"""InteractionDynamics V1 model: action chunk -> interaction -> object effect."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.task.Cm.dense_token import FrozenDenseTokenEncoder
from src.task.InteractionDynamics.uni3d import Uni3DWorldEncoder, gather_points


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.q_norm, self.kv_norm = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, query: torch.Tensor, context: torch.Tensor,
                *, residual: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attn(self.q_norm(query), self.kv_norm(context), self.kv_norm(context),
                                      need_weights=True, average_attn_weights=False)
        output = query + attended if residual else attended
        output = output + self.ffn(self.ffn_norm(output))
        return output, weights


class ActionEncoder(nn.Module):
    def __init__(self, dim: int, heads: int, chunk_len: int, temporal_layers: int,
                 dense_dim: int) -> None:
        super().__init__()
        self.spatial = nn.Sequential(nn.Linear(9, 128), nn.GELU(), nn.Linear(128, 256),
                                     nn.GELU(), nn.Linear(256, dim))
        self.time_embed = nn.Parameter(torch.empty(1, chunk_len, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, temporal_layers, norm=nn.LayerNorm(dim))
        self.hand_position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dense_adapter = nn.Linear(dense_dim + 1, dim)
        self.dense_gate = nn.Parameter(torch.tensor(-4.0))
        nn.init.normal_(self.time_embed, std=.02)

    def forward(self, hand_points: torch.Tensor, hand_normals: torch.Tensor,
                hand_disp_chunk: torch.Tensor, hand_points_object: torch.Tensor,
                hand_knn_idx: torch.Tensor, z_hand: torch.Tensor,
                hand_contact: torch.Tensor, motion_scale: float) -> dict[str, torch.Tensor]:
        local_points = gather_points(hand_points, hand_knn_idx)
        centers_hand = local_points.mean(2)
        relative = local_points - centers_hand[:, :, None]
        normals = gather_points(hand_normals, hand_knn_idx)
        displacement = torch.stack(
            [gather_points(hand_disp_chunk[:, step], hand_knn_idx) for step in range(hand_disp_chunk.shape[1])], 1
        ) * motion_scale
        base = torch.cat([
            relative[:, None].expand(-1, hand_disp_chunk.shape[1], -1, -1, -1),
            normals[:, None].expand(-1, hand_disp_chunk.shape[1], -1, -1, -1), displacement,
        ], -1)
        spatial = self.spatial(base).amax(3) + self.time_embed[:, :hand_disp_chunk.shape[1]]
        batch, steps, patches, dim = spatial.shape
        temporal = self.temporal(spatial.permute(0, 2, 1, 3).reshape(batch * patches, steps, dim))
        action = temporal.mean(1).reshape(batch, patches, dim)
        centers_object = gather_points(hand_points_object, hand_knn_idx).mean(2)
        dense = gather_points(z_hand, hand_knn_idx).mean(2)
        contact = gather_points(hand_contact.unsqueeze(-1), hand_knn_idx).mean(2)
        action = (action + self.hand_position(centers_hand) + self.object_position(centers_object)
                  + self.dense_gate.sigmoid() * self.dense_adapter(torch.cat([dense, contact], -1)))
        return {"action_tokens": action, "hand_patch_center_hand": centers_hand,
                "hand_patch_center_object": centers_object}


class EffectDecoder(nn.Module):
    def __init__(self, dim: int, heads: int, chunk_len: int, dense_dim: int) -> None:
        super().__init__()
        del heads, dense_dim
        self.time_embed = nn.Parameter(torch.empty(1, chunk_len, 1, dim))
        self.flow = nn.Sequential(nn.Linear(dim + 3 + 3 + dim, dim), nn.GELU(), nn.Linear(dim, 3))
        nn.init.normal_(self.time_embed, std=.02)

    def forward(self, points: torch.Tensor, normals: torch.Tensor, z_obj: torch.Tensor,
                interaction: torch.Tensor, patch_centers: torch.Tensor,
                patch_motion_internal: torch.Tensor, valid: torch.Tensor,
                motion_scale: float) -> dict[str, torch.Tensor]:
        del z_obj
        nearest = torch.cdist(points.float(), patch_centers.float()).argmin(-1)
        batch_index = torch.arange(points.shape[0], device=points.device)[:, None]
        routed_interaction = interaction[batch_index, nearest]
        routed_centers = patch_centers[batch_index, nearest]
        routed_patch_motion = patch_motion_internal[
            torch.arange(points.shape[0], device=points.device)[:, None, None],
            torch.arange(patch_motion_internal.shape[1], device=points.device)[None, :, None],
            nearest[:, None, :],
        ]
        steps, queries, dim = patch_motion_internal.shape[1], points.shape[1], interaction.shape[-1]
        context = routed_interaction[:, None].expand(-1, steps, -1, -1)
        local_xyz = (points - routed_centers)[:, None].expand(-1, steps, -1, -1)
        local_normals = normals[:, None].expand(-1, steps, -1, -1)
        time = self.time_embed[:, :steps].expand(points.shape[0], -1, queries, -1)
        residual = self.flow(torch.cat([context, local_xyz, local_normals, time], -1))
        pred_internal = routed_patch_motion + residual
        pred = pred_internal / motion_scale
        pred = pred * valid[:, None, :, None].to(pred.dtype)
        routing = torch.nn.functional.one_hot(nearest, num_classes=interaction.shape[1]).to(interaction.dtype)
        return {"pred_obj_disp_chunk": pred, "pred_obj_disp_internal": pred_internal,
                "effect_patch_index": nearest,
                "effect_to_interaction_attention": routing[:, None, None].expand(-1, 1, steps, -1, -1)}


class InteractionDynamicsModel(nn.Module):
    """The forward signature intentionally contains no future object target."""
    _DENSE_PREFIX = "dense_encoder."
    def __init__(self, cfg: Any, *, dense_encoder: nn.Module | None = None,
                 load_uni3d: bool = True, world_depth: int = 12) -> None:
        super().__init__()
        meta = cfg.meta
        self.motion_scale = float(meta.motion_scale)
        self.action_local_gain = float(getattr(meta, "action_local_gain", 1.0))
        self.dense_encoder = dense_encoder or FrozenDenseTokenEncoder(meta.dense_checkpoint)
        self.dense_encoder.requires_grad_(False).eval()
        dense_dim = int(self.dense_encoder.token_dim)
        self.world = Uni3DWorldEncoder(meta.model_dim, meta.num_hand_patches, meta.num_obj_patches,
                                       meta.patch_size, world_depth, meta.attention_heads)
        if load_uni3d:
            self.world.load_pretrained(meta.uni3d_checkpoint)
        self.world_dense_adapter = nn.Linear(dense_dim + 1, meta.model_dim)
        self.world_dense_gate = nn.Parameter(torch.tensor(-4.0))
        self.action = ActionEncoder(meta.model_dim, meta.attention_heads, meta.chunk_len,
                                    meta.action_temporal_layers, dense_dim)
        self.action_world = nn.ModuleList(
            [CrossAttentionBlock(meta.model_dim, meta.attention_heads) for _ in range(meta.action_world_layers)])
        self.canonicalizer = CrossAttentionBlock(meta.model_dim, meta.attention_heads)
        self.action_reconstruction = nn.Linear(meta.model_dim, meta.chunk_len * 3)
        self.patch_effect = nn.Linear(meta.model_dim, meta.chunk_len * 3)
        self.effect = EffectDecoder(meta.model_dim, meta.attention_heads, meta.chunk_len, dense_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        self.dense_encoder.eval()
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        state = super().state_dict(*args, **kwargs)
        return {key: value for key, value in state.items() if not key.startswith(self._DENSE_PREFIX)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith(self._DENSE_PREFIX)]
        if strict and (missing or incompatible.unexpected_keys):
            raise RuntimeError(f"InteractionDynamics checkpoint mismatch: missing={missing}, "
                               f"unexpected={incompatible.unexpected_keys}")
        return incompatible

    def forward(self, batch: dict[str, torch.Tensor] | None = None, *, world_hand_points_object: torch.Tensor | None = None,
                world_hand_normals_object: torch.Tensor | None = None,
                world_obj_points_object: torch.Tensor | None = None,
                world_obj_normals_object: torch.Tensor | None = None,
                action_hand_points_hand: torch.Tensor | None = None,
                action_hand_normals_hand: torch.Tensor | None = None,
                hand_disp_chunk: torch.Tensor | None = None,
                dense_obj_points_hand: torch.Tensor | None = None,
                dense_obj_normals_hand: torch.Tensor | None = None,
                dense_hand_points_hand: torch.Tensor | None = None,
                dense_hand_normals_hand: torch.Tensor | None = None,
                effect_obj_points_object: torch.Tensor | None = None,
                effect_obj_normals_object: torch.Tensor | None = None,
                effect_obj_valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if batch is not None:
            return self.forward(**{key: batch[key] for key in (
                "world_hand_points_object", "world_hand_normals_object", "world_obj_points_object",
                "world_obj_normals_object", "action_hand_points_hand", "action_hand_normals_hand",
                "hand_disp_chunk", "dense_obj_points_hand", "dense_obj_normals_hand",
                "dense_hand_points_hand", "dense_hand_normals_hand", "effect_obj_points_object",
                "effect_obj_normals_object", "effect_obj_valid_mask")})
        required = (world_hand_points_object, world_hand_normals_object, world_obj_points_object,
                    world_obj_normals_object, action_hand_points_hand, action_hand_normals_hand,
                    hand_disp_chunk, dense_obj_points_hand, dense_obj_normals_hand,
                    dense_hand_points_hand, dense_hand_normals_hand, effect_obj_points_object,
                    effect_obj_normals_object, effect_obj_valid_mask)
        if any(value is None for value in required):
            raise ValueError("InteractionDynamicsModel forward is missing a required current/action tensor")
        # PTv3/spconv's frozen checkpoint runs in FP32; its sparse kernels do not
        # support the runner's bfloat16 autocast path for every valid-mask pattern.
        device_type = dense_obj_points_hand.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            z_obj, z_hand, contact = self.dense_encoder(
                obj_points=dense_obj_points_hand.float(), obj_normals=dense_obj_normals_hand.float(),
                hand_points=dense_hand_points_hand.float(), hand_normals=dense_hand_normals_hand.float(),
                obj_valid_mask=effect_obj_valid_mask)
        world = self.world(world_hand_points_object, world_hand_normals_object,
                           world_obj_points_object, world_obj_normals_object)
        dense_patch = gather_points(z_hand, world["hand_knn_idx"]).mean(2)
        contact_patch = gather_points(contact.unsqueeze(-1), world["hand_knn_idx"]).mean(2)
        world_hand = world["world_hand_tokens"] + self.world_dense_gate.sigmoid() * self.world_dense_adapter(
            torch.cat([dense_patch, contact_patch], -1))
        world_tokens = torch.cat([world_hand, world["world_obj_tokens"]], 1)
        action = self.action(action_hand_points_hand, action_hand_normals_hand, hand_disp_chunk,
                             world_hand_points_object, world["hand_knn_idx"], z_hand, contact, self.motion_scale)
        action_context = action["action_tokens"]
        action_attention = None
        for block in self.action_world:
            action_context, action_attention = block(action_context, world_tokens)
        action_mean = action_context.mean(dim=1, keepdim=True)
        action_context = action_mean + float(getattr(self, "action_local_gain", 1.0)) * (
            action_context - action_mean)
        # No object residual: object tokens only locate canonical interaction queries.
        interaction, object_attention = self.canonicalizer(world["world_obj_tokens"], action_context, residual=False)
        interaction = nn.functional.layer_norm(interaction, (interaction.shape[-1],))
        batch_size = interaction.shape[0]
        pred_hand_patch_disp_internal = self.action_reconstruction(action_context).reshape(
            batch_size, -1, self.effect.time_embed.shape[1], 3).transpose(1, 2)
        pred_obj_patch_disp_internal = self.patch_effect(interaction).reshape(
            batch_size, -1, self.effect.time_embed.shape[1], 3).transpose(1, 2)
        effect = self.effect(effect_obj_points_object, effect_obj_normals_object, z_obj,
                             interaction, world["obj_patch_centers_object"],
                             pred_obj_patch_disp_internal, effect_obj_valid_mask, self.motion_scale)
        return {**world, **action, **effect, "world_hand_tokens": world_hand, "world_tokens": world_tokens,
                "action_context_tokens": action_context, "interaction_tokens": interaction,
                "pred_hand_patch_disp_internal": pred_hand_patch_disp_internal,
                "pred_obj_patch_disp_internal": pred_obj_patch_disp_internal,
                "action_to_world_attention": action_attention,
                "object_to_action_attention": object_attention}
