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
        temporal = temporal.reshape(batch, patches, steps, dim).permute(0, 2, 1, 3)
        action = temporal.mean(1)
        centers_object = gather_points(hand_points_object, hand_knn_idx).mean(2)
        dense = gather_points(z_hand, hand_knn_idx).mean(2)
        contact = gather_points(hand_contact.unsqueeze(-1), hand_knn_idx).mean(2)
        action = (action + self.hand_position(centers_hand) + self.object_position(centers_object)
                  + self.dense_gate.sigmoid() * self.dense_adapter(torch.cat([dense, contact], -1)))
        return {"action_tokens": action, "action_tokens_temporal": temporal,
                "hand_patch_center_hand": centers_hand,
                "hand_patch_center_object": centers_object}


def axis_angle_to_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues map with a stable zero-angle limit."""
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*vector.shape[:-1], 3, 3)
    theta = vector.norm(dim=-1)
    a = torch.sinc(theta / torch.pi)
    b = 0.5 * torch.sinc(theta / (2.0 * torch.pi)).square()
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(*vector.shape[:-1], 3, 3)
    return eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


class SpatiotemporalInteractionField(nn.Module):
    """Continuously splat known hand action onto the current object surface."""
    def __init__(self, model_dim: int, field_dim: int, sigma_m: float) -> None:
        super().__init__()
        self.sigma_m = float(sigma_m)
        self.action_proj = nn.Linear(model_dim, field_dim)
        self.object_proj = nn.Linear(model_dim, field_dim)
        # relative xyz, distance, object/hand normals, hand increment,
        # normal scalar and tangent xyz = 17 dimensions.
        self.geometry_proj = nn.Sequential(nn.Linear(17, field_dim), nn.GELU(),
                                           nn.Linear(field_dim, field_dim))
        self.norm = nn.LayerNorm(field_dim)

    def forward(self, action_temporal: torch.Tensor, object_tokens: torch.Tensor,
                hand_centers: torch.Tensor, object_centers: torch.Tensor,
                hand_normals: torch.Tensor, object_normals: torch.Tensor,
                hand_disp_chunk_object: torch.Tensor) -> dict[str, torch.Tensor]:
        hand_position = hand_centers[:, None] + hand_disp_chunk_object
        hand_increment = torch.diff(hand_disp_chunk_object, dim=1,
                                    prepend=torch.zeros_like(hand_disp_chunk_object[:, :1]))
        relative = hand_position[:, :, None, :, :] - object_centers[:, None, :, None, :]
        distance = relative.norm(dim=-1, keepdim=True)
        weights = torch.exp(-distance.square() / (2.0 * self.sigma_m ** 2))
        normal_value = (hand_increment[:, :, None] * object_normals[:, None, :, None]).sum(-1, keepdim=True)
        tangent = hand_increment[:, :, None] - normal_value * object_normals[:, None, :, None]
        geometry = torch.cat([
            relative, distance,
            object_normals[:, None, :, None].expand(-1, action_temporal.shape[1], -1,
                                                     hand_centers.shape[1], -1),
            hand_normals[:, None, None].expand(-1, action_temporal.shape[1], object_centers.shape[1], -1, -1),
            hand_increment[:, :, None].expand(-1, -1, object_centers.shape[1], -1, -1),
            normal_value.expand(-1, -1, -1, hand_centers.shape[1], -1),
            tangent.expand(-1, -1, object_centers.shape[1], -1, -1),
        ], -1)
        message = torch.nn.functional.gelu(
            self.action_proj(action_temporal)[:, :, None]
            + self.object_proj(object_tokens)[:, None, :, None]
            + self.geometry_proj(geometry))
        density = weights.sum(3)
        normalized_weights = weights / density[:, :, :, None].clamp_min(1e-8)
        field = self.norm((normalized_weights * message).sum(3))
        mean_normal = (normalized_weights * normal_value).sum(3)
        mean_tangent = (normalized_weights * tangent).sum(3)
        descriptor = torch.cat([density / hand_centers.shape[1], mean_normal, mean_tangent], -1)
        return {"interaction_field": field, "interaction_field_descriptor": descriptor,
                "hand_to_object_soft_weights": normalized_weights.squeeze(-1)}


class SE3DynamicsHead(nn.Module):
    def __init__(self, model_dim: int, field_dim: int, motion_scale: float) -> None:
        super().__init__()
        self.motion_scale = float(motion_scale)
        self.object_proj = nn.Linear(model_dim, field_dim)
        self.head = nn.Sequential(nn.Linear(field_dim + 5, field_dim), nn.GELU(), nn.Linear(field_dim, 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, field: torch.Tensor, descriptor: torch.Tensor,
                object_tokens: torch.Tensor, points: torch.Tensor) -> dict[str, torch.Tensor]:
        global_token = (field + self.object_proj(object_tokens)[:, None]).mean(2)
        twist = self.head(torch.cat([global_token, descriptor.mean(2)], -1))
        increment_translation = twist[..., :3] / self.motion_scale
        increment_rotation_vector = twist[..., 3:]
        increment_rotation = axis_angle_to_matrix(increment_rotation_vector)
        batch = points.shape[0]
        rotation = torch.eye(3, dtype=points.dtype, device=points.device).expand(batch, 3, 3).clone()
        translation = torch.zeros(batch, 3, dtype=points.dtype, device=points.device)
        displacements = []
        for step in range(twist.shape[1]):
            translation = translation + torch.einsum("bij,bj->bi", rotation, increment_translation[:, step])
            rotation = rotation @ increment_rotation[:, step]
            future = torch.einsum("bij,bqj->bqi", rotation, points) + translation[:, None]
            displacements.append(future - points)
        return {"pred_obj_increment_translation_internal": twist[..., :3],
                "pred_obj_increment_rotation_vector": increment_rotation_vector,
                "pred_obj_increment_rotation_matrix": increment_rotation,
                "pred_obj_disp_chunk_se3": torch.stack(displacements, 1)}


class DenseEdgeInteractionModel(nn.Module):
    """V8 diagnostic: encode full-resolution local hand-object edges before compression."""
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        meta = cfg.meta
        self.motion_scale = float(meta.motion_scale)
        self.knn = int(getattr(meta, "dense_edge_knn", 4))
        self.sigma_m = float(getattr(meta, "dense_edge_sigma_m", 0.02))
        dim = int(getattr(meta, "dense_edge_dim", 64))
        # relative xyz, distance, hand/object normal, hand increment,
        # normal action scalar and tangent xyz.
        self.edge_mlp = nn.Sequential(nn.Linear(17, dim), nn.GELU(),
                                      nn.Linear(dim, dim), nn.GELU())
        self.prediction = nn.Linear(dim, 4)
        nn.init.zeros_(self.prediction.weight)
        nn.init.zeros_(self.prediction.bias)

    def forward(self, batch: dict[str, torch.Tensor] | None = None, *,
                world_hand_points_object: torch.Tensor | None = None,
                world_hand_normals_object: torch.Tensor | None = None,
                world_obj_points_object: torch.Tensor | None = None,
                world_obj_normals_object: torch.Tensor | None = None,
                action_hand_disp_chunk_object: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if batch is not None:
            return self.forward(**{key: batch[key] for key in (
                "world_hand_points_object", "world_hand_normals_object",
                "world_obj_points_object", "world_obj_normals_object",
                "action_hand_disp_chunk_object")})
        required = (world_hand_points_object, world_hand_normals_object,
                    world_obj_points_object, world_obj_normals_object,
                    action_hand_disp_chunk_object)
        if any(value is None for value in required):
            raise ValueError("DenseEdgeInteractionModel is missing a current/action tensor")
        distance = torch.cdist(world_hand_points_object.float(), world_obj_points_object.float())
        edge_distance, neighbor = distance.topk(self.knn, dim=-1, largest=False, sorted=True)
        object_points = gather_points(world_obj_points_object, neighbor)
        object_normals = gather_points(world_obj_normals_object, neighbor)
        hand_increment = torch.diff(action_hand_disp_chunk_object, dim=1,
                                    prepend=torch.zeros_like(action_hand_disp_chunk_object[:, :1]))
        hand_position = world_hand_points_object[:, None] + action_hand_disp_chunk_object
        relative = hand_position[:, :, :, None] - object_points[:, None]
        step_distance = relative.norm(dim=-1, keepdim=True)
        normal_action = (hand_increment[:, :, :, None] * object_normals[:, None]).sum(-1, keepdim=True)
        tangent_action = hand_increment[:, :, :, None] - normal_action * object_normals[:, None]
        geometry = torch.cat([
            relative, step_distance,
            world_hand_normals_object[:, None, :, None].expand(-1, hand_increment.shape[1], -1,
                                                                self.knn, -1),
            object_normals[:, None].expand(-1, hand_increment.shape[1], -1, -1, -1),
            hand_increment[:, :, :, None].expand(-1, -1, -1, self.knn, -1),
            normal_action, tangent_action,
        ], -1)
        edge = self.edge_mlp(geometry)
        weights = torch.softmax(-step_distance.squeeze(-1).square() / (2 * self.sigma_m ** 2), dim=-1)
        dense_interaction = (weights[..., None] * edge).sum(3)
        return {
            "dense_interaction_tokens": dense_interaction,
            "dense_nearest_obj_point": neighbor[..., 0],
            "dense_edge_distance_m": edge_distance[..., 0],
            "pred_dense_relative_motion_internal": self.prediction(dense_interaction),
        }


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
        self.use_v7_field = bool(getattr(meta, "use_v7_field", False))
        self.field_ablation = str(getattr(meta, "field_ablation", "full"))
        if self.field_ablation not in {"full", "no_descriptor", "global_action_only"}:
            raise ValueError(f"Unsupported field_ablation: {self.field_ablation}")
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
        self.edge_fusion = nn.Sequential(
            nn.Linear(meta.model_dim * 2 + 7, meta.model_dim), nn.GELU(),
            nn.Linear(meta.model_dim, meta.model_dim), nn.LayerNorm(meta.model_dim))
        self.relative_motion = nn.Linear(meta.model_dim, meta.chunk_len * 4)
        field_dim = int(getattr(meta, "field_dim", 128))
        self.interaction_field = SpatiotemporalInteractionField(
            meta.model_dim, field_dim, float(getattr(meta, "field_sigma_m", 0.05)))
        self.se3_dynamics = SE3DynamicsHead(meta.model_dim, field_dim, self.motion_scale)
        self.interaction_reconstruction = bool(
            getattr(meta, "interaction_reconstruction", False))
        if self.interaction_reconstruction:
            self.interaction_reconstruction_head = nn.Sequential(
                nn.Linear(field_dim + meta.model_dim, field_dim), nn.GELU(),
                nn.Linear(field_dim, 6))
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
                action_hand_disp_chunk_object: torch.Tensor | None = None,
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
                "hand_disp_chunk", "action_hand_disp_chunk_object",
                "dense_obj_points_hand", "dense_obj_normals_hand",
                "dense_hand_points_hand", "dense_hand_normals_hand", "effect_obj_points_object",
                "effect_obj_normals_object", "effect_obj_valid_mask")})
        required = (world_hand_points_object, world_hand_normals_object, world_obj_points_object,
                    world_obj_normals_object, action_hand_points_hand, action_hand_normals_hand,
                    hand_disp_chunk, action_hand_disp_chunk_object,
                    dense_obj_points_hand, dense_obj_normals_hand,
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
        action_temporal = action["action_tokens_temporal"] + (
            action_context - action["action_tokens"])[:, None]
        hand_patch_disp_object = torch.stack([
            gather_points(action_hand_disp_chunk_object[:, step], world["hand_knn_idx"]).mean(2)
            for step in range(action_hand_disp_chunk_object.shape[1])
        ], 1)
        hand_patch_normal_object = gather_points(
            world_hand_normals_object, world["hand_knn_idx"]).mean(2)
        hand_patch_normal_object = nn.functional.normalize(hand_patch_normal_object, dim=-1)
        obj_patch_normal_object = gather_points(
            world_obj_normals_object, world["obj_knn_idx"]).mean(2)
        obj_patch_normal_object = nn.functional.normalize(obj_patch_normal_object, dim=-1)
        field = self.interaction_field(
            action_temporal, world["world_obj_tokens"], action["hand_patch_center_object"],
            world["obj_patch_centers_object"], hand_patch_normal_object,
            obj_patch_normal_object, hand_patch_disp_object)
        se3_field = field["interaction_field"]
        se3_descriptor = field["interaction_field_descriptor"]
        if self.field_ablation == "no_descriptor":
            se3_descriptor = torch.zeros_like(se3_descriptor)
        elif self.field_ablation == "global_action_only":
            global_action = torch.nn.functional.gelu(
                self.interaction_field.action_proj(action_temporal.mean(2)))
            global_action = self.interaction_field.norm(global_action)
            se3_field = global_action[:, :, None].expand_as(se3_field)
            se3_descriptor = torch.zeros_like(se3_descriptor)
        se3 = self.se3_dynamics(se3_field, se3_descriptor,
                                world["world_obj_tokens"], effect_obj_points_object)
        global_c = se3_field.mean(2)
        reconstruction = {}
        if self.interaction_reconstruction:
            reconstruction["pred_object_centric_action_field"] = self.interaction_reconstruction_head(
                torch.cat([
                    global_c[:, :, None].expand(-1, -1, world["world_obj_tokens"].shape[1], -1),
                    world["world_obj_tokens"][:, None].expand(-1, global_c.shape[1], -1, -1),
                ], -1))
        # No object residual: object tokens only locate canonical interaction queries.
        interaction, object_attention = self.canonicalizer(world["world_obj_tokens"], action_context, residual=False)
        interaction = nn.functional.layer_norm(interaction, (interaction.shape[-1],))
        batch_size = interaction.shape[0]
        obj_centers = world["obj_patch_centers_object"]
        hand_centers = action["hand_patch_center_object"]
        nearest_obj = torch.cdist(hand_centers.float(), obj_centers.float()).argmin(-1)
        batch_index = torch.arange(batch_size, device=interaction.device)[:, None]
        paired_obj_token = world["world_obj_tokens"][batch_index, nearest_obj]
        paired_obj_center = obj_centers[batch_index, nearest_obj]
        obj_patch_normal = gather_points(world_obj_normals_object, world["obj_knn_idx"]).mean(2)
        obj_patch_normal = nn.functional.normalize(obj_patch_normal, dim=-1)
        paired_obj_normal = obj_patch_normal[batch_index, nearest_obj]
        relative_xyz = hand_centers - paired_obj_center
        edge_geometry = torch.cat([relative_xyz, paired_obj_normal,
                                   relative_xyz.norm(dim=-1, keepdim=True)], -1)
        relative_interaction = self.edge_fusion(torch.cat(
            [action_context, paired_obj_token, edge_geometry], -1))
        pred_relative_motion_internal = self.relative_motion(relative_interaction).reshape(
            batch_size, -1, self.effect.time_embed.shape[1], 4).transpose(1, 2)
        pred_hand_patch_disp_internal = self.action_reconstruction(action_context).reshape(
            batch_size, -1, self.effect.time_embed.shape[1], 3).transpose(1, 2)
        pred_obj_patch_disp_internal = self.patch_effect(interaction).reshape(
            batch_size, -1, self.effect.time_embed.shape[1], 3).transpose(1, 2)
        effect = self.effect(effect_obj_points_object, effect_obj_normals_object, z_obj,
                             interaction, world["obj_patch_centers_object"],
                             pred_obj_patch_disp_internal, effect_obj_valid_mask, self.motion_scale)
        pred_obj_disp = (se3["pred_obj_disp_chunk_se3"] if self.use_v7_field
                         else effect["pred_obj_disp_chunk"])
        return {**world, **action, **effect, **field, **se3, **reconstruction,
                "se3_interaction_field": se3_field,
                "se3_interaction_field_descriptor": se3_descriptor,
                "global_interaction_code": global_c,
                "pred_obj_disp_chunk": pred_obj_disp,
                "world_hand_tokens": world_hand, "world_tokens": world_tokens,
                "action_context_tokens": action_context, "interaction_tokens": interaction,
                "relative_interaction_tokens": relative_interaction,
                "relative_nearest_obj_patch": nearest_obj,
                "relative_edge_distance_m": relative_xyz.norm(dim=-1),
                "pred_relative_motion_internal": pred_relative_motion_internal,
                "pred_hand_patch_disp_internal": pred_hand_patch_disp_internal,
                "pred_obj_patch_disp_internal": pred_obj_patch_disp_internal,
                "action_to_world_attention": action_attention,
                "object_to_action_attention": object_attention}
