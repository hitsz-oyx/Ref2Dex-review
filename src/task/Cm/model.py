"""Hand-anchored temporal Cm tokens and an object point-flow decoder."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.task.Cm.dense_token import FrozenDenseTokenEncoder


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


def _gather_points(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    return values.gather(1, index.unsqueeze(-1).expand(-1, -1, values.shape[-1]))


class CmFlowHead(nn.Module):
    """Learnable temporal bottleneck after the frozen current-frame encoder."""

    def __init__(
        self,
        *,
        dense_token_dim: int,
        cm_dim: int = 256,
        num_cm_tokens: int = 32,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()
        if cm_dim % num_attention_heads != 0:
            raise ValueError("cm_dim must be divisible by num_attention_heads")
        self.dense_token_dim = int(dense_token_dim)
        self.cm_dim = int(cm_dim)
        self.num_cm_tokens = int(num_cm_tokens)
        # z_h, current point/normal, hand flow, dense hand-contact prior,
        # and T_{hand_t <- hand_t1} flattened to 12 values.
        self.hand_motion_encoder = _mlp(self.dense_token_dim + 22, cm_dim, cm_dim)
        self.object_context_encoder = _mlp(self.dense_token_dim + 6, cm_dim, cm_dim)
        self.motion_to_scene = nn.MultiheadAttention(
            cm_dim, num_attention_heads, dropout=0.0, batch_first=True
        )
        self.motion_norm = nn.LayerNorm(cm_dim)
        self.activity_head = nn.Sequential(
            nn.Linear(cm_dim, cm_dim // 2), nn.GELU(), nn.Linear(cm_dim // 2, 1)
        )
        # object feature, Cm feature, relative object-to-hand anchor position,
        # anchor hand flow, and anchor hand normal.
        self.flow_edge = nn.Sequential(
            nn.Linear(cm_dim * 2 + 9, cm_dim),
            nn.GELU(),
            nn.Linear(cm_dim, cm_dim // 2),
            nn.GELU(),
            nn.Linear(cm_dim // 2, 4),
        )
        # Point flow is measured in metres and normally starts close to zero.
        # A zero initialized final decoder is a stable and interpretable
        # baseline for overfit and large-scale training alike.
        nn.init.zeros_(self.flow_edge[-1].weight)
        nn.init.zeros_(self.flow_edge[-1].bias)

    @staticmethod
    def _safe_key_padding_mask(obj_valid_mask: torch.Tensor) -> torch.Tensor:
        """Avoid all-masked attention rows for non-contact frames."""
        mask = ~obj_valid_mask.bool()
        no_valid_object = mask.all(dim=1)
        if no_valid_object.any():
            mask = mask.clone()
            mask[no_valid_object, 0] = False
        return mask

    def forward(
        self,
        *,
        z_obj: torch.Tensor,
        z_hand: torch.Tensor,
        dense_hand_contact: torch.Tensor,
        obj_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_flow: torch.Tensor,
        wrist_delta: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_hand, _ = hand_points.shape
        wrist_features = wrist_delta[:, :3, :4].reshape(batch_size, 1, 12).expand(-1, num_hand, -1)
        hand_input = torch.cat(
            [z_hand, hand_points, hand_normals, hand_flow, dense_hand_contact.unsqueeze(-1), wrist_features], dim=-1
        )
        hand_motion = self.hand_motion_encoder(hand_input)
        obj_context = self.object_context_encoder(torch.cat([z_obj, obj_points, obj_normals], dim=-1))
        scene_motion, _ = self.motion_to_scene(
            hand_motion,
            obj_context,
            obj_context,
            key_padding_mask=self._safe_key_padding_mask(obj_valid_mask),
            need_weights=False,
        )
        hand_action = self.motion_norm(hand_motion + scene_motion)
        activity_logits = self.activity_head(hand_action).squeeze(-1)
        num_tokens = min(self.num_cm_tokens, num_hand)
        cm_anchor_idx = activity_logits.topk(num_tokens, dim=1).indices
        cm_tokens = _gather_points(hand_action, cm_anchor_idx)
        cm_anchor_pos = _gather_points(hand_points, cm_anchor_idx)
        cm_anchor_normal = _gather_points(hand_normals, cm_anchor_idx)
        cm_hand_flow = _gather_points(hand_flow, cm_anchor_idx)
        cm_active = torch.sigmoid(activity_logits.gather(1, cm_anchor_idx))

        num_obj = obj_points.shape[1]
        relative = obj_points.unsqueeze(2) - cm_anchor_pos.unsqueeze(1)
        edge_input = torch.cat(
            [
                obj_context.unsqueeze(2).expand(-1, -1, num_tokens, -1),
                cm_tokens.unsqueeze(1).expand(-1, num_obj, -1, -1),
                relative,
                cm_hand_flow.unsqueeze(1).expand(-1, num_obj, -1, -1),
                cm_anchor_normal.unsqueeze(1).expand(-1, num_obj, -1, -1),
            ],
            dim=-1,
        )
        edge_output = self.flow_edge(edge_input)
        attention_logits = edge_output[..., 0] + torch.log(cm_active.unsqueeze(1).clamp_min(1e-6))
        edge_weight = torch.softmax(attention_logits, dim=2)
        pred_obj_flow = (edge_weight.unsqueeze(-1) * edge_output[..., 1:]).sum(dim=2)
        pred_obj_flow = pred_obj_flow * obj_valid_mask.unsqueeze(-1).float()
        return {
            "pred_obj_flow": pred_obj_flow,
            "cm_tokens": cm_tokens,
            "cm_anchor_idx": cm_anchor_idx,
            "cm_anchor_pos": cm_anchor_pos,
            "cm_anchor_normal": cm_anchor_normal,
            "cm_hand_flow": cm_hand_flow,
            "cm_active": cm_active,
        }


class CmFlowModel(nn.Module):
    """BaseRunner model: frozen dense tokens plus trainable Cm temporal head.

    ``state_dict`` deliberately omits the immutable 1.1GB correspondence
    checkpoint.  Its absolute source path is saved in ``cfg.meta`` by the Base
    checkpoint manager, and construction reloads it before Cm weights are
    restored.  Cm checkpoints therefore contain only the newly trained head.
    """

    _DENSE_PREFIX = "dense_encoder."

    def __init__(
        self,
        cfg: Any,
        *,
        condition_shape: Any = None,
        target_shape: Any = None,
    ) -> None:
        del condition_shape, target_shape
        super().__init__()
        meta = cfg.meta
        self.dense_encoder = FrozenDenseTokenEncoder(meta.dense_checkpoint)
        self.head = CmFlowHead(
            dense_token_dim=self.dense_encoder.token_dim,
            cm_dim=int(meta.cm_dim),
            num_cm_tokens=int(meta.num_cm_tokens),
            num_attention_heads=int(meta.num_attention_heads),
        )

    @property
    def dense_token_dim(self) -> int:
        return self.head.dense_token_dim

    @property
    def cm_dim(self) -> int:
        return self.head.cm_dim

    @property
    def num_cm_tokens(self) -> int:
        return self.head.num_cm_tokens

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_obj, z_hand, dense_contact = self.dense_encoder(
            obj_points=batch["obj_points"],
            obj_normals=batch["obj_normals"],
            hand_points=batch["hand_points"],
            hand_normals=batch["hand_normals"],
            obj_valid_mask=batch["obj_valid_mask"],
        )
        return self.head(
            z_obj=z_obj,
            z_hand=z_hand,
            dense_hand_contact=dense_contact,
            obj_points=batch["obj_points"],
            obj_normals=batch["obj_normals"],
            hand_points=batch["hand_points"],
            hand_normals=batch["hand_normals"],
            hand_flow=batch["hand_flow"],
            wrist_delta=batch["wrist_delta"],
            obj_valid_mask=batch["obj_valid_mask"],
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        state = super().state_dict(*args, **kwargs)
        return {key: value for key, value in state.items() if not key.startswith(self._DENSE_PREFIX)}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = [key for key in incompatible.missing_keys if not key.startswith(self._DENSE_PREFIX)]
        unexpected = list(incompatible.unexpected_keys)
        if strict and (missing or unexpected):
            raise RuntimeError(
                "Cm checkpoint is incompatible with the temporal head: "
                f"missing={missing}, unexpected={unexpected}."
            )
        return incompatible
