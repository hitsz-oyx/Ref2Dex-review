"""Hand-motion Slot Attention bottleneck and current-object flow decoder."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from src.task.Cm.dense_token import FrozenDenseTokenEncoder


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class SlotAttention(nn.Module):
    """Deterministic Slot Attention over hand-point motion features.

    Slots start from distinct learned vectors rather than per-batch random
    noise, which keeps Cm extraction deterministic for a fixed checkpoint.
    The returned slot weights can be projected back onto hand geometry to
    localize each learned slot after training.
    """

    def __init__(self, *, dim: int, num_slots: int, num_iterations: int = 3) -> None:
        super().__init__()
        if dim <= 0 or num_slots <= 0 or num_iterations <= 0:
            raise ValueError("SlotAttention requires positive dim, num_slots, and num_iterations.")
        self.dim = int(dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.slot_init = nn.Parameter(torch.empty(1, self.num_slots, self.dim))
        nn.init.normal_(self.slot_init, std=self.dim**-0.5)
        self.norm_inputs = nn.LayerNorm(self.dim)
        self.norm_slots = nn.LayerNorm(self.dim)
        self.norm_mlp = nn.LayerNorm(self.dim)
        self.to_key = nn.Linear(self.dim, self.dim, bias=False)
        self.to_value = nn.Linear(self.dim, self.dim, bias=False)
        self.to_query = nn.Linear(self.dim, self.dim, bias=False)
        self.gru = nn.GRUCell(self.dim, self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            nn.GELU(),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.scale = self.dim**-0.5

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return sparse slots, point-to-slot assignments, and slot weights.

        Args:
            inputs: `[B, H, D]` hand-point motion features.
        """
        if inputs.ndim != 3 or inputs.shape[-1] != self.dim:
            raise ValueError(
                f"Expected hand motion features [B, H, {self.dim}], got {tuple(inputs.shape)}."
            )
        batch_size = inputs.shape[0]
        encoded = self.norm_inputs(inputs)
        keys = self.to_key(encoded)
        values = self.to_value(encoded)
        slots = self.slot_init.expand(batch_size, -1, -1)
        slot_weights: torch.Tensor | None = None
        slot_assignment: torch.Tensor | None = None
        for _ in range(self.num_iterations):
            previous_slots = slots
            queries = self.to_query(self.norm_slots(slots))
            logits = torch.einsum("bkd,bhd->bkh", queries, keys) * self.scale
            # Each hand point first assigns its mass among slots.  Each slot
            # then normalizes its update across points.
            slot_assignment = torch.softmax(logits, dim=1)
            slot_weights = slot_assignment / slot_assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bkh,bhd->bkd", slot_weights, values)
            slots = self.gru(
                updates.reshape(-1, self.dim),
                previous_slots.reshape(-1, self.dim),
            ).reshape(batch_size, self.num_slots, self.dim)
            slots = slots + self.mlp(self.norm_mlp(slots))
        assert slot_weights is not None and slot_assignment is not None
        return slots, slot_assignment, slot_weights


class CmFlowHead(nn.Module):
    """Full hand-action field compressed into sparse temporal ``C_m`` slots.

    Slot Attention only replaces the old scene cross-attention, activity head,
    and hard Top-K bottleneck.  The current-object geometric flow decoder stays
    separate and sees ``C_m`` only after the hand-side bottleneck.
    """

    def __init__(
        self,
        *,
        dense_token_dim: int,
        cm_dim: int = 256,
        num_cm_tokens: int = 16,
        num_slot_iters: int = 3,
    ) -> None:
        super().__init__()
        self.dense_token_dim = int(dense_token_dim)
        self.cm_dim = int(cm_dim)
        self.num_cm_tokens = int(num_cm_tokens)
        self.num_slot_iters = int(num_slot_iters)
        # Frozen current interaction state, local geometry, local motion,
        # frozen dense contact prior, and whole-wrist SE(3) delta.
        self.hand_motion_encoder = _mlp(self.dense_token_dim + 22, cm_dim, cm_dim)
        self.slot_attention = SlotAttention(
            dim=cm_dim,
            num_slots=num_cm_tokens,
            num_iterations=num_slot_iters,
        )
        self.object_context_encoder = _mlp(self.dense_token_dim + 6, cm_dim, cm_dim)
        # Object feature, Cm feature, relative object-to-soft-anchor position,
        # soft-anchor hand flow, and soft-anchor hand normal.
        self.flow_edge = nn.Sequential(
            nn.Linear(cm_dim * 2 + 9, cm_dim),
            nn.GELU(),
            nn.Linear(cm_dim, cm_dim // 2),
            nn.GELU(),
            nn.Linear(cm_dim // 2, 4),
        )
        # Point flow is measured in metres and normally starts close to zero.
        nn.init.zeros_(self.flow_edge[-1].weight)
        nn.init.zeros_(self.flow_edge[-1].bias)

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
        batch_size, num_hand, _ = hand_flow.shape
        wrist_features = wrist_delta[:, :3, :4].reshape(batch_size, 1, 12).expand(-1, num_hand, -1)
        hand_input = torch.cat(
            [
                z_hand,
                hand_points,
                hand_normals,
                hand_flow,
                dense_hand_contact.unsqueeze(-1),
                wrist_features,
            ],
            dim=-1,
        )
        u_hand = self.hand_motion_encoder(hand_input)
        cm_tokens, cm_assignment, cm_slot_weights = self.slot_attention(u_hand)

        obj_context = self.object_context_encoder(torch.cat([z_obj, obj_points, obj_normals], dim=-1))
        cm_anchor_pos = torch.einsum("bkh,bhd->bkd", cm_slot_weights, hand_points)
        cm_hand_flow = torch.einsum("bkh,bhd->bkd", cm_slot_weights, hand_flow)
        cm_anchor_normal = torch.einsum("bkh,bhd->bkd", cm_slot_weights, hand_normals)
        cm_anchor_normal = F.normalize(cm_anchor_normal, dim=-1, eps=1e-6)

        num_obj = obj_points.shape[1]
        relative = obj_points.unsqueeze(2) - cm_anchor_pos.unsqueeze(1)
        edge_input = torch.cat(
            [
                obj_context.unsqueeze(2).expand(-1, -1, self.num_cm_tokens, -1),
                cm_tokens.unsqueeze(1).expand(-1, num_obj, -1, -1),
                relative,
                cm_hand_flow.unsqueeze(1).expand(-1, num_obj, -1, -1),
                cm_anchor_normal.unsqueeze(1).expand(-1, num_obj, -1, -1),
            ],
            dim=-1,
        )
        edge_output = self.flow_edge(edge_input)
        edge_weight = torch.softmax(edge_output[..., 0], dim=2)
        pred_obj_flow = (edge_weight.unsqueeze(-1) * edge_output[..., 1:]).sum(dim=2)
        pred_obj_flow = pred_obj_flow * obj_valid_mask.unsqueeze(-1).float()
        # q_k: mean decoder attention paid to each slot by valid object points.
        # This is a diagnostic only; it neither gates slots nor changes flow.
        valid_object = obj_valid_mask.unsqueeze(-1).float()
        decoder_slot_usage = (edge_weight * valid_object).sum(dim=1)
        decoder_slot_usage = decoder_slot_usage / valid_object.sum(dim=1).clamp_min(1.0)
        return {
            "pred_obj_flow": pred_obj_flow,
            "cm_tokens": cm_tokens,
            "cm_anchor_pos": cm_anchor_pos,
            "cm_anchor_normal": cm_anchor_normal,
            "cm_hand_flow": cm_hand_flow,
            "cm_assignment": cm_assignment,
            "cm_slot_weights": cm_slot_weights,
            "decoder_slot_usage": decoder_slot_usage,
        }


class CmFlowModel(nn.Module):
    """Frozen dense interaction state plus trainable hand-motion ``C_m`` head.

    ``state_dict`` deliberately omits the immutable dense-token checkpoint.
    Cm checkpoints contain only the temporal Slot Attention head and decoder.
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
            num_slot_iters=int(meta.slot_iters),
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
        z_obj, z_hand, dense_hand_contact = self.dense_encoder(
            obj_points=batch["obj_points"],
            obj_normals=batch["obj_normals"],
            hand_points=batch["hand_points"],
            hand_normals=batch["hand_normals"],
            obj_valid_mask=batch["obj_valid_mask"],
        )
        return self.head(
            z_obj=z_obj,
            z_hand=z_hand,
            dense_hand_contact=dense_hand_contact,
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
                "Cm checkpoint is incompatible with the temporal Slot Attention head: "
                f"missing={missing}, unexpected={unexpected}."
            )
        return incompatible
