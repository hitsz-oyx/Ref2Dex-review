"""Hand-motion Slot Attention bottleneck and current-object flow decoder."""
from __future__ import annotations

import math
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


class HardConcreteGate(nn.Module):
    """Differentiable stochastic gates with an analytic expected L0 penalty."""

    def __init__(
        self,
        *,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("HardConcreteGate temperature must be positive.")
        if gamma >= 0.0 or zeta <= 1.0:
            raise ValueError("HardConcreteGate requires gamma < 0 and zeta > 1.")
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.zeta = float(zeta)
        self.eps = float(eps)

    def nonzero_probability(self, log_alpha: torch.Tensor) -> torch.Tensor:
        offset = self.temperature * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(log_alpha - offset)

    def forward(self, log_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonzero_probability = self.nonzero_probability(log_alpha)
        if self.training:
            uniform = torch.rand_like(log_alpha).clamp(self.eps, 1.0 - self.eps)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            soft_gate = torch.sigmoid((log_alpha + logistic_noise) / self.temperature)
            gate = (soft_gate * (self.zeta - self.gamma) + self.gamma).clamp(0.0, 1.0)
        else:
            # Extraction uses an exact, fixed-size binary mask.  The analytic
            # probability is retained separately for diagnostics and L0 loss.
            gate = (nonzero_probability > 0.5).to(dtype=log_alpha.dtype)
        return gate, nonzero_probability


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
        internal_point_flow_scale: float = 1.0,
        slot_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.dense_token_dim = int(dense_token_dim)
        self.cm_dim = int(cm_dim)
        self.num_cm_tokens = int(num_cm_tokens)
        self.num_dynamic_slots = int(num_cm_tokens)
        self.num_slot_iters = int(num_slot_iters)
        self.internal_point_flow_scale = float(internal_point_flow_scale)
        self.slot_threshold = float(slot_threshold)
        if self.internal_point_flow_scale <= 0.0:
            raise ValueError("internal_point_flow_scale must be positive.")
        # Frozen current interaction state, local geometry, endpoint motion,
        # and frozen dense contact prior.  The dataset uses hand-root poses
        # only to form this current-frame representation; wrist pose/delta is
        # never an input to Cm.
        self.hand_motion_encoder = _mlp(self.dense_token_dim + 10, cm_dim, cm_dim)
        self.slot_attention = SlotAttention(
            dim=cm_dim,
            num_slots=self.num_dynamic_slots,
            num_iterations=num_slot_iters,
        )
        # One shared gate head preserves the permutation symmetry of Slot
        # Attention: slots do not have hard-coded, index-specific roles.
        self.slot_gate_head = nn.Sequential(
            nn.LayerNorm(cm_dim),
            nn.Linear(cm_dim, cm_dim // 2),
            nn.GELU(),
            nn.Linear(cm_dim // 2, 1),
        )
        nn.init.normal_(self.slot_gate_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.slot_gate_head[-1].bias)
        self.hard_concrete = HardConcreteGate()
        self.object_context_encoder = _mlp(self.dense_token_dim + 6, cm_dim, cm_dim)
        # Object feature, Cm feature, relative object-to-soft-anchor position,
        # and soft-anchor hand normal.  Cm already contains the hand motion,
        # so exposing the pooled hand flow here would bypass the bottleneck.
        self.edge_backbone = nn.Sequential(
            nn.Linear(cm_dim * 2 + 6, cm_dim),
            nn.GELU(),
            nn.Linear(cm_dim, cm_dim // 2),
            nn.GELU(),
        )
        # Keep candidate flow at zero initially, while a tiny asymmetric logit
        # initialization lets the decoder learn slot selection immediately.
        self.edge_logit_head = nn.Linear(cm_dim // 2, 1)
        self.edge_flow_head = nn.Linear(cm_dim // 2, 3)
        nn.init.normal_(self.edge_logit_head.weight, std=1e-3)
        nn.init.zeros_(self.edge_logit_head.bias)
        nn.init.zeros_(self.edge_flow_head.weight)
        nn.init.zeros_(self.edge_flow_head.bias)

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
        obj_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # The pretrained dense encoder deliberately remains in metres.  Only
        # the trainable Cm geometry / motion path works in the configurable
        # internal unit (e.g. centimetres for scale=100).
        scale = self.internal_point_flow_scale
        obj_points_internal = obj_points * scale
        hand_points_internal = hand_points * scale
        hand_flow_internal = hand_flow * scale
        hand_input = torch.cat(
            [
                z_hand,
                hand_points_internal,
                hand_normals,
                hand_flow_internal,
                dense_hand_contact.unsqueeze(-1),
            ],
            dim=-1,
        )
        u_hand = self.hand_motion_encoder(hand_input)
        cm_tokens, cm_assignment, cm_slot_weights = self.slot_attention(u_hand)
        slot_gate_logits = self.slot_gate_head(cm_tokens).squeeze(-1)
        slot_nonzero_prob = self.hard_concrete.nonzero_probability(slot_gate_logits)
        hard_slot_mask = slot_nonzero_prob >= self.slot_threshold
        all_below_threshold = ~hard_slot_mask.any(dim=-1, keepdim=True)
        selection_score = slot_gate_logits
        if self.training:
            selection_score = selection_score + 1e-3 * torch.randn_like(selection_score)
        fallback_index = selection_score.argmax(dim=-1, keepdim=True)
        fallback_mask = torch.zeros_like(slot_nonzero_prob).scatter_(1, fallback_index, 1.0).bool()
        hard_slot_mask = torch.where(all_below_threshold, fallback_mask, hard_slot_mask)

        obj_context = self.object_context_encoder(
            torch.cat([z_obj, obj_points_internal, obj_normals], dim=-1)
        )
        cm_anchor_pos_internal = torch.einsum("bkh,bhd->bkd", cm_slot_weights, hand_points_internal)
        cm_anchor_normal = torch.einsum("bkh,bhd->bkd", cm_slot_weights, hand_normals)
        cm_anchor_normal = F.normalize(cm_anchor_normal, dim=-1, eps=1e-6)

        num_obj = obj_points.shape[1]
        edge_input = torch.cat(
            [
                obj_context.unsqueeze(2).expand(-1, -1, self.num_dynamic_slots, -1),
                cm_tokens.unsqueeze(1).expand(-1, num_obj, -1, -1),
                obj_points_internal.unsqueeze(2) - cm_anchor_pos_internal.unsqueeze(1),
                cm_anchor_normal.unsqueeze(1).expand(-1, num_obj, -1, -1),
            ],
            dim=-1,
        )
        edge_features = self.edge_backbone(edge_input)
        dynamic_logits = self.edge_logit_head(edge_features).squeeze(-1)
        dynamic_flow = self.edge_flow_head(edge_features)
        # Forward routing is strictly masked, but its backward pass follows a
        # probability-weighted soft route.  Applying straight-through here
        # (rather than before clamp/log) keeps flow-to-gate gradients alive for
        # currently inactive slots.
        soft_routing_logits = dynamic_logits + torch.log(slot_nonzero_prob[:, None, :].clamp_min(1e-6))
        soft_weight = torch.softmax(soft_routing_logits, dim=2)
        hard_routing_logits = dynamic_logits.masked_fill(~hard_slot_mask[:, None, :], -1e4)
        hard_weight = torch.softmax(hard_routing_logits, dim=2)
        edge_weight = soft_weight + (hard_weight - soft_weight).detach()
        pred_obj_flow_internal = (edge_weight.unsqueeze(-1) * dynamic_flow).sum(dim=2)
        # Restore metres at the public model boundary; runner losses and
        # visualization therefore remain unit-consistent with cached data.
        pred_obj_flow = pred_obj_flow_internal / scale
        pred_obj_flow = pred_obj_flow * obj_valid_mask.unsqueeze(-1).float()
        # q_k: mean decoder attention paid to each slot by valid object points.
        # This is a diagnostic only; it neither gates slots nor changes flow.
        valid_object = obj_valid_mask.unsqueeze(-1).float()
        decoder_slot_usage = (edge_weight * valid_object).sum(dim=1)
        decoder_slot_usage = decoder_slot_usage / valid_object.sum(dim=1).clamp_min(1.0)
        return {
            "pred_obj_flow": pred_obj_flow,
            "cm_tokens": cm_tokens,
            "cm_tokens_masked": cm_tokens * hard_slot_mask.unsqueeze(-1),
            "cm_anchor_pos": cm_anchor_pos_internal / scale,
            "cm_anchor_normal": cm_anchor_normal,
            "cm_assignment": cm_assignment,
            "cm_slot_weights": cm_slot_weights,
            "slot_gate_logits": slot_gate_logits,
            "slot_gate": hard_slot_mask.float(),
            "slot_hard_mask": hard_slot_mask,
            "slot_nonzero_prob": slot_nonzero_prob,
            "slot_fallback_used": all_below_threshold.squeeze(-1),
            "slot_fallback_index": fallback_index.squeeze(-1),
            "decoder_slot_usage": decoder_slot_usage,
            "dynamic_candidate_flow": dynamic_flow / scale,
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
            internal_point_flow_scale=float(meta.internal_point_flow_scale),
            slot_threshold=float(meta.slot_threshold),
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
