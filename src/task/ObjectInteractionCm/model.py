"""Independent object-side dual-hand interaction model for V1.1."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import ObjectFlowDecoder, SharedHandFlowDecoder
from .slot_attention import SlotAttention


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class LocalHandInteraction(nn.Module):
    """KNN + radius-masked cross-attention from an object point to hand edges."""

    def __init__(self, *, dim: int = 32, knn_k: int = 8, radius_m: float = 0.05) -> None:
        super().__init__()
        if dim <= 0 or knn_k <= 0 or radius_m <= 0.0:
            raise ValueError("dim, knn_k and radius_m must be positive")
        self.dim = int(dim)
        self.knn_k = int(knn_k)
        self.radius_m = float(radius_m)
        self.edge_encoder = _mlp(11, 64, self.dim)
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.scale = self.dim ** -0.5

    def forward(
        self,
        object_features: torch.Tensor,
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_flow: torch.Tensor,
        hand_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bsz, num_obj, _ = object_points.shape
        num_hand = hand_points.shape[1]
        k = min(self.knn_k, num_hand)
        distances = torch.cdist(object_points, hand_points)
        edge_distances, indices = torch.topk(distances, k=k, largest=False, dim=-1)
        gather_shape = (bsz, num_obj, k, 3)
        hand_points_expanded = hand_points[:, None, :, :].expand(-1, num_obj, -1, -1)
        hand_normals_expanded = hand_normals[:, None, :, :].expand(-1, num_obj, -1, -1)
        hand_flow_expanded = hand_flow[:, None, :, :].expand(-1, num_obj, -1, -1)
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones((bsz, num_hand), dtype=torch.bool, device=hand_points.device)
        hand_valid_expanded = hand_valid_mask[:, None, :].expand(-1, num_obj, -1)
        index_expanded = indices[..., None].expand(*gather_shape)
        local_hand = torch.gather(hand_points_expanded, 2, index_expanded)
        local_normals = torch.gather(hand_normals_expanded, 2, index_expanded)
        local_flow = torch.gather(hand_flow_expanded, 2, index_expanded)
        local_valid = torch.gather(hand_valid_expanded, 2, indices)
        relative = local_hand - object_points[:, :, None, :]
        normal_dot = (object_normals[:, :, None, :] * local_normals).sum(dim=-1, keepdim=True)
        edge_input = torch.cat(
            [relative, edge_distances[..., None], local_normals, normal_dot, local_flow], dim=-1
        )
        edge_features = self.edge_encoder(edge_input)
        q = self.query(object_features)[:, :, None, :]
        key = self.key(edge_features)
        value = self.value(edge_features)
        logits = (q * key).sum(dim=-1) * self.scale
        valid = (edge_distances <= self.radius_m) & local_valid
        masked_logits = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = weights * valid.to(weights.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(normalizer > 0.0, weights / normalizer.clamp_min(1e-8), torch.zeros_like(weights))
        interaction = (weights[..., None] * value).sum(dim=2)
        diagnostics = {
            "edge_distances": edge_distances,
            "edge_indices": indices,
            "edge_valid_mask": valid,
            "attention_weights": weights,
            "has_interaction": valid.any(dim=-1),
            "mean_valid_neighbors": valid.float().sum(dim=-1),
        }
        return interaction, diagnostics


class ObjectInteractionCmModel(nn.Module):
    """Object tokens plus local dual-hand interaction compressed to 16x32 Cm."""

    def __init__(self, cfg, *, condition_shape=None, target_shape=None) -> None:
        del condition_shape, target_shape
        # BaseRunner attaches the task meta object to ``model_cfg``.  Keeping
        # the fallback makes direct unit construction with a bare model config
        # deterministic and does not alter the runner contract.
        meta = getattr(cfg, "meta", cfg)
        dim = int(getattr(meta, "feature_dim", 32))
        self.feature_dim = dim
        self.num_slots = int(getattr(meta, "num_cm_tokens", 16))
        super().__init__()
        self.object_encoder = _mlp(6, 64, dim)
        self.local_interaction = LocalHandInteraction(
            dim=dim,
            knn_k=int(getattr(meta, "knn_k", 8)),
            radius_m=float(getattr(meta, "interaction_radius_m", 0.05)),
        )
        self.fusion = _mlp(dim * 2, 64, dim)
        self.slot_attention = SlotAttention(
            dim=dim,
            num_slots=self.num_slots,
            num_iterations=int(getattr(meta, "slot_iters", 3)),
        )
        self.object_decoder = ObjectFlowDecoder(dim=dim)
        self.hand_decoder = SharedHandFlowDecoder(dim=dim)

    @staticmethod
    def _hands(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "hand_points" in batch:
            return batch["hand_points"], batch["hand_normals"], batch["hand_flow"]
        required = (
            "left_hand_points", "right_hand_points", "left_hand_normals", "right_hand_normals",
            "left_hand_flow", "right_hand_flow",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing dual-hand fields: {missing}")
        return (
            torch.cat([batch["left_hand_points"], batch["right_hand_points"]], dim=1),
            torch.cat([batch["left_hand_normals"], batch["right_hand_normals"]], dim=1),
            torch.cat([batch["left_hand_flow"], batch["right_hand_flow"]], dim=1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        object_points = batch["obj_points"].float()
        object_normals = F.normalize(batch["obj_normals"].float(), dim=-1, eps=1e-6)
        hand_points, hand_normals, hand_flow = self._hands(batch)
        hand_valid_mask = batch.get("hand_valid_mask")
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones(hand_points.shape[:2], dtype=torch.bool, device=hand_points.device)
        hand_points = hand_points.float()
        hand_normals = F.normalize(hand_normals.float(), dim=-1, eps=1e-6)
        hand_flow = hand_flow.float()
        object_features = self.object_encoder(torch.cat([object_points, object_normals], dim=-1))
        interaction, interaction_diag = self.local_interaction(
            object_features, object_points, object_normals, hand_points, hand_normals, hand_flow, hand_valid_mask
        )
        object_tokens = self.fusion(torch.cat([object_features, interaction], dim=-1))
        cm_tokens, assignment, slot_weights = self.slot_attention(object_tokens)
        anchor_positions = torch.einsum("bsn,bnd->bsd", slot_weights, object_points)
        anchor_normals = F.normalize(
            torch.einsum("bsn,bnd->bsd", slot_weights, object_normals), dim=-1, eps=1e-6
        )
        pred_obj_flow, object_routing, object_candidates = self.object_decoder(
            object_tokens, cm_tokens, object_points, anchor_positions, anchor_normals
        )
        pred_hand_flow, hand_routing = self.hand_decoder(
            hand_points, hand_normals, cm_tokens, anchor_positions, anchor_normals
        )
        output = {
            "pred_obj_flow": pred_obj_flow,
            "pred_hand_flow": pred_hand_flow,
            "cm_tokens": cm_tokens,
            "cm_assignment": assignment,
            "cm_slot_weights": slot_weights,
            "cm_anchor_pos": anchor_positions,
            "cm_anchor_normal": anchor_normals,
            "object_routing_weights": object_routing,
            "object_candidate_flow": object_candidates,
            "hand_routing_weights": hand_routing,
        }
        output.update({f"interaction/{key}": value for key, value in interaction_diag.items()})
        return output
