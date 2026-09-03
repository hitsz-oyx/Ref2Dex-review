"""Object-side interaction-Cm model, V1.2.1."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .decoder import (
    LegacyObjectFlowDecoder,
    LegacySharedHandFlowDecoder,
    ObjectFlowDecoder,
    SharedHandFlowDecoder,
)
from .slot_attention import SlotAttention


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim), nn.GELU())


def _load_scales(meta) -> dict[str, float]:
    values = {
        "s_geo": float(getattr(meta, "geometry_scale_m", 0.05) or 0.05),
        "s_hand_flow": float(getattr(meta, "hand_flow_input_scale", 1.0) or 1.0),
        "s_obj_flow": float(getattr(meta, "object_flow_target_scale", 1.0) or 1.0),
    }
    path_value = str(getattr(meta, "scale_manifest_path", "") or "").strip()
    if path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            scales = payload.get("scales", payload)
            for key in values:
                if key in scales:
                    values[key] = float(scales[key])
    if any(value <= 0.0 for value in values.values()):
        raise ValueError(f"All ObjectInteractionCm scales must be positive, got {values}")
    return values


class LegacyLocalHandInteraction(nn.Module):
    """V1.1 state layout for strict loading of old checkpoints."""

    def __init__(self, *, dim: int = 32, knn_k: int = 8, radius_m: float = 0.05) -> None:
        super().__init__()
        self.dim, self.knn_k, self.radius_m = int(dim), int(knn_k), float(radius_m)
        self.edge_encoder = _mlp(11, 64, self.dim)
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.scale = self.dim ** -0.5

    def forward(self, object_features, object_points, object_normals, hand_points, hand_normals, hand_flow, hand_valid_mask=None):
        bsz, num_obj, _ = object_points.shape
        num_hand = hand_points.shape[1]
        k = min(self.knn_k, num_hand)
        distances = torch.cdist(object_points, hand_points)
        edge_distances, indices = torch.topk(distances, k=k, largest=False, dim=-1)
        hand_points_expanded = hand_points[:, None].expand(-1, num_obj, -1, -1)
        hand_normals_expanded = hand_normals[:, None].expand(-1, num_obj, -1, -1)
        hand_flow_expanded = hand_flow[:, None].expand(-1, num_obj, -1, -1)
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones((bsz, num_hand), dtype=torch.bool, device=hand_points.device)
        local_valid = torch.gather(hand_valid_mask[:, None].expand(-1, num_obj, -1), 2, indices)
        gather_idx = indices[..., None].expand(-1, -1, -1, 3)
        local_hand = torch.gather(hand_points_expanded, 2, gather_idx)
        local_normals = torch.gather(hand_normals_expanded, 2, gather_idx)
        local_flow = torch.gather(hand_flow_expanded, 2, gather_idx)
        relative = local_hand - object_points[:, :, None, :]
        normal_dot = (object_normals[:, :, None, :] * local_normals).sum(dim=-1, keepdim=True)
        edge_features = self.edge_encoder(torch.cat([relative, edge_distances[..., None], local_normals, normal_dot, local_flow], dim=-1))
        logits = (self.query(object_features)[:, :, None, :] * self.key(edge_features)).sum(dim=-1) * self.scale
        valid = (edge_distances <= self.radius_m) & local_valid
        weights = torch.softmax(logits.masked_fill(~valid, -torch.finfo(logits.dtype).max), dim=-1) * valid.to(logits.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(normalizer > 0.0, weights / normalizer.clamp_min(1e-8), torch.zeros_like(weights))
        interaction = (weights[..., None] * self.value(edge_features)).sum(dim=2)
        return interaction, {
            "edge_distances": edge_distances, "edge_indices": indices, "edge_valid_mask": valid,
            "attention_weights": weights, "has_interaction": valid.any(dim=-1),
            "mean_valid_neighbors": valid.float().sum(dim=-1),
        }


class ObjectInteractionCmModel(nn.Module):
    """Delayed-compression object-side Cm with a B-preserving masked path."""

    def __init__(self, cfg, *, condition_shape=None, target_shape=None) -> None:
        del condition_shape, target_shape
        super().__init__()
        meta = getattr(cfg, "meta", cfg)
        version = str(getattr(cfg, "modification_version", "") or getattr(meta, "modification_version", ""))
        self.legacy = version.startswith("V1.1")
        if self.legacy:
            self._init_legacy(meta)
            return
        processing_dim = int(getattr(meta, "processing_dim", getattr(meta, "feature_dim", 128)))
        cm_dim = int(getattr(meta, "cm_dim", 32))
        self.feature_dim = cm_dim
        self.processing_dim = processing_dim
        self.num_slots = int(getattr(meta, "num_cm_tokens", 16))
        self.scale_values = _load_scales(meta)
        self.object_encoder = _mlp(6, 64, processing_dim)
        self.local_interaction = LocalHandInteraction(
            dim=processing_dim, knn_k=int(getattr(meta, "knn_k", 8)),
            radius_m=float(getattr(meta, "interaction_radius_m", 0.05)),
            geometry_scale=self.scale_values["s_geo"], hand_flow_scale=self.scale_values["s_hand_flow"],
        )
        self.fusion = _mlp(processing_dim * 3 + 4, 128, processing_dim)
        self.fusion_norm = nn.LayerNorm(processing_dim)
        self.slot_attention = SlotAttention(dim=processing_dim, num_slots=self.num_slots, num_iterations=int(getattr(meta, "slot_iters", 3)))
        self.cm_projection = nn.Linear(processing_dim, cm_dim)
        self.object_decoder = ObjectFlowDecoder(cm_dim=cm_dim, hidden_dim=processing_dim)
        self.hand_decoder = SharedHandFlowDecoder(cm_dim=cm_dim, processing_dim=processing_dim, hidden_dim=processing_dim)
        self.object_flow_target_scale = float(self.scale_values["s_obj_flow"])
        self.hand_flow_target_scale = float(self.scale_values["s_hand_flow"])

    def _init_legacy(self, meta) -> None:
        dim = int(getattr(meta, "feature_dim", 32))
        self.feature_dim, self.processing_dim = dim, dim
        self.num_slots = int(getattr(meta, "num_cm_tokens", 16))
        self.object_encoder = _mlp(6, 64, dim)
        self.local_interaction = LegacyLocalHandInteraction(dim=dim, knn_k=int(getattr(meta, "knn_k", 8)), radius_m=float(getattr(meta, "interaction_radius_m", 0.05)))
        self.fusion = _mlp(dim * 2, 64, dim)
        self.slot_attention = SlotAttention(dim=dim, num_slots=self.num_slots, num_iterations=int(getattr(meta, "slot_iters", 3)))
        self.object_decoder = LegacyObjectFlowDecoder(dim=dim)
        self.hand_decoder = LegacySharedHandFlowDecoder(dim=dim)

    @staticmethod
    def _hands(batch):
        if "hand_points" in batch:
            return batch["hand_points"], batch["hand_normals"], batch["hand_flow"]
        required = ("left_hand_points", "right_hand_points", "left_hand_normals", "right_hand_normals", "left_hand_flow", "right_hand_flow")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing dual-hand fields: {missing}")
        return (
            torch.cat([batch["left_hand_points"], batch["right_hand_points"]], dim=1),
            torch.cat([batch["left_hand_normals"], batch["right_hand_normals"]], dim=1),
            torch.cat([batch["left_hand_flow"], batch["right_hand_flow"]], dim=1),
        )

    def forward(self, batch):
        object_points = batch["obj_points"].float()
        object_normals = F.normalize(batch["obj_normals"].float(), dim=-1, eps=1e-6)
        hand_points, hand_normals, hand_flow = self._hands(batch)
        hand_valid_mask = batch.get("hand_valid_mask")
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones(hand_points.shape[:2], dtype=torch.bool, device=hand_points.device)
        hand_points, hand_normals, hand_flow = hand_points.float(), F.normalize(hand_normals.float(), dim=-1, eps=1e-6), hand_flow.float()
        object_features = self.object_encoder(torch.cat([object_points, object_normals], dim=-1))
        if self.legacy:
            interaction, diag = self.local_interaction(object_features, object_points, object_normals, hand_points, hand_normals, hand_flow, hand_valid_mask)
            object_tokens = self.fusion(torch.cat([object_features, interaction], dim=-1))
            cm_result = self.slot_attention(object_tokens)
            cm_tokens, assignment, slot_weights = cm_result[:3]
            anchor_positions = torch.einsum("bsn,bnd->bsd", slot_weights, object_points)
            anchor_normals = F.normalize(torch.einsum("bsn,bnd->bsd", slot_weights, object_normals), dim=-1, eps=1e-6)
            pred_obj, routing, candidates = self.object_decoder(object_tokens, cm_tokens, object_points, anchor_positions, anchor_normals)
            pred_hand, hand_routing = self.hand_decoder(hand_points, hand_normals, cm_tokens, anchor_positions, anchor_normals)
            output = {"pred_obj_flow": pred_obj, "pred_hand_flow": pred_hand, "cm_tokens": cm_tokens, "cm_assignment": assignment, "cm_slot_weights": slot_weights, "cm_anchor_pos": anchor_positions, "cm_anchor_normal": anchor_normals, "object_routing_weights": routing, "object_candidate_flow": candidates, "hand_routing_weights": hand_routing, "sample_valid": diag["has_interaction"].any(dim=1), "sampled_active_count": diag["has_interaction"].sum(dim=1)}
            output.update({f"interaction/{key}": value for key, value in diag.items()})
            return output

        interaction, diag = self.local_interaction(object_features, object_points, object_normals, hand_points, hand_normals, hand_flow, hand_valid_mask)
        fused = self.fusion(torch.cat([object_features, diag["interaction_geo"], diag["interaction_flow"], diag["flow_mean"], diag["flow_magnitude"]], dim=-1))
        object_tokens = self.fusion_norm(object_features + fused)
        object_valid = batch.get("obj_valid_mask", torch.ones(object_points.shape[:2], dtype=torch.bool, device=object_points.device)).bool()
        pool_mask = object_valid & diag["has_interaction"]
        sample_valid = pool_mask.any(dim=1)
        cm_raw, assignment, slot_weights, dummy_used = self.slot_attention(object_tokens, pool_mask)
        cm_tokens = self.cm_projection(cm_raw)
        anchor_positions = torch.einsum("bsn,bnd->bsd", slot_weights, object_points)
        anchor_normals = F.normalize(torch.einsum("bsn,bnd->bsd", slot_weights, object_normals), dim=-1, eps=1e-6)
        pred_obj_norm, object_usage = self.object_decoder(object_points, object_normals, cm_tokens, anchor_positions, anchor_normals)
        pred_hand_norm, hand_usage = self.hand_decoder(hand_points, hand_normals, cm_tokens, anchor_positions, anchor_normals)
        output = {
            "pred_obj_flow": pred_obj_norm * self.object_flow_target_scale,
            "pred_hand_flow": pred_hand_norm * self.hand_flow_target_scale,
            "cm_tokens": cm_tokens, "cm_raw": cm_raw, "cm_assignment": assignment,
            "cm_slot_weights": slot_weights, "cm_anchor_pos": anchor_positions,
            "cm_anchor_normal": anchor_normals, "sample_valid": sample_valid,
            "sampled_active_count": pool_mask.sum(dim=1),
            "slot_dummy_used": dummy_used, "object_contribution_usage": object_usage,
            "hand_contribution_usage": hand_usage,
        }
        for key, value in diag.items():
            output[f"interaction/{key}"] = value
        return output


class LocalHandInteraction(nn.Module):
    """Mask-before-top-k local interaction with separate geometry/flow values."""

    def __init__(self, *, dim: int = 128, knn_k: int = 8, radius_m: float = 0.05, geometry_scale: float = 0.05, hand_flow_scale: float = 1.0) -> None:
        super().__init__()
        if min(dim, knn_k, radius_m, geometry_scale, hand_flow_scale) <= 0:
            raise ValueError("dim, knn_k, radius_m and scales must be positive")
        self.dim, self.knn_k, self.radius_m = int(dim), int(knn_k), float(radius_m)
        self.geometry_scale, self.hand_flow_scale = float(geometry_scale), float(hand_flow_scale)
        self.edge_geo_encoder = _mlp(8, 64, self.dim)
        self.edge_flow_encoder = _mlp(3, 64, self.dim)
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value_geo = nn.Linear(self.dim, self.dim, bias=False)
        self.value_flow = nn.Linear(self.dim, self.dim, bias=False)
        self.key_norm = nn.LayerNorm(self.dim)
        self.scale = self.dim ** -0.5

    def forward(self, object_features, object_points, object_normals, hand_points, hand_normals, hand_flow, hand_valid_mask=None):
        bsz, num_obj, _ = object_points.shape
        num_hand = hand_points.shape[1]
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones((bsz, num_hand), dtype=torch.bool, device=hand_points.device)
        distances = torch.cdist(object_points, hand_points)
        distances = distances.masked_fill(~hand_valid_mask[:, None, :].bool(), float("inf"))
        k = min(self.knn_k, num_hand)
        edge_distances, indices = torch.topk(distances, k=k, largest=False, dim=-1)
        gather_idx = indices[..., None].expand(-1, -1, -1, 3)
        local_hand = torch.gather(hand_points[:, None].expand(-1, num_obj, -1, -1), 2, gather_idx)
        local_normals = torch.gather(hand_normals[:, None].expand(-1, num_obj, -1, -1), 2, gather_idx)
        local_flow = torch.gather(hand_flow[:, None].expand(-1, num_obj, -1, -1), 2, gather_idx)
        local_valid = torch.gather(hand_valid_mask[:, None].expand(-1, num_obj, -1), 2, indices)
        relative = local_hand - object_points[:, :, None, :]
        normal_dot = (object_normals[:, :, None, :] * local_normals).sum(dim=-1, keepdim=True)
        # Invalid/padded neighbors have +inf distance for selection; replace
        # that sentinel before the MLP so zero attention weights cannot create
        # 0*NaN in the value aggregation.
        safe_distances = torch.where(torch.isfinite(edge_distances), edge_distances, torch.zeros_like(edge_distances))
        geo_input = torch.cat([relative / self.geometry_scale, safe_distances[..., None] / self.geometry_scale, local_normals, normal_dot], dim=-1)
        flow_scaled = local_flow / self.hand_flow_scale
        edge_geo, edge_flow = self.edge_geo_encoder(geo_input), self.edge_flow_encoder(flow_scaled)
        keys = self.key(self.key_norm(edge_geo + edge_flow))
        logits = (self.query(object_features)[:, :, None, :] * keys).sum(dim=-1) * self.scale
        valid = (edge_distances <= self.radius_m) & local_valid
        weights = torch.softmax(logits.masked_fill(~valid, -torch.finfo(logits.dtype).max), dim=-1) * valid.to(logits.dtype)
        normalizer = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(normalizer > 0.0, weights / normalizer.clamp_min(1e-8), torch.zeros_like(weights))
        interaction_geo = (weights[..., None] * self.value_geo(edge_geo)).sum(dim=2)
        interaction_flow = (weights[..., None] * self.value_flow(edge_flow)).sum(dim=2)
        flow_mean = (weights[..., None] * flow_scaled).sum(dim=2)
        flow_magnitude = (weights * flow_scaled.norm(dim=-1)).sum(dim=2, keepdim=True)
        diag = {
            "edge_distances": safe_distances, "edge_indices": indices, "edge_valid_mask": valid,
            "attention_weights": weights, "has_interaction": valid.any(dim=-1),
            "mean_valid_neighbors": valid.float().sum(dim=-1), "interaction_geo": interaction_geo,
            "interaction_flow": interaction_flow, "flow_mean": flow_mean, "flow_magnitude": flow_magnitude,
        }
        return interaction_geo + interaction_flow, diag
