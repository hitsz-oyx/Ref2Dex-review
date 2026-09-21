"""Rigid structured interaction model with V1.0 static and V1.2 swept modes.

The implementation intentionally keeps the external contract small: complete object
geometry is encoded for structure, while hand evidence is restricted to 2 cm local
edges.  The rigid V1.0 effect is an SE(3) transform and point flow is reconstructed
from that transform, with a small optional residual path. V1.2 selects local edges by
the proposed MANO hand-point sweep while keeping the same rigid effect outputs.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Tuple
from contextlib import nullcontext

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def _profile_stage(profiler, name: str):
    return profiler.stage(name) if profiler is not None else nullcontext()


def _axis_angle_matrix(axis_angle: Tensor) -> Tensor:
    """Convert a batch of axis-angle vectors to rotation matrices."""
    theta = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / theta.clamp_min(1e-8)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(skew)
    sin_t = torch.sin(theta).unsqueeze(-1)
    one_minus = (1.0 - torch.cos(theta)).unsqueeze(-1)
    rot = eye + sin_t * skew + one_minus * (skew @ skew)
    small = theta.squeeze(-1).lt(1e-6).unsqueeze(-1).unsqueeze(-1)
    return torch.where(small, eye + skew, rot)


def transform_points(points: Tensor, delta_xi_root: Tensor) -> Tensor:
    """Apply [translation, axis-angle] to points in the current object frame."""
    if points.ndim != 3 or delta_xi_root.ndim != 2 or delta_xi_root.shape[-1] != 6:
        raise ValueError("points must be [B,N,3] and delta_xi_root must be [B,6]")
    translation, rotation = delta_xi_root[..., :3], delta_xi_root[..., 3:]
    matrix = _axis_angle_matrix(rotation)
    return torch.bmm(points, matrix.transpose(1, 2)) + translation[:, None, :]


def _axis_angle_matrix_stable(axis_angle: Tensor) -> Tensor:
    """Rodrigues map with finite derivatives at zero."""
    theta2 = axis_angle.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-16).sqrt()
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120,
                    torch.sin(theta) / theta.clamp_min(1e-8))
    b = torch.where(small, 0.5 - theta2 / 24 + theta2.square() / 720,
                    (1 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand_as(skew)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


def _stable_smallest_k(distances: Tensor, indices: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """Select lexicographic (distance, index) top-k without sorting the full axis."""
    k = min(k, distances.shape[-1])
    threshold = torch.topk(distances, k=k, largest=False, sorted=False, dim=-1).values.amax(
        dim=-1, keepdim=True)
    strictly_better = distances < threshold
    needed_ties = (k - strictly_better.sum(dim=-1, keepdim=True)).clamp_min(1)
    sentinel = torch.iinfo(indices.dtype).max
    tied_indices = indices.masked_fill(distances != threshold, sentinel)
    smallest_ties = torch.topk(tied_indices, k=k, largest=False, sorted=True, dim=-1).values
    tie_cutoff = torch.gather(smallest_ties, -1, needed_ties.sub(1).clamp_max(k - 1))
    selected = strictly_better | ((distances == threshold) & (indices <= tie_cutoff))
    selected_distances = distances.masked_fill(~selected, float("inf"))
    positions = torch.topk(selected_distances, k=k, largest=False, sorted=False, dim=-1).indices
    result_distances = torch.gather(distances, -1, positions)
    result_indices = torch.gather(indices, -1, positions)
    # Only sort the retained k entries.  Stable two-pass sorting preserves the
    # historical lower-index tie contract at a much smaller width.
    by_index = torch.argsort(result_indices, dim=-1, stable=True)
    result_distances = torch.gather(result_distances, -1, by_index)
    result_indices = torch.gather(result_indices, -1, by_index)
    by_distance = torch.argsort(result_distances, dim=-1, stable=True)
    return (torch.gather(result_distances, -1, by_distance),
            torch.gather(result_indices, -1, by_distance))


def _swept_topk_link_aabb(object_points: Tensor, hand_points: Tensor, hand_flow: Tensor,
                           hand_valid_mask: Tensor, hand_link_index: Tensor,
                           k: int, radius_m: float, object_chunk: int) -> Tuple[Tensor, Tensor]:
    """Conservative link AABB broad phase with exact segment-distance narrow phase."""
    batch, count, _ = object_points.shape
    hands = hand_points.shape[1]
    table = hand_link_index.to(device=hand_points.device, dtype=torch.long)
    if table.ndim != 2 or table.numel() == 0:
        raise ValueError("hand_link_index must be a non-empty padded [L,M] table")
    table_valid = table < hands
    gather_index = table.clamp_max(hands - 1)
    start = hand_points[:, gather_index]
    end = start + hand_flow[:, gather_index]
    point_valid = table_valid[None] & hand_valid_mask[:, gather_index]
    lower = torch.minimum(start, end).masked_fill(~point_valid[..., None], float("inf")).amin(dim=2)
    upper = torch.maximum(start, end).masked_fill(~point_valid[..., None], -float("inf")).amax(dim=2)
    link_count, link_width = table.shape
    selected_distances, selected_indices = [], []
    for o_start in range(0, count, object_chunk):
        obj = object_points[:, o_start:o_start + object_chunk]
        broad = ((obj[:, :, None] >= lower[:, None] - radius_m) &
                 (obj[:, :, None] <= upper[:, None] + radius_m)).all(dim=-1)
        triples = broad.nonzero(as_tuple=False)
        candidate_distances = torch.full(
            (*broad.shape, min(k, link_width)), float("inf"), device=obj.device, dtype=obj.dtype)
        candidate_indices = torch.full(
            candidate_distances.shape, hands, device=obj.device, dtype=torch.long)
        if triples.numel():
            batch_ids, object_ids, link_ids = triples.unbind(-1)
            point_indices = gather_index[link_ids]
            hp = hand_points[batch_ids[:, None], point_indices]
            flow = hand_flow[batch_ids[:, None], point_indices]
            relative = hp - obj[batch_ids, object_ids, None]
            alpha = (-(relative * flow).sum(-1) /
                     flow.square().sum(-1).clamp_min(1e-12)).clamp(0, 1)
            distance2 = (relative + alpha[..., None] * flow).square().sum(-1)
            valid = point_valid[batch_ids, link_ids]
            distance2 = distance2.masked_fill(~valid, float("inf"))
            link_distances, link_indices = _stable_smallest_k(
                distance2, point_indices, min(k, link_width))
            candidate_distances[batch_ids, object_ids, link_ids] = link_distances
            candidate_indices[batch_ids, object_ids, link_ids] = link_indices
        candidate_distances = candidate_distances.flatten(2)
        candidate_indices = candidate_indices.flatten(2)
        if candidate_distances.shape[-1] < k:
            padding = k - candidate_distances.shape[-1]
            candidate_distances = F.pad(candidate_distances, (0, padding), value=float("inf"))
            candidate_indices = F.pad(candidate_indices, (0, padding), value=hands)
        best_distances, best_indices = _stable_smallest_k(candidate_distances, candidate_indices, k)
        selected_distances.append(best_distances)
        selected_indices.append(best_indices)
    distance2 = torch.cat(selected_distances, dim=1)
    return torch.where(torch.isfinite(distance2), distance2.clamp_min(0).sqrt(), distance2), torch.cat(
        selected_indices, dim=1)


def swept_topk(object_points: Tensor, hand_points: Tensor, hand_flow: Tensor,
               hand_valid_mask: Optional[Tensor] = None, k: int = 32,
               object_chunk: int = 128, hand_chunk: int = 256,
               *, algorithm: str = "legacy", hand_link_index: Optional[Tensor] = None,
               radius_m: float = 0.02) -> Tuple[Tensor, Tensor]:
    """Segment-distance top-k with legacy, merge-topk and link-AABB paths."""
    if object_points.ndim != 3 or hand_points.ndim != 3 or hand_flow.shape != hand_points.shape:
        raise ValueError("Expected object [B,N,3] and matching hand/flow [B,H,3]")
    batch, count, _ = object_points.shape
    hands = hand_points.shape[1]
    if hands == 0 or k <= 0:
        raise ValueError("At least one hand point and positive k are required")
    if hand_valid_mask is None:
        hand_valid_mask = torch.ones((batch, hands), dtype=torch.bool, device=hand_points.device)
    k = min(k, hands)
    if algorithm == "link_aabb":
        if hand_link_index is None:
            raise ValueError("link_aabb search requires hand_link_index")
        return _swept_topk_link_aabb(
            object_points, hand_points, hand_flow, hand_valid_mask,
            hand_link_index, k, radius_m, object_chunk)
    if algorithm not in ("legacy", "merge"):
        raise ValueError(f"unknown swept search algorithm: {algorithm}")
    all_dist, all_idx = [], []
    for o_start in range(0, count, object_chunk):
        obj = object_points[:, o_start:o_start + object_chunk]
        best_dist = torch.full((batch, obj.shape[1], k), float("inf"), device=obj.device, dtype=obj.dtype)
        best_idx = torch.full(best_dist.shape, hands, device=obj.device, dtype=torch.long)
        for h_start in range(0, hands, hand_chunk):
            hp = hand_points[:, h_start:h_start + hand_chunk]
            flow = hand_flow[:, h_start:h_start + hand_chunk]
            relative = hp[:, None] - obj[:, :, None]
            alpha = (-(relative * flow[:, None]).sum(-1) /
                     flow.square().sum(-1)[:, None].clamp_min(1e-12)).clamp(0, 1)
            delta = relative + alpha[..., None] * flow[:, None]
            distance = (delta.square().sum(-1) if algorithm == "merge"
                        else torch.linalg.vector_norm(delta, dim=-1))
            distance = distance.masked_fill(~hand_valid_mask[:, None, h_start:h_start + hp.shape[1]], float("inf"))
            idx = torch.arange(h_start, h_start + hp.shape[1], device=obj.device).expand_as(distance)
            candidates_d = torch.cat((best_dist, distance), -1)
            candidates_i = torch.cat((best_idx, idx), -1)
            if algorithm == "legacy":
                by_index = torch.argsort(candidates_i, dim=-1, stable=True)
                candidates_d = torch.gather(candidates_d, -1, by_index)
                candidates_i = torch.gather(candidates_i, -1, by_index)
                by_distance = torch.argsort(candidates_d, dim=-1, stable=True)[..., :k]
                best_dist = torch.gather(candidates_d, -1, by_distance)
                best_idx = torch.gather(candidates_i, -1, by_distance)
            else:
                best_dist, best_idx = _stable_smallest_k(candidates_d, candidates_i, k)
        all_dist.append(best_dist)
        all_idx.append(best_idx)
    distances = torch.cat(all_dist, 1)
    if algorithm == "merge":
        distances = torch.where(torch.isfinite(distances), distances.clamp_min(0).sqrt(), distances)
    return distances, torch.cat(all_idx, 1)


class LocalInteractionEncoder(nn.Module):
    """Object-query attention over valid hand neighbors within 2 cm."""

    def __init__(self, width: int = 128, knn_k: int = 32, radius_m: float = 0.02,
                 interaction_mode: str = "static", feature_scale_m: Optional[float] = None,
                 frame_dt_s: float = 1 / 30) -> None:
        super().__init__()
        self.width, self.knn_k, self.radius_m = int(width), int(knn_k), float(radius_m)
        if interaction_mode not in ("static", "swept"):
            raise ValueError("interaction_mode must be static or swept")
        self.interaction_mode = interaction_mode
        self.feature_scale_m, self.frame_dt_s = feature_scale_m, frame_dt_s
        self.edge = _mlp((24 if feature_scale_m else 21) if interaction_mode == "swept" else 18, width, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, object_features: Tensor, object_points: Tensor, object_normals: Tensor,
                hand_points: Tensor, hand_normals: Tensor, hand_flow: Tensor,
                hand_valid_mask: Optional[Tensor] = None,
                delta_time_s: Optional[Tensor] = None,
                edge_object_chunk: Optional[int] = None,
                latency_profiler=None, swept_algorithm: str = "legacy",
                hand_link_index: Optional[Tensor] = None,
                sparse_valid_edges: bool = False,
                candidate_group_size: Optional[int] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones(hand_points.shape[:2], dtype=torch.bool, device=hand_points.device)
        if delta_time_s is None:
            delta_time_s = torch.full((object_points.shape[0],), 1 / 30, device=object_points.device, dtype=object_points.dtype)
        if delta_time_s.ndim == 2:
            delta_time_s = delta_time_s.squeeze(-1)
        if self.interaction_mode == "swept":
            with _profile_stage(latency_profiler, "swept_topk"):
                edge_distances, indices = swept_topk(
                    object_points, hand_points, hand_flow, hand_valid_mask, self.knn_k,
                    algorithm=swept_algorithm, hand_link_index=hand_link_index,
                    radius_m=self.radius_m)
            indices = indices.clamp_max(hand_points.shape[1] - 1)
        else:
            distances = torch.cdist(object_points, hand_points)
            edge_distances, indices = torch.topk(distances, k=min(self.knn_k, hand_points.shape[1]), largest=False, dim=-1)
        edge_finite = torch.isfinite(edge_distances)
        edge_distances = torch.where(edge_finite, edge_distances,
                                     torch.zeros_like(edge_distances))

        object_count = object_points.shape[1]
        if edge_object_chunk is None:
            edge_object_chunk = object_count
        edge_object_chunk = int(edge_object_chunk)
        if edge_object_chunk <= 0:
            raise ValueError("edge_object_chunk must be positive")

        with _profile_stage(latency_profiler, "edge_contact"):
            contacts, valid_chunks = [], []
            if candidate_group_size is None:
                query_all = self.query(object_features)
            else:
                group = int(candidate_group_size)
                if group <= 0 or object_features.shape[0] % group:
                    raise ValueError("candidate_group_size must divide the candidate batch")
                query_all = self.query(object_features[::group]).repeat_interleave(group, dim=0)
            hand_points_expanded = hand_points[:, None].expand(-1, object_count, -1, -1)
            hand_normals_expanded = hand_normals[:, None].expand(-1, object_count, -1, -1)
            hand_flow_expanded = hand_flow[:, None].expand(-1, object_count, -1, -1)
            hand_valid_expanded = hand_valid_mask[:, None].expand(-1, object_count, -1)
            for start in range(0, object_count, edge_object_chunk):
                stop = min(start + edge_object_chunk, object_count)
                chunk_indices = indices[:, start:stop]
                gather = chunk_indices[..., None].expand(-1, -1, -1, 3)
                hp = torch.gather(hand_points_expanded[:, start:stop], 2, gather)
                hn = torch.gather(hand_normals_expanded[:, start:stop], 2, gather)
                hf = torch.gather(hand_flow_expanded[:, start:stop], 2, gather)
                chunk_distances = edge_distances[:, start:stop]
                valid = torch.gather(hand_valid_expanded[:, start:stop], 2, chunk_indices)
                valid = valid & edge_finite[:, start:stop] & chunk_distances.lt(self.radius_m)
                relative = hp - object_points[:, start:stop, None, :]
                current_distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
                end_distance = torch.linalg.vector_norm(relative + hf, dim=-1, keepdim=True)
                on = object_normals[:, start:stop, None, :].expand_as(hn)
                dot = (on * hn).sum(-1, keepdim=True)
                normal_flow = (hf * on).sum(-1, keepdim=True)
                tangent_flow = hf - normal_flow * on
                dt = delta_time_s[:, None, None, None].expand_as(dot)
                if self.interaction_mode == "swept":
                    if self.feature_scale_m is None:
                        edge_input = torch.cat([relative, current_distance, end_distance, chunk_distances[..., None],
                                                on, hn, dot, hf, normal_flow, tangent_flow, dt], -1)
                    else:
                        s = self.feature_scale_m
                        velocity = (hf / dt.clamp_min(1e-8)) * (self.frame_dt_s / s)
                        edge_input = torch.cat([relative / s, current_distance / s, end_distance / s,
                                                chunk_distances[..., None] / s, on, hn, dot, hf / s,
                                                normal_flow / s, tangent_flow / s, dt / self.frame_dt_s,
                                                velocity], -1)
                else:
                    edge_input = torch.cat([relative, chunk_distances[..., None], on, hn, dot,
                                            hf, normal_flow, tangent_flow], -1)
                if sparse_valid_edges:
                    logits = torch.full(valid.shape, -torch.finfo(edge_input.dtype).max,
                                        device=edge_input.device, dtype=edge_input.dtype)
                    values = torch.zeros((*valid.shape, self.width), device=edge_input.device,
                                         dtype=edge_input.dtype)
                    encoded = self.edge(edge_input[valid])
                    queries = query_all[:, start:stop, None, :].expand(
                        -1, -1, valid.shape[-1], -1)[valid]
                    logits[valid] = (queries * self.key(encoded)).sum(-1) / (self.width ** 0.5)
                    values[valid] = self.value(encoded)
                else:
                    edge = self.edge(edge_input)
                    logits = (query_all[:, start:stop, None, :] * self.key(edge)).sum(-1) / (self.width ** 0.5)
                    values = self.value(edge)
                weights = torch.softmax(logits.masked_fill(~valid, -torch.finfo(logits.dtype).max), -1)
                weights = weights * valid.to(weights.dtype)
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
                contact = (weights[..., None] * values).sum(2)
                contacts.append(contact * valid.any(-1, keepdim=True).to(contact.dtype))
                valid_chunks.append(valid)
            contact = torch.cat(contacts, dim=1)
            valid = torch.cat(valid_chunks, dim=1)
        return contact, {"edge_indices": indices, "edge_distances": edge_distances,
                         "edge_valid_mask": valid, "has_interaction": valid.any(-1)}


class GroundedContactTokens(nn.Module):
    def __init__(self, width: int = 128, num_tokens: int = 16) -> None:
        super().__init__()
        self.width, self.num_tokens = int(width), int(num_tokens)
        self.assignment = nn.Linear(width, self.num_tokens)

    def forward(self, contact: Tensor, points: Tensor, normals: Tensor, active: Tensor):
        logits = self.assignment(contact).transpose(1, 2)
        logits = logits.masked_fill(~active[:, None, :], -torch.finfo(logits.dtype).max)
        weights = torch.softmax(logits, -1) * active[:, None, :].to(contact.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        tokens = torch.bmm(weights, contact)
        anchors = torch.bmm(weights, points)
        token_normals = F.normalize(torch.bmm(weights, normals), dim=-1, eps=1e-8)
        token_mask = weights.sum(-1).gt(0)
        return tokens, anchors, token_normals, token_mask


class ObjectInteractionCmv2Model(nn.Module):
    """V1.0 hybrid direct-part + grounded-token rigid effect model."""

    def __init__(self, cfg=None) -> None:
        super().__init__()
        meta = getattr(cfg, "meta", cfg) if cfg is not None else SimpleNamespace()
        self.width = int(getattr(meta, "hidden_width", 128))
        self.num_tokens = int(getattr(meta, "num_tokens", 16))
        self.use_residual = bool(getattr(meta, "use_residual", True))
        self.geometry_encoder = _mlp(6, self.width, self.width)
        self.local_interaction = LocalInteractionEncoder(self.width, int(getattr(meta, "knn_k", 32)),
                                                         float(getattr(meta, "interaction_radius_m", 0.02)),
                                                         str(getattr(meta, "interaction_mode", "static")))
        self.part_fusion = _mlp(self.width * 2, self.width, self.width)
        self.tokens = GroundedContactTokens(self.width, self.num_tokens)
        self.effect_head = _mlp(self.width * 2, self.width, self.width)
        self.root_head = nn.Linear(self.width, 6)
        self.effect_logit = nn.Linear(self.width, 1)
        self.residual_head = nn.Linear(self.width, 3) if self.use_residual else None

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        required = ("obj_points", "obj_normals", "hand_points", "hand_normals", "hand_flow")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing V1.0 fields: {missing}")
        points, normals = batch["obj_points"], F.normalize(batch["obj_normals"], dim=-1, eps=1e-8)
        geo = self.geometry_encoder(torch.cat([points, normals], -1))
        contact, diagnostics = self.local_interaction(geo, points, normals, batch["hand_points"],
                                                       batch["hand_normals"], batch["hand_flow"],
                                                       batch.get("hand_valid_mask"), batch.get("delta_time_s"),
                                                       batch.get("interaction_object_chunk"))
        active = diagnostics["has_interaction"]
        direct = self.part_fusion(torch.cat([geo.mean(1), contact.sum(1) / active.sum(1, keepdim=True).clamp_min(1).to(contact.dtype)], -1))
        tokens, anchors, token_normals, token_mask = self.tokens(contact, points, normals, active)
        token_summary = tokens.mean(1)
        fused = self.effect_head(torch.cat([direct, token_summary], -1))
        delta_xi = self.root_head(fused)
        effect_probability = torch.sigmoid(self.effect_logit(fused)).squeeze(-1)
        structured_flow = transform_points(points, delta_xi) - points
        residual = self.residual_head(fused)[:, None, :].expand_as(points) if self.residual_head is not None else torch.zeros_like(points)
        return {"delta_xi_root": delta_xi, "p_effect": effect_probability, "obj_flow_structured": structured_flow,
                "obj_flow_residual": residual, "obj_flow_pred": structured_flow + residual,
                "contact_features": contact, "contact_active": active, "tokens": tokens,
                "token_anchors": anchors, "token_normals": token_normals, "token_mask": token_mask,
                **diagnostics}


def object_interaction_loss(output: Dict[str, Tensor], batch: Dict[str, Tensor], residual_weight: float = 1e-3) -> Dict[str, Tensor]:
    """Compute V1.0 losses without changing the point-flow target contract."""
    target = batch["obj_flow_gt"]
    flow = F.smooth_l1_loss(output["obj_flow_pred"], target)
    has_effect = target.square().sum(-1).sqrt().amax(-1).gt(1e-6).to(output["p_effect"].dtype)
    effect = F.binary_cross_entropy(output["p_effect"].clamp(1e-5, 1 - 1e-5), has_effect)
    residual = output["obj_flow_residual"].square().mean()
    total = flow + 0.1 * effect + residual_weight * residual
    return {"total": total, "flow": flow, "effect": effect, "residual": residual}


def _masked_attention_pool(values: Tensor, mask: Tensor, score: nn.Module) -> Tensor:
    logits = score(values).squeeze(-1)
    weights = torch.softmax(logits.masked_fill(~mask, -torch.finfo(logits.dtype).max), dim=1)
    weights = weights * mask.to(weights.dtype)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
    return (weights[..., None] * values).sum(1)


class CompetitiveContactTokens(nn.Module):
    """Assign each active object point across tokens; retain physical mass."""

    def __init__(self, width: int, num_tokens: int) -> None:
        super().__init__()
        self.assignment = _mlp(width + 6, width, num_tokens)

    def forward(self, contact: Tensor, points: Tensor, normals: Tensor, active: Tensor,
                object_scale: Tensor):
        spatial = torch.cat((contact, points / object_scale[:, None, None], normals), dim=-1)
        weights = torch.softmax(self.assignment(spatial), dim=-1) * active[..., None].to(contact.dtype)
        mass = weights.sum(1)
        normalized = weights / mass[:, None].clamp_min(1e-8)
        tokens = torch.bmm(normalized.transpose(1, 2), contact)
        anchors = torch.bmm(normalized.transpose(1, 2), points)
        token_normals = F.normalize(torch.bmm(normalized.transpose(1, 2), normals), dim=-1, eps=1e-8)
        mask = mass.gt(1e-8)
        return tokens, anchors, token_normals, mass, mask


class ObjectInteractionCmv2V13Model(nn.Module):
    """V1.3.2 rigid-only spatial fusion; earlier models remain untouched."""

    architecture_version = "v1_3_rigid_only"

    def __init__(self, cfg) -> None:
        super().__init__()
        width = int(cfg.hidden_width)
        num_tokens = int(cfg.num_tokens)
        if cfg.use_residual or cfg.interaction_mode != "swept":
            raise ValueError("V1.3 requires swept interaction without residual")
        self.feature_scale_m = float(cfg.feature_scale_m)
        self.geometry_encoder = _mlp(7, width, width)
        self.local_interaction = LocalInteractionEncoder(
            width, int(cfg.knn_k), float(cfg.interaction_radius_m), "swept",
            feature_scale_m=self.feature_scale_m, frame_dt_s=float(cfg.frame_dt_s))
        self.global_score = nn.Linear(width, 1)
        self.contact_score = nn.Linear(width, 1)
        self.part_fusion = _mlp(2 * width, width, width)
        self.tokens = CompetitiveContactTokens(width, num_tokens)
        self.token_encoder = _mlp(width + 7, width, width)
        self.cross_attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)
        self.fusion_head = _mlp(2 * width, width, width)
        self.root_head = nn.Linear(width, 6)
        self.cm_projection = nn.Linear(width, 32)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        required = ("obj_points", "obj_normals", "hand_points", "hand_normals", "hand_flow")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Missing V1.3 fields: {missing}")
        points = batch["obj_points"]
        normals = F.normalize(batch["obj_normals"], dim=-1, eps=1e-8)
        candidate_group_size = batch.get("candidate_group_size")
        group = int(candidate_group_size) if candidate_group_size is not None else 1
        if group <= 0 or points.shape[0] % group:
            raise ValueError("candidate_group_size must divide the candidate batch")
        unique_points = points[::group]
        unique_normals = normals[::group]
        unique_centered = unique_points - unique_points.mean(1, keepdim=True)
        unique_scale = unique_centered.square().sum(-1).mean(1).sqrt()
        if not torch.isfinite(unique_scale).all() or (unique_scale <= 1e-8).any():
            raise ValueError("Degenerate object geometry")
        unique_log_scale = torch.log(unique_scale / self.feature_scale_m)
        profiler = batch.get("_latency_profiler")
        with _profile_stage(profiler, "geometry_encoder"):
            unique_geo = self.geometry_encoder(torch.cat((
                unique_centered / unique_scale[:, None, None], unique_normals,
                unique_log_scale[:, None, None].expand(-1, points.shape[1], 1)), dim=-1))
            geo = unique_geo.repeat_interleave(group, dim=0)
        object_scale = unique_scale.repeat_interleave(group, dim=0)
        contact, diagnostics = self.local_interaction(
            geo, points, normals, batch["hand_points"], batch["hand_normals"],
            batch["hand_flow"], batch.get("hand_valid_mask"), batch.get("delta_time_s"),
            batch.get("interaction_object_chunk"), profiler,
            batch.get("swept_algorithm", "legacy"), batch.get("hand_link_index"),
            bool(batch.get("sparse_valid_edges", False)), candidate_group_size)
        with _profile_stage(profiler, "token_attention_effect"):
            active = diagnostics["has_interaction"]
            global_feature = _masked_attention_pool(
                geo, torch.ones_like(active), self.global_score)
            interaction_feature = _masked_attention_pool(contact, active, self.contact_score)
            direct = self.part_fusion(torch.cat((global_feature, interaction_feature), dim=-1))
            tokens, anchors, token_normals, mass, token_mask = self.tokens(
                contact, points, normals, active, object_scale)
            token_input = torch.cat((tokens, anchors / object_scale[:, None, None],
                                     token_normals, torch.log(mass.clamp_min(1e-8))[..., None]), dim=-1)
            spatial_tokens = self.token_encoder(token_input)
            # MultiheadAttention needs at least one unmasked key for each sample.
            safe_mask = token_mask.clone()
            empty = ~safe_mask.any(1)
            safe_mask[empty, 0] = True
            attended, _ = self.cross_attention(
                direct[:, None], spatial_tokens, spatial_tokens,
                key_padding_mask=~safe_mask, need_weights=False)
            attended = attended[:, 0] * (~empty)[:, None].to(direct.dtype)
            fused = self.fusion_head(torch.cat((direct, attended), dim=-1))
            delta_xi = self.root_head(fused)
            rotation = _axis_angle_matrix_stable(delta_xi[:, 3:])
            structured_flow = torch.bmm(points, rotation.transpose(1, 2)) + delta_xi[:, None, :3] - points
            cm_tokens = self.cm_projection(spatial_tokens) * token_mask[..., None].to(points.dtype)
        return {"delta_xi_root": delta_xi,
                "obj_flow_structured": structured_flow,
                "obj_flow_residual": torch.zeros_like(points),
                "obj_flow_pred": structured_flow,
                "contact_features": contact, "contact_active": active,
                "tokens": tokens, "cm_tokens": cm_tokens,
                "token_anchors": anchors, "token_normals": token_normals,
                "token_mass": mass, "token_mask": token_mask, **diagnostics}


def object_interaction_v13_loss(output: Dict[str, Tensor], batch: Dict[str, Tensor],
                                scale_m: float = 0.02) -> Dict[str, Tensor]:
    """Direct rigid supervision with translation, rotation, and flow only."""
    pred = output["delta_xi_root"]
    translation = F.smooth_l1_loss(pred[:, :3] / scale_m,
                                   batch["delta_translation_gt"] / scale_m)
    pred_rotation = _axis_angle_matrix_stable(pred[:, 3:])
    relative = batch["delta_rotation_gt"].transpose(1, 2) @ pred_rotation
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    skew = torch.stack((relative[:, 2, 1] - relative[:, 1, 2],
                        relative[:, 0, 2] - relative[:, 2, 0],
                        relative[:, 1, 0] - relative[:, 0, 1]), dim=-1)
    angle = torch.atan2(torch.linalg.vector_norm(skew, dim=-1), trace - 1)
    points = batch["obj_points"]
    radius = (points - points.mean(1, keepdim=True)).square().sum(-1).mean(1).sqrt()
    rotation = (angle * radius / scale_m).mean()
    flow = F.smooth_l1_loss(output["obj_flow_pred"] / scale_m,
                            batch["obj_flow_gt"] / scale_m)
    total = translation + rotation + flow
    return {"total": total, "translation": translation, "rotation": rotation,
            "flow": flow}
