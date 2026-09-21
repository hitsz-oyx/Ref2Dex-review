"""V1.12 endpoint interaction and permutation-equivariant per-part SE(3)."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import _axis_angle_matrix_stable, _mlp


PART_SE3_VERSION = "v1_12_endpoint_part_se3"


def _stable_smallest(distances: Tensor, indices: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Select by distance with the lower source index winning exact ties."""
    by_index = torch.argsort(indices, dim=-1, stable=True)
    distances = torch.gather(distances, -1, by_index)
    indices = torch.gather(indices, -1, by_index)
    by_distance = torch.argsort(distances, dim=-1, stable=True)[..., :k]
    return torch.gather(distances, -1, by_distance), torch.gather(indices, -1, by_distance)


def endpoint_union_topk(
    object_points: Tensor,
    hand_points: Tensor,
    hand_flow: Tensor,
    hand_valid_mask: Tensor | None = None,
    *,
    k: int = 32,
    object_chunk: int = 128,
    hand_chunk: int = 256,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Top-k of the start/end top-k union, ranked by minimum endpoint distance.

    The object query is fixed at its start-frame position. Returned tensors are
    ``(minimum_distance, index, start_distance, end_distance)``.
    """
    if object_points.ndim != 3 or hand_points.ndim != 3 or hand_flow.shape != hand_points.shape:
        raise ValueError("expected object [B,N,3] and matching hand/flow [B,H,3]")
    if object_points.shape[0] != hand_points.shape[0] or object_points.shape[-1] != 3:
        raise ValueError("endpoint KNN batch or coordinate shape mismatch")
    batch, object_count, _ = object_points.shape
    hand_count = hand_points.shape[1]
    if hand_count <= 0 or k <= 0 or object_chunk <= 0 or hand_chunk <= 0:
        raise ValueError("endpoint KNN requires positive counts, k, and chunks")
    if hand_valid_mask is None:
        hand_valid_mask = torch.ones((batch, hand_count), dtype=torch.bool, device=hand_points.device)
    if hand_valid_mask.shape != (batch, hand_count):
        raise ValueError("hand_valid_mask must be [B,H]")
    keep = min(int(k), hand_count)
    if torch.any(hand_valid_mask.sum(-1) < keep):
        raise ValueError(f"each sample needs at least {keep} valid hand points")
    endpoint = hand_points + hand_flow
    output_min, output_idx, output_start, output_end = [], [], [], []
    for object_start in range(0, object_count, object_chunk):
        query = object_points[:, object_start:object_start + object_chunk]
        width = query.shape[1]
        start_best = torch.full((batch, width, keep), float("inf"), dtype=query.dtype, device=query.device)
        end_best = torch.full_like(start_best, float("inf"))
        start_idx = torch.full(start_best.shape, hand_count, dtype=torch.long, device=query.device)
        end_idx = torch.full_like(start_idx, hand_count)
        for hand_start in range(0, hand_count, hand_chunk):
            stop = min(hand_start + hand_chunk, hand_count)
            start_distance = torch.linalg.vector_norm(
                hand_points[:, None, hand_start:stop] - query[:, :, None], dim=-1)
            end_distance = torch.linalg.vector_norm(
                endpoint[:, None, hand_start:stop] - query[:, :, None], dim=-1)
            valid = hand_valid_mask[:, None, hand_start:stop]
            start_distance = start_distance.masked_fill(~valid, float("inf"))
            end_distance = end_distance.masked_fill(~valid, float("inf"))
            source_idx = torch.arange(hand_start, stop, device=query.device).expand_as(start_distance)
            start_best, start_idx = _stable_smallest(
                torch.cat((start_best, start_distance), -1), torch.cat((start_idx, source_idx), -1), keep)
            end_best, end_idx = _stable_smallest(
                torch.cat((end_best, end_distance), -1), torch.cat((end_idx, source_idx), -1), keep)

        candidates = torch.cat((start_idx, end_idx), -1)
        candidate_count = candidates.shape[-1]
        expanded = candidates[..., None].expand(-1, -1, -1, 3)
        start_points = torch.gather(
            hand_points[:, None].expand(-1, width, -1, -1), 2, expanded)
        end_points = torch.gather(endpoint[:, None].expand(-1, width, -1, -1), 2, expanded)
        start_distance = torch.linalg.vector_norm(start_points - query[:, :, None], dim=-1)
        end_distance = torch.linalg.vector_norm(end_points - query[:, :, None], dim=-1)
        minimum = torch.minimum(start_distance, end_distance)
        candidate_valid = torch.gather(
            hand_valid_mask[:, None].expand(-1, width, -1), 2, candidates)
        minimum = minimum.masked_fill(~candidate_valid, float("inf"))

        earlier = torch.tril(
            torch.ones((candidate_count, candidate_count), dtype=torch.bool, device=query.device), diagonal=-1)
        duplicate = ((candidates[..., :, None] == candidates[..., None, :]) & earlier).any(-1)
        minimum = minimum.masked_fill(duplicate, float("inf"))
        selected_min, selected_idx = _stable_smallest(minimum, candidates, keep)
        selection = (selected_idx[..., None] == candidates[..., None, :])
        first_match = selection.to(torch.int64).argmax(-1)
        selected_start = torch.gather(start_distance, -1, first_match)
        selected_end = torch.gather(end_distance, -1, first_match)
        output_min.append(selected_min)
        output_idx.append(selected_idx)
        output_start.append(selected_start)
        output_end.append(selected_end)
    return tuple(torch.cat(values, 1) for values in (output_min, output_idx, output_start, output_end))


class EndpointInteractionEncoder(nn.Module):
    """Object-query attention over the final V1.12 endpoint top-32 edges."""

    def __init__(self, width: int, k: int, radius_m: float, feature_scale_m: float,
                 frame_dt_s: float) -> None:
        super().__init__()
        self.width, self.k, self.radius_m = int(width), int(k), float(radius_m)
        self.feature_scale_m, self.frame_dt_s = float(feature_scale_m), float(frame_dt_s)
        self.edge = _mlp(24, self.width, self.width)
        self.query = nn.Linear(self.width, self.width, bias=False)
        self.key = nn.Linear(self.width, self.width, bias=False)
        self.value = nn.Linear(self.width, self.width, bias=False)

    def forward(self, object_features: Tensor, object_points: Tensor, object_normals: Tensor,
                hand_points: Tensor, hand_normals: Tensor, hand_flow: Tensor,
                hand_valid_mask: Tensor | None = None, delta_time_s: Tensor | None = None,
                edge_object_chunk: int = 128, edge_hand_chunk: int = 256) -> tuple[Tensor, dict[str, Tensor]]:
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones(hand_points.shape[:2], dtype=torch.bool, device=hand_points.device)
        if delta_time_s is None:
            delta_time_s = torch.full((object_points.shape[0],), self.frame_dt_s,
                                      dtype=object_points.dtype, device=object_points.device)
        delta_time_s = delta_time_s.reshape(object_points.shape[0])
        distance, indices, start_distance, end_distance = endpoint_union_topk(
            object_points, hand_points, hand_flow, hand_valid_mask, k=self.k,
            object_chunk=int(edge_object_chunk), hand_chunk=int(edge_hand_chunk))
        object_count = object_points.shape[1]
        contacts, masks = [], []
        query = self.query(object_features)
        for start in range(0, object_count, int(edge_object_chunk)):
            stop = min(start + int(edge_object_chunk), object_count)
            chunk_indices = indices[:, start:stop]
            gather = chunk_indices[..., None].expand(-1, -1, -1, 3)
            width = stop - start
            hp = torch.gather(hand_points[:, None].expand(-1, width, -1, -1), 2, gather)
            hn = torch.gather(hand_normals[:, None].expand(-1, width, -1, -1), 2, gather)
            hf = torch.gather(hand_flow[:, None].expand(-1, width, -1, -1), 2, gather)
            valid = torch.gather(hand_valid_mask[:, None].expand(-1, width, -1), 2, chunk_indices)
            valid = valid & distance[:, start:stop].lt(self.radius_m)
            relative = hp - object_points[:, start:stop, None]
            on = object_normals[:, start:stop, None].expand_as(hn)
            dot = (on * hn).sum(-1, keepdim=True)
            normal_flow = (hf * on).sum(-1, keepdim=True)
            tangent_flow = hf - normal_flow * on
            dt = delta_time_s[:, None, None, None].expand_as(dot)
            scale = self.feature_scale_m
            velocity = (hf / dt.clamp_min(1e-8)) * (self.frame_dt_s / scale)
            edge_input = torch.cat((
                relative / scale,
                start_distance[:, start:stop, :, None] / scale,
                end_distance[:, start:stop, :, None] / scale,
                distance[:, start:stop, :, None] / scale,
                on, hn, dot, hf / scale, normal_flow / scale, tangent_flow / scale,
                dt / self.frame_dt_s, velocity,
            ), -1)
            edge = self.edge(edge_input)
            logits = (query[:, start:stop, None] * self.key(edge)).sum(-1) / (self.width ** 0.5)
            weights = torch.softmax(logits.masked_fill(~valid, -torch.finfo(logits.dtype).max), -1)
            weights = weights * valid.to(weights.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            contacts.append((weights[..., None] * self.value(edge)).sum(2))
            masks.append(valid)
        valid = torch.cat(masks, 1)
        contact = torch.cat(contacts, 1) * valid.any(-1, keepdim=True).to(object_features.dtype)
        return contact, {"edge_indices": indices, "edge_distances": distance,
                         "edge_start_distances": start_distance, "edge_end_distances": end_distance,
                         "edge_valid_mask": valid, "has_interaction": valid.any(-1)}


class RoutedContactTokens(nn.Module):
    """V1.3 competitive tokens with point-to-token assignment exposed."""

    def __init__(self, width: int, num_tokens: int) -> None:
        super().__init__()
        self.assignment = _mlp(width + 6, width, num_tokens)

    def forward(self, contact: Tensor, points: Tensor, normals: Tensor, active: Tensor,
                object_scale: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        spatial = torch.cat((contact, points / object_scale[:, None, None], normals), -1)
        assignment = torch.softmax(self.assignment(spatial), -1) * active[..., None].to(contact.dtype)
        mass = assignment.sum(1)
        normalized = assignment / mass[:, None].clamp_min(1e-8)
        tokens = torch.bmm(normalized.transpose(1, 2), contact)
        anchors = torch.bmm(normalized.transpose(1, 2), points)
        token_normals = F.normalize(torch.bmm(normalized.transpose(1, 2), normals), dim=-1, eps=1e-8)
        mask = mass.gt(1e-8)
        return tokens, anchors, token_normals, mass, mask, assignment


def _segment_mean(values: Tensor, part_ids: Tensor, part_valid: Tensor,
                  include: Tensor | None = None) -> Tensor:
    output = values.new_zeros((values.shape[0], part_valid.shape[1], values.shape[-1]))
    for part in range(part_valid.shape[1]):
        mask = (part_ids == part) & part_valid[:, part, None]
        if include is not None:
            mask = mask & include
        denominator = mask.sum(1, keepdim=True).clamp_min(1).to(values.dtype)
        output[:, part] = (values * mask[..., None].to(values.dtype)).sum(1) / denominator
    return output


def transform_points_by_part(points: Tensor, delta_xi_part: Tensor, part_ids: Tensor) -> Tensor:
    if points.ndim != 3 or delta_xi_part.ndim != 3 or part_ids.shape != points.shape[:2]:
        raise ValueError("invalid per-part transform shapes")
    if (part_ids < 0).any() or (part_ids >= delta_xi_part.shape[1]).any():
        raise ValueError("obj_part_id outside part range")
    gather = part_ids[..., None].expand(-1, -1, 6)
    point_xi = torch.gather(delta_xi_part, 1, gather)
    rotation = _axis_angle_matrix_stable(point_xi[..., 3:].reshape(-1, 3)).reshape(*points.shape[:2], 3, 3)
    return torch.matmul(points.unsqueeze(-2), rotation.transpose(-1, -2)).squeeze(-2) + point_xi[..., :3]


class PartSE3ObjectInteractionCmv2V112Model(nn.Module):
    """Grounded surface evidence routed to permutation-equivariant part queries."""

    architecture_version = PART_SE3_VERSION

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        width, tokens = int(cfg.hidden_width), int(cfg.num_tokens)
        if int(cfg.knn_k) != 32 or tokens != 16:
            raise ValueError("V1.12 fixes endpoint K=32 and surface K=16")
        if width % 4:
            raise ValueError("V1.12 hidden width must be divisible by four attention heads")
        self.width, self.num_heads = width, 4
        self.feature_scale_m = float(cfg.feature_scale_m)
        self.geometry_encoder = _mlp(7, width, width)
        self.local_interaction = EndpointInteractionEncoder(
            width, 32, float(cfg.interaction_radius_m), self.feature_scale_m, float(cfg.frame_dt_s))
        self.tokens = RoutedContactTokens(width, tokens)
        self.token_encoder = _mlp(width + 7, width, width)
        self.part_encoder = _mlp(2 * width + 1, width, width)
        self.part_surface_attention = nn.MultiheadAttention(width, self.num_heads, batch_first=True)
        self.part_fusion = _mlp(2 * width, width, width)
        self.part_self_attention = nn.MultiheadAttention(width, self.num_heads, batch_first=True)
        self.part_norm = nn.LayerNorm(width)
        self.part_motion_head = nn.Linear(width, 6)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        required = ("obj_points", "obj_normals", "obj_part_id", "part_valid_mask",
                    "hand_points", "hand_normals", "hand_flow", "hand_valid_mask")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"missing V1.12 fields: {missing}")
        points = batch["obj_points"]
        normals = F.normalize(batch["obj_normals"], dim=-1, eps=1e-8)
        part_ids, part_valid = batch["obj_part_id"].long(), batch["part_valid_mask"].bool()
        if (part_ids < 0).any() or (part_ids >= part_valid.shape[1]).any():
            raise ValueError("obj_part_id outside padded part range")
        if not part_valid.any(-1).all():
            raise ValueError("each sample needs at least one valid part")
        point_part_valid = torch.gather(part_valid, 1, part_ids)
        if not point_part_valid.all():
            raise ValueError("obj_part_id refers to a padded part")
        centered = points - points.mean(1, keepdim=True)
        object_scale = centered.square().sum(-1).mean(1).sqrt()
        if not torch.isfinite(object_scale).all() or (object_scale <= 1e-8).any():
            raise ValueError("degenerate object geometry")
        geo = self.geometry_encoder(torch.cat((
            centered / object_scale[:, None, None], normals,
            torch.log(object_scale / self.feature_scale_m)[:, None, None].expand(-1, points.shape[1], 1)), -1))
        contact, diagnostics = self.local_interaction(
            geo, points, normals, batch["hand_points"], batch["hand_normals"], batch["hand_flow"],
            batch["hand_valid_mask"], batch.get("delta_time_s"),
            int(batch.get("interaction_object_chunk", 128)), int(batch.get("interaction_hand_chunk", 256)))
        active = diagnostics["has_interaction"]
        tokens, anchors, token_normals, token_mass, token_mask, assignment = self.tokens(
            contact, points, normals, active, object_scale)
        spatial_tokens = self.token_encoder(torch.cat((
            tokens, anchors / object_scale[:, None, None], token_normals,
            torch.log(token_mass.clamp_min(1e-8))[..., None]), -1))

        geometry = _segment_mean(geo, part_ids, part_valid)
        interaction = _segment_mean(contact, part_ids, part_valid, active)
        active_fraction = _segment_mean(active[..., None].to(points.dtype), part_ids, part_valid)[..., :1]
        parts = self.part_encoder(torch.cat((geometry, interaction, active_fraction), -1))
        parts = parts * part_valid[..., None].to(parts.dtype)

        part_mass = assignment.new_zeros((points.shape[0], part_valid.shape[1], token_mass.shape[1]))
        for part in range(part_valid.shape[1]):
            selector = (part_ids == part) & part_valid[:, part, None]
            part_mass[:, part] = (assignment * selector[..., None].to(assignment.dtype)).sum(1)
        rho = part_mass / token_mass[:, None].clamp_min(1e-8)
        bias = torch.log(rho + 1e-8)
        safe_token_mask = token_mask.clone()
        empty_tokens = ~safe_token_mask.any(1)
        safe_token_mask[empty_tokens, 0] = True
        bias = bias.masked_fill(~safe_token_mask[:, None], -torch.finfo(bias.dtype).max)
        bias[empty_tokens, :, 0] = 0
        attention_mask = bias[:, None].expand(-1, self.num_heads, -1, -1).reshape(
            points.shape[0] * self.num_heads, part_valid.shape[1], token_mass.shape[1])
        routed, _ = self.part_surface_attention(parts, spatial_tokens, spatial_tokens,
                                                 attn_mask=attention_mask, need_weights=False)
        part_has_interaction = part_mass.sum(-1).gt(1e-8) & part_valid
        routed = routed * part_has_interaction[..., None].to(routed.dtype)
        parts = self.part_fusion(torch.cat((parts, routed), -1)) * part_valid[..., None].to(parts.dtype)
        # Use an equivalent distinct K/V view so PyTorch 2.0 does not route the
        # padded attention through its inference fast path, which coerces the
        # padding mask and emits a warning.
        part_context = parts.clone()
        propagated, _ = self.part_self_attention(parts, part_context, part_context,
                                                  key_padding_mask=~part_valid, need_weights=False)
        parts = self.part_norm(parts + propagated) * part_valid[..., None].to(parts.dtype)
        delta = self.part_motion_head(parts) * part_valid[..., None].to(parts.dtype)
        future = transform_points_by_part(points, delta, part_ids)
        return {"delta_xi_part": delta, "obj_flow_pred": future - points,
                "part_features": parts, "part_surface_features": routed,
                "part_token_rho": rho, "part_has_interaction": part_has_interaction,
                "tokens": tokens, "token_anchors": anchors, "token_normals": token_normals,
                "token_mass": token_mass, "token_mask": token_mask, "token_assignment": assignment,
                "contact_features": contact, "contact_active": active, **diagnostics}


def _sample_part_mean(values: Tensor, valid: Tensor) -> Tensor:
    return ((values * valid.to(values.dtype)).sum(1) /
            valid.sum(1).clamp_min(1).to(values.dtype)).mean()


def part_se3_v112_loss(output: Mapping[str, Tensor], batch: Mapping[str, Tensor],
                       scale_m: float = 0.02) -> dict[str, Tensor]:
    prediction = output["delta_xi_part"]
    valid = batch["part_valid_mask"].bool()
    translation_per_part = F.smooth_l1_loss(
        prediction[..., :3] / scale_m, batch["delta_translation_part_gt"] / scale_m,
        reduction="none").mean(-1)
    translation = _sample_part_mean(translation_per_part, valid)
    predicted_rotation = _axis_angle_matrix_stable(prediction[..., 3:].reshape(-1, 3)).reshape(
        *prediction.shape[:2], 3, 3)
    relative = batch["delta_rotation_part_gt"].transpose(-1, -2) @ predicted_rotation
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
    skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                        relative[..., 0, 2] - relative[..., 2, 0],
                        relative[..., 1, 0] - relative[..., 0, 1]), -1)
    angle = torch.atan2(torch.linalg.vector_norm(skew, dim=-1), trace - 1)
    rotation = _sample_part_mean(angle, valid)
    flow = F.smooth_l1_loss(output["obj_flow_pred"] / scale_m, batch["obj_flow_gt"] / scale_m)
    return {"total": translation + rotation + flow, "translation": translation,
            "rotation": rotation, "flow": flow}


def collate_part_se3(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pad only hand and part axes; object samples keep the fixed 1024 points."""
    if not batch:
        raise ValueError("cannot collate an empty V1.12 batch")
    result: dict[str, Any] = {}
    hand_keys = {"hand_points", "hand_normals", "hand_flow", "hand_valid_mask"}
    part_keys = {"part_valid_mask", "delta_translation_part_gt", "delta_rotation_part_gt"}
    max_hand = max(int(sample["hand_points"].shape[0]) for sample in batch)
    max_part = max(int(sample["part_valid_mask"].shape[0]) for sample in batch)
    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if key in hand_keys | part_keys:
            width = max_hand if key in hand_keys else max_part
            tail = values[0].shape[1:]
            padded = torch.zeros((len(batch), width, *tail), dtype=values[0].dtype)
            if key == "delta_rotation_part_gt":
                padded[:] = torch.eye(3, dtype=values[0].dtype)
            for row, value in enumerate(values):
                padded[row, :value.shape[0]] = value
            result[key] = padded
        elif torch.is_tensor(values[0]):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result
