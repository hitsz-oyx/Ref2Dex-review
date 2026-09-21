"""V1.14 narrow endpoint interaction with shared candidate-axis object encoding."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import _axis_angle_matrix_stable, _mlp
from .part_se3 import _segment_mean, endpoint_union_topk


PART_SE3_V114_VERSION = "v1_14_narrow_interaction_candidate_shared"


@dataclass(frozen=True)
class ObjectContextV114:
    """Candidate-independent state produced exactly once per object state."""

    points: Tensor
    normals: Tensor
    part_ids: Tensor
    part_valid: Tensor
    object_scale: Tensor
    point_geometry: Tensor
    part_geometry: Tensor


def _candidate_segment_mean(
    values: Tensor,
    part_ids: Tensor,
    part_valid: Tensor,
    include: Tensor | None = None,
) -> Tensor:
    """Segment mean for values shaped [B,K,N,D]."""
    batch, candidates, _, width = values.shape
    output = values.new_zeros((batch, candidates, part_valid.shape[1], width))
    for part in range(part_valid.shape[1]):
        mask = (part_ids[:, None] == part) & part_valid[:, None, part, None]
        if include is not None:
            mask = mask & include
        denominator = mask.sum(2, keepdim=True).clamp_min(1).to(values.dtype)
        output[:, :, part] = (values * mask[..., None].to(values.dtype)).sum(2) / denominator
    return output


def transform_points_by_part_candidates(
    points: Tensor,
    delta_xi_part: Tensor,
    part_ids: Tensor,
) -> Tensor:
    """Apply candidate-specific part transforms without re-encoding object geometry."""
    if points.ndim != 3 or delta_xi_part.ndim != 4 or part_ids.shape != points.shape[:2]:
        raise ValueError("invalid candidate per-part transform shapes")
    batch, candidates, part_count, _ = delta_xi_part.shape
    if batch != points.shape[0] or (part_ids < 0).any() or (part_ids >= part_count).any():
        raise ValueError("obj_part_id outside candidate part range")
    gather = part_ids[:, None, :, None].expand(-1, candidates, -1, 6)
    point_xi = torch.gather(delta_xi_part, 2, gather)
    rotation = _axis_angle_matrix_stable(point_xi[..., 3:].reshape(-1, 3)).reshape(
        batch, candidates, points.shape[1], 3, 3)
    return (torch.matmul(points[:, None, :, None], rotation.transpose(-1, -2)).squeeze(-2)
            + point_xi[..., :3])


class EndpointInteractionEncoderV114(nn.Module):
    """Low-width object-query attention over endpoint top-32 edges."""

    def __init__(self, geometry_width: int, interaction_width: int, k: int,
                 radius_m: float, feature_scale_m: float, frame_dt_s: float) -> None:
        super().__init__()
        self.interaction_width = int(interaction_width)
        self.k, self.radius_m = int(k), float(radius_m)
        self.feature_scale_m, self.frame_dt_s = float(feature_scale_m), float(frame_dt_s)
        self.edge = _mlp(24, self.interaction_width, self.interaction_width)
        self.query = nn.Linear(int(geometry_width), self.interaction_width, bias=False)
        self.key = nn.Linear(self.interaction_width, self.interaction_width, bias=False)
        self.value = nn.Linear(self.interaction_width, self.interaction_width, bias=False)

    def _edges(
        self,
        context: ObjectContextV114,
        candidate: Mapping[str, Tensor],
        object_chunk: int,
        hand_chunk: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        points = context.points
        batch, object_count, _ = points.shape
        compact_keys = ("edge_hand_points", "edge_hand_normals", "edge_hand_flow",
                        "edge_source_id", "edge_valid_mask")
        reference_keys = ("hand_points", "hand_normals", "hand_flow", "hand_valid_mask")
        compact = all(key in candidate for key in compact_keys)
        reference = all(key in candidate for key in reference_keys)
        if compact == reference:
            raise KeyError("provide exactly one candidate reference stream or compact endpoint edges")
        if compact:
            hp = candidate["edge_hand_points"]
            hn = candidate["edge_hand_normals"]
            hf = candidate["edge_hand_flow"]
            indices = candidate["edge_source_id"].long()
            input_valid = candidate["edge_valid_mask"].bool()
            if hp.ndim != 5:
                raise ValueError("compact candidate edges must be [B,K,N,E,3]")
            candidates = hp.shape[1]
            expected = (batch, candidates, object_count, self.k, 3)
            if hp.shape != expected or hn.shape != expected or hf.shape != expected:
                raise ValueError("compact candidate edge tensors have invalid shape")
            if indices.shape != expected[:-1] or input_valid.shape != expected[:-1]:
                raise ValueError("compact candidate IDs or masks have invalid shape")
            relative = hp - points[:, None, :, None]
            start_distance = torch.linalg.vector_norm(relative, dim=-1)
            end_distance = torch.linalg.vector_norm(relative + hf, dim=-1)
            distance = torch.minimum(start_distance, end_distance)
            return hp, hn, hf, indices, input_valid, distance, start_distance, end_distance

        hand_points = candidate["hand_points"]
        hand_normals = candidate["hand_normals"]
        hand_flow = candidate["hand_flow"]
        hand_valid = candidate["hand_valid_mask"].bool()
        if hand_points.ndim != 4 or hand_normals.shape != hand_points.shape or hand_flow.shape != hand_points.shape:
            raise ValueError("reference candidate hand tensors must be [B,K,H,3]")
        candidates, hand_count = hand_points.shape[1:3]
        if hand_valid.shape != (batch, candidates, hand_count):
            raise ValueError("candidate hand_valid_mask must be [B,K,H]")
        flat_points = points[:, None].expand(-1, candidates, -1, -1).reshape(
            batch * candidates, object_count, 3)
        flat_hand = hand_points.reshape(batch * candidates, hand_count, 3)
        flat_flow = hand_flow.reshape_as(flat_hand)
        distance, indices, start_distance, end_distance = endpoint_union_topk(
            flat_points, flat_hand, flat_flow, hand_valid.reshape(batch * candidates, hand_count),
            k=self.k, object_chunk=object_chunk, hand_chunk=hand_chunk)
        edge_shape = (batch, candidates, object_count, self.k)
        distance = distance.reshape(edge_shape)
        indices = indices.reshape(edge_shape)
        start_distance = start_distance.reshape(edge_shape)
        end_distance = end_distance.reshape(edge_shape)
        gather = indices[..., None].expand(-1, -1, -1, -1, 3)
        hp = torch.gather(
            hand_points[:, :, None].expand(-1, -1, object_count, -1, -1), 3, gather)
        hn = torch.gather(
            hand_normals[:, :, None].expand(-1, -1, object_count, -1, -1), 3, gather)
        hf = torch.gather(
            hand_flow[:, :, None].expand(-1, -1, object_count, -1, -1), 3, gather)
        input_valid = torch.gather(
            hand_valid[:, :, None].expand(-1, -1, object_count, -1), 3, indices)
        return hp, hn, hf, indices, input_valid, distance, start_distance, end_distance

    def forward_candidates(
        self,
        context: ObjectContextV114,
        candidate: Mapping[str, Tensor],
        object_chunk: int = 128,
        hand_chunk: int = 256,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if object_chunk <= 0 or hand_chunk <= 0:
            raise ValueError("interaction chunks must be positive")
        hp, hn, hf, indices, input_valid, distance, start_distance, end_distance = self._edges(
            context, candidate, int(object_chunk), int(hand_chunk))
        batch, candidates, object_count = distance.shape[:3]
        delta_time = candidate.get("delta_time_s")
        if delta_time is None:
            delta_time = context.points.new_full((batch, candidates), self.frame_dt_s)
        if delta_time.shape != (batch, candidates):
            raise ValueError("candidate delta_time_s must be [B,K]")

        projected_query = self.query(context.point_geometry)
        contacts, masks = [], []
        for start in range(0, object_count, int(object_chunk)):
            stop = min(start + int(object_chunk), object_count)
            valid = input_valid[:, :, start:stop] & distance[:, :, start:stop].lt(self.radius_m)
            object_points = context.points[:, None, start:stop, None]
            relative = hp[:, :, start:stop] - object_points
            object_normals = context.normals[:, None, start:stop, None].expand_as(hn[:, :, start:stop])
            hand_normals = hn[:, :, start:stop]
            hand_flow = hf[:, :, start:stop]
            dot = (object_normals * hand_normals).sum(-1, keepdim=True)
            normal_flow = (hand_flow * object_normals).sum(-1, keepdim=True)
            tangent_flow = hand_flow - normal_flow * object_normals
            dt = delta_time[:, :, None, None, None].expand_as(dot)
            scale = self.feature_scale_m
            velocity = (hand_flow / dt.clamp_min(1e-8)) * (self.frame_dt_s / scale)
            edge_input = torch.cat((
                relative / scale,
                start_distance[:, :, start:stop, :, None] / scale,
                end_distance[:, :, start:stop, :, None] / scale,
                distance[:, :, start:stop, :, None] / scale,
                object_normals, hand_normals, dot, hand_flow / scale,
                normal_flow / scale, tangent_flow / scale,
                dt / self.frame_dt_s, velocity,
            ), -1)
            edge = self.edge(edge_input)
            query = projected_query[:, None, start:stop, None]
            logits = (query * self.key(edge)).sum(-1) / (self.interaction_width ** 0.5)
            weights = torch.softmax(
                logits.masked_fill(~valid, -torch.finfo(logits.dtype).max), -1)
            weights = weights * valid.to(weights.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
            contacts.append((weights[..., None] * self.value(edge)).sum(3))
            masks.append(valid)
        valid = torch.cat(masks, 2)
        contact = torch.cat(contacts, 2) * valid.any(-1, keepdim=True).to(context.points.dtype)
        return contact, {
            "edge_indices": indices,
            "edge_distances": distance,
            "edge_start_distances": start_distance,
            "edge_end_distances": end_distance,
            "edge_valid_mask": valid,
            "has_interaction": valid.any(-1),
        }


class RoutedContactTokensV114(nn.Module):
    def __init__(self, interaction_width: int, num_tokens: int) -> None:
        super().__init__()
        self.assignment = _mlp(interaction_width + 6, interaction_width, num_tokens)

    def forward(self, contact: Tensor, context: ObjectContextV114, active: Tensor):
        points = context.points[:, None].expand(-1, contact.shape[1], -1, -1)
        normals = context.normals[:, None].expand_as(points)
        scale = context.object_scale[:, None, None, None]
        spatial = torch.cat((contact, points / scale, normals), -1)
        assignment = torch.softmax(self.assignment(spatial), -1) * active[..., None].to(contact.dtype)
        mass = assignment.sum(2)
        normalized = assignment / mass[:, :, None].clamp_min(1e-8)
        tokens = torch.einsum("bknt,bknd->bktd", normalized, contact)
        anchors = torch.einsum("bknt,bnd->bktd", normalized, context.points)
        token_normals = F.normalize(
            torch.einsum("bknt,bnd->bktd", normalized, context.normals), dim=-1, eps=1e-8)
        return tokens, anchors, token_normals, mass, mass.gt(1e-8), assignment


class PartSE3ObjectInteractionCmv2V114Model(nn.Module):
    """V1.14 model with one static object encode and K dynamic candidates."""

    architecture_version = PART_SE3_V114_VERSION

    def __init__(self, cfg: SimpleNamespace) -> None:
        super().__init__()
        width = int(cfg.hidden_width)
        interaction_width = int(cfg.interaction_dim)
        tokens = int(cfg.num_tokens)
        if int(cfg.knn_k) != 32 or tokens != 16:
            raise ValueError("V1.14 fixes endpoint K=32 and surface K=16")
        if width % 4 or interaction_width <= 0:
            raise ValueError("invalid V1.14 hidden or interaction width")
        self.width, self.interaction_width, self.num_heads = width, interaction_width, 4
        self.feature_scale_m = float(cfg.feature_scale_m)
        self.geometry_encoder = _mlp(7, width, width)
        self.local_interaction = EndpointInteractionEncoderV114(
            width, interaction_width, 32, float(cfg.interaction_radius_m),
            self.feature_scale_m, float(cfg.frame_dt_s))
        self.tokens = RoutedContactTokensV114(interaction_width, tokens)
        self.token_encoder = _mlp(interaction_width + 7, width, width)
        self.part_encoder = _mlp(width + interaction_width + 1, width, width)
        self.part_surface_attention = nn.MultiheadAttention(width, self.num_heads, batch_first=True)
        self.part_fusion = _mlp(2 * width, width, width)
        self.part_self_attention = nn.MultiheadAttention(width, self.num_heads, batch_first=True)
        self.part_norm = nn.LayerNorm(width)
        self.part_motion_head = nn.Linear(width, 6)

    def encode_object(self, object_batch: Mapping[str, Tensor]) -> ObjectContextV114:
        required = ("obj_points", "obj_normals", "obj_part_id", "part_valid_mask")
        missing = [key for key in required if key not in object_batch]
        if missing:
            raise KeyError(f"missing V1.14 object fields: {missing}")
        points = object_batch["obj_points"]
        normals = F.normalize(object_batch["obj_normals"], dim=-1, eps=1e-8)
        part_ids = object_batch["obj_part_id"].long()
        part_valid = object_batch["part_valid_mask"].bool()
        if points.ndim != 3 or normals.shape != points.shape or part_ids.shape != points.shape[:2]:
            raise ValueError("invalid V1.14 object tensor shapes")
        if (part_ids < 0).any() or (part_ids >= part_valid.shape[1]).any():
            raise ValueError("obj_part_id outside padded part range")
        if not part_valid.any(-1).all() or not torch.gather(part_valid, 1, part_ids).all():
            raise ValueError("obj_part_id refers to an invalid part")
        centered = points - points.mean(1, keepdim=True)
        object_scale = centered.square().sum(-1).mean(1).sqrt()
        if not torch.isfinite(object_scale).all() or (object_scale <= 1e-8).any():
            raise ValueError("degenerate object geometry")
        point_geometry = self.geometry_encoder(torch.cat((
            centered / object_scale[:, None, None], normals,
            torch.log(object_scale / self.feature_scale_m)[:, None, None].expand(
                -1, points.shape[1], 1)), -1))
        part_geometry = _segment_mean(point_geometry, part_ids, part_valid)
        return ObjectContextV114(
            points, normals, part_ids, part_valid, object_scale, point_geometry, part_geometry)

    def forward_candidates(
        self,
        context: ObjectContextV114,
        candidate_batch: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        object_chunk = int(candidate_batch.get("interaction_object_chunk", 128))
        hand_chunk = int(candidate_batch.get("interaction_hand_chunk", 256))
        contact, diagnostics = self.local_interaction.forward_candidates(
            context, candidate_batch, object_chunk, hand_chunk)
        active = diagnostics["has_interaction"]
        tokens, anchors, token_normals, token_mass, token_mask, assignment = self.tokens(
            contact, context, active)
        spatial_tokens = self.token_encoder(torch.cat((
            tokens,
            anchors / context.object_scale[:, None, None, None],
            token_normals,
            torch.log(token_mass.clamp_min(1e-8))[..., None],
        ), -1))

        interaction = _candidate_segment_mean(
            contact, context.part_ids, context.part_valid, active)
        active_fraction = _candidate_segment_mean(
            active[..., None].to(contact.dtype), context.part_ids, context.part_valid)[..., :1]
        geometry = context.part_geometry[:, None].expand(-1, contact.shape[1], -1, -1)
        parts = self.part_encoder(torch.cat((geometry, interaction, active_fraction), -1))
        parts = parts * context.part_valid[:, None, :, None].to(parts.dtype)

        batch, candidates, _, token_count = assignment.shape
        part_count = context.part_valid.shape[1]
        part_mass = assignment.new_zeros((batch, candidates, part_count, token_count))
        for part in range(part_count):
            selector = ((context.part_ids == part) & context.part_valid[:, part, None])[:, None]
            part_mass[:, :, part] = (
                assignment * selector[..., None].to(assignment.dtype)).sum(2)
        rho = part_mass / token_mass[:, :, None].clamp_min(1e-8)
        bias = torch.log(rho + 1e-8)
        safe_token_mask = token_mask.clone()
        empty_tokens = ~safe_token_mask.any(2)
        flat_safe_token_mask = safe_token_mask.reshape(batch * candidates, token_count)
        flat_empty_tokens = empty_tokens.reshape(batch * candidates)
        flat_safe_token_mask[flat_empty_tokens, 0] = True
        safe_token_mask = flat_safe_token_mask.reshape(batch, candidates, token_count)
        bias = bias.masked_fill(~safe_token_mask[:, :, None], -torch.finfo(bias.dtype).max)
        flat_bias = bias.reshape(batch * candidates, part_count, token_count)
        flat_bias[flat_empty_tokens, :, 0] = 0
        bias = flat_bias.reshape(batch, candidates, part_count, token_count)

        flat_count = batch * candidates
        attention_mask = bias[:, :, None].expand(
            -1, -1, self.num_heads, -1, -1).reshape(
                flat_count * self.num_heads, part_count, token_count)
        flat_parts = parts.reshape(flat_count, part_count, self.width)
        flat_tokens = spatial_tokens.reshape(flat_count, token_count, self.width)
        routed, _ = self.part_surface_attention(
            flat_parts, flat_tokens, flat_tokens, attn_mask=attention_mask, need_weights=False)
        routed = routed.reshape(batch, candidates, part_count, self.width)
        part_has_interaction = part_mass.sum(-1).gt(1e-8) & context.part_valid[:, None]
        routed = routed * part_has_interaction[..., None].to(routed.dtype)
        parts = self.part_fusion(torch.cat((parts, routed), -1))
        parts = parts * context.part_valid[:, None, :, None].to(parts.dtype)

        flat_parts = parts.reshape(flat_count, part_count, self.width)
        propagated, _ = self.part_self_attention(
            flat_parts, flat_parts.clone(), flat_parts.clone(),
            key_padding_mask=(~context.part_valid[:, None].expand(
                -1, candidates, -1)).reshape(flat_count, part_count), need_weights=False)
        parts = self.part_norm(flat_parts + propagated).reshape(
            batch, candidates, part_count, self.width)
        parts = parts * context.part_valid[:, None, :, None].to(parts.dtype)
        delta = self.part_motion_head(parts) * context.part_valid[:, None, :, None].to(parts.dtype)
        future = transform_points_by_part_candidates(context.points, delta, context.part_ids)
        return {
            "delta_xi_part": delta,
            "obj_flow_pred": future - context.points[:, None],
            "part_features": parts,
            "part_surface_features": routed,
            "part_token_rho": rho,
            "part_has_interaction": part_has_interaction,
            "tokens": tokens,
            "token_anchors": anchors,
            "token_normals": token_normals,
            "token_mass": token_mass,
            "token_mask": token_mask,
            "token_assignment": assignment,
            "contact_features": contact,
            "contact_active": active,
            **diagnostics,
        }

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        context = self.encode_object(batch)
        dynamic_keys = (
            "hand_points", "hand_normals", "hand_flow", "hand_valid_mask", "delta_time_s",
            "edge_hand_points", "edge_hand_normals", "edge_hand_flow", "edge_source_id",
            "edge_valid_mask",
        )
        candidate = {key: batch[key].unsqueeze(1) for key in dynamic_keys if key in batch}
        for key in ("interaction_object_chunk", "interaction_hand_chunk"):
            if key in batch:
                candidate[key] = batch[key]
        output = self.forward_candidates(context, candidate)
        return {key: value[:, 0] for key, value in output.items()}
