"""V1.0 structured interaction model.

The implementation intentionally keeps the external contract small: complete object
geometry is encoded for structure, while hand evidence is restricted to 2 cm local
edges.  The rigid V1.0 effect is an SE(3) transform and point flow is reconstructed
from that transform, with a small optional residual path.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


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


class LocalInteractionEncoder(nn.Module):
    """Object-query attention over valid hand neighbors within 2 cm."""

    def __init__(self, width: int = 128, knn_k: int = 32, radius_m: float = 0.02) -> None:
        super().__init__()
        self.width, self.knn_k, self.radius_m = int(width), int(knn_k), float(radius_m)
        self.edge = _mlp(18, width, width)
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)

    def forward(self, object_features: Tensor, object_points: Tensor, object_normals: Tensor,
                hand_points: Tensor, hand_normals: Tensor, hand_flow: Tensor,
                hand_valid_mask: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        if hand_valid_mask is None:
            hand_valid_mask = torch.ones(hand_points.shape[:2], dtype=torch.bool, device=hand_points.device)
        k = min(self.knn_k, hand_points.shape[1])
        distances = torch.cdist(object_points, hand_points)
        edge_distances, indices = torch.topk(distances, k=k, largest=False, dim=-1)
        gather = indices[..., None].expand(-1, -1, -1, 3)
        hp = torch.gather(hand_points[:, None].expand(-1, object_points.shape[1], -1, -1), 2, gather)
        hn = torch.gather(hand_normals[:, None].expand(-1, object_points.shape[1], -1, -1), 2, gather)
        hf = torch.gather(hand_flow[:, None].expand(-1, object_points.shape[1], -1, -1), 2, gather)
        valid = torch.gather(hand_valid_mask[:, None].expand(-1, object_points.shape[1], -1), 2, indices)
        valid = valid & edge_distances.lt(self.radius_m)
        relative = hp - object_points[:, :, None, :]
        on = object_normals[:, :, None, :].expand_as(hn)
        dot = (on * hn).sum(-1, keepdim=True)
        normal_flow = (hf * on).sum(-1, keepdim=True)
        tangent_flow = hf - normal_flow * on
        edge = self.edge(torch.cat([relative, edge_distances[..., None], on, hn, dot, hf, normal_flow, tangent_flow], -1))
        logits = (self.query(object_features)[:, :, None, :] * self.key(edge)).sum(-1) / (self.width ** 0.5)
        weights = torch.softmax(logits.masked_fill(~valid, -torch.finfo(logits.dtype).max), -1)
        weights = weights * valid.to(weights.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        contact = (weights[..., None] * self.value(edge)).sum(2)
        contact = contact * valid.any(-1, keepdim=True).to(contact.dtype)
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
        self.local_interaction = LocalInteractionEncoder(self.width, int(getattr(meta, "knn_k", 32)), float(getattr(meta, "interaction_radius_m", 0.02)))
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
                                                       batch["hand_normals"], batch["hand_flow"], batch.get("hand_valid_mask"))
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
