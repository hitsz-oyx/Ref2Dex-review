"""Additive flow decoders used by ObjectInteractionCm V1.2.1."""
from __future__ import annotations

import torch
from torch import nn


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class ObjectFlowDecoder(nn.Module):
    """Decode raw object geometry and Cm with additive per-slot contributions."""

    def __init__(self, *, cm_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        self.cm_dim = int(cm_dim)
        self.hidden_dim = int(hidden_dim)
        input_dim = 3 + 3 + self.cm_dim + 3 + 3
        self.backbone = _mlp(input_dim, self.hidden_dim, self.hidden_dim)
        self.flow = nn.Linear(self.hidden_dim, 3)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, object_points, object_normals, cm_tokens, anchor_positions, anchor_normals):
        bsz, num_points, _ = object_points.shape
        num_slots = cm_tokens.shape[1]
        relative = object_points[:, :, None, :] - anchor_positions[:, None, :, :]
        anchor_normal = anchor_normals[:, None, :, :].expand(-1, num_points, -1, -1)
        features = torch.cat([
            object_points[:, :, None, :].expand(-1, -1, num_slots, -1),
            object_normals[:, :, None, :].expand(-1, -1, num_slots, -1),
            cm_tokens[:, None, :, :].expand(-1, num_points, -1, -1),
            relative, anchor_normal,
        ], dim=-1)
        hidden = self.backbone(features)
        contribution = self.flow(hidden)
        prediction = contribution.sum(dim=2)
        usage = contribution.norm(dim=-1)
        usage = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return prediction, usage


class SharedHandFlowDecoder(nn.Module):
    """Shared hand-flow head with delayed compression and additive slots."""

    def __init__(self, *, cm_dim: int = 32, processing_dim: int = 128, hidden_dim: int = 128, dim: int | None = None) -> None:
        super().__init__()
        if dim is not None:
            cm_dim = int(dim)  # backwards-compatible direct constructor alias
        self.cm_dim = int(cm_dim)
        self.processing_dim = int(processing_dim)
        self.hand_encoder = _mlp(6, 64, self.processing_dim)
        input_dim = self.processing_dim + self.cm_dim + 3 + 3
        self.backbone = _mlp(input_dim, int(hidden_dim), int(hidden_dim))
        self.flow = nn.Linear(int(hidden_dim), 3)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, hand_points, hand_normals, cm_tokens, anchor_positions, anchor_normals):
        _, num_hand, _ = hand_points.shape
        num_slots = cm_tokens.shape[1]
        hand_feat = self.hand_encoder(torch.cat([hand_points, hand_normals], dim=-1))
        relative = hand_points[:, :, None, :] - anchor_positions[:, None, :, :]
        anchor_normal = anchor_normals[:, None, :, :].expand(-1, num_hand, -1, -1)
        features = torch.cat([
            hand_feat[:, :, None, :].expand(-1, -1, num_slots, -1),
            cm_tokens[:, None, :, :].expand(-1, num_hand, -1, -1),
            relative, anchor_normal,
        ], dim=-1)
        hidden = self.backbone(features)
        contribution = self.flow(hidden)
        prediction = contribution.sum(dim=2)
        usage = contribution.norm(dim=-1)
        usage = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return prediction, usage


class LegacyObjectFlowDecoder(nn.Module):
    """V1.1 decoder kept solely for loading old ObjectInteractionCm checkpoints."""

    def __init__(self, dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = dim + dim + 3 + 3
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.logit = nn.Linear(hidden_dim, 1)
        self.flow = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, object_tokens, slots, object_points, anchor_positions, anchor_normals):
        _, num_points, _ = object_tokens.shape
        num_slots = slots.shape[1]
        relative = object_points[:, :, None, :] - anchor_positions[:, None, :, :]
        normals = anchor_normals[:, None, :, :].expand(-1, num_points, -1, -1)
        obj = object_tokens[:, :, None, :].expand(-1, -1, num_slots, -1)
        cm = slots[:, None, :, :].expand(-1, num_points, -1, -1)
        hidden = self.backbone(torch.cat([obj, cm, relative, normals], dim=-1))
        weights = torch.softmax(self.logit(hidden).squeeze(-1), dim=2)
        candidate = self.flow(hidden)
        return (weights.unsqueeze(-1) * candidate).sum(dim=2), weights, candidate


class LegacySharedHandFlowDecoder(nn.Module):
    """V1.1 hand decoder kept solely for old checkpoint compatibility."""

    def __init__(self, dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = 6 + dim + 3 + 3
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.logit = nn.Linear(hidden_dim, 1)
        self.flow = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(self, hand_points, hand_normals, slots, anchor_positions, anchor_normals):
        _, num_hand, _ = hand_points.shape
        num_slots = slots.shape[1]
        raw = torch.cat([hand_points, hand_normals], dim=-1)[:, :, None, :].expand(-1, -1, num_slots, -1)
        cm = slots[:, None, :, :].expand(-1, num_hand, -1, -1)
        relative = hand_points[:, :, None, :] - anchor_positions[:, None, :, :]
        normals = anchor_normals[:, None, :, :].expand(-1, num_hand, -1, -1)
        hidden = self.backbone(torch.cat([raw, cm, relative, normals], dim=-1))
        weights = torch.softmax(self.logit(hidden).squeeze(-1), dim=2)
        candidate = self.flow(hidden)
        return (weights.unsqueeze(-1) * candidate).sum(dim=2), weights
