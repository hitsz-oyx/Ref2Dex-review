"""Flow decoders used by ObjectInteractionCm."""
from __future__ import annotations

import torch
from torch import nn


class ObjectFlowDecoder(nn.Module):
    """Decode object-point flow from object tokens and object-side Cm anchors."""

    def __init__(self, dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = dim + dim + 3 + 3
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.logit = nn.Linear(hidden_dim, 1)
        self.flow = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(
        self,
        object_tokens: torch.Tensor,
        slots: torch.Tensor,
        object_points: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_points, dim = object_tokens.shape
        num_slots = slots.shape[1]
        relative = object_points[:, :, None, :] - anchor_positions[:, None, :, :]
        normals = anchor_normals[:, None, :, :].expand(-1, num_points, -1, -1)
        obj = object_tokens[:, :, None, :].expand(-1, -1, num_slots, -1)
        cm = slots[:, None, :, :].expand(-1, num_points, -1, -1)
        features = torch.cat([obj, cm, relative, normals], dim=-1)
        hidden = self.backbone(features)
        logits = self.logit(hidden).squeeze(-1)
        weights = torch.softmax(logits, dim=2)
        candidate_flow = self.flow(hidden)
        prediction = (weights.unsqueeze(-1) * candidate_flow).sum(dim=2)
        return prediction, weights, candidate_flow


class SharedHandFlowDecoder(nn.Module):
    """One shared hand-flow head for the concatenated left/right hand points."""

    def __init__(self, dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        input_dim = 6 + dim + 3 + 3
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.logit = nn.Linear(hidden_dim, 1)
        self.flow = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.flow.weight)
        nn.init.zeros_(self.flow.bias)

    def forward(
        self,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        slots: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_normals: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_hand = hand_points.shape[1]
        num_slots = slots.shape[1]
        raw = torch.cat([hand_points, hand_normals], dim=-1)
        raw = raw[:, :, None, :].expand(-1, -1, num_slots, -1)
        cm = slots[:, None, :, :].expand(-1, num_hand, -1, -1)
        relative = hand_points[:, :, None, :] - anchor_positions[:, None, :, :]
        normals = anchor_normals[:, None, :, :].expand(-1, num_hand, -1, -1)
        hidden = self.backbone(torch.cat([raw, cm, relative, normals], dim=-1))
        weights = torch.softmax(self.logit(hidden).squeeze(-1), dim=2)
        contribution = self.flow(hidden)
        return (weights.unsqueeze(-1) * contribution).sum(dim=2), weights
