from __future__ import annotations

import torch
import torch.nn as nn


class DirectForward(nn.Module):
    """不经过 Cm bottleneck、直接使用 O/H/ΔH 的 object-flow baseline。"""

    def __init__(self, hidden: int = 128):
        super().__init__()
        # Global hand state makes the baseline see both current hand pose and action.
        self.hand_encoder = nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.object_decoder = nn.Sequential(
            nn.Linear(3 + 3 + hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )

    def forward(self, object_points, object_normals, hand_points, hand_normals, hand_flow):
        hand_features = torch.cat([hand_points, hand_normals, hand_flow], dim=-1)
        hand_state = self.hand_encoder(hand_features).mean(1, keepdim=True)
        hand_state = hand_state.expand(-1, object_points.shape[1], -1)
        return self.object_decoder(torch.cat([object_points, object_normals, hand_state], dim=-1))
