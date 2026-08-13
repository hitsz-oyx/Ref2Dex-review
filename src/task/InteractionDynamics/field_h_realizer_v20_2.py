"""V20.2：只从预测交互场、物体几何和当前手态恢复未来 MANO-H。"""
from __future__ import annotations

import torch
from torch import nn


class FieldHRealizerV20_2(nn.Module):
    def __init__(self, horizon: int = 8, dim: int = 192, heads: int = 4,
                 temporal_layers: int = 2) -> None:
        super().__init__(); self.horizon = horizon
        self.y_encoder = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor_encoder = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_encoder = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.h_encoder = nn.Sequential(nn.Linear(33, dim), nn.GELU(), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(horizon, dim) * .02)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, temporal_layers, nn.LayerNorm(dim))
        self.head = nn.Linear(dim, 30)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, predicted_y: torch.Tensor, anchors_cm: torch.Tensor,
                object_patches: torch.Tensor, current_h: torch.Tensor) -> torch.Tensor:
        """返回 normalized delta H `[B,8,30]`；不读取 p channel。"""
        batch, anchors, horizon, _ = predicted_y.shape
        if horizon != self.horizon:
            raise ValueError(f"Expected horizon {self.horizon}, got {horizon}")
        geometry = self.anchor_encoder(anchors_cm) + self.object_encoder(object_patches).amax(2)
        spatial = self.y_encoder(predicted_y[..., :7]) + geometry[:, :, None]
        spatial = spatial.permute(0, 2, 1, 3).reshape(batch * horizon, anchors, -1)
        query = self.queries[None] + self.h_encoder(current_h)[:, None]
        query = query.reshape(batch * horizon, 1, -1)
        frame = query + self.cross_attention(query, self.cross_norm(spatial),
                                             self.cross_norm(spatial), need_weights=False)[0]
        frame = frame.reshape(batch, horizon, -1)
        return self.head(self.temporal(frame))
