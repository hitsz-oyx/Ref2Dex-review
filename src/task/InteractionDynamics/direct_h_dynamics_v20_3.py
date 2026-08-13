"""V20.3：由当前 causal field、物体几何与当前 H 直接预测未来 MANO-H。"""
from __future__ import annotations

import torch
from torch import nn


class DirectHDynamicsV20_3(nn.Module):
    def __init__(self, horizon: int = 8, y_dim: int = 8, dim: int = 192,
                 heads: int = 4, temporal_layers: int = 2) -> None:
        super().__init__(); self.horizon = horizon
        self.y_encoder = nn.Sequential(nn.Linear(y_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor_encoder = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_encoder = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.h_encoder = nn.Sequential(nn.Linear(33, dim), nn.GELU(), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(horizon, dim) * .02)
        self.spatial_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, temporal_layers, nn.LayerNorm(dim))
        self.head = nn.Linear(dim, 30)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, current_y: torch.Tensor, anchors_cm: torch.Tensor,
                object_patches: torch.Tensor, current_h: torch.Tensor) -> torch.Tensor:
        spatial = (self.y_encoder(current_y) + self.anchor_encoder(anchors_cm)
                   + self.object_encoder(object_patches).amax(2))
        query = self.queries[None] + self.h_encoder(current_h)[:, None]
        normalized = self.spatial_norm(spatial)
        future = query + self.cross_attention(query, normalized, normalized, need_weights=False)[0]
        return self.head(self.temporal(future))
