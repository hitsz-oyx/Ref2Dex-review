from __future__ import annotations

import torch
import torch.nn as nn


class SpatialQueryPool(nn.Module):
    """Learned cross-attention queries that preserve more than global max pooling."""

    def __init__(self, channels: int, num_queries: int, heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, channels))
        self.query_norm = nn.LayerNorm(channels)
        self.feature_norm = nn.LayerNorm(channels)
        self.cross_attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        nn.init.normal_(self.queries, std=0.02)

    def forward(
        self, features: torch.Tensor, conditioning: torch.Tensor | None = None
    ) -> torch.Tensor:
        queries = self.queries.expand(len(features), -1, -1)
        if conditioning is not None:
            queries = queries + conditioning[:, None]
        update, _ = self.cross_attention(
            self.query_norm(queries),
            self.feature_norm(features),
            self.feature_norm(features),
            need_weights=False,
        )
        queries = queries + update
        return queries + self.mlp(self.output_norm(queries))
