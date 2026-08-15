from __future__ import annotations

import torch
import torch.nn as nn


class ActionReadout(nn.Module):
    """One learned query per timestep reads all world/left/right temporal tokens."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, channels))
        self.query_norm = nn.LayerNorm(channels)
        self.token_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, 4 * channels),
            nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, chunk_size, query_count, channels = tokens.shape
        key_value = tokens.reshape(batch_size * chunk_size, query_count, channels)
        query = self.query.expand(batch_size * chunk_size, -1, -1)
        update, _ = self.attention(
            self.query_norm(query),
            self.token_norm(key_value),
            self.token_norm(key_value),
            need_weights=False,
        )
        output = query + update
        output = output + self.mlp(output)
        return output[:, 0].reshape(batch_size, chunk_size, channels)
