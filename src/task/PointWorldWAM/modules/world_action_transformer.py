from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class WorldActionTransformer(nn.Module):
    """Bidirectional temporal mixing over object/left/right compact queries."""

    def __init__(
        self,
        spatial_dim: int,
        temporal_dim: int,
        depth: int,
        heads: int,
        chunk_size: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(spatial_dim, temporal_dim)
        self.time_embedding = nn.Embedding(chunk_size, temporal_dim)
        self.type_embedding = nn.Parameter(torch.empty(3, 1, temporal_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=temporal_dim,
            nhead=heads,
            dim_feedforward=4 * temporal_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, norm=nn.LayerNorm(temporal_dim)
        )
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(
        self,
        object_tokens: torch.Tensor,
        left_tokens: torch.Tensor,
        right_tokens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        groups = (object_tokens, left_tokens, right_tokens)
        batch_size, chunk_size = object_tokens.shape[:2]
        time = self.time_embedding(
            torch.arange(chunk_size, device=object_tokens.device)
        )[None, :, None]
        projected = []
        lengths = []
        for group_index, tokens in enumerate(groups):
            value = self.input_projection(tokens)
            value = value + time + self.type_embedding[group_index]
            projected.append(value)
            lengths.append(value.shape[2])
        per_time = torch.cat(projected, dim=2)
        output = self.transformer(per_time.flatten(1, 2)).reshape(
            batch_size, chunk_size, sum(lengths), -1
        )
        object_output, left_output, right_output = torch.split(
            output, lengths, dim=2
        )
        return {
            "object": object_output,
            "left": left_output,
            "right": right_output,
        }
