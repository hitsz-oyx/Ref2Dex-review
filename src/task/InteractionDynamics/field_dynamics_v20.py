"""V20 无 pooling 的 object-indexed field dynamics。"""
from __future__ import annotations

import torch
from torch import nn


class FieldBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__(); self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        value = value + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return value + self.ffn(self.norm2(value))


class FieldDynamicsTransition(nn.Module):
    def __init__(self, horizon: int = 8, y_dim: int = 8, dim: int = 128,
                 heads: int = 4, layers: int = 4) -> None:
        super().__init__(); self.horizon = horizon; self.y_dim = y_dim
        self.y_encoder = nn.Sequential(nn.Linear(y_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor_encoder = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_encoder = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([FieldBlock(dim, heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
                                    nn.Linear(dim, horizon * y_dim))

    def encode(self, current_y: torch.Tensor, anchors_cm: torch.Tensor,
               object_patches: torch.Tensor) -> torch.Tensor:
        tokens = (self.y_encoder(current_y) + self.anchor_encoder(anchors_cm)
                  + self.object_encoder(object_patches).amax(2))
        for block in self.blocks: tokens = block(tokens)
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.output(tokens).reshape(len(tokens), tokens.shape[1], self.horizon, self.y_dim)

    def forward(self, current_y: torch.Tensor, anchors_cm: torch.Tensor,
                object_patches: torch.Tensor, intervention: str = "normal") -> torch.Tensor:
        tokens = self.encode(current_y, anchors_cm, object_patches)
        if intervention == "mean": tokens = tokens.mean(1, keepdim=True).expand_as(tokens)
        elif intervention == "shuffle":
            tokens = tokens[:, torch.randperm(tokens.shape[1], device=tokens.device)]
        elif intervention != "normal": raise ValueError(f"Unknown intervention: {intervention}")
        return self.decode(tokens)
