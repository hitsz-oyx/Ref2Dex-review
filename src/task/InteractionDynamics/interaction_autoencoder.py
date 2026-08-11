"""V16.2 coordinate-conditioned Y-Teacher V1 slot autoencoder。"""
from __future__ import annotations

import torch
from torch import nn


class InteractionAutoencoder(nn.Module):
    def __init__(self, slots: int = 8, dim: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.geometry_input = nn.Sequential(nn.Linear(7, dim), nn.GELU(), nn.LayerNorm(dim))
        self.motion_input = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.LayerNorm(dim))
        self.geometry_query = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.LayerNorm(dim))
        self.motion_query = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.LayerNorm(dim))
        self.time_embedding = nn.Embedding(9, dim)
        self.type_embedding = nn.Embedding(2, dim)
        self.latent_queries = nn.Parameter(torch.randn(slots, dim) * .02)
        self.encoder_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(
            dim, heads, dim_feedforward=4 * dim, batch_first=True, norm_first=True)
        self.latent_transformer = nn.TransformerEncoder(layer, 1)
        self.decoder_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.geometry_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, 4))
        self.motion_head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, 3))

    @staticmethod
    def _expand_anchor_time(anchors_cm: torch.Tensor, frames: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count, _ = anchors_cm.shape
        coordinates = anchors_cm[:, None].expand(-1, frames, -1, -1)
        time = torch.arange(frames, device=anchors_cm.device)[None, :, None].expand(
            batch, -1, count)
        return coordinates, time

    def encode(self, anchors_cm: torch.Tensor, relative_geometry_cm: torch.Tensor,
               relative_distance_cm: torch.Tensor,
               relative_motion_cm: torch.Tensor) -> torch.Tensor:
        geometry_x, geometry_t = self._expand_anchor_time(anchors_cm, 9)
        motion_x, motion_t = self._expand_anchor_time(anchors_cm, 8)
        geometry = torch.cat(
            [geometry_x, relative_geometry_cm, relative_distance_cm[..., None]], -1)
        motion = torch.cat([motion_x, relative_motion_cm], -1)
        geometry_tokens = (self.geometry_input(geometry)
                           + self.time_embedding(geometry_t)
                           + self.type_embedding.weight[0])
        motion_tokens = (self.motion_input(motion)
                         + self.time_embedding(motion_t)
                         + self.type_embedding.weight[1])
        tokens = torch.cat([geometry_tokens.flatten(1, 2), motion_tokens.flatten(1, 2)], 1)
        queries = self.latent_queries[None].expand(anchors_cm.shape[0], -1, -1)
        attended, _ = self.encoder_attention(queries, tokens, tokens, need_weights=False)
        latent = queries + attended
        return self.latent_transformer(latent)

    def decode(self, latent: torch.Tensor, anchors_cm: torch.Tensor) -> dict[str, torch.Tensor]:
        geometry_x, geometry_t = self._expand_anchor_time(anchors_cm, 9)
        motion_x, motion_t = self._expand_anchor_time(anchors_cm, 8)
        geometry_query = (self.geometry_query(geometry_x)
                          + self.time_embedding(geometry_t)
                          + self.type_embedding.weight[0])
        motion_query = (self.motion_query(motion_x)
                        + self.time_embedding(motion_t)
                        + self.type_embedding.weight[1])
        queries = torch.cat([geometry_query.flatten(1, 2), motion_query.flatten(1, 2)], 1)
        attended, _ = self.decoder_attention(queries, latent, latent, need_weights=False)
        decoded = queries + attended
        geometry_count = 9 * anchors_cm.shape[1]
        geometry = self.geometry_head(decoded[:, :geometry_count]).reshape(
            anchors_cm.shape[0], 9, anchors_cm.shape[1], 4)
        motion = self.motion_head(decoded[:, geometry_count:]).reshape(
            anchors_cm.shape[0], 8, anchors_cm.shape[1], 3)
        return {"relative_geometry": geometry[..., :3],
                "relative_distance": geometry[..., 3], "relative_motion": motion}

    def forward(self, anchors_cm: torch.Tensor, relative_geometry_cm: torch.Tensor,
                relative_distance_cm: torch.Tensor,
                relative_motion_cm: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(anchors_cm, relative_geometry_cm,
                             relative_distance_cm, relative_motion_cm)
        output = self.decode(latent, anchors_cm)
        output["latent"] = latent
        return output
