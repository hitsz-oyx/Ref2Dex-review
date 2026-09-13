"""Matched Temporal-D2 realizers for the V1.1.16 F7 and direct-H gates.

The module deliberately contains no MANO layer and no retargeting optimizer.
Both variants produce the same ``delta_q6 + delta_wrist6`` contract as the
existing decoder.  ``F7`` is the frozen V20 object-indexed field
``[r(3), d(1), v(3)]``; ``H`` is a direct MANO-H source condition used only for
the representation attribution control.
"""
from __future__ import annotations

from typing import Final

import torch
from torch import nn

from src.task.InteractionDynamics.interaction_field import (
    build_relative_distance,
    build_relative_geometry,
    build_soft_correspondence,
)


F7_DIM: Final[int] = 7
F7_ANCHORS: Final[int] = 128
MANO_H_DIM: Final[int] = 43  # object-frame wrist(3+6), pose(24), betas(10)


def build_f7_field(
    hand_points_object: torch.Tensor,
    object_anchors: torch.Tensor,
    *,
    tau_m: float = 0.015,
) -> torch.Tensor:
    """Build causal V20 ``F7=[r,d,v]`` without penetration/contact channels.

    The returned tensor has shape ``[T-1, N, 7]``.  The velocity uses the
    correspondence weights at the *new* frame, exactly as V20's causal
    implementation does.  Points must already be in a consistent object
    frame; this function never consumes object pose or future task effect.
    """
    if hand_points_object.ndim != 3 or object_anchors.ndim != 2:
        raise ValueError("Expected hand_points_object [T,P,3] and object_anchors [N,3]")
    if hand_points_object.shape[-1] != 3 or object_anchors.shape[-1] != 3:
        raise ValueError("F7 geometry must be three-dimensional")
    if hand_points_object.shape[0] < 2:
        raise ValueError("F7 requires at least two causal frames")
    weights = build_soft_correspondence(hand_points_object, object_anchors, tau_m)
    relative = build_relative_geometry(hand_points_object, object_anchors, weights)
    distance = build_relative_distance(hand_points_object, object_anchors, weights)
    velocity = (
        weights[1:, ..., None]
        * (hand_points_object[1:] - hand_points_object[:-1])[:, None]
    ).sum(-2)
    return torch.cat([relative[1:], distance[1:, :, None], velocity], dim=-1)


def zero_f7(field: torch.Tensor) -> torch.Tensor:
    """C0 intervention: zero physical F7 before any model normalization."""
    if field.shape[-1] != F7_DIM:
        raise ValueError(f"Expected last dimension {F7_DIM}, got {field.shape[-1]}")
    return torch.zeros_like(field)


def spatial_shuffle_f7(field: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Cs intervention: shuffle field vectors while keeping anchor geometry.

    ``permutation`` is shared by every time step in a sample.  The caller may
    choose one permutation per rollout; no anchor positions are modified here.
    """
    if field.ndim != 4 or field.shape[-1] != F7_DIM:
        raise ValueError("Expected field [B,K,N,7]")
    if permutation.ndim != 2 or permutation.shape != (field.shape[0], field.shape[2]):
        raise ValueError("Expected permutation [B,N]")
    if permutation.dtype != torch.long:
        raise TypeError("permutation must have dtype torch.long")
    if permutation.device != field.device:
        raise ValueError("permutation and field must share a device")
    gather = permutation[:, None, :, None].expand_as(field)
    return torch.gather(field, dim=2, index=gather)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class MatchedTemporalD2Core(nn.Module):
    """Temporal-D2 with a configurable source token dimension/count.

    Transformer width, depth, query construction and action heads mirror the
    existing ``TemporalD2Core``.  Only source token dimension and count vary
    between F7 and direct-H controls, so parameter/FLOP differences are
    measurable rather than hidden behind separate architectures.
    """

    def __init__(
        self,
        *,
        window_size: int = 4,
        token_count: int = F7_ANCHORS,
        token_dim: int = F7_DIM,
        link_count: int = 18,
        link_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if window_size < 1 or token_count < 1:
            raise ValueError("window_size and token_count must be positive")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.window_size = int(window_size)
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.link_count = int(link_count)
        self.token_projection = _mlp(token_dim + 6, hidden_dim, hidden_dim)
        self.state_projection = _mlp(15, hidden_dim, hidden_dim)
        self.link_projection = _mlp(link_feature_dim, hidden_dim, hidden_dim)
        self.time_embedding = nn.Embedding(window_size, hidden_dim)
        self.future_embedding = nn.Embedding(window_size, hidden_dim)
        self.link_embedding = nn.Embedding(link_count, hidden_dim)
        self.summary_type = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.readout = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 6)
        self.translation_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 3)
        for head in (self.q_head, self.translation_head, self.rotation_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        source_tokens: torch.Tensor,
        source_geometry: torch.Tensor,
        current_state: torch.Tensor,
        current_link_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if source_tokens.ndim != 4:
            raise ValueError("Expected source_tokens [B,K,N,D]")
        batch, window, count, dimension = source_tokens.shape
        if (window, count, dimension) != (self.window_size, self.token_count, self.token_dim):
            raise ValueError(
                f"Expected source shape [B,{self.window_size},{self.token_count},{self.token_dim}], "
                f"got {tuple(source_tokens.shape)}"
            )
        if source_geometry.shape != (batch, window, count, 6):
            raise ValueError("source_geometry must be [B,K,N,6]")
        if current_state.shape != (batch, 15):
            raise ValueError("current_state must be [B,15]")
        if current_link_features.shape != (batch, self.link_count, 10):
            raise ValueError(f"current_link_features must be [B,{self.link_count},10]")
        time_ids = torch.arange(window, device=source_tokens.device)
        memory = self.token_projection(torch.cat([source_tokens, source_geometry], dim=-1))
        memory = memory + self.time_embedding(time_ids)[None, :, None, :]
        memory = memory.reshape(batch, window * count, -1)
        future = self.future_embedding(time_ids)
        summary = self.state_projection(current_state)[:, None, :] + future[None, :, :] + self.summary_type
        link_ids = torch.arange(self.link_count, device=source_tokens.device)
        links = self.link_projection(current_link_features) + self.link_embedding(link_ids)[None, :, :]
        links = links[:, None, :, :] + future[None, :, None, :]
        queries = torch.cat([summary[:, :, None, :], links], dim=2).reshape(
            batch, window * (self.link_count + 1), -1
        )
        decoded = self.decoder(tgt=queries, memory=memory).reshape(batch, window, self.link_count + 1, -1)
        horizon_feature = self.readout(torch.cat([decoded[:, :, 0], decoded[:, :, 1:].mean(dim=2)], dim=-1))
        return {
            "pred_q_delta": self.q_head(horizon_feature),
            "pred_wrist_translation": self.translation_head(horizon_feature),
            "pred_wrist_rotvec": self.rotation_head(horizon_feature),
        }


class InspireFieldRealizer(nn.Module):
    """B1 realizer: oracle MANO F7 to direct Inspire action targets."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.core = MatchedTemporalD2Core(token_count=F7_ANCHORS, token_dim=F7_DIM, **kwargs)

    def forward(
        self,
        f7: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_normal: torch.Tensor,
        current_state: torch.Tensor,
        current_link_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if anchor_pos.shape != (*f7.shape[:-1], 3) or anchor_normal.shape != anchor_pos.shape:
            raise ValueError("F7 anchor positions/normals must match [B,K,128,3]")
        geometry = torch.cat([anchor_pos, anchor_normal], dim=-1)
        return self.core(f7, geometry, current_state, current_link_features)


class DirectManoHRealizer(nn.Module):
    """D control: direct MANO-H tokens with the same action/query core."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.core = MatchedTemporalD2Core(token_count=1, token_dim=MANO_H_DIM, **kwargs)

    def forward(
        self,
        mano_h: torch.Tensor,
        current_state: torch.Tensor,
        current_link_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if mano_h.ndim != 3 or mano_h.shape[-1] != MANO_H_DIM:
            raise ValueError("Expected mano_h [B,K,43] = wrist(9)+pose(24)+betas(10)")
        batch, window, _ = mano_h.shape
        geometry = mano_h.new_zeros((batch, window, 1, 6))
        return self.core(mano_h[:, :, None, :], geometry, current_state, current_link_features)
