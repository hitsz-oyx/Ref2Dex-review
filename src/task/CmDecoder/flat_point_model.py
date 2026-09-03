"""Fast per-point baseline: current hand xyz + flattened frozen Cm tokens."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.base.checkpoint import load_checkpoint
from src.base.base_config import task_config_from_dict
from src.task.Cm.src.model import CmFlowModel


class CmFlatPointFlowModel(nn.Module):
    """Shared point MLP conditioned on one flattened Cm vector per sample."""

    def __init__(self, cfg: Any, *, condition_shape: Any = None, target_shape: Any = None) -> None:
        del condition_shape, target_shape
        super().__init__()
        checkpoint = load_checkpoint(cfg.meta.cm_checkpoint, map_location="cpu")
        cm_cfg = task_config_from_dict(checkpoint["config"])
        self.cm = CmFlowModel(cm_cfg)
        self.cm.load_state_dict(checkpoint["model"], strict=True)
        self.cm.eval()
        for parameter in self.cm.parameters():
            parameter.requires_grad_(False)
        self.decoder_input = "hand_points_cm_flat"
        self.num_slots = int(self.cm.num_cm_tokens)
        self.cm_dim = int(self.cm.cm_dim)
        self.point_scale = float(getattr(cfg.meta, "flat_point_input_scale", 1.0))
        input_dim = 3 + self.num_slots * self.cm_dim
        hidden = int(getattr(cfg.meta, "flat_point_hidden_dim", 512))
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 3),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.cm.eval()
        return self

    @torch.no_grad()
    def _tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "cm_tokens" in batch:
            return batch["cm_tokens"]
        return self.cm(batch)["cm_tokens"]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        tokens = self._tokens(batch)
        flat = tokens.flatten(1)
        points = batch["hand_points"] * self.point_scale
        condition = flat[:, None, :].expand(-1, points.shape[1], -1)
        pred = self.point_mlp(torch.cat([points, condition], dim=-1))
        return {
            "pred_hand_flow": pred,
            "pred_hand_flow_scaled": pred,
            "pred_hand_points_next": batch["hand_points"] + pred,
            "cm_tokens": tokens,
        }
