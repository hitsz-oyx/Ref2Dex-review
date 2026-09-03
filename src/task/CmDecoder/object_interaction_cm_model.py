"""V1.2 hand-flow Decoder backed by a frozen ObjectInteractionCm encoder."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.task.ObjectInteractionCm.decoder import SharedHandFlowDecoder
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


class ObjectInteractionCmHandFlowAutoencoder(nn.Module):
    """Decode Inspire hand flow from frozen ObjectInteractionCm tokens.

    The encoder keeps the checkpoint's 3076-point padded-union contract.  The
    Decoder is a fresh, trainable 1538-point head, so no checkpoint decoder
    parameters are reused or updated.
    """

    def __init__(self, cfg: Any, *, condition_shape: Any = None, target_shape: Any = None) -> None:
        del condition_shape, target_shape
        super().__init__()
        checkpoint_path = str(cfg.meta.cm_checkpoint)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        if "config" not in checkpoint or "model" not in checkpoint:
            raise ValueError(f"Invalid ObjectInteractionCm checkpoint payload: {checkpoint_path}")
        cm_cfg = task_config_from_dict(checkpoint["config"])
        self.cm = ObjectInteractionCmModel(cm_cfg)
        self.cm.load_state_dict(checkpoint["model"], strict=True)
        self.cm.eval()
        for parameter in self.cm.parameters():
            parameter.requires_grad_(False)

        self.num_hand_points = int(getattr(cfg.meta, "num_hand_points", 1538))
        self.max_hand_points = int(getattr(cfg.meta, "max_hand_points", 3076))
        self.num_obj_points = int(getattr(cfg.meta, "num_obj_points", 1024))
        if self.num_hand_points != 1538:
            raise ValueError(f"V1.2 requires num_hand_points=1538, got {self.num_hand_points}")
        if self.max_hand_points != 3076:
            raise ValueError(f"ObjectInteractionCm V1.1 contract requires max_hand_points=3076, got {self.max_hand_points}")
        if self.num_obj_points != 1024:
            raise ValueError(f"ObjectInteractionCm V1.1 contract requires num_obj_points=1024, got {self.num_obj_points}")
        if int(self.cm.feature_dim) != 32 or int(self.cm.num_slots) != 16:
            raise ValueError(
                "Unsupported ObjectInteractionCm token contract: "
                f"feature_dim={self.cm.feature_dim}, num_slots={self.cm.num_slots}"
            )
        self.decoder = SharedHandFlowDecoder(dim=int(self.cm.feature_dim))
        self.point_flow_target_scale = float(getattr(cfg.meta, "point_flow_target_scale", 1.0))
        if self.point_flow_target_scale <= 0.0:
            raise ValueError("point_flow_target_scale must be positive")

    def train(self, mode: bool = True):
        super().train(mode)
        # BaseRunner calls train() before every training epoch.  Keep the
        # frozen encoder in eval mode so no stateful layer can drift.
        self.cm.eval()
        return self

    def _padded_hand_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hand_points = batch["hand_points"].float()
        hand_normals = batch["hand_normals"].float()
        hand_flow = batch["hand_flow"].float()
        if hand_points.ndim != 3 or hand_points.shape[1] != self.num_hand_points:
            raise ValueError(f"Expected real Inspire hand points [B,1538,3], got {tuple(hand_points.shape)}")
        if hand_normals.shape != hand_points.shape or hand_flow.shape != hand_points.shape:
            raise ValueError(
                "hand_points/hand_normals/hand_flow must have identical [B,1538,3] shapes: "
                f"{tuple(hand_points.shape)}, {tuple(hand_normals.shape)}, {tuple(hand_flow.shape)}"
            )
        pad = self.max_hand_points - self.num_hand_points
        if pad:
            zeros = torch.zeros((hand_points.shape[0], pad, 3), dtype=hand_points.dtype, device=hand_points.device)
            hand_points = torch.cat([hand_points, zeros], dim=1)
            hand_normals = torch.cat([hand_normals, zeros], dim=1)
            hand_flow = torch.cat([hand_flow, zeros], dim=1)
        valid = torch.zeros(
            (hand_points.shape[0], self.max_hand_points), dtype=torch.bool, device=hand_points.device
        )
        valid[:, : self.num_hand_points] = True
        cm_batch = dict(batch)
        cm_batch.update(
            hand_points=hand_points,
            hand_normals=hand_normals,
            hand_flow=hand_flow,
            hand_valid_mask=valid,
        )
        return cm_batch

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cm_batch = self._padded_hand_batch(batch)
        with torch.no_grad():
            cm_output = self.cm(cm_batch)
        pred_hand_flow = self.decoder(
            batch["hand_points"].float(),
            batch["hand_normals"].float(),
            cm_output["cm_tokens"],
            cm_output["cm_anchor_pos"],
            cm_output["cm_anchor_normal"],
        )[0]
        scale = self.point_flow_target_scale
        return {
            "pred_hand_flow": pred_hand_flow,
            "pred_hand_flow_scaled": pred_hand_flow * scale,
            "pred_hand_points_next": batch["hand_points"] + pred_hand_flow,
            "cm_tokens": cm_output["cm_tokens"],
            "cm_anchor_pos": cm_output["cm_anchor_pos"],
            "cm_anchor_normal": cm_output["cm_anchor_normal"],
        }
