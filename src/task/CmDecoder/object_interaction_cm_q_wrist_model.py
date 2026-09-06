"""Direct q+wrist decoder whose only learned input is frozen ObjectInteractionCm."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel


class ObjectInteractionCmQWristDecoder(nn.Module):
    """Decode joint and wrist motion directly from ObjectInteractionCm tokens.

    The ObjectInteractionCm encoder is loaded from a checkpoint and is kept in
    eval/no-grad mode.  The trainable head consumes only ``cm_tokens``; q_t is
    used after the head to form ``pred_q_next`` for the residual target and
    metrics, never as a network feature.
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
            raise ValueError(f"V1.2.13 requires num_hand_points=1538, got {self.num_hand_points}")
        if self.max_hand_points != 3076:
            raise ValueError(f"ObjectInteractionCm contract requires max_hand_points=3076, got {self.max_hand_points}")
        if self.num_obj_points != 1024:
            raise ValueError(f"ObjectInteractionCm contract requires num_obj_points=1024, got {self.num_obj_points}")
        if int(self.cm.feature_dim) != 32 or int(self.cm.num_slots) != 16:
            raise ValueError(
                "Unsupported ObjectInteractionCm token contract: "
                f"feature_dim={self.cm.feature_dim}, num_slots={self.cm.num_slots}"
            )

        self.prediction_target = str(getattr(cfg.meta, "prediction_target", "delta_q"))
        if self.prediction_target != "delta_q":
            raise ValueError("V1.2.13 direct q+wrist decoder requires prediction_target='delta_q'")
        self.q_target_scale = float(getattr(cfg.meta, "q_target_scale", 1.0))
        self.wrist_translation_target_scale = float(
            getattr(cfg.meta, "wrist_translation_target_scale", 100.0)
        )
        self.wrist_rotation_target_scale = float(
            getattr(cfg.meta, "wrist_rotation_target_scale", 1.0)
        )
        if min(self.q_target_scale, self.wrist_translation_target_scale, self.wrist_rotation_target_scale) <= 0.0:
            raise ValueError("q/wrist target scales must be positive")

        token_dim = int(self.cm.num_slots * self.cm.feature_dim)
        self.decoder = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 12),
        )
        # Start from the identity q/wrist motion baseline.  The final layer is
        # still fully trainable; zero initialization only prevents a random
        # first-step pose jump before the first optimizer update.
        final = self.decoder[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cm.eval()
        return self

    def _padded_hand_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        hand_points = batch["hand_points"].float()
        hand_normals = batch["hand_normals"].float()
        hand_flow = batch["hand_flow"].float()
        expected = (self.num_hand_points, 3)
        if hand_points.ndim != 3 or tuple(hand_points.shape[1:]) != expected:
            raise ValueError(f"Expected hand_points [B,1538,3], got {tuple(hand_points.shape)}")
        if hand_normals.shape != hand_points.shape or hand_flow.shape != hand_points.shape:
            raise ValueError("hand_points, hand_normals and hand_flow must have identical [B,1538,3] shapes")
        pad = self.max_hand_points - self.num_hand_points
        if pad:
            zeros = torch.zeros(
                hand_points.shape[0], pad, 3, dtype=hand_points.dtype, device=hand_points.device
            )
            hand_points = torch.cat([hand_points, zeros], dim=1)
            hand_normals = torch.cat([hand_normals, zeros], dim=1)
            hand_flow = torch.cat([hand_flow, zeros], dim=1)
        valid = torch.zeros(
            hand_points.shape[0], self.max_hand_points, dtype=torch.bool, device=hand_points.device
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
        cm_tokens = cm_output["cm_tokens"]
        if cm_tokens.ndim != 3 or cm_tokens.shape[1:] != (self.cm.num_slots, self.cm.feature_dim):
            raise ValueError(f"Unexpected cm_tokens shape: {tuple(cm_tokens.shape)}")
        prediction_scaled = self.decoder(cm_tokens.flatten(1))
        pred_delta_q = prediction_scaled[:, :6] / self.q_target_scale
        pred_q_next = batch["q_t"].float() + pred_delta_q
        pred_translation = prediction_scaled[:, 6:9] / self.wrist_translation_target_scale
        pred_rotvec = prediction_scaled[:, 9:12] / self.wrist_rotation_target_scale
        return {
            "pred_q_next": pred_q_next,
            "pred_q_next_scaled": pred_q_next * self.q_target_scale,
            "pred_delta_q": pred_delta_q,
            "pred_delta_q_scaled": prediction_scaled[:, :6],
            "pred_target_scaled": prediction_scaled[:, :6],
            "pred_wrist_delta_translation": pred_translation,
            "pred_wrist_delta_translation_scaled": prediction_scaled[:, 6:9],
            "pred_wrist_delta_rotvec": pred_rotvec,
            "pred_wrist_delta_rotvec_scaled": prediction_scaled[:, 9:12],
            "cm_tokens": cm_tokens,
        }
