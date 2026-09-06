"""Training losses and BaseRunner integration for CmDecoderv2."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from pytorch3d.transforms import axis_angle_to_matrix as _pytorch3d_axis_angle_to_matrix

from src.base import BaseRunner, RunnerOutput

from .dataset import make_dataloaders


def axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Quaternion-backed exponential map with finite identity gradients."""
    return _pytorch3d_axis_angle_to_matrix(rotvec, fast=False)


def rotation_geodesic(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = prediction.transpose(-1, -2) @ target
    vector = torch.stack([relative[..., 2, 1] - relative[..., 1, 2], relative[..., 0, 2] - relative[..., 2, 0], relative[..., 1, 0] - relative[..., 0, 1]], dim=-1)
    # A tiny squared-norm floor keeps the identity rotation's gradient finite;
    # ``linalg.vector_norm`` has an undefined derivative at exactly zero.
    sine = 0.5 * torch.sqrt((vector * vector).sum(dim=-1) + 1e-12)
    cosine = 0.5 * (relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0)
    return torch.atan2(sine, cosine)


class CmDecoderV2Runner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(data_cfg, seed, meta_cfg=self.cfg.meta, distributed=self.distributed)

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        super().configure_data(metadata, train_dataset)
        expected = {
            "modification_version": str(self.cfg.modification_version),
            "coordinate_frame": str(self.cfg.meta.coordinate_frame),
            "window_size": int(self.cfg.meta.window_size),
            "effective_fps": float(self.cfg.meta.effective_fps),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"CmDecoderv2 metadata {key}={metadata.get(key)!r} != {value!r}")
        if metadata.get("split_contract") != {"train": "inspire_rl", "val": "inspire_rl", "test": "mano_qualitative_only"}:
            raise ValueError("CmDecoderv2 must train/validate on RL-Inspire and reserve MANO test for visualization")
        checkpoint = Path(str(self.cfg.model.oicm_checkpoint))
        if not checkpoint.is_absolute():
            checkpoint = (Path.cwd() / checkpoint).resolve()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.metadata["oicm_checkpoint"] = str(checkpoint)
        self.metadata["oicm_checkpoint_sha256"] = digest.hexdigest()

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg)

    def _weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = int(self.cfg.meta.window_size)
        gamma = float(self.cfg.meta.horizon_gamma)
        values = torch.pow(torch.tensor(gamma, device=device, dtype=dtype), torch.arange(k, device=device, dtype=dtype))
        return values / values.sum()

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        del mode
        output = model(batch)
        pred_q = output["pred_q_delta"]
        pred_t = output["pred_wrist_translation"]
        pred_r = axis_angle_to_matrix(output["pred_wrist_rotvec"])
        target_q = batch["target_q_delta"].float()
        target_t = batch["target_wrist_translation"].float()
        target_r = batch["target_wrist_rotation"].float()
        weights = self._weights(pred_q.device, pred_q.dtype)
        q_per_h = F.smooth_l1_loss(pred_q, target_q, beta=float(self.cfg.meta.q_smooth_l1_beta_rad), reduction="none").mean(dim=(0, 2))
        translation_scale = float(self.cfg.meta.translation_scale)
        translation_beta = float(self.cfg.meta.translation_smooth_l1_beta_m) * float(self.cfg.meta.translation_scale)
        translation_per_h = F.smooth_l1_loss(
            pred_t * translation_scale,
            target_t * translation_scale,
            beta=translation_beta,
            reduction="none",
        ).mean(dim=(0, 2))
        rotation_values = rotation_geodesic(pred_r, target_r)
        rotation_per_h = rotation_values.mean(dim=0)
        q_loss = (weights * q_per_h).sum()
        translation_loss = (weights * translation_per_h).sum()
        rotation_loss = (weights * rotation_per_h).sum()
        loss = (
            float(self.cfg.meta.loss_q_weight) * q_loss
            + float(self.cfg.meta.loss_translation_weight) * translation_loss
            + float(self.cfg.meta.loss_rotation_weight) * rotation_loss
        )
        q_mae = (pred_q - target_q).abs().mean(dim=(0, 2))
        translation_mm = torch.linalg.vector_norm(pred_t - target_t, dim=-1).mean(dim=0) * 1000.0
        rotation_deg = rotation_values.mean(dim=0) * (180.0 / math.pi)
        identity_q = target_q.abs().mean(dim=(0, 2))
        identity_translation = torch.linalg.vector_norm(target_t, dim=-1).mean(dim=0) * 1000.0
        identity_rotation = rotation_geodesic(torch.eye(3, device=target_r.device, dtype=target_r.dtype).expand_as(target_r), target_r).mean(dim=0) * (180.0 / math.pi)
        metrics: dict[str, Any] = {
            "loss": loss,
            "loss/q": q_loss,
            "loss/wrist_translation": translation_loss,
            "loss/wrist_rotation": rotation_loss,
            "cm/valid_frame_ratio": output["cm_sample_valid"].float().mean(),
            "q/mae_rad": q_mae.mean(),
            "wrist/translation_mm": translation_mm.mean(),
            "wrist/rotation_deg": rotation_deg.mean(),
            "baseline_identity/q_mae_rad": identity_q.mean(),
            "baseline_identity/wrist_translation_mm": identity_translation.mean(),
            "baseline_identity/wrist_rotation_deg": identity_rotation.mean(),
        }
        for horizon in range(pred_q.shape[1]):
            suffix = f"h{horizon + 1}"
            metrics[f"q/mae_rad_{suffix}"] = q_mae[horizon]
            metrics[f"wrist/translation_mm_{suffix}"] = translation_mm[horizon]
            metrics[f"wrist/rotation_deg_{suffix}"] = rotation_deg[horizon]
            metrics[f"baseline_identity/q_mae_rad_{suffix}"] = identity_q[horizon]
            metrics[f"baseline_identity/wrist_translation_mm_{suffix}"] = identity_translation[horizon]
            metrics[f"baseline_identity/wrist_rotation_deg_{suffix}"] = identity_rotation[horizon]
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=int(pred_q.shape[0]))

    def select_step_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        keys = {"loss", "loss/q", "loss/wrist_translation", "loss/wrist_rotation", "q/mae_rad_h1", "wrist/translation_mm_h1", "wrist/rotation_deg_h1", "cm/valid_frame_ratio"}
        return {key: value for key, value in metrics.items() if key in keys}
