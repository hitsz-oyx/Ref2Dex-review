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


def _masked_horizon_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over batch/feature axes for each horizon, ignoring invalid frames."""
    if values.ndim < 2 or mask.shape != values.shape[:2]:
        raise ValueError(f"Expected values [B,K,...] and mask [B,K], got {tuple(values.shape)} and {tuple(mask.shape)}")
    expanded = mask.to(dtype=values.dtype)
    for _ in range(values.ndim - 2):
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(values)
    reduce_dims = (0, *range(2, values.ndim))
    numerator = (values * expanded).sum(dim=reduce_dims)
    denominator = expanded.sum(dim=reduce_dims)
    zero = values.sum() * 0.0
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1.0), zero.expand_as(numerator))


def _masked_weighted_horizon_loss(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted loss over valid [B,K] horizons, renormalized after masking."""
    if values.ndim < 2 or mask.shape != values.shape[:2] or weights.shape != (values.shape[1],):
        raise ValueError(
            f"Expected values [B,K,...], mask [B,K], weights [K], got {tuple(values.shape)}, "
            f"{tuple(mask.shape)}, {tuple(weights.shape)}"
        )
    if values.ndim > 2:
        per_sample_horizon = values.mean(dim=tuple(range(2, values.ndim)))
    else:
        per_sample_horizon = values
    weighted_mask = mask.to(dtype=values.dtype) * weights[None, :]
    denominator = weighted_mask.sum()
    numerator = (per_sample_horizon * weighted_mask).sum()
    zero = values.sum() * 0.0
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1.0), zero)


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
        if "pred_hand_flow" not in output or "current_hand_points_object" not in output:
            raise KeyError("CmDecoderv2 point-flow training requires differentiable FK outputs from the model")
        target_hand_points = batch["target_hand_points_object"].float()
        target_hand_flow = target_hand_points - output["current_hand_points_object"].detach()[:, None]
        pred_hand_flow = output["pred_hand_flow"]
        weights = self._weights(pred_q.device, pred_q.dtype)
        cm_valid = output["cm_sample_valid"].bool()
        if bool(getattr(self.cfg.data, "active_only", False)):
            if "active_mask" not in batch:
                raise KeyError("active_only=True requires the dataset to provide active_mask")
            active_mask = batch["active_mask"].bool()
            if active_mask.shape != cm_valid.shape:
                raise ValueError(f"active_mask shape {active_mask.shape} != cm_sample_valid shape {cm_valid.shape}")
            supervision_mask = active_mask & cm_valid
        else:
            active_mask = torch.ones_like(cm_valid)
            supervision_mask = active_mask
        flow_values = F.smooth_l1_loss(
            pred_hand_flow,
            target_hand_flow,
            beta=float(self.cfg.meta.point_flow_smooth_l1_beta_m),
            reduction="none",
        )
        rotation_values = rotation_geodesic(pred_r, target_r)
        point_flow_loss = _masked_weighted_horizon_loss(flow_values, supervision_mask, weights)
        loss = float(self.cfg.meta.loss_point_flow_weight) * point_flow_loss
        point_flow_epe = _masked_horizon_mean(torch.linalg.vector_norm(pred_hand_flow - target_hand_flow, dim=-1), supervision_mask) * 1000.0
        q_mae = _masked_horizon_mean((pred_q - target_q).abs(), supervision_mask)
        translation_mm = _masked_horizon_mean(torch.linalg.vector_norm(pred_t - target_t, dim=-1), supervision_mask) * 1000.0
        rotation_deg = _masked_horizon_mean(rotation_values, supervision_mask) * (180.0 / math.pi)
        identity_q = _masked_horizon_mean(target_q.abs(), supervision_mask)
        identity_translation = _masked_horizon_mean(torch.linalg.vector_norm(target_t, dim=-1), supervision_mask) * 1000.0
        identity_rotation = _masked_horizon_mean(
            rotation_geodesic(torch.eye(3, device=target_r.device, dtype=target_r.dtype).expand_as(target_r), target_r),
            supervision_mask,
        ) * (180.0 / math.pi)
        metrics: dict[str, Any] = {
            "loss": loss,
            "loss/point_flow": point_flow_loss,
            "cm/valid_frame_ratio": cm_valid.float().mean(),
            "cm/active_frame_ratio": active_mask.float().mean(),
            "cm/supervision_frame_ratio": supervision_mask.float().mean(),
            "hand/point_flow_epe_mm": point_flow_epe.mean(),
            "q/mae_rad": q_mae.mean(),
            "wrist/translation_mm": translation_mm.mean(),
            "wrist/rotation_deg": rotation_deg.mean(),
            "baseline_identity/q_mae_rad": identity_q.mean(),
            "baseline_identity/wrist_translation_mm": identity_translation.mean(),
            "baseline_identity/wrist_rotation_deg": identity_rotation.mean(),
        }
        for horizon in range(pred_q.shape[1]):
            suffix = f"h{horizon + 1}"
            metrics[f"hand/point_flow_epe_mm_{suffix}"] = point_flow_epe[horizon]
            metrics[f"q/mae_rad_{suffix}"] = q_mae[horizon]
            metrics[f"wrist/translation_mm_{suffix}"] = translation_mm[horizon]
            metrics[f"wrist/rotation_deg_{suffix}"] = rotation_deg[horizon]
            metrics[f"baseline_identity/q_mae_rad_{suffix}"] = identity_q[horizon]
            metrics[f"baseline_identity/wrist_translation_mm_{suffix}"] = identity_translation[horizon]
            metrics[f"baseline_identity/wrist_rotation_deg_{suffix}"] = identity_rotation[horizon]
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=int(pred_q.shape[0]))

    def select_step_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        keys = {"loss", "loss/point_flow", "hand/point_flow_epe_mm_h1", "q/mae_rad_h1", "wrist/translation_mm_h1", "wrist/rotation_deg_h1", "cm/valid_frame_ratio", "cm/active_frame_ratio", "cm/supervision_frame_ratio"}
        return {key: value for key, value in metrics.items() if key in keys}
