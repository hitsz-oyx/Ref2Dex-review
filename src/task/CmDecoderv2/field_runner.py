"""V1.1.16 B1 training runner with the existing point-flow objective."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.base import BaseRunner, RunnerOutput
from src.base.distributed import make_default_eval_sampler, make_default_train_sampler

from .dataset import _resolve
from .field_dataset import FieldRealizerDataset
from .runner import axis_angle_to_matrix, rotation_geodesic, _masked_horizon_mean, _masked_weighted_horizon_loss


class FieldRealizerRunner(BaseRunner):
    """Train B1 using parent-only F7 and actual Inspire point-flow labels."""

    def make_dataloaders(self, data_cfg: Any, seed: int):
        index = __import__("json").loads(_resolve(data_cfg.index_path).read_text(encoding="utf-8"))
        root = _resolve(data_cfg.field_root)
        train = FieldRealizerDataset(index["sequences"]["train"], field_root=root, urdf_path=_resolve(data_cfg.urdf_path), window_size=int(self.cfg.meta.window_size), active_only=bool(data_cfg.active_only))
        val = FieldRealizerDataset(index["sequences"]["val"], field_root=root, urdf_path=_resolve(data_cfg.urdf_path), window_size=int(self.cfg.meta.window_size), active_only=bool(data_cfg.active_only))
        train_sampler = make_default_train_sampler(train, shuffle=True, seed=seed, distributed=self.distributed, drop_last=False)
        val_sampler = make_default_eval_sampler(val, distributed=self.distributed)
        common = dict(num_workers=int(data_cfg.num_workers), pin_memory=bool(data_cfg.pin_memory), persistent_workers=bool(data_cfg.persistent_workers) and int(data_cfg.num_workers) > 0)
        train_loader = DataLoader(train, batch_size=int(data_cfg.batch_size), shuffle=train_sampler is None, sampler=train_sampler, **common)
        val_loader = DataLoader(val, batch_size=int(getattr(data_cfg, "val_batch_size", data_cfg.batch_size)), shuffle=False, sampler=val_sampler, **common)
        metadata = {
            "schema_name": "ref2dex_v1_1_16_parent_f7_field_realizer",
            "work_version": str(self.cfg.work_version),
            "coordinate_frame": "object_pose_t",
            "window_size": int(self.cfg.meta.window_size),
            "effective_fps": 30.0,
            "split_contract": "mano_source_actual_inspire",
            "field_definition": "F7=[r,d,v]; no p/c/contact/E",
            "field_root": str(root),
            "index_path": str(_resolve(data_cfg.index_path)),
            "training_windows": len(train),
            "validation_windows": len(val),
        }
        return train_loader, val_loader, None, metadata, {"val/": val_loader}, {}

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        super().configure_data(metadata, train_dataset)
        if metadata.get("field_definition") != "F7=[r,d,v]; no p/c/contact/E":
            raise ValueError("FieldRealizer requires the frozen V1.1.16 F7 contract")
        if int(metadata.get("window_size", -1)) != int(self.cfg.meta.window_size):
            raise ValueError("FieldRealizer window size changed")

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg)

    def _weights(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        values = torch.pow(torch.tensor(float(self.cfg.meta.horizon_gamma), device=device, dtype=dtype), torch.arange(int(self.cfg.meta.window_size), device=device, dtype=dtype))
        return values / values.sum()

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        del mode
        output = model(batch)
        pred_r = axis_angle_to_matrix(output["pred_wrist_rotvec"])
        target_r = batch["target_wrist_rotation"].float()
        target_flow = batch["target_hand_points_object"].float() - output["current_hand_points_object"].detach()[:, None]
        valid = batch["active_mask"].bool()
        weights = self._weights(output["pred_q_delta"].device, output["pred_q_delta"].dtype)
        flow = F.smooth_l1_loss(output["pred_hand_flow"], target_flow, beta=float(self.cfg.meta.point_flow_smooth_l1_beta_m), reduction="none")
        flow_loss = _masked_weighted_horizon_loss(flow, valid, weights)
        loss = float(self.cfg.meta.loss_point_flow_weight) * flow_loss
        epe = _masked_horizon_mean(torch.linalg.vector_norm(output["pred_hand_flow"] - target_flow, dim=-1), valid) * 1000.0
        q_mae = _masked_horizon_mean((output["pred_q_delta"] - batch["target_q_delta"].float()).abs(), valid)
        wrist_mm = _masked_horizon_mean(torch.linalg.vector_norm(output["pred_wrist_translation"] - batch["target_wrist_translation"].float(), dim=-1), valid) * 1000.0
        rot_deg = _masked_horizon_mean(rotation_geodesic(pred_r, target_r), valid) * (180.0 / math.pi)
        return RunnerOutput(loss=loss, metrics={"loss": loss, "loss/point_flow": flow_loss, "hand/point_flow_epe_mm": epe.mean(), "q/mae_rad": q_mae.mean(), "wrist/translation_mm": wrist_mm.mean(), "wrist/rotation_deg": rot_deg.mean()}, batch_size=int(output["pred_q_delta"].shape[0]))
