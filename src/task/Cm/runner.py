"""BaseRunner integration for CmAction temporal point-flow learning."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.Cm.dataset import make_dataloaders


class CmActionRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(
            data_cfg,
            seed,
            meta_cfg=self.cfg.meta,
            distributed=self.distributed,
        )

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        super().configure_data(metadata, train_dataset)
        stage4_frame = str(metadata.get("coordinate_frame", ""))
        expected_frame = str(self.cfg.meta.coordinate_frame)
        if stage4_frame != expected_frame:
            raise ValueError(
                f"Stage 4 coordinate_frame={stage4_frame!r}, but Cm config expects "
                f"{expected_frame!r}. Regenerate Stage 4 or set meta.coordinate_frame explicitly."
            )
        if int(metadata.get("num_obj_points", self.cfg.meta.num_obj_points)) != int(self.cfg.meta.num_obj_points):
            raise ValueError("Stage 4 runtime object sample count does not match Cm config.")
        if int(metadata.get("num_hand_points", self.cfg.meta.num_hand_points)) != int(self.cfg.meta.num_hand_points):
            raise ValueError("Stage 4 hand point count does not match Cm config.")

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def step(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> RunnerOutput:
        del mode
        prediction = model(batch)
        valid = batch["obj_valid_mask"].bool()
        valid_count = valid.sum()
        if int(valid_count.detach().item()) <= 0:
            raise RuntimeError(
                "CmAction received a batch with no valid object candidate. "
                "Use data.active_only=true for flow training."
            )
        pred_flow = prediction["pred_obj_flow"]
        gt_flow = batch["obj_flow_gt"].float()
        beta = float(self.cfg.meta.flow_smooth_l1_beta)
        smooth_l1_map = F.smooth_l1_loss(pred_flow, gt_flow, beta=beta, reduction="none").mean(dim=-1)
        flow_smooth_l1 = (smooth_l1_map * valid.float()).sum() / valid_count.float()
        squared_map = (pred_flow - gt_flow).square().mean(dim=-1)
        absolute_map = (pred_flow - gt_flow).abs().mean(dim=-1)
        flow_mse = (squared_map * valid.float()).sum() / valid_count.float()
        flow_mae = (absolute_map * valid.float()).sum() / valid_count.float()
        gt_flow_norm = (torch.linalg.norm(gt_flow, dim=-1) * valid.float()).sum() / valid_count.float()
        pred_flow_norm = (torch.linalg.norm(pred_flow, dim=-1) * valid.float()).sum() / valid_count.float()
        cm_assignment = prediction["cm_assignment"].clamp_min(1e-8)
        slot_assignment_entropy = -(cm_assignment * cm_assignment.log()).sum(dim=1).mean()
        cm_slot_weights = prediction["cm_slot_weights"]
        num_slots = cm_slot_weights.shape[1]
        if num_slots > 1:
            normalized_weights = torch.nn.functional.normalize(cm_slot_weights, dim=-1, eps=1e-8)
            slot_similarity = normalized_weights @ normalized_weights.transpose(1, 2)
            off_diagonal = ~torch.eye(num_slots, device=slot_similarity.device, dtype=torch.bool)
            slot_weight_overlap = slot_similarity[:, off_diagonal].mean()
        else:
            slot_weight_overlap = cm_slot_weights.new_zeros(())
        total_loss = float(self.cfg.meta.loss_flow_weight) * flow_smooth_l1
        metrics: dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow_smooth_l1": flow_smooth_l1,
            "flow_mse": flow_mse,
            "flow_mae": flow_mae,
            "gt_flow_norm": gt_flow_norm,
            "pred_flow_norm": pred_flow_norm,
            "slot_assignment_entropy": slot_assignment_entropy,
            "slot_weight_overlap": slot_weight_overlap,
            "valid_object_count": valid_count.float(),
        }
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(pred_flow.shape[0]))
