"""BaseRunner integration for CmAction temporal point-flow learning."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.Cm.dataset import make_dataloaders


def internal_flow_smooth_l1(
    pred_flow_m: torch.Tensor,
    gt_flow_m: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta_m: float,
    internal_scale: float,
) -> torch.Tensor:
    """Compute the training loss in Cm-head internal units.

    Public Cm inputs, predictions, and metrics remain metres.  Multiplying
    both endpoints and the Huber transition by the internal scale removes the
    reciprocal scale factor from the decoder's gradient.
    """
    scale = float(internal_scale)
    if scale <= 0.0:
        raise ValueError("internal_point_flow_scale must be positive.")
    valid_count = valid_mask.sum()
    pred_internal = pred_flow_m * scale
    gt_internal = gt_flow_m * scale
    smooth_l1_map = F.smooth_l1_loss(
        pred_internal,
        gt_internal,
        beta=float(beta_m) * scale,
        reduction="none",
    ).mean(dim=-1)
    return (smooth_l1_map * valid_mask.float()).sum() / valid_count.float()


class CmActionRunner(BaseRunner):
    def evaluate_all(self) -> dict[str, float]:
        metrics = super().evaluate_all()
        stride_mse = [value for key, value in metrics.items() if key.endswith("/flow_mse") and "/stride_" in key]
        if stride_mse:
            metrics["val/mean_stride_flow_mse"] = float(sum(stride_mse) / len(stride_mse))
        return metrics

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
        flow_smooth_l1 = internal_flow_smooth_l1(
            pred_flow,
            gt_flow,
            valid,
            beta_m=float(self.cfg.meta.flow_smooth_l1_beta),
            internal_scale=float(self.cfg.meta.internal_point_flow_scale),
        )
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
        decoder_slot_usage = prediction["decoder_slot_usage"]
        mean_decoder_slot_usage = decoder_slot_usage.mean(dim=0)
        decoder_slot_usage_entropy = -(
            mean_decoder_slot_usage.clamp_min(1e-8)
            * mean_decoder_slot_usage.clamp_min(1e-8).log()
        ).sum()
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
            "decoder_slot_usage_entropy": decoder_slot_usage_entropy,
            "decoder_slot_usage_max": mean_decoder_slot_usage.max(),
            "valid_object_count": valid_count.float(),
        }
        metrics.update(
            {
                f"decoder_slot_usage/slot_{slot_idx:02d}": usage
                for slot_idx, usage in enumerate(mean_decoder_slot_usage)
            }
        )
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(pred_flow.shape[0]))
