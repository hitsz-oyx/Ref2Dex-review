"""BaseRunner integration for ObjectInteractionCm V1.1."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, MetricStat, RunnerOutput

from .dataset import make_dataloaders


def masked_vector_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta: float = 0.005,
) -> torch.Tensor:
    """Huber loss on vector EPE with a safe empty-mask branch."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    if mask.shape != prediction.shape[:-1]:
        raise ValueError(f"Mask shape mismatch: {mask.shape} != {prediction.shape[:-1]}")
    residual = torch.linalg.vector_norm(prediction - target, dim=-1)
    values = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=float(beta), reduction="none")
    mask_float = mask.to(dtype=values.dtype)
    denominator = mask_float.sum()
    # Preserve a zero-gradient graph when no hand point falls within 3 cm.
    return torch.where(
        denominator > 0,
        (values * mask_float).sum() / denominator.clamp_min(1.0),
        prediction.sum() * 0.0,
    )


class ObjectInteractionCmRunner(BaseRunner):
    """Train/evaluate the independent object-side interaction representation."""

    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(
            data_cfg,
            seed,
            meta_cfg=self.cfg.meta,
            distributed=self.distributed,
        )

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        super().configure_data(metadata, train_dataset)
        expected = {
            "coordinate_frame": str(self.cfg.meta.coordinate_frame),
            "num_obj_pool": int(self.cfg.meta.num_obj_pool),
            "num_obj_points": int(self.cfg.meta.num_obj_points),
            "num_hand_points": int(self.cfg.meta.num_hand_points),
            "max_hand_points": int(getattr(self.cfg.meta, "max_hand_points", int(self.cfg.meta.num_hand_points) * 2)),
        }
        for key, value in expected.items():
            if key in metadata and (str(metadata[key]) if isinstance(value, str) else int(metadata[key])) != value:
                raise ValueError(f"ObjectInteractionCm metadata {key}={metadata[key]!r} != config {value!r}")

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    @staticmethod
    def _hands(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if "hand_flow" in batch:
            valid = batch.get("hand_valid_mask")
            if valid is None:
                valid = torch.ones(batch["hand_flow"].shape[:2], dtype=torch.bool, device=batch["hand_flow"].device)
            return batch["hand_flow"].float(), batch["hand_supervision_mask"].bool() & valid.bool()
        flow = torch.cat([batch["left_hand_flow"], batch["right_hand_flow"]], dim=1).float()
        mask = batch.get("hand_supervision_mask")
        if mask is None:
            raise KeyError("Batch must contain hand_supervision_mask for 3 cm hand loss")
        return flow, mask.bool()

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train"):
        del mode
        prediction = model(batch)
        obj_mask = batch["obj_valid_mask"].bool()
        obj_target = batch["obj_flow_gt"].float()
        obj_prediction = prediction["pred_obj_flow"]
        obj_loss = masked_vector_huber(
            obj_prediction,
            obj_target,
            obj_mask,
            beta=float(self.cfg.meta.flow_smooth_l1_beta),
        )
        hand_target, hand_mask = self._hands(batch)
        hand_prediction = prediction["pred_hand_flow"]
        hand_loss = masked_vector_huber(
            hand_prediction,
            hand_target,
            hand_mask,
            beta=float(self.cfg.meta.flow_smooth_l1_beta),
        )
        total_loss = (
            float(getattr(self.cfg.meta, "loss_obj_flow_weight", 1.0)) * obj_loss
            + float(getattr(self.cfg.meta, "loss_hand_flow_weight", 1.0)) * hand_loss
        )
        obj_epe = torch.linalg.vector_norm(obj_prediction - obj_target, dim=-1)
        hand_epe = torch.linalg.vector_norm(hand_prediction - hand_target, dim=-1)
        obj_count = float(obj_mask.sum().detach())
        hand_count = float(hand_mask.sum().detach())
        assignment = prediction["cm_assignment"].clamp_min(1e-8)
        assignment_entropy = -(assignment * assignment.log()).sum(dim=1).mean()
        slot_weights = prediction["cm_slot_weights"].clamp_min(1e-8)
        slot_usage = slot_weights.mean(dim=-1)
        slot_usage = slot_usage / slot_usage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slot_entropy = -(slot_usage * slot_usage.log()).sum(dim=-1).mean()
        attention = prediction["interaction/attention_weights"]
        valid_edges = prediction["interaction/edge_valid_mask"]
        metrics: dict[str, Any] = {
            "loss": total_loss,
            "obj/loss": obj_loss,
            "hand/loss_3cm": hand_loss,
            "obj/flow_epe_mm": MetricStat(float((obj_epe * obj_mask.float() * 1000.0).sum().detach()), obj_count),
            "hand/flow_epe_3cm_mm": MetricStat(float((hand_epe * hand_mask.float() * 1000.0).sum().detach()), hand_count),
            "hand/active_points": hand_mask.float().sum(),
            "hand/active_ratio": hand_mask.float().mean(),
            "data/hand_valid_points": batch["hand_valid_mask"].float().sum(dim=-1).mean(),
            "data/min_hand_object_distance_mm": batch["min_hand_object_distance_mm"].float().mean(),
            "interaction/object_points_with_hand_neighbor": prediction["interaction/has_interaction"].float().sum(dim=-1).mean(),
            "interaction/mean_valid_neighbors": valid_edges.float().sum(dim=-1).mean(),
            "interaction/attention_max": attention.max(dim=-1).values.mean(),
            "interaction/attention_entropy": -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1).mean(),
            "slot/assignment_entropy": assignment_entropy,
            "slot/entropy": slot_entropy,
            "slot/effective_count": slot_entropy.exp(),
            "slot/max_mass": slot_usage.max(dim=-1).values.mean(),
        }
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(obj_prediction.shape[0]))

    def select_step_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        keep = {
            key: value for key, value in metrics.items()
            if key in {"loss", "obj/loss", "hand/loss_3cm", "obj/flow_epe_mm", "hand/flow_epe_3cm_mm", "hand/active_ratio", "data/hand_valid_points", "data/min_hand_object_distance_mm", "interaction/attention_max", "slot/effective_count"}
        }
        return keep

    def evaluate_all(self) -> dict[str, float]:
        metrics = super().evaluate_all()
        # Model selection is defined on the equal-source mean, not on the
        # frame-count-weighted aggregate loader (the two source datasets have
        # different sequence/frame counts).  Keep per-source metrics intact.
        source_values = [
            metrics[key]
            for key in ("val/grab/obj/flow_epe_mm", "val/inspire_f1/obj/flow_epe_mm")
            if key in metrics
        ]
        if source_values:
            metrics["val/obj/flow_epe_mm"] = sum(source_values) / len(source_values)
        return metrics
