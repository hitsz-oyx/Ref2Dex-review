"""BaseRunner integration for ObjectInteractionCm V1.1."""
from __future__ import annotations

from contextlib import nullcontext
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
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Huber loss on vector EPE with a safe empty-mask branch."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}")
    if mask.shape != prediction.shape[:-1]:
        raise ValueError(f"Mask shape mismatch: {mask.shape} != {prediction.shape[:-1]}")
    if sample_mask is not None:
        if sample_mask.shape != (prediction.shape[0],):
            raise ValueError(f"sample_mask shape mismatch: {sample_mask.shape}")
        mask = mask.bool() & sample_mask[:, None].bool()
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


def per_sample_vector_huber(prediction, target, mask, *, beta: float = 0.005) -> torch.Tensor:
    """Point-mean Huber loss per sample; empty point masks produce zero."""
    if prediction.shape != target.shape or mask.shape != prediction.shape[:-1]:
        raise ValueError("Prediction, target and mask shapes are inconsistent")
    residual = torch.linalg.vector_norm(prediction - target, dim=-1)
    values = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=float(beta), reduction="none")
    weights = mask.to(values.dtype)
    return torch.where(weights.sum(dim=1) > 0, (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0), prediction.sum(dim=(-1, -2)) * 0.0)


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
            "max_knn_hand_points": int(getattr(self.cfg.meta, "max_knn_hand_points", 0) or 0),
        }
        for key, value in expected.items():
            if key in metadata and (str(metadata[key]) if isinstance(value, str) else int(metadata[key])) != value:
                raise ValueError(f"ObjectInteractionCm metadata {key}={metadata[key]!r} != config {value!r}")
        scale_path = str(getattr(self.cfg.meta, "scale_manifest_path", "") or "").strip()
        if scale_path:
            from pathlib import Path
            import json
            path = Path(scale_path)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.metadata["scale_manifest_path"] = str(path)
                self.metadata["scale_values"] = payload.get("scales", payload)

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def train_step(self, batch: Any) -> dict[str, float]:
        """Task-local variant that skips optimizer updates for an all-invalid batch."""
        self._require_train_ready()
        self.optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self._autocast_dtype(), enabled=True)
            if self.cfg.train.amp and self.device.type == "cuda" else nullcontext()
        )
        with self._performance_section("forward_and_loss"):
            with autocast_ctx:
                output = self.step(self.model, batch, mode="train")
                loss = output.loss
        metrics = dict(output.metrics)
        all_invalid = bool(metrics.pop("_all_samples_invalid", False))
        if all_invalid:
            metrics["optimizer_skipped_all_invalid"] = 1.0
            metrics.setdefault("loss", loss.detach())
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["grad_norm"] = 0.0
            return metrics
        with self._performance_section("backward"):
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        with self._performance_section("optimizer"):
            grad_clip_norm = self.cfg.train.grad_clip_norm
            if grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in self.model.parameters() if p.grad is not None))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
        metrics.setdefault("loss", loss.detach())
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        metrics["grad_norm"] = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
        if grad_clip_norm is not None:
            metrics["grad_clipped"] = float(metrics["grad_norm"] > float(grad_clip_norm))
        return metrics

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
        sample_valid = prediction.get("sample_valid", torch.ones(obj_prediction.shape[0], dtype=torch.bool, device=obj_prediction.device)).bool()
        obj_loss_per_sample = per_sample_vector_huber(obj_prediction, obj_target, obj_mask, beta=float(self.cfg.meta.flow_smooth_l1_beta))
        hand_target, hand_mask = self._hands(batch)
        hand_prediction = prediction["pred_hand_flow"]
        hand_loss_per_sample = per_sample_vector_huber(hand_prediction, hand_target, hand_mask, beta=float(self.cfg.meta.flow_smooth_l1_beta))
        valid_count = sample_valid.to(obj_prediction.dtype).sum()
        per_sample_total = float(getattr(self.cfg.meta, "loss_obj_flow_weight", 1.0)) * obj_loss_per_sample + float(getattr(self.cfg.meta, "loss_hand_flow_weight", 1.0)) * hand_loss_per_sample
        total_loss = torch.where(valid_count > 0, (per_sample_total * sample_valid.to(per_sample_total.dtype)).sum() / valid_count.clamp_min(1.0), obj_prediction.sum() * 0.0)
        obj_loss = torch.where(valid_count > 0, (obj_loss_per_sample * sample_valid.to(obj_loss_per_sample.dtype)).sum() / valid_count.clamp_min(1.0), obj_prediction.sum() * 0.0)
        hand_loss = torch.where(valid_count > 0, (hand_loss_per_sample * sample_valid.to(hand_loss_per_sample.dtype)).sum() / valid_count.clamp_min(1.0), hand_prediction.sum() * 0.0)
        obj_epe = torch.linalg.vector_norm(obj_prediction - obj_target, dim=-1)
        hand_epe = torch.linalg.vector_norm(hand_prediction - hand_target, dim=-1)
        valid_obj_mask = obj_mask & sample_valid[:, None]
        valid_hand_mask = hand_mask & sample_valid[:, None]
        obj_count = float(valid_obj_mask.sum().detach())
        hand_count = float(valid_hand_mask.sum().detach())
        assignment = prediction["cm_assignment"].clamp_min(1e-8)
        assignment_entropy = -(assignment * assignment.clamp_min(1e-8).log()).sum(dim=1)
        slot_weights = prediction["cm_slot_weights"].clamp_min(1e-8)
        slot_usage = slot_weights.mean(dim=-1)
        slot_usage = slot_usage / slot_usage.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        slot_entropy = -(slot_usage * slot_usage.clamp_min(1e-8).log()).sum(dim=-1)
        scalar_sample_mask = sample_valid.to(obj_prediction.dtype)
        def sample_mean(value):
            if value.ndim == 1:
                per = value
            else:
                per = value.reshape(value.shape[0], -1).mean(dim=1)
            return torch.where(valid_count > 0, (per * scalar_sample_mask).sum() / valid_count.clamp_min(1.0), value.sum() * 0.0)
        attention = prediction["interaction/attention_weights"]
        valid_edges = prediction["interaction/edge_valid_mask"]
        metrics: dict[str, Any] = {
            "loss": total_loss,
            "obj/loss": obj_loss,
            "hand/loss_3cm": hand_loss,
            "obj/flow_epe_mm": MetricStat(float((obj_epe * valid_obj_mask.float() * 1000.0).sum().detach()), obj_count),
            "hand/flow_epe_3cm_mm": MetricStat(float((hand_epe * valid_hand_mask.float() * 1000.0).sum().detach()), hand_count),
            "hand/active_points": sample_mean(hand_mask.float().sum(dim=1)),
            "hand/active_ratio": sample_mean(hand_mask.float().mean(dim=1)),
            "data/hand_valid_points": batch["hand_valid_mask"].float().sum(dim=-1).mean(),
            "data/min_hand_object_distance_mm": batch["min_hand_object_distance_mm"].float().mean(),
            "interaction/object_points_with_hand_neighbor": sample_mean(prediction["interaction/has_interaction"].float().sum(dim=-1)),
            "interaction/mean_valid_neighbors": sample_mean(valid_edges.float().sum(dim=-1)),
            "interaction/attention_max": sample_mean(attention.max(dim=-1).values),
            "interaction/attention_entropy": sample_mean(-(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(dim=-1)),
            "slot/assignment_entropy": sample_mean(assignment_entropy),
            "slot/entropy": sample_mean(slot_entropy),
            "slot/effective_count": sample_mean(slot_entropy.exp()),
            "slot/max_mass": sample_mean(slot_usage.max(dim=-1).values),
            "sample/valid_ratio": sample_valid.float().mean(),
            "sample/valid_count": sample_valid.float().sum(),
            "sample/sampling_miss_ratio": ((batch.get("full_active_count", torch.zeros_like(sample_valid, dtype=torch.long)) > 0) & ~sample_valid).float().mean(),
            "sample/true_no_interaction_ratio": (batch.get("full_active_count", torch.ones_like(sample_valid, dtype=torch.long)) == 0).float().mean(),
            "sample/sampled_active_count": prediction.get("sampled_active_count", prediction["interaction/has_interaction"].float().sum(dim=1)).float().mean(),
            "object/contribution_usage": sample_mean(prediction.get("object_contribution_usage", torch.zeros_like(obj_prediction))),
            "hand/contribution_usage": sample_mean(prediction.get("hand_contribution_usage", torch.zeros_like(hand_prediction))),
        }
        metrics["_all_samples_invalid"] = bool(valid_count.detach().item() == 0)
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(obj_prediction.shape[0]))

    def select_step_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        keep = {
            key: value for key, value in metrics.items()
            if key in {"loss", "obj/loss", "hand/loss_3cm", "obj/flow_epe_mm", "hand/flow_epe_3cm_mm", "hand/active_ratio", "data/hand_valid_points", "data/min_hand_object_distance_mm", "interaction/attention_max", "slot/effective_count"}
        }
        return keep

    def evaluate_all(self) -> dict[str, float]:
        if bool(getattr(self.cfg.train, "skip_eval", False)):
            return {}
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

    def _handle_validation(self, metrics: dict[str, float], epoch: int) -> bool:
        if bool(getattr(self.cfg.train, "skip_eval", False)):
            return False
        return super()._handle_validation(metrics, epoch)
