from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.correspondence_ptv3.dataset import make_dataloaders
from src.utils.correspondence import soft_contact_label


class CorrespondencePTV3Runner(BaseRunner):
    """Runner for the PTv3-style static hand-object correspondence model."""

    def __init__(
        self,
        cfg: TaskConfig,
        mode: str = "train",
        checkpoint: str | None = None,
        device: str | None = None,
        build_data: bool = True,
    ) -> None:
        super().__init__(cfg=cfg, mode=mode, checkpoint=checkpoint, device=device, build_data=build_data)

    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(data_cfg, meta_cfg=self.cfg.meta, seed=seed)

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        super().configure_data(metadata, train_dataset)
        meta = self.cfg.meta
        for field in ("num_obj_points", "num_hand_points", "k_cross"):
            if field in metadata:
                setattr(meta, field, int(metadata[field]))
        for field in ("num_fingers", "num_regions"):
            if field in metadata and int(metadata[field]) > 0:
                setattr(meta, field, int(metadata[field]))

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        preds = model(batch)
        losses, aux_metrics = self._compute_losses(preds, batch)
        total_loss = sum(losses.values())
        metrics = {key: float(val.detach().cpu()) for key, val in losses.items()}
        metrics.update({key: float(val.detach().cpu()) for key, val in aux_metrics.items()})
        metrics["loss"] = float(total_loss.detach().cpu())
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(batch["points"].shape[0]))

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        meta = self.cfg.meta
        obj_label_valid_mask = batch["obj_label_valid_mask"].float()
        corr_valid_mask = self._build_corr_valid_mask(batch).float()

        pred_contact = preds["pred_obj_contact"]
        target_contact = batch["obj_contact_label"].float()
        contact_loss = F.binary_cross_entropy_with_logits(
            pred_contact,
            target_contact,
            weight=obj_label_valid_mask,
            reduction="sum",
        ) / obj_label_valid_mask.sum().clamp(min=1)

        cross_valid_mask = batch["obj_to_hand_knn_valid_mask"].float()
        edge_contact_labels = self._compute_dynamic_edge_labels(batch)
        cross_loss = F.binary_cross_entropy_with_logits(
            preds["pred_cross_contact"],
            edge_contact_labels,
            weight=cross_valid_mask,
            reduction="sum",
        ) / cross_valid_mask.sum().clamp(min=1)

        pred_cross_cano = preds["pred_cross_cano"]
        target_cross_cano = batch["obj_to_hand_cano_points"].float().unsqueeze(2).expand_as(pred_cross_cano)
        edge_cano_valid_mask = cross_valid_mask * (
            edge_contact_labels > float(getattr(meta, "corr_contact_label_min", 0.1))
        ).float()
        cross_cano_diff = F.smooth_l1_loss(pred_cross_cano, target_cross_cano, reduction="none").sum(dim=-1)
        cano_loss = (cross_cano_diff * edge_cano_valid_mask).sum() / edge_cano_valid_mask.sum().clamp(min=1)

        obj_contact_oracle = self._soft_bce_entropy_floor(target_contact, obj_label_valid_mask)
        cross_edge_oracle = self._soft_bce_entropy_floor(edge_contact_labels, cross_valid_mask)

        zero = pred_contact.new_tensor(0.0)
        finger_loss = zero
        region_loss = zero

        if "pred_obj_to_hand_finger" in preds:
            pred_finger = preds["pred_obj_to_hand_finger"]
            target_finger = batch["obj_to_hand_finger_id"].long()
            finger_valid = corr_valid_mask * (target_finger >= 0).float()
            if finger_valid.sum() > 0:
                finger_loss_map = F.cross_entropy(
                    pred_finger.reshape(-1, pred_finger.shape[-1]),
                    target_finger.reshape(-1).clamp(min=0),
                    reduction="none",
                ).reshape(target_finger.shape)
                finger_loss = (finger_loss_map * finger_valid).sum() / finger_valid.sum().clamp(min=1)

        if "pred_obj_to_hand_region" in preds:
            pred_region = preds["pred_obj_to_hand_region"]
            target_region = batch["obj_to_hand_region_id"].long()
            region_valid = corr_valid_mask * (target_region >= 0).float()
            if region_valid.sum() > 0:
                region_loss_map = F.cross_entropy(
                    pred_region.reshape(-1, pred_region.shape[-1]),
                    target_region.reshape(-1).clamp(min=0),
                    reduction="none",
                ).reshape(target_region.shape)
                region_loss = (region_loss_map * region_valid).sum() / region_valid.sum().clamp(min=1)

        losses = {
            "obj_contact": float(meta.loss_contact_weight) * contact_loss,
            "cross_cano": float(meta.loss_cano_weight) * cano_loss,
            "obj_finger": float(meta.loss_finger_weight) * finger_loss,
            "obj_region": float(meta.loss_region_weight) * region_loss,
            "cross_edge_contact": float(meta.loss_cross_edge_weight) * cross_loss,
        }
        aux_metrics = {
            "obj_contact_oracle_bce": obj_contact_oracle,
            "obj_contact_excess_bce": contact_loss - obj_contact_oracle,
            "cross_edge_oracle_bce": cross_edge_oracle,
            "cross_edge_excess_bce": cross_loss - cross_edge_oracle,
        }
        return losses, aux_metrics

    def _build_corr_valid_mask(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        meta = self.cfg.meta
        corr_valid_mask = batch["obj_label_valid_mask"].bool()
        if "obj_to_hand_nn_id" in batch:
            corr_valid_mask = corr_valid_mask & (batch["obj_to_hand_nn_id"] >= 0)
        corr_valid_mask = corr_valid_mask & (
            batch["obj_contact_label"] > float(getattr(meta, "corr_contact_label_min", 0.1))
        )
        return corr_valid_mask

    def _compute_dynamic_edge_labels(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        meta = self.cfg.meta
        points = batch.get("gt_points", batch["points"])
        num_obj_points = int(meta.num_obj_points)
        num_hand_points = int(meta.num_hand_points)

        obj_points = points[:, :num_obj_points]
        hand_points = points[:, num_obj_points : num_obj_points + num_hand_points]
        obj_to_hand_knn_idx = batch["obj_to_hand_knn_idx"].long()

        batch_size, _, k_cross = obj_to_hand_knn_idx.shape
        labels = torch.zeros(batch_size, num_obj_points, k_cross, device=points.device, dtype=points.dtype)

        for batch_idx in range(batch_size):
            safe_idx = obj_to_hand_knn_idx[batch_idx].clamp(min=0)
            neighbor_hand = hand_points[batch_idx, safe_idx]
            delta = neighbor_hand - obj_points[batch_idx].unsqueeze(1)
            dist = torch.norm(delta, dim=-1)
            labels[batch_idx] = soft_contact_label(
                dist,
                d_pos=float(meta.d_pos),
                d_neg=float(meta.d_neg),
                gamma=float(meta.gamma),
            )

        return labels

    def _soft_bce_entropy_floor(self, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        target = target.clamp(min=eps, max=1.0 - eps)
        entropy = -(target * torch.log(target) + (1.0 - target) * torch.log(1.0 - target))
        return (entropy * mask).sum() / mask.sum().clamp(min=1.0)

    def inference(self, model: torch.nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
        inputs = self.prepare_batch(inputs)
        return model(inputs)
