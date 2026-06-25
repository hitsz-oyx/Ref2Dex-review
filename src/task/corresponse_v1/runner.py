from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import (
    BaseRunner,
    RunnerOutput,
    TaskConfig,
)
from src.task.corresponse_v1.dataset import make_dataloaders
from src.utils.correspondence import soft_contact_label


class CorrResponseRunner(BaseRunner):
    """Runner for Static Hand-Object Correspondence training."""

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
        for field in ("num_obj_points", "num_hand_points", "k_obj_local", "k_hand_local", "k_cross"):
            if field in metadata:
                setattr(meta, field, int(metadata[field]))
        for field in ("num_fingers", "num_regions"):
            if field in metadata and int(metadata[field]) > 0:
                setattr(meta, field, int(metadata[field]))

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(
            model_cfg,
            condition_shape=None,
            target_shape=None,
        )

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        preds = model(batch)

        # Compute losses
        losses = self._compute_losses(preds, batch)

        total_loss = sum(losses.values())
        metrics = {key: float(val.detach().cpu()) for key, val in losses.items()}
        metrics["loss"] = float(total_loss.detach().cpu())

        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(batch["points"].shape[0]))

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute all losses for the correspondence model.

        Losses:
            1. obj_contact_loss: BCE on object contact probability
            2. obj_cano_loss: SmoothL1 on object-to-hand canonical correspondence
            3. obj_finger_loss: CrossEntropy on object-to-hand finger classification
            4. obj_region_loss: CrossEntropy on object-to-hand region classification
            5. cross_edge_contact_loss: BCE on cross edge contact probability
        """
        meta = self.cfg.meta
        obj_label_valid_mask = batch["obj_label_valid_mask"].float()  # [B, No]
        obj_corr_valid_mask = batch.get("obj_corr_valid_mask")
        if obj_corr_valid_mask is None:
            obj_corr_valid_mask = batch["obj_label_valid_mask"].bool()
            if "obj_to_hand_nn_id" in batch:
                obj_corr_valid_mask = obj_corr_valid_mask & (batch["obj_to_hand_nn_id"] >= 0)
            obj_corr_valid_mask = obj_corr_valid_mask & (
                batch["obj_contact_label"] > float(getattr(meta, "corr_contact_label_min", 0.0))
            )
        obj_corr_valid_mask = obj_corr_valid_mask.float()

        # ---- 1. Object contact loss ----
        pred_contact = preds["pred_obj_contact"]  # [B, No]
        target_contact = batch["obj_contact_label"].float()  # [B, No]
        contact_loss = F.binary_cross_entropy_with_logits(
            pred_contact, target_contact, weight=obj_label_valid_mask, reduction="sum"
        ) / obj_label_valid_mask.sum().clamp(min=1)

        # ---- 2. Object-to-hand canonical correspondence loss ----
        pred_cano = preds["pred_obj_to_hand_cano"]  # [B, No, 3]
        target_cano = batch["obj_to_hand_cano_points"].float()  # [B, No, 3]
        cano_diff = F.smooth_l1_loss(pred_cano, target_cano, reduction="none").sum(dim=-1)  # [B, No]
        cano_loss = (cano_diff * obj_corr_valid_mask).sum() / obj_corr_valid_mask.sum().clamp(min=1)

        # ---- 3. Object-to-hand finger classification loss ----
        pred_finger = preds["pred_obj_to_hand_finger"]  # [B, No, num_fingers]
        target_finger = batch["obj_to_hand_finger_id"].long()  # [B, No]
        # Only compute on valid object points with valid finger labels (>= 0)
        finger_valid = obj_corr_valid_mask * (target_finger >= 0).float()
        if finger_valid.sum() > 0:
            finger_loss = F.cross_entropy(
                pred_finger.reshape(-1, pred_finger.shape[-1]),
                target_finger.reshape(-1).clamp(min=0),
                reduction="none",
            ).reshape(target_finger.shape)
            finger_loss = (finger_loss * finger_valid).sum() / finger_valid.sum().clamp(min=1)
        else:
            finger_loss = torch.tensor(0.0, device=pred_finger.device)

        # ---- 4. Object-to-hand region classification loss ----
        pred_region = preds["pred_obj_to_hand_region"]  # [B, No, num_regions]
        target_region = batch["obj_to_hand_region_id"].long()  # [B, No]
        region_valid = obj_corr_valid_mask * (target_region >= 0).float()
        if region_valid.sum() > 0:
            region_loss = F.cross_entropy(
                pred_region.reshape(-1, pred_region.shape[-1]),
                target_region.reshape(-1).clamp(min=0),
                reduction="none",
            ).reshape(target_region.shape)
            region_loss = (region_loss * region_valid).sum() / region_valid.sum().clamp(min=1)
        else:
            region_loss = torch.tensor(0.0, device=pred_region.device)

        # ---- 5. Cross edge contact loss ----
        pred_cross = preds["pred_cross_contact"]  # [B, Nh, K_cross]
        cross_valid_mask = batch["hand_to_obj_knn_valid_mask"].float()  # [B, Nh, K_cross]

        # Dynamically generate edge labels from distances
        # We need to compute distances for each cross edge
        edge_contact_labels = self._compute_dynamic_edge_labels(batch)

        cross_loss = F.binary_cross_entropy_with_logits(
            pred_cross, edge_contact_labels, weight=cross_valid_mask, reduction="sum"
        ) / cross_valid_mask.sum().clamp(min=1)

        weight_contact = float(meta.loss_contact_weight)
        weight_cano = float(meta.loss_cano_weight)
        weight_finger = float(meta.loss_finger_weight)
        weight_region = float(meta.loss_region_weight)
        weight_cross = float(meta.loss_cross_edge_weight)

        return {
            "obj_contact": weight_contact * contact_loss,
            "obj_cano": weight_cano * cano_loss,
            "obj_finger": weight_finger * finger_loss,
            "obj_region": weight_region * region_loss,
            "cross_edge_contact": weight_cross * cross_loss,
        }

    def _compute_dynamic_edge_labels(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute dynamic edge contact labels from cross edge distances.

        Uses the soft_contact_label function with meta parameters.
        """
        meta = self.cfg.meta
        points = batch["points"]
        No = int(meta.num_obj_points)
        Nh = int(meta.num_hand_points)
        B = points.shape[0]

        obj_points = points[:, :No]  # [B, No, 3]
        hand_points = points[:, No:]  # [B, Nh, 3]

        hand_to_obj_knn_idx = batch["hand_to_obj_knn_idx"].long()  # [B, Nh, K]

        K = hand_to_obj_knn_idx.shape[2]
        labels = torch.zeros(B, Nh, K, device=points.device)

        for b in range(B):
            # Gather neighbor object points
            safe_idx = hand_to_obj_knn_idx[b].clamp(min=0)  # [Nh, K]
            neighbor_obj = obj_points[b, safe_idx]  # [Nh, K, 3]
            delta = hand_points[b].unsqueeze(1) - neighbor_obj  # [Nh, K, 3]
            dist = torch.norm(delta, dim=-1)  # [Nh, K]

            labels[b] = soft_contact_label(
                dist,
                d_pos=float(meta.d_pos),
                d_neg=float(meta.d_neg),
                gamma=float(meta.gamma),
            )

        return labels

    def inference(self, model: torch.nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
        inputs = self.prepare_batch(inputs)
        return model(inputs)
