from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.correspondence_ptv3.dataset import make_dataloaders
from src.utils.correspondence import soft_contact_label


class CorrespondencePTV3Runner(BaseRunner):
    """Training runner for the runtime-sampled Stage 3 representation."""

    def __init__(
        self,
        cfg: TaskConfig,
        mode: str = "train",
        checkpoint: str | None = None,
        device: str | None = None,
        build_data: bool = True,
    ) -> None:
        super().__init__(
            cfg=cfg,
            mode=mode,
            checkpoint=checkpoint,
            device=device,
            build_data=build_data,
        )

    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(data_cfg, meta_cfg=self.cfg.meta, seed=seed)

    def configure_data(
        self,
        metadata: dict[str, Any],
        train_dataset: Any | None = None,
    ) -> None:
        super().configure_data(metadata, train_dataset)
        for field in (
            "num_obj_pool",
            "num_obj_points",
            "num_hand_points",
            "k_cross",
            "num_fingers",
            "num_regions",
        ):
            if field in metadata and int(metadata[field]) > 0:
                setattr(self.cfg.meta, field, int(metadata[field]))

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(
            model_cfg,
            condition_shape=None,
            target_shape=None,
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Make runtime object sampling a deterministic function of the epoch."""
        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = getattr(getattr(self, "train_loader", None), "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        return super().train_epoch(epoch)

    def step(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> RunnerOutput:
        del mode
        preds = model(batch)
        losses, aux_metrics = self._compute_losses(preds, batch)
        total_loss = sum(losses.values())
        metrics = {
            key: float(value.detach().cpu())
            for key, value in {**losses, **aux_metrics}.items()
        }
        metrics["loss"] = float(total_loss.detach().cpu())
        return RunnerOutput(
            loss=total_loss,
            metrics=metrics,
            batch_size=int(batch["points"].shape[0]),
        )

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        meta = self.cfg.meta
        obj_valid = batch["runtime_obj_valid_mask"].float()
        input_edge_valid = batch["input_obj_to_hand_knn_valid_mask"].float()
        target_contact = batch["obj_contact_label"].float()

        contact_loss = self._masked_bce(
            preds["pred_obj_contact"],
            target_contact,
            obj_valid,
        )

        edge_contact_target = self._compute_dynamic_edge_labels(batch)
        edge_contact_loss = self._masked_bce(
            preds["pred_cross_contact"],
            edge_contact_target,
            input_edge_valid,
        )

        target_obj_cano, corr_valid, target_finger, target_region = (
            self._build_clean_correspondence_targets(batch)
        )
        cano_diff = F.smooth_l1_loss(
            preds["pred_obj_cano"],
            target_obj_cano,
            reduction="none",
        ).sum(dim=-1)
        cano_loss = (
            (cano_diff * corr_valid).sum()
            / corr_valid.sum().clamp(min=1.0)
        )

        zero = contact_loss.new_tensor(0.0)
        finger_loss = zero
        if "pred_obj_finger" in preds:
            finger_valid = corr_valid * (target_finger >= 0).float()
            if bool(finger_valid.any()):
                finger_map = F.cross_entropy(
                    preds["pred_obj_finger"].reshape(
                        -1,
                        preds["pred_obj_finger"].shape[-1],
                    ),
                    target_finger.reshape(-1).clamp(min=0),
                    reduction="none",
                ).reshape_as(target_finger)
                finger_loss = (
                    (finger_map * finger_valid).sum()
                    / finger_valid.sum().clamp(min=1.0)
                )

        region_loss = zero
        if "pred_obj_region" in preds:
            region_valid = corr_valid * (target_region >= 0).float()
            if bool(region_valid.any()):
                region_map = F.cross_entropy(
                    preds["pred_obj_region"].reshape(
                        -1,
                        preds["pred_obj_region"].shape[-1],
                    ),
                    target_region.reshape(-1).clamp(min=0),
                    reduction="none",
                ).reshape_as(target_region)
                region_loss = (
                    (region_map * region_valid).sum()
                    / region_valid.sum().clamp(min=1.0)
                )

        losses = {
            "obj_contact": float(meta.loss_contact_weight) * contact_loss,
            "obj_cano": float(meta.loss_cano_weight) * cano_loss,
            "obj_finger": float(meta.loss_finger_weight) * finger_loss,
            "obj_region": float(meta.loss_region_weight) * region_loss,
            "cross_edge_contact": (
                float(meta.loss_cross_edge_weight) * edge_contact_loss
            ),
        }
        aux_metrics = {
            "obj_contact_oracle_bce": self._soft_bce_entropy_floor(
                target_contact,
                obj_valid,
            ),
            "cross_edge_oracle_bce": self._soft_bce_entropy_floor(
                edge_contact_target,
                input_edge_valid,
            ),
            "num_valid_obj": obj_valid.sum(),
            "num_valid_cano": corr_valid.sum(),
        }
        aux_metrics["obj_contact_excess_bce"] = (
            contact_loss - aux_metrics["obj_contact_oracle_bce"]
        )
        aux_metrics["cross_edge_excess_bce"] = (
            edge_contact_loss - aux_metrics["cross_edge_oracle_bce"]
        )
        return losses, aux_metrics

    @staticmethod
    def _masked_bce(
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            pred,
            target,
            weight=mask,
            reduction="sum",
        ) / mask.sum().clamp(min=1.0)

    def _build_clean_correspondence_targets(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use clean KNN rank 0 as the one target for each sampled object point."""
        clean_knn = batch["gt_obj_to_hand_knn_idx"].long()
        nearest_idx = clean_knn[..., 0]
        safe_idx = nearest_idx.clamp(min=0)

        hand_cano = batch["hand_cano_points"].float()
        hand_finger = batch["hand_finger_id"].long()
        hand_region = batch["hand_region_id"].long()
        if hand_cano.dim() == 2:
            hand_cano = hand_cano.unsqueeze(0)
            hand_finger = hand_finger.unsqueeze(0)
            hand_region = hand_region.unsqueeze(0)

        target_cano = self._batch_gather(hand_cano, safe_idx)
        target_finger = self._batch_gather(hand_finger, safe_idx)
        target_region = self._batch_gather(hand_region, safe_idx)
        corr_valid = (
            batch["runtime_obj_valid_mask"].bool()
            & (nearest_idx >= 0)
            & (
                batch["obj_contact_label"]
                > float(getattr(self.cfg.meta, "corr_contact_label_min", 0.1))
            )
        ).float()
        return target_cano, corr_valid, target_finger, target_region

    @staticmethod
    def _batch_gather(
        values: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_idx = torch.arange(
            values.shape[0],
            device=values.device,
        ).view(-1, 1)
        return values[batch_idx, indices]

    def _compute_dynamic_edge_labels(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Build edge labels from noisy input KNN and clean GT geometry."""
        meta = self.cfg.meta
        points = batch.get("gt_points", batch["points"])
        num_obj = int(meta.num_obj_points)
        num_hand = int(meta.num_hand_points)
        obj_points = points[:, :num_obj]
        hand_points = points[:, num_obj : num_obj + num_hand]
        knn_idx = batch["input_obj_to_hand_knn_idx"].long()
        edge_valid = batch["input_obj_to_hand_knn_valid_mask"].bool()
        safe_idx = knn_idx.clamp(min=0)

        batch_idx = torch.arange(
            points.shape[0],
            device=points.device,
        ).view(-1, 1, 1)
        neighbor_hand = hand_points[batch_idx, safe_idx]
        distance = torch.norm(
            neighbor_hand - obj_points.unsqueeze(2),
            dim=-1,
        )
        labels = soft_contact_label(
            distance,
            d_pos=float(meta.d_pos),
            d_neg=float(meta.d_neg),
            gamma=float(meta.gamma),
        )
        return labels * edge_valid.float()

    @staticmethod
    def _soft_bce_entropy_floor(
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        target = target.clamp(min=eps, max=1.0 - eps)
        entropy = -(
            target * torch.log(target)
            + (1.0 - target) * torch.log(1.0 - target)
        )
        return (entropy * mask).sum() / mask.sum().clamp(min=1.0)

    def inference(
        self,
        model: torch.nn.Module,
        inputs: Any,
    ) -> dict[str, torch.Tensor]:
        return model(self.prepare_batch(inputs))
