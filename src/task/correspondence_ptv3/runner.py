from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.correspondence_ptv3.dataset import make_dataloaders
from src.utils.correspondence import (
    contact_prob_to_bins,
    soft_contact_label,
)


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
            "k_ctx",
            "k_logit",
            "k_logit_hard_neg",
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
        num_bins = int(getattr(meta, "num_contact_bins", 10))
        obj_valid_bool = batch["runtime_obj_valid_mask"].bool()
        obj_valid = obj_valid_bool.float()
        input_edge_valid = batch.get("input_obj_to_hand_logit_valid_mask")
        if input_edge_valid is None:
            input_edge_valid = batch["input_obj_to_hand_knn_valid_mask"]
        input_edge_valid = input_edge_valid.bool()
        input_edge_weight = batch.get("input_obj_to_hand_logit_loss_weight")
        if input_edge_weight is None:
            input_edge_weight = input_edge_valid.float()
        else:
            input_edge_weight = input_edge_weight.float() * input_edge_valid.float()
        target_contact_prob = batch["obj_contact_label"].float()
        target_contact_bin = contact_prob_to_bins(
            target_contact_prob,
            num_bins=num_bins,
        )
        bin_weights = self._get_contact_bin_weights(
            preds["pred_obj_contact_bin"].device,
            preds["pred_obj_contact_bin"].dtype,
        )
        contact_loss = self._masked_cross_entropy(
            preds["pred_obj_contact_bin"],
            target_contact_bin,
            obj_valid,
            class_weight=bin_weights,
        )

        edge_contact_prob = self._compute_dynamic_edge_labels(batch)
        edge_contact_bin = contact_prob_to_bins(
            edge_contact_prob,
            num_bins=num_bins,
        )
        edge_contact_loss = self._masked_cross_entropy(
            preds["pred_cross_contact_bin"],
            edge_contact_bin,
            input_edge_weight,
            class_weight=bin_weights,
        )
        cross_edge_rankk_loss, cross_edge_rankk_pairs = self._compute_cross_edge_rankk_loss(
            preds["pred_cross_contact_prob"],
            edge_contact_prob,
            input_edge_valid & obj_valid_bool.unsqueeze(-1),
        )

        need_clean_targets = any(
            key in preds
            for key in ("pred_obj_cano", "pred_obj_finger", "pred_obj_region")
        )
        zero = contact_loss.new_tensor(0.0)
        corr_valid = zero.unsqueeze(0).expand_as(target_contact_prob)
        target_obj_cano: torch.Tensor | None = None
        target_finger: torch.Tensor | None = None
        target_region: torch.Tensor | None = None
        if need_clean_targets:
            (
                target_obj_cano,
                corr_valid,
                target_finger,
                target_region,
            ) = self._build_clean_correspondence_targets(batch)

        cano_loss = zero
        if "pred_obj_cano" in preds:
            assert target_obj_cano is not None
            cano_diff = F.smooth_l1_loss(
                preds["pred_obj_cano"],
                target_obj_cano,
                reduction="none",
            ).sum(dim=-1)
            cano_loss = (
                (cano_diff * corr_valid).sum()
                / corr_valid.sum().clamp(min=1.0)
            )

        finger_loss = zero
        if "pred_obj_finger" in preds:
            assert target_finger is not None
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
            assert target_region is not None
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
            "obj_contact": float(
                getattr(meta, "loss_obj_contact_weight", getattr(meta, "loss_contact_weight", 1.0))
            ) * contact_loss,
            "obj_cano": float(meta.loss_cano_weight) * cano_loss,
            "obj_finger": float(meta.loss_finger_weight) * finger_loss,
            "obj_region": float(meta.loss_region_weight) * region_loss,
            "cross_edge_contact": (
                float(meta.loss_cross_edge_weight) * edge_contact_loss
            ),
            "cross_edge_rankk": (
                float(getattr(meta, "loss_cross_edge_rankk_weight", 0.0))
                * cross_edge_rankk_loss
            ),
        }
        pr_label_threshold = float(getattr(meta, "pr_label_threshold", 0.5))
        with torch.no_grad():
            obj_contact_auprc = self._batched_binary_auprc(
                preds["pred_obj_contact_prob"].detach(),
                (target_contact_prob > pr_label_threshold),
                obj_valid_bool,
            )
            cross_edge_auprc = self._batched_binary_auprc(
                preds["pred_cross_contact_prob"].detach(),
                (edge_contact_prob > pr_label_threshold),
                input_edge_valid & obj_valid_bool.unsqueeze(-1),
            )
            cross_edge_rank1 = self._batched_cross_edge_rank_at_k(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob,
                input_edge_valid & obj_valid_bool.unsqueeze(-1),
                k=1,
            )
            cross_edge_rank4 = self._batched_cross_edge_rank_at_k(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob,
                input_edge_valid & obj_valid_bool.unsqueeze(-1),
                k=4,
            )
            cross_edge_rank8 = self._batched_cross_edge_rank_at_k(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob,
                input_edge_valid & obj_valid_bool.unsqueeze(-1),
                k=8,
            )
            obj_contact_decoded_bce = self._masked_bce_from_prob(
                preds["pred_obj_contact_prob"].detach(),
                target_contact_prob,
                obj_valid,
            )
            cross_edge_decoded_bce = self._masked_bce_from_prob(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob,
                input_edge_weight,
            )
        aux_metrics = {
            "obj_contact_oracle_bce": self._soft_bce_entropy_floor(
                target_contact_prob,
                obj_valid,
            ),
            "cross_edge_oracle_bce": self._soft_bce_entropy_floor(
                edge_contact_prob,
                input_edge_weight,
            ),
            "num_valid_obj": obj_valid.sum(),
            "num_valid_cano": corr_valid.sum(),
            "obj_contact_decoded_bce": obj_contact_decoded_bce,
            "cross_edge_decoded_bce": cross_edge_decoded_bce,
            "obj_contact_auprc": obj_contact_auprc,
            "cross_edge_auprc": cross_edge_auprc,
            "cross_edge_rank1": cross_edge_rank1,
            "cross_edge_rank4": cross_edge_rank4,
            "cross_edge_rank8": cross_edge_rank8,
            "cross_edge_rankk_loss_raw": cross_edge_rankk_loss.detach(),
            "cross_edge_rankk_pairs": cross_edge_rankk_pairs.detach(),
        }
        aux_metrics["obj_contact_excess_bce"] = (
            obj_contact_decoded_bce - aux_metrics["obj_contact_oracle_bce"]
        )
        aux_metrics["cross_edge_excess_bce"] = (
            cross_edge_decoded_bce - aux_metrics["cross_edge_oracle_bce"]
        )
        return losses, aux_metrics

    @staticmethod
    def _masked_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        class_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_classes = int(logits.shape[-1])
        loss_map = F.cross_entropy(
            logits.reshape(-1, num_classes),
            target.reshape(-1),
            reduction="none",
            weight=class_weight,
        )
        loss_map = loss_map.view_as(target).float()
        return (loss_map * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

    @staticmethod
    def _masked_bce_from_prob(
        pred_prob: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        pred_prob = pred_prob.clamp(min=eps, max=1.0 - eps)
        return F.binary_cross_entropy(
            pred_prob,
            target,
            weight=mask.float(),
            reduction="sum",
        ) / mask.float().sum().clamp(min=1.0)

    def _get_contact_bin_weights(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        raw = getattr(self.cfg.meta, "contact_bin_weights", None)
        if raw is None:
            return None
        weight = torch.as_tensor(raw, device=device, dtype=dtype)
        if weight.numel() != int(getattr(self.cfg.meta, "num_contact_bins", 10)):
            raise ValueError(
                "contact_bin_weights must match num_contact_bins, got "
                f"{weight.numel()} vs {int(getattr(self.cfg.meta, 'num_contact_bins', 10))}."
            )
        return weight

    def _compute_cross_edge_rankk_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        meta = self.cfg.meta
        k = max(0, int(getattr(meta, "rankk_loss_k", 4)))
        if k <= 0:
            zero = pred.new_tensor(0.0)
            return zero, zero
        label_gap_min = float(getattr(meta, "rankk_label_gap", 0.15))
        margin = float(getattr(meta, "rankk_margin", 0.2))
        num_slots = pred.shape[-1]
        topk = min(k, num_slots)
        if topk <= 0:
            zero = pred.new_tensor(0.0)
            return zero, zero

        masked_target = target.masked_fill(~valid_mask, float("-inf"))
        anchor_idx = torch.topk(masked_target, k=topk, dim=-1).indices
        valid_count = valid_mask.sum(dim=-1)
        rank_range = torch.arange(topk, device=pred.device).view(1, 1, topk)
        anchor_valid = rank_range < valid_count.unsqueeze(-1)

        anchor_target = target.gather(dim=-1, index=anchor_idx)
        anchor_pred = pred.gather(dim=-1, index=anchor_idx)
        label_gap = anchor_target.unsqueeze(-1) - target.unsqueeze(-2)
        score_gap = anchor_pred.unsqueeze(-1) - pred.unsqueeze(-2)

        slot_idx = torch.arange(num_slots, device=pred.device).view(1, 1, 1, num_slots)
        pair_valid = (
            anchor_valid.unsqueeze(-1)
            & valid_mask.unsqueeze(-2)
            & (slot_idx != anchor_idx.unsqueeze(-1))
            & (label_gap >= label_gap_min)
        )
        pair_weight = label_gap.clamp(min=0.0)
        pair_loss = F.relu(margin - score_gap) * pair_weight
        pair_valid_f = pair_valid.float()
        pair_count = pair_valid_f.sum()
        loss = (pair_loss * pair_valid_f).sum() / pair_count.clamp(min=1.0)
        return loss, pair_count

    @staticmethod
    def _binary_auprc(
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if scores.numel() == 0:
            return scores.new_tensor(0.0)
        labels = labels.bool()
        num_pos = int(labels.sum().item())
        if num_pos <= 0:
            return scores.new_tensor(0.0)
        order = torch.argsort(scores, descending=True)
        sorted_labels = labels[order].float()
        cumsum_pos = torch.cumsum(sorted_labels, dim=0)
        rank = torch.arange(
            1,
            sorted_labels.numel() + 1,
            device=scores.device,
            dtype=scores.dtype,
        )
        precision = cumsum_pos / rank
        return (precision * sorted_labels).sum() / float(num_pos)

    def _batched_binary_auprc(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_sample: list[torch.Tensor] = []
        for batch_idx in range(scores.shape[0]):
            sample_valid = valid_mask[batch_idx].reshape(-1)
            if not bool(sample_valid.any()):
                continue
            sample_scores = scores[batch_idx].reshape(-1)[sample_valid]
            sample_labels = labels[batch_idx].reshape(-1)[sample_valid]
            per_sample.append(self._binary_auprc(sample_scores, sample_labels))
        if not per_sample:
            return scores.new_tensor(0.0)
        return torch.stack(per_sample).mean()

    @staticmethod
    def _batched_cross_edge_rank_at_k(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        k: int,
    ) -> torch.Tensor:
        per_sample: list[torch.Tensor] = []
        for batch_idx in range(pred.shape[0]):
            per_object_hits: list[torch.Tensor] = []
            sample_valid_mask = valid_mask[batch_idx]
            sample_pred = pred[batch_idx]
            sample_target = target[batch_idx]
            valid_objects = torch.nonzero(
                sample_valid_mask.any(dim=-1),
                as_tuple=False,
            ).squeeze(-1)
            for obj_idx in valid_objects.tolist():
                edge_valid = sample_valid_mask[obj_idx]
                valid_idx = torch.nonzero(edge_valid, as_tuple=False).squeeze(-1)
                if valid_idx.numel() == 0:
                    continue
                kk = min(int(k), int(valid_idx.numel()))
                gt_scores = sample_target[obj_idx, valid_idx]
                pred_scores = sample_pred[obj_idx, valid_idx]
                gt_topk = torch.topk(gt_scores, k=kk, dim=-1).indices
                pred_topk = torch.topk(pred_scores, k=kk, dim=-1).indices
                hit = (
                    pred_topk.unsqueeze(-1) == gt_topk.unsqueeze(-2)
                ).any().float()
                per_object_hits.append(hit)
            if per_object_hits:
                per_sample.append(torch.stack(per_object_hits).mean())
        if not per_sample:
            return pred.new_tensor(0.0)
        return torch.stack(per_sample).mean()

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
        knn_idx = batch.get("input_obj_to_hand_logit_idx")
        if knn_idx is None:
            knn_idx = batch["input_obj_to_hand_knn_idx"]
        edge_valid = batch.get("input_obj_to_hand_logit_valid_mask")
        if edge_valid is None:
            edge_valid = batch["input_obj_to_hand_knn_valid_mask"]
        knn_idx = knn_idx.long()
        edge_valid = edge_valid.bool()
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
