from __future__ import annotations

from typing import Any

import torch

from src.base import BaseRunner, MetricStat, RunnerOutput, TaskConfig, set_config_default_if_not_explicit
from src.task.correspondence_ptv3_v2.dataset import make_dataloaders
from src.task.correspondence_ptv3_v2.losses import (
    binary_cross_entropy_with_logits_map,
    binary_entropy_floor_map,
    quality_focal_loss_map,
    reduce_loss_map,
    reduce_loss_map_per_object,
    target_strength_bin_mask,
    zero_predictor_bce_map,
    zero_predictor_mae_map,
    zero_predictor_qfl_map,
)


_TARGET_STRENGTH_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.25, "edge_y_0_025"),
    (0.25, 0.50, "edge_y_025_050"),
    (0.50, 0.75, "edge_y_050_075"),
    (0.75, 1.00, "edge_y_075_100"),
)


class CorrespondencePTV3V2Runner(BaseRunner):
    @classmethod
    def configure_overfit_mode(
        cls,
        cfg: TaskConfig,
        explicit_override_keys: set[str],
    ) -> None:
        super().configure_overfit_mode(cfg, explicit_override_keys)
        for key, value in (
            ("meta.augment", False),
            ("meta.apply_obj_perturb", False),
            ("meta.augment_rotation", False),
            ("meta.augment_translation", False),
            ("meta.obj_perturb_prob", 0.0),
            ("meta.val_augment", False),
            ("meta.ptv3_drop_path", 0.0),
            ("meta.ptv3_shuffle_orders", False),
        ):
            set_config_default_if_not_explicit(
                cfg,
                key=key,
                value=value,
                explicit_override_keys=explicit_override_keys,
            )

    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(
            data_cfg,
            meta_cfg=self.cfg.meta,
            seed=seed,
            distributed=self.distributed,
        )

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
        super().configure_data(metadata, train_dataset)
        # Validate coordinate_frame BEFORE we let metadata overwrite the cfg
        # value; the data is the source of truth and a mismatch must raise
        # rather than be silently coerced by the update loop below.
        if "coordinate_frame" not in metadata:
            raise ValueError(
                "Stage 3 metadata is missing 'coordinate_frame'. Re-run Stage 3 "
                "with the current prepare_corr_static.py."
            )
        stage3_frame = metadata["coordinate_frame"]
        if stage3_frame != self.cfg.meta.coordinate_frame:
            raise ValueError(
                f"Stage 3 data was generated in coordinate_frame={stage3_frame!r} "
                f"but cfg.meta.coordinate_frame={self.cfg.meta.coordinate_frame!r}. "
                f"Re-run Stage 3 with --coordinate-frame {self.cfg.meta.coordinate_frame} "
                f"on the same Stage 2 root, or pass "
                f"--set meta.coordinate_frame={stage3_frame} to this training run."
            )
        for field in (
            "num_obj_pool",
            "num_obj_points",
            "num_hand_points",
            "num_supervision_edges",
            "contact_radius",
            "coordinate_frame",
        ):
            if field in metadata:
                setattr(self.cfg.meta, field, metadata[field])

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        dataset_epoch = 0 if bool(getattr(self.cfg.train, "overfit_mode", False)) else epoch
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(dataset_epoch)
        sampler = getattr(getattr(self, "train_loader", None), "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        return super().train_epoch(epoch)

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        del mode
        preds = model(batch)
        losses, aux_metrics = self._compute_losses(preds, batch)
        total_loss = sum(losses.values())
        metrics: dict[str, float | MetricStat] = {**losses, **aux_metrics}
        metrics["loss"] = total_loss
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(batch["points"].shape[0]))

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        meta = self.cfg.meta
        obj_valid_mask = batch["runtime_obj_valid_mask"].bool()
        supervision_edge_valid_mask = batch["supervision_edge_valid_mask"].bool()
        edge_valid_mask = supervision_edge_valid_mask & obj_valid_mask.unsqueeze(-1)

        edge_target = batch["edge_contact_target"].float()
        edge_logits = preds["pred_cross_contact_logits"]
        edge_prob = preds["pred_cross_contact_prob"]
        beta = float(meta.quality_focal_beta)

        edge_qfl_map = quality_focal_loss_map(edge_logits, edge_target, beta=beta)
        edge_bce_map = binary_cross_entropy_with_logits_map(edge_logits, edge_target)
        edge_mae_map = torch.abs(edge_prob - edge_target)

        cross_edge_qfl = reduce_loss_map_per_object(edge_qfl_map, edge_valid_mask, obj_valid_mask)
        cross_edge_bce = reduce_loss_map_per_object(edge_bce_map, edge_valid_mask, obj_valid_mask)
        cross_edge_mae = reduce_loss_map_per_object(edge_mae_map, edge_valid_mask, obj_valid_mask)

        # Hand contact head: per-hand-point soft contact score supervised by
        # the frame-invariant min distance to the full 4096 object pool.
        # All 1538 hand points are valid (no padding/valid mask), so the
        # loss is averaged over B*1538 (= numel) for batch-size invariance:
        # duplicating a sample along the batch dim must not change the loss.
        hand_target = batch["hand_contact_target"].float()
        hand_logits = preds["pred_hand_contact_logits"]
        hand_prob = preds["pred_hand_contact_prob"]
        hand_qfl_map = quality_focal_loss_map(hand_logits, hand_target, beta=beta)
        hand_bce_map = binary_cross_entropy_with_logits_map(hand_logits, hand_target)
        hand_mae_map = torch.abs(hand_prob - hand_target)
        hand_contact_qfl = hand_qfl_map.mean()
        hand_contact_bce = hand_bce_map.mean()
        hand_contact_mae = hand_mae_map.mean()

        hand_nonzero_mask = hand_target > 0
        hand_nonzero_count = hand_nonzero_mask.sum()
        if bool(hand_nonzero_count > 0):
            hand_nonzero_mae = (hand_prob - hand_target).abs()[hand_nonzero_mask].sum() / hand_nonzero_count.float()
            hand_nonzero_pred_mean = hand_prob[hand_nonzero_mask].mean()
            hand_nonzero_target_mean = hand_target[hand_nonzero_mask].mean()
        else:
            zero_tensor = hand_target.sum() * 0.0
            hand_nonzero_mae = zero_tensor
            hand_nonzero_pred_mean = zero_tensor
            hand_nonzero_target_mean = zero_tensor

        oracle_bce_map = binary_entropy_floor_map(edge_target)
        cross_edge_oracle_bce = reduce_loss_map_per_object(
            oracle_bce_map, edge_valid_mask, obj_valid_mask
        )
        cross_edge_oracle_bce_global = reduce_loss_map(
            oracle_bce_map, edge_valid_mask
        )
        cross_edge_bce_global = reduce_loss_map(edge_bce_map, edge_valid_mask)
        cross_edge_excess_bce = (cross_edge_bce - cross_edge_oracle_bce).detach()
        cross_edge_excess_bce_global = (cross_edge_bce_global - cross_edge_oracle_bce_global).detach()

        sampled_nonzero_mask = (edge_target > 0) & edge_valid_mask
        sampled_nonzero_edge_count = sampled_nonzero_mask.sum()
        valid_edge_count = edge_valid_mask.sum()
        per_obj_has_nonzero = sampled_nonzero_mask.any(dim=-1) & obj_valid_mask
        num_valid_obj = obj_valid_mask.sum()
        num_obj_with_nonzero = per_obj_has_nonzero.sum()

        aux_metrics: dict[str, float | MetricStat] = {
            "cross_edge_qfl": cross_edge_qfl,
            "cross_edge_bce": cross_edge_bce,
            "cross_edge_mae": cross_edge_mae,
            "cross_edge_oracle_bce": cross_edge_oracle_bce,
            "cross_edge_oracle_bce_global": cross_edge_oracle_bce_global,
            "cross_edge_excess_bce": cross_edge_excess_bce,
            "cross_edge_excess_bce_global": cross_edge_excess_bce_global,
            "num_valid_obj": num_valid_obj.float(),
            "num_valid_edges": valid_edge_count.float(),
            "sampled_nonzero_edge_count": sampled_nonzero_edge_count.float(),
            "sampled_nonzero_edge_fraction": MetricStat(
                total=float(sampled_nonzero_edge_count.detach().cpu()),
                count=float(valid_edge_count.detach().cpu()),
            ),
            "object_nonzero_edge_coverage": MetricStat(
                total=float(num_obj_with_nonzero.detach().cpu()),
                count=float(num_valid_obj.detach().cpu()),
            ),
        }

        aux_metrics.update(self._compute_diagnostic_metrics(
            edge_target=edge_target,
            edge_prob=edge_prob,
            edge_valid_mask=edge_valid_mask,
            obj_valid_mask=obj_valid_mask,
            beta=beta,
        ))

        aux_metrics["hand_contact_qfl"] = hand_contact_qfl
        aux_metrics["hand_contact_bce"] = hand_contact_bce
        aux_metrics["hand_contact_mae"] = hand_contact_mae
        aux_metrics["hand_contact_nonzero_mae"] = hand_nonzero_mae
        aux_metrics["hand_contact_nonzero_pred_mean"] = hand_nonzero_pred_mean
        aux_metrics["hand_contact_nonzero_target_mean"] = hand_nonzero_target_mean
        aux_metrics["hand_contact_nonzero_count"] = MetricStat(
            total=float(hand_nonzero_count.detach().cpu()),
            count=float(hand_nonzero_count.detach().cpu()),
            expose_validity=True,
        )

        losses = {
            "cross_edge_contact": float(meta.loss_cross_edge_weight) * cross_edge_qfl,
            "hand_contact": float(meta.loss_hand_contact_weight) * hand_contact_qfl,
        }
        return losses, aux_metrics

    def _compute_diagnostic_metrics(
        self,
        *,
        edge_target: torch.Tensor,
        edge_prob: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        obj_valid_mask: torch.Tensor,
        beta: float,
    ) -> dict[str, float | MetricStat]:
        """Diagnostic metrics for the 144 nonzero / 787456 total imbalance.

        See ``docs/指导.md`` for the design rationale.
        """
        metrics: dict[str, float | MetricStat] = {}

        # ---- Nonzero edges ----
        nonzero_mask = (edge_target > 0) & edge_valid_mask
        nonzero_count = nonzero_mask.sum()
        if bool(nonzero_count > 0):
            metrics["cross_edge_nonzero_mae"] = (
                (edge_prob - edge_target).abs()[nonzero_mask].sum() / nonzero_count.float()
            )
            metrics["cross_edge_nonzero_bce"] = (
                -(
                    edge_target[nonzero_mask] * torch.log(edge_prob[nonzero_mask].clamp(min=1e-6))
                    + (1.0 - edge_target[nonzero_mask])
                    * torch.log((1.0 - edge_prob[nonzero_mask]).clamp(min=1e-6))
                ).mean()
            )
            metrics["cross_edge_nonzero_pred_mean"] = edge_prob[nonzero_mask].mean()
            metrics["cross_edge_nonzero_target_mean"] = edge_target[nonzero_mask].mean()
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["cross_edge_nonzero_mae"] = zero_tensor
            metrics["cross_edge_nonzero_bce"] = zero_tensor
            metrics["cross_edge_nonzero_pred_mean"] = zero_tensor
            metrics["cross_edge_nonzero_target_mean"] = zero_tensor
        metrics["cross_edge_nonzero_count"] = MetricStat(
            total=float(nonzero_count.detach().cpu()),
            count=float(nonzero_count.detach().cpu()),
            expose_validity=True,
        )

        # ---- Target strength bins (count + MAE) ----
        for lower, upper, name in _TARGET_STRENGTH_BINS:
            bin_mask = target_strength_bin_mask(
                edge_target, lower=lower, upper=upper, upper_inclusive=(upper < 1.0)
            ) & edge_valid_mask
            bin_count = bin_mask.sum()
            metrics[f"{name}_count"] = MetricStat(
                total=float(bin_count.detach().cpu()),
                count=float(bin_count.detach().cpu()),
                expose_validity=True,
            )
            if bool(bin_count > 0):
                metrics[f"{name}_mae"] = (
                    (edge_prob - edge_target).abs()[bin_mask].sum() / bin_count.float()
                )
            else:
                metrics[f"{name}_mae"] = edge_target.sum() * 0.0

        # ---- Zero edge prediction distribution ----
        zero_edge_mask = (edge_target == 0) & edge_valid_mask
        zero_edge_count = zero_edge_mask.sum()
        if bool(zero_edge_count > 0):
            zero_pred = edge_prob[zero_edge_mask].float()
            metrics["cross_edge_zero_pred_mean"] = zero_pred.mean()
            metrics["cross_edge_zero_pred_p95"] = torch.quantile(zero_pred, 0.95)
            metrics["cross_edge_zero_pred_p99"] = torch.quantile(zero_pred, 0.99)
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["cross_edge_zero_pred_mean"] = zero_tensor
            metrics["cross_edge_zero_pred_p95"] = zero_tensor
            metrics["cross_edge_zero_pred_p99"] = zero_tensor

        # ---- Zero predictor baseline ----
        zero_qfl_map = zero_predictor_qfl_map(edge_target, beta=beta)
        zero_bce_map = zero_predictor_bce_map(edge_target)
        zero_mae_map = zero_predictor_mae_map(edge_target)
        metrics["zero_baseline_qfl"] = reduce_loss_map_per_object(
            zero_qfl_map, edge_valid_mask, obj_valid_mask
        )
        metrics["zero_baseline_bce"] = reduce_loss_map_per_object(
            zero_bce_map, edge_valid_mask, obj_valid_mask
        )
        metrics["zero_baseline_mae"] = reduce_loss_map_per_object(
            zero_mae_map, edge_valid_mask, obj_valid_mask
        )
        return metrics

    def inference(self, model: torch.nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
        return model(self.prepare_batch(inputs))
