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
            ("meta.apply_obj_perturb", False),
            ("meta.obj_perturb_prob", 0.0),
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
        beta = float(meta.quality_focal_beta)

        # ---- Random-edge stream (L_r) ----
        random_edge_valid_mask = batch["random_edge_valid_mask"].bool() & obj_valid_mask.unsqueeze(-1)
        random_target = batch["random_edge_contact_target"].float()
        random_logits = preds["pred_cross_random_logits"]
        random_prob = preds["pred_cross_random_prob"]

        random_qfl_map = quality_focal_loss_map(random_logits, random_target, beta=beta)
        random_bce_map = binary_cross_entropy_with_logits_map(random_logits, random_target)
        random_mae_map = torch.abs(random_prob - random_target)

        cross_edge_random_qfl = reduce_loss_map_per_object(
            random_qfl_map, random_edge_valid_mask, obj_valid_mask
        )
        cross_edge_random_bce = reduce_loss_map_per_object(
            random_bce_map, random_edge_valid_mask, obj_valid_mask
        )
        cross_edge_random_mae = reduce_loss_map_per_object(
            random_mae_map, random_edge_valid_mask, obj_valid_mask
        )

        # ---- Contact-aware auxiliary stream (L_c) ----
        contact_edge_valid_mask = batch["contact_edge_valid_mask"].bool() & obj_valid_mask.unsqueeze(-1)
        contact_edge_target = batch["contact_edge_contact_target"].float()
        contact_logits = preds["pred_cross_contact_aux_logits"]
        contact_prob = preds["pred_cross_contact_aux_prob"]

        contact_binary_target = (contact_edge_target > 0).float()
        contact_qfl_map = quality_focal_loss_map(contact_logits, contact_edge_target, beta=beta)
        contact_soft_bce_map = binary_cross_entropy_with_logits_map(contact_logits, contact_edge_target)
        contact_binary_bce_map = binary_cross_entropy_with_logits_map(contact_logits, contact_binary_target)
        contact_mae_map = torch.abs(contact_prob - contact_edge_target)

        cross_edge_contact_aux_qfl = reduce_loss_map_per_object(
            contact_qfl_map, contact_edge_valid_mask, obj_valid_mask
        )
        cross_edge_contact_aux_soft_bce = reduce_loss_map_per_object(
            contact_soft_bce_map, contact_edge_valid_mask, obj_valid_mask
        )
        cross_edge_contact_aux_bce = reduce_loss_map_per_object(
            contact_binary_bce_map, contact_edge_valid_mask, obj_valid_mask
        )
        cross_edge_contact_aux_mae = reduce_loss_map_per_object(
            contact_mae_map, contact_edge_valid_mask, obj_valid_mask
        )

        # ---- Hand contact is GT-only in v2.1 ----
        # We intentionally do not build or supervise a hand prediction head
        # in this ablation. Only the GT distribution is logged.
        hand_target = batch["hand_contact_target"].float()

        # ---- Random-stream diagnostic (must NOT be polluted by aux edges) ----
        random_oracle_bce_map = binary_entropy_floor_map(random_target)
        cross_edge_random_oracle_bce = reduce_loss_map_per_object(
            random_oracle_bce_map, random_edge_valid_mask, obj_valid_mask
        )
        cross_edge_random_oracle_bce_global = reduce_loss_map(
            random_oracle_bce_map, random_edge_valid_mask
        )
        cross_edge_random_bce_global = reduce_loss_map(
            random_bce_map, random_edge_valid_mask
        )
        cross_edge_random_excess_bce = (cross_edge_random_bce - cross_edge_random_oracle_bce).detach()
        cross_edge_random_excess_bce_global = (
            cross_edge_random_bce_global - cross_edge_random_oracle_bce_global
        ).detach()

        random_sampled_nonzero_mask = (random_target > 0) & random_edge_valid_mask
        random_sampled_nonzero_edge_count = random_sampled_nonzero_mask.sum()
        random_valid_edge_count = random_edge_valid_mask.sum()
        random_per_obj_has_nonzero = random_sampled_nonzero_mask.any(dim=-1) & obj_valid_mask
        random_num_valid_obj = obj_valid_mask.sum()
        random_num_obj_with_nonzero = random_per_obj_has_nonzero.sum()

        aux_metrics: dict[str, float | MetricStat] = {
            # Loss-aligned random stream metrics (names match the loss key).
            "cross_edge_random_qfl": cross_edge_random_qfl,
            "cross_edge_random_bce": cross_edge_random_bce,
            "cross_edge_random_mae": cross_edge_random_mae,
            "cross_edge_random_oracle_bce": cross_edge_random_oracle_bce,
            "cross_edge_random_oracle_bce_global": cross_edge_random_oracle_bce_global,
            "cross_edge_random_excess_bce": cross_edge_random_excess_bce,
            "cross_edge_random_excess_bce_global": cross_edge_random_excess_bce_global,
            "random_num_valid_obj": random_num_valid_obj.float(),
            "random_num_valid_edges": random_valid_edge_count.float(),
            "random_sampled_nonzero_edge_count": random_sampled_nonzero_edge_count.float(),
            "random_sampled_nonzero_edge_fraction": MetricStat(
                total=float(random_sampled_nonzero_edge_count.detach().cpu()),
                count=float(random_valid_edge_count.detach().cpu()),
            ),
            "random_object_nonzero_edge_coverage": MetricStat(
                total=float(random_num_obj_with_nonzero.detach().cpu()),
                count=float(random_num_valid_obj.detach().cpu()),
            ),
            # Contact auxiliary stream metrics (separate diagnostic).
            "contact_aux_qfl": cross_edge_contact_aux_qfl,
            "contact_aux_bce": cross_edge_contact_aux_bce,
            "contact_aux_soft_bce": cross_edge_contact_aux_soft_bce,
            "contact_aux_mae": cross_edge_contact_aux_mae,
        }

        aux_metrics.update(self._compute_random_diagnostic_metrics(
            edge_target=random_target,
            edge_prob=random_prob,
            edge_valid_mask=random_edge_valid_mask,
            obj_valid_mask=obj_valid_mask,
            beta=beta,
        ))
        aux_metrics.update(self._compute_contact_aux_diagnostic_metrics(
            edge_target=contact_edge_target,
            edge_prob=contact_prob,
            edge_valid_mask=contact_edge_valid_mask,
            obj_valid_mask=obj_valid_mask,
        ))
        aux_metrics.update(self._compute_hand_contact_gt_metrics(hand_target=hand_target))

        losses = {
            "cross_edge_random": float(meta.loss_cross_edge_weight) * cross_edge_random_qfl,
            "cross_edge_contact_aux": float(meta.loss_contact_aux_weight) * cross_edge_contact_aux_bce,
        }
        return losses, aux_metrics

    def _compute_random_diagnostic_metrics(
        self,
        *,
        edge_target: torch.Tensor,
        edge_prob: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        obj_valid_mask: torch.Tensor,
        beta: float,
    ) -> dict[str, float | MetricStat]:
        """Diagnostic metrics for the random128 stream.

        These are the metrics that must remain comparable to the
        ``random128 baseline`` runs, so they MUST NOT mix in any
        contact-aux edges.
        """
        metrics: dict[str, float | MetricStat] = {}

        # ---- Nonzero edges ----
        nonzero_mask = (edge_target > 0) & edge_valid_mask
        nonzero_count = nonzero_mask.sum()
        if bool(nonzero_count > 0):
            metrics["cross_edge_random_nonzero_mae"] = (
                (edge_prob - edge_target).abs()[nonzero_mask].sum() / nonzero_count.float()
            )
            metrics["cross_edge_random_nonzero_bce"] = (
                -(
                    edge_target[nonzero_mask] * torch.log(edge_prob[nonzero_mask].clamp(min=1e-6))
                    + (1.0 - edge_target[nonzero_mask])
                    * torch.log((1.0 - edge_prob[nonzero_mask]).clamp(min=1e-6))
                ).mean()
            )
            metrics["cross_edge_random_nonzero_pred_mean"] = edge_prob[nonzero_mask].mean()
            metrics["cross_edge_random_nonzero_target_mean"] = edge_target[nonzero_mask].mean()
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["cross_edge_random_nonzero_mae"] = zero_tensor
            metrics["cross_edge_random_nonzero_bce"] = zero_tensor
            metrics["cross_edge_random_nonzero_pred_mean"] = zero_tensor
            metrics["cross_edge_random_nonzero_target_mean"] = zero_tensor
        metrics["cross_edge_random_nonzero_count"] = MetricStat(
            total=float(nonzero_count.detach().cpu()),
            count=float(1),
            expose_validity=True,
        )

        # ---- Target strength bins (count + MAE) ----
        for lower, upper, name in _TARGET_STRENGTH_BINS:
            bin_mask = target_strength_bin_mask(edge_target, lower=lower, upper=upper) & edge_valid_mask
            bin_count = bin_mask.sum()
            metrics[f"random_{name}_count"] = MetricStat(
                total=float(bin_count.detach().cpu()),
                count=float(1),
                expose_validity=True,
            )
            if bool(bin_count > 0):
                metrics[f"random_{name}_mae"] = (
                    (edge_prob - edge_target).abs()[bin_mask].sum() / bin_count.float()
                )
            else:
                metrics[f"random_{name}_mae"] = edge_target.sum() * 0.0

        # ---- Zero edge prediction distribution ----
        zero_edge_mask = (edge_target == 0) & edge_valid_mask
        zero_edge_count = zero_edge_mask.sum()
        if bool(zero_edge_count > 0):
            zero_pred = edge_prob[zero_edge_mask].float()
            metrics["cross_edge_random_zero_pred_mean"] = zero_pred.mean()
            metrics["cross_edge_random_zero_pred_p95"] = torch.quantile(zero_pred, 0.95)
            metrics["cross_edge_random_zero_pred_p99"] = torch.quantile(zero_pred, 0.99)
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["cross_edge_random_zero_pred_mean"] = zero_tensor
            metrics["cross_edge_random_zero_pred_p95"] = zero_tensor
            metrics["cross_edge_random_zero_pred_p99"] = zero_tensor

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

    def _compute_contact_aux_diagnostic_metrics(
        self,
        *,
        edge_target: torch.Tensor,
        edge_prob: torch.Tensor,
        edge_valid_mask: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> dict[str, float | MetricStat]:
        """Diagnostic metrics for the contact auxiliary stream only.

        These must NEVER be mixed with the random stream metrics: the
        contact stream is biased by construction (only y > 0 edges), so
        diagnostic quantities like nonzero_fraction, target_mean, and
        nonzero_pred_mean would shift on their own.
        """
        metrics: dict[str, float | MetricStat] = {}

        # Per-stream means and MAE
        if bool(edge_valid_mask.sum() > 0):
            metrics["contact_aux_pred_mean"] = edge_prob[edge_valid_mask].mean()
            metrics["contact_aux_target_mean"] = edge_target[edge_valid_mask].mean()
            metrics["contact_aux_nonzero_mae"] = (
                (edge_prob - edge_target).abs()[edge_valid_mask].sum()
                / edge_valid_mask.sum().float()
            )
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["contact_aux_pred_mean"] = zero_tensor
            metrics["contact_aux_target_mean"] = zero_tensor
            metrics["contact_aux_nonzero_mae"] = zero_tensor

        metrics["contact_aux_valid_edge_count"] = MetricStat(
            total=float(edge_valid_mask.sum().detach().cpu()),
            count=float(1),
            expose_validity=True,
        )
        per_obj_has_edge = edge_valid_mask.any(dim=-1) & obj_valid_mask
        num_valid_obj = obj_valid_mask.sum()
        num_obj_with_edge = per_obj_has_edge.sum()
        metrics["contact_aux_object_coverage"] = MetricStat(
            total=float(num_obj_with_edge.detach().cpu()),
            count=float(num_valid_obj.detach().cpu()),
        )

        # Per-bin count and MAE (matches the sampler order weak/medium/strong/very_strong).
        for lower, upper, name in _TARGET_STRENGTH_BINS:
            bin_mask = target_strength_bin_mask(edge_target, lower=lower, upper=upper) & edge_valid_mask
            bin_count = bin_mask.sum()
            metrics[f"contact_aux_{name}_count"] = MetricStat(
                total=float(bin_count.detach().cpu()),
                count=float(1),
                expose_validity=True,
            )
            if bool(bin_count > 0):
                metrics[f"contact_aux_{name}_mae"] = (
                    (edge_prob - edge_target).abs()[bin_mask].sum() / bin_count.float()
                )
            else:
                metrics[f"contact_aux_{name}_mae"] = edge_target.sum() * 0.0
        return metrics

    def _compute_hand_contact_gt_metrics(
        self,
        *,
        hand_target: torch.Tensor,
    ) -> dict[str, float | MetricStat]:
        """GT-only hand contact observation.

        The hand head is NOT supervised in v2.1, so any prediction-side
        metric (MAE, QFL, prob mean) would be meaningless. We only log
        the GT distribution to detect train/val domain shift and to keep
        the option open for a future detached probe on z_hand.
        """
        metrics: dict[str, float | MetricStat] = {}
        nonzero_mask = hand_target > 0
        nonzero_count = nonzero_mask.sum()
        total = float(hand_target.numel())
        if bool(nonzero_count > 0):
            metrics["hand_contact_target_nonzero_fraction"] = float(nonzero_count) / max(total, 1.0)
            metrics["hand_contact_target_nonzero_mean"] = hand_target[nonzero_mask].mean()
        else:
            zero_tensor = hand_target.sum() * 0.0
            metrics["hand_contact_target_nonzero_fraction"] = zero_tensor
            metrics["hand_contact_target_nonzero_mean"] = zero_tensor
        metrics["hand_contact_target_mean"] = hand_target.mean()
        metrics["hand_contact_target_count"] = MetricStat(
            total=total,
            count=float(1),
            expose_validity=True,
        )
        for lower, upper, name in _TARGET_STRENGTH_BINS:
            bin_mask = target_strength_bin_mask(hand_target, lower=lower, upper=upper)
            bin_count = bin_mask.sum()
            metrics[f"hand_target_{name}_fraction"] = (
                float(bin_count) / max(total, 1.0)
            )
        return metrics

    def inference(self, model: torch.nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
        return model(self.prepare_batch(inputs))
