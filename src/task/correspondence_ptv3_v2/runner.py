from __future__ import annotations

from typing import Any

import torch

from src.base import BaseRunner, MetricStat, RunnerOutput, TaskConfig, set_config_default_if_not_explicit
from src.task.correspondence_ptv3_v2.dataset import make_dataloaders
from src.task.correspondence_ptv3_v2.losses import (
    contact_target_from_distance,
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
from src.utils.correspondence import gather_batched_knn_features


_TARGET_STRENGTH_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.25, "edge_y_0_025"),
    (0.25, 0.50, "edge_y_025_050"),
    (0.50, 0.75, "edge_y_050_075"),
    (0.75, 1.00, "edge_y_075_100"),
)


class CorrespondencePTV3V2Runner(BaseRunner):
    _CONTACT_POSITIVE_EPS = 1e-4
    _CONTACT_SAMPLE_CHUNK = 64

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
            "num_fingers",
            "num_regions",
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

    def prepare_batch(self, batch: Any) -> Any:
        if not isinstance(batch, dict):
            return super().prepare_batch(batch)

        contact_seed = batch.pop("contact_seed", None)
        batch = super().prepare_batch(batch)
        if contact_seed is None:
            return batch

        if torch.is_tensor(contact_seed):
            contact_seed_list = [int(value) for value in contact_seed.reshape(-1).tolist()]
        else:
            contact_seed_list = [int(contact_seed)]
        with torch.no_grad():
            self._build_supervision_gpu(batch, contact_seed_list=contact_seed_list)
        return batch

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        preds = model(batch)
        diagnostic_every = max(int(getattr(self.cfg.train, "diagnostic_every_steps", 20)), 1)
        compute_diagnostics = mode == "eval" or ((self.global_step + 1) % diagnostic_every == 0)
        losses, aux_metrics = self._compute_losses(
            preds,
            batch,
            compute_diagnostics=compute_diagnostics,
        )
        total_loss = sum(losses.values())
        metrics: dict[str, float | torch.Tensor | MetricStat] = {**losses, **aux_metrics}
        metrics["loss"] = total_loss
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(batch["points"].shape[0]))

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        *,
        compute_diagnostics: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float | torch.Tensor | MetricStat]]:
        meta = self.cfg.meta
        obj_valid_mask = batch["runtime_obj_valid_mask"].bool()
        beta = float(meta.quality_focal_beta)

        # ---- Random-edge stream (L_r) ----
        random_edge_valid_mask = batch["random_edge_valid_mask"].bool() & obj_valid_mask.unsqueeze(-1)
        random_target = batch["random_edge_contact_target"].float()
        random_logits = preds["pred_cross_random_logits"]
        random_prob = preds["pred_cross_random_prob"]

        random_qfl_map = quality_focal_loss_map(random_logits, random_target, beta=beta)
        cross_edge_random_qfl = reduce_loss_map_per_object(
            random_qfl_map, random_edge_valid_mask, obj_valid_mask
        )

        # ---- Contact-aware auxiliary stream (L_c) ----
        contact_edge_valid_mask = batch["contact_edge_valid_mask"].bool() & obj_valid_mask.unsqueeze(-1)
        contact_edge_target = batch["contact_edge_contact_target"].float()
        contact_logits = preds["pred_cross_contact_aux_logits"]
        contact_prob = preds["pred_cross_contact_aux_prob"]

        contact_qfl_map = quality_focal_loss_map(contact_logits, contact_edge_target, beta=beta)
        cross_edge_contact_aux_qfl = reduce_loss_map_per_object(
            contact_qfl_map, contact_edge_valid_mask, obj_valid_mask
        )

        # ---- Dense hand-contact heatmap (L_h) ----
        hand_contact_weight = float(getattr(meta, "loss_hand_contact_weight", 0.005))
        hand_target: torch.Tensor | None = None
        hand_prob: torch.Tensor | None = None
        hand_contact_bce: torch.Tensor | None = None
        if hand_contact_weight > 0.0:
            hand_target = batch["hand_contact_target"].float()
            hand_logits = preds["pred_hand_contact_logits"]
            hand_prob = preds["pred_hand_contact_prob"]
            if hand_logits.shape != hand_target.shape:
                raise ValueError(
                    "pred_hand_contact_logits and hand_contact_target must have the same "
                    f"shape, got {tuple(hand_logits.shape)} and {tuple(hand_target.shape)}."
                )
            hand_contact_bce = reduce_loss_map(
                binary_cross_entropy_with_logits_map(hand_logits, hand_target),
                torch.ones_like(hand_target, dtype=torch.bool),
            )

        aux_metrics: dict[str, float | torch.Tensor | MetricStat] = {
            "cross_edge_random_qfl": cross_edge_random_qfl,
            "contact_aux_qfl": cross_edge_contact_aux_qfl,
        }
        if hand_contact_bce is not None:
            aux_metrics["hand_contact_bce"] = hand_contact_bce
        if compute_diagnostics:
            random_bce_map = binary_cross_entropy_with_logits_map(random_logits, random_target)
            random_mae_map = torch.abs(random_prob - random_target)
            cross_edge_random_bce = reduce_loss_map_per_object(
                random_bce_map, random_edge_valid_mask, obj_valid_mask
            )
            cross_edge_random_mae = reduce_loss_map_per_object(
                random_mae_map, random_edge_valid_mask, obj_valid_mask
            )
            contact_soft_bce_map = binary_cross_entropy_with_logits_map(contact_logits, contact_edge_target)
            contact_mae_map = torch.abs(contact_prob - contact_edge_target)
            cross_edge_contact_aux_soft_bce = reduce_loss_map_per_object(
                contact_soft_bce_map, contact_edge_valid_mask, obj_valid_mask
            )
            cross_edge_contact_aux_bce = reduce_loss_map_per_object(
                contact_soft_bce_map, contact_edge_valid_mask, obj_valid_mask
            )
            cross_edge_contact_aux_mae = reduce_loss_map_per_object(
                contact_mae_map, contact_edge_valid_mask, obj_valid_mask
            )
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
            aux_metrics.update({
                "cross_edge_random_bce": cross_edge_random_bce,
                "cross_edge_random_mae": cross_edge_random_mae,
                "cross_edge_random_oracle_bce": cross_edge_random_oracle_bce,
                "cross_edge_random_oracle_bce_global": cross_edge_random_oracle_bce_global,
                "cross_edge_random_excess_bce": cross_edge_random_excess_bce,
                "cross_edge_random_excess_bce_global": cross_edge_random_excess_bce_global,
                "random_num_valid_obj": random_num_valid_obj.float(),
                "random_num_valid_edges": random_valid_edge_count.float(),
                "random_sampled_nonzero_edge_count": random_sampled_nonzero_edge_count.float(),
                "contact_aux_bce": cross_edge_contact_aux_bce,
                "contact_aux_soft_bce": cross_edge_contact_aux_soft_bce,
                "contact_aux_mae": cross_edge_contact_aux_mae,
                "random_sampled_nonzero_edge_fraction": MetricStat(
                    total=float(random_sampled_nonzero_edge_count.detach().cpu()),
                    count=float(random_valid_edge_count.detach().cpu()),
                ),
                "random_object_nonzero_edge_coverage": MetricStat(
                    total=float(random_num_obj_with_nonzero.detach().cpu()),
                    count=float(random_num_valid_obj.detach().cpu()),
                ),
            })
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
            if hand_target is not None and hand_prob is not None:
                aux_metrics.update(self._compute_hand_contact_gt_metrics(hand_target=hand_target))
                aux_metrics.update(
                    self._compute_hand_contact_diagnostic_metrics(
                        hand_target=hand_target,
                        hand_prob=hand_prob,
                    )
                )

        losses = {
            "cross_edge_random": float(meta.loss_cross_edge_weight) * cross_edge_random_qfl,
            "cross_edge_contact_aux": float(meta.loss_contact_aux_weight) * cross_edge_contact_aux_qfl,
        }
        if hand_contact_bce is not None:
            losses["hand_contact"] = hand_contact_weight * hand_contact_bce
        return losses, aux_metrics

    def _build_supervision_gpu(
        self,
        batch: dict[str, torch.Tensor],
        *,
        contact_seed_list: list[int],
    ) -> None:
        num_obj = int(self.cfg.meta.num_obj_points)
        num_hand = int(self.cfg.meta.num_hand_points)
        gt_points = batch["gt_points"].float()
        gt_obj = gt_points[:, :num_obj]
        gt_hand = gt_points[:, num_obj : num_obj + num_hand]
        obj_valid_mask = batch["runtime_obj_valid_mask"].bool()
        random_edge_idx = batch["random_edge_idx"].long()
        random_edge_valid_mask = batch["random_edge_valid_mask"].bool()

        batch["random_edge_contact_target"] = self._build_random_edge_target(
            gt_obj=gt_obj,
            gt_hand=gt_hand,
            random_edge_idx=random_edge_idx,
            random_edge_valid_mask=random_edge_valid_mask,
        )
        contact_edge_idx, contact_edge_valid_mask, contact_edge_target = self._build_contact_supervision_edges(
            gt_obj=gt_obj,
            gt_hand=gt_hand,
            obj_valid_mask=obj_valid_mask,
            contact_seed_list=contact_seed_list,
        )
        batch["contact_edge_idx"] = contact_edge_idx
        batch["contact_edge_valid_mask"] = contact_edge_valid_mask
        batch["contact_edge_contact_target"] = contact_edge_target
        batch["hand_contact_target"] = contact_target_from_distance(
            batch["hand_min_dist"].float(),
            contact_radius=float(self.cfg.meta.contact_radius),
        ).float()

    def _build_random_edge_target(
        self,
        *,
        gt_obj: torch.Tensor,
        gt_hand: torch.Tensor,
        random_edge_idx: torch.Tensor,
        random_edge_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_hand = gather_batched_knn_features(
            gt_hand,
            random_edge_idx,
            random_edge_valid_mask,
        )
        distance = torch.norm(neighbor_hand - gt_obj.unsqueeze(2), dim=-1)
        target = contact_target_from_distance(
            distance,
            contact_radius=float(self.cfg.meta.contact_radius),
        )
        return target * random_edge_valid_mask.float()

    def _build_contact_supervision_edges(
        self,
        *,
        gt_obj: torch.Tensor,
        gt_hand: torch.Tensor,
        obj_valid_mask: torch.Tensor,
        contact_seed_list: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_obj, _ = gt_obj.shape
        num_hand = gt_hand.shape[1]
        quotas = tuple(int(value) for value in getattr(self.cfg.meta, "contact_supervision_quotas", (16, 16, 16, 16)))
        hard_negative_quota = int(getattr(self.cfg.meta, "contact_supervision_hard_negative_quota", 16))
        neg_min, neg_max = tuple(
            float(value)
            for value in getattr(self.cfg.meta, "contact_supervision_hard_negative_distance_range", (0.02, 0.03))
        )
        total_quota = sum(quotas) + hard_negative_quota
        edge_idx = torch.full((batch_size, num_obj, total_quota), -1, device=gt_obj.device, dtype=torch.long)
        edge_valid = torch.zeros((batch_size, num_obj, total_quota), device=gt_obj.device, dtype=torch.bool)
        edge_target = torch.zeros((batch_size, num_obj, total_quota), device=gt_obj.device, dtype=torch.float32)
        seed_tensor = torch.as_tensor(contact_seed_list, device=gt_obj.device, dtype=torch.long)
        contact_radius = float(self.cfg.meta.contact_radius)

        for start in range(0, num_obj, self._CONTACT_SAMPLE_CHUNK):
            end = min(start + self._CONTACT_SAMPLE_CHUNK, num_obj)
            dist = torch.cdist(gt_obj[:, start:end], gt_hand)
            target = contact_target_from_distance(dist, contact_radius=contact_radius)
            row_valid = obj_valid_mask[:, start:end].unsqueeze(-1)
            score = self._deterministic_contact_scores(
                seed_tensor=seed_tensor,
                obj_start=start,
                obj_count=end - start,
                num_hand=num_hand,
                device=gt_obj.device,
            )
            masks = (
                row_valid & (target > self._CONTACT_POSITIVE_EPS) & (target <= 0.25),
                row_valid & (target > 0.25) & (target <= 0.50),
                row_valid & (target > 0.50) & (target <= 0.75),
                row_valid & (target > 0.75),
                row_valid & (dist >= neg_min) & (dist < neg_max) & (target <= 0.0),
            )
            widths = (*quotas, hard_negative_quota)
            write_col = 0
            for mask, width in zip(masks, widths):
                sampled_idx, sampled_valid, sampled_target = self._sample_contact_edges_from_mask(
                    mask=mask,
                    score=score,
                    target=target,
                    quota=width,
                )
                next_col = write_col + width
                edge_idx[:, start:end, write_col:next_col] = sampled_idx
                edge_valid[:, start:end, write_col:next_col] = sampled_valid
                edge_target[:, start:end, write_col:next_col] = sampled_target
                write_col = next_col
        return edge_idx, edge_valid, edge_target

    @staticmethod
    def _sample_contact_edges_from_mask(
        *,
        mask: torch.Tensor,
        score: torch.Tensor,
        target: torch.Tensor,
        quota: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_obj_chunk, _ = mask.shape
        if quota <= 0:
            empty_idx = torch.empty((batch_size, num_obj_chunk, 0), device=mask.device, dtype=torch.long)
            empty_valid = torch.empty((batch_size, num_obj_chunk, 0), device=mask.device, dtype=torch.bool)
            empty_target = torch.empty((batch_size, num_obj_chunk, 0), device=mask.device, dtype=target.dtype)
            return empty_idx, empty_valid, empty_target
        masked_score = score.masked_fill(~mask, float("inf"))
        values, idx = torch.topk(masked_score, k=quota, dim=-1, largest=False)
        valid = torch.isfinite(values)
        sampled_target = torch.gather(target, dim=-1, index=idx) * valid.float()
        return idx.masked_fill(~valid, -1), valid, sampled_target

    @staticmethod
    def _deterministic_contact_scores(
        *,
        seed_tensor: torch.Tensor,
        obj_start: int,
        obj_count: int,
        num_hand: int,
        device: torch.device,
    ) -> torch.Tensor:
        obj_idx = torch.arange(obj_start, obj_start + obj_count, device=device, dtype=torch.long).view(1, obj_count, 1)
        hand_idx = torch.arange(num_hand, device=device, dtype=torch.long).view(1, 1, num_hand)
        seed = seed_tensor.view(-1, 1, 1)
        hashed = seed ^ (obj_idx * 1000003) ^ (hand_idx * 9176)
        hashed = (hashed * 1103515245 + 12345) & 0x7FFFFFFF
        return hashed.to(torch.float32) / 2147483648.0

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
        contact stream is biased by construction (stratified positives
        plus a narrow hard-negative band), so diagnostic quantities must
        be interpreted on that auxiliary distribution only.
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
        hard_negative_mask = (edge_target <= 0) & edge_valid_mask
        hard_negative_count = hard_negative_mask.sum()
        metrics["contact_aux_hard_neg_count"] = MetricStat(
            total=float(hard_negative_count.detach().cpu()),
            count=float(1),
            expose_validity=True,
        )
        if bool(hard_negative_count > 0):
            hard_negative_pred = edge_prob[hard_negative_mask].float()
            metrics["contact_aux_hard_neg_pred_mean"] = hard_negative_pred.mean()
            metrics["contact_aux_hard_neg_pred_p95"] = torch.quantile(hard_negative_pred, 0.95)
        else:
            zero_tensor = edge_target.sum() * 0.0
            metrics["contact_aux_hard_neg_pred_mean"] = zero_tensor
            metrics["contact_aux_hard_neg_pred_p95"] = zero_tensor
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

    @staticmethod
    def _compute_hand_contact_diagnostic_metrics(
        *,
        hand_target: torch.Tensor,
        hand_prob: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Prediction-side diagnostics for the object-conditioned hand heatmap."""
        if hand_prob.shape != hand_target.shape:
            raise ValueError(
                "pred_hand_contact_prob and hand_contact_target must have the same "
                f"shape, got {tuple(hand_prob.shape)} and {tuple(hand_target.shape)}."
            )
        nonzero_mask = hand_target > 0
        if bool(nonzero_mask.any()):
            nonzero_mae = (hand_prob - hand_target).abs()[nonzero_mask].mean()
            nonzero_pred_mean = hand_prob[nonzero_mask].mean()
        else:
            nonzero_mae = hand_target.sum() * 0.0
            nonzero_pred_mean = hand_target.sum() * 0.0
        return {
            "hand_contact_mae": (hand_prob - hand_target).abs().mean(),
            "hand_contact_pred_mean": hand_prob.mean(),
            "hand_contact_nonzero_mae": nonzero_mae,
            "hand_contact_nonzero_pred_mean": nonzero_pred_mean,
        }

    def inference(self, model: torch.nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
        return model(self.prepare_batch(inputs))
