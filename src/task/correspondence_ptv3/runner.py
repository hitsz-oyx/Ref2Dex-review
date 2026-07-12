from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from hydra.utils import instantiate

from src.base import BaseRunner, MetricStat, RunnerOutput, TaskConfig, set_config_default_if_not_explicit
from src.task.correspondence_ptv3.data import make_dataloaders
from src.task.correspondence_ptv3.metrics import (
    batched_binary_auprc_stat,
    batched_cross_edge_rank_at_k_stat,
)
from src.task.correspondence_ptv3.objectives import CrossEdgeRankKObjective
from src.task.correspondence_ptv3.supervision.base import reduce_loss_map_per_object
from src.task.correspondence_ptv3.supervision.soft import (
    binary_entropy_floor_map,
    flatten_binary_logits,
    masked_bce_with_logits,
    masked_bce_with_logits_per_object,
)
from src.task.correspondence_ptv3.targets import (
    build_clean_correspondence_targets,
    build_dynamic_edge_contact_targets,
)


def prefix_metrics(
    prefix: str,
    metrics: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def set_model_config_default_if_not_explicit(
    cfg: TaskConfig,
    *,
    key: str,
    value: Any,
    explicit_override_keys: set[str] | None,
) -> bool:
    explicit_override_keys = set(explicit_override_keys or set())
    dotted_key = f"model.{key}"
    if dotted_key in explicit_override_keys:
        return False
    model_cfg = getattr(cfg, "model", None)
    if not isinstance(model_cfg, dict):
        raise TypeError("Correspondence runner expects cfg.model to be a dict-backed Hydra component config.")
    cursor = model_cfg
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value
    return True


class CorrespondencePTV3Runner(BaseRunner):
    @classmethod
    def configure_overfit_mode(
        cls,
        cfg: TaskConfig,
        explicit_override_keys: set[str],
    ) -> None:
        super().configure_overfit_mode(cfg, explicit_override_keys)
        set_config_default_if_not_explicit(
            cfg,
            key="meta.fix_overfit_seed",
            value=True,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="meta.augment",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="meta.apply_hand_perturb",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="meta.augment_rotation",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="meta.augment_translation",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="meta.augment_scale",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="meta.hand_perturb_prob",
            value=0.0,
            explicit_override_keys=explicit_override_keys,
        )
        set_config_default_if_not_explicit(
            cfg,
            key="meta.val_augment",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )
        set_model_config_default_if_not_explicit(
            cfg,
            key="ptv3.drop_path",
            value=0.0,
            explicit_override_keys=explicit_override_keys,
        )
        set_model_config_default_if_not_explicit(
            cfg,
            key="ptv3.shuffle_orders",
            value=False,
            explicit_override_keys=explicit_override_keys,
        )

    @property
    def edge_sampler(self):
        sampler = getattr(self, "_edge_sampler", None)
        if sampler is None:
            sampler = instantiate(self.cfg.edge_sampler)
            self._edge_sampler = sampler
        return sampler

    @property
    def contact_supervision(self):
        supervision = getattr(self, "_contact_supervision", None)
        if supervision is None:
            supervision = instantiate(self.cfg.contact_supervision)
            self._contact_supervision = supervision
        return supervision

    @property
    def rankk_objective(self) -> CrossEdgeRankKObjective:
        objective = getattr(self, "_rankk_objective", None)
        if objective is None:
            meta = self.cfg.meta
            objective = CrossEdgeRankKObjective(
                k=int(getattr(meta, "rankk_loss_k", 4)),
                label_gap=float(getattr(meta, "rankk_label_gap", 0.15)),
                margin=float(getattr(meta, "rankk_margin", 0.2)),
            )
            self._rankk_objective = objective
        return objective

    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(
            data_cfg,
            meta_cfg=self.cfg.meta,
            edge_sampler=self.edge_sampler,
            seed=seed,
            distributed=self.distributed,
        )

    def configure_data(
        self,
        metadata: dict[str, Any],
        train_dataset: Any | None = None,
    ) -> None:
        super().configure_data(metadata, train_dataset)
        for field in (
            "num_obj_pool",
            "num_hand_points",
            "k_cross",
            "num_fingers",
            "num_regions",
        ):
            if field not in metadata:
                continue
            value = metadata[field]
            if int(value) > 0:
                setattr(self.cfg.meta, field, int(value))

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return instantiate(
            model_cfg,
            _convert_="object",
            _recursive_=False,
            task_meta=self.cfg.meta,
            contact_supervision=self.contact_supervision,
        )

    def train_epoch(self, epoch: int) -> dict[str, float]:
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
        metrics: dict[str, float | MetricStat] = {}
        for key, value in {**losses, **aux_metrics}.items():
            if isinstance(value, MetricStat):
                metrics[key] = value
            else:
                metrics[key] = float(value.detach().cpu())
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
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | MetricStat]]:
        meta = self.cfg.meta
        contact_supervision = self.contact_supervision
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
        pred_obj_contact_logits = preds.get(
            "pred_obj_contact_logits",
            preds.get("pred_obj_contact_bin"),
        )
        if pred_obj_contact_logits is None:
            raise KeyError("preds must contain pred_obj_contact_logits or pred_obj_contact_bin.")
        pred_cross_contact_logits = preds.get(
            "pred_cross_contact_logits",
            preds.get("pred_cross_contact_bin"),
        )
        if pred_cross_contact_logits is None:
            raise KeyError("preds must contain pred_cross_contact_logits or pred_cross_contact_bin.")

        obj_result = contact_supervision.compute_object_loss(
            logits=pred_obj_contact_logits,
            target_probability=target_contact_prob,
            valid_mask=obj_valid,
        )

        edge_valid = input_edge_valid & obj_valid_bool.unsqueeze(-1)
        edge_contact_prob = build_dynamic_edge_contact_targets(
            batch,
            num_obj_points=int(meta.num_obj_points),
            num_hand_points=int(meta.num_hand_points),
            d_pos=float(meta.d_pos),
            d_neg=float(meta.d_neg),
            gamma=float(meta.gamma),
        )
        edge_result = contact_supervision.compute_edge_loss(
            logits=pred_cross_contact_logits,
            target_probability=edge_contact_prob,
            edge_weight=input_edge_weight,
            obj_valid_mask=obj_valid_bool,
        )
        rankk_weight = float(getattr(meta, "loss_cross_edge_rankk_weight", 0.0))
        if rankk_weight > 0.0:
            cross_edge_rankk_loss, cross_edge_rankk_pairs = self.rankk_objective.compute(
                preds["pred_cross_contact_prob"],
                edge_contact_prob,
                edge_valid,
            )
        else:
            cross_edge_rankk_loss = edge_result.loss.new_tensor(0.0)
            cross_edge_rankk_pairs = edge_result.loss.new_tensor(0.0)

        need_clean_targets = any(
            key in preds
            for key in ("pred_obj_cano", "pred_obj_finger", "pred_obj_region")
        )
        zero = obj_result.loss.new_tensor(0.0)
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
            ) = build_clean_correspondence_targets(
                batch,
                corr_contact_label_min=float(
                    getattr(self.cfg.meta, "corr_contact_label_min", 0.1)
                ),
            )

        cano_loss = zero
        if "pred_obj_cano" in preds:
            assert target_obj_cano is not None
            cano_diff = F.smooth_l1_loss(
                preds["pred_obj_cano"],
                target_obj_cano,
                reduction="none",
            ).sum(dim=-1)
            cano_loss = ((cano_diff * corr_valid).sum() / corr_valid.sum().clamp(min=1.0))

        finger_loss = zero
        if "pred_obj_finger" in preds:
            assert target_finger is not None
            finger_valid = corr_valid * (target_finger >= 0).float()
            if bool(finger_valid.any()):
                finger_map = F.cross_entropy(
                    preds["pred_obj_finger"].reshape(-1, preds["pred_obj_finger"].shape[-1]),
                    target_finger.reshape(-1).clamp(min=0),
                    reduction="none",
                ).reshape_as(target_finger)
                finger_loss = (finger_map * finger_valid).sum() / finger_valid.sum().clamp(min=1.0)

        region_loss = zero
        if "pred_obj_region" in preds:
            assert target_region is not None
            region_valid = corr_valid * (target_region >= 0).float()
            if bool(region_valid.any()):
                region_map = F.cross_entropy(
                    preds["pred_obj_region"].reshape(-1, preds["pred_obj_region"].shape[-1]),
                    target_region.reshape(-1).clamp(min=0),
                    reduction="none",
                ).reshape_as(target_region)
                region_loss = (region_map * region_valid).sum() / region_valid.sum().clamp(min=1.0)

        losses = {
            "obj_contact": float(getattr(meta, "loss_obj_contact_weight", 1.0)) * obj_result.loss,
            "obj_cano": float(meta.loss_cano_weight) * cano_loss,
            "obj_finger": float(meta.loss_finger_weight) * finger_loss,
            "obj_region": float(meta.loss_region_weight) * region_loss,
            "cross_edge_contact": float(meta.loss_cross_edge_weight) * edge_result.loss,
            "cross_edge_rankk": rankk_weight * cross_edge_rankk_loss,
        }

        pr_label_threshold = float(getattr(meta, "pr_label_threshold", 0.5))
        with torch.no_grad():
            obj_contact_auprc = batched_binary_auprc_stat(
                preds["pred_obj_contact_prob"].detach(),
                target_contact_prob > pr_label_threshold,
                obj_valid_bool,
            )
            cross_edge_auprc = batched_binary_auprc_stat(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob > pr_label_threshold,
                edge_valid,
            )
            cross_edge_rank1 = batched_cross_edge_rank_at_k_stat(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob,
                edge_valid,
                k=1,
            )
            cross_edge_rank4 = batched_cross_edge_rank_at_k_stat(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob,
                edge_valid,
                k=4,
            )
            cross_edge_rank8 = batched_cross_edge_rank_at_k_stat(
                preds["pred_cross_contact_prob"].detach(),
                edge_contact_prob,
                edge_valid,
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

        aux_metrics: dict[str, torch.Tensor | MetricStat] = {
            **prefix_metrics("obj_contact", obj_result.metrics),
            **prefix_metrics("cross_edge", edge_result.metrics),
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
        return losses, aux_metrics

    @staticmethod
    def _masked_bce_from_prob(
        pred_prob: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        with torch.autocast(device_type=pred_prob.device.type, enabled=False):
            pred_prob = pred_prob.float().clamp(min=eps, max=1.0 - eps)
            target = target.float()
            mask = mask.float()
            return F.binary_cross_entropy(
                pred_prob,
                target,
                weight=mask,
                reduction="sum",
            ) / mask.sum().clamp(min=1.0)

    # -------------------------------------------------------------------------
    # Legacy test/import compatibility.
    # New code must import from supervision/, metrics/, objectives.py, or targets/.
    # -------------------------------------------------------------------------

    @staticmethod
    def _flatten_binary_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return flatten_binary_logits(logits, target)

    @staticmethod
    def _masked_bce_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return masked_bce_with_logits(logits, target, mask)

    @staticmethod
    def _masked_bce_with_logits_per_object(
        logits: torch.Tensor,
        target: torch.Tensor,
        edge_weight: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return masked_bce_with_logits_per_object(
            logits,
            target,
            edge_weight,
            obj_valid_mask,
        )

    @staticmethod
    def _reduce_loss_map_per_object(
        loss_map: torch.Tensor,
        edge_weight: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return reduce_loss_map_per_object(loss_map, edge_weight, obj_valid_mask)

    @staticmethod
    def _binary_entropy_floor_map(
        target: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        return binary_entropy_floor_map(target, eps=eps)

    @staticmethod
    def _soft_bce_entropy_floor(
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        entropy = binary_entropy_floor_map(target)
        mask = mask.float()
        return (entropy * mask).sum() / mask.sum().clamp(min=1.0)

    @staticmethod
    def _soft_bce_entropy_floor_per_object(
        target: torch.Tensor,
        edge_weight: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return reduce_loss_map_per_object(
            binary_entropy_floor_map(target),
            edge_weight,
            obj_valid_mask,
        )

    @staticmethod
    def _batched_binary_auprc_stat(
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> MetricStat:
        return batched_binary_auprc_stat(scores, labels, valid_mask)

    @staticmethod
    def _batched_cross_edge_rank_at_k_stat(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        k: int,
    ) -> MetricStat:
        return batched_cross_edge_rank_at_k_stat(
            pred,
            target,
            valid_mask,
            k=k,
        )

    @staticmethod
    def _batched_binary_auprc(
        scores: torch.Tensor,
        labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        stat = batched_binary_auprc_stat(scores, labels, valid_mask)
        if stat.count <= 0.0:
            return scores.new_tensor(0.0)
        return scores.new_tensor(stat.total / stat.count)

    @staticmethod
    def _batched_cross_edge_rank_at_k(
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        k: int,
    ) -> torch.Tensor:
        stat = batched_cross_edge_rank_at_k_stat(pred, target, valid_mask, k=k)
        if stat.count <= 0.0:
            return pred.new_tensor(0.0)
        return pred.new_tensor(stat.total / stat.count)

    def inference(
        self,
        model: torch.nn.Module,
        inputs: Any,
    ) -> dict[str, torch.Tensor]:
        return model(self.prepare_batch(inputs))
