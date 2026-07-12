from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig, set_config_default_if_not_explicit
from src.task.correspondence_ptv3_v2.dataset import make_dataloaders
from src.task.correspondence_ptv3_v2.losses import (
    quality_focal_loss_map,
    reduce_loss_map,
    reduce_loss_map_per_object,
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
            ("meta.apply_hand_perturb", False),
            ("meta.augment_rotation", False),
            ("meta.augment_translation", False),
            ("meta.augment_scale", False),
            ("meta.hand_perturb_prob", 0.0),
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
        for field in (
            "num_obj_pool",
            "num_obj_points",
            "num_hand_points",
            "k_cross",
            "k_ctx",
            "ctx_radius",
            "num_supervision_edges",
            "contact_radius",
        ):
            if field in metadata:
                setattr(self.cfg.meta, field, metadata[field])

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def train_epoch(self, epoch: int) -> dict[str, float]:
        dataset = getattr(getattr(self, "train_loader", None), "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        sampler = getattr(getattr(self, "train_loader", None), "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        return super().train_epoch(epoch)

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        del mode
        preds = model(batch)
        losses, aux_metrics = self._compute_losses(preds, batch)
        total_loss = sum(losses.values())
        metrics = {key: float(value.detach().cpu()) for key, value in {**losses, **aux_metrics}.items()}
        metrics["loss"] = float(total_loss.detach().cpu())
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

        obj_target = batch["contact_target"].float()
        edge_target = batch["edge_contact_target"].float()
        obj_logits = preds["pred_obj_contact_logits"]
        edge_logits = preds["pred_cross_contact_logits"]
        beta = float(meta.quality_focal_beta)

        obj_qfl_map = quality_focal_loss_map(obj_logits, obj_target, beta=beta)
        edge_qfl_map = quality_focal_loss_map(edge_logits, edge_target, beta=beta)
        obj_bce_map = F.binary_cross_entropy_with_logits(obj_logits.float(), obj_target, reduction="none")
        edge_bce_map = F.binary_cross_entropy_with_logits(edge_logits.float(), edge_target, reduction="none")
        obj_mae_map = torch.abs(preds["pred_obj_contact_prob"] - obj_target)
        edge_mae_map = torch.abs(preds["pred_cross_contact_prob"] - edge_target)

        obj_contact_qfl = reduce_loss_map(obj_qfl_map, obj_valid_mask)
        cross_edge_qfl = reduce_loss_map_per_object(edge_qfl_map, edge_valid_mask, obj_valid_mask)
        obj_contact_bce = reduce_loss_map(obj_bce_map, obj_valid_mask)
        cross_edge_bce = reduce_loss_map_per_object(edge_bce_map, edge_valid_mask, obj_valid_mask)
        obj_contact_mae = reduce_loss_map(obj_mae_map, obj_valid_mask)
        cross_edge_mae = reduce_loss_map_per_object(edge_mae_map, edge_valid_mask, obj_valid_mask)

        sampled_nonzero_mask = (edge_target > 0) & edge_valid_mask
        sampled_nonzero_edge_count = sampled_nonzero_mask.sum()
        valid_edge_count = edge_valid_mask.sum()
        sampled_nonzero_edge_fraction = sampled_nonzero_edge_count.float() / valid_edge_count.clamp(min=1).float()
        per_obj_has_nonzero = sampled_nonzero_mask.any(dim=-1) & obj_valid_mask
        num_valid_obj = obj_valid_mask.sum()
        object_nonzero_edge_coverage = per_obj_has_nonzero.sum().float() / num_valid_obj.clamp(min=1).float()

        losses = {
            "obj_contact": float(meta.loss_obj_contact_weight) * obj_contact_qfl,
            "cross_edge_contact": float(meta.loss_cross_edge_weight) * cross_edge_qfl,
        }
        aux_metrics = {
            "obj_contact_qfl": obj_contact_qfl,
            "cross_edge_qfl": cross_edge_qfl,
            "obj_contact_bce": obj_contact_bce,
            "cross_edge_bce": cross_edge_bce,
            "obj_contact_mae": obj_contact_mae,
            "cross_edge_mae": cross_edge_mae,
            "num_valid_obj": num_valid_obj.float(),
            "num_valid_edges": valid_edge_count.float(),
            "sampled_nonzero_edge_count": sampled_nonzero_edge_count.float(),
            "sampled_nonzero_edge_fraction": sampled_nonzero_edge_fraction,
            "object_nonzero_edge_coverage": object_nonzero_edge_coverage,
        }
        return losses, aux_metrics

    def inference(self, model: torch.nn.Module, inputs: Any) -> dict[str, torch.Tensor]:
        return model(self.prepare_batch(inputs))
