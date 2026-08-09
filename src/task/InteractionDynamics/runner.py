"""BaseRunner integration and V1 trajectory metrics."""
from __future__ import annotations

from typing import Any

import torch

from src.base import BaseRunner, MetricStat, RunnerOutput
from src.base.data import make_file_split_dataloaders
from src.base.checkpoint import unwrap_model
from src.task.InteractionDynamics.dataset import (
    InteractionDynamicsDataset, ensure_sequence_disjoint, sequence_group_key,
)
from src.task.InteractionDynamics.uni3d import gather_points


def masked_effect_mse(pred_m: torch.Tensor, gt_m: torch.Tensor, valid: torch.Tensor,
                      motion_scale: float = 100.0) -> torch.Tensor:
    mask = valid[:, None, :, None].expand_as(pred_m)
    if not mask.any():
        raise RuntimeError("InteractionDynamics batch has no valid effect points")
    return ((pred_m - gt_m) * motion_scale).square()[mask].mean()


def patch_motion_target(displacement_m: torch.Tensor, knn_idx: torch.Tensor,
                        motion_scale: float = 100.0) -> torch.Tensor:
    """Pool stable-index point trajectories into patch trajectories, in internal cm units."""
    return torch.stack([gather_points(displacement_m[:, step], knn_idx).mean(2)
                        for step in range(displacement_m.shape[1])], 1) * motion_scale


def trajectory_statistics(pred_m: torch.Tensor, gt_m: torch.Tensor,
                          valid: torch.Tensor) -> dict[str, torch.Tensor]:
    error_mm = torch.linalg.vector_norm(pred_m - gt_m, dim=-1) * 1000.0
    mask = valid[:, None].expand_as(error_mm)
    per_step = [(error_mm[:, step] * valid).sum() / valid.sum() for step in range(error_mm.shape[1])]
    result = {"object/ade_mm": error_mm[mask].mean(), "object/fde_mm": per_step[-1]}
    result.update({f"object/epe_step_{step + 1}_mm": value for step, value in enumerate(per_step)})
    return result


def collapse_statistics(tokens: torch.Tensor, attention: torch.Tensor) -> dict[str, torch.Tensor]:
    centered = tokens - tokens.mean(1, keepdim=True)
    normalized = torch.nn.functional.normalize(tokens, dim=-1)
    similarity = normalized @ normalized.transpose(1, 2)
    off_diagonal = ~torch.eye(tokens.shape[1], dtype=torch.bool, device=tokens.device)
    probability = attention.mean(1).clamp_min(1e-8)
    entropy = -(probability * probability.log()).sum(-1).mean()
    return {"interaction/feature_variance": centered.square().mean(),
            "interaction/pairwise_cosine": similarity[:, off_diagonal].mean(),
            "interaction/l2_norm": tokens.norm(dim=-1).mean(),
            "interaction/object_to_action_entropy": entropy}


class InteractionDynamicsRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        common = {"dominant_hand_manifest": data_cfg.dominant_hand_manifest,
                  "num_effect_points": self.cfg.meta.num_effect_points,
                  "chunk_len": self.cfg.meta.chunk_len,
                  "temporal_stride": self.cfg.meta.temporal_stride, "base_seed": seed,
                  "max_samples_per_sequence": getattr(data_cfg, "max_samples_per_sequence", None)}
        loaders = make_file_split_dataloaders(
            data_cfg, seed, dataset_cls=InteractionDynamicsDataset, file_pattern="**/*.npz",
            train_dataset_kwargs={**common,
                "max_samples": getattr(data_cfg, "max_train_samples", None),
                "min_object_effect_norm": getattr(data_cfg, "min_object_effect_norm", 0.0)},
            val_dataset_kwargs={**common, "max_samples": getattr(data_cfg, "max_val_samples", None)},
            test_dataset_kwargs=common,
            split_group_fn=sequence_group_key, distributed=self.distributed)
        train, val, test, metadata = loaders
        ensure_sequence_disjoint(train.dataset.file_paths,
            [] if val is None else val.dataset.file_paths, [] if test is None else test.dataset.file_paths)
        val_loaders = {} if val is None else {"val/": val}
        test_loaders = {} if test is None else {"test/": test}
        return train, val, test, metadata, val_loaders, test_loaders

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        model = unwrap_model(self.model)
        uni3d = list(model.world.transformer.parameters())
        uni3d_ids = {id(parameter) for parameter in uni3d}
        new = [parameter for parameter in model.parameters()
               if parameter.requires_grad and id(parameter) not in uni3d_ids]
        return torch.optim.AdamW([
            {"params": new, "lr": float(self.cfg.train.lr)},
            {"params": uni3d, "lr": float(self.cfg.train.uni3d_lr)},
        ], weight_decay=float(self.cfg.train.weight_decay))

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        del mode
        prediction = model(batch)
        valid = batch["effect_obj_valid_mask"].bool()
        gt = batch["effect_obj_disp_gt"].float()
        pred = prediction["pred_obj_disp_chunk"].float()
        scale = float(self.cfg.meta.motion_scale)
        effect_loss = masked_effect_mse(pred, gt, valid, scale)
        action_target = patch_motion_target(batch["hand_disp_chunk"].float(),
                                            prediction["hand_knn_idx"], scale)
        patch_effect_target = patch_motion_target(batch["obj_disp_chunk_gt"].float(),
                                                  prediction["obj_knn_idx"], scale)
        action_loss = torch.nn.functional.mse_loss(
            prediction["pred_hand_patch_disp_internal"].float(), action_target)
        patch_effect_loss = torch.nn.functional.mse_loss(
            prediction["pred_obj_patch_disp_internal"].float(), patch_effect_target)
        loss = (effect_loss + float(self.cfg.train.action_loss_weight) * action_loss
                + float(self.cfg.train.patch_effect_loss_weight) * patch_effect_loss)
        metrics: dict[str, Any] = {"loss": loss, "loss/effect": effect_loss,
                                  "loss/action": action_loss,
                                  "loss/patch_effect": patch_effect_loss}
        metrics.update(trajectory_statistics(pred, gt, valid))
        metrics.update(collapse_statistics(prediction["interaction_tokens"],
                                           prediction["object_to_action_attention"]))
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=pred.shape[0])
