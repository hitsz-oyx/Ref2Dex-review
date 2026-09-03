"""BaseRunner integration for CmAction temporal point-flow learning."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.base import BaseRunner, MetricStat, RunnerOutput, TaskConfig, resolve_optimizer, unwrap_model
from src.task.Cm.dataset import make_dataloaders


def scaled_flow_smooth_l1(
    pred_flow_m: torch.Tensor,
    gt_flow_m: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta_m: float,
    target_scale: float,
) -> torch.Tensor:
    """Compute vector Huber loss in normalized object-flow target units.

    Public predictions and metrics remain metres.  Scaling both endpoints and
    the Huber transition restores the FlowHead's normalized target space
    without shrinking the gradient through its metres-valued public output.
    """
    scale = float(target_scale)
    if scale <= 0.0:
        raise ValueError("object_flow_target_scale must be positive.")
    valid_count = valid_mask.sum()
    residual_norm_internal = torch.linalg.vector_norm(
        (pred_flow_m - gt_flow_m) * scale,
        dim=-1,
    )
    smooth_l1_map = F.smooth_l1_loss(
        residual_norm_internal,
        torch.zeros_like(residual_norm_internal),
        beta=float(beta_m) * scale,
        reduction="none",
    )
    return (smooth_l1_map * valid_mask.float()).sum() / valid_count.float()


def internal_flow_smooth_l1(
    pred_flow_m: torch.Tensor,
    gt_flow_m: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta_m: float,
    internal_scale: float,
) -> torch.Tensor:
    """Compatibility alias for older callers using the former argument name."""
    return scaled_flow_smooth_l1(
        pred_flow_m,
        gt_flow_m,
        valid_mask,
        beta_m=beta_m,
        target_scale=internal_scale,
    )


def candidate_mixture_smooth_l1(
    candidate_flow_m: torch.Tensor,
    gt_flow_m: torch.Tensor,
    valid_mask: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    beta_m: float,
    target_scale: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Soft-minimize per-slot candidate flow losses.

    ``candidate_flow_m`` is ``[B, N, K, 3]`` and ``routing_weights`` is
    ``[B, N, K]``.  The returned responsibility tensor is detached only by
    callers that use it for diagnostics; the loss itself keeps gradients
    through both candidate flows and routing probabilities.
    """
    if temperature <= 0.0:
        raise ValueError("candidate_mixture_temperature must be positive.")
    if candidate_flow_m.ndim != 4 or routing_weights.ndim != 3:
        raise ValueError("Candidate mixture expects [B,N,K,3] flows and [B,N,K] routing weights.")
    if candidate_flow_m.shape[:-1] != routing_weights.shape:
        raise ValueError("Candidate flow and routing weight shapes are inconsistent.")
    scale = float(target_scale)
    if scale <= 0.0:
        raise ValueError("object_flow_target_scale must be positive.")
    valid_count = valid_mask.sum()
    residual_norm_internal = torch.linalg.vector_norm(
        (candidate_flow_m - gt_flow_m.unsqueeze(2)) * scale,
        dim=-1,
    )
    candidate_loss = F.smooth_l1_loss(
        residual_norm_internal,
        torch.zeros_like(residual_norm_internal),
        beta=float(beta_m) * scale,
        reduction="none",
    )
    log_pi = routing_weights.clamp_min(1e-8).log()
    responsibility_logits = log_pi - candidate_loss / float(temperature)
    mixture_loss_map = -float(temperature) * torch.logsumexp(responsibility_logits, dim=-1)
    mixture_loss = (mixture_loss_map * valid_mask.float()).sum() / valid_count.float()
    responsibilities = torch.softmax(responsibility_logits, dim=-1)
    return mixture_loss, responsibilities


def additive_slot_group_sparsity(
    contribution_flow_m: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    target_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return batch-invariant group sparsity and per-sample slot strengths.

    Each sample first averages the 3-D contribution norm over its own valid
    object points.  Summing those strengths over slots and then averaging the
    batch prevents the regularizer from shrinking when batch size increases.
    """
    if contribution_flow_m.ndim != 4 or contribution_flow_m.shape[-1] != 3:
        raise ValueError("Additive contributions must have shape [B,N,K,3].")
    if contribution_flow_m.shape[:2] != valid_mask.shape:
        raise ValueError("Contribution flow and valid-mask shapes are inconsistent.")
    scale = float(target_scale)
    if scale <= 0.0:
        raise ValueError("object_flow_target_scale must be positive.")
    contribution_norm_internal = torch.linalg.vector_norm(
        contribution_flow_m * scale,
        dim=-1,
    )
    valid_per_sample = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).float()
    per_slot_strength = (
        contribution_norm_internal * valid_mask.unsqueeze(-1).float()
    ).sum(dim=1) / valid_per_sample
    group_sparsity_loss = per_slot_strength.sum(dim=-1).mean()
    return group_sparsity_loss, per_slot_strength


class CmActionRunner(BaseRunner):
    def __init__(self, cfg: Any, mode: str = "train", checkpoint: str | Path | None = None,
                 device: str | None = None, build_data: bool = True) -> None:
        # ``train.resume`` is for continuing the same run.  Fine-tuning from a
        # completed base checkpoint needs a fresh output directory and fresh
        # optimizer/scheduler counters, so it uses the Cm-specific
        # ``train.init_checkpoint`` field instead.
        init_checkpoint = getattr(cfg.train, "init_checkpoint", None)
        super().__init__(cfg, mode=mode, checkpoint=checkpoint, device=device, build_data=build_data)
        if mode == "train" and build_data and init_checkpoint:
            self.load(init_checkpoint, load_optimizer=False, map_location=self.device)
            self.global_step = 0
            self.start_epoch = 0
            self.best_metric = None
            self.early_stopping_metric = None
            self.epochs_without_improvement = 0
            self.evals_without_improvement = 0
            if self.is_primary:
                self._log_line(
                    f"Initialized fine-tuning weights from {init_checkpoint}; "
                    "reset optimizer, scheduler, epoch and step counters."
                )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build a decoder-only optimizer for the explicit freeze-resume phase.

        Historical Cm runs retain BaseRunner's parameter registration.  The
        decoder-only phase opts in explicitly so old checkpoints/configs keep
        their original optimizer contract.
        """
        optimizer_cls = resolve_optimizer(self.cfg.train.optimizer)
        if bool(getattr(self.cfg.train, "decoder_only_resume", False)):
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        else:
            parameters = self.model.parameters()
        return optimizer_cls(parameters, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

    def _load_checkpoint_payload(self, checkpoint: Dict[str, Any], load_optimizer: bool) -> None:
        """Resume decoder weights/state while dropping stale DenseToken moments."""
        if not bool(getattr(self.cfg.train, "decoder_only_resume", False)):
            return super()._load_checkpoint_payload(checkpoint, load_optimizer=load_optimizer)
        self._require_model_ready()
        unwrap_model(self.model).load_state_dict(checkpoint["model"])
        # Deliberately skip optimizer/scaler state: the optimizer now contains
        # only trainable decoder parameters.  Keep the global step and epoch.
        self.global_step = int(checkpoint.get("step", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        if load_optimizer and self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            last_lr = getattr(self.scheduler, "_last_lr", None)
            if isinstance(last_lr, (list, tuple)) and len(last_lr) == len(self.optimizer.param_groups):
                for group, lr in zip(self.optimizer.param_groups, last_lr):
                    group["lr"] = float(lr)
            if self.is_primary:
                self._log_line(
                    "Decoder-only resume: skipped source optimizer/DenseToken moments; "
                    f"restored scheduler/global_step at {self.global_step}."
                )
        state = checkpoint.get("runner_state")
        self.best_metric = checkpoint.get("best_metric")
        self.load_state_dict(state or {})
        self._early_stopping_triggered = False

    def _gate_warmup_state(self, epoch: int) -> Tuple[bool, float, float, float]:
        """Return force-all, threshold, count weight, and ramp progress."""
        base_threshold = float(self.cfg.meta.slot_threshold)
        base_count_weight = float(self.cfg.meta.loss_slot_count_weight)
        if not bool(getattr(self.cfg.meta, "gate_warmup_enabled", False)):
            return False, base_threshold, base_count_weight, 1.0
        if not bool(getattr(self.cfg.meta, "use_slot_gate", True)):
            raise ValueError("gate_warmup_enabled requires meta.use_slot_gate=true.")
        full_epochs = int(getattr(self.cfg.meta, "gate_warmup_full_epochs", 0))
        ramp_epochs = int(getattr(self.cfg.meta, "gate_warmup_ramp_epochs", 0))
        if full_epochs < 0 or ramp_epochs < 2:
            raise ValueError("Gate warm-up requires full_epochs >= 0 and ramp_epochs >= 2.")
        if epoch < full_epochs:
            return True, 0.0, 0.0, 0.0
        ramp_index = epoch - full_epochs
        if ramp_index < ramp_epochs:
            progress = float(ramp_index) / float(ramp_epochs - 1)
            return False, base_threshold * progress, base_count_weight * progress, progress
        return False, base_threshold, base_count_weight, 1.0

    def _set_train_epoch(self, epoch: int) -> None:
        super()._set_train_epoch(epoch)
        force_all, threshold, count_weight, progress = self._gate_warmup_state(epoch)
        self._runtime_slot_count_weight = count_weight
        self._runtime_gate_warmup_progress = progress
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.set_slot_gate_runtime(force_all_slots=force_all, threshold=threshold)

    def evaluate_all(self) -> Dict[str, float]:
        detailed = super().evaluate_all()
        return {**detailed, **self._summarize_stride_metrics(detailed, split="val")}

    def evaluate_test_all(self) -> Dict[str, float]:
        detailed = super().evaluate_test_all()
        return {**detailed, **self._summarize_stride_metrics(detailed, split="test")}

    @staticmethod
    def _summarize_stride_metrics(metrics: Dict[str, float], *, split: str) -> Dict[str, float]:
        """Summarize fixed-stride metrics, optionally split by source bucket.

        Object-v2 historically used ``val/stride_1/flow/epe_mm``.  The
        HRDexDB fine-tune path uses ``val/stride_1/hrdexdb/flow/epe_mm`` so
        source-specific results remain visible.  Both forms are accepted and
        the global mean is weighted equally over the reported source/stride
        panels, matching the balanced three-source evaluation contract.
        """
        panels: Dict[str, Dict[str, List[float]]] = {}
        prefix = f"{split}/stride_"
        for key, value in metrics.items():
            if not key.startswith(prefix):
                continue
            remainder = key[len(prefix):]
            stride_text, separator, tail = remainder.partition("/")
            if not separator or not stride_text.isdigit():
                continue
            source, separator, metric_name = tail.partition("/")
            if not separator:
                continue
            # Legacy object-v2 loaders omit a source component.
            if metric_name not in {
                "flow/epe_mm", "flow/relative_epe", "flow/zero_flow_improvement", "flow/norm_ratio",
                "hand_flow/epe_mm", "hand_flow/relative_epe",
                "hand_flow/zero_flow_improvement", "hand_flow/norm_ratio",
            }:
                source, metric_name = "all", tail
            panels.setdefault(source, {}).setdefault(metric_name, []).append(float(value))

        def values(name: str) -> List[float]:
            return [value for panel in panels.values() for value in panel.get(name, [])]

        epe = values("flow/epe_mm")
        relative_epe = values("flow/relative_epe")
        improvement = values("flow/zero_flow_improvement")
        norm_ratio = values("flow/norm_ratio")
        summary: Dict[str, float] = {}
        if epe:
            summary[f"{split}/mean_stride_epe_mm"] = float(sum(epe) / len(epe))
        if relative_epe:
            summary[f"{split}/mean_stride_relative_epe"] = float(sum(relative_epe) / len(relative_epe))
        if improvement:
            summary[f"{split}/zero_flow_improvement"] = float(sum(improvement) / len(improvement))
        if norm_ratio:
            summary[f"{split}/norm_ratio"] = float(sum(norm_ratio) / len(norm_ratio))
        hand_epe = values("hand_flow/epe_mm")
        hand_relative_epe = values("hand_flow/relative_epe")
        hand_improvement = values("hand_flow/zero_flow_improvement")
        hand_norm_ratio = values("hand_flow/norm_ratio")
        if hand_epe:
            summary[f"{split}/mean_stride_hand_epe_mm"] = float(
                sum(hand_epe) / len(hand_epe)
            )
        if hand_relative_epe:
            summary[f"{split}/mean_stride_hand_relative_epe"] = float(
                sum(hand_relative_epe) / len(hand_relative_epe)
            )
        if hand_improvement:
            summary[f"{split}/hand_zero_flow_improvement"] = float(
                sum(hand_improvement) / len(hand_improvement)
            )
        if hand_norm_ratio:
            summary[f"{split}/hand_norm_ratio"] = float(
                sum(hand_norm_ratio) / len(hand_norm_ratio)
            )
        for source, panel in panels.items():
            if source == "all":
                continue
            if panel.get("flow/epe_mm"):
                summary[f"{split}/{source}/mean_stride_epe_mm"] = float(sum(panel["flow/epe_mm"]) / len(panel["flow/epe_mm"]))
            if panel.get("flow/relative_epe"):
                summary[f"{split}/{source}/mean_stride_relative_epe"] = float(sum(panel["flow/relative_epe"]) / len(panel["flow/relative_epe"]))
            if panel.get("flow/zero_flow_improvement"):
                summary[f"{split}/{source}/zero_flow_improvement"] = float(sum(panel["flow/zero_flow_improvement"]) / len(panel["flow/zero_flow_improvement"]))
            if panel.get("hand_flow/epe_mm"):
                summary[f"{split}/{source}/mean_stride_hand_epe_mm"] = float(
                    sum(panel["hand_flow/epe_mm"]) / len(panel["hand_flow/epe_mm"])
                )
            if panel.get("hand_flow/relative_epe"):
                summary[f"{split}/{source}/mean_stride_hand_relative_epe"] = float(
                    sum(panel["hand_flow/relative_epe"])
                    / len(panel["hand_flow/relative_epe"])
                )
            if panel.get("hand_flow/zero_flow_improvement"):
                summary[f"{split}/{source}/hand_zero_flow_improvement"] = float(
                    sum(panel["hand_flow/zero_flow_improvement"])
                    / len(panel["hand_flow/zero_flow_improvement"])
                )
        # Scene V1 diagnostics (V1.md §25): object vs environment must be
        # reported separately so a zero-flow collapse on environment points
        # cannot hide behind an attractive overall EPE.
        for name, key in (
            ("flow/epe_object_mm", "mean_stride_epe_object_mm"),
            ("flow/epe_environment_mm", "mean_stride_epe_environment_mm"),
            ("flow/pred_norm_environment_mm", "mean_stride_pred_norm_environment_mm"),
        ):
            panel = values(name)
            if panel:
                summary[f"{split}/{key}"] = float(sum(panel) / len(panel))
        for stride in (1, 5, 10):
            stride_values = [
                value for source, panel in panels.items()
                for value in panel.get("flow/epe_mm", [])
                if any(key == f"{split}/stride_{stride}/{source}/flow/epe_mm" for key in metrics)
            ]
            # The source-aware branch above needs a direct key scan to avoid
            # conflating values from different strides.
            direct = [
                float(value) for key, value in metrics.items()
                if key.startswith(f"{split}/stride_{stride}/") and key.endswith("/flow/epe_mm")
            ]
            if direct:
                summary[f"{split}/stride_{stride}_epe_mm"] = float(sum(direct) / len(direct))
            direct_hand = [
                float(value) for key, value in metrics.items()
                if key.startswith(f"{split}/stride_{stride}/")
                and key.endswith("/hand_flow/epe_mm")
            ]
            if direct_hand:
                summary[f"{split}/stride_{stride}_hand_epe_mm"] = float(
                    sum(direct_hand) / len(direct_hand)
                )
        return summary

    def evaluate_loader(self, loader: Any, *, prefix: str) -> Dict[str, float]:
        """Add ratios after point-weighted aggregation across one loader."""
        metrics = super().evaluate_loader(loader, prefix=prefix)
        epe_mm = metrics.get(f"{prefix}flow/epe_mm")
        gt_norm_mm = metrics.get(f"{prefix}flow/gt_norm_mm")
        pred_norm_mm = metrics.get(f"{prefix}flow/pred_norm_mm")
        if epe_mm is not None and gt_norm_mm is not None:
            relative_epe = epe_mm / max(gt_norm_mm, 1e-8)
            metrics[f"{prefix}flow/relative_epe"] = relative_epe
            metrics[f"{prefix}flow/zero_flow_improvement"] = 1.0 - relative_epe
        if pred_norm_mm is not None and gt_norm_mm is not None:
            metrics[f"{prefix}flow/norm_ratio"] = pred_norm_mm / max(gt_norm_mm, 1e-8)
        if gt_norm_mm is not None:
            metrics[f"{prefix}flow/zero_baseline_epe_mm"] = gt_norm_mm
        hand_epe_mm = metrics.get(f"{prefix}hand_flow/epe_mm")
        hand_gt_norm_mm = metrics.get(f"{prefix}hand_flow/gt_norm_mm")
        hand_pred_norm_mm = metrics.get(f"{prefix}hand_flow/pred_norm_mm")
        if hand_epe_mm is not None and hand_gt_norm_mm is not None:
            hand_relative_epe = hand_epe_mm / max(hand_gt_norm_mm, 1e-8)
            metrics[f"{prefix}hand_flow/relative_epe"] = hand_relative_epe
            metrics[f"{prefix}hand_flow/zero_flow_improvement"] = 1.0 - hand_relative_epe
        if hand_pred_norm_mm is not None and hand_gt_norm_mm is not None:
            metrics[f"{prefix}hand_flow/norm_ratio"] = (
                hand_pred_norm_mm / max(hand_gt_norm_mm, 1e-8)
            )
        if hand_gt_norm_mm is not None:
            metrics[f"{prefix}hand_flow/zero_baseline_epe_mm"] = hand_gt_norm_mm
        if any(hasattr(loader.dataset, name) for name in ("rows", "datasets")):
            metrics.update(self._evaluate_action_interventions(loader, prefix=prefix))
        return metrics

    def _evaluate_action_interventions(self, loader: Any, *, prefix: str) -> Dict[str, float]:
        """Measure action shuffle/reverse EPE without changing model inputs or loss."""
        totals = {"shuffle": [0.0, 0.0], "reverse": [0.0, 0.0]}
        dataset_totals: Dict[Tuple[str, str], List[float]] = {}
        was_training = self.model.training
        self.eval_mode()
        for batch in loader:
            batch = self.prepare_batch(batch)
            valid = batch["obj_valid_mask"].bool()
            dataset_ids = batch.get("dataset_id")
            for mode in ("shuffle", "reverse"):
                changed = dict(batch)
                changed["hand_flow"] = (
                    torch.roll(batch["hand_flow"], shifts=1, dims=0)
                    if mode == "shuffle" else -batch["hand_flow"]
                )
                with self.eval_context():
                    prediction = self.model(changed)["pred_obj_flow"]
                epe = torch.linalg.norm(prediction - batch["obj_flow_gt"].float(), dim=-1)
                totals[mode][0] += float((epe * valid.float()).sum())
                totals[mode][1] += float(valid.sum())
                if dataset_ids is not None:
                    for name in set(str(value) for value in dataset_ids):
                        sample_mask = torch.tensor([str(value) == name for value in dataset_ids], device=valid.device).unsqueeze(-1)
                        mask = valid & sample_mask
                        entry = dataset_totals.setdefault((mode, name), [0.0, 0.0])
                        entry[0] += float((epe * mask.float()).sum())
                        entry[1] += float(mask.sum())
        if was_training:
            self.train_mode()
        result = {
            f"{prefix}flow/action_{mode}_epe_mm": 1000.0 * values[0] / max(values[1], 1.0)
            for mode, values in totals.items()
        }
        for (mode, name), values in dataset_totals.items():
            key = name.lower().replace("/", "_").replace(" ", "_")
            result[f"{prefix}flow/action_{mode}_{key}_epe_mm"] = 1000.0 * values[0] / max(values[1], 1.0)
        return result

    def select_eval_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Keep JSONL exhaustive while limiting W&B validation curves."""
        keep = {
            key: value
            for key, value in metrics.items()
            if not (key.startswith("val/stride_") or key.startswith("test/stride_"))
        }
        for split in ("val", "test"):
            for stride in (1, 5, 10):
                for metric_name in ("flow/epe_mm", "hand_flow/epe_mm"):
                    key = f"{split}/stride_{stride}/{metric_name}"
                    if key in metrics:
                        keep[key] = metrics[key]
                    source_suffix = f"/{metric_name}"
                    source_prefix = f"{split}/stride_{stride}/"
                    for source_key, value in metrics.items():
                        if source_key.startswith(source_prefix) and source_key.endswith(source_suffix):
                            keep[source_key] = value
        return keep

    def select_step_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        core_keys = (
            "loss", "flow/loss_scaled", "flow/epe_mm",
            "hand_flow/loss_scaled", "hand_flow/epe_mm", "lr", "grad_norm",
            "slot/expected_active_mean", "slot/hard_active_mean", "slot/fallback_ratio",
            "slot/effective_branch_count", "slot/global_top1_usage", "slot/per_sample_top1_usage",
            "slot/gate_threshold", "slot/gate_force_all", "slot/gate_warmup_progress",
            "flow/epe_object_mm", "flow/epe_environment_mm", "flow/pred_norm_environment_mm",
        )
        aliases = {"lr": "optim/lr", "grad_norm": "optim/grad_norm"}
        return {
            aliases.get(key, key): metrics[key]
            for key in core_keys
            if key in metrics
        }

    def select_epoch_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        suffix = "train_epoch/"
        core_keys = (
            "loss", "flow/loss_scaled", "flow/epe_mm", "flow/gt_norm_mm", "flow/pred_norm_mm",
            "hand_flow/loss_scaled", "hand_flow/epe_mm",
            "hand_flow/gt_norm_mm", "hand_flow/pred_norm_mm",
            "slot/expected_active_mean", "slot/hard_active_mean", "slot/fallback_ratio",
            "slot/effective_branch_count", "slot/global_top1_usage", "slot/per_sample_top1_usage",
            "slot/assignment_entropy",
            "slot/max_probability_mean", "slot/count_loss", "slot/confidence_loss",
            "slot/gate_threshold", "slot/gate_force_all", "slot/gate_warmup_progress",
            "flow/epe_object_mm", "flow/gt_norm_object_mm",
            "flow/epe_environment_mm", "flow/gt_norm_environment_mm",
            "flow/pred_norm_environment_mm",
        )
        if float(self.cfg.meta.loss_active_overlap_weight) != 0.0:
            core_keys += ("slot/active_overlap",)
        result = {
            f"epoch/{key}": metrics[f"{suffix}{key}"]
            for key in core_keys
            if f"{suffix}{key}" in metrics
        }
        epe_mm = metrics.get(f"{suffix}flow/epe_mm")
        gt_norm_mm = metrics.get(f"{suffix}flow/gt_norm_mm")
        pred_norm_mm = metrics.get(f"{suffix}flow/pred_norm_mm")
        if epe_mm is not None and gt_norm_mm is not None:
            relative_epe = epe_mm / max(gt_norm_mm, 1e-8)
            result["epoch/flow/relative_epe"] = relative_epe
            result["epoch/flow/zero_flow_improvement"] = 1.0 - relative_epe
        if pred_norm_mm is not None and gt_norm_mm is not None:
            result["epoch/flow/norm_ratio"] = pred_norm_mm / max(gt_norm_mm, 1e-8)
        hand_epe_mm = metrics.get(f"{suffix}hand_flow/epe_mm")
        hand_gt_norm_mm = metrics.get(f"{suffix}hand_flow/gt_norm_mm")
        hand_pred_norm_mm = metrics.get(f"{suffix}hand_flow/pred_norm_mm")
        if hand_epe_mm is not None and hand_gt_norm_mm is not None:
            hand_relative_epe = hand_epe_mm / max(hand_gt_norm_mm, 1e-8)
            result["epoch/hand_flow/relative_epe"] = hand_relative_epe
            result["epoch/hand_flow/zero_flow_improvement"] = 1.0 - hand_relative_epe
        if hand_pred_norm_mm is not None and hand_gt_norm_mm is not None:
            result["epoch/hand_flow/norm_ratio"] = (
                hand_pred_norm_mm / max(hand_gt_norm_mm, 1e-8)
            )
        if f"{suffix}grad_clipped" in metrics:
            result["epoch/optim/grad_clipped_fraction"] = metrics[f"{suffix}grad_clipped"]
        if f"{suffix}grad_norm" in metrics:
            result["epoch/optim/grad_norm_mean"] = metrics[f"{suffix}grad_norm"]
        return result

    def make_dataloaders(self, data_cfg: Any, seed: int):
        if str(getattr(data_cfg, "finetune_mode", "")).lower() == "hrdexdb":
            from src.task.Cm.dataset.hrdexdb import make_dataloaders as make_hrdexdb_dataloaders
            return make_hrdexdb_dataloaders(
                data_cfg,
                seed,
                meta_cfg=self.cfg.meta,
                distributed=self.distributed,
            )
        scene_loader = self._resolve_scene_dataloaders(data_cfg, seed)
        if scene_loader is not None:
            return scene_loader
        return make_dataloaders(
            data_cfg,
            seed,
            meta_cfg=self.cfg.meta,
            distributed=self.distributed,
        )

    def _resolve_scene_dataloaders(self, data_cfg: Any, seed: int):
        """Dispatch to the Scene Cache V1 dataset when the root declares it.

        Detection is by the root ``meta.json`` schema; a scene root that then
        fails fingerprint validation raises loudly inside the scene loader
        (V1.md §22) instead of silently falling back to the NPZ dataset.
        """
        root_value = str(getattr(data_cfg, "root", "") or data_cfg.train_path or "").strip()
        if not root_value:
            return None
        meta_path = Path(root_value) / "meta.json"
        if not meta_path.is_file():
            # Combined object-v2 roots may contain dataset-specific children
            # (grab/ and arctic/) without duplicating a root meta.json.
            if list(Path(root_value).glob("**/shared/meta.json")):
                from src.task.Cm.dataset.object_v2 import make_dataloaders as make_object_dataloaders
                return make_object_dataloaders(
                    data_cfg, seed, meta_cfg=self.cfg.meta, distributed=self.distributed,
                )
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or payload.get("schema_name") != "ref2dex_cm_scene_v1_1":
            if payload.get("schema_name") != "ref2dex_cm_object_v2":
                return None
            from src.task.Cm.dataset.object_v2 import make_dataloaders as make_object_dataloaders

            return make_object_dataloaders(
                data_cfg,
                seed,
                meta_cfg=self.cfg.meta,
                distributed=self.distributed,
            )
        from src.task.Cm.dataset.scene import make_dataloaders as make_scene_dataloaders

        return make_scene_dataloaders(
            data_cfg,
            seed,
            meta_cfg=self.cfg.meta,
            distributed=self.distributed,
        )

    def configure_data(self, metadata: Dict[str, Any], train_dataset: Optional[Any] = None) -> None:
        super().configure_data(metadata, train_dataset)
        stage4_frame = str(metadata.get("coordinate_frame", ""))
        expected_frame = str(self.cfg.meta.coordinate_frame)
        if stage4_frame != expected_frame:
            raise ValueError(
                f"Stage 4 coordinate_frame={stage4_frame!r}, but Cm config expects "
                f"{expected_frame!r}. Regenerate Stage 4 or set meta.coordinate_frame explicitly."
            )
        if int(metadata.get("num_obj_points", self.cfg.meta.num_obj_points)) != int(self.cfg.meta.num_obj_points):
            raise ValueError("Stage 4 runtime object sample count does not match Cm config.")
        if int(metadata.get("num_hand_points", self.cfg.meta.num_hand_points)) != int(self.cfg.meta.num_hand_points):
            raise ValueError("Stage 4 hand point count does not match Cm config.")
        rms_m = self.cfg.meta.flow_target_rms_m
        target_scale = float(self.cfg.meta.object_flow_target_scale)
        if rms_m is not None:
            normalized_rms = float(rms_m) * target_scale
            if abs(normalized_rms - 1.0) > 1e-3:
                raise ValueError(
                    "flow_target_rms_m and object_flow_target_scale are inconsistent: "
                    f"{rms_m} * {target_scale} = {normalized_rms}."
                )
        hand_rms_m = getattr(self.cfg.meta, "hand_flow_target_rms_m", None)
        hand_target_scale = float(
            getattr(self.cfg.meta, "hand_flow_target_scale", 1.0)
        )
        if bool(getattr(self.cfg.meta, "use_hand_flow_decoder", False)):
            if hand_rms_m is None:
                raise ValueError(
                    "Hand-flow decoding requires meta.hand_flow_target_rms_m from "
                    "a train-only calibration."
                )
            normalized_hand_rms = float(hand_rms_m) * hand_target_scale
            if abs(normalized_hand_rms - 1.0) > 1e-3:
                raise ValueError(
                    "hand_flow_target_rms_m and hand_flow_target_scale are inconsistent: "
                    f"{hand_rms_m} * {hand_target_scale} = {normalized_hand_rms}."
                )
        metadata_scale = metadata.get("flow_target_scale")
        if metadata_scale is not None and not math.isclose(
            target_scale, float(metadata_scale), rel_tol=1e-5, abs_tol=0.0
        ):
            raise ValueError(
                "Cm config object_flow_target_scale does not match train metadata: "
                f"{target_scale} != {metadata_scale}. Recalibrate or update the config."
            )
        required_calibration_keys = {
            "flow_target_rms_m",
            "flow_target_scale",
            "statistics_split",
            "statistics_active_only",
            "statistics_num_obj_points",
            "statistics_stride_distribution",
            "statistics_stride_weighting",
            "statistics_point_weighting",
        }
        if bool(getattr(self.cfg.meta, "require_flow_calibration", False)):
            missing = sorted(required_calibration_keys.difference(metadata))
            if missing:
                raise ValueError(
                    "Missing Cm flow calibration metadata in the train root: "
                    f"{missing}. Run src.task.Cm.tools.data.compute_flow_scale or explicitly "
                    "set meta.require_flow_calibration=false for an ablation."
                )
        calibration_checks = {
            "statistics_split": ("train", str),
            "statistics_active_only": (bool(self.cfg.data.active_only), bool),
            "statistics_num_obj_points": (int(self.cfg.meta.num_obj_points), int),
            "statistics_stride_distribution": (
                f"uniform_{int(self.cfg.data.min_stride)}_to_{int(self.cfg.data.max_stride)}", str,
            ),
            "statistics_stride_weighting": ("equal_per_stride", str),
            "statistics_point_weighting": ("per_pair_capped_at_num_obj_points", str),
        }
        for key, (expected, cast) in calibration_checks.items():
            if key in metadata and cast(metadata[key]) != expected:
                raise ValueError(
                    f"Cm train calibration {key}={metadata[key]!r} does not match "
                    f"the current training setting {expected!r}. Recalibrate first."
                )

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def step(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        mode: str = "train",
    ) -> RunnerOutput:
        prediction = model(batch)
        valid = batch["obj_valid_mask"].bool()
        valid_count = valid.sum()
        if int(valid_count.detach().item()) <= 0:
            raise RuntimeError(
                "CmAction received a batch with no valid object candidate. "
                "Use data.active_only=true for flow training."
            )
        pred_flow = prediction["pred_obj_flow"]
        gt_flow = batch["obj_flow_gt"].float()
        flow_smooth_l1 = scaled_flow_smooth_l1(
            pred_flow,
            gt_flow,
            valid,
            beta_m=float(self.cfg.meta.flow_smooth_l1_beta),
            target_scale=float(self.cfg.meta.object_flow_target_scale),
        )
        candidate_mixture_weight = float(
            getattr(self.cfg.meta, "loss_candidate_mixture_weight", 0.0)
        )
        additive_slot_mode = bool(
            getattr(self.cfg.meta, "use_additive_slot_contributions", False)
        )
        if additive_slot_mode:
            slot_group_sparsity_loss, per_slot_strength = additive_slot_group_sparsity(
                prediction["slot_contribution_flow"],
                valid,
                target_scale=float(self.cfg.meta.object_flow_target_scale),
            )
        else:
            per_slot_strength = prediction["cm_tokens"].new_zeros(
                prediction["cm_tokens"].shape[:2]
            )
            slot_group_sparsity_loss = flow_smooth_l1.new_zeros(())
        if candidate_mixture_weight != 0.0:
            mixture_loss, mixture_responsibilities = candidate_mixture_smooth_l1(
                prediction["dynamic_candidate_flow"],
                gt_flow,
                valid,
                prediction["decoder_routing_weights"],
                beta_m=float(self.cfg.meta.flow_smooth_l1_beta),
                target_scale=float(self.cfg.meta.object_flow_target_scale),
                temperature=float(self.cfg.meta.candidate_mixture_temperature),
            )
        else:
            mixture_loss = flow_smooth_l1.new_zeros(())
            mixture_responsibilities = prediction["decoder_slot_usage"].unsqueeze(1)
        # EPE is invariant to an orthogonal change of xyz axes, unlike
        # component-wise MSE/MAE.  It is the sole flow-quality metric.
        residual_norm_map = torch.linalg.norm(pred_flow - gt_flow, dim=-1)
        gt_norm_map = torch.linalg.norm(gt_flow, dim=-1)
        pred_norm_map = torch.linalg.norm(pred_flow, dim=-1)
        residual_sum = (residual_norm_map * valid.float()).sum()
        gt_norm_sum = (gt_norm_map * valid.float()).sum()
        pred_norm_sum = (pred_norm_map * valid.float()).sum()
        use_hand_flow_decoder = bool(
            getattr(self.cfg.meta, "use_hand_flow_decoder", False)
        )
        if use_hand_flow_decoder:
            if "pred_hand_flow" not in prediction:
                raise RuntimeError(
                    "meta.use_hand_flow_decoder=true but the model did not return pred_hand_flow."
                )
            pred_hand_flow = prediction["pred_hand_flow"]
            gt_hand_flow = batch["hand_flow"].float()
            if pred_hand_flow.shape != gt_hand_flow.shape:
                raise ValueError(
                    "Cm hand-flow prediction/target shape mismatch: "
                    f"{tuple(pred_hand_flow.shape)} != {tuple(gt_hand_flow.shape)}"
                )
            hand_valid = torch.ones_like(gt_hand_flow[..., 0], dtype=torch.bool)
            hand_valid_count = hand_valid.sum()
            hand_flow_smooth_l1 = scaled_flow_smooth_l1(
                pred_hand_flow,
                gt_hand_flow,
                hand_valid,
                beta_m=float(self.cfg.meta.flow_smooth_l1_beta),
                target_scale=float(self.cfg.meta.hand_flow_target_scale),
            )
            hand_residual_norm_map = torch.linalg.norm(
                pred_hand_flow - gt_hand_flow, dim=-1
            )
            hand_gt_norm_map = torch.linalg.norm(gt_hand_flow, dim=-1)
            hand_pred_norm_map = torch.linalg.norm(pred_hand_flow, dim=-1)
            hand_residual_sum = hand_residual_norm_map.sum()
            hand_gt_norm_sum = hand_gt_norm_map.sum()
            hand_pred_norm_sum = hand_pred_norm_map.sum()
        else:
            hand_flow_smooth_l1 = flow_smooth_l1.new_zeros(())
        # Scene V1 (V1.md §25): scene_source_id is diagnostics-only metadata
        # (0=object, 1=environment) and never enters the model.  Splitting the
        # EPE by source exposes a zero-flow collapse on environment points
        # that the overall EPE could otherwise hide.
        source_metrics: Dict[str, MetricStat] = {}
        scene_source_id = batch.get("scene_source_id")
        if scene_source_id is not None:
            environment = scene_source_id.bool() & valid
            object_mask = (~scene_source_id.bool()) & valid
            for name, mask, norm_map in (
                ("flow/epe_object_mm", object_mask, residual_norm_map),
                ("flow/gt_norm_object_mm", object_mask, gt_norm_map),
                ("flow/epe_environment_mm", environment, residual_norm_map),
                ("flow/gt_norm_environment_mm", environment, gt_norm_map),
                ("flow/pred_norm_environment_mm", environment, pred_norm_map),
            ):
                count = float(mask.sum())
                if count > 0.0:
                    source_metrics[name] = MetricStat(
                        float((norm_map * 1000.0 * mask.float()).sum().detach()), count
                    )
        dataset_ids = batch.get("dataset_id")
        if dataset_ids is not None:
            for dataset_name in sorted(set(str(value) for value in dataset_ids)):
                sample_mask = torch.tensor(
                    [str(value) == dataset_name for value in dataset_ids],
                    dtype=torch.bool,
                    device=valid.device,
                ).unsqueeze(-1)
                mask = valid & sample_mask
                count = float(mask.sum())
                if count <= 0.0:
                    continue
                key = dataset_name.lower().replace("/", "_").replace(" ", "_")
                source_metrics[f"flow/epe_{key}_mm"] = MetricStat(
                    float((residual_norm_map * 1000.0 * mask.float()).sum().detach()), count
                )
                source_metrics[f"flow/gt_norm_{key}_mm"] = MetricStat(
                    float((gt_norm_map * 1000.0 * mask.float()).sum().detach()), count
                )
                if use_hand_flow_decoder:
                    hand_sample_mask = sample_mask.expand(-1, hand_valid.shape[1])
                    hand_count = float(hand_sample_mask.sum())
                    source_metrics[f"hand_flow/epe_{key}_mm"] = MetricStat(
                        float(
                            (hand_residual_norm_map * 1000.0 * hand_sample_mask.float())
                            .sum()
                            .detach()
                        ),
                        hand_count,
                    )
                    source_metrics[f"hand_flow/gt_norm_{key}_mm"] = MetricStat(
                        float(
                            (hand_gt_norm_map * 1000.0 * hand_sample_mask.float())
                            .sum()
                            .detach()
                        ),
                        hand_count,
                    )
        cm_assignment = prediction["cm_assignment"].clamp_min(1e-8)
        # cm_assignment: [B, N, S]  每个物点分到 S 个 slot 的概率分布（已 clamp 防 log(0)）
        slot_assignment_entropy = -(cm_assignment * cm_assignment.log()).sum(dim=1).mean()
        # 沿 slot 维度求熵再对 N 求平均：惩罚「每个点都均匀分到所有 slot」的情况（鼓励点选最确定的 slot）
        cm_slot_weights = prediction["cm_slot_weights"]
        # cm_slot_weights: [B, S, D]  S = num_slots, D = slot 隐维度
        num_slots = cm_slot_weights.shape[1]
        if num_slots > 1 and float(self.cfg.meta.loss_active_overlap_weight) != 0.0:
            # 多个 slot 才有「去相关」意义；= 1 时强行算会得到 mean([1.0])=1.0，物理上无意义
            normalized_weights = torch.nn.functional.normalize(cm_slot_weights, dim=-1, eps=1e-8)
            # 沿 D 维做 L2 归一化：每个 slot 向量变成单位向量，让后续内积 = cosine 相似度
            # eps=1e-8 防止某个 slot 全 0 时除零
            slot_similarity = normalized_weights @ normalized_weights.transpose(1, 2)
            # [B, S, D] @ [B, D, S]  →  [B, S, S]
            # 对 batch 内每个样本构造 S×S 的「slot vs slot」余弦相似度矩阵
            # 对角线 = 1（自身 vs 自身），矩阵对称
            off_diagonal = ~torch.eye(num_slots, device=slot_similarity.device, dtype=torch.bool)
            # 构造 off-diagonal mask：对角线 0、其它 1，用于排除「自身 vs 自身」的 1.0
            slot_weight_overlap = slot_similarity[:, off_diagonal].mean()
            # 用 bool mask 沿最后一维挑掉对角线 → [B, S*(S-1)]，再求平均 → 标量
            # ∈ [-1, 1]：当前只作为 metric 监控 slot 是否塌缩（同向 → 接近 1 / 反向 → 接近 -1），
            # 并没有被加到 total_loss 里（total_loss 现在只有 flow_smooth_l1 一项）
        decoder_slot_usage = prediction["slot_contribution_usage"]
        # decoder_slot_usage: [B, S]  每个样本里每个 slot 被 decoder 实际使用到的程度（例如被分配到的 token 数 / 总数）
        mean_decoder_slot_usage = decoder_slot_usage.mean(dim=0)
        # 沿 batch 维求平均：得到每个 slot 在整个 batch 上的平均使用率 → [S]
        full_usage_entropy = -(
            decoder_slot_usage.clamp_min(1e-8) * decoder_slot_usage.clamp_min(1e-8).log()
        ).sum(dim=-1)
        decoder_slot_usage_entropy = full_usage_entropy.mean()
        # 对 S 个 slot 的平均使用率求熵：鼓励各 slot 使用率接近均匀分布
        # 防止某些 slot 完全没被用上（collapse）
        slot_nonzero_prob = prediction["slot_nonzero_prob"]
        max_slot_probability = slot_nonzero_prob.max(dim=-1).values
        # A fallback guarantees one slot, so its probability is free; only
        # extra slots pay the L0-style count cost.
        slot_count_loss = (slot_nonzero_prob.sum(dim=-1) - max_slot_probability).mean()
        # Only compute this diagnostic when it affects optimization.
        if num_slots > 1 and float(self.cfg.meta.loss_active_overlap_weight) != 0.0:
            slot_gate = prediction["slot_gate"]
            active_pair_weight = slot_gate.unsqueeze(2) * slot_gate.unsqueeze(1)
            active_pair_weight = active_pair_weight * off_diagonal.unsqueeze(0)
            active_overlap_loss = (slot_similarity * active_pair_weight).sum() / active_pair_weight.sum().clamp_min(1e-8)
        else:
            active_overlap_loss = cm_slot_weights.new_zeros(())
        effective_threshold = prediction["slot_gate_threshold"]
        confidence_loss = torch.relu(effective_threshold - max_slot_probability).square().mean()
        slot_count_weight = float(getattr(
            self, "_runtime_slot_count_weight", self.cfg.meta.loss_slot_count_weight,
        ))
        total_loss = (
            float(self.cfg.meta.loss_flow_weight) * flow_smooth_l1
            + float(getattr(self.cfg.meta, "loss_hand_flow_weight", 0.0))
            * hand_flow_smooth_l1
            + candidate_mixture_weight * mixture_loss
            + float(getattr(
                self.cfg.meta, "loss_slot_group_sparsity_weight", 0.0
            )) * slot_group_sparsity_loss
            + slot_count_weight * slot_count_loss
            + float(self.cfg.meta.loss_slot_confidence_weight) * confidence_loss
            + float(self.cfg.meta.loss_active_overlap_weight) * active_overlap_loss
        )
        expected_active_count = slot_nonzero_prob.sum(dim=-1)
        sampled_active_count = prediction["slot_hard_mask"].sum(dim=-1).float()
        fallback_used = prediction["slot_fallback_used"]
        effective_branch_count = full_usage_entropy.exp().mean()
        metrics: Dict[str, Union[torch.Tensor, MetricStat]] = {
            "loss": total_loss,
            "flow/loss_scaled": flow_smooth_l1,
            "flow/mixture_loss_scaled": mixture_loss,
            "slot/group_sparsity_loss": slot_group_sparsity_loss,
            "flow/epe_mm": MetricStat(float((residual_sum * 1000.0).detach()), float(valid_count.detach())),
            "flow/gt_norm_mm": MetricStat(float((gt_norm_sum * 1000.0).detach()), float(valid_count.detach())),
            "flow/pred_norm_mm": MetricStat(float((pred_norm_sum * 1000.0).detach()), float(valid_count.detach())),
            "slot/expected_active_mean": expected_active_count.mean(),
            "slot/hard_active_mean": sampled_active_count.mean(),
            "slot/fallback_ratio": fallback_used.float().mean(),
            "slot/max_probability_mean": max_slot_probability.mean(),
            "slot/effective_branch_count": effective_branch_count,
            "slot/global_top1_usage": mean_decoder_slot_usage.max(),
            "slot/per_sample_top1_usage": decoder_slot_usage.max(dim=-1).values.mean(),
            "slot/assignment_entropy": slot_assignment_entropy,
            "slot/count_loss": slot_count_loss,
            "slot/confidence_loss": confidence_loss,
            "slot/gate_threshold": effective_threshold,
            "slot/gate_force_all": prediction["slot_gate_force_all"],
            "slot/gate_warmup_progress": cm_slot_weights.new_tensor(float(getattr(
                self, "_runtime_gate_warmup_progress", 1.0,
            ))),
        }
        if use_hand_flow_decoder:
            metrics.update(
                {
                    "hand_flow/loss_scaled": hand_flow_smooth_l1,
                    "hand_flow/epe_mm": MetricStat(
                        float((hand_residual_sum * 1000.0).detach()),
                        float(hand_valid_count.detach()),
                    ),
                    "hand_flow/gt_norm_mm": MetricStat(
                        float((hand_gt_norm_sum * 1000.0).detach()),
                        float(hand_valid_count.detach()),
                    ),
                    "hand_flow/pred_norm_mm": MetricStat(
                        float((hand_pred_norm_sum * 1000.0).detach()),
                        float(hand_valid_count.detach()),
                    ),
                }
            )
        if candidate_mixture_weight != 0.0:
            valid_for_resp = valid.unsqueeze(-1).float()
            resp_usage = (mixture_responsibilities * valid_for_resp).sum(dim=1)
            resp_usage = resp_usage / valid_for_resp.sum(dim=1).clamp_min(1.0)
            resp_entropy = -(
                mixture_responsibilities.clamp_min(1e-8)
                * mixture_responsibilities.clamp_min(1e-8).log()
            ).sum(dim=-1)
            resp_entropy_global = -(
                resp_usage.clamp_min(1e-8) * resp_usage.clamp_min(1e-8).log()
            ).sum(dim=-1)
            resp_effective_count = resp_entropy_global.exp().mean()
            metrics.update({
                "slot/mixture_responsibility_entropy": (
                    (resp_entropy * valid.float()).sum() / valid_count.float()
                ),
                "slot/mixture_effective_branch_count": resp_effective_count,
                "slot/mixture_global_top1_usage": resp_usage.mean(dim=0).max(),
                "slot/mixture_per_sample_top1_usage": resp_usage.max(dim=-1).values.mean(),
            })
        if additive_slot_mode:
            usage_entropy = -(
                decoder_slot_usage.clamp_min(1e-8)
                * decoder_slot_usage.clamp_min(1e-8).log()
            ).sum(dim=-1)
            metrics.update({
                "slot/contribution_effective_branch_count": usage_entropy.exp().mean(),
                "slot/contribution_global_top1_usage": decoder_slot_usage.mean(dim=0).max(),
                "slot/contribution_per_sample_top1_usage": decoder_slot_usage.max(dim=-1).values.mean(),
                "slot/contribution_strength_mean": per_slot_strength.mean(),
            })
        metrics.update(source_metrics)
        if float(self.cfg.meta.loss_active_overlap_weight) != 0.0:
            metrics["slot/active_overlap"] = active_overlap_loss
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(pred_flow.shape[0]))
