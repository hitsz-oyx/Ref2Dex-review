"""BaseRunner integration for CmAction temporal point-flow learning."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, MetricStat, RunnerOutput, TaskConfig
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


class CmActionRunner(BaseRunner):
    def evaluate_all(self) -> dict[str, float]:
        detailed = super().evaluate_all()
        return {**detailed, **self._summarize_stride_metrics(detailed, split="val")}

    def evaluate_test_all(self) -> dict[str, float]:
        detailed = super().evaluate_test_all()
        return {**detailed, **self._summarize_stride_metrics(detailed, split="test")}

    @staticmethod
    def _summarize_stride_metrics(metrics: dict[str, float], *, split: str) -> dict[str, float]:
        """Keep a compact validation/test panel instead of 10× metric curves."""
        def values(name: str) -> list[float]:
            return [
                value for key, value in metrics.items()
                if key.startswith(f"{split}/stride_") and key.endswith(f"/{name}")
            ]

        epe = values("flow/epe_mm")
        relative_epe = values("flow/relative_epe")
        improvement = values("flow/zero_flow_improvement")
        norm_ratio = values("flow/norm_ratio")
        summary: dict[str, float] = {}
        if epe:
            summary[f"{split}/mean_stride_epe_mm"] = float(sum(epe) / len(epe))
        if relative_epe:
            summary[f"{split}/mean_stride_relative_epe"] = float(sum(relative_epe) / len(relative_epe))
        if improvement:
            summary[f"{split}/zero_flow_improvement"] = float(sum(improvement) / len(improvement))
        if norm_ratio:
            summary[f"{split}/norm_ratio"] = float(sum(norm_ratio) / len(norm_ratio))
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
            key = f"{split}/stride_{stride}/flow/epe_mm"
            if key in metrics:
                summary[f"{split}/stride_{stride}_epe_mm"] = metrics[key]
        return summary

    def evaluate_loader(self, loader: Any, *, prefix: str) -> dict[str, float]:
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
        return metrics

    def select_eval_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        """Keep JSONL exhaustive while limiting W&B validation curves."""
        keep = {
            key: value
            for key, value in metrics.items()
            if not (key.startswith("val/stride_") or key.startswith("test/stride_"))
        }
        for split in ("val", "test"):
            for stride in (1, 5, 10):
                key = f"{split}/stride_{stride}/flow/epe_mm"
                if key in metrics:
                    keep[key] = metrics[key]
        return keep

    def select_step_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        core_keys = (
            "loss", "flow/loss_scaled", "flow/epe_mm", "lr", "grad_norm",
            "slot/expected_active_mean", "slot/hard_active_mean", "slot/fallback_ratio",
            "slot/effective_branch_count", "slot/global_top1_usage", "slot/per_sample_top1_usage",
            "flow/epe_object_mm", "flow/epe_environment_mm", "flow/pred_norm_environment_mm",
        )
        aliases = {"lr": "optim/lr", "grad_norm": "optim/grad_norm"}
        return {
            aliases.get(key, key): metrics[key]
            for key in core_keys
            if key in metrics
        }

    def select_epoch_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        suffix = "train_epoch/"
        core_keys = (
            "loss", "flow/loss_scaled", "flow/epe_mm", "flow/gt_norm_mm", "flow/pred_norm_mm",
            "slot/expected_active_mean", "slot/hard_active_mean", "slot/fallback_ratio",
            "slot/effective_branch_count", "slot/global_top1_usage", "slot/per_sample_top1_usage",
            "slot/assignment_entropy",
            "slot/max_probability_mean", "slot/count_loss", "slot/confidence_loss",
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
        if f"{suffix}grad_clipped" in metrics:
            result["epoch/optim/grad_clipped_fraction"] = metrics[f"{suffix}grad_clipped"]
        if f"{suffix}grad_norm" in metrics:
            result["epoch/optim/grad_norm_mean"] = metrics[f"{suffix}grad_norm"]
        return result

    def make_dataloaders(self, data_cfg: Any, seed: int):
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
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or payload.get("schema_name") != "ref2dex_cm_scene_v1":
            return None
        from src.task.Cm.dataset_scene import make_dataloaders as make_scene_dataloaders

        return make_scene_dataloaders(
            data_cfg,
            seed,
            meta_cfg=self.cfg.meta,
            distributed=self.distributed,
        )

    def configure_data(self, metadata: dict[str, Any], train_dataset: Any | None = None) -> None:
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
                    f"{missing}. Run src.task.Cm.compute_flow_scale or explicitly "
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
        batch: dict[str, torch.Tensor],
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
        # EPE is invariant to an orthogonal change of xyz axes, unlike
        # component-wise MSE/MAE.  It is the sole flow-quality metric.
        residual_norm_map = torch.linalg.norm(pred_flow - gt_flow, dim=-1)
        gt_norm_map = torch.linalg.norm(gt_flow, dim=-1)
        pred_norm_map = torch.linalg.norm(pred_flow, dim=-1)
        residual_sum = (residual_norm_map * valid.float()).sum()
        gt_norm_sum = (gt_norm_map * valid.float()).sum()
        pred_norm_sum = (pred_norm_map * valid.float()).sum()
        # Scene V1 (V1.md §25): scene_source_id is diagnostics-only metadata
        # (0=object, 1=environment) and never enters the model.  Splitting the
        # EPE by source exposes a zero-flow collapse on environment points
        # that the overall EPE could otherwise hide.
        source_metrics: dict[str, MetricStat] = {}
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
        decoder_slot_usage = prediction["decoder_slot_usage"]
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
        confidence_loss = torch.relu(
            float(self.cfg.meta.slot_threshold) - max_slot_probability
        ).square().mean()
        total_loss = (
            float(self.cfg.meta.loss_flow_weight) * flow_smooth_l1
            + float(self.cfg.meta.loss_slot_count_weight) * slot_count_loss
            + float(self.cfg.meta.loss_slot_confidence_weight) * confidence_loss
            + float(self.cfg.meta.loss_active_overlap_weight) * active_overlap_loss
        )
        expected_active_count = slot_nonzero_prob.sum(dim=-1)
        sampled_active_count = prediction["slot_hard_mask"].sum(dim=-1).float()
        fallback_used = prediction["slot_fallback_used"]
        effective_branch_count = full_usage_entropy.exp().mean()
        metrics: dict[str, torch.Tensor | MetricStat] = {
            "loss": total_loss,
            "flow/loss_scaled": flow_smooth_l1,
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
        }
        metrics.update(source_metrics)
        if float(self.cfg.meta.loss_active_overlap_weight) != 0.0:
            metrics["slot/active_overlap"] = active_overlap_loss
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(pred_flow.shape[0]))
