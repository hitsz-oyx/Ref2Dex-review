"""BaseRunner integration for CmAction temporal point-flow learning."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.Cm.dataset import make_dataloaders


def internal_flow_smooth_l1(
    pred_flow_m: torch.Tensor,
    gt_flow_m: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    beta_m: float,
    internal_scale: float,
) -> torch.Tensor:
    """Compute the training loss in Cm-head internal units.

    Public Cm inputs, predictions, and metrics remain metres.  Multiplying
    both endpoints and the Huber transition by the internal scale removes the
    reciprocal scale factor from the decoder's gradient.
    """
    scale = float(internal_scale)
    if scale <= 0.0:
        raise ValueError("internal_point_flow_scale must be positive.")
    valid_count = valid_mask.sum()
    pred_internal = pred_flow_m * scale
    gt_internal = gt_flow_m * scale
    smooth_l1_map = F.smooth_l1_loss(
        pred_internal,
        gt_internal,
        beta=float(beta_m) * scale,
        reduction="none",
    ).mean(dim=-1)
    return (smooth_l1_map * valid_mask.float()).sum() / valid_count.float()


class CmActionRunner(BaseRunner):
    def evaluate_all(self) -> dict[str, float]:
        metrics = super().evaluate_all()
        stride_mse = [value for key, value in metrics.items() if key.endswith("/flow_mse") and "/stride_" in key]
        if stride_mse:
            metrics["val/mean_stride_flow_mse"] = float(sum(stride_mse) / len(stride_mse))
        return metrics

    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(
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

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def step(
        self,
        model: torch.nn.Module,
        batch: dict[str, torch.Tensor],
        mode: str = "train",
    ) -> RunnerOutput:
        del mode
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
        flow_smooth_l1 = internal_flow_smooth_l1(
            pred_flow,
            gt_flow,
            valid,
            beta_m=float(self.cfg.meta.flow_smooth_l1_beta),
            internal_scale=float(self.cfg.meta.internal_point_flow_scale),
        )
        squared_map = (pred_flow - gt_flow).square().mean(dim=-1)
        absolute_map = (pred_flow - gt_flow).abs().mean(dim=-1)
        flow_mse = (squared_map * valid.float()).sum() / valid_count.float()
        flow_mae = (absolute_map * valid.float()).sum() / valid_count.float()
        gt_flow_norm = (torch.linalg.norm(gt_flow, dim=-1) * valid.float()).sum() / valid_count.float()
        pred_flow_norm = (torch.linalg.norm(pred_flow, dim=-1) * valid.float()).sum() / valid_count.float()
        cm_assignment = prediction["all_cm_assignment"].clamp_min(1e-8)
        # cm_assignment: [B, N, S]  每个物点分到 S 个 slot 的概率分布（已 clamp 防 log(0)）
        slot_assignment_entropy = -(cm_assignment * cm_assignment.log()).sum(dim=1).mean()
        # 沿 slot 维度求熵再对 N 求平均：惩罚「每个点都均匀分到所有 slot」的情况（鼓励点选最确定的 slot）
        cm_slot_weights = prediction["cm_slot_weights"]
        # cm_slot_weights: [B, S, D]  S = num_slots, D = slot 隐维度
        num_slots = cm_slot_weights.shape[1]
        if num_slots > 1:
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
        else:
            # 单 slot 占位：返 0 让下游代码不必做分支
            slot_weight_overlap = cm_slot_weights.new_zeros(())
        decoder_slot_usage = prediction["decoder_slot_usage"]
        # decoder_slot_usage: [B, S]  每个样本里每个 slot 被 decoder 实际使用到的程度（例如被分配到的 token 数 / 总数）
        decoder_null_usage = prediction["decoder_null_usage"]
        mean_decoder_slot_usage = decoder_slot_usage.mean(dim=0)
        # 沿 batch 维求平均：得到每个 slot 在整个 batch 上的平均使用率 → [S]
        full_decoder_usage = torch.cat([decoder_null_usage[:, None], decoder_slot_usage], dim=-1)
        decoder_slot_usage_entropy = -(
            full_decoder_usage.clamp_min(1e-8) * full_decoder_usage.clamp_min(1e-8).log()
        ).sum(dim=-1).mean()
        # 对 S 个 slot 的平均使用率求熵：鼓励各 slot 使用率接近均匀分布
        # 防止某些 slot 完全没被用上（collapse）
        slot_nonzero_prob = prediction["slot_nonzero_prob"]
        slot_count_loss = slot_nonzero_prob.mean()
        # Only pairs that are currently active contribute.  Slot weights are
        # probability distributions across hand points, so their cosine
        # similarity measures redundant hand-region assignment.
        if num_slots > 1:
            slot_gate = prediction["slot_gate"]
            active_pair_weight = slot_gate.unsqueeze(2) * slot_gate.unsqueeze(1)
            active_pair_weight = active_pair_weight * off_diagonal.unsqueeze(0)
            active_overlap_loss = (slot_similarity * active_pair_weight).sum() / active_pair_weight.sum().clamp_min(1e-8)
        else:
            active_overlap_loss = cm_slot_weights.new_zeros(())
        null_candidate_flow = prediction["null_candidate_flow"]
        null_zero_loss = internal_flow_smooth_l1(
            null_candidate_flow,
            torch.zeros_like(null_candidate_flow),
            valid,
            beta_m=float(self.cfg.meta.flow_smooth_l1_beta),
            internal_scale=float(self.cfg.meta.internal_point_flow_scale),
        )
        total_loss = (
            float(self.cfg.meta.loss_flow_weight) * flow_smooth_l1
            + float(self.cfg.meta.loss_slot_count_weight) * slot_count_loss
            + float(self.cfg.meta.loss_null_zero_weight) * null_zero_loss
            + float(self.cfg.meta.loss_active_overlap_weight) * active_overlap_loss
        )
        expected_active_count = slot_nonzero_prob.sum(dim=-1)
        sampled_active_count = (prediction["slot_gate"] > 0).sum(dim=-1).float()
        null_assignment_mass = prediction["null_assignment"].mean()
        null_candidate_flow_norm = (
            torch.linalg.norm(null_candidate_flow, dim=-1) * valid.float()
        ).sum() / valid_count.float()
        metrics: dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow_smooth_l1": flow_smooth_l1,
            "flow_mse": flow_mse,
            "flow_mae": flow_mae,
            "gt_flow_norm": gt_flow_norm,
            "pred_flow_norm": pred_flow_norm,
            "slot_assignment_entropy": slot_assignment_entropy,
            "slot_weight_overlap": slot_weight_overlap,
            "decoder_slot_usage_entropy": decoder_slot_usage_entropy,
            "decoder_slot_usage_max": mean_decoder_slot_usage.max(),
            "slot_count_loss": slot_count_loss,
            "null_zero_loss": null_zero_loss,
            "active_overlap_loss": active_overlap_loss,
            "slot/expected_active_mean": expected_active_count.mean(),
            "slot/sampled_active_mean": sampled_active_count.mean(),
            "slot/sampled_active_min": sampled_active_count.min(),
            "slot/sampled_active_max": sampled_active_count.max(),
            "slot/gate_probability_mean": slot_nonzero_prob.mean(),
            "slot/active_overlap": active_overlap_loss,
            "null/candidate_flow_norm": null_candidate_flow_norm,
            "null/decoder_usage": decoder_null_usage.mean(),
            "null/assignment_mass": null_assignment_mass,
            "valid_object_count": valid_count.float(),
        }
        metrics.update(
            {
                f"decoder_slot_usage/slot_{slot_idx:02d}": usage
                for slot_idx, usage in enumerate(mean_decoder_slot_usage)
            }
        )
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(pred_flow.shape[0]))
