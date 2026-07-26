"""BaseRunner integration for CmAction temporal point-flow learning."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput, TaskConfig
from src.task.Cm.dataset import make_dataloaders


def _wrist_targets(
    hand_points: torch.Tensor,
    hand_flow: torch.Tensor,
    wrist_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split total hand flow into the documented wrist-rigid and residual parts.

    Stage 4 stores ``T_hand_t<-hand_t1``.  With row-vector point storage the
    inverse mapping is ``y_t1 = (y_t - t) @ R``.
    """
    rotation = wrist_delta[:, :3, :3]
    translation = -torch.einsum("bi,bij->bj", wrist_delta[:, :3, 3], rotation)
    rigid_next = torch.einsum("bhi,bij->bhj", hand_points, rotation) + translation.unsqueeze(1)
    rigid_flow = rigid_next - hand_points
    articulation_flow = hand_flow - rigid_flow
    return rotation, translation, rigid_flow, articulation_flow


def _rotation_geodesic(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = prediction.transpose(-1, -2) @ target
    # ``acos`` has an infinite derivative at +/-1.  Exact or near-exact
    # rotations are common during single-example overfitting, so keep the
    # argument strictly inside the domain to avoid NaN gradients.
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(
        -1.0 + 1e-7,
        1.0 - 1e-7,
    )
    return torch.acos(cosine)


def _weighted_point_flow_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    point_weight: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    loss_map = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none").mean(dim=-1)
    return (loss_map * point_weight).sum() / point_weight.sum().clamp_min(1e-6)


class CmActionRunner(BaseRunner):
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
        model = self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)
        warm_start = getattr(self.cfg.meta, "warm_start_checkpoint", None)
        if warm_start:
            missing = model.load_legacy_warm_start(warm_start)
            if self.is_primary:
                print(f"[cm] warm-started from {warm_start}; new tensors: {missing}")
        if bool(getattr(self.cfg.meta, "freeze_cm_encoder", False)):
            trainable_prefixes = (
                "head.hand_context_encoder.",
                "head.hand_token_score.",
                "head.hand_articulation_decoder.",
                "head.wrist_decoder.",
                "mano_pose_head.",
            )
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(trainable_prefixes))
            if self.is_primary:
                print("[cm] frozen legacy Cm encoder/object decoder; training hand decoder only.")
        return model

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
        beta = float(self.cfg.meta.flow_smooth_l1_beta)
        smooth_l1_map = F.smooth_l1_loss(pred_flow, gt_flow, beta=beta, reduction="none").mean(dim=-1)
        flow_smooth_l1 = (smooth_l1_map * valid.float()).sum() / valid_count.float()
        squared_map = (pred_flow - gt_flow).square().mean(dim=-1)
        absolute_map = (pred_flow - gt_flow).abs().mean(dim=-1)
        flow_mse = (squared_map * valid.float()).sum() / valid_count.float()
        flow_mae = (absolute_map * valid.float()).sum() / valid_count.float()
        gt_flow_norm = (torch.linalg.norm(gt_flow, dim=-1) * valid.float()).sum() / valid_count.float()
        pred_flow_norm = (torch.linalg.norm(pred_flow, dim=-1) * valid.float()).sum() / valid_count.float()

        target_wrist_rotation, target_wrist_translation, target_rigid_flow, target_articulation_flow = _wrist_targets(
            batch["hand_points"].float(), batch["hand_flow"].float(), batch["wrist_delta"].float()
        )
        pred_wrist_translation = prediction["pred_wrist_translation"]
        wrist_translation_loss = F.smooth_l1_loss(
            pred_wrist_translation,
            target_wrist_translation,
            beta=beta,
        )
        wrist_rotation_error = _rotation_geodesic(prediction["pred_wrist_rotation"], target_wrist_rotation)
        wrist_rotation_loss = F.smooth_l1_loss(wrist_rotation_error, torch.zeros_like(wrist_rotation_error), beta=beta)
        wrist_loss = wrist_translation_loss + float(self.cfg.meta.wrist_rotation_weight_m_per_rad) * wrist_rotation_loss

        pred_articulation_flow = prediction["pred_hand_articulation_flow"]
        articulation_map = F.smooth_l1_loss(
            pred_articulation_flow, target_articulation_flow, beta=beta, reduction="none"
        ).mean(dim=-1)
        hand_distance = batch["hand_to_obj_min_dist"].float()
        contact_weight = torch.exp(-0.5 * (hand_distance / float(self.cfg.meta.hand_contact_sigma_m)).square())
        articulation_weight = torch.linalg.norm(target_articulation_flow, dim=-1)
        articulation_weight = (articulation_weight / float(self.cfg.meta.hand_articulation_scale_m)).clamp(0.0, 1.0)
        point_weight = (
            float(self.cfg.meta.hand_loss_base_weight)
            + float(self.cfg.meta.hand_loss_contact_weight) * contact_weight
            + float(self.cfg.meta.hand_loss_articulation_weight) * articulation_weight
        ).clamp(max=float(self.cfg.meta.hand_loss_max_weight))
        point_weight = point_weight / point_weight.mean(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_articulation_loss = (articulation_map * point_weight).sum() / point_weight.sum().clamp_min(1e-6)
        global_articulation_loss = articulation_map.mean()
        hand_loss = (
            float(self.cfg.meta.loss_wrist_weight) * wrist_loss
            + float(self.cfg.meta.loss_articulation_weight) * weighted_articulation_loss
            + float(self.cfg.meta.loss_global_articulation_weight) * global_articulation_loss
        )
        pred_hand_flow = prediction["pred_hand_flow"]
        hand_epe = torch.linalg.norm(pred_hand_flow - batch["hand_flow"].float(), dim=-1)
        contact_denominator = contact_weight.sum().clamp_min(1e-6)
        hand_epe_contact = (hand_epe * contact_weight).sum() / contact_denominator
        mano_flow_loss = pred_flow.new_zeros(())
        mano_consistency_loss = pred_flow.new_zeros(())
        mano_pose_loss = pred_flow.new_zeros(())
        mano_current_fit_mm = pred_flow.new_zeros(())
        mano_epe_mm = pred_flow.new_zeros(())
        if bool(getattr(self.cfg.meta, "use_mano_aux", False)):
            pred_mano_flow = prediction["pred_mano_flow"]
            pred_mano_articulation_flow = prediction["pred_mano_articulation_flow"]
            mano_flow_loss = _weighted_point_flow_loss(
                pred_mano_flow,
                batch["hand_flow"].float(),
                point_weight,
                beta=beta,
            )
            mano_consistency_loss = _weighted_point_flow_loss(
                prediction["pred_hand_articulation_flow"],
                pred_mano_articulation_flow.detach(),
                point_weight,
                beta=beta,
            )
            target_delta_pose = batch["next_mano_hand_pose"].float() - batch["mano_hand_pose"].float()
            mano_pose_loss = F.smooth_l1_loss(
                prediction["pred_mano_delta_pose"], target_delta_pose, beta=beta
            )
            mano_current_fit_mm = (
                torch.linalg.norm(prediction["pred_mano_current_points"] - batch["hand_points"].float(), dim=-1).mean()
                * 1000.0
            )
            mano_epe_mm = torch.linalg.norm(pred_mano_flow - batch["hand_flow"].float(), dim=-1).mean() * 1000.0
        cm_assignment = prediction["cm_assignment"].clamp_min(1e-8)
        slot_assignment_entropy = -(cm_assignment * cm_assignment.log()).sum(dim=1).mean()
        cm_slot_weights = prediction["cm_slot_weights"]
        num_slots = cm_slot_weights.shape[1]
        if num_slots > 1:
            normalized_weights = torch.nn.functional.normalize(cm_slot_weights, dim=-1, eps=1e-8)
            slot_similarity = normalized_weights @ normalized_weights.transpose(1, 2)
            off_diagonal = ~torch.eye(num_slots, device=slot_similarity.device, dtype=torch.bool)
            slot_weight_overlap = slot_similarity[:, off_diagonal].mean()
        else:
            slot_weight_overlap = cm_slot_weights.new_zeros(())
        decoder_slot_usage = prediction["decoder_slot_usage"]
        mean_decoder_slot_usage = decoder_slot_usage.mean(dim=0)
        decoder_slot_usage_entropy = -(
            mean_decoder_slot_usage.clamp_min(1e-8)
            * mean_decoder_slot_usage.clamp_min(1e-8).log()
        ).sum()
        total_loss = (
            float(self.cfg.meta.loss_object_weight) * flow_smooth_l1
            + float(self.cfg.meta.loss_hand_weight) * hand_loss
            + float(self.cfg.meta.loss_mano_flow_weight) * mano_flow_loss
            + float(self.cfg.meta.loss_mano_consistency_weight) * mano_consistency_loss
            + float(self.cfg.meta.loss_mano_pose_weight) * mano_pose_loss
        )
        metrics: dict[str, torch.Tensor] = {
            "loss": total_loss,
            "flow_smooth_l1": flow_smooth_l1,
            "flow_mse": flow_mse,
            "flow_mae": flow_mae,
            "gt_flow_norm": gt_flow_norm,
            "pred_flow_norm": pred_flow_norm,
            "hand_loss": hand_loss,
            "wrist_loss": wrist_loss,
            "wrist_translation_loss": wrist_translation_loss,
            "wrist_rotation_loss": wrist_rotation_loss,
            "weighted_articulation_loss": weighted_articulation_loss,
            "global_articulation_loss": global_articulation_loss,
            "hand_epe_mm": hand_epe.mean() * 1000.0,
            "hand_epe_contact_mm": hand_epe_contact * 1000.0,
            "mano_flow_loss": mano_flow_loss,
            "mano_consistency_loss": mano_consistency_loss,
            "mano_pose_loss": mano_pose_loss,
            "mano_current_fit_mm": mano_current_fit_mm,
            "mano_flow_epe_mm": mano_epe_mm,
            "wrist_translation_error_mm": torch.linalg.norm(
                pred_wrist_translation - target_wrist_translation, dim=-1
            ).mean() * 1000.0,
            "wrist_rotation_error_deg": wrist_rotation_error.mean() * (180.0 / torch.pi),
            "hand_contact_weight_mean": contact_weight.mean(),
            "hand_supervision_weight_mean": point_weight.mean(),
            "slot_assignment_entropy": slot_assignment_entropy,
            "slot_weight_overlap": slot_weight_overlap,
            "decoder_slot_usage_entropy": decoder_slot_usage_entropy,
            "decoder_slot_usage_max": mean_decoder_slot_usage.max(),
            "valid_object_count": valid_count.float(),
        }
        metrics.update(
            {
                f"decoder_slot_usage/slot_{slot_idx:02d}": usage
                for slot_idx, usage in enumerate(mean_decoder_slot_usage)
            }
        )
        return RunnerOutput(loss=total_loss, metrics=metrics, batch_size=int(pred_flow.shape[0]))
