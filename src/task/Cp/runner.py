"""BaseRunner integration for Cp effect learning and frozen-Cm closure learning."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput
from src.task.Cm.runner import _rotation_geodesic, _wrist_targets

from .dataset import make_dataloaders


def _cm_weighted_hand_loss(
    prediction: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    meta: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reuse Cm's wrist/articulation weighting for the frozen-decoder closure."""
    beta = float(meta.flow_smooth_l1_beta)
    target_rotation, target_translation, _, target_articulation = _wrist_targets(
        batch["hand_points"].float(), batch["hand_flow"].float(), batch["wrist_delta"].float()
    )
    wrist_translation_loss = F.smooth_l1_loss(prediction["pred_wrist_translation"], target_translation, beta=beta)
    wrist_rotation_error = _rotation_geodesic(prediction["pred_wrist_rotation"], target_rotation)
    wrist_rotation_loss = F.smooth_l1_loss(wrist_rotation_error, torch.zeros_like(wrist_rotation_error), beta=beta)
    wrist_loss = wrist_translation_loss + float(meta.wrist_rotation_weight_m_per_rad) * wrist_rotation_loss

    articulation_map = F.smooth_l1_loss(
        prediction["pred_hand_articulation_flow"], target_articulation, beta=beta, reduction="none"
    ).mean(dim=-1)
    hand_distance = batch["hand_to_obj_min_dist"].float()
    contact_weight = torch.exp(-0.5 * (hand_distance / float(meta.hand_contact_sigma_m)).square())
    articulation_weight = (
        torch.linalg.norm(target_articulation, dim=-1) / float(meta.hand_articulation_scale_m)
    ).clamp(0.0, 1.0)
    point_weight = (
        float(meta.hand_loss_base_weight)
        + float(meta.hand_loss_contact_weight) * contact_weight
        + float(meta.hand_loss_articulation_weight) * articulation_weight
    ).clamp(max=float(meta.hand_loss_max_weight))
    point_weight = point_weight / point_weight.mean(dim=1, keepdim=True).clamp_min(1e-6)
    weighted_articulation_loss = (articulation_map * point_weight).sum() / point_weight.sum().clamp_min(1e-6)
    global_articulation_loss = articulation_map.mean()
    loss = (
        float(meta.loss_wrist_weight) * wrist_loss
        + float(meta.loss_articulation_weight) * weighted_articulation_loss
        + float(meta.loss_global_articulation_weight) * global_articulation_loss
    )
    hand_epe = torch.linalg.norm(prediction["pred_hand_flow"] - batch["hand_flow"], dim=-1)
    return loss, {
        "hand_flow_epe_mm": hand_epe.mean() * 1000,
        "hand_flow_epe_p90_mm": torch.quantile(hand_epe, 0.9) * 1000,
        "wrist_loss": wrist_loss,
        "weighted_articulation_loss": weighted_articulation_loss,
        "global_articulation_loss": global_articulation_loss,
    }


class CpHumanClosureRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(data_cfg, seed, meta_cfg=self.cfg.meta, distributed=self.distributed)

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        model = self.build_model_from_config(model_cfg)
        if str(self.cfg.meta.stage) == "closed_loop":
            if not self.cfg.meta.cp_effect_checkpoint:
                raise ValueError("closed_loop requires meta.cp_effect_checkpoint.")
            model.load_cp_effect(self.cfg.meta.cp_effect_checkpoint)
        return model

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        del mode
        prediction = model(batch)
        stage = str(self.cfg.meta.stage)
        valid = batch["obj_valid_mask"].float()
        denominator = valid.sum().clamp_min(1)

        if stage == "cp_effect":
            flow_scale = float(self.cfg.meta.object_flow_scale_m)
            if flow_scale <= 0.0:
                raise ValueError("meta.object_flow_scale_m must be positive.")
            flow_map = F.smooth_l1_loss(
                prediction["pred_obj_flow"] / flow_scale,
                batch["obj_flow_gt"] / flow_scale,
                beta=1.0,
                reduction="none",
            ).mean(dim=-1)
            flow_loss = (flow_map * valid).sum() / denominator
            contact_map = F.binary_cross_entropy_with_logits(
                prediction["pred_obj_contact_logits"], batch["obj_contact_gt"], reduction="none"
            )
            contact_target = batch["obj_contact_gt"].float()
            contact_entropy_map = -(
                torch.xlogy(contact_target, contact_target)
                + torch.xlogy(1.0 - contact_target, 1.0 - contact_target)
            )
            contact_lower_bound = (contact_entropy_map * valid).sum() / denominator
            contact_excess_loss = ((contact_map - contact_entropy_map) * valid).sum() / denominator
            loss = (
                float(self.cfg.meta.loss_object_flow_weight) * flow_loss
                # Subtracting the target entropy is constant with respect to
                # the prediction, so this has the same gradients as BCE while
                # making zero the meaningful theoretical optimum.
                + float(self.cfg.meta.loss_object_contact_weight) * contact_excess_loss
            )
            metrics: dict[str, torch.Tensor] = {
                "loss": loss,
                "obj_flow_loss": flow_loss,
                "obj_flow_epe_mm": (
                    torch.linalg.norm(prediction["pred_obj_flow"] - batch["obj_flow_gt"], dim=-1) * valid
                ).sum() / denominator * 1000,
                "contact_bce_lower_bound": contact_lower_bound,
                "excess_contact_bce": contact_excess_loss.clamp_min(0.0),
            }
        elif stage == "closed_loop":
            cm_decoder = model.cm_hand_decoder
            if cm_decoder is None:
                raise RuntimeError("closed_loop model has no frozen Cm hand decoder.")
            loss, metrics = _cm_weighted_hand_loss(prediction, batch, cm_decoder.loss_meta)
            metrics["loss"] = loss
        else:
            raise ValueError(f"Unknown meta.stage={stage!r}.")
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=int(batch["hand_points"].shape[0]))
