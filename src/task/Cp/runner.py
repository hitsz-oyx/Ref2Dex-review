"""BaseRunner integration for Cp's three sequential training stages."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base import BaseRunner, RunnerOutput

from .dataset import make_dataloaders


class CpHumanClosureRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(data_cfg, seed, meta_cfg=self.cfg.meta, distributed=self.distributed)

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        model = self.build_model_from_config(model_cfg)
        if str(self.cfg.meta.stage) == "closed_loop":
            if not self.cfg.meta.hand_decoder_checkpoint or not self.cfg.meta.cp_effect_checkpoint:
                raise ValueError("closed_loop requires hand_decoder_checkpoint and cp_effect_checkpoint.")
            model.load_stage(self.cfg.meta.hand_decoder_checkpoint, ("hand_decoder",))
            model.load_stage(self.cfg.meta.cp_effect_checkpoint, ("cp_encoder", "cp_slots", "cp_decode"))
        return model

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        del mode
        prediction = model(batch)
        stage = str(self.cfg.meta.stage)
        beta = float(self.cfg.meta.flow_smooth_l1_beta)

        if stage == "hand_decoder":
            predicted_hand_flow = prediction["pred_hand_flow_teacher"]
            loss = F.smooth_l1_loss(predicted_hand_flow, batch["hand_flow"], beta=beta)
            extras: dict[str, torch.Tensor] = {}
        elif stage == "cp_effect":
            valid = batch["obj_valid_mask"].float()
            denominator = valid.sum().clamp_min(1)
            flow_map = F.smooth_l1_loss(prediction["pred_obj_flow"], batch["obj_flow_gt"], beta=beta, reduction="none").mean(-1)
            flow_loss = (flow_map * valid).sum() / denominator
            contact_map = F.binary_cross_entropy_with_logits(prediction["pred_obj_contact_logits"], batch["obj_contact_gt"], reduction="none")
            contact_loss = (contact_map * valid).sum() / denominator
            loss = flow_loss + contact_loss
            predicted_hand_flow = prediction["pred_hand_flow"]
            extras = {"obj_flow_epe_mm": (torch.linalg.norm(prediction["pred_obj_flow"] - batch["obj_flow_gt"], dim=-1) * valid).sum() / denominator * 1000, "obj_contact_bce": contact_loss}
        elif stage == "closed_loop":
            predicted_hand_flow = prediction["pred_hand_flow"]
            loss = F.smooth_l1_loss(predicted_hand_flow, batch["hand_flow"], beta=beta)
            extras = {}
        else:
            raise ValueError(f"Unknown meta.stage={stage!r}.")

        hand_epe = torch.linalg.norm(predicted_hand_flow - batch["hand_flow"], dim=-1)
        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "hand_flow_epe_mm": hand_epe.mean() * 1000,
            "hand_flow_epe_p90_mm": torch.quantile(hand_epe, 0.9) * 1000,
            **extras,
        }
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=int(batch["hand_points"].shape[0]))
