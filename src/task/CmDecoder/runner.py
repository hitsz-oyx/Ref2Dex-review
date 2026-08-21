from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base.base_runner import BaseRunner, RunnerOutput
from src.task.CmDecoder.dataset import make_dataloaders


class CmDecoderRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(data_cfg, seed, meta_cfg=self.cfg.meta, distributed=self.distributed)

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)

    def prepare_batch(self, batch: Any) -> Any:
        return {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        prediction = model(batch)
        residual = prediction["pred_q_next"] - batch["q_next"]
        prediction_target = str(getattr(self.cfg.meta, "prediction_target", "q_next"))
        if prediction_target == "delta_q":
            target = batch["q_next"] - batch["q_t"]
        elif prediction_target == "q_next":
            target = batch["q_next"]
        else:
            raise ValueError(f"Unsupported prediction_target: {prediction_target}")
        loss = F.smooth_l1_loss(
            prediction["pred_target_scaled"], target * float(self.cfg.meta.q_target_scale),
            beta=float(self.cfg.meta.q_loss_beta_rad) * float(self.cfg.meta.q_target_scale)
        )
        mae_rad = residual.abs().mean()
        identity_mae_rad = (batch["q_t"] - batch["q_next"]).abs().mean()
        metrics = {
            "loss": loss,
            "q/mae_rad": mae_rad,
            "q/mae_deg": mae_rad * (180.0 / 3.141592653589793),
            "q/rmse_rad": residual.square().mean().sqrt(),
            "identity/mae_deg": identity_mae_rad * (180.0 / 3.141592653589793),
        }
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=int(batch["q_t"].shape[0]))

    def select_step_metrics(self, metrics: dict[str, float]):
        return {key: value for key, value in metrics.items() if key in {"loss", "q/mae_deg", "q/mae_rad", "identity/mae_deg"}}
