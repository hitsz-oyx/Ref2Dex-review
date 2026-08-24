from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.base.base_runner import BaseRunner, RunnerOutput
from src.task.CmDecoder.dataset import make_dataloaders
from src.task.CmDecoder.q_optimizer import (
    DifferentiableInspireHand,
    reconstruct_hand_points,
    rotvec_to_matrix,
)


class CmDecoderRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        return make_dataloaders(data_cfg, seed, meta_cfg=self.cfg.meta, distributed=self.distributed)

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        model = self.build_model_from_config(model_cfg, condition_shape=None, target_shape=None)
        self.baseline_hand_model = None
        if bool(getattr(self.cfg.meta, "predict_wrist_motion", False)) and float(
            getattr(self.cfg.meta, "baseline_point_loss_weight", 0.0)
        ) > 0.0:
            self.baseline_hand_model = DifferentiableInspireHand(
                self.cfg.meta.robot_urdf,
                num_hand_points=int(self.cfg.meta.num_hand_points),
                sample_seed=int(self.cfg.meta.sample_seed),
                device=self.device,
            )
        return model

    def prepare_batch(self, batch: Any) -> Any:
        return {key: value.to(self.device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode: str = "train") -> RunnerOutput:
        prediction = model(batch)
        if "pred_hand_flow" in prediction:
            scale = float(getattr(self.cfg.meta, "point_flow_target_scale", 1.0))
            beta = float(getattr(self.cfg.meta, "point_flow_loss_beta_m", 0.01)) * scale
            residual = prediction["pred_hand_flow"] - batch["hand_flow"]
            loss = F.smooth_l1_loss(
                prediction["pred_hand_flow_scaled"], batch["hand_flow"] * scale, beta=beta
            )
            epe = torch.linalg.vector_norm(residual, dim=-1).mean()
            return RunnerOutput(
                loss=loss,
                metrics={
                    "loss": loss,
                    "hand_flow/epe_m": epe,
                    "hand_flow/epe_mm": epe * 1000.0,
                    "hand_flow/rmse_m": residual.square().mean().sqrt(),
                    "zero_flow/epe_mm": torch.linalg.vector_norm(batch["hand_flow"], dim=-1).mean() * 1000.0,
                },
                batch_size=int(batch["hand_flow"].shape[0]),
            )
        residual = prediction["pred_q_next"] - batch["q_next"]
        prediction_target = str(getattr(self.cfg.meta, "prediction_target", "q_next"))
        if prediction_target == "delta_q":
            target = batch["q_next"] - batch["q_t"]
        elif prediction_target == "q_next":
            target = batch["q_next"]
        else:
            raise ValueError(f"Unsupported prediction_target: {prediction_target}")
        q_loss = F.smooth_l1_loss(
            prediction["pred_target_scaled"], target * float(self.cfg.meta.q_target_scale),
            beta=float(self.cfg.meta.q_loss_beta_rad) * float(self.cfg.meta.q_target_scale)
        )
        loss = float(getattr(self.cfg.meta, "q_loss_weight", 1.0)) * q_loss
        mae_rad = residual.abs().mean()
        identity_mae_rad = (batch["q_t"] - batch["q_next"]).abs().mean()
        metrics = {
            "loss": loss,
            "q/loss": q_loss,
            "q/mae_rad": mae_rad,
            "q/mae_deg": mae_rad * (180.0 / 3.141592653589793),
            "q/rmse_rad": residual.square().mean().sqrt(),
            "identity/mae_deg": identity_mae_rad * (180.0 / 3.141592653589793),
        }
        if "pred_wrist_delta_translation" in prediction:
            translation_scale = float(self.cfg.meta.wrist_translation_target_scale)
            rotation_scale = float(self.cfg.meta.wrist_rotation_target_scale)
            translation_loss = F.smooth_l1_loss(
                prediction["pred_wrist_delta_translation_scaled"],
                batch["wrist_delta_translation"] * translation_scale,
                beta=float(self.cfg.meta.wrist_translation_loss_beta_m) * translation_scale,
            )
            rotation_loss = F.smooth_l1_loss(
                prediction["pred_wrist_delta_rotvec_scaled"],
                batch["wrist_delta_rotvec"] * rotation_scale,
                beta=float(self.cfg.meta.wrist_rotation_loss_beta_rad) * rotation_scale,
            )
            loss = (
                float(getattr(self.cfg.meta, "q_loss_weight", 1.0)) * q_loss
                + float(self.cfg.meta.wrist_translation_loss_weight) * translation_loss
                + float(self.cfg.meta.wrist_rotation_loss_weight) * rotation_loss
            )
            translation_epe = torch.linalg.vector_norm(
                prediction["pred_wrist_delta_translation"] - batch["wrist_delta_translation"], dim=-1
            ).mean()
            pred_rotation = rotvec_to_matrix(prediction["pred_wrist_delta_rotvec"])
            gt_rotation = rotvec_to_matrix(batch["wrist_delta_rotvec"])
            relative_rotation = pred_rotation.transpose(-1, -2) @ gt_rotation
            cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
            rotation_error = torch.acos(cosine).mean()
            metrics.update(
                {
                    "loss": loss,
                    "wrist/translation_loss": translation_loss,
                    "wrist/rotation_loss": rotation_loss,
                    "wrist/translation_epe_mm": translation_epe * 1000.0,
                    "wrist/rotation_error_deg": rotation_error * (180.0 / 3.141592653589793),
                }
            )
            point_weight = float(getattr(self.cfg.meta, "baseline_point_loss_weight", 0.0))
            if point_weight > 0.0:
                if self.baseline_hand_model is None:
                    raise RuntimeError("baseline point loss requires DifferentiableInspireHand")
                predicted_points = reconstruct_hand_points(
                    self.baseline_hand_model,
                    q_t=batch["q_t"],
                    current_hand_points=batch["hand_points"],
                predicted_q=prediction["pred_q_next"],
                wrist_translation=prediction["pred_wrist_delta_translation"],
                wrist_rotvec=prediction["pred_wrist_delta_rotvec"],
                cached_link_index=batch.get("hand_point_link_index"),
                cached_local_points=batch.get("hand_points_local"),
            )
                target_points = batch["hand_points"] + batch["hand_flow"]
                point_scale = float(getattr(self.cfg.meta, "baseline_point_target_scale", 1.0))
                point_beta = float(getattr(self.cfg.meta, "baseline_point_loss_beta_m", 0.01))
                point_loss = F.smooth_l1_loss(
                    predicted_points * point_scale,
                    target_points * point_scale,
                    beta=point_beta * point_scale,
                )
                point_residual = predicted_points - target_points
                point_epe = torch.linalg.vector_norm(point_residual, dim=-1).mean()
                loss = loss + point_weight * point_loss
                metrics.update(
                    {
                        "loss": loss,
                        "hand_points/loss": point_loss,
                        "hand_points/epe_mm": point_epe * 1000.0,
                        "hand_points/rmse_mm": point_residual.square().mean().sqrt() * 1000.0,
                        "identity_hand/epe_mm": torch.linalg.vector_norm(
                            batch["hand_flow"], dim=-1
                        ).mean()
                        * 1000.0,
                    }
                )
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=int(batch["q_t"].shape[0]))

    def select_step_metrics(self, metrics: dict[str, float]):
        return {
            key: value
            for key, value in metrics.items()
            if key
            in {
                "loss",
                "q/mae_deg",
                "q/mae_rad",
                "identity/mae_deg",
                "wrist/translation_epe_mm",
                "wrist/rotation_error_deg",
                "hand_flow/epe_mm",
                "zero_flow/epe_mm",
                "hand_points/epe_mm",
                "identity_hand/epe_mm",
            }
        }
