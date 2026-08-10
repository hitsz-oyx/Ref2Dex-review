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


def action_decomposition_losses(
        batch: dict[str, torch.Tensor], prediction: dict[str, torch.Tensor], motion_scale: float
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """V12-A adjacent wrist SE(3) and wrist-frame articulated patch-flow losses."""
    articulation_target = patch_motion_target(
        batch["hand_articulation_increment_gt"].float(), prediction["hand_knn_idx"], motion_scale)
    articulation = torch.nn.functional.mse_loss(
        prediction["pred_hand_articulation_increment_internal"].float(), articulation_target)
    zero_articulation = articulation_target.square().mean()
    root_target = batch["hand_root_increment_pose_gt"].float()
    translation_target = root_target[..., :3, 3] * motion_scale
    translation = torch.nn.functional.mse_loss(
        prediction["pred_hand_root_increment_translation_internal"].float(), translation_target)
    zero_translation = translation_target.square().mean()
    rotation = torch.nn.functional.mse_loss(
        prediction["pred_hand_root_increment_rotation_matrix"].float(), root_target[..., :3, :3])
    relative_rotation = (prediction["pred_hand_root_increment_rotation_matrix"].float()
                         .transpose(-1, -2) @ root_target[..., :3, :3])
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(
        -1 + 1e-7, 1 - 1e-7)
    return articulation, translation, rotation, {
        "action/articulation_rmse_cm": articulation.sqrt(),
        "action/zero_articulation_rmse_cm": zero_articulation.sqrt(),
        "action/root_translation_rmse_cm": translation.sqrt(),
        "action/zero_root_translation_rmse_cm": zero_translation.sqrt(),
        "action/root_rotation_error_deg": torch.acos(cosine).mean() * (180 / torch.pi),
    }


def static_pose_reconstruction_loss(
        batch: dict[str, torch.Tensor], prediction: dict[str, torch.Tensor], motion_scale: float
        ) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.stack([
        gather_points(batch["action_hand_points_local_sequence"][:, step].float(),
                      prediction["hand_knn_idx"])
        for step in range(batch["action_hand_points_local_sequence"].shape[1])
    ], 1).mean(3) * motion_scale
    prediction_points = prediction["pred_hand_pose_patch_center_internal"].float()
    loss = torch.nn.functional.mse_loss(prediction_points, target)
    zero = (target - target.mean(dim=2, keepdim=True)).square().mean()
    return loss, zero


def object_centric_action_field_target(
        batch: dict[str, torch.Tensor], prediction: dict[str, torch.Tensor],
        sigma_m: float, motion_scale: float = 100.0) -> torch.Tensor:
    """Body-index-free [density, distance, local normal/tangent motion] at object patches."""
    hand_position = (batch["world_hand_points_object"].float()[:, None]
                     + batch["action_hand_disp_chunk_object"].float())
    object_center = prediction["obj_patch_centers_object"].float()
    object_normal = gather_points(batch["world_obj_normals_object"].float(),
                                  prediction["obj_knn_idx"]).mean(2)
    object_normal = torch.nn.functional.normalize(object_normal, dim=-1)
    relative = hand_position[:, :, None] - object_center[:, None, :, None]
    distance = relative.norm(dim=-1)
    weights = torch.exp(-distance.square() / (2.0 * sigma_m ** 2))
    density = weights.mean(-1, keepdim=True)
    nearest_distance = distance.amin(-1, keepdim=True) * motion_scale
    hand_motion = batch["action_hand_disp_chunk_object"].float()
    local_motion = hand_motion - hand_motion.mean(2, keepdim=True)
    normalized = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    local = (normalized[..., None] * local_motion[:, :, None]).sum(3) * motion_scale
    normal = (local * object_normal[:, None]).sum(-1, keepdim=True)
    tangent = local - normal * object_normal[:, None]
    return torch.cat([density, nearest_distance, normal, tangent], -1)


def action_field_losses(prediction: torch.Tensor, target: torch.Tensor
                        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equalize density, distance and local-motion groups instead of raw 6-D scales."""
    density = torch.nn.functional.mse_loss(prediction[..., :1], target[..., :1])
    distance = torch.nn.functional.mse_loss(prediction[..., 1:2], target[..., 1:2])
    motion = torch.nn.functional.mse_loss(prediction[..., 2:], target[..., 2:])
    zero_density = target[..., :1].square().mean()
    zero_distance = target[..., 1:2].square().mean()
    zero_motion = target[..., 2:].square().mean()
    return (density + distance + motion) / 3.0, {
        "interaction_field/rmse_density": density.sqrt(),
        "interaction_field/zero_rmse_density": zero_density.sqrt(),
        "interaction_field/rmse_distance_cm": distance.sqrt(),
        "interaction_field/zero_rmse_distance_cm": zero_distance.sqrt(),
        "interaction_field/rmse_local_motion_cm": motion.sqrt(),
        "interaction_field/zero_rmse_local_motion_cm": zero_motion.sqrt(),
    }


def relative_motion_target(batch: dict[str, torch.Tensor], prediction: dict[str, torch.Tensor],
                           motion_scale: float = 100.0,
                           mode: str = "cumulative_fixed") -> torch.Tensor:
    """Return hand->nearest-object edge [normal scalar, tangent xyz] trajectories in cm."""
    hand_motion = patch_motion_target(batch["hand_disp_chunk_object_gt"].float(),
                                      prediction["hand_knn_idx"], motion_scale)
    obj_motion_all = patch_motion_target(batch["obj_disp_chunk_gt"].float(),
                                         prediction["obj_knn_idx"], motion_scale)
    nearest = prediction["relative_nearest_obj_patch"]
    obj_motion = obj_motion_all[
        torch.arange(obj_motion_all.shape[0], device=obj_motion_all.device)[:, None, None],
        torch.arange(obj_motion_all.shape[1], device=obj_motion_all.device)[None, :, None],
        nearest[:, None, :],
    ]
    obj_normal = gather_points(batch["world_obj_normals_object"].float(),
                               prediction["obj_knn_idx"]).mean(2)
    obj_normal = torch.nn.functional.normalize(obj_normal, dim=-1)
    paired_normal = obj_normal[
        torch.arange(obj_normal.shape[0], device=obj_normal.device)[:, None], nearest]
    if mode == "cumulative_fixed":
        relative = hand_motion - obj_motion
        normal_basis = paired_normal[:, None]
    elif mode == "increment_fixed":
        hand_increment = torch.diff(hand_motion, dim=1, prepend=torch.zeros_like(hand_motion[:, :1]))
        obj_increment = torch.diff(obj_motion, dim=1, prepend=torch.zeros_like(obj_motion[:, :1]))
        relative = hand_increment - obj_increment
        future_normals = torch.stack([
            gather_points(batch["obj_normals_chunk_object_gt"][:, step].float(),
                          prediction["obj_knn_idx"]).mean(2)
            for step in range(relative.shape[1])
        ], 1)
        future_normals = torch.nn.functional.normalize(future_normals, dim=-1)
        normal_basis = future_normals[
            torch.arange(future_normals.shape[0], device=future_normals.device)[:, None, None],
            torch.arange(future_normals.shape[1], device=future_normals.device)[None, :, None],
            nearest[:, None, :],
        ]
    else:
        raise ValueError(f"Unsupported relative target mode: {mode}")
    normal = (relative * normal_basis).sum(-1, keepdim=True)
    tangent = relative - normal * normal_basis
    return torch.cat([normal, tangent], -1)


def masked_relative_mse(prediction: torch.Tensor, target: torch.Tensor,
                        edge_distance_m: torch.Tensor, radius_cm: float) -> torch.Tensor:
    edge_mask = edge_distance_m * 100.0 < radius_cm
    mask = edge_mask[:, None, :, None].expand_as(target)
    if not mask.any():
        raise RuntimeError(f"InteractionDynamics batch has no hand-object edge within {radius_cm:g} cm")
    return (prediction - target).square()[mask].mean()


def dense_relative_motion_target(batch: dict[str, torch.Tensor], prediction: dict[str, torch.Tensor],
                                 motion_scale: float = 100.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-point fixed-edge incremental target and known-hand analytic baseline."""
    hand_motion = batch["hand_disp_chunk_object_gt"].float() * motion_scale
    hand_increment = torch.diff(hand_motion, dim=1, prepend=torch.zeros_like(hand_motion[:, :1]))
    obj_motion = batch["obj_disp_chunk_gt"].float() * motion_scale
    obj_increment = torch.diff(obj_motion, dim=1, prepend=torch.zeros_like(obj_motion[:, :1]))
    nearest = prediction["dense_nearest_obj_point"]
    batch_index = torch.arange(obj_increment.shape[0], device=obj_increment.device)[:, None, None]
    step_index = torch.arange(obj_increment.shape[1], device=obj_increment.device)[None, :, None]
    paired_obj_increment = obj_increment[batch_index, step_index, nearest[:, None]]
    future_normal = batch["obj_normals_chunk_object_gt"].float()
    paired_normal = future_normal[batch_index, step_index, nearest[:, None]]
    paired_normal = torch.nn.functional.normalize(paired_normal, dim=-1)
    relative = hand_increment - paired_obj_increment
    relative_normal = (relative * paired_normal).sum(-1, keepdim=True)
    relative_tangent = relative - relative_normal * paired_normal
    hand_normal = (hand_increment * paired_normal).sum(-1, keepdim=True)
    hand_tangent = hand_increment - hand_normal * paired_normal
    return (torch.cat([relative_normal, relative_tangent], -1),
            torch.cat([hand_normal, hand_tangent], -1))


def trajectory_statistics(pred_m: torch.Tensor, gt_m: torch.Tensor,
                          valid: torch.Tensor) -> dict[str, torch.Tensor]:
    error_mm = torch.linalg.vector_norm(pred_m - gt_m, dim=-1) * 1000.0
    mask = valid[:, None].expand_as(error_mm)
    per_step = [(error_mm[:, step] * valid).sum() / valid.sum() for step in range(error_mm.shape[1])]
    result = {"object/ade_mm": error_mm[mask].mean(), "object/fde_mm": per_step[-1]}
    result.update({f"object/epe_step_{step + 1}_mm": value for step, value in enumerate(per_step)})
    return result


def se3_statistics_and_loss(prediction: dict[str, torch.Tensor],
                            gt_increment_pose: torch.Tensor,
                            motion_scale: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    pred_translation = prediction["pred_obj_increment_translation_internal"].float()
    gt_translation = gt_increment_pose[..., :3, 3].float() * motion_scale
    translation_loss = torch.nn.functional.mse_loss(pred_translation, gt_translation)
    pred_rotation = prediction["pred_obj_increment_rotation_matrix"].float()
    gt_rotation = gt_increment_pose[..., :3, :3].float()
    rotation_loss = torch.nn.functional.mse_loss(pred_rotation, gt_rotation)
    relative_rotation = pred_rotation.transpose(-1, -2) @ gt_rotation
    cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(
        -1 + 1e-7, 1 - 1e-7)
    angle_deg = torch.acos(cosine) * (180.0 / torch.pi)
    return translation_loss, rotation_loss, {
        "se3/translation_rmse_cm": translation_loss.sqrt(),
        "se3/translation_error_cm": (pred_translation - gt_translation).norm(dim=-1).mean(),
        "se3/rotation_error_deg": angle_deg.mean(),
    }


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


def field_statistics(field: torch.Tensor, descriptor: torch.Tensor,
                     soft_weights: torch.Tensor) -> dict[str, torch.Tensor]:
    """Measure V7's actual spatiotemporal field, independently of legacy tokens."""
    spatial_centered = field - field.mean(2, keepdim=True)
    temporal_centered = field - field.mean(1, keepdim=True)
    probability = soft_weights.clamp_min(1e-8)
    hand_entropy = -(probability * probability.log()).sum(-1).mean()
    return {
        "field/spatial_feature_variance": spatial_centered.square().mean(),
        "field/temporal_feature_variance": temporal_centered.square().mean(),
        "field/descriptor_variance": descriptor.var(dim=(0, 1, 2), unbiased=False).mean(),
        "field/proximity_density_mean": descriptor[..., 0].mean(),
        "field/hand_weight_entropy": hand_entropy,
    }


class InteractionDynamicsRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        common = {"dominant_hand_manifest": data_cfg.dominant_hand_manifest,
                  "hand_side": getattr(data_cfg, "hand_side", None),
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
        if bool(getattr(data_cfg, "intervention_on_train", False)):
            val_loaders["train/"] = train
        test_loaders = {} if test is None else {"test/": test}
        return train, val, test, metadata, val_loaders, test_loaders

    def build_model(self, model_cfg: Any) -> torch.nn.Module:
        return self.build_model_from_config(model_cfg)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        model = unwrap_model(self.model)
        if not hasattr(model, "world"):
            return torch.optim.AdamW([parameter for parameter in model.parameters()
                                      if parameter.requires_grad], lr=float(self.cfg.train.lr),
                                     weight_decay=float(self.cfg.train.weight_decay))
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
        if "cm_tokens" in prediction:
            return self._step_v14(prediction, batch)
        if "pred_dense_relative_motion_internal" in prediction:
            target, hand_baseline = dense_relative_motion_target(batch, prediction)
            radius_cm = float(self.cfg.train.relative_edge_radius_cm)
            pred_dense = prediction["pred_dense_relative_motion_internal"].float()
            dense_loss = masked_relative_mse(
                pred_dense, target, prediction["dense_edge_distance_m"], radius_cm)
            zero_loss = masked_relative_mse(
                torch.zeros_like(target), target, prediction["dense_edge_distance_m"], radius_cm)
            hand_loss = masked_relative_mse(
                hand_baseline, target, prediction["dense_edge_distance_m"], radius_cm)
            tokens = prediction["dense_interaction_tokens"]
            centered = tokens - tokens.mean(2, keepdim=True)
            metrics = {
                "loss": dense_loss * float(self.cfg.train.dense_relative_loss_weight),
                "dense_relative/rmse_cm": dense_loss.sqrt(),
                "dense_relative/zero_rmse_cm": zero_loss.sqrt(),
                "dense_relative/hand_only_rmse_cm": hand_loss.sqrt(),
                "dense_relative/edge_fraction": (
                    prediction["dense_edge_distance_m"] * 100 < radius_cm).float().mean(),
                "dense_relative/feature_variance": centered.square().mean(),
            }
            return RunnerOutput(loss=metrics["loss"], metrics=metrics,
                                batch_size=pred_dense.shape[0])
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
        relative_target = relative_motion_target(
            batch, prediction, scale, str(self.cfg.train.relative_target_mode))
        edge_radius_cm = float(self.cfg.train.relative_edge_radius_cm)
        relative_loss = masked_relative_mse(
            prediction["pred_relative_motion_internal"].float(), relative_target,
            prediction["relative_edge_distance_m"], edge_radius_cm)
        relative_zero_loss = masked_relative_mse(
            torch.zeros_like(relative_target), relative_target,
            prediction["relative_edge_distance_m"], edge_radius_cm)
        translation_loss, rotation_loss, se3_metrics = se3_statistics_and_loss(
            prediction, batch["obj_increment_pose_gt"], scale)
        action_field_weight = float(self.cfg.train.interaction_reconstruction_loss_weight)
        if action_field_weight:
            action_field_target = object_centric_action_field_target(
                batch, prediction, float(self.cfg.meta.field_sigma_m), scale)
            action_field_loss, action_field_metrics = action_field_losses(
                prediction["pred_object_centric_action_field"].float(), action_field_target)
        else:
            action_field_loss = pred.new_zeros(())
            action_field_metrics = {}
        articulation_weight = float(self.cfg.train.action_articulation_loss_weight)
        root_translation_weight = float(self.cfg.train.action_root_translation_loss_weight)
        root_rotation_weight = float(self.cfg.train.action_root_rotation_loss_weight)
        if articulation_weight or root_translation_weight or root_rotation_weight:
            articulation_loss, root_translation_loss, root_rotation_loss, action_metrics = (
                action_decomposition_losses(batch, prediction, scale))
        else:
            articulation_loss = root_translation_loss = root_rotation_loss = pred.new_zeros(())
            action_metrics = {}
        static_pose_weight = float(self.cfg.train.static_pose_reconstruction_loss_weight)
        if static_pose_weight:
            static_pose_loss, static_pose_zero = static_pose_reconstruction_loss(
                batch, prediction, scale)
        else:
            static_pose_loss = static_pose_zero = pred.new_zeros(())
        loss = (float(self.cfg.train.effect_loss_weight) * effect_loss
                + float(self.cfg.train.se3_translation_loss_weight) * translation_loss
                + float(self.cfg.train.se3_rotation_loss_weight) * rotation_loss
                + float(self.cfg.train.action_loss_weight) * action_loss
                + float(self.cfg.train.patch_effect_loss_weight) * patch_effect_loss
                + float(self.cfg.train.relative_loss_weight) * relative_loss
                + action_field_weight * action_field_loss
                + articulation_weight * articulation_loss
                + root_translation_weight * root_translation_loss
                + root_rotation_weight * root_rotation_loss)
        loss = loss + static_pose_weight * static_pose_loss
        metrics: dict[str, Any] = {"loss": loss, "loss/effect": effect_loss,
                                  "loss/action": action_loss,
                                  "loss/patch_effect": patch_effect_loss,
                                  "loss/relative": relative_loss,
                                  "loss/se3_translation": translation_loss,
                                  "loss/se3_rotation": rotation_loss,
                                  "loss/interaction_reconstruction": action_field_loss,
                                  "loss/action_articulation": articulation_loss,
                                  "loss/action_root_translation": root_translation_loss,
                                  "loss/action_root_rotation": root_rotation_loss,
                                  "loss/static_pose_reconstruction": static_pose_loss,
                                  "relative/rmse_cm": relative_loss.sqrt(),
                                  "relative/zero_rmse_cm": relative_zero_loss.sqrt(),
                                  "relative/edge_fraction": (
                                      prediction["relative_edge_distance_m"] * 100.0 < edge_radius_cm
                                  ).float().mean()}
        metrics.update(trajectory_statistics(pred, gt, valid))
        metrics.update(se3_metrics)
        metrics.update(action_field_metrics)
        metrics.update(action_metrics)
        if static_pose_weight:
            metrics["pose/reconstruction_rmse_cm"] = static_pose_loss.sqrt()
            metrics["pose/centered_zero_rmse_cm"] = static_pose_zero.sqrt()
        metrics.update(collapse_statistics(prediction["interaction_tokens"],
                                           prediction["object_to_action_attention"]))
        if "interaction_field" in prediction:
            metrics.update(field_statistics(
                prediction.get("se3_interaction_field", prediction["interaction_field"]),
                prediction.get("se3_interaction_field_descriptor",
                               prediction["interaction_field_descriptor"]),
                prediction["hand_to_object_soft_weights"]))
        edge = prediction["relative_interaction_tokens"]
        edge_centered = edge - edge.mean(1, keepdim=True)
        edge_normalized = torch.nn.functional.normalize(edge, dim=-1)
        edge_similarity = edge_normalized @ edge_normalized.transpose(1, 2)
        off_diagonal = ~torch.eye(edge.shape[1], dtype=torch.bool, device=edge.device)
        metrics["relative/feature_variance"] = edge_centered.square().mean()
        metrics["relative/pairwise_cosine"] = edge_similarity[:, off_diagonal].mean()
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=pred.shape[0])

    def _step_v14(self, prediction: dict[str, torch.Tensor],
                  batch: dict[str, torch.Tensor]) -> RunnerOutput:
        valid = batch["effect_obj_valid_mask"].bool()
        gt = batch["effect_obj_disp_gt"].float()
        pred = prediction["pred_obj_disp_chunk"].float()
        scale = float(self.cfg.meta.motion_scale)
        translation_loss, rotation_loss, se3_metrics = se3_statistics_and_loss(
            prediction, batch["obj_increment_pose_gt"], scale)
        loss = (float(self.cfg.train.se3_translation_loss_weight) * translation_loss
                + float(self.cfg.train.se3_rotation_loss_weight) * rotation_loss)
        cm = prediction["cm_tokens"].float()
        spatial = cm - cm.mean(2, keepdim=True)
        temporal = cm - cm.mean(1, keepdim=True)
        normalized = torch.nn.functional.normalize(cm, dim=-1)
        similarity = normalized @ normalized.transpose(-1, -2)
        off = ~torch.eye(cm.shape[2], dtype=torch.bool, device=cm.device)
        action_probability = prediction["action_to_world_attention"].float().mean(2).clamp_min(1e-8)
        effect_probability = prediction["effect_to_cm_attention"].float().mean(2).clamp_min(1e-8)
        metrics: dict[str, torch.Tensor] = {
            "loss": loss,
            "loss/se3_translation": translation_loss,
            "loss/se3_rotation": rotation_loss,
            "cm/spatial_variance": spatial.square().mean(),
            "cm/temporal_variance": temporal.square().mean(),
            "cm/pairwise_cosine": similarity[..., off].mean(),
            "cm/l2_norm": cm.norm(dim=-1).mean(),
            "cm/action_to_world_entropy": -(action_probability * action_probability.log()).sum(-1).mean(),
            "cm/effect_attention_entropy": -(effect_probability * effect_probability.log()).sum(-1).mean(),
        }
        metrics.update(trajectory_statistics(pred, gt, valid))
        metrics.update(se3_metrics)
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=pred.shape[0])
