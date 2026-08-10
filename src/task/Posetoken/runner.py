"""PoseToken V1 的 BaseRunner 数据与重建指标。"""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.base import BaseRunner, RunnerOutput
from src.base.checkpoint import load_checkpoint
from src.base.data import make_dataloader_kwargs
from src.task.InteractionDynamics.uni3d import gather_points
from src.task.Posetoken.dataset import (MANO, ManoPoseParameterDataset,
                                        SyntheticManoPoseDataset, face_centers)


class PoseTokenRunner(BaseRunner):
    def __init__(self, *args, **kwargs):
        self._mano_layer = None
        self._mano_faces = None
        super().__init__(*args, **kwargs)

    def make_dataloaders(self, data_cfg: Any, seed: int):
        common = dict(
            mano_path=self.cfg.meta.mano_path, side=self.cfg.meta.side,
            num_pca_comps=self.cfg.meta.num_pca_comps,
            num_patches=self.cfg.meta.num_patches, patch_size=self.cfg.meta.patch_size,
            generation_batch_size=data_cfg.generation_batch_size,
            beta_std=data_cfg.beta_std)
        dataset_class = (ManoPoseParameterDataset if bool(getattr(data_cfg, "on_the_fly", False))
                         else SyntheticManoPoseDataset)
        train = dataset_class(
            **common, num_samples=data_cfg.train_samples, seed=seed,
            pose_group_size=data_cfg.train_pose_group_size)
        val = dataset_class(
            **common, num_samples=data_cfg.val_samples, seed=data_cfg.val_seed,
            pose_group_size=data_cfg.eval_pose_group_size)
        test = dataset_class(
            **common, num_samples=data_cfg.test_samples, seed=data_cfg.test_seed,
            pose_group_size=data_cfg.eval_pose_group_size)
        kwargs = make_dataloader_kwargs(data_cfg, seed)
        train_loader = DataLoader(train, batch_size=int(data_cfg.batch_size),
                                  shuffle=bool(data_cfg.shuffle), **kwargs)
        eval_batch = int(data_cfg.val_batch_size or data_cfg.batch_size)
        val_loader = DataLoader(val, batch_size=eval_batch, shuffle=False,
                                **make_dataloader_kwargs(data_cfg, seed + 1))
        test_loader = DataLoader(test, batch_size=int(data_cfg.test_batch_size or eval_batch),
                                 shuffle=False, **make_dataloader_kwargs(data_cfg, seed + 2))
        metadata = {"data_source": "synthetic_mano",
                    "morphology": ("random_beta_to_zero" if float(data_cfg.beta_std) > 0
                                   else "beta_zero"),
                    "surface_generation": ("batch_mano" if dataset_class is ManoPoseParameterDataset
                                           else "precomputed"),
                    "side": str(self.cfg.meta.side), "train_samples": len(train),
                    "val_samples": len(val), "test_samples": len(test)}
        return (train_loader, val_loader, test_loader, metadata,
                {"val/": val_loader}, {"test/": test_loader})

    def build_model(self, model_cfg):
        model = self.build_model_from_config(model_cfg)
        checkpoint = getattr(model_cfg, "pretrained_checkpoint", None)
        if checkpoint:
            payload = load_checkpoint(checkpoint, map_location="cpu")
            model.load_state_dict(payload["model"])
        return model

    def prepare_batch(self, batch: Any) -> Any:
        batch = super().prepare_batch(batch)
        if "hand_points_root" in batch:
            return batch
        if self._mano_layer is None:
            self._mano_layer = MANO(
                str(self.cfg.meta.mano_path), is_rhand=str(self.cfg.meta.side) == "right",
                use_pca=True, num_pca_comps=int(self.cfg.meta.num_pca_comps),
                flat_hand_mean=True).to(self.device).eval().requires_grad_(False)
            self._mano_faces = torch.as_tensor(
                self._mano_layer.faces.astype("int64"), device=self.device)
        pose = batch["mano_pose"].float()
        beta = batch["mano_beta"].float()
        zeros3 = torch.zeros(len(pose), 3, device=pose.device)
        with torch.no_grad():
            current = self._mano_layer(hand_pose=pose, betas=beta,
                                       global_orient=zeros3, transl=zeros3)
            target = self._mano_layer(hand_pose=pose, betas=torch.zeros_like(beta),
                                      global_orient=zeros3, transl=zeros3)
            batch["hand_points_root"] = face_centers(
                current.vertices - current.joints[:, :1], self._mano_faces).float()
            batch["hand_target_points_root"] = face_centers(
                target.vertices - target.joints[:, :1], self._mano_faces).float()
        return batch

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor],
             mode: str = "train") -> RunnerOutput:
        prediction = model(batch)
        scale = float(self.cfg.meta.motion_scale)
        current = batch["hand_points_root"].float()
        target_points = batch.get("hand_target_points_root", current).float()
        canonical = batch["hand_cano_points"].float()
        target_patch = gather_points(target_points, batch["patch_knn_idx"])
        canonical_patch = gather_points(canonical, batch["patch_knn_idx"])
        dense_target = (target_patch - canonical_patch) * scale
        global_target = (target_points - canonical) * scale
        pred_dense = prediction["pred_patch_deformation_internal"].float()
        pred_global = prediction["pred_global_deformation_internal"].float()
        dense_loss = torch.nn.functional.mse_loss(pred_dense, dense_target)
        global_loss = torch.nn.functional.mse_loss(pred_global, global_target)
        loss = (float(self.cfg.train.dense_reconstruction_loss_weight) * dense_loss
                + float(self.cfg.train.global_reconstruction_loss_weight) * global_loss)
        group = batch.get("pose_group_id")
        pair_left, pair_right = [], []
        if mode != "train" and group is not None:
            for group_id in group.unique().tolist():
                members = (group == group_id).nonzero(as_tuple=False).flatten().tolist()
                if len(members) > 1:
                    pair_left.append(members[0])
                    pair_right.append(members[1])
        dense_zero = dense_target.square().mean()
        global_zero = global_target.square().mean()
        reconstructed = canonical_patch + pred_dense / scale
        epe_mm = (reconstructed - target_patch).norm(dim=-1).mean() * 1000.0
        centered = prediction["global_pose_token"] - prediction["global_pose_token"].mean(0)
        normalized = torch.nn.functional.normalize(prediction["global_pose_token"], dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        if similarity.shape[0] > 1:
            off_diagonal = ~torch.eye(similarity.shape[0], dtype=torch.bool,
                                      device=similarity.device)
            pairwise = similarity[off_diagonal].mean()
        else:
            pairwise = similarity.new_tensor(1.0)
        metrics = {
            "loss": loss,
            "loss/dense_reconstruction": dense_loss,
            "loss/global_reconstruction": global_loss,
            "pose/dense_rmse_cm": dense_loss.sqrt(),
            "pose/zero_dense_rmse_cm": dense_zero.sqrt(),
            "pose/dense_relative_improvement": 1 - dense_loss.sqrt() / dense_zero.sqrt(),
            "pose/global_rmse_cm": global_loss.sqrt(),
            "pose/zero_global_rmse_cm": global_zero.sqrt(),
            "pose/global_relative_improvement": 1 - global_loss.sqrt() / global_zero.sqrt(),
            "pose/reconstruction_epe_mm": epe_mm,
            "pose/global_feature_variance": centered.square().mean(),
            "pose/global_pairwise_cosine": pairwise,
            "pose/global_l2_norm": prediction["global_pose_token"].norm(dim=-1).mean(),
        }
        if pair_left:
            global_token = prediction["global_pose_token"].float()
            local_token = prediction["pose_tokens"].float()
            global_morph = (global_token[pair_left] - global_token[pair_right]).square().mean().sqrt()
            local_morph = (local_token[pair_left] - local_token[pair_right]).square().mean().sqrt()
            shifted = torch.roll(torch.arange(len(group), device=group.device), 1)
            different = group != group[shifted]
            global_between = (global_token[different] - global_token[shifted[different]]
                              ).square().mean().sqrt()
            local_between = (local_token[different] - local_token[shifted[different]]
                             ).square().mean().sqrt()
            metrics.update({
                "pose/global_morphology_rmse": global_morph,
                "pose/global_between_pose_rmse": global_between,
                "pose/global_morphology_ratio": global_morph / global_between.clamp_min(1e-8),
                "pose/local_morphology_rmse": local_morph,
                "pose/local_between_pose_rmse": local_between,
                "pose/local_morphology_ratio": local_morph / local_between.clamp_min(1e-8),
            })
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=current.shape[0])
