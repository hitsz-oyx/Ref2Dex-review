from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader

from src.base import BaseRunner, RunnerOutput
from src.base.checkpoint import unwrap_model
from src.base.data import make_dataloader_kwargs
from src.task.Actiontoken.generator import (SyntheticManoActionDataset,
                                            SyntheticManoTransitionDataset)
from src.task.InteractionDynamics.uni3d import gather_points


class ActionTokenRunner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        common = dict(mano_path=self.cfg.meta.mano_path, side=self.cfg.meta.side,
                      num_pca_comps=self.cfg.meta.num_pca_comps,
                      num_patches=self.cfg.meta.num_patches, patch_size=self.cfg.meta.patch_size,
                      chunk_len=self.cfg.meta.chunk_len,
                      generation_batch_size=data_cfg.generation_batch_size,
                      velocity_decay=data_cfg.velocity_decay, velocity_std=data_cfg.velocity_std)
        train = SyntheticManoActionDataset(**common, num_samples=data_cfg.train_samples, seed=seed)
        val = SyntheticManoActionDataset(**common, num_samples=data_cfg.val_samples,
                                         seed=data_cfg.val_seed)
        test = SyntheticManoActionDataset(**common, num_samples=data_cfg.test_samples,
                                          seed=data_cfg.test_seed)
        train_loader = DataLoader(train, batch_size=data_cfg.batch_size,
                                  shuffle=bool(data_cfg.shuffle),
                                  **make_dataloader_kwargs(data_cfg, seed))
        eval_batch = int(data_cfg.val_batch_size or data_cfg.batch_size)
        val_loader = DataLoader(val, batch_size=eval_batch, shuffle=False,
                                **make_dataloader_kwargs(data_cfg, seed + 1))
        test_loader = DataLoader(test, batch_size=int(data_cfg.test_batch_size or eval_batch),
                                 shuffle=False, **make_dataloader_kwargs(data_cfg, seed + 2))
        metadata = {"source": "synthetic_mano_trajectory", "beta": 0,
                    "train_samples": len(train), "val_samples": len(val),
                    "test_samples": len(test)}
        return train_loader, val_loader, test_loader, metadata, {"val/": val_loader}, {"test/": test_loader}

    def build_model(self, model_cfg):
        return self.build_model_from_config(model_cfg)

    @staticmethod
    def _target(batch):
        sequence = batch["hand_points_root_sequence"].float()
        patches = torch.stack([gather_points(sequence[:, frame], batch["patch_knn_idx"])
                               for frame in range(sequence.shape[1])], 1)
        return patches[:, 1:] - patches[:, :-1]

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode="train"):
        output = model(batch)
        target = self._target(batch)
        scale = float(self.cfg.meta.motion_scale)
        target_internal = target * scale
        pred = output["pred_dense_flow_internal"].float()
        loss = torch.nn.functional.mse_loss(pred, target_internal)
        zero = target_internal.square().mean()
        epe = (pred / scale - target).norm(dim=-1).mean() * 1000
        tokens = output["action_tokens"].float()
        centered = tokens - tokens.mean((1, 2), keepdim=True)
        normalized = torch.nn.functional.normalize(tokens.flatten(1, 2), dim=-1)
        similarity = normalized @ normalized.transpose(1, 2)
        off = ~torch.eye(similarity.shape[-1], dtype=torch.bool, device=similarity.device)
        metrics = {"loss": loss, "action/dense_flow_rmse_cm": loss.sqrt(),
                   "action/zero_dense_flow_rmse_cm": zero.sqrt(),
                   "action/dense_flow_relative_improvement": 1 - loss.sqrt() / zero.sqrt(),
                   "action/dense_flow_epe_mm": epe,
                   "action/token_variance": centered.square().mean(),
                   "action/token_pairwise_cosine": similarity[:, off].mean(),
                   "action/token_l2_norm": tokens.norm(dim=-1).mean()}
        speed = target_internal.norm(dim=-1)
        quantiles = torch.quantile(speed.flatten(), torch.tensor([1 / 3, 2 / 3], device=speed.device))
        for name, mask in (("slow", speed <= quantiles[0]),
                           ("medium", (speed > quantiles[0]) & (speed <= quantiles[1])),
                           ("fast", speed > quantiles[1])):
            metrics[f"action/{name}_flow_rmse_cm"] = (pred.sub(target_internal).square().sum(-1)[mask].mean()).sqrt()
        if mode != "train":
            dynamic = unwrap_model(model).dynamic
            local = output["pose_tokens_sequence"]
            global_pose = output["global_pose_token_sequence"]
            # Pairwise reverse/static checks do not use temporal order: run one repeated pair per transition.
            reverse_flow = dynamic.forward_pairs(
                local[:, 1:], local[:, :-1], global_pose[:, 1:], global_pose[:, :-1]
            )["pred_dense_flow_internal"]
            static_flow = dynamic.forward_pairs(
                local[:, :-1], local[:, :-1], global_pose[:, :-1], global_pose[:, :-1]
            )["pred_dense_flow_internal"]
            metrics["action/reverse_pair_residual_rmse_cm"] = (
                reverse_flow + pred).square().mean().sqrt()
            metrics["action/static_pair_rmse_cm"] = static_flow.square().mean().sqrt()
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=pred.shape[0])


class ActionTokenV2Runner(BaseRunner):
    def make_dataloaders(self, data_cfg: Any, seed: int):
        common = dict(mano_path=self.cfg.meta.mano_path, side=self.cfg.meta.side,
                      num_pca_comps=self.cfg.meta.num_pca_comps,
                      num_patches=self.cfg.meta.num_patches, patch_size=self.cfg.meta.patch_size,
                      generation_batch_size=data_cfg.generation_batch_size,
                      motion_stds=tuple(data_cfg.motion_stds),
                      motion_probs=tuple(data_cfg.motion_probs),
                      sparse_probability=data_cfg.sparse_probability,
                      motion_scale=self.cfg.meta.motion_scale)
        train = SyntheticManoTransitionDataset(**common, num_samples=data_cfg.train_samples,
                                               seed=seed)
        val = SyntheticManoTransitionDataset(**common, num_samples=data_cfg.val_samples,
                                             seed=data_cfg.val_seed)
        test = SyntheticManoTransitionDataset(**common, num_samples=data_cfg.test_samples,
                                              seed=data_cfg.test_seed)
        kwargs = make_dataloader_kwargs(data_cfg, seed)
        train_loader = DataLoader(train, batch_size=data_cfg.batch_size,
                                  shuffle=bool(data_cfg.shuffle), **kwargs)
        eval_batch = int(data_cfg.val_batch_size or data_cfg.batch_size)
        val_loader = DataLoader(val, batch_size=eval_batch, shuffle=False,
                                **make_dataloader_kwargs(data_cfg, seed + 1))
        test_loader = DataLoader(test, batch_size=int(data_cfg.test_batch_size or eval_batch),
                                 shuffle=False, **make_dataloader_kwargs(data_cfg, seed + 2))
        metadata = {"source": "synthetic_mano_transition", "beta": 0, "fps": 30,
                    "train_samples": len(train), "val_samples": len(val),
                    "test_samples": len(test)}
        return (train_loader, val_loader, test_loader, metadata,
                {"val/": val_loader}, {"test/": test_loader})

    def build_model(self, model_cfg):
        return self.build_model_from_config(model_cfg)

    def step(self, model: torch.nn.Module, batch: dict[str, torch.Tensor], mode="train"):
        output = model(batch)
        target = batch["patch_flow_internal"].float()
        pred = output["pred_dense_flow_internal"].float()
        loss = torch.nn.functional.mse_loss(pred, target)
        zero = target.square().mean()
        scale = float(self.cfg.meta.motion_scale)
        tokens = output["action_tokens"].float()
        metrics = {"loss": loss, "action/dense_flow_rmse_cm": loss.sqrt(),
                   "action/zero_dense_flow_rmse_cm": zero.sqrt(),
                   "action/dense_flow_relative_improvement": 1 - loss.sqrt() / zero.sqrt(),
                   "action/dense_flow_epe_mm": ((pred - target) / scale).norm(dim=-1).mean() * 1000,
                   "action/token_variance": tokens.var(dim=1, unbiased=False).mean(),
                   "action/token_l2_norm": tokens.norm(dim=-1).mean()}
        speed = target.norm(dim=-1)
        quantiles = torch.quantile(speed.flatten(),
                                   torch.tensor([1 / 3, 2 / 3], device=speed.device))
        error_sq = pred.sub(target).square().sum(-1)
        for name, mask in (("slow", speed <= quantiles[0]),
                           ("medium", (speed > quantiles[0]) & (speed <= quantiles[1])),
                           ("fast", speed > quantiles[1])):
            metrics[f"action/{name}_flow_rmse_cm"] = error_sq[mask].mean().sqrt()
        if mode != "train":
            encoder = unwrap_model(model).flow_action
            reverse = encoder(-target)
            static = encoder(torch.zeros_like(target))
            metrics["action/reverse_token_residual"] = (
                reverse["action_tokens"] + tokens).square().mean().sqrt()
            metrics["action/reverse_flow_residual_rmse_cm"] = (
                reverse["pred_dense_flow_internal"] + pred).square().mean().sqrt()
            metrics["action/static_token_l2_norm"] = static["action_tokens"].norm(dim=-1).mean()
            metrics["action/static_flow_rmse_cm"] = (
                static["pred_dense_flow_internal"].square().mean().sqrt())
        return RunnerOutput(loss=loss, metrics=metrics, batch_size=pred.shape[0])
