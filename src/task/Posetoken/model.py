"""局部 surface token 与紧凑全局 static-pose token。"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.task.InteractionDynamics.uni3d import gather_points


class StaticPoseEncoder(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        meta = cfg.meta
        dim = int(meta.model_dim)
        global_dim = int(meta.global_dim)
        self.patch_size = int(meta.patch_size)
        self.num_patches = int(meta.num_patches)
        self.num_hand_points = int(meta.num_hand_points)
        self.point_mlp = nn.Sequential(
            nn.Linear(12, 128), nn.GELU(), nn.Linear(128, 256), nn.GELU(),
            nn.Linear(256, dim))
        self.patch_identity = nn.Parameter(torch.randn(1, self.num_patches, dim) * .02)
        self.global_query = nn.Parameter(torch.randn(1, 1, dim) * .02)
        self.global_attention = nn.MultiheadAttention(
            dim, int(meta.attention_heads), batch_first=True)
        self.global_projection = nn.Linear(dim, global_dim)
        self.local_decoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, self.patch_size * 3))
        self.global_decoder = nn.Sequential(
            nn.Linear(global_dim, dim), nn.GELU(), nn.Linear(dim, self.num_hand_points * 3))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        points = batch["hand_points_root"]
        canonical = batch["hand_cano_points"]
        indices = batch["patch_knn_idx"]
        current_patch = gather_points(points, indices)
        canonical_patch = gather_points(canonical, indices)
        current_center = current_patch.mean(2, keepdim=True)
        canonical_center = canonical_patch.mean(2, keepdim=True)
        features = torch.cat([
            current_patch, current_patch - current_center,
            canonical_patch, canonical_patch - canonical_center,
        ], -1)
        pose_tokens = self.point_mlp(features).amax(2) + self.patch_identity
        query = self.global_query.expand(points.shape[0], -1, -1)
        pooled, attention = self.global_attention(query, pose_tokens, pose_tokens)
        global_pose = self.global_projection(pooled[:, 0])
        return {
            "pose_tokens": pose_tokens,
            "global_pose_token": global_pose,
            "global_pose_attention": attention,
            "pred_patch_deformation_internal": self.local_decoder(pose_tokens).reshape(
                points.shape[0], self.num_patches, self.patch_size, 3),
            "pred_global_deformation_internal": self.global_decoder(global_pose).reshape(
                points.shape[0], self.num_hand_points, 3),
        }
