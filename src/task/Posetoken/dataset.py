"""固定 morphology 的 synthetic MANO 静态姿态数据。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

for _legacy_name, _legacy_value in {
    "bool": bool, "int": int, "float": float, "complex": complex,
    "object": object, "unicode": str, "str": str,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)

from smplx import MANO

from src.task.InteractionDynamics.uni3d import patchify


def face_centers(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    return vertices[:, faces].mean(2)


def build_canonical_patch_map(canonical_points: torch.Tensor, num_patches: int = 64,
                              patch_size: int = 32) -> torch.Tensor:
    """只在 canonical surface 上建立一次稳定 patch atlas。"""
    _, _, indices = patchify(canonical_points[None], num_patches, patch_size)
    return indices[0]


def sample_mano_pose(count: int, components: int, seed: int) -> torch.Tensor:
    """确定性的 local/wide/sparse PCA mixture，避免无界随机关节姿态。"""
    generator = torch.Generator().manual_seed(int(seed))
    base_scale = torch.linspace(1.25, .35, components)
    family = torch.rand(count, generator=generator)
    pose = torch.randn(count, components, generator=generator)
    scale = torch.where(family[:, None] < .5, .45, 1.0) * base_scale
    pose = pose * scale
    sparse = family >= .8
    if sparse.any():
        keep = torch.rand(int(sparse.sum()), components, generator=generator) < .2
        pose[sparse] *= keep
    return pose.clamp(-2.5, 2.5)


class SyntheticManoPoseDataset(Dataset):
    """预生成 β=0、无 global orientation/translation 的 MANO surface。"""
    def __init__(self, *, mano_path: str | Path, side: str, num_samples: int, seed: int,
                 num_pca_comps: int = 24, num_patches: int = 64, patch_size: int = 32,
                 generation_batch_size: int = 256, beta_std: float = 0.0,
                 pose_group_size: int = 1) -> None:
        super().__init__()
        self.side = str(side)
        layer = MANO(str(mano_path), is_rhand=self.side == "right", use_pca=True,
                     num_pca_comps=int(num_pca_comps), flat_hand_mean=True)
        layer.eval().requires_grad_(False)
        faces = torch.as_tensor(layer.faces.astype(np.int64))
        group_size = int(pose_group_size)
        if group_size < 1:
            raise ValueError("pose_group_size 必须为正整数")
        unique_count = (int(num_samples) + group_size - 1) // group_size
        pose = sample_mano_pose(unique_count, int(num_pca_comps), int(seed)).repeat_interleave(
            group_size, 0)[:int(num_samples)]
        generator = torch.Generator().manual_seed(int(seed) + 100003)
        betas = (torch.randn(int(num_samples), 10, generator=generator) * float(beta_std)).clamp(-2, 2)
        surfaces, targets = [], []
        with torch.no_grad():
            for start in range(0, len(pose), int(generation_batch_size)):
                current = pose[start:start + int(generation_batch_size)]
                current_betas = betas[start:start + len(current)]
                output = layer(hand_pose=current, betas=current_betas,
                               global_orient=torch.zeros(len(current), 3),
                               transl=torch.zeros(len(current), 3))
                target = layer(hand_pose=current, betas=torch.zeros(len(current), 10),
                               global_orient=torch.zeros(len(current), 3),
                               transl=torch.zeros(len(current), 3))
                root_local = output.vertices - output.joints[:, :1]
                target_root_local = target.vertices - target.joints[:, :1]
                surfaces.append(face_centers(root_local, faces).float())
                targets.append(face_centers(target_root_local, faces).float())
            zero = layer(hand_pose=torch.zeros(1, int(num_pca_comps)),
                         betas=torch.zeros(1, 10), global_orient=torch.zeros(1, 3),
                         transl=torch.zeros(1, 3))
        canonical = face_centers(zero.vertices - zero.joints[:, :1], faces)[0].float()
        self.hand_points_root = torch.cat(surfaces)
        self.hand_target_points_root = torch.cat(targets)
        self.hand_cano_points = canonical
        self.patch_knn_idx = build_canonical_patch_map(canonical, num_patches, patch_size)
        self.pose_parameters = pose
        self.betas = betas
        self.pose_group_id = torch.arange(int(num_samples)) // group_size

    def __len__(self) -> int:
        return self.hand_points_root.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "hand_points_root": self.hand_points_root[index],
            "hand_cano_points": self.hand_cano_points,
            "hand_target_points_root": self.hand_target_points_root[index],
            "patch_knn_idx": self.patch_knn_idx,
            "mano_pose": self.pose_parameters[index],
            "mano_beta": self.betas[index],
            "pose_group_id": self.pose_group_id[index],
            "side_id": torch.tensor(self.side == "right", dtype=torch.long),
            "sample_id": torch.tensor(index, dtype=torch.long),
        }
