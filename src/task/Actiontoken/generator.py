"""Synthetic MANO articulation trajectory 与单步动作数据。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

for _name, _value in {"bool": bool, "int": int, "float": float, "complex": complex,
                      "object": object, "unicode": str, "str": str}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)
from smplx import MANO

from src.task.Posetoken.dataset import build_canonical_patch_map, face_centers, sample_mano_pose


def sample_smooth_pose_trajectory(count: int, components: int, steps: int, seed: int,
                                  decay: float, velocity_std: float) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed))
    pose = sample_mano_pose(count, components, seed)
    velocity = torch.randn(count, components, generator=generator) * float(velocity_std)
    trajectory = [pose]
    component_scale = torch.linspace(1.0, .35, components)
    for _ in range(steps - 1):
        noise = torch.randn(count, components, generator=generator) * float(velocity_std) * .25
        velocity = float(decay) * velocity + noise * component_scale
        pose = (pose + velocity).clamp(-2.5, 2.5)
        trajectory.append(pose)
    return torch.stack(trajectory, 1)


class SyntheticManoActionDataset(Dataset):
    def __init__(self, *, mano_path: str | Path, side: str, num_samples: int, seed: int,
                 num_pca_comps: int, num_patches: int, patch_size: int, chunk_len: int,
                 generation_batch_size: int, velocity_decay: float, velocity_std: float) -> None:
        layer = MANO(str(mano_path), is_rhand=side == "right", use_pca=True,
                     num_pca_comps=num_pca_comps, flat_hand_mean=True)
        layer.eval().requires_grad_(False)
        faces = torch.as_tensor(layer.faces.astype(np.int64))
        poses = sample_smooth_pose_trajectory(
            num_samples, num_pca_comps, chunk_len + 1, seed, velocity_decay, velocity_std)
        flat = poses.reshape(-1, num_pca_comps)
        surfaces = []
        with torch.no_grad():
            for start in range(0, len(flat), generation_batch_size):
                current = flat[start:start + generation_batch_size]
                output = layer(hand_pose=current, betas=torch.zeros(len(current), 10),
                               global_orient=torch.zeros(len(current), 3),
                               transl=torch.zeros(len(current), 3))
                surfaces.append(face_centers(output.vertices - output.joints[:, :1], faces).float())
            zero = layer(hand_pose=torch.zeros(1, num_pca_comps), betas=torch.zeros(1, 10),
                         global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
        canonical = face_centers(zero.vertices - zero.joints[:, :1], faces)[0].float()
        self.surfaces = torch.cat(surfaces).reshape(num_samples, chunk_len + 1, -1, 3)
        self.canonical = canonical
        self.patch_knn_idx = build_canonical_patch_map(canonical, num_patches, patch_size)
        self.poses = poses

    def __len__(self):
        return len(self.surfaces)

    def __getitem__(self, index):
        return {"hand_points_root_sequence": self.surfaces[index],
                "hand_cano_points": self.canonical,
                "patch_knn_idx": self.patch_knn_idx,
                "mano_pose_sequence": self.poses[index],
                "sample_id": torch.tensor(index, dtype=torch.long)}


def sample_pose_pairs(count: int, components: int, seed: int,
                      motion_stds: tuple[float, float, float],
                      motion_probs: tuple[float, float, float],
                      sparse_probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    """采样独立 pose pair；混合动作尺度并包含稀疏 PCA 分量运动。"""
    generator = torch.Generator().manual_seed(int(seed))
    pose = sample_mano_pose(count, components, seed)
    category = torch.multinomial(torch.tensor(motion_probs), count, replacement=True,
                                 generator=generator)
    std = torch.tensor(motion_stds)[category, None]
    delta = torch.randn(count, components, generator=generator) * std
    sparse = torch.rand(count, generator=generator) < float(sparse_probability)
    for row in sparse.nonzero(as_tuple=False).flatten().tolist():
        keep = int(torch.randint(1, min(5, components) + 1, (), generator=generator))
        active = torch.randperm(components, generator=generator)[:keep]
        mask = torch.zeros(components, dtype=torch.bool)
        mask[active] = True
        delta[row, ~mask] = 0
    return pose, delta


class SyntheticManoTransitionDataset(Dataset):
    """V2 单 transition 数据；所有 surface 都在各自 wrist/root frame。"""

    def __init__(self, *, mano_path: str | Path, side: str, num_samples: int, seed: int,
                 num_pca_comps: int, num_patches: int, patch_size: int,
                 generation_batch_size: int, motion_stds: tuple[float, float, float],
                 motion_probs: tuple[float, float, float], sparse_probability: float,
                 motion_scale: float) -> None:
        layer = MANO(str(mano_path), is_rhand=side == "right", use_pca=True,
                     num_pca_comps=num_pca_comps, flat_hand_mean=True)
        layer.eval().requires_grad_(False)
        faces = torch.as_tensor(layer.faces.astype(np.int64))
        pose, delta = sample_pose_pairs(num_samples, num_pca_comps, seed, motion_stds,
                                        motion_probs, sparse_probability)
        after_pose = (pose + delta).clamp(-2.5, 2.5)
        all_pose = torch.stack((pose, after_pose), 1).reshape(-1, num_pca_comps)
        surfaces = []
        with torch.no_grad():
            for start in range(0, len(all_pose), generation_batch_size):
                current = all_pose[start:start + generation_batch_size]
                output = layer(hand_pose=current, betas=torch.zeros(len(current), 10),
                               global_orient=torch.zeros(len(current), 3),
                               transl=torch.zeros(len(current), 3))
                surfaces.append(face_centers(output.vertices - output.joints[:, :1], faces).float())
            zero = layer(hand_pose=torch.zeros(1, num_pca_comps), betas=torch.zeros(1, 10),
                         global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
        canonical = face_centers(zero.vertices - zero.joints[:, :1], faces)[0].float()
        pair = torch.cat(surfaces).reshape(num_samples, 2, -1, 3)
        patch_idx = build_canonical_patch_map(canonical, num_patches, patch_size)
        dense_flow = (pair[:, 1] - pair[:, 0]) * float(motion_scale)
        self.patch_flow = dense_flow[:, patch_idx]
        self.pose = pose
        self.delta = after_pose - pose
        self.patch_knn_idx = patch_idx

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, index):
        return {"patch_flow_internal": self.patch_flow[index],
                "patch_knn_idx": self.patch_knn_idx,
                "mano_pose": self.pose[index], "mano_pose_delta": self.delta[index],
                "sample_id": torch.tensor(index, dtype=torch.long)}


class ManoTransitionParameterDataset(Dataset):
    """大规模 V2 数据：只保存 pose pair 参数，batch 内再执行 MANO。"""

    def __init__(self, *, mano_path: str | Path, side: str, num_samples: int, seed: int,
                 num_pca_comps: int, num_patches: int, patch_size: int,
                 motion_stds: tuple[float, float, float],
                 motion_probs: tuple[float, float, float], sparse_probability: float,
                 **_: object) -> None:
        self.side = str(side)
        pose, requested_delta = sample_pose_pairs(
            num_samples, num_pca_comps, seed, motion_stds, motion_probs, sparse_probability)
        after = (pose + requested_delta).clamp(-2.5, 2.5)
        self.pose = pose
        self.delta = after - pose
        self.clip_fraction = (after != pose + requested_delta).float().mean(-1)

        layer = MANO(str(mano_path), is_rhand=self.side == "right", use_pca=True,
                     num_pca_comps=num_pca_comps, flat_hand_mean=True)
        layer.eval().requires_grad_(False)
        faces = torch.as_tensor(layer.faces.astype(np.int64))
        with torch.no_grad():
            zero = layer(hand_pose=torch.zeros(1, num_pca_comps), betas=torch.zeros(1, 10),
                         global_orient=torch.zeros(1, 3), transl=torch.zeros(1, 3))
        self.canonical = face_centers(zero.vertices - zero.joints[:, :1], faces)[0].float()
        self.patch_knn_idx = build_canonical_patch_map(
            self.canonical, num_patches, patch_size)

    def __len__(self):
        return len(self.pose)

    def __getitem__(self, index):
        return {"hand_cano_points": self.canonical,
                "patch_knn_idx": self.patch_knn_idx,
                "mano_pose": self.pose[index], "mano_pose_delta": self.delta[index],
                "clip_fraction": self.clip_fraction[index],
                "sample_id": torch.tensor(index, dtype=torch.long)}
