"""平滑 synthetic MANO articulation trajectory。"""
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
