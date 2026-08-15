from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import trimesh

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

from smplx import MANO


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    theta2 = (axis_angle * axis_angle).sum(-1, keepdim=True)
    theta = theta2.sqrt()
    x, y, z = axis_angle.unbind(-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(*axis_angle.shape[:-1], 3, 3)
    a = torch.where(
        theta2 > 1e-8,
        torch.sin(theta) / theta.clamp_min(1e-8),
        1.0 - theta2 / 6.0,
    )
    b = torch.where(
        theta2 > 1e-8,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-8),
        0.5 - theta2 / 24.0,
    )
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    vector = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    ) * 0.5
    sin_angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    cos_angle = ((matrix.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5).unsqueeze(-1)
    angle = torch.atan2(sin_angle, cos_angle.clamp(-1.0, 1.0))
    scale = torch.where(
        sin_angle > 1e-7,
        angle / sin_angle.clamp_min(1e-7),
        torch.ones_like(sin_angle),
    )
    return vector * scale


def compose_relative_action(
    global_orient: torch.Tensor,
    hand_pose: torch.Tensor,
    transl: torch.Tensor,
    action: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    future_rotation = axis_angle_to_matrix(global_orient) @ axis_angle_to_matrix(
        action[..., 3:6]
    )
    return (
        matrix_to_axis_angle(future_rotation),
        hand_pose + action[..., 6:30],
        transl + action[..., :3],
    )


class BimanualManoAction(nn.Module):
    def __init__(
        self,
        mano_model_dir: str,
        left_vtemplate: str,
        right_vtemplate: str,
        statistics: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict()
        for side, is_right, template_path in (
            ("left", False, left_vtemplate),
            ("right", True, right_vtemplate),
        ):
            vertices = trimesh.load(Path(template_path), process=False).vertices.astype(
                np.float32
            )
            layer = MANO(
                mano_model_dir,
                is_rhand=is_right,
                use_pca=True,
                num_pca_comps=24,
                flat_hand_mean=True,
                v_template=vertices,
            )
            layer.requires_grad_(False)
            self.layers[side] = layer
            self.register_buffer(
                f"{side}_faces", torch.from_numpy(np.asarray(layer.faces, np.int64))
            )
            self.register_buffer(f"{side}_action_mean", statistics[f"{side}_mean"].clone())
            self.register_buffer(f"{side}_action_std", statistics[f"{side}_std"].clone())

    def normalize(self, side: str, action: torch.Tensor) -> torch.Tensor:
        return (action - getattr(self, f"{side}_action_mean")) / getattr(
            self, f"{side}_action_std"
        )

    def denormalize(self, side: str, action: torch.Tensor) -> torch.Tensor:
        return action * getattr(self, f"{side}_action_std") + getattr(
            self, f"{side}_action_mean"
        )

    def points_from_action(
        self,
        side: str,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        transl: torch.Tensor,
        betas: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        action = self.denormalize(side, normalized_action)
        orient, pose, future_transl = compose_relative_action(
            global_orient, hand_pose, transl, action
        )
        vertices = self.layers[side](
            global_orient=orient,
            hand_pose=pose,
            transl=future_transl,
            betas=betas,
        ).vertices
        faces = getattr(self, f"{side}_faces")
        return vertices[:, faces].mean(2)
