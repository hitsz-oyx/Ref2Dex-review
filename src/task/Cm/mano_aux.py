"""Differentiable MANO auxiliary branch for Cm hand-motion training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

# smplx may import the legacy chumpy dependency when opening MANO pickle
# assets.  The repository's GRAB adapter carries the same compatibility shim.
for _legacy_name, _legacy_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)

from smplx import MANO
from smplx.lbs import lbs


class ManoForwardAuxiliary(nn.Module):
    """Turn a predicted MANO PCA-pose increment into legal face-centre flow.

    The MANO layers are fixed assets, not trainable parameters.  Their forward
    pass deliberately remains differentiable so loss gradients reach the pose
    adapter that produces ``delta_pose``.
    """

    def __init__(self, model_dir: str | Path, *, num_pca_comps: int = 24) -> None:
        super().__init__()
        common = {
            "model_path": str(model_dir),
            "use_pca": True,
            "num_pca_comps": int(num_pca_comps),
            "flat_hand_mean": True,
        }
        self.right_mano = MANO(is_rhand=True, **common)
        self.left_mano = MANO(is_rhand=False, **common)
        self.right_mano.requires_grad_(False)
        self.left_mano.requires_grad_(False)
        self.right_mano.eval()
        self.left_mano.eval()
        self.register_buffer("right_faces", torch.as_tensor(self.right_mano.faces, dtype=torch.long), persistent=False)
        self.register_buffer("left_faces", torch.as_tensor(self.left_mano.faces, dtype=torch.long), persistent=False)

    @staticmethod
    def _apply_wrist(points: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bni,bij->bnj", points, rotation) + translation.unsqueeze(1)

    @staticmethod
    def _face_centres(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        return vertices[:, faces].mean(dim=2)

    @staticmethod
    def _match_betas(layer: MANO, betas: torch.Tensor) -> torch.Tensor:
        """Match stored GRAB beta width to the installed MANO asset."""
        num_betas = int(layer.shapedirs.shape[-1])
        if betas.shape[-1] == num_betas:
            return betas
        betas = betas[..., :num_betas]
        if betas.shape[-1] < num_betas:
            betas = torch.cat(
                [betas, torch.zeros(*betas.shape[:-1], num_betas - betas.shape[-1], device=betas.device, dtype=betas.dtype)],
                dim=-1,
            )
        return betas

    def _forward_side(
        self,
        layer: MANO,
        faces: torch.Tensor,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        v_template: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = hand_pose.shape[0]
        zeros = torch.zeros(batch_size, 3, device=hand_pose.device, dtype=hand_pose.dtype)
        if v_template.ndim == 2:
            v_template = v_template.unsqueeze(0).expand(batch_size, -1, -1)
        if v_template.shape != (batch_size, 778, 3):
            raise ValueError(f"Expected v_template [B, 778, 3], got {tuple(v_template.shape)}")
        # MANO.forward hard-codes its module-level template.  Calling LBS
        # directly lets every sample retain the subject-specific GRAB template
        # saved in Stage 4, including mixed-subject batches.
        expanded_hand_pose = torch.einsum("bi,ij->bj", hand_pose, layer.hand_components)
        full_pose = torch.cat([zeros, expanded_hand_pose], dim=1) + layer.pose_mean
        vertices, joints = lbs(
            self._match_betas(layer, betas),
            full_pose,
            v_template,
            layer.shapedirs,
            layer.posedirs,
            layer.J_regressor,
            layer.parents,
            layer.lbs_weights,
            pose2rot=True,
        )
        vertices = vertices - joints[:, :1]
        return self._face_centres(vertices, faces)

    def forward(
        self,
        *,
        current_pose: torch.Tensor,
        delta_pose: torch.Tensor,
        betas: torch.Tensor,
        v_template: torch.Tensor,
        is_right: torch.Tensor,
        mirror_x: torch.Tensor,
        wrist_rotation: torch.Tensor,
        wrist_translation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        next_pose = current_pose + delta_pose
        current_right = self._forward_side(self.right_mano, self.right_faces, current_pose, betas, v_template)
        next_right = self._forward_side(self.right_mano, self.right_faces, next_pose, betas, v_template)
        current_left = self._forward_side(self.left_mano, self.left_faces, current_pose, betas, v_template)
        next_left = self._forward_side(self.left_mano, self.left_faces, next_pose, betas, v_template)
        side_mask = is_right.bool()[:, None, None]
        current_local = torch.where(side_mask, current_right, current_left)
        next_local = torch.where(side_mask, next_right, next_left)

        mirror_scale = torch.ones(
            current_local.shape[0], 1, 3, device=current_local.device, dtype=current_local.dtype
        )
        mirror_scale[..., 0] = torch.where(
            mirror_x.bool(), mirror_scale.new_tensor(-1.0), mirror_scale.new_tensor(1.0)
        )
        current_local = current_local * mirror_scale
        next_local = next_local * mirror_scale

        rigid_next = self._apply_wrist(current_local, wrist_rotation, wrist_translation)
        full_next = self._apply_wrist(next_local, wrist_rotation, wrist_translation)
        return {
            "pred_mano_current_points": current_local,
            "pred_mano_next_points": full_next,
            "pred_mano_articulation_flow": full_next - rigid_next,
            "pred_mano_flow": full_next - current_local,
            "pred_mano_next_pose": next_pose,
        }
