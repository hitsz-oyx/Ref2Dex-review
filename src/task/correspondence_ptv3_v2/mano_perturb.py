from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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


class ManoPcaHandPerturber:
    def __init__(self, model_dir: str | Path, *, num_pca_comps: int) -> None:
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
        self.right_faces = torch.as_tensor(self.right_mano.faces, dtype=torch.long)
        self.left_faces = torch.as_tensor(self.left_mano.faces, dtype=torch.long)

    @staticmethod
    def _match_betas(layer: MANO, betas: torch.Tensor) -> torch.Tensor:
        num_betas = int(layer.shapedirs.shape[-1])
        if betas.shape[-1] == num_betas:
            return betas
        betas = betas[..., :num_betas]
        if betas.shape[-1] < num_betas:
            pad = torch.zeros(
                *betas.shape[:-1],
                num_betas - betas.shape[-1],
                device=betas.device,
                dtype=betas.dtype,
            )
            betas = torch.cat([betas, pad], dim=-1)
        return betas

    @staticmethod
    def _face_centers(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        return vertices[:, faces].mean(dim=2)

    @staticmethod
    def _face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        tri = vertices[:, faces]
        edge1 = tri[:, :, 1] - tri[:, :, 0]
        edge2 = tri[:, :, 2] - tri[:, :, 0]
        normal = torch.cross(edge1, edge2, dim=-1)
        return normal / torch.clamp(torch.linalg.norm(normal, dim=-1, keepdim=True), min=1e-8)

    def _forward_side(
        self,
        layer: MANO,
        faces: torch.Tensor,
        *,
        hand_pose: torch.Tensor,
        betas: torch.Tensor,
        v_template: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(hand_pose.shape[0])
        zeros = torch.zeros(batch_size, 3, device=hand_pose.device, dtype=hand_pose.dtype)
        if v_template.ndim == 2:
            v_template = v_template.unsqueeze(0)
        v_template = v_template.expand(batch_size, -1, -1)
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
        return self._face_centers(vertices, faces), self._face_normals(vertices, faces)

    def reconstruct(
        self,
        *,
        hand_pose: np.ndarray,
        betas: np.ndarray,
        v_template: np.ndarray,
        is_right: bool,
        mirror_x: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        hand_pose_t = torch.as_tensor(np.asarray(hand_pose, dtype=np.float32)).view(1, -1)
        betas_t = torch.as_tensor(np.asarray(betas, dtype=np.float32)).view(1, -1)
        v_template_t = torch.as_tensor(np.asarray(v_template, dtype=np.float32))
        with torch.no_grad():
            if is_right:
                points_t, normals_t = self._forward_side(
                    self.right_mano,
                    self.right_faces,
                    hand_pose=hand_pose_t,
                    betas=betas_t,
                    v_template=v_template_t,
                )
            else:
                points_t, normals_t = self._forward_side(
                    self.left_mano,
                    self.left_faces,
                    hand_pose=hand_pose_t,
                    betas=betas_t,
                    v_template=v_template_t,
                )
        points = points_t[0].detach().cpu().numpy().astype(np.float32)
        normals = normals_t[0].detach().cpu().numpy().astype(np.float32)
        if mirror_x:
            points[:, 0] *= -1.0
            normals[:, 0] *= -1.0
        return points, normals
