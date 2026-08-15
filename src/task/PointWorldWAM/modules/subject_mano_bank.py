from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import trimesh

from ..mano_action import MANO, compose_relative_action


class SubjectManoBank(nn.Module):
    """Frozen subject-specific MANO layers with mixed-subject batch dispatch."""

    def __init__(
        self,
        mano_model_dir: str,
        subject_template_root: str,
        subjects: Sequence[str],
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleDict()
        template_root = Path(subject_template_root)
        for subject in sorted(set(subjects)):
            for side, is_right, suffix in (
                ("left", False, "lhand.ply"),
                ("right", True, "rhand.ply"),
            ):
                matches = list(template_root.glob(f"*/{subject}_{suffix}"))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"{subject}/{side} v_template 应唯一，实际为 {matches}"
                    )
                vertices = trimesh.load(matches[0], process=False).vertices.astype(
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
                key = self._key(subject, side)
                self.layers[key] = layer
                self.register_buffer(
                    f"{key}_faces",
                    torch.from_numpy(np.asarray(layer.faces, dtype=np.int64)),
                )

    @staticmethod
    def _key(subject: str, side: str) -> str:
        return f"{subject}_{side}"

    def points_from_action_chunk(
        self,
        subject_ids: Sequence[str],
        side: str,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        transl: torch.Tensor,
        betas: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        if action_chunk.ndim != 3 or action_chunk.shape[-1] != 30:
            raise ValueError(f"action chunk 应为 [B,K,30]，实际 {action_chunk.shape}")
        batch_size, chunk_size = action_chunk.shape[:2]
        if len(subject_ids) != batch_size:
            raise ValueError("subject_ids 与 batch size 不一致")
        output = action_chunk.new_zeros(batch_size, chunk_size, 1538, 3)
        for subject in sorted(set(subject_ids)):
            indices = torch.tensor(
                [i for i, value in enumerate(subject_ids) if value == subject],
                device=action_chunk.device,
                dtype=torch.long,
            )
            action = action_chunk.index_select(0, indices)
            count = len(indices)
            orient = global_orient.index_select(0, indices)[:, None].expand(
                -1, chunk_size, -1
            )
            pose = hand_pose.index_select(0, indices)[:, None].expand(
                -1, chunk_size, -1
            )
            translation = transl.index_select(0, indices)[:, None].expand(
                -1, chunk_size, -1
            )
            shape = betas.index_select(0, indices)[:, None].expand(
                -1, chunk_size, -1
            )
            future_orient, future_pose, future_transl = compose_relative_action(
                orient.reshape(-1, 3),
                pose.reshape(-1, 24),
                translation.reshape(-1, 3),
                action.reshape(-1, 30),
            )
            vertices = self.layers[self._key(subject, side)](
                global_orient=future_orient,
                hand_pose=future_pose,
                transl=future_transl,
                betas=shape.reshape(-1, 10),
            ).vertices
            faces = getattr(self, f"{self._key(subject, side)}_faces")
            points = vertices[:, faces].mean(2).reshape(count, chunk_size, 1538, 3)
            output = output.index_copy(0, indices, points)
        return output
