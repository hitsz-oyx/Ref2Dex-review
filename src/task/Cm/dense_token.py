"""Frozen adapter around the existing correspondence_ptv3_v2 dense-token checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.base import task_config_from_dict
from src.task.correspondence_ptv3_v2.model import StaticHOCPTv3V2


class FrozenDenseTokenEncoder(nn.Module):
    """Expose frozen current-frame dense interaction tokens and contact prior.

    The wrapped model receives `(O_t, H_t)` only.  In particular, neither
    `hand_flow` nor any future object field enters the old PTv3 encoder.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        super().__init__()
        self.checkpoint = str(Path(checkpoint).resolve())
        checkpoint_data = torch.load(self.checkpoint, map_location="cpu")
        if "config" not in checkpoint_data or "model" not in checkpoint_data:
            raise ValueError(f"Unsupported dense-token checkpoint: {checkpoint}")
        self.cfg = task_config_from_dict(checkpoint_data["config"])
        meta = self.cfg.meta
        # Older checkpoints record the training host's absolute PTv3 path.
        # The backbone source is repository-local, so relocate only a missing
        # path rather than requiring users to recreate the dense checkpoint.
        if not Path(str(meta.ptv3_repo_path)).is_dir():
            local_ptv3 = Path(__file__).resolve().parents[3] / "third_party" / "PointTransformerV3"
            if not local_ptv3.is_dir():
                raise FileNotFoundError(
                    f"PointTransformerV3 is missing at checkpoint path {meta.ptv3_repo_path!r} "
                    f"and local fallback {local_ptv3}."
                )
            meta.ptv3_repo_path = str(local_ptv3)
        if int(meta.num_obj_points) != 512 or int(meta.num_hand_points) != 1538:
            raise ValueError(
                "Cm bootstrap expects the current 512-object / 1538-hand dense-token checkpoint, "
                f"got {meta.num_obj_points} / {meta.num_hand_points}."
            )
        self.model = StaticHOCPTv3V2(self.cfg)
        self.model.load_state_dict(checkpoint_data["model"], strict=True)
        self.model.requires_grad_(False)
        self.model.eval()
        self.token_dim = int(self.model.token_dim)
        self.num_obj_points = int(meta.num_obj_points)
        self.num_hand_points = int(meta.num_hand_points)

    @torch.no_grad()
    def forward(
        self,
        *,
        obj_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.model.eval()
        batch_size = obj_points.shape[0]
        points = torch.cat([obj_points.float(), hand_points.float()], dim=1)
        normals = torch.cat([obj_normals.float(), hand_normals.float()], dim=1)
        point_valid_mask = torch.cat(
            [obj_valid_mask.bool(), torch.ones_like(hand_points[..., 0], dtype=torch.bool)], dim=1
        )
        # `encode_points` checks these keys for its original batch contract but
        # does not consume their values.  A single invalid placeholder avoids
        # fabricating an edge-supervision target during Cm extraction.
        empty_edges = torch.zeros(
            batch_size, self.num_obj_points, 1, device=points.device, dtype=torch.long
        )
        batch = {
            "points": points,
            "normals": normals,
            "point_valid_mask": point_valid_mask,
            "runtime_obj_valid_mask": obj_valid_mask.bool(),
            "random_edge_idx": empty_edges,
            "random_edge_valid_mask": torch.zeros_like(empty_edges, dtype=torch.bool),
        }
        z_obj, z_hand = self.model.encode_points(batch)
        if self.model.use_hand_contact_head:
            hand_contact = torch.sigmoid(self.model.hand_contact_head(z_hand).squeeze(-1))
        else:
            hand_contact = torch.zeros_like(z_hand[..., 0])
        return z_obj, z_hand, hand_contact
