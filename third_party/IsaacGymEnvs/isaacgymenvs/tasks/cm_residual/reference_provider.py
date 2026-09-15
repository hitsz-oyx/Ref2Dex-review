"""Reference playback for the DExplore 1-step/16-step observation offsets."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

from .contract import QUERY_LINKS


class ReferenceProvider:
    """Load one audited reference sequence and serve phase-indexed tensors."""

    def __init__(self, path: str | Path, device: torch.device | str):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Reference artifact not found: {self.path}")
        manifest = self.path.with_name("manifest.json")
        if not manifest.is_file():
            raise FileNotFoundError(f"Reference manifest not found: {manifest}")
        meta = json.loads(manifest.read_text())
        if meta.get("coordinate_frame") != "world" or meta.get("quaternion_order") != "xyzw":
            raise ValueError("Reference must use world-frame xyzw poses")
        payload = np.load(self.path, allow_pickle=False)
        required = ("q_native_ref", "dq_native_ref", "link_pose_world_ref", "object_pose_world_ref", "object_twist_world_ref", "phase")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Reference artifact missing fields: {missing}")
        self.device = torch.device(device)
        self.q = torch.as_tensor(payload["q_native_ref"], dtype=torch.float32, device=self.device)
        self.dq = torch.as_tensor(payload["dq_native_ref"], dtype=torch.float32, device=self.device)
        self.links = torch.as_tensor(payload["link_pose_world_ref"], dtype=torch.float32, device=self.device)
        self.object = torch.as_tensor(payload["object_pose_world_ref"], dtype=torch.float32, device=self.device)
        self.object_twist = torch.as_tensor(payload["object_twist_world_ref"], dtype=torch.float32, device=self.device)
        self.phase = torch.as_tensor(payload["phase"], dtype=torch.float32, device=self.device)
        self.dt = float(meta.get("control_dt", 1.0 / 30.0))
        if self.q.ndim != 2 or self.q.shape[1] != 18 or self.links.ndim != 4 or self.links.shape[1] != 25:
            raise ValueError("Reference q/link dimensions do not match Inspire contract")
        names = tuple(meta.get("link_order", ()))
        if len(names) != 25 or any(name not in names for name in QUERY_LINKS):
            raise ValueError("Reference manifest link_order is incompatible with QUERY_LINKS")
        self.query_indices = torch.as_tensor([names.index(name) for name in QUERY_LINKS], device=self.device)
        self.length = int(self.q.shape[0])
        self.link_velocity = self._finite_difference(self.links[:, :, :3, 3])
        self.link_ang_velocity = self._angular_velocity(self.links[:, :, :3, :3])

    def _finite_difference(self, values):
        velocity = torch.zeros_like(values)
        if self.length > 1:
            velocity[1:-1] = (values[2:] - values[:-2]) / (2.0 * self.dt)
            velocity[0] = (values[1] - values[0]) / self.dt
            velocity[-1] = (values[-1] - values[-2]) / self.dt
        return velocity

    def _angular_velocity(self, rotations):
        velocity = torch.zeros((self.length, rotations.shape[1], 3), device=rotations.device, dtype=rotations.dtype)
        if self.length <= 1:
            return velocity
        relative = rotations[1:] @ rotations[:-1].transpose(-1, -2)
        skew = torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                            relative[..., 0, 2] - relative[..., 2, 0],
                            relative[..., 1, 0] - relative[..., 0, 1]), dim=-1) / (2.0 * self.dt)
        velocity[1:] = skew
        velocity[0] = skew[0]
        return velocity

    def frame(self, indices: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, ...]:
        idx = indices.to(self.device, dtype=torch.long).clamp(0, self.length - 1)
        idx = (idx + int(offset)).clamp(0, self.length - 1)
        return (self.q.index_select(0, idx), self.dq.index_select(0, idx),
                self.links.index_select(0, idx).index_select(1, self.query_indices),
                self.object.index_select(0, idx), self.phase.index_select(0, idx),
                self.object_twist.index_select(0, idx),
                self.link_velocity.index_select(0, idx).index_select(1, self.query_indices),
                self.link_ang_velocity.index_select(0, idx).index_select(1, self.query_indices))
