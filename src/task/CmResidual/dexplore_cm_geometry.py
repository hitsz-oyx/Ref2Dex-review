"""Task-local geometry bridge from DExplore live state to frozen-Cmv2 inputs.

This module deliberately has no Isaac Gym or PPO dependency.  It preserves
the released DExplore action interpretation and uses the pinned Inspire URDF
to turn pre-action native state plus nominal policy actions into the 1024/1538
world-space geometry required by Cmv2.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import torch

from src.task.CmResidual.v118_planner import ACTION_DIM, QUERY_LINKS, TorchInspireKinematics


OBJECT_POINTS = 1024
HAND_POINTS = 1538
DELTA_TIME_S = 1.0 / 30.0


def _surface_geometry_class():
    """Load the geometry-only vendor module without importing its task registry."""
    name = "ref2dex_cmv2_surface_geometry"
    if name not in sys.modules:
        path = (Path(__file__).resolve().parents[3] / "third_party/IsaacGymEnvs/isaacgymenvs/"
                "tasks/cm_residual/cm_geometry.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].SurfaceGeometry


def _require(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.ndim != len(shape) or tuple(value.shape[-len(shape) + 1:]) != shape[1:]:
        raise ValueError(f"{name} must have shape {shape} up to batch dimension, got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} must be finite")


def dexplore_action_to_native_targets(action: torch.Tensor, current_native: torch.Tensor,
                                      lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Exact `Dexplore_Inspire._action_to_pd_targets` mapping, out of vendor code."""
    _require("action", action, (-1, ACTION_DIM))
    _require("current_native", current_native, (-1, ACTION_DIM))
    if action.shape != current_native.shape or lower.shape != (ACTION_DIM,) or upper.shape != (ACTION_DIM,):
        raise ValueError("action/current_native must match [B,18]; limits must be [18]")
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or (upper < lower).any():
        raise ValueError("invalid DExplore native limits")
    scale = (upper - lower).to(device=action.device, dtype=action.dtype).clone()
    scale[:3] = 1.0
    scale[3:6] = torch.pi
    normalized = action.clamp(-1.0, 1.0)
    pd_action = torch.cat((normalized[:, :6], (1.0 + normalized[:, 6:]) * 0.5), dim=-1)
    targets = scale * pd_action
    targets[:, :6] += current_native[:, :6]
    targets = targets.clone()
    targets[:, 7] = targets[:, 6] * 1.05
    targets[:, 9] = targets[:, 8] * 1.05
    targets[:, 11] = targets[:, 10] * 1.05
    targets[:, 13] = targets[:, 12] * 1.05
    targets[:, 16] = targets[:, 15] * 0.6
    targets[:, 17] = targets[:, 15] * 0.8
    return targets


def dexplore_root_pose(root_state: torch.Tensor) -> torch.Tensor:
    """Convert DExplore Isaac-Gym root `[x,y,z,qx,qy,qz,qw,...]` to `[B,4,4]`."""
    _require("root_state", root_state, (-1, 13))
    quat = torch.nn.functional.normalize(root_state[:, 3:7], dim=-1, eps=1e-8)
    x, y, z, w = quat.unbind(-1)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz, wx, wy, wz = x*y, x*z, y*z, w*x, w*y, w*z
    rotation = torch.stack((
        1 - 2*(yy + zz), 2*(xy - wz), 2*(xz + wy),
        2*(xy + wz), 1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy), 2*(yz + wx), 1 - 2*(xx + yy),
    ), dim=-1).view(-1, 3, 3)
    pose = torch.eye(4, device=root_state.device, dtype=root_state.dtype).expand(root_state.shape[0], 4, 4).clone()
    pose[:, :3, :3] = rotation
    pose[:, :3, 3] = root_state[:, :3]
    return pose


@dataclass(frozen=True)
class DExploreCmv2Geometry:
    object_points: torch.Tensor
    object_normals: torch.Tensor
    hand_points: torch.Tensor
    hand_normals: torch.Tensor
    object_pose: torch.Tensor
    link_poses: torch.Tensor


class DExploreCmv2GeometryBridge:
    """Pinned 1024/1538 geometry and native-action nominal sweep builder."""

    def __init__(self, *, hand_urdf: str | Path, object_urdf: str | Path,
                 device: torch.device | str, seed: int = 42, geometry=None) -> None:
        self.device = torch.device(device)
        self.hand_urdf = Path(hand_urdf).resolve()
        self.object_urdf = Path(object_urdf).resolve()
        if not self.hand_urdf.is_file() or not self.object_urdf.is_file():
            raise FileNotFoundError("DExplore Cm geometry URDF is missing")
        self.kinematics = TorchInspireKinematics(self.hand_urdf, self.device)
        self.geometry = geometry if geometry is not None else _surface_geometry_class()(
            hand_urdf=self.hand_urdf, object_urdf=self.object_urdf, query_links=QUERY_LINKS,
            object_count=OBJECT_POINTS, hand_count=HAND_POINTS, seed=seed, device=self.device)

    def current(self, current_native: torch.Tensor, object_root_state: torch.Tensor) -> DExploreCmv2Geometry:
        _require("current_native", current_native, (-1, ACTION_DIM))
        _require("object_root_state", object_root_state, (-1, 13))
        if current_native.shape[0] != object_root_state.shape[0] or current_native.device != self.device:
            raise ValueError("live DExplore state must share the bridge device and batch")
        link_poses = self.kinematics.forward(current_native[:, None])[:, 0]
        object_pose = dexplore_root_pose(object_root_state)
        object_points, object_normals = self.geometry.object(object_pose)
        hand_points, hand_normals = self.geometry.hand(link_poses)
        expected = current_native.shape[0]
        for name, value, tail in (("object_points", object_points, (OBJECT_POINTS, 3)),
                                  ("object_normals", object_normals, (OBJECT_POINTS, 3)),
                                  ("hand_points", hand_points, (HAND_POINTS, 3)),
                                  ("hand_normals", hand_normals, (HAND_POINTS, 3))):
            if value.shape != (expected, *tail) or not torch.isfinite(value).all():
                raise RuntimeError(f"invalid bridge {name}: {tuple(value.shape)}")
        return DExploreCmv2Geometry(object_points, object_normals, hand_points, hand_normals,
                                    object_pose, link_poses)

    def nominal_hand_sweep(self, current_native: torch.Tensor, candidate_actions: torch.Tensor,
                           lower: torch.Tensor, upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return current hand points/normals and `[B,K,1538,3]` nominal flow."""
        _require("current_native", current_native, (-1, ACTION_DIM))
        if candidate_actions.ndim != 3 or candidate_actions.shape[0] != current_native.shape[0] or candidate_actions.shape[-1] != ACTION_DIM:
            raise ValueError("candidate_actions must be [B,K,18]")
        batch, candidates = candidate_actions.shape[:2]
        current_links = self.kinematics.forward(current_native[:, None])[:, 0]
        current_points, current_normals = self.geometry.hand(current_links)
        expanded_current = current_native[:, None].expand(-1, candidates, -1).reshape(-1, ACTION_DIM)
        targets = dexplore_action_to_native_targets(candidate_actions.reshape(-1, ACTION_DIM), expanded_current,
                                                     lower, upper)
        next_links = self.kinematics.forward(targets.view(batch, candidates, ACTION_DIM))
        next_points, _ = self.geometry.hand(next_links.reshape(-1, len(QUERY_LINKS), 4, 4))
        flow = next_points.view(batch, candidates, HAND_POINTS, 3) - current_points[:, None]
        if not torch.isfinite(flow).all():
            raise RuntimeError("non-finite nominal DExplore hand flow")
        return current_points, current_normals, flow
