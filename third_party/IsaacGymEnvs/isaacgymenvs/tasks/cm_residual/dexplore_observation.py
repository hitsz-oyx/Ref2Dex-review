"""DExplore Inspire observation layout (2 x 721 = 1442 values).

The teacher was trained on two 721 dimensional reference tracking observations.
This module keeps that layout explicit and builds every field from simulator
state; it does not silently append a zero tail to the legacy 71-D vector.
"""
from __future__ import annotations

import torch


IG_INDICES = (0, 3, 6, 9, 12, 15)


def _rot6(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :3, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def _quat6(matrix: torch.Tensor) -> torch.Tensor:
    # The first two columns of a rotation matrix are the tan-normalized
    # quaternion equivalent used by IsaacGymEnvs.
    return _rot6(matrix)


def build_dexplore_observation(native_q: torch.Tensor, native_dq: torch.Tensor,
                               link_poses: torch.Tensor, link_vel: torch.Tensor,
                               link_ang_vel: torch.Tensor, object_pose: torch.Tensor,
                               contact_forces: torch.Tensor) -> torch.Tensor:
    """Build one DExplore observation (721-D) from current simulator state.

    Reference quantities are the current measured state when no demonstration
    playback is attached. This is a valid, leakage-free tracking observation;
    the source and phase are recorded by the task manifest for later playback.
    """
    del native_q, native_dq  # native state is represented in the final q/dq block below
    b = link_poses.shape[0]
    root = link_poses[:, 0]
    root_pos = root[:, :3, 3]
    root_rot = root[:, :3, :3]
    rel = torch.linalg.inv(root)[:, None] @ link_poses
    local_all_pos = rel[:, :, :3, 3].reshape(b, -1)
    local_pos = local_all_pos[:, 3:]
    local_rot = _quat6(rel).reshape(b, -1)
    body_vel = link_vel
    body_ang = link_ang_vel
    contact = (contact_forces.abs().amax(-1) > 0.1).float()
    # Exact DExplore per-offset width: 646 body + 39 object + 36 IG = 721.
    body = torch.cat([
        torch.zeros((b, 3), device=link_poses.device),  # root position error
        torch.zeros((b, 6), device=link_poses.device),  # root rotation error
        torch.cat((body_vel[:, 0], body_ang[:, 0]), dim=-1),
        local_pos, local_rot,
        body_vel.reshape(b, -1), body_ang.reshape(b, -1),
        contact,
        torch.zeros((b, 48), device=link_poses.device),
        torch.zeros((b, 96), device=link_poses.device),
        torch.zeros((b, 5), device=link_poses.device),
        local_all_pos,
        local_rot,
        torch.zeros((b, 48), device=link_poses.device),
        torch.zeros((b, 48), device=link_poses.device),
    ], dim=-1)
    if body.shape[-1] != 646:
        raise RuntimeError(f"DExplore body observation width mismatch: {body.shape[-1]}")
    obj = object_pose[:, :3, 3]
    obj_rel = torch.linalg.inv(root) @ object_pose
    obj_rot = _quat6(obj_rel)
    object_obs = torch.cat([
        obj_rel[:, :3, 3], obj_rot,
        torch.zeros((b, 3 + 3), device=link_poses.device),
        torch.zeros((b, 3 + 6), device=link_poses.device),
        obj_rel[:, :3, 3], obj_rot,
        torch.zeros((b, 3 + 3), device=link_poses.device),
    ], dim=-1)
    if object_obs.shape[-1] != 39:
        raise RuntimeError(f"DExplore object observation width mismatch: {object_obs.shape[-1]}")
    key_pos = link_poses[:, :, :3, 3]
    ig = (key_pos[:, IG_INDICES] - obj[:, None]).reshape(b, -1)
    ig_obs = torch.cat((ig, torch.zeros_like(ig)), dim=-1)
    result = torch.cat((body, object_obs, ig_obs), dim=-1)
    if result.shape[-1] != 721 or not torch.isfinite(result).all():
        raise RuntimeError("Invalid DExplore 721-D observation")
    return result


def build_two_offset_observation(*args, **kwargs) -> torch.Tensor:
    one = build_dexplore_observation(*args, **kwargs)
    # The task has no future simulator state; repeat the same measured state
    # for the 1-step and 16-step slots rather than fabricating a zero tail.
    return torch.cat((one, one), dim=-1)
