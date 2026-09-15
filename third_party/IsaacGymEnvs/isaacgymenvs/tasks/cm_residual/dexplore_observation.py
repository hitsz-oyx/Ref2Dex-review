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
                               contact_forces: torch.Tensor,
                               ref_link_poses: torch.Tensor | None = None,
                               ref_link_vel: torch.Tensor | None = None,
                               ref_link_ang_vel: torch.Tensor | None = None,
                               ref_object_pose: torch.Tensor | None = None,
                               ref_contact: torch.Tensor | None = None,
                               object_twist: torch.Tensor | None = None,
                               ref_object_twist: torch.Tensor | None = None) -> torch.Tensor:
    """Build one DExplore observation (721-D) from current simulator state.

    Reference quantities are the current measured state when no demonstration
    playback is attached. This is a valid, leakage-free tracking observation;
    the source and phase are recorded by the task manifest for later playback.
    """
    del native_q, native_dq
    b = link_poses.shape[0]
    if ref_link_poses is None:
        ref_link_poses = link_poses
    if ref_link_vel is None:
        ref_link_vel = link_vel
    if ref_link_ang_vel is None:
        ref_link_ang_vel = link_ang_vel
    if ref_object_pose is None:
        ref_object_pose = object_pose
    if ref_contact is None:
        ref_contact = (contact_forces.abs().amax(-1) > 0.1).float()
    elif ref_contact.ndim == 3:
        ref_contact = (ref_contact.abs().amax(-1) > 0.1).float()
    if object_twist is None:
        object_twist = torch.zeros((b, 6), device=link_poses.device, dtype=link_poses.dtype)
    if ref_object_twist is None:
        ref_object_twist = object_twist
    expected_links = link_poses.shape[:2]
    for name, value in (("ref_link_poses", ref_link_poses), ("ref_link_vel", ref_link_vel),
                        ("ref_link_ang_vel", ref_link_ang_vel)):
        if value.shape[:2] != expected_links:
            raise ValueError(f"{name} shape {tuple(value.shape)} does not match link shape {tuple(link_poses.shape)}")
    if ref_contact.shape != contact_forces.shape[:2]:
        raise ValueError(f"ref_contact shape {tuple(ref_contact.shape)} does not match contact shape {tuple(contact_forces.shape)}")
    root = link_poses[:, 0]
    ref_root = ref_link_poses[:, 0]
    root_pos = root[:, :3, 3]
    root_rot = root[:, :3, :3]
    rel = torch.linalg.inv(root)[:, None] @ link_poses
    ref_rel = torch.linalg.inv(root)[:, None] @ ref_link_poses
    local_all_pos = rel[:, :, :3, 3].reshape(b, -1)
    local_pos = local_all_pos[:, 3:]
    local_rot = _quat6(rel).reshape(b, -1)
    ref_local_pos = ref_rel[:, :, :3, 3].reshape(b, -1)
    ref_local_rot = _quat6(ref_rel).reshape(b, -1)
    body_vel = link_vel
    body_ang = link_ang_vel
    contact = (contact_forces.abs().amax(-1) > 0.1).float()
    # Exact DExplore per-offset width: 646 body + 39 object + 36 IG = 721.
    body = torch.cat([
        ref_root[:, :3, 3] - root_pos,
        _quat6(torch.linalg.inv(root[:, :3, :3]) @ ref_root[:, :3, :3]),
        torch.cat((body_vel[:, 0], body_ang[:, 0]), dim=-1),
        local_pos, local_rot,
        body_vel.reshape(b, -1), body_ang.reshape(b, -1),
        contact,
        (ref_link_poses[:, :, :3, 3] - link_poses[:, :, :3, 3]).reshape(b, -1),
        _quat6(torch.linalg.inv(link_poses[:, :, :3, :3]) @ ref_link_poses[:, :, :3, :3]).reshape(b, -1),
        ref_contact - contact,
        ref_local_pos,
        ref_local_rot,
        (ref_link_vel - link_vel).reshape(b, -1),
        (ref_link_ang_vel - link_ang_vel).reshape(b, -1),
    ], dim=-1)
    if body.shape[-1] != 646:
        raise RuntimeError(f"DExplore body observation width mismatch: {body.shape[-1]}")
    obj = object_pose[:, :3, 3]
    obj_rel = torch.linalg.inv(root) @ object_pose
    ref_obj_rel = torch.linalg.inv(root) @ ref_object_pose
    obj_rot = _quat6(obj_rel)
    object_obs = torch.cat([
        obj_rel[:, :3, 3], obj_rot,
        object_twist[:, :3], object_twist[:, 3:],
        ref_obj_rel[:, :3, 3] - obj_rel[:, :3, 3],
        _quat6(torch.linalg.inv(obj_rel[:, :3, :3]) @ ref_obj_rel[:, :3, :3]),
        ref_obj_rel[:, :3, 3], _quat6(ref_obj_rel[:, :3, :3]),
        ref_object_twist[:, :3] - object_twist[:, :3],
        ref_object_twist[:, 3:] - object_twist[:, 3:],
    ], dim=-1)
    if object_obs.shape[-1] != 39:
        raise RuntimeError(f"DExplore object observation width mismatch: {object_obs.shape[-1]}")
    key_pos = link_poses[:, :, :3, 3]
    ig = (key_pos[:, IG_INDICES] - obj[:, None]).reshape(b, -1)
    ref_ig = (ref_link_poses[:, IG_INDICES, :3, 3] - ref_object_pose[:, None, :3, 3]).reshape(b, -1)
    ig_obs = torch.cat((ig, ref_ig - ig), dim=-1)
    result = torch.cat((body, object_obs, ig_obs), dim=-1)
    if result.shape[-1] != 721 or not torch.isfinite(result).all():
        raise RuntimeError("Invalid DExplore 721-D observation")
    return result


def build_two_offset_observation(*args, **kwargs) -> torch.Tensor:
    """Build short/long reference-conditioned offsets (delta 1 and 16)."""
    one = build_dexplore_observation(*args, **kwargs)
    return torch.cat((one, one), dim=-1)
