"""Pure, deterministic ranking primitives for the CmResidual V1.21c gate.

The module intentionally has no Isaac Gym, DExplore-agent, or checkpoint import.  It
contains the parts of the V1.21c contract that can be tested with tensors alone:
candidate generation, state identity and phase selection, canonical interaction
geometry, the Cmv2 score, and episode-blocked ranking statistics.  The simulator
collector is expected to provide the tensors consumed here and must keep the same
native-action and reference semantics as the released DExplore task.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import math
from typing import Sequence

import numpy as np
import torch


ACTION_DIM = 18
CANDIDATE_COUNT = 8
H_REF = 6
IG_KEY_INDICES = (0, 3, 6, 9, 12, 15)
IG_POINT_COUNT = 6
IG_SIGMA_M = 0.02
OBJECT_POSITION_SIGMA_M = 0.02
OBJECT_ROTATION_SIGMA_RAD = 0.05
CM_TIE_TOLERANCE = 1e-6
PHYSX_MIN_TOLERANCE = 1e-6

PHASE_MOVING = 0
PHASE_CONTACT = 1
PHASE_PRECONTACT = 2
PHASE_NAMES = {PHASE_MOVING: "moving", PHASE_CONTACT: "contact", PHASE_PRECONTACT: "precontact"}
PHASE_QUOTAS = {PHASE_MOVING: 192, PHASE_CONTACT: 192, PHASE_PRECONTACT: 128}


class CollectionInsufficient(RuntimeError):
    """Raised when a collection batch cannot fill all fixed phase quotas."""


def _finite(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite torch.Tensor")


def _check_last(name: str, value: torch.Tensor, size: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must have last dimension {size}")
    _finite(name, value)


def _finite_where(name: str, value: torch.Tensor, valid: torch.Tensor) -> None:
    """Require finite values only on entries that participate in a metric."""
    if not torch.isfinite(value[valid]).all():
        raise ValueError(f"{name} has non-finite valid entries")


def candidate_seed(collection_batch_id: str, state_id: str) -> int:
    """Return the V1.21c uint64 little-endian candidate seed."""
    if not collection_batch_id or not state_id:
        raise ValueError("collection_batch_id and state_id are required")
    digest = hashlib.sha256((str(collection_batch_id) + str(state_id)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def generate_candidate_actions(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    collection_batch_id: str,
    state_ids: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate deterministic native-action candidates around policy ``mu``.

    Candidate zero is the clipped policy mean.  Candidates one through seven use
    iid standard-normal CPU samples from a per-state ``torch.Generator`` and are
    clipped only in native action space.  The return tensor is float32 even when a
    caller supplied another floating dtype, matching the replay schema.
    """
    if mu.ndim != 2 or tuple(mu.shape[1:]) != (ACTION_DIM,):
        raise ValueError("mu must be [B,18]")
    if sigma.shape != mu.shape:
        raise ValueError("sigma must match mu [B,18]")
    if not mu.is_floating_point() or not sigma.is_floating_point():
        raise ValueError("mu and sigma must be floating point")
    _finite("mu", mu)
    _finite("sigma", sigma)
    if (sigma < 0).any():
        raise ValueError("policy sigma must be non-negative")
    if len(state_ids) != mu.shape[0] or any(not str(item) for item in state_ids):
        raise ValueError("state_ids must contain one non-empty id per state")

    seeds = torch.tensor(
        [candidate_seed(collection_batch_id, str(state_id)) for state_id in state_ids],
        dtype=torch.uint64,
        device=mu.device,
    )
    result = torch.empty((mu.shape[0], CANDIDATE_COUNT, ACTION_DIM), dtype=torch.float32,
                         device=mu.device)
    mean = mu.to(dtype=torch.float32)
    std = sigma.to(dtype=torch.float32)
    result[:, 0] = mean.clamp(-1.0, 1.0)
    for batch_index, seed in enumerate(seeds.detach().cpu().tolist()):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        epsilon = torch.randn((CANDIDATE_COUNT - 1, ACTION_DIM), generator=generator,
                              dtype=torch.float32)
        result[batch_index, 1:] = (mean[batch_index].cpu() + std[batch_index].cpu() * epsilon
                                   ).clamp(-1.0, 1.0).to(mu.device)
    if not torch.isfinite(result).all():
        raise FloatingPointError("candidate actions are non-finite")
    return result, seeds


def canonical_state_id(
    checkpoint_sha256: str,
    seed: int,
    episode_id: int,
    frame_id: int,
    raw_obs: torch.Tensor | np.ndarray,
    executed_action_prefix_sha256: str,
) -> str:
    """Hash the canonical V1.21c state identity.

    ``raw_obs`` is converted to contiguous little-endian float32 bytes; all scalar
    fields use their unambiguous ASCII representation.  Prefix hashing is accepted
    as a digest string because the collector can stream the action history without
    retaining a second copy of the prefix.
    """
    if len(checkpoint_sha256) != 64 or len(executed_action_prefix_sha256) != 64:
        raise ValueError("checkpoint and action-prefix SHA256 values are required")
    if isinstance(raw_obs, torch.Tensor):
        if raw_obs.numel() != 1442:
            raise ValueError("raw_obs must contain 1442 values")
        obs = raw_obs.detach().cpu().numpy().astype("<f4", copy=True).reshape(-1)
    else:
        obs = np.asarray(raw_obs, dtype="<f4").reshape(-1)
        if obs.size != 1442:
            raise ValueError("raw_obs must contain 1442 values")
        obs = np.ascontiguousarray(obs)
    if not np.isfinite(obs).all():
        raise ValueError("raw_obs must be finite")
    payload = b"".join((
        checkpoint_sha256.lower().encode("ascii"),
        int(seed).to_bytes(8, "little", signed=True),
        int(episode_id).to_bytes(8, "little", signed=True),
        int(frame_id).to_bytes(8, "little", signed=True),
        obs.tobytes(order="C"),
        executed_action_prefix_sha256.lower().encode("ascii"),
    ))
    return hashlib.sha256(payload).hexdigest()


def active_phase(
    actual_contact_force: torch.Tensor,
    actual_tip_distance_m: torch.Tensor,
    reference_contact_label: torch.Tensor,
    reference_motion_translation_m: torch.Tensor,
    reference_motion_rotation_rad: torch.Tensor,
    reference_tip_distance_m: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the V1.21c active formula and mutually exclusive phase priority.

    Returns ``(active, reason_mask, phase_id)``.  Reason bits are actual contact,
    actual near, reference contact, and reference moving+near respectively.
    """
    tensors = (actual_contact_force, actual_tip_distance_m, reference_contact_label,
               reference_motion_translation_m, reference_motion_rotation_rad,
               reference_tip_distance_m)
    if actual_contact_force.ndim != 3 or actual_contact_force.shape[-1] != 3:
        raise ValueError("actual_contact_force must be [B,5,3]")
    batch = actual_contact_force.shape[0]
    for name, value in zip(("actual_contact_force", "actual_tip_distance_m",
                            "reference_contact_label", "reference_motion_translation_m",
                            "reference_motion_rotation_rad", "reference_tip_distance_m"), tensors):
        if value.shape[0] != batch:
            raise ValueError(f"{name} batch mismatch")
        _finite(name, value)
    if actual_contact_force.shape[1] != 5:
        raise ValueError("actual_contact_force must contain the five contact bodies")
    for name, value in (("actual_tip_distance_m", actual_tip_distance_m),
                        ("reference_contact_label", reference_contact_label),
                        ("reference_tip_distance_m", reference_tip_distance_m)):
        if value.ndim != 2 or value.shape[1] != IG_POINT_COUNT - 1:
            raise ValueError(f"{name} must be [B,5]")
    for name, value in (("reference_motion_translation_m", reference_motion_translation_m),
                        ("reference_motion_rotation_rad", reference_motion_rotation_rad)):
        if value.ndim != 1:
            raise ValueError(f"{name} must be [B]")
    actual_contact = (actual_contact_force.abs().amax(dim=(-1, -2)) > 0.1)
    actual_near = actual_tip_distance_m.amin(dim=-1) <= 0.04
    reference_contact = (reference_contact_label > 0.5).any(dim=-1)
    reference_moving = ((reference_motion_translation_m >= 0.002) |
                        (reference_motion_rotation_rad >= 0.01))
    reference_near = reference_tip_distance_m.amin(dim=-1) <= 0.04
    reference_moving_near = reference_moving & reference_near
    active = actual_contact | actual_near | reference_contact | reference_moving_near
    reason = (actual_contact.to(torch.uint8) |
              (actual_near.to(torch.uint8) << 1) |
              (reference_contact.to(torch.uint8) << 2) |
              (reference_moving_near.to(torch.uint8) << 3))
    phase = torch.full((batch,), PHASE_PRECONTACT, dtype=torch.int64,
                       device=actual_contact_force.device)
    contact = (~reference_moving) & (actual_contact | reference_contact)
    phase = torch.where(reference_moving, torch.full_like(phase, PHASE_MOVING), phase)
    phase = torch.where(contact, torch.full_like(phase, PHASE_CONTACT), phase)
    # Inactive samples have no phase and cannot enter a quota bucket.
    phase = torch.where(active, phase, torch.full_like(phase, -1))
    return active, reason, phase


def select_phase_quota(
    state_ids: Sequence[str],
    phase_ids: torch.Tensor,
    quotas: dict[int, int] | None = None,
) -> torch.Tensor:
    """Select at most each phase quota by canonical state-id order.

    No phase is back-filled from another phase.  The returned indices are sorted by
    ``(phase_id, state_id)`` to make downstream serialization deterministic.
    """
    quotas = dict(PHASE_QUOTAS if quotas is None else quotas)
    if phase_ids.ndim != 1 or len(state_ids) != phase_ids.numel():
        raise ValueError("state_ids and phase_ids must have matching length")
    if len(set(map(str, state_ids))) != len(state_ids):
        raise ValueError("state_ids must be unique")
    chosen: list[tuple[int, str, int]] = []
    phase_cpu = phase_ids.detach().cpu().tolist()
    for phase, quota in quotas.items():
        if quota < 0:
            raise ValueError("phase quotas must be non-negative")
        candidates = sorted((str(state_ids[index]), index) for index, value in enumerate(phase_cpu)
                            if int(value) == int(phase))
        chosen.extend((int(phase), state_id, index) for state_id, index in candidates[:quota])
    chosen.sort(key=lambda item: (item[0], item[1]))
    return torch.tensor([item[2] for item in chosen], dtype=torch.long, device=phase_ids.device)


def collect_active_phase_quota(
    state_ids: Sequence[str],
    active: torch.Tensor,
    phase_ids: torch.Tensor,
    quotas: dict[int, int] | None = None,
) -> torch.Tensor:
    """Apply the collection stop gate without cross-phase backfilling."""
    if active.ndim != 1 or active.dtype != torch.bool or active.numel() != phase_ids.numel():
        raise ValueError("active must be boolean and match phase_ids")
    phase = phase_ids.clone()
    phase[~active] = -1
    quotas = dict(PHASE_QUOTAS if quotas is None else quotas)
    selected = select_phase_quota(state_ids, phase, quotas)
    counts = {int(value): int(((phase == value)).sum().item()) for value in quotas}
    missing = {PHASE_NAMES.get(key, str(key)): max(0, int(value) - counts.get(int(key), 0))
               for key, value in quotas.items() if counts.get(int(key), 0) < int(value)}
    if missing:
        raise CollectionInsufficient(f"phase quotas are insufficient: {missing}")
    return selected


def _axis_angle_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    _check_last("axis_angle", axis_angle, 3)
    theta2 = axis_angle.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-16).sqrt()
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    small = theta2 < 1e-8
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120,
                    torch.sin(theta) / theta.clamp_min(1e-8))
    b = torch.where(small, 0.5 - theta2 / 24 + theta2.square() / 720,
                    (1 - torch.cos(theta)) / theta2.clamp_min(1e-8))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(skew)
    return (eye + a.reshape(-1, 1, 1) * skew + b.reshape(-1, 1, 1) * (skew @ skew)
            ).reshape(*axis_angle.shape[:-1], 3, 3)


def rotation_geodesic(rotation_a: torch.Tensor, rotation_b: torch.Tensor) -> torch.Tensor:
    if rotation_a.shape != rotation_b.shape or rotation_a.shape[-2:] != (3, 3):
        raise ValueError("rotation matrices must have matching [...,3,3] shapes")
    _finite("rotation_a", rotation_a)
    _finite("rotation_b", rotation_b)
    relative = rotation_a.transpose(-1, -2) @ rotation_b
    cosine = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5
    return torch.acos(cosine.clamp(-1.0, 1.0))


def canonical_ig(
    key_body_positions: torch.Tensor,
    object_surface_points: torch.Tensor,
    heading_inverse: torch.Tensor,
) -> torch.Tensor:
    """Build six-key-body DExplore interaction geometry in heading coordinates."""
    if key_body_positions.ndim != 3 or key_body_positions.shape[1:] != (IG_POINT_COUNT, 3):
        raise ValueError("key_body_positions must be [B,6,3]")
    if object_surface_points.ndim != 3 or object_surface_points.shape[1:] != (256, 3):
        raise ValueError("object_surface_points must be the DExplore canonical [B,256,3] surface")
    if heading_inverse.shape != (key_body_positions.shape[0], 3, 3):
        raise ValueError("heading_inverse must be [B,3,3]")
    _finite("key_body_positions", key_body_positions)
    _finite("object_surface_points", object_surface_points)
    _finite("heading_inverse", heading_inverse)
    displacement = key_body_positions[:, :, None, :] - object_surface_points[:, None, :, :]
    nearest = displacement.square().sum(-1).argmin(-1)
    batch = torch.arange(key_body_positions.shape[0], device=key_body_positions.device)[:, None]
    key = torch.arange(IG_POINT_COUNT, device=key_body_positions.device)[None, :]
    nearest_displacement = displacement[batch, key, nearest]
    local = torch.einsum("bij,bkj->bki", heading_inverse, nearest_displacement)
    return local.reshape(key_body_positions.shape[0], IG_POINT_COUNT * 3)


def canonical_ig_from_body_positions(
    body_positions: torch.Tensor,
    object_surface_points: torch.Tensor,
    heading_inverse: torch.Tensor,
) -> torch.Tensor:
    """Select DExplore's six canonical key bodies before building IG.

    The public DExplore rigid-body tensor contains the full body list; V1.21c
    fixes the interaction order to ``[0, 3, 6, 9, 12, 15]`` and never lets a
    caller accidentally change it by passing a different link ordering.
    """
    if body_positions.ndim != 3 or body_positions.shape[1] <= max(IG_KEY_INDICES) or body_positions.shape[-1] != 3:
        raise ValueError("body_positions must be [B,N>=16,3]")
    return canonical_ig(body_positions[:, IG_KEY_INDICES], object_surface_points, heading_inverse)


def ig_error(ig: torch.Tensor, reference_ig: torch.Tensor) -> torch.Tensor:
    if ig.shape != reference_ig.shape or ig.shape[-1] != ACTION_DIM:
        raise ValueError("IG tensors must have matching [...,18] shapes")
    _finite("ig", ig)
    _finite("reference_ig", reference_ig)
    return (ig - reference_ig).reshape(*ig.shape[:-1], IG_POINT_COUNT, 3).square().sum(-1).mean(-1) / (IG_SIGMA_M ** 2)


def compose_current_local_delta(current_pose: torch.Tensor, delta_xi: torch.Tensor) -> torch.Tensor:
    """Right-compose a current-object-local ``[translation, rotvec]`` block."""
    if current_pose.ndim != 3 or current_pose.shape[-2:] != (4, 4):
        raise ValueError("current_pose must be [B,4,4]")
    if delta_xi.ndim != 3 or delta_xi.shape[0] != current_pose.shape[0] or delta_xi.shape[-1] != 6:
        raise ValueError("delta_xi must be [B,K,6]")
    _finite("current_pose", current_pose)
    _finite("delta_xi", delta_xi)
    batch, candidates = delta_xi.shape[:2]
    delta = torch.eye(4, dtype=delta_xi.dtype, device=delta_xi.device).expand(batch, candidates, 4, 4).clone()
    delta[..., :3, :3] = _axis_angle_matrix(delta_xi[..., 3:])
    delta[..., :3, 3] = delta_xi[..., :3]
    return current_pose[:, None] @ delta


def object_distance(pose: torch.Tensor, goal_pose: torch.Tensor) -> torch.Tensor:
    if pose.shape != goal_pose.shape or pose.shape[-2:] != (4, 4):
        raise ValueError("pose and goal_pose must have matching [...,4,4] shapes")
    _finite("pose", pose)
    _finite("goal_pose", goal_pose)
    translation = (pose[..., :3, 3] - goal_pose[..., :3, 3]).square().sum(-1) / (OBJECT_POSITION_SIGMA_M ** 2)
    rotation = rotation_geodesic(pose[..., :3, :3], goal_pose[..., :3, :3])
    return translation + rotation.square() / (OBJECT_ROTATION_SIGMA_RAD ** 2)


def pose_xyzw_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """Convert world ``xyz + xyzw`` poses to homogeneous matrices."""
    if not isinstance(pose, torch.Tensor) or pose.ndim < 1 or pose.shape[-1] != 7:
        raise ValueError("pose must have shape [...,7] in xyz+xyzw order")
    _finite("pose", pose)
    quaternion = pose[..., 3:7]
    norm = quaternion.norm(dim=-1, keepdim=True)
    if (norm <= 0).any():
        raise ValueError("pose quaternion must have non-zero norm")
    x, y, z, w = (quaternion / norm).unbind(-1)
    rotation = torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), dim=-1).reshape(*pose.shape[:-1], 3, 3)
    result = torch.eye(4, dtype=pose.dtype, device=pose.device).expand(
        *pose.shape[:-1], 4, 4
    ).clone()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = pose[..., :3]
    return result


def next_state_cost(
    goal_pose_t6: torch.Tensor,
    reference_ig_t1: torch.Tensor,
    next_pose: torch.Tensor,
    next_ig: torch.Tensor,
) -> torch.Tensor:
    """Return the V1.21c next-state cost with explicit ``t+6``/``t+1`` clocks.

    The next-state tensors are ``[B,K,...]``.  This primitive deliberately has
    no current-state input, so duplicate calibration cannot accidentally use a
    branch-local pre-candidate baseline.
    """
    if goal_pose_t6.ndim != 3 or goal_pose_t6.shape[-2:] != (4, 4):
        raise ValueError("goal_pose_t6 must be [B,4,4]")
    if reference_ig_t1.ndim != 2 or reference_ig_t1.shape[-1] != ACTION_DIM:
        raise ValueError("reference_ig_t1 must be [B,18]")
    if next_pose.ndim != 4 or next_pose.shape[-2:] != (4, 4):
        raise ValueError("next_pose must be [B,K,4,4]")
    if next_ig.ndim != 3 or next_ig.shape[-1] != ACTION_DIM:
        raise ValueError("next_ig must be [B,K,18]")
    if (goal_pose_t6.shape[0] != next_pose.shape[0] or
            reference_ig_t1.shape[0] != next_pose.shape[0] or
            next_ig.shape[:2] != next_pose.shape[:2]):
        raise ValueError("next-state cost batch/candidate shapes do not match")
    goal = goal_pose_t6[:, None].expand_as(next_pose)
    reference_ig = reference_ig_t1[:, None].expand_as(next_ig)
    return object_distance(next_pose, goal) + 0.5 * ig_error(next_ig, reference_ig)


def physics_scores(
    canonical_current_pose: torch.Tensor,
    goal_pose_t6: torch.Tensor,
    canonical_current_ig: torch.Tensor,
    reference_ig_t1: torch.Tensor,
    next_pose: torch.Tensor,
    next_ig: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Score PhysX candidates against one shared collected pre-action state.

    ``canonical_current_pose`` and ``canonical_current_ig`` have no candidate
    dimension by construction.  Replay-env-local pre-candidate states cannot be
    passed without violating the shape contract.
    """
    if canonical_current_pose.ndim != 3 or canonical_current_pose.shape[-2:] != (4, 4):
        raise ValueError("canonical_current_pose must be shared [B,4,4]")
    if canonical_current_ig.ndim != 2 or canonical_current_ig.shape[-1] != ACTION_DIM:
        raise ValueError("canonical_current_ig must be shared [B,18]")
    if (canonical_current_pose.shape[0] != goal_pose_t6.shape[0] or
            canonical_current_ig.shape[0] != goal_pose_t6.shape[0]):
        raise ValueError("shared baseline batch does not match goal batch")
    baseline = object_distance(canonical_current_pose, goal_pose_t6)
    baseline = baseline + 0.5 * ig_error(canonical_current_ig, reference_ig_t1)
    next_cost = next_state_cost(goal_pose_t6, reference_ig_t1, next_pose, next_ig)
    return {"baseline_cost": baseline, "next_cost": next_cost,
            "score": baseline[:, None] - next_cost}


def calibration_reference_targets(
    hoi_data: torch.Tensor,
    data_ids: torch.Tensor,
    progress: int,
    key_body_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the calibration target clocks: object ``t+6`` and IG ``t+1``."""
    if hoi_data.ndim != 3 or data_ids.ndim != 1:
        raise ValueError("hoi_data and data_ids must be [D,T,F] and [B]")
    if key_body_count <= max(IG_KEY_INDICES):
        raise ValueError("key_body_count must include all canonical IG indices")
    if progress < 0 or progress + H_REF >= hoi_data.shape[1]:
        raise ValueError("progress does not admit the fixed t+6 goal")
    data_ids = data_ids.to(device=hoi_data.device, dtype=torch.long)
    goal_pose_xyzw = hoi_data[data_ids, progress + H_REF, 106:113]
    reference_t1 = hoi_data[data_ids, progress + 1]
    reference_ig_start = 119 + key_body_count * 3 + 1 + 16
    reference_ig = reference_t1[
        ..., reference_ig_start:reference_ig_start + key_body_count * 3
    ].view(-1, key_body_count, 3)
    selector = torch.tensor(IG_KEY_INDICES, device=hoi_data.device, dtype=torch.long)
    return pose_xyzw_to_matrix(goal_pose_xyzw), reference_ig[:, selector].reshape(-1, ACTION_DIM)


def cm_scores(
    current_pose: torch.Tensor,
    goal_pose: torch.Tensor,
    predicted_delta_xi: torch.Tensor,
    current_ig: torch.Tensor,
    reference_ig_next: torch.Tensor,
    predicted_ig: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute ``P_obj + .5 P_IG`` for Cm one-step predictions."""
    if predicted_delta_xi.ndim != 3 or predicted_delta_xi.shape[-1] != 6:
        raise ValueError("predicted_delta_xi must be [B,K,6]")
    next_pose = compose_current_local_delta(current_pose, predicted_delta_xi)
    scored = physics_scores(
        current_pose, goal_pose, current_ig, reference_ig_next, next_pose, predicted_ig
    )
    predicted_object = object_distance(next_pose, goal_pose[:, None].expand_as(next_pose))
    predicted_ig_error = ig_error(predicted_ig, reference_ig_next[:, None].expand_as(predicted_ig))
    object_progress = object_distance(current_pose, goal_pose)[:, None] - predicted_object
    ig_progress = ig_error(current_ig, reference_ig_next)[:, None] - predicted_ig_error
    return {
        "next_pose": next_pose,
        "object_progress": object_progress,
        "ig_progress": ig_progress,
        "score": scored["score"],
    }


def cm_valid_mask(
    candidate_valid: torch.Tensor,
    delta_xi: torch.Tensor,
    token_mask: torch.Tensor,
    token_mass: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact candidate/Cmv2 validity contract."""
    if candidate_valid.ndim != 2 or delta_xi.ndim != 3 or delta_xi.shape[:2] != candidate_valid.shape:
        raise ValueError("candidate_valid and delta_xi must be [B,K] and [B,K,6]")
    if token_mask.shape[:2] != candidate_valid.shape or token_mask.dtype != torch.bool:
        raise ValueError("token_mask must be boolean [B,K,T]")
    if token_mass.shape != token_mask.shape:
        raise ValueError("token_mass must match token_mask")
    valid = candidate_valid & torch.isfinite(delta_xi).all(-1)
    valid = valid & torch.isfinite(token_mass).all(-1)
    valid = valid & token_mask.any(-1) & (token_mass.sum(-1) > 0)
    return valid


def ranking_eligible(cm_valid: torch.Tensor, minimum: int = 4) -> torch.Tensor:
    if cm_valid.ndim != 2 or cm_valid.dtype != torch.bool:
        raise ValueError("cm_valid must be boolean [B,K]")
    if minimum < 1:
        raise ValueError("minimum must be positive")
    return cm_valid.sum(-1) >= minimum


@dataclass(frozen=True)
class PairwiseResult:
    state_accuracy: torch.Tensor
    pair_count: torch.Tensor
    valid_state: torch.Tensor
    accuracy: float


def pairwise_metrics(
    cm_score: torch.Tensor,
    physics_score: torch.Tensor,
    cm_valid: torch.Tensor,
    physics_valid: torch.Tensor,
    epsilon_physx: float,
    cm_tie_tolerance: float = CM_TIE_TOLERANCE,
) -> PairwiseResult:
    """Compare candidate directions, averaging pairs within each state first."""
    if cm_score.shape != physics_score.shape or cm_score.ndim != 2:
        raise ValueError("scores must have matching [B,K] shapes")
    if cm_valid.shape != cm_score.shape or physics_valid.shape != cm_score.shape:
        raise ValueError("valid masks must match scores")
    if cm_valid.dtype != torch.bool or physics_valid.dtype != torch.bool:
        raise ValueError("valid masks must be boolean")
    if not math.isfinite(epsilon_physx) or epsilon_physx < PHYSX_MIN_TOLERANCE:
        raise ValueError("epsilon_physx must be finite and >= 1e-6")
    _finite_where("cm_score", cm_score, cm_valid)
    _finite_where("physics_score", physics_score, physics_valid)
    state_accuracy = torch.full((cm_score.shape[0],), float("nan"), dtype=cm_score.dtype,
                                device=cm_score.device)
    pair_count = torch.zeros((cm_score.shape[0],), dtype=torch.int64, device=cm_score.device)
    for batch_index in range(cm_score.shape[0]):
        indices = torch.where(cm_valid[batch_index] & physics_valid[batch_index])[0].tolist()
        agreements: list[float] = []
        for first, second in itertools.combinations(indices, 2):
            physics_delta = float(physics_score[batch_index, first] - physics_score[batch_index, second])
            if abs(physics_delta) <= epsilon_physx:
                continue
            cm_delta = float(cm_score[batch_index, first] - cm_score[batch_index, second])
            if abs(cm_delta) <= cm_tie_tolerance:
                agreements.append(0.5)
            else:
                agreements.append(float((cm_delta > 0) == (physics_delta > 0)))
        pair_count[batch_index] = len(agreements)
        if agreements:
            state_accuracy[batch_index] = torch.tensor(agreements, dtype=cm_score.dtype,
                                                       device=cm_score.device).mean()
    valid_state = torch.isfinite(state_accuracy)
    accuracy = float(state_accuracy[valid_state].mean().item()) if valid_state.any() else float("nan")
    return PairwiseResult(state_accuracy, pair_count, valid_state, accuracy)


@dataclass(frozen=True)
class Top1Result:
    selected_index: torch.Tensor
    valid_state: torch.Tensor
    success: torch.Tensor
    tie: torch.Tensor
    improvement: torch.Tensor


def top1_metrics(
    cm_score: torch.Tensor,
    physics_score: torch.Tensor,
    cm_valid: torch.Tensor,
    physics_valid: torch.Tensor,
    epsilon_physx: float,
) -> Top1Result:
    """Evaluate Cm's stable top-1 candidate against the policy mean candidate 0."""
    if cm_score.shape != physics_score.shape or cm_score.ndim != 2 or cm_score.shape[1] != CANDIDATE_COUNT:
        raise ValueError("scores must be matching [B,8] tensors")
    if cm_valid.shape != cm_score.shape or physics_valid.shape != cm_score.shape:
        raise ValueError("valid masks must match scores")
    if cm_valid.dtype != torch.bool or physics_valid.dtype != torch.bool:
        raise ValueError("valid masks must be boolean")
    _finite_where("cm_score", cm_score, cm_valid)
    # Invalid Cm scores are masked to -inf; argmax is stable and selects the lowest index on ties.
    masked = cm_score.masked_fill(~cm_valid, float("-inf"))
    selected = masked.argmax(-1)
    row = torch.arange(cm_score.shape[0], device=cm_score.device)
    selected_valid = cm_valid[row, selected]
    baseline_valid = physics_valid[:, 0] & torch.isfinite(physics_score[:, 0])
    chosen_valid = physics_valid[row, selected] & torch.isfinite(physics_score[row, selected])
    valid_state = selected_valid & baseline_valid & chosen_valid
    improvement = torch.full((cm_score.shape[0],), float("nan"), dtype=cm_score.dtype,
                             device=cm_score.device)
    improvement[valid_state] = physics_score[row[valid_state], selected[valid_state]] - physics_score[valid_state, 0]
    tie = torch.zeros_like(valid_state)
    tie[valid_state] = improvement[valid_state].abs() <= epsilon_physx
    success = valid_state & (improvement > epsilon_physx)
    return Top1Result(selected, valid_state, success, tie, improvement)


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lower: float
    upper: float
    draws: int
    episode_count: int


def episode_block_bootstrap(
    values: Sequence[float] | np.ndarray,
    episode_ids: Sequence[str] | np.ndarray,
    *,
    draws: int = 10_000,
    seed: int = 2021,
) -> BootstrapCI:
    """Compute an episode-block percentile CI, omitting invalid/NaN values."""
    values_array = np.asarray(values, dtype=np.float64).reshape(-1)
    episodes_array = np.asarray(episode_ids).reshape(-1).astype(str)
    if values_array.size != episodes_array.size or values_array.size == 0:
        raise ValueError("values and episode_ids must have matching non-empty lengths")
    if draws <= 0:
        raise ValueError("draws must be positive")
    valid = np.isfinite(values_array)
    values_array, episodes_array = values_array[valid], episodes_array[valid]
    episodes = np.unique(episodes_array)
    if episodes.size < 1:
        return BootstrapCI(float("nan"), float("nan"), float("nan"), draws, 0)
    estimate = float(values_array.mean())
    grouped = [np.flatnonzero(episodes_array == episode) for episode in episodes]
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.integers(0, len(grouped), size=len(grouped))
        selected = np.concatenate([grouped[index] for index in chosen])
        bootstrap[draw] = values_array[selected].mean()
    lower, upper = np.percentile(bootstrap, (2.5, 97.5))
    return BootstrapCI(estimate, float(lower), float(upper), draws, int(episodes.size))


def freeze_physx_epsilon(duplicate_score_delta: Sequence[float] | np.ndarray) -> float:
    """Freeze ``max(5*p99(abs(delta)), 1e-6)`` from calibration duplicates."""
    values = np.asarray(duplicate_score_delta, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("duplicate score deltas must be non-empty and finite")
    return max(5.0 * float(np.percentile(np.abs(values), 99.0)), PHYSX_MIN_TOLERANCE)


def calibrate_duplicate_anchor(
    duplicate_position_m: Sequence[float] | np.ndarray,
    duplicate_rotation_rad: Sequence[float] | np.ndarray,
    duplicate_score_delta: Sequence[float] | np.ndarray,
    *,
    position_ceiling_m: float = 5e-4,
    rotation_ceiling_rad: float = 5e-3,
) -> dict[str, float]:
    """Check the 64-state duplicate hard ceiling and freeze score tie tolerance."""
    position = np.asarray(duplicate_position_m, dtype=np.float64).reshape(-1)
    rotation = np.asarray(duplicate_rotation_rad, dtype=np.float64).reshape(-1)
    score = np.asarray(duplicate_score_delta, dtype=np.float64).reshape(-1)
    if position.size == 0 or position.size != rotation.size or position.size != score.size:
        raise ValueError("duplicate calibration arrays must have equal non-zero length")
    if (not np.isfinite(position).all() or not np.isfinite(rotation).all() or
            not np.isfinite(score).all()):
        raise ValueError("duplicate calibration must be finite")
    max_position = float(np.max(np.abs(position)))
    max_rotation = float(np.max(np.abs(rotation)))
    if max_position > position_ceiling_m or max_rotation > rotation_ceiling_rad:
        raise RuntimeError(
            "INVALID_IMPLEMENTATION: duplicate anchor exceeds the PhysX hard ceiling "
            f"(position={max_position:g}, rotation={max_rotation:g})")
    return {
        "position_p99_m": float(np.percentile(np.abs(position), 99.0)),
        "rotation_p99_rad": float(np.percentile(np.abs(rotation), 99.0)),
        "score_p99": float(np.percentile(np.abs(score), 99.0)),
        "epsilon_physx": freeze_physx_epsilon(score),
    }


def branch_state_parity(
    public_reference_state: torch.Tensor,
    public_branch_states: torch.Tensor,
    reference_task_indices: torch.Tensor,
    branch_task_indices: torch.Tensor,
    *,
    tolerance: float = 1e-5,
) -> torch.Tensor:
    """Return per-branch parity for replayed public state and task indices."""
    if public_reference_state.ndim != 1 or public_branch_states.ndim != 2:
        raise ValueError("public states must be [D] and [B,D]")
    if public_branch_states.shape[1] != public_reference_state.numel():
        raise ValueError("public branch state width mismatch")
    if reference_task_indices.ndim != 1 or branch_task_indices.shape != (public_branch_states.shape[0], reference_task_indices.numel()):
        raise ValueError("task indices must be [D] and [B,D]")
    if reference_task_indices.dtype != branch_task_indices.dtype:
        raise ValueError("task index dtypes must match")
    _finite("public_reference_state", public_reference_state)
    _finite("public_branch_states", public_branch_states)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be non-negative and finite")
    state_ok = (public_branch_states - public_reference_state[None]).abs().amax(-1) <= tolerance
    task_ok = (branch_task_indices == reference_task_indices[None]).all(-1)
    return state_ok & task_ok


def phase_metric(values: torch.Tensor, phase_ids: torch.Tensor, phase: int) -> torch.Tensor:
    """Return a phase slice for diagnostics without silently back-filling buckets."""
    if values.shape[0] != phase_ids.shape[0]:
        raise ValueError("values and phase_ids must have matching first dimension")
    return values[phase_ids == phase]
