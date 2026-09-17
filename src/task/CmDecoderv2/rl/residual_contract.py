"""Tensor-only contract for a frozen base target plus residual policy."""
from __future__ import annotations

import torch
from pytorch3d.transforms import matrix_to_euler_angles, matrix_to_quaternion, quaternion_to_matrix

from src.task.CmDecoderv2.kinematics import QUERY_LINK_NATIVE_Q_INDICES

ACTION_DIM = 12  # independent finger residual q6 + wrist translation/rotation residual6
OBSERVATION_DIM = 71  # native q18, native dq18, base q6, wrist pose7, object pose7, five tip offsets15
INDEPENDENT_NATIVE = (6, 8, 10, 12, 14, 15)
MIMIC_NATIVE = (7, 9, 11, 13, 16, 17)
MIMIC_SOURCE = (0, 1, 2, 3, 5, 5)  # indices in independent q6, not native18
MIMIC_SCALE = (1.05, 1.05, 1.05, 1.05, 0.6, 0.8)
NATIVE_TO_URDF = (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9)
NATIVE_DOF_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6",
    "index_proximal_joint", "index_intermediate_joint",
    "middle_proximal_joint", "middle_intermediate_joint",
    "pinky_proximal_joint", "pinky_intermediate_joint",
    "ring_proximal_joint", "ring_intermediate_joint",
    "thumb_proximal_yaw_joint", "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint", "thumb_distal_joint",
)


def native_sim_indices(sim_names: list[str]) -> list[int]:
    if len(sim_names) != 18 or set(sim_names) != set(NATIVE_DOF_NAMES):
        raise ValueError(f"Unexpected Inspire DOF names: {sim_names}")
    return [sim_names.index(name) for name in NATIVE_DOF_NAMES]


def sim_to_native(sim_q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return sim_q.index_select(-1, indices)


def native_to_sim(native_q: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    result = torch.empty_like(native_q)
    result[..., indices] = native_q
    return result


def pose_matrix(pose_xyzw: torch.Tensor) -> torch.Tensor:
    result = torch.eye(4, device=pose_xyzw.device, dtype=pose_xyzw.dtype).expand(*pose_xyzw.shape[:-1], 4, 4).clone()
    result[..., :3, :3] = quaternion_to_matrix(pose_xyzw[..., [6, 3, 4, 5]])
    result[..., :3, 3] = pose_xyzw[..., :3]
    return result


def matrix_pose(matrix: torch.Tensor) -> torch.Tensor:
    quat = matrix_to_quaternion(matrix[..., :3, :3])
    return torch.cat([matrix[..., :3, 3], quat[..., [1, 2, 3, 0]]], dim=-1)


def inverse_pose(matrix: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(matrix)
    rotation = matrix[..., :3, :3].transpose(-1, -2)
    result[..., :3, :3] = rotation
    result[..., :3, 3] = -(rotation @ matrix[..., :3, 3, None]).squeeze(-1)
    result[..., 3, 3] = 1
    return result


def rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :3, :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def actual_queries(native_q: torch.Tensor, link_world: torch.Tensor,
                   object_world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use all actual link poses, including non-mimic finger deflections."""
    relative = inverse_pose(object_world)[:, None] @ link_world
    indices = torch.as_tensor(QUERY_LINK_NATIVE_Q_INDICES, device=native_q.device)
    q_features = native_q[:, indices.clamp_min(0)].clone()
    q_features[:, indices < 0] = 0
    links = torch.cat([q_features[..., None], relative[..., :3, 3], rotation_6d(relative)], dim=-1)
    state = torch.cat([native_q[:, list(INDEPENDENT_NATIVE)], relative[:, 0, :3, 3], rotation_6d(relative[:, 0])], dim=-1)
    return state, links


def wrist_native_target(wrist_world: torch.Tensor, zero_base_inverse: torch.Tensor,
                        current_wrist_q: torch.Tensor) -> torch.Tensor:
    """Invert the URDF's XYZ translation, Rx Ry Rz and fixed base_joint."""
    root = wrist_world @ zero_base_inverse
    angles = matrix_to_euler_angles(root[..., :3, :3], "XYZ")
    alternative = torch.stack([angles[..., 0] + torch.pi, torch.pi - angles[..., 1], angles[..., 2] + torch.pi], dim=-1)
    candidates = torch.stack([angles, alternative], dim=-2)
    current = current_wrist_q[..., 3:6, None].transpose(-1, -2)
    candidates = current + torch.remainder(candidates - current + torch.pi, 2 * torch.pi) - torch.pi
    select = (candidates - current).square().sum(-1).argmin(-1)
    selected = candidates.gather(-2, select[..., None, None].expand(*select.shape, 1, 3)).squeeze(-2)
    return torch.cat([root[..., :3, 3], selected], dim=-1)


def coupled_finger_bounds(native_lower: torch.Tensor, native_upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    lower = native_lower[list(INDEPENDENT_NATIVE)].clone()
    upper = native_upper[list(INDEPENDENT_NATIVE)].clone()
    for mimic, source, scale in zip(MIMIC_NATIVE, MIMIC_SOURCE, MIMIC_SCALE):
        lower[source] = torch.maximum(lower[source], native_lower[mimic] / scale)
        upper[source] = torch.minimum(upper[source], native_upper[mimic] / scale)
    if (lower > upper).any():
        raise ValueError("Empty coupled control range")
    return lower, upper


def residual_action_bounds(device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized residual limits used by the first physics wiring smoke."""
    lower = torch.tensor([-1.0] * ACTION_DIM, dtype=torch.float32, device=device)
    upper = torch.tensor([1.0] * ACTION_DIM, dtype=torch.float32, device=device)
    return lower, upper


def expand_native_targets(finger_q: torch.Tensor) -> torch.Tensor:
    """Expand q6 into native18 targets with the DExplore mimic convention.

    The function deliberately does not clamp: the simulator task clamps against
    its loaded URDF limits after composing base target and residual.
    """
    if finger_q.shape[-1] != 6:
        raise ValueError(f"Expected [...,6] independent finger q, got {tuple(finger_q.shape)}")
    result = torch.zeros(*finger_q.shape[:-1], 18, dtype=finger_q.dtype, device=finger_q.device)
    result[..., :6] = 0.0
    result[..., list(INDEPENDENT_NATIVE)] = finger_q
    source = finger_q[..., list(MIMIC_SOURCE)]
    result[..., list(MIMIC_NATIVE)] = source * torch.tensor(MIMIC_SCALE, dtype=finger_q.dtype, device=finger_q.device)
    return result


def native_to_urdf(native_q: torch.Tensor) -> torch.Tensor:
    if native_q.shape[-1] != 18:
        raise ValueError(f"Expected [...,18] native q, got {tuple(native_q.shape)}")
    result = torch.empty_like(native_q)
    result[..., list(NATIVE_TO_URDF)] = native_q
    return result
