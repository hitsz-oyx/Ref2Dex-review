"""Differentiable Inspire surface FK for CmDecoderv2 point-flow supervision."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix
from torch import nn

from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel

from .kinematics import InspireKinematics


class DifferentiableInspireSurface(nn.Module):
    """Generate the cache's fixed 1538-point Inspire surface with torch gradients.

    The mesh sampling and URDF ordering are delegated to the same deterministic
    builder used by the ObjectInteractionCm geometry cache.  FK itself is
    implemented with torch operations so point-flow loss gradients reach q and
    wrist outputs.
    """

    def __init__(self, urdf_path: str | Path, *, sample_count: int = 1538, surface_seed: int = 2024) -> None:
        super().__init__()
        helper = InspireUrdfModel(Path(urdf_path).resolve())
        limits = InspireKinematics(urdf_path)
        sampled_points, _, sampled_visual_ids = helper.surface_samples(int(sample_count), int(surface_seed))
        link_names = tuple(helper.link_names)
        link_index = {name: index for index, name in enumerate(link_names)}
        visual_link_indices = np.asarray([link_index[visual.link] for visual in helper.visuals], dtype=np.int64)
        sample_link_indices = visual_link_indices[np.asarray(sampled_visual_ids, dtype=np.int64)]

        ordered_joints = []

        def visit(parent: str) -> None:
            for joint in helper.children.get(parent, []):
                ordered_joints.append(joint)
                visit(joint.child)

        visit(helper.root_link)
        if len(ordered_joints) != len(helper.joints):
            raise ValueError("Inspire URDF joint tree is disconnected")

        self.link_names = link_names
        self.joint_types = tuple(str(joint.joint_type) for joint in ordered_joints)
        self.joint_parent_indices = torch.tensor([link_index[joint.parent] for joint in ordered_joints], dtype=torch.long)
        self.joint_child_indices = torch.tensor([link_index[joint.child] for joint in ordered_joints], dtype=torch.long)
        self.joint_q_indices = torch.tensor([int(joint.q_index) for joint in ordered_joints], dtype=torch.long)
        self.register_buffer(
            "joint_origins",
            torch.from_numpy(np.stack([np.asarray(joint.origin, dtype=np.float32) for joint in ordered_joints])),
            persistent=False,
        )
        self.register_buffer(
            "joint_axes",
            torch.from_numpy(np.stack([np.asarray(joint.axis, dtype=np.float32) for joint in ordered_joints])),
            persistent=False,
        )
        self.register_buffer("native_to_urdf_inverse", torch.argsort(torch.tensor([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9], dtype=torch.long)), persistent=False)
        self.register_buffer("independent_native_indices", torch.tensor([6, 8, 10, 12, 14, 15], dtype=torch.long), persistent=False)
        self.register_buffer("mimic_native_indices", torch.tensor([7, 9, 11, 13, 16, 17], dtype=torch.long), persistent=False)
        self.register_buffer("finger_lower", torch.from_numpy(limits.finger_lower.astype(np.float32)), persistent=False)
        self.register_buffer("finger_upper", torch.from_numpy(limits.finger_upper.astype(np.float32)), persistent=False)
        self.register_buffer("surface_points_local", torch.from_numpy(np.asarray(sampled_points, dtype=np.float32)), persistent=False)
        self.register_buffer("surface_link_indices", torch.from_numpy(sample_link_indices), persistent=False)
        self.register_buffer("surface_visual_ids", torch.from_numpy(np.asarray(sampled_visual_ids, dtype=np.int64)), persistent=False)
        self.register_buffer(
            "visual_local_transforms",
            torch.from_numpy(np.stack([np.asarray(visual.local_transform, dtype=np.float32) for visual in helper.visuals])),
            persistent=False,
        )
        self.root_index = int(link_index[helper.root_link])
        self.hand_base_index = int(link_index["hand_base_link"])
        zero_native = np.zeros(18, dtype=np.float64)
        zero_canonical = helper.link_transforms(helper.qpos_to_urdf_order(zero_native))["hand_base_link"]
        self.register_buffer("zero_hand_base_inverse", torch.from_numpy(np.linalg.inv(zero_canonical).astype(np.float32)), persistent=False)

    @staticmethod
    def _expand_native(finger_q: torch.Tensor) -> torch.Tensor:
        native = finger_q.new_zeros((*finger_q.shape[:-1], 18))
        native[..., 6] = finger_q[..., 0]
        native[..., 8] = finger_q[..., 1]
        native[..., 10] = finger_q[..., 2]
        native[..., 12] = finger_q[..., 3]
        native[..., 14] = finger_q[..., 4]
        native[..., 15] = finger_q[..., 5]
        native[..., 7] = native[..., 6] * 1.05
        native[..., 9] = native[..., 8] * 1.05
        native[..., 11] = native[..., 10] * 1.05
        native[..., 13] = native[..., 12] * 1.05
        native[..., 16] = native[..., 15] * 0.6
        native[..., 17] = native[..., 15] * 0.8
        return native

    def _canonical_links(self, finger_q: torch.Tensor) -> torch.Tensor:
        native = self._expand_native(finger_q)
        urdf_q = native.index_select(-1, self.native_to_urdf_inverse)
        batch = int(finger_q.reshape(-1, 6).shape[0])
        eye = torch.eye(4, device=finger_q.device, dtype=finger_q.dtype).expand(batch, 4, 4)
        links: list[torch.Tensor | None] = [None] * len(self.link_names)
        links[self.root_index] = eye
        for index, joint_type in enumerate(self.joint_types):
            parent = links[int(self.joint_parent_indices[index])]
            if parent is None:
                raise RuntimeError("Inspire FK encountered a joint before its parent")
            origin = self.joint_origins[index].to(dtype=finger_q.dtype)
            if joint_type == "fixed":
                motion = eye
            else:
                q_index = int(self.joint_q_indices[index])
                q = urdf_q[..., q_index]
                axis = self.joint_axes[index].to(dtype=finger_q.dtype)
                if joint_type == "prismatic":
                    motion = eye.clone()
                    motion[:, :3, 3] = axis[None, :] * q[:, None]
                elif joint_type in {"revolute", "continuous"}:
                    motion = eye.clone()
                    motion[:, :3, :3] = axis_angle_to_matrix(axis[None, :] * q[:, None], fast=False)
                else:
                    raise ValueError(f"Unsupported Inspire joint type {joint_type!r}")
            links[int(self.joint_child_indices[index])] = parent @ origin @ motion
        if any(value is None for value in links):
            raise RuntimeError("Inspire FK did not resolve all links")
        return torch.stack([value for value in links if value is not None], dim=1)

    def forward(self, finger_q: torch.Tensor, wrist_pose_world: torch.Tensor) -> torch.Tensor:
        if finger_q.shape[-1] != 6 or wrist_pose_world.shape[-2:] != (4, 4):
            raise ValueError(f"Expected finger_q [...,6] and wrist_pose [...,4,4], got {finger_q.shape}, {wrist_pose_world.shape}")
        leading = finger_q.shape[:-1]
        finger_q = torch.maximum(torch.minimum(finger_q, self.finger_upper.to(dtype=finger_q.dtype)), self.finger_lower.to(dtype=finger_q.dtype))
        flat_q = finger_q.reshape(-1, 6)
        flat_wrist = wrist_pose_world.reshape(-1, 4, 4).to(dtype=finger_q.dtype)
        canonical = self._canonical_links(flat_q)
        world_from_canonical = flat_wrist @ self.zero_hand_base_inverse.to(dtype=finger_q.dtype)
        world_links = world_from_canonical[:, None] @ canonical
        link_transforms = world_links.index_select(1, self.surface_link_indices)
        local_transforms = self.visual_local_transforms.to(dtype=finger_q.dtype).index_select(0, self.surface_visual_ids)
        transforms = link_transforms @ local_transforms[None]
        local_points = self.surface_points_local.to(dtype=finger_q.dtype)[None, :, :].expand(flat_q.shape[0], -1, -1)
        points = torch.einsum("bnj,bnkj->bnk", local_points, transforms[..., :3, :3]) + transforms[..., :3, 3]
        return points.reshape(*leading, self.surface_points_local.shape[0], 3)


def make_relative_transform(rotvec: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Build current-wrist-local SE(3) deltas from decoder outputs."""
    if rotvec.shape[-1] != 3 or translation.shape[-1] != 3 or rotvec.shape != translation.shape:
        raise ValueError(f"Expected matching [...,3] rotvec/translation, got {rotvec.shape}, {translation.shape}")
    result = torch.eye(4, device=rotvec.device, dtype=rotvec.dtype).expand(*rotvec.shape[:-1], 4, 4).clone()
    result[..., :3, :3] = axis_angle_to_matrix(rotvec, fast=False)
    result[..., :3, 3] = translation
    return result


def world_to_object(points_world: torch.Tensor, object_pose_world: torch.Tensor) -> torch.Tensor:
    """Transform row-vector points from world to the object_pose_t frame."""
    rotation = object_pose_world[..., :3, :3]
    translation = object_pose_world[..., :3, 3]
    for _ in range(max(0, points_world.ndim - 2)):
        rotation = rotation.unsqueeze(-3)
        translation = translation.unsqueeze(-2)
    relative = points_world - translation
    return torch.matmul(relative.unsqueeze(-2), rotation).squeeze(-2)
