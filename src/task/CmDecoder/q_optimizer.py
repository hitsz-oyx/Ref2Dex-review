"""Fit Inspire F1 finger q to a predicted corresponding hand point cloud."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.task.CmDecoder.dataset import _load_hrdex_io, _surface_spec, _urdf_link_order


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert batched axis-angle vectors `[...,3]` to rotation matrices."""
    if rotvec.shape[-1] != 3:
        raise ValueError(f"Expected rotvec [...,3], got {tuple(rotvec.shape)}")
    x, y, z = rotvec.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    skew = torch.stack(
        [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=-1
    ).reshape(*rotvec.shape[:-1], 3, 3)
    theta2 = rotvec.square().sum(dim=-1)
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    safe_theta = theta.clamp_min(1e-6)
    safe_theta2 = theta2.clamp_min(1e-12)
    a = torch.where(theta2 > 1e-12, torch.sin(theta) / safe_theta, 1.0 - theta2 / 6.0)
    b = torch.where(
        theta2 > 1e-12,
        (1.0 - torch.cos(theta)) / safe_theta2,
        0.5 - theta2 / 24.0,
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def _axis_motion(axis: np.ndarray, value: torch.Tensor, joint_type: str) -> torch.Tensor:
    """Batched differentiable URDF joint motion matrices."""
    batch = value.shape[0]
    result = torch.eye(4, dtype=value.dtype, device=value.device).expand(batch, -1, -1).clone()
    axis_tensor = torch.as_tensor(axis, dtype=value.dtype, device=value.device)
    axis_tensor = axis_tensor / torch.linalg.vector_norm(axis_tensor).clamp_min(1e-12)
    if joint_type in {"revolute", "continuous"}:
        x, y, z = axis_tensor.unbind()
        zeros = torch.zeros_like(x)
        skew = torch.stack(
            [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=0
        ).reshape(3, 3)
        eye = torch.eye(3, dtype=value.dtype, device=value.device)
        sin = torch.sin(value)[:, None, None]
        cos = torch.cos(value)[:, None, None]
        result[:, :3, :3] = eye + sin * skew + (1.0 - cos) * (skew @ skew)
    elif joint_type == "prismatic":
        result[:, :3, 3] = value[:, None] * axis_tensor
    return result


class DifferentiableInspireHand:
    """Exact fixed-correspondence hand samples driven by differentiable URDF FK."""

    def __init__(
        self,
        robot_urdf: str | Path,
        *,
        num_hand_points: int = 1538,
        sample_seed: int = 42,
        device: str | torch.device = "cpu",
    ) -> None:
        self.robot_urdf = Path(robot_urdf).expanduser().resolve()
        self.device = torch.device(device)
        self.io = _load_hrdex_io(self.robot_urdf.parents[2])
        self.urdf = self.io.parse_urdf(self.robot_urdf)
        self.link_order = _urdf_link_order(self.urdf)

        vertices: list[np.ndarray] = []
        faces: list[np.ndarray] = []
        face_links: list[str] = []
        offset = 0
        for visual in self.urdf.visuals:
            if self.io.is_arm_visual(visual):
                continue
            mesh = self.io.transformed_mesh(
                self.io.load_mesh(visual.mesh_path), visual.origin, visual.scale
            )
            local_vertices = np.asarray(mesh.vertices, dtype=np.float32)
            local_faces = np.asarray(mesh.indices, dtype=np.int64)
            vertices.append(local_vertices)
            faces.append(local_faces + offset)
            face_links.extend([visual.link] * len(local_faces))
            offset += len(local_vertices)
        if not vertices:
            raise RuntimeError(f"No hand visuals found in {self.robot_urdf}")
        all_vertices = np.concatenate(vertices, axis=0)
        all_faces = np.concatenate(faces, axis=0)
        selected_faces, bary = _surface_spec(
            all_vertices, all_faces, int(num_hand_points), int(sample_seed) + 991
        )
        triangles = all_vertices[all_faces[selected_faces]]
        local_points = (triangles * bary[:, :, None]).sum(axis=1).astype(np.float32)
        local_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        local_normals /= np.linalg.norm(local_normals, axis=1, keepdims=True).clip(min=1e-8)
        self.point_normals = torch.from_numpy(local_normals.astype(np.float32)).to(self.device)
        selected_links = np.asarray(face_links, dtype=object)[selected_faces]
        self.point_groups: list[tuple[str, torch.Tensor, torch.Tensor]] = []
        for link in dict.fromkeys(selected_links.tolist()):
            indices = np.flatnonzero(selected_links == link).astype(np.int64)
            self.point_groups.append(
                (
                    str(link),
                    torch.from_numpy(indices).to(self.device),
                    torch.from_numpy(local_points[indices]).to(self.device),
                )
            )
        self.num_hand_points = int(num_hand_points)
        self.q_lower, self.q_upper = self._joint_limits()

    def normals(self, hand_q: torch.Tensor) -> torch.Tensor:
        """Return sampled hand normals in the hand-root frame [B,1538,3]."""
        hand_q = hand_q.to(self.device, dtype=torch.float32)
        relative = self._link_transforms(hand_q)
        result = hand_q.new_zeros((hand_q.shape[0], self.num_hand_points, 3))
        for link, indices, _ in self.point_groups:
            rotation = relative[link][:, :3, :3]
            normals = self.point_normals.index_select(0, indices)
            transformed = torch.matmul(rotation[:, None], normals[None, :, :, None]).squeeze(-1)
            transformed = transformed / torch.linalg.vector_norm(transformed, dim=-1, keepdim=True).clamp_min(1e-8)
            result.index_copy_(1, indices, transformed)
        return result

    def _joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        xml_joints = {joint.attrib["name"]: joint for joint in ET.parse(self.robot_urdf).getroot().findall("joint")}
        independent = sorted(
            (
                joint for joint in self.urdf.joints
                if joint.qpos_index is not None and 6 <= joint.qpos_index < 12
            ),
            key=lambda joint: int(joint.qpos_index),
        )
        if len(independent) != 6:
            raise ValueError(f"Expected six independent finger joints, found {[joint.name for joint in independent]}")
        limits = []
        for joint in independent:
            limit = xml_joints[joint.name].find("limit")
            if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
                raise ValueError(f"Missing finite limits for {joint.name}")
            limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
        value = torch.tensor(limits, dtype=torch.float32, device=self.device)
        return value[:, 0], value[:, 1]

    def _link_transforms(self, hand_q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return every link transform relative to the hand ``base_link``."""
        if hand_q.ndim != 2 or hand_q.shape[1] != 6:
            raise ValueError(f"Expected hand_q [B,6], got {tuple(hand_q.shape)}")
        hand_q = hand_q.to(self.device, dtype=torch.float32)
        batch = hand_q.shape[0]
        transforms: dict[str, torch.Tensor] = {
            self.urdf.root_link: torch.eye(4, device=self.device).expand(batch, -1, -1)
        }
        joint_values: dict[str, torch.Tensor] = {}
        stack = [self.urdf.root_link]
        while stack:
            parent = stack.pop()
            parent_transform = transforms[parent]
            for joint in self.urdf.child_joints.get(parent, []):
                if joint.mimic_joint is not None:
                    value = joint_values[joint.mimic_joint] * float(joint.mimic_multiplier)
                    value = value + float(joint.mimic_offset)
                elif joint.qpos_index is not None and joint.qpos_index >= 6:
                    value = hand_q[:, int(joint.qpos_index) - 6]
                else:
                    value = hand_q.new_zeros(batch)
                joint_values[joint.name] = value
                origin = torch.as_tensor(joint.origin, dtype=hand_q.dtype, device=self.device)
                origin = origin.expand(batch, -1, -1)
                motion = _axis_motion(joint.axis, value, joint.joint_type)
                transforms[joint.child] = parent_transform @ origin @ motion
                stack.append(joint.child)

        base_inverse = torch.linalg.inv(transforms["base_link"])
        return {link: base_inverse @ transform for link, transform in transforms.items()}

    def bind_points(
        self, hand_q: torch.Tensor, current_points: torch.Tensor
    ) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
        """Bind exact cached points to their URDF links at the current q.

        Cache construction samples faces after float32 FK. Tiny area-rounding
        differences can move a random draw to an adjacent face, so the cached
        current points—not a separately repeated surface draw—are transformed
        back into link-local coordinates before q fitting.
        """
        if current_points.shape != (hand_q.shape[0], self.num_hand_points, 3):
            raise ValueError(
                f"Expected current_points [B,{self.num_hand_points},3], got {tuple(current_points.shape)}"
            )
        relative = self._link_transforms(hand_q)
        groups = []
        for link, indices, _ in self.point_groups:
            transform_inverse = torch.linalg.inv(relative[link])
            selected = current_points.to(self.device, dtype=torch.float32).index_select(1, indices)
            points_h = torch.cat(
                [selected, torch.ones((*selected.shape[:2], 1), device=self.device)], dim=-1
            )
            local = torch.matmul(transform_inverse, points_h.transpose(1, 2))[:, :3].transpose(1, 2)
            groups.append((link, indices, local.detach()))
        return groups

    def points(
        self,
        hand_q: torch.Tensor,
        *,
        bindings: list[tuple[str, torch.Tensor, torch.Tensor]] | None = None,
        cached_link_index: torch.Tensor | None = None,
        cached_local_points: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return hand-root-frame samples `[B,1538,3]` for q `[B,6]`.

        ``cached_link_index``/``cached_local_points`` are the static binding
        written by the layered cache.  They avoid recomputing the current-q FK
        inverse and use one batched gather for all sampled points.
        """
        hand_q = hand_q.to(self.device, dtype=torch.float32)
        relative = self._link_transforms(hand_q)
        batch = hand_q.shape[0]
        if (cached_link_index is None) != (cached_local_points is None):
            raise ValueError("cached_link_index and cached_local_points must be provided together")
        if cached_link_index is not None:
            link_index = cached_link_index.to(self.device, dtype=torch.long)
            local_points = cached_local_points.to(self.device, dtype=torch.float32)
            if link_index.ndim == 2:
                if not torch.equal(link_index, link_index[:1].expand_as(link_index)):
                    raise ValueError("Cached point-to-link binding differs within a batch")
                link_index = link_index[0]
            if local_points.ndim == 3:
                if not torch.allclose(local_points, local_points[:1].expand_as(local_points), atol=1e-7, rtol=0.0):
                    raise ValueError("Cached link-local points differ within a batch")
                local_points = local_points[0]
            if link_index.shape != (self.num_hand_points,) or local_points.shape != (self.num_hand_points, 3):
                raise ValueError(
                    f"Invalid cached point binding shapes: link={tuple(link_index.shape)}, "
                    f"local={tuple(local_points.shape)}"
                )
            group_transforms = torch.stack(
                [relative[link] for link in self.link_order], dim=1
            )
            if int(link_index.min().item()) < 0 or int(link_index.max().item()) >= group_transforms.shape[1]:
                raise ValueError("Cached point binding refers to an unknown hand link")
            point_transforms = group_transforms.index_select(1, link_index)
            points_h = torch.cat(
                [local_points.unsqueeze(0).expand(batch, -1, -1),
                 torch.ones((batch, self.num_hand_points, 1), device=self.device, dtype=local_points.dtype)],
                dim=-1,
            )
            return torch.matmul(point_transforms, points_h.unsqueeze(-1)).squeeze(-1)[..., :3]
        output = hand_q.new_zeros((batch, self.num_hand_points, 3))
        groups = self.point_groups if bindings is None else bindings
        point_indices = torch.cat([indices for _, indices, _ in groups], dim=0)
        local_chunks = []
        transform_chunks = []
        for link, indices, local_points in groups:
            if local_points.ndim == 2:
                local_points = local_points.unsqueeze(0).expand(batch, -1, -1)
            local_chunks.append(local_points)
            transform_chunks.append(relative[link].unsqueeze(1).expand(-1, local_points.shape[1], -1, -1))
        local_points = torch.cat(local_chunks, dim=1)
        point_transforms = torch.cat(transform_chunks, dim=1)
        points_h = torch.cat(
            [local_points, torch.ones((*local_points.shape[:2], 1), device=self.device)], dim=-1
        )
        transformed = torch.matmul(point_transforms, points_h.unsqueeze(-1)).squeeze(-1)[..., :3]
        return output.index_copy(1, point_indices, transformed)


def reconstruct_hand_points(
    hand_model: DifferentiableInspireHand,
    *,
    q_t: torch.Tensor,
    current_hand_points: torch.Tensor,
    predicted_q: torch.Tensor,
    wrist_translation: torch.Tensor,
    wrist_rotvec: torch.Tensor,
    cached_link_index: torch.Tensor | None = None,
    cached_local_points: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reconstruct corresponding target points from predicted articulation and wrist SE(3)."""
    if cached_link_index is None:
        bindings = hand_model.bind_points(q_t, current_hand_points)
        articulated_points = hand_model.points(predicted_q, bindings=bindings)
    else:
        articulated_points = hand_model.points(
            predicted_q,
            cached_link_index=cached_link_index,
            cached_local_points=cached_local_points,
        )
    wrist_rotation = rotvec_to_matrix(wrist_rotvec)
    return articulated_points @ wrist_rotation.transpose(-1, -2) + wrist_translation[:, None, :]


def optimize_q_from_hand_points(
    hand_model: DifferentiableInspireHand,
    *,
    q_t: torch.Tensor,
    current_hand_points: torch.Tensor,
    target_hand_points: torch.Tensor,
    steps: int = 100,
    lr: float = 0.05,
    prior_weight: float = 1e-4,
    wrist_prior_weight: float = 1e-6,
) -> dict[str, Any]:
    """Jointly fit relative wrist SE(3) and bounded finger q outside training."""
    if steps < 0 or lr <= 0.0 or prior_weight < 0.0 or wrist_prior_weight < 0.0:
        raise ValueError("q fitting requires non-negative steps/priors and positive lr")
    q_t = q_t.to(hand_model.device, dtype=torch.float32)
    current = current_hand_points.to(hand_model.device, dtype=torch.float32)
    target = target_hand_points.to(hand_model.device, dtype=torch.float32)
    bindings = hand_model.bind_points(q_t, current)
    q = q_t.detach().clone().requires_grad_(True)
    wrist_translation = q_t.new_zeros((q_t.shape[0], 3), requires_grad=True)
    wrist_rotvec = q_t.new_zeros((q_t.shape[0], 3), requires_grad=True)
    optimizer = torch.optim.Adam([q, wrist_translation, wrist_rotvec], lr=float(lr))
    history = []
    best_loss = float("inf")
    best_q = q.detach().clone()
    best_wrist_translation = wrist_translation.detach().clone()
    best_wrist_rotvec = wrist_rotvec.detach().clone()
    for step in range(int(steps) + 1):
        articulated_points = hand_model.points(q, bindings=bindings)
        wrist_rotation = rotvec_to_matrix(wrist_rotvec)
        points = articulated_points @ wrist_rotation.transpose(-1, -2) + wrist_translation[:, None, :]
        point_loss = (points - target).square().mean()
        prior_loss = (q - q_t).square().mean()
        wrist_prior = wrist_translation.square().mean() + wrist_rotvec.square().mean()
        loss = (
            point_loss
            + float(prior_weight) * prior_loss
            + float(wrist_prior_weight) * wrist_prior
        )
        loss_value = float(loss.detach())
        if loss_value < best_loss:
            best_loss = loss_value
            best_q = q.detach().clone()
            best_wrist_translation = wrist_translation.detach().clone()
            best_wrist_rotvec = wrist_rotvec.detach().clone()
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                q.clamp_(hand_model.q_lower, hand_model.q_upper)
        if step in {0, int(steps)}:
            history.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "point_epe_mm": float(
                        torch.linalg.vector_norm(points.detach() - target, dim=-1).mean() * 1000.0
                    ),
                }
            )
    with torch.no_grad():
        articulated_points = hand_model.points(best_q, bindings=bindings)
        fitted_points = (
            articulated_points @ rotvec_to_matrix(best_wrist_rotvec).transpose(-1, -2)
            + best_wrist_translation[:, None, :]
        )
    return {
        "q": best_q,
        "wrist_delta_translation": best_wrist_translation,
        "wrist_delta_rotvec": best_wrist_rotvec,
        "points": fitted_points.detach(),
        "history": history,
    }
