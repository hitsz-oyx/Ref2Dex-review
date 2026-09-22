"""Frozen-Cmv2 one-step counterfactual teacher for CmResidual V1.18.

This module intentionally has no Isaac Gym dependency.  It provides the GPU
FK and batched candidate path used by the task, while keeping the frozen model
out of the actor, critic, reward, and executed rollout action.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

ACTION_DIM = 18


def _profile_stage(profiler, name: str):
    return profiler.stage(name) if profiler is not None else nullcontext()


QUERY_LINKS = (
    "hand_base_link", "thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_tip", "middle_proximal", "middle_intermediate", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_tip", "pinky_proximal", "pinky_intermediate", "pinky_tip",
)
KEY_LINKS = (
    "hand_base_link", "index_proximal", "index_intermediate", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_tip",
    "ring_proximal", "ring_intermediate", "ring_tip",
    "thumb_proximal_base", "thumb_intermediate", "thumb_tip",
)
IG_KEY_INDICES = (0, 3, 6, 9, 12, 15)
TIP_LINKS = ("index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip")
NATIVE_TO_URDF = (0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9)


def _vec(text: str | None) -> np.ndarray:
    return np.asarray([float(value) for value in (text or "0 0 0").split()], dtype=np.float32)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    rx, ry, rz = rpy
    sx, cx, sy, cy, sz, cz = math.sin(rx), math.cos(rx), math.sin(ry), math.cos(ry), math.sin(rz), math.cos(rz)
    return np.asarray(((cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
                       (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
                       (-sy, cy * sx, cy * cx)), dtype=np.float32)


def _origin(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    if element is not None:
        result[:3, 3] = _vec(element.get("xyz"))
        result[:3, :3] = _rpy_matrix(_vec(element.get("rpy")))
    return result


@dataclass(frozen=True)
class _Joint:
    parent: str
    child: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray
    q_index: int


class TorchInspireKinematics:
    """URDF-pinned vectorized Torch FK for ``[B,K,18]`` native targets."""

    def __init__(self, urdf_path: str | Path, device: torch.device | str) -> None:
        path = Path(urdf_path).resolve()
        root = ET.parse(path).getroot()
        links = tuple(str(link.get("name")) for link in root.findall("link"))
        joints, children, child_links = [], {}, set()
        q_index = 0
        for element in root.findall("joint"):
            kind = str(element.get("type", "fixed"))
            parent = str(element.find("parent").get("link"))
            child = str(element.find("child").get("link"))
            axis_element = element.find("axis")
            joint = _Joint(parent, child, kind, _origin(element.find("origin")),
                           _vec(axis_element.get("xyz") if axis_element is not None else None),
                           q_index if kind != "fixed" else -1)
            if joint.q_index >= 0:
                q_index += 1
            joints.append(joint)
            children.setdefault(parent, []).append(joint)
            child_links.add(child)
        if q_index != ACTION_DIM:
            raise ValueError(f"Expected 18 URDF joints, got {q_index}")
        missing = set(QUERY_LINKS) - set(links)
        if missing:
            raise ValueError(f"URDF misses V1.18 query links: {sorted(missing)}")
        self.device = torch.device(device)
        self.root_link = next(link for link in links if link not in child_links)
        self.children = children
        self.origin = {joint.child: torch.as_tensor(joint.origin, device=self.device) for joint in joints}
        self.axis = {joint.child: torch.as_tensor(joint.axis, device=self.device) for joint in joints}

    @staticmethod
    def _motion(kind: str, axis: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        shape = value.shape
        result = torch.eye(4, device=value.device, dtype=value.dtype).expand(*shape, 4, 4).clone()
        if kind == "fixed":
            return result
        axis = axis.to(dtype=value.dtype)
        axis = axis / axis.norm().clamp_min(1e-9)
        if kind == "prismatic":
            result[..., :3, 3] = axis * value[..., None]
            return result
        if kind not in ("revolute", "continuous"):
            raise ValueError(f"Unsupported URDF joint type {kind!r}")
        x, y, z = axis
        cosine, sine = value.cos(), value.sin()
        one = 1.0 - cosine
        result[..., :3, :3] = torch.stack((
            cosine + x * x * one, x * y * one - z * sine, x * z * one + y * sine,
            y * x * one + z * sine, cosine + y * y * one, y * z * one - x * sine,
            z * x * one - y * sine, z * y * one + x * sine, cosine + z * z * one,
        ), dim=-1).reshape(*shape, 3, 3)
        return result

    def forward(self, native_q: torch.Tensor) -> torch.Tensor:
        if native_q.ndim != 3 or native_q.shape[-1] != ACTION_DIM:
            raise ValueError(f"native_q must be [B,K,18], got {tuple(native_q.shape)}")
        if native_q.device != self.device or not torch.isfinite(native_q).all():
            raise ValueError("native_q must be finite and on the FK device")
        urdf_q = torch.empty_like(native_q)
        urdf_q[..., list(NATIVE_TO_URDF)] = native_q
        transforms = {self.root_link: torch.eye(4, device=self.device, dtype=native_q.dtype).expand(
            *native_q.shape[:2], 4, 4)}

        def visit(parent: str) -> None:
            for joint in self.children.get(parent, ()):
                origin = self.origin[joint.child].to(dtype=native_q.dtype)
                origin = origin.expand(*native_q.shape[:2], 4, 4)
                value = (urdf_q[..., joint.q_index] if joint.q_index >= 0
                         else torch.zeros(native_q.shape[:2], device=self.device, dtype=native_q.dtype))
                transforms[joint.child] = transforms[parent] @ origin @ self._motion(
                    joint.kind, self.axis[joint.child], value)
                visit(joint.child)

        visit(self.root_link)
        result = torch.stack([transforms[name] for name in QUERY_LINKS], dim=2)
        if result.shape != (*native_q.shape[:2], len(QUERY_LINKS), 4, 4) or not torch.isfinite(result).all():
            raise RuntimeError("V1.18 GPU FK returned invalid link transforms")
        return result


def _rotvec_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True)
    axis = rotvec / angle.clamp_min(1e-9)
    x, y, z = axis.unbind(-1)
    cosine, sine, one = angle[..., 0].cos(), angle[..., 0].sin(), 1.0 - angle[..., 0].cos()
    return torch.stack((
        cosine + x * x * one, x * y * one - z * sine, x * z * one + y * sine,
        y * x * one + z * sine, cosine + y * y * one, y * z * one - x * sine,
        z * x * one - y * sine, z * y * one + x * sine, cosine + z * z * one,
    ), dim=-1).reshape(*rotvec.shape[:-1], 3, 3)


def _pose_from_delta(current: torch.Tensor, delta_xi: torch.Tensor) -> torch.Tensor:
    """Apply a current-object-local [translation, rotvec] effect."""
    result = current[:, None].expand(-1, delta_xi.shape[1], -1, -1).clone()
    local = torch.eye(4, device=current.device, dtype=current.dtype).expand_as(result).clone()
    local[..., :3, :3] = _rotvec_matrix(delta_xi[..., 3:])
    local[..., :3, 3] = delta_xi[..., :3]
    return result @ local


@dataclass(frozen=True)
class PlannerConfig:
    candidates: int = 8
    perturbation_std: float = 0.15
    temperature: float = 1.0
    effect_weight: float = 1.0
    ig_weight: float = 0.5
    trust_weight: float = 0.05
    translation_scale_m: float = 0.02
    rotation_scale_rad: float = 0.05
    gate_distance_m: float = 0.04
    effect_translation_gate_m: float = 0.002
    effect_rotation_gate_rad: float = 0.01
    planner_env_microbatch: int = 16
    interaction_object_chunk: int = 32
    interaction_hand_chunk: int = 256


class FrozenCmv2Planner:
    """Creates a detached teacher by streaming active env and Cmv2 object chunks."""

    def __init__(self, adapter, geometry, kinematics: TorchInspireKinematics, config: PlannerConfig) -> None:
        if config.candidates != 8:
            raise ValueError("V1.18 contract fixes K=8")
        if min(config.perturbation_std, config.temperature, config.translation_scale_m,
               config.rotation_scale_rad, config.gate_distance_m) <= 0:
            raise ValueError("V1.18 planner scales must be positive")
        self.adapter, self.geometry, self.kinematics, self.config = adapter, geometry, kinematics, config
        if config.planner_env_microbatch <= 0:
            raise ValueError("planner_env_microbatch must be positive")
        if config.interaction_object_chunk <= 0:
            raise ValueError("interaction_object_chunk must be positive")
        if config.interaction_hand_chunk <= 0:
            raise ValueError("interaction_hand_chunk must be positive")
        self.key_query_indices = torch.as_tensor([QUERY_LINKS.index(name) for name in KEY_LINKS], dtype=torch.long,
                                                 device=kinematics.device)
        self.tip_query_indices = torch.as_tensor([QUERY_LINKS.index(name) for name in TIP_LINKS], dtype=torch.long,
                                                 device=kinematics.device)
        self.hand_link_index = None
        if hasattr(geometry, "hand_link"):
            link_ids = np.asarray(geometry.hand_link, dtype=np.int64)
            groups = [np.flatnonzero(link_ids == link) for link in range(len(QUERY_LINKS))]
            width = max(map(len, groups))
            table = np.full((len(groups), width), len(link_ids), dtype=np.int64)
            for link, values in enumerate(groups):
                table[link, :len(values)] = values
            self.hand_link_index = torch.as_tensor(table, device=kinematics.device)

    def _candidates(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.ndim != 2 or mu.shape[-1] != ACTION_DIM:
            raise ValueError("policy mean must be [B,18]")
        eps = torch.randn((mu.shape[0], 3, ACTION_DIM), device=mu.device, dtype=mu.dtype) * self.config.perturbation_std
        result = torch.cat((mu[:, None], torch.zeros_like(mu[:, None]), mu[:, None] + eps, mu[:, None] - eps), dim=1)
        return result.clamp(-1.0, 1.0)

    def _activation(self, current_links: torch.Tensor, object_points: torch.Tensor,
                    reference_transport: torch.Tensor, desired_delta: torch.Tensor) -> torch.Tensor:
        tips = current_links.index_select(1, self.tip_query_indices)[..., :3, 3]
        current_near = torch.cdist(tips, object_points).amin(dim=(1, 2)) <= self.config.gate_distance_m
        ref_key = reference_transport[:, 119:167].view(-1, 16, 3)
        ref_near = torch.cdist(ref_key[:, list(IG_KEY_INDICES)], object_points).amin(dim=(1, 2)) <= self.config.gate_distance_m
        effect = ((desired_delta[:, :3].norm(dim=-1) >= self.config.effect_translation_gate_m) |
                  (desired_delta[:, 3:].norm(dim=-1) >= self.config.effect_rotation_gate_rad))
        return current_near | ref_near | effect

    @torch.inference_mode()
    def teacher(self, *, mu: torch.Tensor, current_native: torch.Tensor, base_target: torch.Tensor,
                native_lower: torch.Tensor, native_upper: torch.Tensor, mimic_scales: tuple[float, ...],
                current_links: torch.Tensor, object_pose: torch.Tensor, reference_transport: torch.Tensor,
                desired_delta_xi: torch.Tensor, _latency_profiler=None,
                _diagnostic_candidate_count: int | None = None,
                _interaction_variant: str = "baseline") -> dict[str, torch.Tensor]:
        """Return detached action/weight.  All inputs are pre-action simulator state."""
        batch = mu.shape[0]
        candidate_count = self.config.candidates if _diagnostic_candidate_count is None else _diagnostic_candidate_count
        if candidate_count not in (1, 2, 4, 8):
            raise ValueError("diagnostic candidate count must be one of 1, 2, 4, 8")
        if _interaction_variant not in ("baseline", "merge", "link_aabb", "link_sparse"):
            raise ValueError(f"unknown interaction variant: {_interaction_variant}")
        if _interaction_variant.startswith("link") and self.hand_link_index is None:
            raise ValueError("link interaction variants require geometry.hand_link")
        fallback = {"teacher_action": mu.detach(), "teacher_weight": torch.zeros(batch, device=mu.device),
                    "activation": torch.zeros(batch, device=mu.device), "valid_fraction": torch.zeros(batch, device=mu.device),
                    "predicted_cost_improvement": torch.zeros(batch, device=mu.device)}
        with _profile_stage(_latency_profiler, "candidate_generation"):
            candidate_actions = self._candidates(mu)[:, :candidate_count]
        with torch.no_grad():
            # Delayed imports avoid making this task-local module initialize the
            # global IsaacGym task registry during FK-only tests.
            from isaacgymenvs.tasks.cm_residual.action_mapping import compose_reference_residual
            from isaacgymenvs.tasks.cm_residual.dexplore_observation import calc_heading_quat_inv, quat_rotate
            with _profile_stage(_latency_profiler, "state_geometry"):
                object_points, object_normals = self.geometry.object(object_pose)
                active = self._activation(current_links, object_points, reference_transport, desired_delta_xi)
            if not bool(active.any()):
                return fallback
            active_ids = active.nonzero(as_tuple=False).flatten()
            # Process every active env exactly once; only the transient Cmv2
            # candidate batch is bounded by planner_env_microbatch.
            for ids in active_ids.split(self.config.planner_env_microbatch):
                actions = candidate_actions.index_select(0, ids)
                count = len(ids)
                expanded_base = base_target.index_select(0, ids)[:, None].expand(-1, candidate_count, -1).reshape(-1, ACTION_DIM)
                expanded_actions = actions.reshape(-1, ACTION_DIM)
                lower = native_lower[None].expand_as(expanded_base)
                upper = native_upper[None].expand_as(expanded_base)
                targets, details = compose_reference_residual(
                    expanded_base, expanded_actions, lower, upper,
                    translation_scale_m=0.015, rotation_scale_rad=0.20, finger_scale_rad=0.08,
                    mimic_scales=mimic_scales)
                targets = targets.view(count, candidate_count, ACTION_DIM)
                applied_delta = details["applied_delta"].view(count, candidate_count, ACTION_DIM)
                feasible = torch.isfinite(targets).all(dim=-1) & torch.isfinite(applied_delta).all(dim=-1)
                with _profile_stage(_latency_profiler, "fk"):
                    next_links = self.kinematics.forward(targets)
                with _profile_stage(_latency_profiler, "hand_surface"):
                    current_points, current_normals = self.geometry.hand(current_links.index_select(0, ids))
                    next_points, next_normals = self.geometry.hand(next_links.reshape(-1, len(QUERY_LINKS), 4, 4))
                    next_points = next_points.view(count, candidate_count, -1, 3)
                    next_normals = next_normals.view(count, candidate_count, -1, 3)
                hand_count = next_points.shape[2]
                if hasattr(self.adapter, "predict_candidates"):
                    if _interaction_variant != "baseline" or _latency_profiler is not None:
                        raise ValueError("V1.14a adapter does not use legacy V1.18b diagnostic variants")
                    # V1.14a was trained in the current object frame.  Keep the
                    # planner's world-frame activation/scoring, but transform its
                    # learned-model inputs into that pinned local convention.
                    pose = object_pose.index_select(0, ids)
                    rotation, translation = pose[:, :3, :3], pose[:, :3, 3]

                    def local_points(value):
                        return torch.matmul(
                            (value - translation[(slice(None),) + (None,) * (value.ndim - 2)]).unsqueeze(-2),
                            rotation[(slice(None),) + (None,) * (value.ndim - 3)]).squeeze(-2)

                    def local_vectors(value):
                        return torch.matmul(
                            value.unsqueeze(-2),
                            rotation[(slice(None),) + (None,) * (value.ndim - 3)]).squeeze(-2)

                    object_local = local_points(object_points.index_select(0, ids))
                    object_normal_local = local_vectors(object_normals.index_select(0, ids))
                    current_local = local_points(current_points)
                    current_normal_local = local_vectors(current_normals)
                    next_local = local_points(next_points)
                    output = self.adapter.predict_candidates(
                        object_local, object_normal_local, current_local, current_normal_local,
                        next_local - current_local[:, None], 1.0 / 30.0,
                        torch.ones((count, hand_count), dtype=torch.bool, device=mu.device),
                        interaction_object_chunk=self.config.interaction_object_chunk,
                        interaction_hand_chunk=self.config.interaction_hand_chunk)
                else:
                    obj_points = object_points.index_select(0, ids)[:, None].expand(-1, candidate_count, -1, -1).reshape(-1, object_points.shape[1], 3)
                    obj_normals = object_normals.index_select(0, ids)[:, None].expand(-1, candidate_count, -1, -1).reshape(-1, object_normals.shape[1], 3)
                    current_batch = current_points[:, None].expand(-1, candidate_count, -1, -1).reshape(-1, hand_count, 3)
                    current_normal_batch = current_normals[:, None].expand(-1, candidate_count, -1, -1).reshape(-1, hand_count, 3)
                    adapter_kwargs = {"interaction_object_chunk": self.config.interaction_object_chunk}
                    if _latency_profiler is not None:
                        adapter_kwargs["latency_profiler"] = _latency_profiler
                    if _interaction_variant != "baseline":
                        adapter_kwargs.update({
                            "swept_algorithm": ("merge" if _interaction_variant == "merge" else "link_aabb"),
                            "hand_link_index": (self.hand_link_index if _interaction_variant.startswith("link") else None),
                            "sparse_valid_edges": _interaction_variant == "link_sparse",
                            "candidate_group_size": candidate_count,
                        })
                    output = self.adapter.predict_effect_only(
                        obj_points, obj_normals, current_batch, current_normal_batch,
                        next_points.reshape(-1, hand_count, 3) - current_batch,
                        1.0 / 30.0,
                        torch.ones((count * candidate_count, hand_count),
                                   dtype=torch.bool, device=mu.device),
                        **adapter_kwargs)
                with _profile_stage(_latency_profiler, "scoring"):
                    predicted = output["delta_xi_root"].view(count, candidate_count, 6)
                    token_mask = output["token_mask"].view(count, candidate_count, -1)
                    token_mass = output["token_mass"].view(count, candidate_count, -1)
                    valid = feasible & token_mask.any(dim=-1) & (token_mass.sum(dim=-1) > 0) & torch.isfinite(predicted).all(dim=-1)
                    desired = desired_delta_xi.index_select(0, ids)[:, None]
                    translation_cost = (predicted[..., :3] - desired[..., :3]).square().sum(dim=-1) / self.config.translation_scale_m ** 2
                    rotation_cost = (predicted[..., 3:] - desired[..., 3:]).square().sum(dim=-1) / self.config.rotation_scale_rad ** 2
                    next_object = _pose_from_delta(object_pose.index_select(0, ids), predicted)
                # object_points_world only needs position/quaternion; use transformed sampled points directly for stable costs.
                    local_obj = torch.as_tensor(self.geometry.object_local, device=mu.device, dtype=mu.dtype)
                    predicted_points = torch.matmul(local_obj[None, None, :, None, :], next_object[:, :, None, :3, :3].transpose(-1, -2)).squeeze(-2) + next_object[:, :, None, :3, 3]
                    key_pos = next_links.index_select(2, self.key_query_indices)[:, :, list(IG_KEY_INDICES), :3, 3]
                    root_quat = torch.zeros((count, candidate_count, 4), dtype=mu.dtype, device=mu.device)
                # Heading only needs a unit quaternion; derive it from the root transform via a stable trace conversion.
                    root_rot = next_links[:, :, 0, :3, :3]
                    trace = root_rot.diagonal(dim1=-2, dim2=-1).sum(-1)
                    root_quat[..., 3] = torch.sqrt((1.0 + trace).clamp_min(1e-8)) * 0.5
                    root_quat[..., 0] = (root_rot[..., 2, 1] - root_rot[..., 1, 2]) / (4.0 * root_quat[..., 3]).clamp_min(1e-8)
                    root_quat[..., 1] = (root_rot[..., 0, 2] - root_rot[..., 2, 0]) / (4.0 * root_quat[..., 3]).clamp_min(1e-8)
                    root_quat[..., 2] = (root_rot[..., 1, 0] - root_rot[..., 0, 1]) / (4.0 * root_quat[..., 3]).clamp_min(1e-8)
                    displacement = key_pos[:, :, :, None, :] - predicted_points[:, :, None, :, :]
                    nearest = displacement.norm(dim=-1).argmin(dim=-1)
                    gather = nearest[..., None, None].expand(-1, -1, -1, 1, 3)
                    candidate_ig = displacement.gather(3, gather).squeeze(3)
                    heading = calc_heading_quat_inv(root_quat.reshape(-1, 4)).view(count, candidate_count, 4)
                    candidate_ig = quat_rotate(heading[:, :, None].expand(-1, -1, 6, -1).reshape(-1, 4),
                                               candidate_ig.reshape(-1, 3)).view(count, candidate_count, -1)
                    reference_ig = reference_transport.index_select(0, ids)[:, 184:232].view(count, 16, 3)[:, list(IG_KEY_INDICES)].reshape(count, 1, -1)
                    ig_cost = (candidate_ig - reference_ig).square().mean(dim=-1) / self.config.translation_scale_m ** 2
                    trust_cost = (actions - mu.index_select(0, ids)[:, None]).square().mean(dim=-1)
                    costs = self.config.effect_weight * (translation_cost + rotation_cost) + self.config.ig_weight * ig_cost + self.config.trust_weight * trust_cost
                    costs = costs.masked_fill(~valid, float("inf"))
                    has_valid = valid.any(dim=-1)
                    weights = torch.softmax((-costs / self.config.temperature).masked_fill(~valid, -float("inf")), dim=-1)
                    weights = torch.where(has_valid[:, None], weights, torch.zeros_like(weights))
                    teacher = (weights[..., None] * actions).sum(dim=1)
                    fallback["teacher_action"].index_copy_(0, ids, torch.where(has_valid[:, None], teacher, mu.index_select(0, ids)))
                    fallback["teacher_weight"].index_copy_(0, ids, has_valid.to(mu.dtype))
                    fallback["activation"].index_copy_(0, ids, torch.ones(count, device=mu.device))
                    fallback["valid_fraction"].index_copy_(0, ids, valid.to(mu.dtype).mean(dim=-1))
                    baseline = costs[:, 0]
                    selected = (weights * costs.masked_fill(~valid, 0.0)).sum(dim=-1)
                    improvement = torch.where(has_valid & torch.isfinite(baseline), baseline - selected, torch.zeros_like(selected))
                    fallback["predicted_cost_improvement"].index_copy_(0, ids, improvement)
        return fallback
