"""Dexplore Inspire q contract and deterministic Task-local URDF kinematics."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


NUM_NATIVE_DOFS = 18
NATIVE_Q_START = 373
NATIVE_TO_URDF = np.asarray(
    [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9],
    dtype=np.int64,
)
INDEPENDENT_FINGER_NATIVE_INDICES = np.asarray([6, 8, 10, 12, 14, 15], dtype=np.int64)
QUERY_LINKS = (
    "hand_base_link",
    "thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_tip",
)
# ``finger_q`` follows Dexplore native independent order:
# index, middle, pinky, ring, thumb-yaw, thumb-pitch.
QUERY_LINK_Q_INDICES = np.asarray([-1, 4, 5, 5, 5, 5, 0, 0, 0, 1, 1, 1, 3, 3, 3, 2, 2, 2])
QUERY_LINK_NATIVE_Q_INDICES = np.asarray([-1, 14, 15, 15, 15, 15, 6, 6, 6, 8, 8, 8, 12, 12, 12, 10, 10, 10])


def expand_finger_q(finger_q: np.ndarray, wrist_native: np.ndarray | None = None) -> np.ndarray:
    """Expand six actuated finger values to Dexplore's 18-value native q vector."""
    finger_q = np.asarray(finger_q, dtype=np.float64)
    if finger_q.shape[-1] != 6:
        raise ValueError(f"Expected finger_q[...,6], got {finger_q.shape}")
    native = np.zeros((*finger_q.shape[:-1], NUM_NATIVE_DOFS), dtype=np.float64)
    if wrist_native is not None:
        wrist_native = np.asarray(wrist_native, dtype=np.float64)
        if wrist_native.shape != (*finger_q.shape[:-1], 6):
            raise ValueError(f"Expected wrist_native {(*finger_q.shape[:-1], 6)}, got {wrist_native.shape}")
        native[..., :6] = wrist_native
    native[..., INDEPENDENT_FINGER_NATIVE_INDICES] = finger_q
    native[..., 7] = native[..., 6] * 1.05
    native[..., 9] = native[..., 8] * 1.05
    native[..., 11] = native[..., 10] * 1.05
    native[..., 13] = native[..., 12] * 1.05
    native[..., 16] = native[..., 15] * 0.6
    native[..., 17] = native[..., 15] * 0.8
    return native


def extract_finger_q(native_q: np.ndarray) -> np.ndarray:
    native_q = np.asarray(native_q)
    if native_q.shape[-1] != NUM_NATIVE_DOFS:
        raise ValueError(f"Expected native_q[...,18], got {native_q.shape}")
    return np.asarray(native_q[..., INDEPENDENT_FINGER_NATIVE_INDICES], dtype=np.float64)


def _vec(text: str | None, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(3, default, dtype=np.float64)
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (3,):
        raise ValueError(f"Expected three values, got {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    rx, ry, rz = (float(value) for value in rpy)
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rx_m = np.asarray([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry_m = np.asarray([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz_m = np.asarray([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz_m @ ry_m @ rx_m


def _origin(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if element is not None:
        result[:3, 3] = _vec(element.get("xyz"))
        result[:3, :3] = _rpy_matrix(_vec(element.get("rpy")))
    return result


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.asarray([
        [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
        [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
        [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
    ], dtype=np.float64)


def _motion(joint_type: str, axis: np.ndarray, value: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if joint_type == "prismatic":
        result[:3, 3] = axis * value
    elif joint_type in {"revolute", "continuous"}:
        result[:3, :3] = _axis_angle(axis, value)
    elif joint_type != "fixed":
        raise ValueError(f"Unsupported joint type {joint_type!r}")
    return result


@dataclass(frozen=True)
class Joint:
    parent: str
    child: str
    joint_type: str
    origin: np.ndarray
    axis: np.ndarray
    q_index: int
    lower: float
    upper: float


class InspireKinematics:
    """Minimal FK implementation pinned to the Dexplore right-hand URDF."""

    def __init__(self, urdf_path: str | Path) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Missing Inspire URDF: {self.urdf_path}")
        root = ET.parse(self.urdf_path).getroot()
        self.link_names = tuple(str(link.get("name")) for link in root.findall("link"))
        child_links: set[str] = set()
        joints: list[Joint] = []
        q_index = 0
        for element in root.findall("joint"):
            joint_type = str(element.get("type", "fixed"))
            parent = str(element.find("parent").get("link"))
            child = str(element.find("child").get("link"))
            axis_element = element.find("axis")
            axis = _vec(axis_element.get("xyz") if axis_element is not None else None)
            limit = element.find("limit")
            lower = float(limit.get("lower", "-inf")) if limit is not None else -math.inf
            upper = float(limit.get("upper", "inf")) if limit is not None else math.inf
            current_q = q_index if joint_type != "fixed" else -1
            if current_q >= 0:
                q_index += 1
            joints.append(Joint(parent, child, joint_type, _origin(element.find("origin")), axis, current_q, lower, upper))
            child_links.add(child)
        if q_index != NUM_NATIVE_DOFS:
            raise ValueError(f"Expected 18 actuated URDF joints, got {q_index}")
        self.root_link = next(link for link in self.link_names if link not in child_links)
        self.joints = tuple(joints)
        self.children: dict[str, list[Joint]] = {}
        for joint in self.joints:
            self.children.setdefault(joint.parent, []).append(joint)
        missing = set(QUERY_LINKS) - set(self.link_names)
        if missing:
            raise ValueError(f"URDF misses decoder query links: {sorted(missing)}")
        zero_links = self.link_transforms_native(np.zeros(NUM_NATIVE_DOFS, dtype=np.float64))
        self._zero_hand_base_inverse = np.linalg.inv(zero_links["hand_base_link"])
        lower_urdf = np.asarray([j.lower for j in self.joints if j.q_index >= 0], dtype=np.float64)
        upper_urdf = np.asarray([j.upper for j in self.joints if j.q_index >= 0], dtype=np.float64)
        lower_native = lower_urdf[NATIVE_TO_URDF]
        upper_native = upper_urdf[NATIVE_TO_URDF]
        self.finger_lower = lower_native[INDEPENDENT_FINGER_NATIVE_INDICES]
        self.finger_upper = upper_native[INDEPENDENT_FINGER_NATIVE_INDICES]

    @staticmethod
    def native_to_urdf(native_q: np.ndarray) -> np.ndarray:
        native_q = np.asarray(native_q, dtype=np.float64)
        if native_q.shape != (NUM_NATIVE_DOFS,):
            raise ValueError(f"Expected native q [18], got {native_q.shape}")
        urdf_q = np.empty_like(native_q)
        urdf_q[NATIVE_TO_URDF] = native_q
        return urdf_q

    def link_transforms_native(self, native_q: np.ndarray) -> dict[str, np.ndarray]:
        urdf_q = self.native_to_urdf(native_q)
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}

        def visit(parent: str) -> None:
            for joint in self.children.get(parent, []):
                value = urdf_q[joint.q_index] if joint.q_index >= 0 else 0.0
                transforms[joint.child] = transforms[parent] @ joint.origin @ _motion(joint.joint_type, joint.axis, value)
                visit(joint.child)

        visit(self.root_link)
        if len(transforms) != len(self.link_names):
            raise ValueError("Disconnected Inspire URDF")
        return transforms

    def wrist_pose_from_native(self, native_q: np.ndarray) -> np.ndarray:
        return self.link_transforms_native(native_q)["hand_base_link"].copy()

    def link_transforms_from_state(self, finger_q: np.ndarray, wrist_pose_world: np.ndarray) -> dict[str, np.ndarray]:
        finger_q = self.clamp_finger_q(finger_q)
        pose = np.asarray(wrist_pose_world, dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected wrist pose [4,4], got {pose.shape}")
        canonical = self.link_transforms_native(expand_finger_q(finger_q))
        world_from_canonical = pose @ self._zero_hand_base_inverse
        return {name: world_from_canonical @ transform for name, transform in canonical.items()}

    def clamp_finger_q(self, finger_q: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(finger_q, dtype=np.float64), self.finger_lower, self.finger_upper)

    def query_features(self, finger_q: np.ndarray, wrist_pose_world: np.ndarray, object_pose_world: np.ndarray) -> np.ndarray:
        """Return [18,10] = scalar q + object-frame position + rotation-6D."""
        finger_q = self.clamp_finger_q(finger_q)
        links = self.link_transforms_from_state(finger_q, wrist_pose_world)
        object_from_world = np.linalg.inv(np.asarray(object_pose_world, dtype=np.float64))
        result = np.empty((len(QUERY_LINKS), 10), dtype=np.float32)
        for index, name in enumerate(QUERY_LINKS):
            transform = object_from_world @ links[name]
            q_index = int(QUERY_LINK_Q_INDICES[index])
            result[index, 0] = 0.0 if q_index < 0 else float(finger_q[q_index])
            result[index, 1:4] = transform[:3, 3]
            result[index, 4:10] = transform[:3, :2].T.reshape(-1)
        return result

    def query_features_native(
        self, native_q: np.ndarray, object_pose_world: np.ndarray
    ) -> np.ndarray:
        """Compute observation link features from the complete native 18-DOF state.

        This path deliberately does not expand six controls or impose mimic
        ratios.  It is used for actual Inspire observations; six-dimensional
        control targets continue to use :meth:`query_features`.
        """
        native_q = np.asarray(native_q, dtype=np.float64)
        if native_q.shape != (NUM_NATIVE_DOFS,):
            raise ValueError(f"Expected native_q [18], got {native_q.shape}")
        object_from_world = np.linalg.inv(np.asarray(object_pose_world, dtype=np.float64))
        links = self.link_transforms_native(native_q)
        result = np.empty((len(QUERY_LINKS), 10), dtype=np.float32)
        for index, name in enumerate(QUERY_LINKS):
            transform = object_from_world @ links[name]
            q_index = int(QUERY_LINK_NATIVE_Q_INDICES[index])
            result[index, 0] = 0.0 if q_index < 0 else float(native_q[q_index])
            result[index, 1:4] = transform[:3, 3]
            result[index, 4:10] = transform[:3, :2].T.reshape(-1)
        return result


def relative_pose(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(current, dtype=np.float64)) @ np.asarray(future, dtype=np.float64)


def rotation_matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    angle = math.acos(float(cosine))
    if angle < 1e-7:
        return 0.5 * np.asarray([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    if math.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        axis[0] = math.copysign(axis[0], matrix[2, 1] - matrix[1, 2])
        axis[1] = math.copysign(axis[1], matrix[0, 2] - matrix[2, 0])
        axis /= max(np.linalg.norm(axis), 1e-8)
        return axis * angle
    axis = np.asarray([matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]])
    return axis * (angle / (2.0 * math.sin(angle)))


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    return _axis_angle(rotvec / max(angle, 1e-12), angle)
