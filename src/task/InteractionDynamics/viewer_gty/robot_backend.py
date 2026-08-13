"""零额外依赖的最小 URDF visual-mesh FK backend。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import axis_angle_to_matrix, euler_angles_to_matrix


ROBOT_URDFS = {
    "Allegro": "allegro_hand/allegro_hand_right.urdf",
    "LEAP": "leap_hand/leap_hand_right.urdf",
    "Shadow": "shadow_hand/shadow_hand_right.urdf",
    "Barrett": "barrett_hand/bhand_model.urdf",
}


def scan_robot_assets(root: str | Path) -> dict[str, Path]:
    hands = Path(root) / "dex-urdf/robots/hands"
    return {name: hands / relative for name, relative in ROBOT_URDFS.items()
            if (hands / relative).is_file()}


def _values(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray(default if text is None else [float(value) for value in text.split()], np.float32)


def _origin(element: ET.Element | None) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)
    if element is None:
        return result
    xyz = _values(element.get("xyz"), (0, 0, 0)); rpy = _values(element.get("rpy"), (0, 0, 0))
    result[:3, :3] = euler_angles_to_matrix(torch.from_numpy(rpy), "XYZ").numpy()
    result[:3, 3] = xyz
    return result


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: torch.Tensor
    axis: torch.Tensor
    index: int | None


@dataclass(frozen=True)
class Visual:
    link: str
    vertices: torch.Tensor
    faces: np.ndarray
    origin: torch.Tensor


class UrdfHandBackend:
    """只实现 V20.5 所需的 fixed/revolute joint 与 triangle visual mesh。"""

    def __init__(self, urdf: str | Path, device: torch.device | str = "cpu") -> None:
        self.urdf = Path(urdf); self.device = torch.device(device)
        tree = ET.parse(self.urdf); root = tree.getroot(); children = set(); joint_rows = []
        limits, names = [], []
        for element in root.findall("joint"):
            kind = element.get("type"); parent = element.find("parent").get("link")
            child = element.find("child").get("link"); children.add(child)
            index = None
            if kind in {"revolute", "continuous"}:
                index = len(names); names.append(element.get("name")); limit = element.find("limit")
                limits.append((-np.pi, np.pi) if limit is None else
                              (float(limit.get("lower", -np.pi)), float(limit.get("upper", np.pi))))
            elif kind != "fixed":
                raise ValueError(f"暂不支持 URDF joint type: {kind}")
            joint_rows.append(Joint(element.get("name"), parent, child,
                torch.from_numpy(_origin(element.find("origin"))).to(self.device),
                torch.from_numpy(_values(None if element.find("axis") is None else
                    element.find("axis").get("xyz"), (1, 0, 0))).to(self.device), index))
        links = {element.get("name") for element in root.findall("link")}
        roots = links - children
        if len(roots) != 1:
            raise ValueError(f"URDF 需要唯一 root link，实际为 {sorted(roots)}")
        self.root_link = next(iter(roots)); self.joints = joint_rows; self.joint_names = tuple(names)
        self.limits = torch.tensor(limits, dtype=torch.float32, device=self.device)
        visuals = []
        for link in root.findall("link"):
            for visual in link.findall("visual"):
                mesh_element = visual.find("geometry/mesh")
                if mesh_element is None:
                    continue
                path = (self.urdf.parent / mesh_element.get("filename")).resolve()
                mesh = trimesh.load(path, force="mesh", process=False)
                scale = _values(mesh_element.get("scale"), (1, 1, 1))
                vertices = np.asarray(mesh.vertices, np.float32) * scale
                visuals.append(Visual(link.get("name"), torch.from_numpy(vertices).to(self.device),
                                      np.asarray(mesh.faces, np.int32),
                                      torch.from_numpy(_origin(visual.find("origin"))).to(self.device)))
        if not visuals:
            raise ValueError(f"URDF 没有可用 visual triangle mesh: {self.urdf}")
        self.visuals = visuals
        faces, offset = [], 0
        for visual in visuals:
            faces.append(visual.faces + offset); offset += len(visual.vertices)
        self.faces = np.concatenate(faces).astype(np.int32)

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def vertices(self, q: torch.Tensor, root_rotation: torch.Tensor,
                 root_translation: torch.Tensor,
                 vertex_indices: torch.Tensor | None = None) -> torch.Tensor:
        """输入 `q[T,D]`、axis-angle/translation `[T,3]`，返回 `[T,V,3]`。"""
        frames = len(q)

        def homogeneous(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
            upper = torch.cat((rotation, translation[..., None]), -1)
            lower = torch.zeros((*rotation.shape[:-2], 1, 4), device=rotation.device,
                                dtype=rotation.dtype); lower[..., 0, 3] = 1
            return torch.cat((upper, lower), -2)

        transforms = {self.root_link: homogeneous(axis_angle_to_matrix(root_rotation), root_translation)}
        pending = list(self.joints)
        while pending:
            progress = False
            for joint in pending[:]:
                if joint.parent not in transforms:
                    continue
                local = joint.origin[None].repeat(frames, 1, 1)
                if joint.index is not None:
                    rotation = axis_angle_to_matrix(q[:, joint.index, None] * joint.axis[None])
                    joint_rotation = homogeneous(rotation, torch.zeros(frames, 3, device=q.device))
                    local = local @ joint_rotation
                transforms[joint.child] = transforms[joint.parent] @ local
                pending.remove(joint); progress = True
            if not progress:
                raise ValueError("URDF joint graph 不连通")
        outputs = []; start = 0
        for visual in self.visuals:
            transform = transforms[visual.link] @ visual.origin[None]
            vertices = visual.vertices
            if vertex_indices is not None:
                mask = (vertex_indices >= start) & (vertex_indices < start + len(vertices))
                vertices = vertices[vertex_indices[mask] - start]
            start += len(visual.vertices)
            if len(vertices):
                outputs.append(vertices[None] @ transform[:, :3, :3].transpose(-1, -2)
                               + transform[:, None, :3, 3])
        return torch.cat(outputs, 1)

    def joint_margin(self, q: torch.Tensor) -> torch.Tensor:
        return torch.minimum(q - self.limits[:, 0], self.limits[:, 1] - q).amin(-1)
