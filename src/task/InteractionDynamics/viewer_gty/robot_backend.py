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


@dataclass(frozen=True)
class SurfaceSamples:
    visual_index: torch.Tensor
    face_index: torch.Tensor
    barycentric: torch.Tensor


class UrdfHandBackend:
    """实现 V20.6 所需的 fixed/revolute joint、完整 mesh 与固定表面采样。"""

    def __init__(self, urdf: str | Path, device: torch.device | str = "cpu",
                 surface_sample_count: int = 1538, surface_sample_seed: int = 0) -> None:
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
        self.visual_triangle_count = len(self.faces)
        self.surface_sample_count = surface_sample_count; self.surface_sample_seed = surface_sample_seed
        areas=[]; owners=[]; local_faces=[]
        for visual_index, visual in enumerate(visuals):
            face = torch.as_tensor(visual.faces, dtype=torch.long, device=self.device)
            triangle = visual.vertices[face]
            areas.append(torch.linalg.cross(triangle[:, 1]-triangle[:, 0],
                                            triangle[:, 2]-triangle[:, 0]).norm(dim=-1)/2)
            owners.append(torch.full((len(face),), visual_index, dtype=torch.long, device=self.device))
            local_faces.append(torch.arange(len(face), dtype=torch.long, device=self.device))
        area = torch.cat(areas).cpu(); generator = torch.Generator(device="cpu")
        generator.manual_seed(surface_sample_seed)
        selected = torch.multinomial(area / area.sum(), surface_sample_count, replacement=True,
                                     generator=generator)
        u = torch.rand(surface_sample_count, generator=generator); v = torch.rand(surface_sample_count,
                                                                                   generator=generator)
        sqrt_u = u.sqrt(); barycentric = torch.stack((1-sqrt_u, sqrt_u*(1-v), sqrt_u*v), -1)
        self.surface_samples = SurfaceSamples(torch.cat(owners).cpu()[selected].to(self.device),
            torch.cat(local_faces).cpu()[selected].to(self.device), barycentric.to(self.device))

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    def _transforms(self, q: torch.Tensor, root_rotation: torch.Tensor,
                    root_translation: torch.Tensor) -> dict[str, torch.Tensor]:
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
        return transforms

    def vertices(self, q: torch.Tensor, root_rotation: torch.Tensor,
                 root_translation: torch.Tensor) -> torch.Tensor:
        """返回用于 viewer/penetration 的完整 visual mesh `[T,V,3]`。"""
        transforms = self._transforms(q, root_rotation, root_translation); outputs = []
        for visual in self.visuals:
            transform = transforms[visual.link] @ visual.origin[None]
            outputs.append(visual.vertices[None] @ transform[:, :3, :3].transpose(-1, -2)
                               + transform[:, None, :3, 3])
        return torch.cat(outputs, 1)

    def surface_points(self, q: torch.Tensor, root_rotation: torch.Tensor,
                       root_translation: torch.Tensor) -> torch.Tensor:
        """返回固定 triangle+barycentric 对应的面积均匀表面点 `[T,1538,3]`。"""
        transforms = self._transforms(q, root_rotation, root_translation)
        output = torch.empty((len(q), self.surface_sample_count, 3), device=q.device, dtype=q.dtype)
        for visual_index, visual in enumerate(self.visuals):
            mask = self.surface_samples.visual_index == visual_index
            if not mask.any():
                continue
            face = torch.as_tensor(visual.faces, dtype=torch.long, device=q.device)
            triangles = visual.vertices[face[self.surface_samples.face_index[mask]]]
            local = (triangles * self.surface_samples.barycentric[mask, :, None]).sum(1)
            transform = transforms[visual.link] @ visual.origin[None]
            output[:, mask] = (local[None] @ transform[:, :3, :3].transpose(-1, -2)
                               + transform[:, None, :3, 3])
        return output

    def joint_margin(self, q: torch.Tensor) -> torch.Tensor:
        return torch.minimum(q - self.limits[:, 0], self.limits[:, 1] - q).amin(-1)
