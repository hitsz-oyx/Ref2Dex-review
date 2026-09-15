"""Deterministic surface geometry adapter for the frozen ObjectInteractionCm model."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch


def _origin(element):
    xyz = np.fromstring((element.get("xyz", "0 0 0") if element is not None else "0 0 0"), sep=" ", dtype=np.float32)
    rpy = np.fromstring((element.get("rpy", "0 0 0") if element is not None else "0 0 0"), sep=" ", dtype=np.float32)
    rx, ry, rz = rpy
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    rot = np.array([[cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
                    [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
                    [-sy, cy * sx, cy * cx]], dtype=np.float32)
    out = np.eye(4, dtype=np.float32); out[:3, :3] = rot; out[:3, 3] = xyz
    return out


class SurfaceGeometry:
    """Sample URDF visual surfaces once and transform them from live link poses."""

    def __init__(self, *, hand_urdf: str | Path, object_urdf: str | Path,
                 query_links: tuple[str, ...], object_count: int = 1024,
                 hand_count: int = 1538, seed: int = 42, device="cpu"):
        try:
            import trimesh
        except ImportError as exc:  # pragma: no cover - environment dependency
            raise RuntimeError("CmResidual geometry requires trimesh") from exc
        self.device = torch.device(device)
        self.query_links = tuple(query_links)
        self.object_local, self.object_normal = self._sample(trimesh, Path(object_urdf), object_count, seed)
        self.hand_local, self.hand_normal, self.hand_link = self._sample_hand(
            trimesh, Path(hand_urdf), hand_count, seed + 1)
        self.prev_hand = None

    @staticmethod
    def _mesh_path(urdf: Path, filename: str) -> Path:
        path = Path(filename)
        if path.is_absolute():
            return path
        if filename.startswith("package://"):
            path = Path(filename.split("package://", 1)[1].split("/", 1)[1])
        return (urdf.parent / path).resolve()

    def _visuals(self, trimesh, urdf: Path):
        root = ET.parse(urdf).getroot()
        result = []
        for link in root.findall("link"):
            name = link.get("name")
            for visual in link.findall("visual"):
                mesh = visual.find("./geometry/mesh")
                if mesh is None or mesh.get("filename") is None:
                    continue
                loaded = trimesh.load(str(self._mesh_path(urdf, mesh.get("filename"))), force="mesh", process=False)
                if not isinstance(loaded, trimesh.Trimesh):
                    continue
                scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ", dtype=np.float32)
                if scale.size == 1: scale = np.repeat(scale, 3)
                result.append((str(name), loaded, _origin(visual.find("origin")), scale))
        if not result:
            raise ValueError(f"No visual mesh found in {urdf}")
        return result

    def _sample(self, trimesh, urdf: Path, count: int, seed: int):
        visuals = self._visuals(trimesh, urdf)
        points, normals, areas = [], [], []
        for _, mesh, transform, scale in visuals:
            vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale[None, :]
            tri = vertices[np.asarray(mesh.faces, dtype=np.int64)]
            cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = 0.5 * np.linalg.norm(cross, axis=1)
            valid = area > 1e-10
            n = cross[valid] / np.clip(np.linalg.norm(cross[valid], axis=1, keepdims=True), 1e-8, None)
            tri = tri[valid]
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(tri), size=max(1, count // max(1, len(visuals))), replace=True, p=area[valid] / area[valid].sum())
            r1, r2 = np.sqrt(rng.random(len(idx))), rng.random(len(idx))
            p = tri[idx, 0] + r1[:, None] * (tri[idx, 1] - tri[idx, 0]) + r2[:, None] * (tri[idx, 2] - tri[idx, 0])
            p = p @ transform[:3, :3].T + transform[:3, 3]
            n = n[idx] @ transform[:3, :3].T
            points.append(p); normals.append(n); areas.append(np.ones(len(p)))
        points, normals = np.concatenate(points), np.concatenate(normals)
        rng = np.random.default_rng(seed + 17)
        idx = rng.choice(len(points), size=count, replace=len(points) < count)
        return points[idx].astype(np.float32), normals[idx].astype(np.float32)

    def _sample_hand(self, trimesh, urdf: Path, count: int, seed: int):
        visuals = self._visuals(trimesh, urdf)
        chunks = []
        per = max(1, count // len(visuals))
        for i, (name, mesh, transform, scale) in enumerate(visuals):
            p, n = self._sample_visual(mesh, transform, scale, per, seed + i)
            if name in self.query_links:
                chunks.append((p, n, self.query_links.index(name)))
        if not chunks:
            raise ValueError("Hand URDF visual links do not overlap QUERY_LINKS")
        points = np.concatenate([x[0] for x in chunks]); normals = np.concatenate([x[1] for x in chunks])
        links = np.concatenate([np.full(len(x[0]), x[2], dtype=np.int64) for x in chunks])
        rng = np.random.default_rng(seed + 17); idx = rng.choice(len(points), size=count, replace=len(points) < count)
        return points[idx].astype(np.float32), normals[idx].astype(np.float32), links[idx]

    @staticmethod
    def _sample_visual(mesh, transform, scale, count, seed):
        vertices = np.asarray(mesh.vertices, dtype=np.float32) * scale[None, :]
        tri = vertices[np.asarray(mesh.faces, dtype=np.int64)]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); area = 0.5 * np.linalg.norm(cross, axis=1)
        valid = area > 1e-10; tri, cross, area = tri[valid], cross[valid], area[valid]
        rng = np.random.default_rng(seed); idx = rng.choice(len(tri), size=count, replace=True, p=area / area.sum())
        r1, r2 = np.sqrt(rng.random(count)), rng.random(count)
        p = tri[idx, 0] + r1[:, None] * (tri[idx, 1] - tri[idx, 0]) + r2[:, None] * (tri[idx, 2] - tri[idx, 0])
        n = cross[idx] / np.clip(np.linalg.norm(cross[idx], axis=1, keepdims=True), 1e-8, None)
        return p @ transform[:3, :3].T + transform[:3, 3], n @ transform[:3, :3].T

    def _transform(self, local, normal, transforms):
        # local/normal: [N,3], transforms: [B,L,4,4]
        device, dtype = transforms.device, transforms.dtype
        local = torch.as_tensor(local, device=device, dtype=dtype); normal = torch.as_tensor(normal, device=device, dtype=dtype)
        rotations = transforms[:, self.hand_link, :3, :3]
        translations = transforms[:, self.hand_link, :3, 3]
        points = torch.matmul(local[None, :, None, :], rotations.transpose(-1, -2)).squeeze(-2) + translations
        normals = torch.matmul(normal[None, :, None, :], rotations.transpose(-1, -2)).squeeze(-2)
        return points, torch.nn.functional.normalize(normals, dim=-1)

    def hand(self, link_poses):
        return self._transform(self.hand_local, self.hand_normal, link_poses)

    def object(self, object_pose):
        local = torch.as_tensor(self.object_local, device=object_pose.device, dtype=object_pose.dtype)
        normal = torch.as_tensor(self.object_normal, device=object_pose.device, dtype=object_pose.dtype)
        points = torch.matmul(local[None, :, None, :], object_pose[:, None, :3, :3].transpose(-1, -2)).squeeze(-2) + object_pose[:, None, :3, 3]
        normals = torch.matmul(normal[None, :, None, :], object_pose[:, None, :3, :3].transpose(-1, -2)).squeeze(-2)
        return points, torch.nn.functional.normalize(normals, dim=-1)

    def flow(self, points, reset_mask=None):
        flow = torch.zeros_like(points) if self.prev_hand is None else points - self.prev_hand
        if reset_mask is not None and reset_mask.any(): flow[reset_mask] = 0
        self.prev_hand = points.detach().clone()
        return flow

    def contact_targets(self, link_poses, object_pose, radius=0.02):
        points, _ = self.object(object_pose)
        contact_links = (7, 10, 16, 13, 4)
        centers = link_poses[:, contact_links, :3, 3]
        distances = torch.cdist(centers, points)
        return (distances.amin(-1) <= radius).to(points.dtype)
