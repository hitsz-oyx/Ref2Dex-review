#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import trimesh


OBJECT_SDF_GRID_CACHE_VERSION = 1


@dataclass
class ObjectSDFGrid:
    grid_zyx: torch.Tensor
    bounds_min: torch.Tensor
    bounds_max: torch.Tensor
    device: torch.device

    @classmethod
    def from_numpy(
        cls,
        sdf_grid_xyz: np.ndarray,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        device: torch.device,
    ) -> "ObjectSDFGrid":
        grid_zyx = torch.from_numpy(
            np.asarray(sdf_grid_xyz, dtype=np.float32).transpose(2, 1, 0)
        ).to(device=device, dtype=torch.float32)
        return cls(
            grid_zyx=grid_zyx.unsqueeze(0).unsqueeze(0).contiguous(),
            bounds_min=torch.as_tensor(bounds_min, dtype=torch.float32, device=device),
            bounds_max=torch.as_tensor(bounds_max, dtype=torch.float32, device=device),
            device=device,
        )

    def query_sdf(self, points_obj: torch.Tensor) -> torch.Tensor:
        orig_shape = points_obj.shape[:-1]
        points_flat = points_obj.to(device=self.device, dtype=torch.float32).reshape(-1, 3)
        denom = (self.bounds_max - self.bounds_min).clamp_min(1e-6)
        coords = 2.0 * (points_flat - self.bounds_min) / denom - 1.0
        sample_grid = coords.view(1, -1, 1, 1, 3)
        sdf = F.grid_sample(
            self.grid_zyx,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sdf.view(-1).reshape(orig_shape)

    def penetration_loss(
        self,
        points_obj: torch.Tensor,
        penetration_tol_m: float,
        penetration_scale_m: float,
    ) -> torch.Tensor:
        sdf = self.query_sdf(points_obj)
        penetration = torch.relu(-(sdf + float(penetration_tol_m)))
        active = penetration > 0.0
        if not bool(active.any()):
            return torch.zeros((), dtype=sdf.dtype, device=sdf.device)
        scale = max(float(penetration_scale_m), 1e-6)
        return torch.mean(torch.square(penetration[active] / scale))


def default_object_sdf_cache_root(repo_root: Path) -> Path:
    return repo_root / "outputs" / f"_object_sdf_grid_cache_v{OBJECT_SDF_GRID_CACHE_VERSION}"


def resolve_object_sdf_cache_path(
    cache_root: Path,
    dataset_name: str,
    object_name: str,
    resolution: int,
    padding_m: float,
) -> Path:
    padding_mm = int(round(float(padding_m) * 1000.0))
    return cache_root / dataset_name / f"{object_name}_res{int(resolution)}_pad{padding_mm:03d}mm.npz"


def _is_cache_valid(
    cache_data: np.lib.npyio.NpzFile,
    mesh_path: Path,
    resolution: int,
    padding_m: float,
) -> bool:
    try:
        cache_version = int(np.asarray(cache_data["cache_version"]).item())
        cached_mesh_path = str(np.asarray(cache_data["mesh_path"]).item())
        cached_mesh_mtime_ns = int(np.asarray(cache_data["mesh_mtime_ns"]).item())
        cached_resolution = int(np.asarray(cache_data["resolution"]).item())
        cached_padding_m = float(np.asarray(cache_data["padding_m"]).item())
    except Exception:
        return False
    if cache_version != OBJECT_SDF_GRID_CACHE_VERSION:
        return False
    if cached_mesh_path != str(mesh_path.resolve()):
        return False
    if cached_resolution != int(resolution):
        return False
    if abs(cached_padding_m - float(padding_m)) > 1e-8:
        return False
    try:
        stat = mesh_path.stat()
    except FileNotFoundError:
        return False
    return cached_mesh_mtime_ns == int(stat.st_mtime_ns)


def load_mesh_trimesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected Trimesh from {mesh_path}, got {type(mesh)!r}")
    return mesh


def build_object_sdf_grid_np(
    mesh_path: Path,
    resolution: int,
    padding_m: float,
    chunk_size: int = 262144,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh = load_mesh_trimesh(mesh_path)
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    bounds_min = verts.min(axis=0) - float(padding_m)
    bounds_max = verts.max(axis=0) + float(padding_m)

    x_lin = np.linspace(bounds_min[0], bounds_max[0], int(resolution), dtype=np.float32)
    y_lin = np.linspace(bounds_min[1], bounds_max[1], int(resolution), dtype=np.float32)
    z_lin = np.linspace(bounds_min[2], bounds_max[2], int(resolution), dtype=np.float32)
    xx, yy, zz = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)

    mesh_legacy = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.astype(np.float64)),
        o3d.utility.Vector3iVector(faces),
    )
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_t)

    sdf_chunks = []
    num_points = int(points.shape[0])
    for start in range(0, num_points, int(chunk_size)):
        end = min(start + int(chunk_size), num_points)
        if progress:
            print(f"[sdf-grid] build {mesh_path.name} points {start}:{end}/{num_points}")
        sdf_chunk = scene.compute_signed_distance(
            o3d.core.Tensor(points[start:end], dtype=o3d.core.Dtype.Float32)
        ).numpy()
        sdf_chunks.append(np.asarray(sdf_chunk, dtype=np.float32))
    sdf_grid_xyz = np.concatenate(sdf_chunks, axis=0).reshape(int(resolution), int(resolution), int(resolution))
    return sdf_grid_xyz.astype(np.float32), bounds_min.astype(np.float32), bounds_max.astype(np.float32)


def load_or_build_object_sdf_grid(
    mesh_path: Path,
    dataset_name: str,
    object_name: str,
    device: torch.device,
    cache_root: Path,
    resolution: int,
    padding_m: float,
    rebuild: bool = False,
    progress: bool = False,
) -> ObjectSDFGrid:
    cache_path = resolve_object_sdf_cache_path(
        cache_root=cache_root,
        dataset_name=dataset_name,
        object_name=object_name,
        resolution=resolution,
        padding_m=padding_m,
    )
    if (not rebuild) and cache_path.exists():
        with np.load(str(cache_path), allow_pickle=False) as cache_data:
            if _is_cache_valid(cache_data, mesh_path, resolution, padding_m):
                return ObjectSDFGrid.from_numpy(
                    sdf_grid_xyz=np.asarray(cache_data["sdf_grid_xyz"], dtype=np.float32),
                    bounds_min=np.asarray(cache_data["bounds_min"], dtype=np.float32),
                    bounds_max=np.asarray(cache_data["bounds_max"], dtype=np.float32),
                    device=device,
                )
        print(f"[sdf-grid] stale {cache_path}, rebuilding")

    sdf_grid_xyz, bounds_min, bounds_max = build_object_sdf_grid_np(
        mesh_path=mesh_path,
        resolution=resolution,
        padding_m=padding_m,
        progress=progress,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        cache_version=np.asarray(OBJECT_SDF_GRID_CACHE_VERSION, dtype=np.int32),
        mesh_path=np.asarray(str(mesh_path.resolve())),
        mesh_mtime_ns=np.asarray(mesh_path.stat().st_mtime_ns, dtype=np.int64),
        resolution=np.asarray(int(resolution), dtype=np.int32),
        padding_m=np.asarray(float(padding_m), dtype=np.float32),
        bounds_min=bounds_min.astype(np.float32),
        bounds_max=bounds_max.astype(np.float32),
        sdf_grid_xyz=sdf_grid_xyz.astype(np.float32),
    )
    return ObjectSDFGrid.from_numpy(
        sdf_grid_xyz=sdf_grid_xyz,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        device=device,
    )
