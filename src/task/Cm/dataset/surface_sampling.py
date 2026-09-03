"""Deterministic random surface sampling for the single-frame Cm cache.

The sampler returns one surface specification (face ids and barycentric
coordinates).  A transition evaluates that same specification on both the
current and future meshes, preserving point-wise flow correspondence.
"""
from __future__ import annotations

import numpy as np


def sample_surface_spec(vertices: np.ndarray, faces: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("surface mesh must have vertices [V,3] and faces [F,3]")
    if count <= 0 or len(faces) == 0:
        raise ValueError("surface mesh must contain faces and count must be positive")
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=-1)
    total = float(areas.sum())
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError("surface mesh has no positive-area faces")
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    face = rng.choice(len(faces), size=int(count), replace=True, p=areas / total).astype(np.int64)
    uv = rng.random((int(count), 2), dtype=np.float32)
    over = (uv[:, 0] + uv[:, 1]) > 1.0
    uv[over] = 1.0 - uv[over]
    bary = np.concatenate((1.0 - uv.sum(axis=1, keepdims=True), uv), axis=1).astype(np.float32)
    return face, bary


def evaluate_surface(vertices: np.ndarray, faces: np.ndarray, face: np.ndarray, bary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int64)[np.asarray(face, dtype=np.int64)]]
    bary = np.asarray(bary, dtype=np.float32)
    points = (triangles * bary[:, :, None]).sum(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return points.astype(np.float32), normals.astype(np.float32)
