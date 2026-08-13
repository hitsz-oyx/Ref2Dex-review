"""基于封闭 object mesh 的解析穿透诊断。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PenetrationResult:
    p_anchor: np.ndarray | None
    max_penetration_mm: float | None
    mean_penetration_mm: float | None
    penetrating_point_ratio: float | None
    valid: bool


def _watertight(faces: np.ndarray) -> bool:
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                    faces[:, [2, 0]]]), axis=1)
    return bool(np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2))


def _point_triangle_distance(points: np.ndarray, triangles: np.ndarray,
                             point_chunk: int = 128, face_chunk: int = 2048) -> np.ndarray:
    """Ericson closest-point regions 的向量化实现。"""
    result = np.full(len(points), np.inf, np.float64)
    for start in range(0, len(points), point_chunk):
      p = points[start:start + point_chunk, None]
      local_result = np.full(len(p), np.inf, np.float64)
      for face_start in range(0, len(triangles), face_chunk):
        tri = triangles[face_start:face_start + face_chunk]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, ac, bc = b - a, c - a, c - b
        ap, bp, cp = p - a, p - b, p - c
        d1, d2 = np.sum(ab * ap, -1), np.sum(ac * ap, -1)
        d3, d4 = np.sum(ab * bp, -1), np.sum(ac * bp, -1)
        d5, d6 = np.sum(ab * cp, -1), np.sum(ac * cp, -1)
        shape = (len(p), len(tri))
        closest = np.empty((*shape, 3), np.float64)
        assigned = np.zeros(shape, bool)

        def put(mask, value):
            nonlocal assigned
            mask &= ~assigned
            closest[mask] = np.broadcast_to(value, closest.shape)[mask]
            assigned |= mask

        put((d1 <= 0) & (d2 <= 0), a)
        put((d3 >= 0) & (d4 <= d3), b)
        vc = d1 * d4 - d3 * d2
        v = d1 / np.where(d1 - d3 == 0, 1, d1 - d3)
        put((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + v[..., None] * ab)
        put((d6 >= 0) & (d5 <= d6), c)
        vb = d5 * d2 - d1 * d6
        w = d2 / np.where(d2 - d6 == 0, 1, d2 - d6)
        put((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + w[..., None] * ac)
        va = d3 * d6 - d5 * d4
        w_bc = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1,
                                     (d4 - d3) + (d5 - d6))
        put((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
            b + w_bc[..., None] * bc)
        denom = np.where(va + vb + vc == 0, 1, va + vb + vc)
        v_face, w_face = vb / denom, vc / denom
        put(~assigned, a + v_face[..., None] * ab + w_face[..., None] * ac)
        local_result = np.minimum(local_result, np.linalg.norm(p - closest, axis=-1).min(1))
      result[start:start + len(p)] = local_result
    return result


def _inside_mesh(points: np.ndarray, triangles: np.ndarray, point_chunk: int = 128,
                 face_chunk: int = 4096) -> np.ndarray:
    """以固定非轴向射线做奇偶相交测试，避免依赖 rtree。"""
    inside = np.zeros(len(points), bool)
    direction = np.array([1., .37139068, .69474659])
    direction /= np.linalg.norm(direction)
    for start in range(0, len(points), point_chunk):
        origin = points[start:start + point_chunk]; hits = np.zeros(len(origin), np.int64)
        for face_start in range(0, len(triangles), face_chunk):
            tri = triangles[face_start:face_start + face_chunk]
            edge1, edge2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
            h = np.cross(direction, edge2); a = np.einsum("fi,fi->f", edge1, h)
            usable = np.abs(a) > 1e-10
            inverse = np.zeros_like(a); inverse[usable] = 1 / a[usable]
            s = origin[:, None] - tri[None, :, 0]
            u = np.einsum("pfi,fi->pf", s, h) * inverse
            q = np.cross(s, edge1[None])
            v = np.einsum("i,pfi->pf", direction, q) * inverse
            t = np.einsum("fi,pfi->pf", edge2, q) * inverse
            hit = usable[None] & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
            hits += hit.sum(1)
        inside[start:start + len(origin)] = (hits % 2) == 1
    return inside


def penetration_from_mesh(hand_points_object: np.ndarray, object_vertices: np.ndarray,
                          object_faces: np.ndarray, anchors: np.ndarray | None = None,
                          tau_m: float = .015) -> PenetrationResult:
    """约定 signed distance 外正内负；非封闭/退化 mesh 明确返回无效。"""
    points = np.asarray(hand_points_object, np.float64)
    vertices = np.asarray(object_vertices, np.float64)
    faces = np.asarray(object_faces, np.int64)
    if (points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or len(faces) == 0 or not _watertight(faces)):
        return PenetrationResult(None, None, None, None, False)
    triangles = vertices[faces]
    if np.any(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                      triangles[:, 2] - triangles[:, 0]), axis=1) < 1e-12):
        return PenetrationResult(None, None, None, None, False)
    inside = _inside_mesh(points, triangles)
    penetration = np.zeros(len(points), np.float64)
    if inside.any():
        penetration[inside] = _point_triangle_distance(points[inside], triangles)
    p_anchor = None
    if anchors is not None:
        squared = ((np.asarray(anchors)[:, None] - points[None]) ** 2).sum(-1)
        logits = -squared / tau_m ** 2
        weights = np.exp(logits - logits.max(1, keepdims=True))
        weights /= weights.sum(1, keepdims=True)
        p_anchor = (weights * penetration[None]).sum(1).astype(np.float32)
    penetrating = penetration > 1e-7
    return PenetrationResult(p_anchor, float(penetration.max(initial=0) * 1000),
                             float(penetration.mean() * 1000),
                             float(penetrating.mean()), True)
