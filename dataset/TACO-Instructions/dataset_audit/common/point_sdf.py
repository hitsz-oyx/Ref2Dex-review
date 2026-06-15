from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .mesh_io import get_object_mesh, import_numpy, iter_query_meshes, load_cache
from .statistics import DEFAULT_THRESHOLDS, summarize_frame_metrics
from .topology import mesh_topology


def audit_point_sdf_cache(
    cache_path: Path,
    *,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
    surface_mode: str = "vertices",
    unit_scale_to_mm: float = 1000.0,
    thresholds: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np = import_numpy()
    cache = load_cache(cache_path)
    query_meshes = iter_query_meshes(cache)
    if not query_meshes:
        raise ValueError("cache has no verts_right, verts_left, verts_hand, or verts_body query mesh")
    object_verts, object_faces = get_object_mesh(cache)
    if object_faces is None:
        raise ValueError("cache has no faces_object; cannot query signed distance")
    frame_count = min([object_verts.shape[0]] + [mesh[1].shape[0] for mesh in query_meshes])
    end_idx = frame_count if end is None else min(end, frame_count)
    indices = list(range(max(start, 0), end_idx, max(stride, 1)))
    topo = mesh_topology(object_verts[0], object_faces)
    reliability = "normal" if topo["is_watertight"] else "heuristic"
    rows: list[dict[str, Any]] = []
    for frame_index in indices:
        object_frame = object_verts[frame_index]
        signed_chunks: list[Any] = []
        query_labels: list[str] = []
        for label, verts, faces in query_meshes:
            points = _surface_points(verts[frame_index], faces, surface_mode)
            if points.size == 0:
                continue
            signed = signed_distance_mm(
                object_frame,
                object_faces,
                points,
                unit_scale_to_mm=unit_scale_to_mm,
                prefer_trimesh=True,
                allow_heuristic=True,
            )
            signed_chunks.append(signed)
            query_labels.append(label)
        if signed_chunks:
            signed_all = np.concatenate(signed_chunks)
        else:
            signed_all = np.zeros((0,), dtype="float64")
        depths = np.maximum(0.0, -signed_all)
        negative = signed_all < 0.0
        inside_count = int(negative.sum())
        num_query = int(len(signed_all))
        rows.append(
            {
                "frame_index": int(frame_index),
                "query_labels": ",".join(query_labels),
                "num_query_points": num_query,
                "inside_count": inside_count,
                "inside_ratio": float(inside_count / num_query) if num_query else 0.0,
                "frame_min_signed_distance_mm": float(signed_all.min()) if num_query else 0.0,
                "frame_max_negative_depth_mm": float(depths.max()) if num_query else 0.0,
                "frame_mean_negative_depth_mm": float(depths[negative].mean()) if inside_count else 0.0,
                "reliability": reliability,
            }
        )
    summary = summarize_frame_metrics(rows, thresholds or DEFAULT_THRESHOLDS)
    summary.update(
        {
            "status": "ok",
            "cache_path": str(cache_path),
            "dataset_name": str(cache.get("dataset_name", "")),
            "subject": str(cache.get("subject", "")),
            "sequence": str(cache.get("sequence", "")),
            "object_name": str(cache.get("object_name", "")),
            "surface_mode": surface_mode,
            "stride": int(stride),
            "unit_scale_to_mm": float(unit_scale_to_mm),
            "object_topology": topo,
            "reliability": reliability,
            "signed_distance_convention": "negative means query point is inside object/collision proxy",
        }
    )
    return rows, summary


def _surface_points(vertices: Any, faces: Any | None, surface_mode: str) -> Any:
    np = import_numpy()
    chunks = []
    if surface_mode in ("vertices", "vertices_and_face_centers"):
        chunks.append(np.asarray(vertices, dtype="float64"))
    if surface_mode in ("face_centers", "vertices_and_face_centers"):
        if faces is None:
            warnings.warn("face centers requested but query faces are missing; using vertices only", RuntimeWarning)
            if not chunks:
                chunks.append(np.asarray(vertices, dtype="float64"))
        else:
            tri = vertices[faces]
            chunks.append(tri.mean(axis=1))
    if not chunks:
        raise ValueError(f"unsupported surface_mode: {surface_mode}")
    return np.concatenate(chunks, axis=0)


def signed_distance_mm(
    mesh_vertices: Any,
    mesh_faces: Any,
    points: Any,
    *,
    unit_scale_to_mm: float = 1000.0,
    prefer_trimesh: bool = True,
    allow_heuristic: bool = True,
) -> Any:
    np = import_numpy()
    vertices = np.asarray(mesh_vertices, dtype="float64")
    faces = np.asarray(mesh_faces, dtype="int64")
    points = np.asarray(points, dtype="float64")
    if prefer_trimesh:
        try:
            return _signed_distance_trimesh(vertices, faces, points, unit_scale_to_mm)
        except Exception:
            pass
    distances = _closest_distances_to_triangles(vertices, faces, points)
    inside = _points_inside_mesh(vertices, faces, points)
    if not allow_heuristic:
        inside = np.zeros_like(inside, dtype=bool)
    signed = distances * float(unit_scale_to_mm)
    signed[inside] *= -1.0
    return signed


def _signed_distance_trimesh(vertices: Any, faces: Any, points: Any, unit_scale_to_mm: float) -> Any:
    np = import_numpy()
    import trimesh  # type: ignore

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    raw = trimesh.proximity.signed_distance(mesh, points)
    # Trimesh convention is positive inside and negative outside. Audit convention is opposite.
    return -np.asarray(raw, dtype="float64") * float(unit_scale_to_mm)


def _closest_distances_to_triangles(vertices: Any, faces: Any, points: Any) -> Any:
    np = import_numpy()
    triangles = vertices[faces]
    out = np.empty((len(points),), dtype="float64")
    for idx, point in enumerate(points):
        out[idx] = float(np.sqrt(_point_triangle_min_distance_sq(np, point, triangles).min()))
    return out


def _point_triangle_min_distance_sq(np: Any, point: Any, triangles: Any) -> Any:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    dist = np.full((len(triangles),), np.inf, dtype="float64")

    mask = (d1 <= 0.0) & (d2 <= 0.0)
    dist[mask] = np.einsum("ij,ij->i", ap[mask], ap[mask])

    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    mask = (d3 >= 0.0) & (d4 <= d3)
    dist[mask] = np.einsum("ij,ij->i", bp[mask], bp[mask])

    vc = d1 * d4 - d3 * d2
    mask = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denom = d1 - d3
    valid = mask & (np.abs(denom) > 1e-12)
    v = np.zeros_like(d1)
    v[valid] = d1[valid] / denom[valid]
    proj = a + v[:, None] * ab
    diff = point - proj
    dist[valid] = np.einsum("ij,ij->i", diff[valid], diff[valid])

    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    mask = (d6 >= 0.0) & (d5 <= d6)
    dist[mask] = np.einsum("ij,ij->i", cp[mask], cp[mask])

    vb = d5 * d2 - d1 * d6
    mask = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denom = d2 - d6
    valid = mask & (np.abs(denom) > 1e-12)
    w = np.zeros_like(d2)
    w[valid] = d2[valid] / denom[valid]
    proj = a + w[:, None] * ac
    diff = point - proj
    dist[valid] = np.einsum("ij,ij->i", diff[valid], diff[valid])

    va = d3 * d6 - d5 * d4
    mask = (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    denom = (d4 - d3) + (d5 - d6)
    valid = mask & (np.abs(denom) > 1e-12)
    w = np.zeros_like(d3)
    w[valid] = (d4[valid] - d3[valid]) / denom[valid]
    proj = b + w[:, None] * (c - b)
    diff = point - proj
    dist[valid] = np.einsum("ij,ij->i", diff[valid], diff[valid])

    valid = np.isinf(dist)
    denom = va + vb + vc
    good = valid & (np.abs(denom) > 1e-12)
    v = np.zeros_like(denom)
    w = np.zeros_like(denom)
    v[good] = vb[good] / denom[good]
    w[good] = vc[good] / denom[good]
    proj = a + ab * v[:, None] + ac * w[:, None]
    diff = point - proj
    dist[good] = np.einsum("ij,ij->i", diff[good], diff[good])
    return dist


def _points_inside_mesh(vertices: Any, faces: Any, points: Any) -> Any:
    np = import_numpy()
    triangles = vertices[faces]
    direction = np.asarray([1.0, 0.173, 0.317], dtype="float64")
    direction = direction / np.linalg.norm(direction)
    inside = np.zeros((len(points),), dtype=bool)
    eps = 1e-10
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - triangles[:, 0]
    e2 = triangles[:, 2] - triangles[:, 0]
    h = np.cross(np.repeat(direction[None, :], len(triangles), axis=0), e2)
    a = np.einsum("ij,ij->i", e1, h)
    valid_a = np.abs(a) > eps
    for idx, point in enumerate(points):
        f = np.zeros_like(a)
        f[valid_a] = 1.0 / a[valid_a]
        s = point - v0
        u = f * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = f * np.einsum("j,ij->i", direction, q)
        t = f * np.einsum("ij,ij->i", e2, q)
        hits = valid_a & (u >= -eps) & (v >= -eps) & ((u + v) <= 1.0 + eps) & (t > eps)
        # A ray can hit the two triangles of one rectangular face at the same
        # distance; count unique intersection depths instead of raw triangles.
        unique_depths = np.unique(np.round(t[hits], decimals=10))
        inside[idx] = bool(int(len(unique_depths)) % 2 == 1)
    return inside
