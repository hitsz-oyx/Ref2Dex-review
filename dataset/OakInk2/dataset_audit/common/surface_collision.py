from __future__ import annotations

from pathlib import Path
from typing import Any

from .mesh_io import get_object_mesh, import_numpy, iter_query_meshes, load_cache
from .point_sdf import signed_distance_mm


def audit_surface_collision_cache(
    cache_path: Path,
    *,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
    unit_scale_to_mm: float = 1000.0,
    eps_mm: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    np = import_numpy()
    cache = load_cache(cache_path)
    query_meshes = iter_query_meshes(cache)
    object_verts, object_faces = get_object_mesh(cache)
    frame_count = min([object_verts.shape[0]] + [mesh[1].shape[0] for mesh in query_meshes]) if query_meshes else object_verts.shape[0]
    end_idx = frame_count if end is None else min(end, frame_count)
    indices = list(range(max(start, 0), end_idx, max(stride, 1)))
    rows: list[dict[str, Any]] = []
    skipped = 0
    frames_with_collision: set[int] = set()
    for frame_index in indices:
        object_frame = object_verts[frame_index]
        if object_faces is None:
            skipped += 1
            rows.append(_skip_row(frame_index, "object", "missing_faces_object"))
            continue
        for label, verts, faces in query_meshes:
            if faces is None:
                skipped += 1
                rows.append(_skip_row(frame_index, label, f"missing_faces_{label}"))
                continue
            query_frame = verts[frame_index]
            centers = query_frame[faces].mean(axis=1)
            signed = signed_distance_mm(
                object_frame,
                object_faces,
                centers,
                unit_scale_to_mm=unit_scale_to_mm,
                prefer_trimesh=True,
                allow_heuristic=True,
            )
            colliding_triangles = signed <= float(eps_mm)
            count = int(colliding_triangles.sum())
            if count:
                frames_with_collision.add(frame_index)
            rows.append(
                {
                    "frame_index": int(frame_index),
                    "query_label": label,
                    "method": "triangle_center_inside_proxy",
                    "num_query_triangles": int(len(faces)),
                    "collision_pairs": count,
                    "colliding_query_triangle_ratio": float(count / len(faces)) if len(faces) else 0.0,
                    "surface_collision_flag": bool(count > 0),
                    "note": "Uses triangle centers as a lightweight intersection proxy; no penetration depth is reported.",
                }
            )
    checked_rows = [row for row in rows if not str(row.get("method", "")).startswith("skipped")]
    checked_frames = len(indices)
    summary = {
        "status": "ok" if checked_rows else "skipped_missing_faces",
        "cache_path": str(cache_path),
        "dataset_name": str(cache.get("dataset_name", "")),
        "subject": str(cache.get("subject", "")),
        "sequence": str(cache.get("sequence", "")),
        "object_name": str(cache.get("object_name", "")),
        "num_frames_checked": int(checked_frames),
        "num_rows_checked": int(len(checked_rows)),
        "num_rows_skipped": int(skipped),
        "frames_with_surface_collision": int(len(frames_with_collision)),
        "surface_collision_frame_ratio": float(len(frames_with_collision) / checked_frames) if checked_frames else 0.0,
        "max_collision_pairs": int(max([int(row.get("collision_pairs", 0)) for row in checked_rows] or [0])),
        "max_colliding_query_triangle_ratio": float(
            max([float(row.get("colliding_query_triangle_ratio", 0.0)) for row in checked_rows] or [0.0])
        ),
        "does_not_report_penetration_depth": True,
        "limitations": [
            "Triangle-center proxy can miss edge-edge intersections.",
            "Coplanar touching surfaces and fitted visual meshes may be counted as collisions.",
            "Use point-SDF statistics and MeshCat review together for audit decisions.",
        ],
    }
    return rows, summary


def _skip_row(frame_index: int, label: str, reason: str) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
        "query_label": label,
        "method": "skipped_missing_faces",
        "num_query_triangles": 0,
        "collision_pairs": 0,
        "colliding_query_triangle_ratio": 0.0,
        "surface_collision_flag": False,
        "note": reason,
    }

