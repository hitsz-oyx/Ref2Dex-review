from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .mesh_io import get_object_mesh, import_numpy, iter_query_meshes, load_cache


def headless_check(cache_path: Path) -> dict[str, Any]:
    cache = load_cache(cache_path)
    object_verts, object_faces = get_object_mesh(cache)
    query_meshes = iter_query_meshes(cache)
    return {
        "status": "ok",
        "cache_path": str(cache_path),
        "dataset_name": str(cache.get("dataset_name", "")),
        "subject": str(cache.get("subject", "")),
        "sequence": str(cache.get("sequence", "")),
        "object_name": str(cache.get("object_name", "")),
        "num_frames": int(object_verts.shape[0]),
        "object_vertices": int(object_verts.shape[1]),
        "object_faces": int(0 if object_faces is None else len(object_faces)),
        "query_meshes": [
            {"label": label, "vertices": int(verts.shape[1]), "faces": int(0 if faces is None else len(faces))}
            for label, verts, faces in query_meshes
        ],
        "contact_available": str(cache.get("contact_available", "")),
        "contact_source": str(cache.get("contact_source", "")),
    }


def play_meshcat(
    cache_path: Path,
    *,
    fps: float = 3.0,
    stride: int = 1,
    start: int = 0,
    end: int | None = None,
    loop: bool = False,
    show_centers: bool = False,
) -> None:
    np = import_numpy()
    try:
        import meshcat  # type: ignore
        import meshcat.geometry as g  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: meshcat. Install it inside Docker/Miniconda to use the browser viewer. "
            "Use --headless-check for cache validation without opening a browser."
        ) from exc
    cache = load_cache(cache_path)
    object_verts, object_faces = get_object_mesh(cache)
    if object_faces is None:
        raise SystemExit("cache has no faces_object; MeshCat viewer needs faces for object rendering")
    query_meshes = iter_query_meshes(cache)
    vis = meshcat.Visualizer().open()
    frame_count = min([object_verts.shape[0]] + [mesh[1].shape[0] for mesh in query_meshes]) if query_meshes else object_verts.shape[0]
    indices = list(range(max(start, 0), frame_count if end is None else min(end, frame_count), max(stride, 1)))
    delay = 1.0 / max(fps, 0.1)
    while True:
        for frame_index in indices:
            vis["object"].set_object(g.TriangularMeshGeometry(object_verts[frame_index], object_faces))
            for label, verts, faces in query_meshes:
                if faces is None:
                    points = verts[frame_index]
                    vis[label].set_object(g.PointCloud(points.T))
                else:
                    vis[label].set_object(g.TriangularMeshGeometry(verts[frame_index], faces))
            if show_centers:
                centers = [object_verts[frame_index].mean(axis=0)]
                centers.extend([verts[frame_index].mean(axis=0) for _, verts, _ in query_meshes])
                vis["centers"].set_object(g.PointCloud(np.asarray(centers).T))
            time.sleep(delay)
        if not loop:
            break

