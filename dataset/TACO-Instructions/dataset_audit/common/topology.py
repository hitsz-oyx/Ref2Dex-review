from __future__ import annotations

from collections import Counter, defaultdict, deque
from typing import Any

from .mesh_io import import_numpy


def mesh_topology(vertices: Any, faces: Any) -> dict[str, Any]:
    np = import_numpy()
    vertices = np.asarray(vertices, dtype="float64")
    faces = np.asarray(faces, dtype="int64")
    edges: list[tuple[int, int]] = []
    for tri in faces:
        a, b, c = [int(x) for x in tri]
        edges.extend([tuple(sorted((a, b))), tuple(sorted((b, c))), tuple(sorted((c, a)))])
    edge_counts = Counter(edges)
    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    components = _connected_components(faces)
    area = _surface_area(np, vertices, faces)
    volume = _signed_volume(np, vertices, faces)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    is_watertight = boundary_edges == 0 and nonmanifold_edges == 0
    reliability = "normal" if is_watertight else "non_watertight_visual_mesh"
    return {
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "num_edges": int(len(edge_counts)),
        "is_watertight": bool(is_watertight),
        "num_boundary_edges": int(boundary_edges),
        "num_nonmanifold_edges": int(nonmanifold_edges),
        "num_connected_components": int(components),
        "euler_number": int(len(vertices) - len(edge_counts) + len(faces)),
        "surface_area": float(area),
        "signed_volume": float(volume),
        "abs_volume": float(abs(volume)),
        "bbox_min": [float(x) for x in bbox_min],
        "bbox_max": [float(x) for x in bbox_max],
        "bbox_extent": [float(x) for x in (bbox_max - bbox_min)],
        "reliability": reliability,
    }


def _connected_components(faces: Any) -> int:
    vertex_to_faces: dict[int, list[int]] = defaultdict(list)
    for face_idx, tri in enumerate(faces):
        for vertex_idx in tri:
            vertex_to_faces[int(vertex_idx)].append(face_idx)
    seen: set[int] = set()
    components = 0
    for start in range(len(faces)):
        if start in seen:
            continue
        components += 1
        queue: deque[int] = deque([start])
        seen.add(start)
        while queue:
            face_idx = queue.popleft()
            for vertex_idx in faces[face_idx]:
                for neighbor in vertex_to_faces[int(vertex_idx)]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
    return components


def _surface_area(np: Any, vertices: Any, faces: Any) -> float:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(0.5 * np.linalg.norm(cross, axis=1).sum())


def _signed_volume(np: Any, vertices: Any, faces: Any) -> float:
    tri = vertices[faces]
    return float((np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0).sum())

