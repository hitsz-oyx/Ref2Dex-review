from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def import_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: numpy. Install it inside the repository Docker/"
            "Miniconda environment; do not install dependencies on the macOS host."
        ) from exc
    return np


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_parent(path)
    if fieldnames is None:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        fieldnames = fields or ["message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return value


def repo_root_from_script(script_file: str) -> Path:
    return Path(script_file).resolve().parents[2]


def decode_np_scalar(value: Any) -> Any:
    np = import_numpy()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_np_scalar(value.item())
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_cache(path: Path) -> dict[str, Any]:
    np = import_numpy()
    if not path.exists():
        raise FileNotFoundError(f"cache not found: {path}")
    out: dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as cache:
        for key in cache.files:
            out[key] = decode_np_scalar(cache[key])
    return out


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    np = import_numpy()
    ensure_parent(path)
    serializable: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, (str, bool, int, float)):
            serializable[key] = np.array(value)
        else:
            serializable[key] = value
    np.savez_compressed(path, **serializable)


def as_float_vertices(value: Any, name: str) -> Any:
    np = import_numpy()
    arr = np.asarray(value, dtype="float64")
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [T, N, 3], got {arr.shape}")
    return arr


def as_faces(value: Any, name: str) -> Any:
    np = import_numpy()
    arr = np.asarray(value, dtype="int64")
    if arr.ndim != 2 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [F, 3], got {arr.shape}")
    return arr


def find_first_cache_mesh(cache: dict[str, Any], names: list[str]) -> tuple[str, Any, Any | None] | None:
    for name in names:
        verts_key = f"verts_{name}"
        if verts_key in cache:
            faces = cache.get(f"faces_{name}")
            return name, as_float_vertices(cache[verts_key], verts_key), None if faces is None else as_faces(faces, f"faces_{name}")
    return None


def iter_query_meshes(cache: dict[str, Any]) -> list[tuple[str, Any, Any | None]]:
    meshes: list[tuple[str, Any, Any | None]] = []
    for name in ("right", "left", "hand", "body"):
        verts_key = f"verts_{name}"
        if verts_key in cache:
            faces_key = f"faces_{name}"
            faces = as_faces(cache[faces_key], faces_key) if faces_key in cache else None
            meshes.append((name, as_float_vertices(cache[verts_key], verts_key), faces))
    return meshes


def get_object_mesh(cache: dict[str, Any]) -> tuple[Any, Any | None]:
    verts = as_float_vertices(cache["verts_object"], "verts_object")
    faces = as_faces(cache["faces_object"], "faces_object") if "faces_object" in cache else None
    return verts, faces


def load_obj_mesh(path: Path) -> tuple[Any, Any]:
    np = import_numpy()
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                raw = line.strip().split()[1:]
                idx: list[int] = []
                for token in raw:
                    head = token.split("/")[0]
                    if not head:
                        continue
                    value = int(head)
                    idx.append(value - 1 if value > 0 else len(vertices) + value)
                if len(idx) >= 3:
                    for i in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[i], idx[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"OBJ file has no usable vertices/faces: {path}")
    return np.asarray(vertices, dtype="float64"), np.asarray(faces, dtype="int64")


def mesh_entries_for_topology(cache: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    entries: list[tuple[str, Any, Any]] = []
    for name in ("object", "right", "left", "hand", "body"):
        verts_key = f"verts_{name}"
        faces_key = f"faces_{name}"
        if verts_key in cache and faces_key in cache:
            verts = as_float_vertices(cache[verts_key], verts_key)
            faces = as_faces(cache[faces_key], faces_key)
            entries.append((name, verts[0], faces))
    return entries


def create_toy_cache(path: Path, frames: int = 5) -> dict[str, Any]:
    np = import_numpy()
    cube_v = np.asarray(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype="float64",
    )
    cube_f = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype="int64",
    )
    object_scale = 0.06
    hand_scale = 0.025
    offsets = np.asarray(
        [
            [-0.11, 0.0, 0.0],
            [-0.05, 0.0, 0.0],
            [-0.015, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
        ],
        dtype="float64",
    )
    offsets = offsets[:frames]
    verts_object = np.repeat((cube_v * object_scale)[None, :, :], len(offsets), axis=0)
    verts_hand = np.stack([(cube_v * hand_scale) + offset for offset in offsets], axis=0)
    payload = {
        "verts_hand": verts_hand,
        "faces_hand": cube_f,
        "verts_object": verts_object,
        "faces_object": cube_f,
        "object_name": "toy_cube",
        "subject": "toy_subject",
        "sequence": "toy_hand_object",
        "dataset_name": "toy",
        "frame_indices": np.arange(len(offsets), dtype="int64"),
        "source_paths": np.asarray(["synthetic_toy_cache"], dtype=object),
        "contact_available": False,
        "contact_source": "none; synthetic cache for smoke tests",
    }
    save_cache(path, payload)
    return payload


def candidate_data_roots(repo_root: Path, env_names: list[str], relative_names: list[str]) -> list[Path]:
    roots: list[Path] = []
    for name in env_names:
        value = os.environ.get(name)
        if value:
            roots.append(Path(value).expanduser())
    for rel in relative_names:
        roots.append(repo_root / rel)
    deduped: list[Path] = []
    for path in roots:
        resolved = path.resolve()
        if resolved not in deduped:
            deduped.append(resolved)
    return deduped

