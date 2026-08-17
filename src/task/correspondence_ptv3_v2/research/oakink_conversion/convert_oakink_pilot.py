"""Convert a small OakInk annotation subset to compact correspondence Stage 3.

OakInk's released annotations are per-view/per-frame pickles rather than the
project's Stage 2 schema.  This pilot keeps the camera frame as the common
``object`` frame, samples the static object mesh to 4096 points, upsamples the
778 MANO vertices to the 1538-point training contract, and derives contact
distance targets with a KD-tree.  It is intentionally bounded by group count.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _parse_name(path: Path) -> tuple[str, str, int, int, int]:
    parts = path.stem.split("__")
    if len(parts) < 5:
        raise ValueError(f"Unexpected OakInk filename: {path.name}")
    # OakInk names are ``sequence__camera__frame__view``.  The final field is
    # the view identifier used by seq_all.json; camera is retained only for
    # diagnostics.
    return parts[0], parts[1], int(parts[4]), int(parts[3]), int(parts[2])


def _mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v" and len(fields) >= 4:
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
        elif fields[0] == "f" and len(fields) >= 4:
            face = []
            for token in fields[1:4]:
                face.append(int(token.split("/")[0]) - 1)
            faces.append(face)
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise ValueError(f"Empty/invalid mesh: {path}")
    normals = np.zeros_like(v)
    if len(f):
        tri = v[f]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        for col in range(3):
            np.add.at(normals, f[:, col], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.clip(norms, 1e-8, None)
    return v, normals


def _transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def _transform_normals(normals: np.ndarray, transform: np.ndarray) -> np.ndarray:
    out = normals @ transform[:3, :3].T
    return (out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-8, None)).astype(np.float32)


def _sample(points: np.ndarray, normals: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic farthest-like coverage without introducing a dependency on
    # a second mesh sampler; OakInk meshes are already densely tessellated.
    idx = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[idx], normals[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oakink-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--max-groups", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.oakink_root).resolve()
    anno = root / "downloads" / "image" / "anno"
    obj_root = root / "downloads" / "image" / "obj"
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    hand_dir = anno / "hand_v"
    groups: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in hand_dir.iterdir():
        if path.suffix != ".pkl":
            continue
        obj_id, timestamp, view, frame, camera = _parse_name(path)
        groups[(obj_id, timestamp, view)].append(path)
    group_keys = sorted(groups)[: int(args.max_groups)]

    mesh_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows = []
    for group_idx, key in enumerate(group_keys):
        obj_id, timestamp, view = key
        paths = sorted(groups[key], key=lambda p: _parse_name(p)[3])
        if args.max_frames > 0:
            paths = paths[: int(args.max_frames)]
        mesh_id = obj_id.split("_", 1)[0]
        mesh_path = obj_root / f"{mesh_id}.obj"
        if mesh_id not in mesh_cache:
            mesh_cache[mesh_id] = _mesh(mesh_path)
        canonical_v, canonical_n = mesh_cache[mesh_id]
        obj_points: list[np.ndarray] = []
        obj_normals: list[np.ndarray] = []
        hand_points: list[np.ndarray] = []
        hand_normals: list[np.ndarray] = []
        raw_ids = []
        for path in paths:
            obj_id2, _timestamp, _view, frame, _camera = _parse_name(path)
            hand_cam = np.asarray(_load_pickle(path), dtype=np.float32)
            transf = np.asarray(_load_pickle(anno / "obj_transf" / path.name), dtype=np.float32)
            # obj_transf maps object coordinates to camera coordinates.  Invert
            # it so object and hand are represented in one object-fixed frame.
            cam_to_obj = np.linalg.inv(transf).astype(np.float32)
            obj_cam = _transform(canonical_v, transf)
            obj_n_cam = _transform_normals(canonical_n, transf)
            obj_o = _transform(obj_cam, cam_to_obj)
            obj_n = _transform_normals(obj_n_cam, cam_to_obj)
            hand_o = _transform(hand_cam, cam_to_obj)
            # MANO vertex normals are not distributed by OakInk.  A radial
            # proxy is sufficient for this pilot and is recorded explicitly.
            wrist = hand_o[0]
            hand_n = hand_o - wrist
            hand_n /= np.clip(np.linalg.norm(hand_n, axis=1, keepdims=True), 1e-8, None)
            hand_idx = np.linspace(0, len(hand_o) - 1, 1538, dtype=np.int64)
            obj_p, obj_nn = _sample(obj_o, obj_n, 4096)
            hand_p, hand_nn = hand_o[hand_idx], hand_n[hand_idx]
            obj_points.append(obj_p)
            obj_normals.append(obj_nn)
            hand_points.append(hand_p)
            hand_normals.append(hand_nn)
            raw_ids.append(frame)
        obj_arr = np.asarray(obj_points, dtype=np.float32)
        obj_n_arr = np.asarray(obj_normals, dtype=np.float32)
        hand_arr = np.asarray(hand_points, dtype=np.float32)
        hand_n_arr = np.asarray(hand_normals, dtype=np.float32)
        distances = np.empty((len(paths), 1538), dtype=np.float32)
        for i in range(len(paths)):
            distances[i] = cKDTree(obj_arr[i]).query(hand_arr[i], workers=1)[0]
        seq_id = f"{obj_id}/{timestamp}/view{view}"
        out = out_root / f"{group_idx:05d}_{obj_id}_view{view}.npz"
        np.savez_compressed(
            out,
            schema_name=np.asarray("train_corr_static_v2"),
            schema_version=np.asarray("2.0.0"),
            dataset_id=np.asarray("oakink"), dataset_name=np.asarray("OakInk"),
            seq_id=np.asarray(seq_id), side=np.asarray("right"),
            raw_frame_id=np.asarray(raw_ids, dtype=np.int32),
            obj_points=obj_arr, obj_normals=obj_n_arr,
            hand_points=hand_arr, hand_normals=hand_n_arr,
            hand_to_obj_min_dist=distances,
            coordinate_frame=np.asarray("object"),
        )
        rows.append({
            "seq_id": seq_id, "object_id": obj_id, "view": view,
            "frames": len(paths), "file": str(out),
            "contact_fraction_2cm": float((distances < 0.02).mean()),
            "distance_mean_m": float(distances.mean()),
            "distance_p50_m": float(np.quantile(distances, 0.50)),
            "distance_p95_m": float(np.quantile(distances, 0.95)),
        })
        print(f"[{group_idx + 1}/{len(group_keys)}] {seq_id} frames={len(paths)}", flush=True)
    summary = {
        "source": str(root), "output_root": str(out_root),
        "num_groups": len(rows), "num_frames": int(sum(x["frames"] for x in rows)),
        "schema_version": "2.0.0", "coordinate_frame": "object",
        "hand_points_source": "OakInk hand_v 778 vertices, deterministic upsample to 1538",
        "hand_normals_source": "radial proxy from wrist (pilot only)",
        "contact_source": "KD-tree distance to 4096 sampled object vertices",
        "groups": rows,
    }
    (out_root / "oakink_pilot_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
