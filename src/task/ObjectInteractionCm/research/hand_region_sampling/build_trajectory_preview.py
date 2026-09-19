#!/usr/bin/env python3
"""Build a small variable-hand-point trajectory preview for ``visualize_grab``.

The production cache keeps 1538 hand points.  This research-only builder
reuses its object trajectories and provenance, but reconstructs the hand
surface with the latest area-density contract: MANO=2048 and Inspire=10135.
The same canonical triangle/barycentric samples are transformed across every
frame, so the output is a trajectory rather than a static PLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task.ObjectInteractionCm.research.hand_region_sampling.run import (  # noqa: E402
    INSPIRE_RATIO_POINTS,
    NUM_POINTS,
    SURFACE_SEED,
    _build_inspire_pool,
    _build_mano_pool,
)
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (  # noqa: E402
    InspireUrdfModel,
    NATIVE_Q_START,
    NUM_DOFS,
    _load_tensor,
)


OBJECT_POOL_POINTS = 4096
HAND_POINT_COUNTS = {"mano": NUM_POINTS, "inspire_rl": INSPIRE_RATIO_POINTS}
TARGET_VARIANTS = ("mano", "inspire_rl")
DEFAULT_COUNT_PER_VARIANT = 3
ACTIVE_THRESHOLD_M = 0.05


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_info() -> Tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _safe_name(sequence_id: str) -> str:
    return sequence_id.replace("/", "_")


def _save(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value))


def _sample_correspondence(pool: Any, count: int, seed: int) -> Dict[str, np.ndarray]:
    """Match ``run.py`` global area sampling and retain transform metadata."""

    rng = np.random.default_rng(int(seed))
    probabilities = np.asarray(pool.areas, dtype=np.float64)
    probabilities /= probabilities.sum()
    chosen = rng.choice(len(pool.triangles), size=int(count), replace=True, p=probabilities)
    u = np.sqrt(rng.random(int(count)))
    v = rng.random(int(count))
    barycentric = np.stack((1.0 - u, u * (1.0 - v), u * v), axis=1).astype(np.float32)
    return {
        "face_ids": np.asarray(pool.source_face_ids[chosen], dtype=np.int64),
        "visual_ids": np.asarray(pool.source_visual_ids[chosen], dtype=np.int64),
        "barycentric": barycentric,
    }


def _sample_mano_frames(
    parent_root: Path,
    source_frame_ids: np.ndarray,
    target_pose: np.ndarray,
    correspondence: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    vertices_path = parent_root / "right" / "hand_mesh_vertices_world.npy"
    faces_path = parent_root / "right" / "hand_mesh_faces.npy"
    old_pose_path = parent_root / "shared" / "obj_pose_world.npy"
    old_raw_path = parent_root / "shared" / "raw_frame_id.npy"
    for path in (vertices_path, faces_path, old_pose_path, old_raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"MANO provenance file is unavailable: {path}")
    vertices_world = np.load(vertices_path, mmap_mode="r")
    faces = np.asarray(np.load(faces_path, mmap_mode="r"), dtype=np.int64)
    old_pose = np.load(old_pose_path, mmap_mode="r")
    old_raw = np.asarray(np.load(old_raw_path, mmap_mode="r"), dtype=np.int64)
    if vertices_world.ndim != 3 or vertices_world.shape[1:] != (778, 3):
        raise ValueError(f"invalid MANO mesh vertices {vertices_path}: {vertices_world.shape}")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError(f"invalid MANO mesh faces {faces_path}: {faces.shape}")
    if old_pose.shape != (len(old_raw), 4, 4):
        raise ValueError(f"invalid MANO object poses {old_pose_path}: {old_pose.shape}")
    raw_to_index = {int(value): index for index, value in enumerate(old_raw)}
    face_ids = correspondence["face_ids"]
    barycentric = correspondence["barycentric"]
    if np.any(face_ids < 0) or np.any(face_ids >= len(faces)):
        raise ValueError("MANO correspondence contains an out-of-range face id")
    points = np.empty((len(source_frame_ids), len(face_ids), 3), dtype=np.float32)
    normals = np.empty_like(points)
    for frame, source_id in enumerate(np.asarray(source_frame_ids, dtype=np.int64)):
        parent_frame = raw_to_index.get(int(source_id))
        if parent_frame is None:
            raise KeyError(f"source frame {source_id} is absent from {old_raw_path}")
        old_vertices = np.asarray(vertices_world[parent_frame], dtype=np.float32)
        old_object_pose = np.asarray(old_pose[parent_frame], dtype=np.float32)
        local_vertices = (old_vertices - old_object_pose[:3, 3]) @ old_object_pose[:3, :3]
        world_vertices = local_vertices @ target_pose[frame, :3, :3].T + target_pose[frame, :3, 3]
        triangles = world_vertices[faces[face_ids]]
        points[frame] = np.einsum("ni,nij->nj", barycentric, triangles)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals[frame] = cross / np.clip(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12, None)
    return points, normals


def _sample_inspire_frames(
    urdf_path: Path,
    q_tensor_path: Path,
    target_frames: int,
    correspondence: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    model = InspireUrdfModel(urdf_path)
    data = _load_tensor(q_tensor_path)
    q_native = data[:, NATIVE_Q_START : NATIVE_Q_START + NUM_DOFS]
    if len(q_native) != target_frames:
        raise ValueError(f"Inspire q/geometry frame mismatch: {len(q_native)} != {target_frames}")
    face_ids = correspondence["face_ids"]
    visual_ids = correspondence["visual_ids"]
    barycentric = correspondence["barycentric"]
    if np.any(visual_ids < 0) or np.any(visual_ids >= len(model.visuals)):
        raise ValueError("Inspire correspondence contains an out-of-range visual id")
    points = np.empty((target_frames, len(face_ids), 3), dtype=np.float32)
    normals = np.empty_like(points)
    for frame, native_q in enumerate(np.asarray(q_native, dtype=np.float64)):
        links = model.link_transforms(model.qpos_to_urdf_order(native_q))
        for visual_id in np.unique(visual_ids):
            sample_mask = visual_ids == visual_id
            visual = model.visuals[int(visual_id)]
            transform = links[visual.link] @ visual.local_transform
            triangles_local = visual.vertices[visual.faces[face_ids[sample_mask]]]
            triangles = triangles_local @ transform[:3, :3].T + transform[:3, 3]
            points[frame, sample_mask] = np.einsum(
                "ni,nij->nj", barycentric[sample_mask], triangles
            )
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals[frame, sample_mask] = cross / np.clip(
                np.linalg.norm(cross, axis=1, keepdims=True), 1e-12, None
            )
    return points, normals


def _active_mask(object_points: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("scipy is required to compute the preview 5 cm mask") from exc
    result = np.zeros((len(object_points),), dtype=bool)
    for frame, (objects, hands) in enumerate(zip(object_points, hand_points)):
        nearest = cKDTree(np.asarray(objects, dtype=np.float32)).query(
            np.asarray(hands, dtype=np.float32), k=1, workers=1
        )[0]
        result[frame] = bool(np.min(nearest) <= ACTIVE_THRESHOLD_M)
    return result


def _select_entries(index: Dict[str, Any], split: str, count_per_variant: int) -> List[Dict[str, Any]]:
    entries = list(index.get("sequences", {}).get(split, []))
    selected: List[Dict[str, Any]] = []
    for variant in TARGET_VARIANTS:
        candidates = sorted((e for e in entries if e.get("variant") == variant), key=lambda e: str(e["id"]))
        if len(candidates) < count_per_variant:
            raise ValueError(f"split={split} has only {len(candidates)} {variant} sequences")
        selected.extend(candidates[:count_per_variant])
    return selected


def _copy_sequence(
    entry: Dict[str, Any],
    *,
    source_root: Path,
    output_root: Path,
    mano_pool: Any,
    inspire_pool: Any,
    inspire_model_path: Path,
    surface_seed: int,
) -> Dict[str, Any]:
    source_path = source_root / str(entry["path"])
    source_geometry = source_path / "geometry"
    source_manifest_path = source_geometry / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"source sequence manifest is unavailable: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    object_points = np.asarray(np.load(source_geometry / "obj_points_pool_world.npy", mmap_mode="r"), dtype=np.float32)
    object_normals = np.asarray(np.load(source_geometry / "obj_normals_pool_world.npy", mmap_mode="r"), dtype=np.float32)
    object_pose = np.asarray(np.load(source_geometry / "obj_pose_world.npy", mmap_mode="r"), dtype=np.float32)
    source_frame = np.asarray(np.load(source_geometry / "source_frame_id.npy", mmap_mode="r"), dtype=np.int32)
    frame_time = np.asarray(np.load(source_geometry / "frame_time.npy", mmap_mode="r"), dtype=np.float32)
    if object_points.ndim != 3 or object_points.shape[1:] != (OBJECT_POOL_POINTS, 3):
        raise ValueError(f"unexpected object pool shape for {entry['id']}: {object_points.shape}")
    T = len(object_points)
    if object_normals.shape != object_points.shape or object_pose.shape != (T, 4, 4):
        raise ValueError(f"object metadata shape mismatch for {entry['id']}")
    if source_frame.shape != (T,) or frame_time.shape != (T,):
        raise ValueError(f"frame metadata shape mismatch for {entry['id']}")

    variant = str(entry["variant"])
    count = HAND_POINT_COUNTS[variant]
    if variant == "mano":
        correspondence = _sample_correspondence(mano_pool, count, surface_seed)
        hand_points, hand_normals = _sample_mano_frames(
            Path(str(source_manifest["input_parent_cache"])), source_frame, object_pose, correspondence
        )
        asset = "mano_parent_mesh_topology"
        source_type = "mano_parent_geometry_area_uniform_preview"
        sampling_asset = str(mano_pool.source_name)
        rl_q = None
    elif variant == "inspire_rl":
        correspondence = _sample_correspondence(inspire_pool, count, surface_seed)
        rl_q = source_manifest.get("rl_q")
        if not isinstance(rl_q, dict) or not rl_q.get("tensor"):
            raise ValueError(f"Inspire sequence {entry['id']} lacks rl_q.tensor")
        hand_points, hand_normals = _sample_inspire_frames(
            inspire_model_path,
            Path(str(rl_q["tensor"])).expanduser().resolve(),
            T,
            correspondence,
        )
        asset = "inspire_hand_right_urdf_visuals"
        source_type = "dexplore_rl_native_q_area_uniform_preview"
        sampling_asset = str(inspire_model_path.resolve())
    else:
        raise ValueError(f"unsupported preview variant {variant!r}")
    if hand_points.shape != (T, count, 3) or hand_normals.shape != hand_points.shape:
        raise ValueError(f"hand shape mismatch for {entry['id']}: {hand_points.shape}")
    if not np.isfinite(hand_points).all() or not np.isfinite(hand_normals).all():
        raise ValueError(f"non-finite hand geometry for {entry['id']}")
    active = _active_mask(object_points, hand_points)

    seq_out = output_root / "sequences" / str(entry["split"]) / variant / _safe_name(str(entry["id"]))
    geometry = seq_out / "geometry"
    geometry.mkdir(parents=True, exist_ok=False)
    _save(geometry / "obj_points_pool_world.npy", object_points)
    _save(geometry / "obj_normals_pool_world.npy", object_normals)
    _save(geometry / "obj_pose_world.npy", object_pose)
    _save(geometry / "source_frame_id.npy", source_frame)
    _save(geometry / "frame_time.npy", frame_time)
    _save(geometry / "hand_points_world.npy", hand_points)
    _save(geometry / "hand_normals_world.npy", hand_normals)
    _save(geometry / "obj_candidate_mask_5cm.npy", active)
    np.savez_compressed(
        geometry / "sampling_correspondence.npz",
        face_ids=correspondence["face_ids"],
        visual_ids=correspondence["visual_ids"],
        barycentric=correspondence["barycentric"],
    )
    manifest = {
        "schema_name": source_manifest["schema_name"],
        "schema_version": source_manifest.get("schema_version", "1.0.0"),
        "sequence_id": str(entry["id"]),
        "parent_seq_id": str(entry.get("parent_seq_id", entry["id"])),
        "split": str(entry["split"]),
        "variant": variant,
        "source": str(entry["source"]),
        "source_type": source_type,
        "coordinate_frame": source_manifest["coordinate_frame"],
        "world_frame": source_manifest["world_frame"],
        "hand_side": source_manifest["hand_side"],
        "object_pool_points": OBJECT_POOL_POINTS,
        "hand_points": count,
        "effective_fps": source_manifest.get("effective_fps", 30.0),
        "source_fps": source_manifest.get("source_fps", 120.0),
        "ds_rate": source_manifest.get("ds_rate", 4),
        "candidate_threshold_m": ACTIVE_THRESHOLD_M,
        "candidate_semantics": "preview_sampled_hand_to_object_surface_5cm",
        "candidate_mask_shape": [T],
        "surface_sampling": {
            "method": "global_surface_area_uniform_triangle_barycentric",
            "seed": int(surface_seed),
            "asset": asset,
            "point_count": count,
            "correspondence_file": "sampling_correspondence.npz",
            "sampling_asset": sampling_asset,
        },
        "rl_q": rl_q,
        "input_parent_cache": source_manifest.get("input_parent_cache"),
        "preview_source_cache": str(source_path.resolve()),
    }
    (geometry / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {
        "source": str(entry["source"]),
        "id": str(entry["id"]),
        "parent_seq_id": str(entry.get("parent_seq_id", entry["id"])),
        "path": str(seq_out.relative_to(output_root)),
        "variant": variant,
        "split": str(entry["split"]),
        "subject_id": entry.get("subject_id"),
        "object_name": str(entry["object_name"]),
        "action_name": entry.get("action_name"),
        "frame_count": T,
        "source_frame_first": int(source_frame[0]),
        "source_frame_last": int(source_frame[-1]),
        "hand_points": count,
        "active_frames": int(active.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", default="data/processed_data/object_interaction_cm_dexplore_rl_v1_2_5/index.json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--urdf",
        default="/home2/wyy/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf",
    )
    parser.add_argument("--mano-model-dir", default="dataset/arctic/data/body_models/mano")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--count-per-variant", type=int, default=DEFAULT_COUNT_PER_VARIANT)
    parser.add_argument("--surface-seed", type=int, default=SURFACE_SEED)
    parser.add_argument("--work-version", default="V1.2.15")
    args = parser.parse_args()
    if args.surface_seed != SURFACE_SEED:
        raise ValueError(f"area-ratio preview is pinned to surface_seed={SURFACE_SEED}")
    if args.count_per_variant <= 0:
        raise ValueError("count-per-variant must be positive")
    output_root = Path(args.output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    source_index_path = Path(args.source_index).expanduser().resolve()
    source_root = source_index_path.parent
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    if source_index.get("schema_name") != "ref2dex_object_interaction_cm_index_v1_1":
        raise ValueError(f"unsupported source index schema: {source_index.get('schema_name')!r}")
    urdf_path = Path(args.urdf).expanduser().resolve()
    mano_model_dir = Path(args.mano_model_dir).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(urdf_path)
    if not (mano_model_dir / "MANO_RIGHT.pkl").is_file():
        raise FileNotFoundError(mano_model_dir / "MANO_RIGHT.pkl")

    selected = _select_entries(source_index, args.split, args.count_per_variant)
    mano_pool, mano_info = _build_mano_pool(mano_model_dir, 0.20)
    inspire_pool, inspire_info = _build_inspire_pool(urdf_path, 0.20)
    records = []
    for entry in selected:
        records.append(
            _copy_sequence(
                entry,
                source_root=source_root,
                output_root=output_root,
                mano_pool=mano_pool,
                inspire_pool=inspire_pool,
                inspire_model_path=urdf_path,
                surface_seed=args.surface_seed,
            )
        )
    assignment = {
        "schema_name": "ref2dex_object_interaction_cm_variable_hand_preview_assignment_v1",
        "work_version": args.work_version,
        "source_index": str(source_index_path),
        "split": args.split,
        "count_per_variant": args.count_per_variant,
        "sampling": {
            "method": "global_surface_area_uniform_triangle_barycentric",
            "surface_seed": args.surface_seed,
            "point_counts": HAND_POINT_COUNTS,
        },
        "sequences": records,
    }
    (output_root / "assignment.json").write_text(
        json.dumps(assignment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    index = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_1",
        "schema_version": "1.1.0",
        "created_at": _now(),
        "experiment_schema": "ref2dex_object_interaction_cm_variable_hand_preview_v1",
        "source_probability": {"mano": 0.5, "inspire_rl": 0.5},
        "object_pool_points": OBJECT_POOL_POINTS,
        "model_object_points": int(source_index.get("model_object_points", 1024)),
        "hand_points_per_stream": HAND_POINT_COUNTS,
        "max_union_hand_points": max(HAND_POINT_COUNTS.values()),
        "split_policy": "selected deterministic preview subset from source index",
        "source_index": str(source_index_path),
        "assignment_manifest": "assignment.json",
        "sequences": {args.split: records},
        "counts": {args.split: {variant: sum(r["variant"] == variant for r in records) for variant in TARGET_VARIANTS}},
        "sampling_assets": {"mano": mano_info, "inspire": inspire_info},
    }
    (output_root / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    commit, dirty = _git_info()
    run_manifest = {
        "schema_name": "ref2dex_object_interaction_cm_variable_hand_preview_run_v1",
        "run_id": output_root.name,
        "run_status": "COMPLETED",
        "created_at": _now(),
        "work_version": args.work_version,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": " ".join([sys.executable, *sys.argv]),
        "source_index": str(source_index_path),
        "split": args.split,
        "count_per_variant": args.count_per_variant,
        "point_counts": HAND_POINT_COUNTS,
        "surface_seed": args.surface_seed,
        "sampling_method": "global_surface_area_uniform_triangle_barycentric",
        "outputs": {"index": "index.json", "assignment": "assignment.json"},
        "sequence_count": len(records),
        "records": records,
        "asset_sha256": {
            "urdf": _sha256(urdf_path),
            "mano_model": _sha256(mano_model_dir / "MANO_RIGHT.pkl"),
        },
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_root), "records": records}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
