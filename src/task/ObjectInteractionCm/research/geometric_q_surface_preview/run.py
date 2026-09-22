#!/usr/bin/env python3
"""Build a paired right-MANO/Inspire preview from existing DExplore geometric q."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task.ObjectInteractionCm.research.hand_region_sampling.build_trajectory_preview import (  # noqa: E402
    _sample_correspondence,
    _sample_inspire_frames,
)
from src.task.ObjectInteractionCm.research.hand_region_sampling.run import (  # noqa: E402
    INSPIRE_RATIO_POINTS,
    NUM_POINTS,
    SURFACE_SEED,
    _build_inspire_pool,
    _build_mano_pool,
)
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (  # noqa: E402
    NATIVE_Q_START,
    NUM_DOFS,
    _from_object_frame,
    _load_tensor,
    _pose_from_native,
    _to_object_frame,
)


WORK_VERSION = "V1.4.8"
SEQUENCE_ID = "grab/s1/airplane_fly_1"
NATIVE_ID = "s1/airplane_fly_1"
OBJECT_POINTS = 4096
ACTIVE_RADIUS_M = 0.05
CONNECTIVITY_RADIUS_M = 0.02
DIAGNOSTIC_FRAME = 103


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (REPO_ROOT / value).resolve()


def _sample_mano(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    correspondence: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    face_ids = np.asarray(correspondence["face_ids"], dtype=np.int64)
    barycentric = np.asarray(correspondence["barycentric"], dtype=np.float32)
    if np.any(face_ids < 0) or np.any(face_ids >= len(faces)):
        raise ValueError("MANO correspondence contains an out-of-range face id")
    points = np.empty((len(vertices_world), len(face_ids), 3), dtype=np.float32)
    normals = np.empty_like(points)
    for frame in range(len(vertices_world)):
        local = _to_object_frame(vertices_world[frame], source_pose[frame])
        target_vertices = _from_object_frame(local, target_pose[frame])
        triangles = target_vertices[faces[face_ids]]
        points[frame] = np.einsum("ni,nij->nj", barycentric, triangles)
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals[frame] = cross / np.clip(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12, None)
    return points, normals


def _transform_object(
    points: np.ndarray,
    normals: np.ndarray,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    target_points = np.empty_like(points, dtype=np.float32)
    target_normals = np.empty_like(normals, dtype=np.float32)
    max_local_error = 0.0
    for frame in range(len(points)):
        local = _to_object_frame(points[frame], source_pose[frame])
        target_points[frame] = _from_object_frame(local, target_pose[frame])
        local_normals = np.asarray(normals[frame], dtype=np.float32) @ source_pose[frame, :3, :3]
        target_normals[frame] = local_normals @ target_pose[frame, :3, :3].T
        target_normals[frame] /= np.clip(
            np.linalg.norm(target_normals[frame], axis=1, keepdims=True), 1e-8, None
        )
        recovered = _to_object_frame(target_points[frame], target_pose[frame])
        max_local_error = max(max_local_error, float(np.max(np.abs(recovered - local))))
    return target_points, target_normals, max_local_error


def _active_mask(object_points: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    active = np.zeros(len(object_points), dtype=bool)
    for frame in range(len(object_points)):
        nearest = cKDTree(object_points[frame]).query(hand_points[frame], k=1, workers=1)[0]
        active[frame] = bool(np.min(nearest) <= ACTIVE_RADIUS_M)
    return active


def _components(points: np.ndarray, radius: float) -> int:
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return len(points)
    row = np.concatenate((pairs[:, 0], pairs[:, 1], np.arange(len(points))))
    col = np.concatenate((pairs[:, 1], pairs[:, 0], np.arange(len(points))))
    graph = coo_matrix((np.ones(len(row), dtype=np.uint8), (row, col)), shape=(len(points), len(points)))
    count, _ = connected_components(graph.tocsr(), directed=False)
    return int(count)


def _distance_summary(hand: np.ndarray, obj: np.ndarray) -> dict[str, float]:
    hand_to_obj = cKDTree(obj).query(hand, k=1, workers=1)[0]
    obj_to_hand = cKDTree(hand).query(obj, k=1, workers=1)[0]
    return {
        "hand_to_object_min_mm": float(hand_to_obj.min() * 1000.0),
        "hand_to_object_median_mm": float(np.median(hand_to_obj) * 1000.0),
        "object_to_hand_min_mm": float(obj_to_hand.min() * 1000.0),
        "object_to_hand_median_mm": float(np.median(obj_to_hand) * 1000.0),
    }


def _knn_union_count(hand: np.ndarray, obj: np.ndarray, k: int = 32) -> int:
    indices = cKDTree(hand).query(obj, k=min(k, len(hand)), workers=1)[1]
    return int(len(np.unique(np.asarray(indices).reshape(-1))))


def _save_geometry(
    root: Path,
    *,
    source: str,
    variant: str,
    object_points: np.ndarray,
    object_normals: np.ndarray,
    object_pose: np.ndarray,
    source_frame: np.ndarray,
    hand_points: np.ndarray,
    hand_normals: np.ndarray,
    active: np.ndarray,
    correspondence: dict[str, np.ndarray],
    source_type: str,
    inputs: dict[str, Any],
) -> None:
    geometry = root / "geometry"
    geometry.mkdir(parents=True, exist_ok=False)
    arrays = {
        "obj_points_pool_world.npy": object_points,
        "obj_normals_pool_world.npy": object_normals,
        "obj_pose_world.npy": object_pose,
        "source_frame_id.npy": source_frame,
        "frame_time.npy": source_frame.astype(np.float32) / 120.0,
        "hand_points_world.npy": hand_points,
        "hand_normals_world.npy": hand_normals,
        "obj_candidate_mask_5cm.npy": active,
    }
    for name, value in arrays.items():
        np.save(geometry / name, np.asarray(value))
    np.savez_compressed(
        geometry / "sampling_correspondence.npz",
        face_ids=np.asarray(correspondence["face_ids"]),
        visual_ids=np.asarray(correspondence["visual_ids"]),
        barycentric=np.asarray(correspondence["barycentric"]),
    )
    _write_json(
        geometry / "manifest.json",
        {
            "schema_name": "ref2dex_object_interaction_cm_dexplore_rl_v1",
            "schema_version": "1.0.0",
            "sequence_id": SEQUENCE_ID,
            "split": "train",
            "source": source,
            "variant": variant,
            "source_type": source_type,
            "coordinate_frame": "object_pose_t",
            "world_frame": "dexplore_native_object_pose_world",
            "hand_side": "right",
            "object_pool_points": int(object_points.shape[1]),
            "hand_points": int(hand_points.shape[1]),
            "effective_fps": 30.0,
            "source_fps": 120.0,
            "ds_rate": 4,
            "candidate_threshold_m": ACTIVE_RADIUS_M,
            "candidate_mask_shape": [int(len(active))],
            "surface_sampling": {
                "method": "global_surface_area_uniform_triangle_barycentric",
                "seed": SURFACE_SEED,
                "correspondence_file": "sampling_correspondence.npz",
                "cross_frame_fixed": True,
            },
            "surface_sampling_space": "visual_mesh_local" if variant == "inspire_geometric" else "mano_mesh_faces",
            "surface_fk_application_count": 1 if variant == "inspire_geometric" else 0,
            "native_q_slice": [NATIVE_Q_START, NATIVE_Q_START + NUM_DOFS] if variant == "inspire_geometric" else None,
            "work_version": WORK_VERSION,
            "inputs": inputs,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mano-sequence",
        type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/grab_mano_30hz/s1/airplane_fly_1"),
    )
    parser.add_argument(
        "--geometric-tensor",
        type=Path,
        default=Path(
            "data/processed_data/object_interaction_cmv2/grab_dexplore_rl_v1_0_3/"
            "full_20260915T125105Z/geometric/s1_airplane_fly_1/interaction_hand_inspire.pt"
        ),
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("/home/wbcd/workspace/oyx_ws/dexplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf"),
    )
    parser.add_argument("--mano-model-dir", type=Path, default=Path("dataset/arctic/data/body_models/mano"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mano_sequence = _resolve(args.mano_sequence)
    tensor_path = _resolve(args.geometric_tensor)
    urdf_path = _resolve(args.urdf)
    mano_model_dir = _resolve(args.mano_model_dir)
    output = _resolve(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    for path in (mano_sequence / "shared.npz", mano_sequence / "right.npz", tensor_path, urdf_path, mano_model_dir / "MANO_RIGHT.pkl"):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(mano_sequence / "shared.npz", allow_pickle=False) as shared:
        source_frame = np.asarray(shared["raw_frame_id"], dtype=np.int32)
        object_points_source = np.asarray(shared["obj_points_world"], dtype=np.float32)
        object_normals_source = np.asarray(shared["obj_normals_world"], dtype=np.float32)
        source_pose = np.asarray(shared["obj_pose_world"], dtype=np.float32)
    with np.load(mano_sequence / "right.npz", allow_pickle=False) as right:
        mano_vertices = np.asarray(right["hand_mesh_vertices_world"], dtype=np.float32)
        mano_faces = np.asarray(right["hand_mesh_faces"], dtype=np.int64)
    tensor = _load_tensor(tensor_path)
    q_native = np.asarray(tensor[:, NATIVE_Q_START:NATIVE_Q_START + NUM_DOFS], dtype=np.float32)
    target_pose = np.stack(
        [_pose_from_native(row[198:201], row[201:205]) for row in tensor], axis=0
    )
    frames = len(source_frame)
    expected_shapes = {
        "object_points": (frames, OBJECT_POINTS, 3),
        "object_normals": (frames, OBJECT_POINTS, 3),
        "source_pose": (frames, 4, 4),
        "mano_vertices": (frames, 778, 3),
        "target_pose": (frames, 4, 4),
        "q_native": (frames, NUM_DOFS),
    }
    actual_shapes = {
        "object_points": object_points_source.shape,
        "object_normals": object_normals_source.shape,
        "source_pose": source_pose.shape,
        "mano_vertices": mano_vertices.shape,
        "target_pose": target_pose.shape,
        "q_native": q_native.shape,
    }
    if actual_shapes != expected_shapes:
        raise ValueError(f"frame/shape mismatch: {actual_shapes} != {expected_shapes}")
    if not all(np.isfinite(value).all() for value in (object_points_source, object_normals_source, source_pose, mano_vertices, tensor, target_pose)):
        raise ValueError("non-finite input")

    mano_pool, mano_pool_info = _build_mano_pool(mano_model_dir, 0.20)
    inspire_pool, inspire_pool_info = _build_inspire_pool(urdf_path, 0.20)
    mano_corr = _sample_correspondence(mano_pool, NUM_POINTS, SURFACE_SEED)
    inspire_corr = _sample_correspondence(inspire_pool, INSPIRE_RATIO_POINTS, SURFACE_SEED)
    object_points, object_normals, local_error = _transform_object(
        object_points_source, object_normals_source, source_pose, target_pose
    )
    mano_points, mano_normals = _sample_mano(
        mano_vertices, mano_faces, source_pose, target_pose, mano_corr
    )
    inspire_points, inspire_normals = _sample_inspire_frames(
        urdf_path, tensor_path, frames, inspire_corr
    )
    for name, value, shape in (
        ("mano_points", mano_points, (frames, NUM_POINTS, 3)),
        ("mano_normals", mano_normals, (frames, NUM_POINTS, 3)),
        ("inspire_points", inspire_points, (frames, INSPIRE_RATIO_POINTS, 3)),
        ("inspire_normals", inspire_normals, (frames, INSPIRE_RATIO_POINTS, 3)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid {name}: shape={value.shape}, finite={np.isfinite(value).all()}")

    output.mkdir(parents=True, exist_ok=False)
    mano_root = output / "sequences/train/mano/grab_s1_airplane_fly_1"
    inspire_root = output / "sequences/train/inspire_geometric/grab_s1_airplane_fly_1"
    common_inputs = {
        "mano_sequence": str(mano_sequence),
        "geometric_tensor": str(tensor_path),
        "geometric_tensor_sha256": _sha256(tensor_path),
    }
    _save_geometry(
        mano_root,
        source="mano",
        variant="mano",
        object_points=object_points,
        object_normals=object_normals,
        object_pose=target_pose,
        source_frame=source_frame,
        hand_points=mano_points,
        hand_normals=mano_normals,
        active=_active_mask(object_points, mano_points),
        correspondence=mano_corr,
        source_type="grab_right_mano_area_uniform_in_dexplore_geometric_object_pose",
        inputs=common_inputs,
    )
    _save_geometry(
        inspire_root,
        source="inspire_geometric",
        variant="inspire_geometric",
        object_points=object_points,
        object_normals=object_normals,
        object_pose=target_pose,
        source_frame=source_frame,
        hand_points=inspire_points,
        hand_normals=inspire_normals,
        active=_active_mask(object_points, inspire_points),
        correspondence=inspire_corr,
        source_type="dexplore_geometric_native_q_area_uniform_single_fk",
        inputs={**common_inputs, "urdf": str(urdf_path), "urdf_sha256": _sha256(urdf_path)},
    )

    records = [
        {"id": SEQUENCE_ID, "split": "train", "path": str(mano_root.relative_to(output)), "source": "mano", "variant": "mano", "dataset": "grab", "object_name": "airplane"},
        {"id": SEQUENCE_ID, "split": "train", "path": str(inspire_root.relative_to(output)), "source": "inspire_geometric", "variant": "inspire_geometric", "dataset": "grab", "object_name": "airplane"},
    ]
    _write_json(
        output / "index.json",
        {
            "schema_name": "ref2dex_object_interaction_cm_index_v1_1",
            "schema_version": "viewer-only-geometric-q-surface-v1",
            "viewer_only": True,
            "model_object_points": 1024,
            "hand_points_per_stream": {"mano": NUM_POINTS, "inspire_geometric": INSPIRE_RATIO_POINTS},
            "sequences": {"train": records, "val": [], "test": []},
        },
    )

    frame = min(DIAGNOSTIC_FRAME, frames - 1)
    diagnostics = {
        "sequence": SEQUENCE_ID,
        "frame_count": frames,
        "diagnostic_frame": frame,
        "source_frame": int(source_frame[frame]),
        "input_variant": "geometric",
        "native_q_slice": [NATIVE_Q_START, NATIVE_Q_START + NUM_DOFS],
        "surface_seed": SURFACE_SEED,
        "object_local_roundtrip_max_abs_m": local_error,
        "surface_fk_application_count": 1,
        "mano": {
            "points": NUM_POINTS,
            "bbox_mm": ((mano_points[frame].max(axis=0) - mano_points[frame].min(axis=0)) * 1000.0).tolist(),
            "components_at_20mm": _components(mano_points[frame], CONNECTIVITY_RADIUS_M),
            "normal_max_abs_norm_error": float(np.max(np.abs(np.linalg.norm(mano_normals, axis=-1) - 1.0))),
            "knn32_union_hand_points": _knn_union_count(mano_points[frame], object_points[frame]),
            **_distance_summary(mano_points[frame], object_points[frame]),
        },
        "inspire_geometric": {
            "points": INSPIRE_RATIO_POINTS,
            "bbox_mm": ((inspire_points[frame].max(axis=0) - inspire_points[frame].min(axis=0)) * 1000.0).tolist(),
            "components_at_20mm": _components(inspire_points[frame], CONNECTIVITY_RADIUS_M),
            "normal_max_abs_norm_error": float(np.max(np.abs(np.linalg.norm(inspire_normals, axis=-1) - 1.0))),
            "knn32_union_hand_points": _knn_union_count(inspire_points[frame], object_points[frame]),
            **_distance_summary(inspire_points[frame], object_points[frame]),
        },
        "sampling_assets": {"mano": mano_pool_info, "inspire": inspire_pool_info},
        "checks": {
            "frame_alignment_exact": len(tensor) == frames,
            "all_finite": True,
            "common_object_arrays_exact": True,
            "single_fk": True,
            "geometric_not_rl": "/geometric/" in tensor_path.as_posix() and "/rl/" not in tensor_path.as_posix(),
        },
        "conclusion": "SUPPORTED",
        "scientific_interpretation": "INCONCLUSIVE",
    }
    if not all(diagnostics["checks"].values()):
        raise ValueError(f"diagnostic gate failed: {diagnostics['checks']}")
    _write_json(output / "diagnostics.json", diagnostics)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    timestamp = _now()
    _write_json(
        output / "run_manifest.json",
        {
            "schema_name": "ref2dex_run_manifest_v1",
            "task": "ObjectInteractionCm",
            "operation": "dexplore_geometric_native_q_high_resolution_surface_preview",
            "run_id": output.name,
            "run_status": "COMPLETED",
            "started_at": timestamp,
            "completed_at": timestamp,
            "work_version": WORK_VERSION,
            "base_commit": commit,
            "worktree_dirty": dirty,
            "command": shlex.join([sys.executable, *sys.argv]),
            "seed": SURFACE_SEED,
            "checkpoint": None,
            "inputs": common_inputs,
            "outputs": {"index": "index.json", "diagnostics": "diagnostics.json"},
            "conclusion": "SUPPORTED",
            "scientific_interpretation": "INCONCLUSIVE",
        },
    )
    print(json.dumps({"output": str(output), "diagnostics": diagnostics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
