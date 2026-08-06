from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np


STAGE2_SCHEMA_NAME = "ref2dex_opti"
STAGE2_SCHEMA_VERSION = "1.0.0"


def _as_array(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"Missing required source field: {key}")
    return np.asarray(payload[key])


def _approx_penetration_depth(
    obj_points: np.ndarray,
    obj_normals: np.ndarray,
    hand_points: np.ndarray,
    hand_to_obj_idx: np.ndarray | None,
) -> np.ndarray:
    """Nearest-sampled-object normal approximation, returned in meters."""
    num_frames, num_hand, _ = hand_points.shape
    if hand_to_obj_idx is None:
        out = np.zeros((num_frames,), dtype=np.float32)
        for frame_idx in range(num_frames):
            delta = hand_points[frame_idx, :, None] - obj_points[frame_idx, None]
            dist2 = np.sum(delta * delta, axis=-1)
            nearest = np.argmin(dist2, axis=1)
            nearest_points = obj_points[frame_idx, nearest]
            nearest_normals = obj_normals[frame_idx, nearest]
            signed = np.sum((hand_points[frame_idx] - nearest_points) * nearest_normals, axis=-1)
            out[frame_idx] = float(np.maximum(-signed, 0.0).max(initial=0.0))
        return out

    nearest = np.asarray(hand_to_obj_idx, dtype=np.int64)
    nearest = np.clip(nearest, 0, obj_points.shape[1] - 1)
    frame_axis = np.arange(num_frames, dtype=np.int64)[:, None]
    nearest_points = obj_points[frame_axis, nearest]
    nearest_normals = obj_normals[frame_axis, nearest]
    signed = np.sum((hand_points - nearest_points) * nearest_normals, axis=-1)
    return np.maximum(-signed, 0.0).max(axis=1).astype(np.float32)


def pack_stage2_hand(
    source: dict[str, Any],
    *,
    side: str,
    source_raw_file: str,
    processing_mode: str,
    frame_keep_threshold: float,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    """Convert an in-memory dataset-specific payload to the common Stage 2 schema."""
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported side: {side}")

    raw_frame_id = _as_array(source, "raw_frame_id").astype(np.int32)
    obj_points = _as_array(source, "obj_points_world").astype(np.float32)
    obj_normals = _as_array(source, "obj_normals_world").astype(np.float32)
    obj_root_pose = _as_array(source, "obj_root_pose").astype(np.float32)
    hand_points = _as_array(source, f"{side}_hand_points_world").astype(np.float32)
    hand_normals = _as_array(source, f"{side}_hand_normals_world").astype(np.float32)

    min_dist_key = f"{side}_hand_min_dist_to_obj"
    if min_dist_key in source:
        min_dist = np.asarray(source[min_dist_key], dtype=np.float32)
    else:
        delta = hand_points[:, :, None] - obj_points[:, None]
        min_dist = np.sqrt(np.sum(delta * delta, axis=-1).min(axis=(1, 2))).astype(np.float32)

    keep = np.isfinite(min_dist) & (min_dist <= float(frame_keep_threshold))
    if not np.any(keep):
        return None

    hand_to_obj_key = f"{side}_hand_to_obj_nn_id"
    hand_to_obj_idx = (
        np.asarray(source[hand_to_obj_key], dtype=np.int64)
        if hand_to_obj_key in source
        else None
    )
    penetration_depth = _approx_penetration_depth(
        obj_points,
        obj_normals,
        hand_points,
        hand_to_obj_idx,
    )

    hand_root_pose = None
    hand_root_pose_key = f"{side}_hand_root_pose"
    if hand_root_pose_key in source:
        hand_root_pose = np.asarray(source[hand_root_pose_key], dtype=np.float32)

    kept_count = int(keep.sum())
    payload: dict[str, Any] = {
        "schema_name": STAGE2_SCHEMA_NAME,
        "schema_version": STAGE2_SCHEMA_VERSION,
        "source_raw_file": str(source_raw_file),
        "processing_mode": str(processing_mode),
        "dataset_name": str(source["dataset_name"]),
        "seq_id": str(source["seq_id"]),
        "subject_id": str(source["subject_id"]),
        "seq_name": str(source["seq_name"]),
        "object_name": str(source["object_name"]),
        "side": side,
        "raw_frame_id": raw_frame_id[keep],
        "stage2_frame_idx": np.arange(kept_count, dtype=np.int32),
        "obj_points_world": obj_points[keep],
        "obj_normals_world": obj_normals[keep],
        "obj_point_id": _as_array(source, "obj_point_id").astype(np.int32),
        "obj_root_pose": obj_root_pose[keep],
        "hand_points_world": hand_points[keep],
        "hand_normals_world": hand_normals[keep],
        "hand_point_id": _as_array(source, f"{side}_hand_point_id").astype(np.int32),
        "hand_cano_points": _as_array(source, f"{side}_hand_cano_points").astype(np.float32),
        "hand_finger_id": _as_array(source, f"{side}_hand_finger_id").astype(np.int32),
        "hand_region_id": _as_array(source, f"{side}_hand_region_id").astype(np.int32),
        "frame_min_hand_obj_dist": min_dist[keep],
        "penetration_depth": penetration_depth[keep],
        "processing_success": np.ones((kept_count,), dtype=bool),
        "processing_message": ["init_only"] * kept_count,
        "config": {
            **config,
            "frame_keep_threshold": float(frame_keep_threshold),
            "penetration_metric": "nearest_sampled_object_point_normal_approx",
            "length_unit": "meter",
        },
    }
    if hand_root_pose is not None:
        payload["hand_root_pose"] = hand_root_pose[keep]

    # ---- MANO fields (cross-dataset compatibility, docs/指导.md) ----
    # Forward raw MANO parameters + descriptive config so stage 3 can write
    # them to the npz and the train side can re-run MANO forward / perturb
    # in PCA space. All fields are optional at this layer; missing fields
    # simply mean the upstream adapter did not produce them.
    for mano_key in (
        "mano_global_orient",
        "mano_transl",
        "mano_pose",
        "mano_betas",
    ):
        source_key = f"{side}_{mano_key}"
        if source_key in source and source[source_key] is not None:
            value = _as_array(source, source_key).astype(np.float32)
            if value.ndim > 0 and value.shape[0] == int(raw_frame_id.shape[0]):
                value = value[keep]
            payload[mano_key] = value
    # Configuration booleans / ints / strings are frame-invariant; pass
    # through as-is when the upstream adapter emitted them.
    for cfg_key, caster in (
        ("mano_use_pca", bool),
        ("mano_num_pca_comps", int),
        ("mano_flat_hand_mean", bool),
        ("mano_pose_repr", str),
    ):
        source_key = f"{side}_{cfg_key}"
        if source_key in source and source[source_key] is not None:
            payload[cfg_key] = caster(source[source_key])
    # v_template is a per-subject asset, NOT per-frame. It must NOT be
    # indexed by `keep` (which is a per-frame mask).
    vtemp_key = f"{side}_mano_v_template"
    if vtemp_key in source and source[vtemp_key] is not None:
        payload["mano_v_template"] = _as_array(source, vtemp_key).astype(np.float32)

    # ---- Object parametric/canonical fields (Stage 3 v2.1 optional schema) ----
    # The per-frame obj_points_world/obj_normals_world remain the authoritative
    # training path for now.  These fields preserve enough information for a
    # later canonical-object runtime path without making GRAB/ARCTIC diverge at
    # the Stage 3 file level.
    for obj_key in ("obj_points_canonical", "obj_normals_canonical"):
        if obj_key in source and source[obj_key] is not None:
            payload[obj_key] = _as_array(source, obj_key).astype(np.float32)
    if "obj_repr" in source and source["obj_repr"] is not None:
        payload["obj_repr"] = str(source["obj_repr"])
    if "obj_part_id" in source and source["obj_part_id"] is not None:
        payload["obj_part_id"] = _as_array(source, "obj_part_id").astype(np.int32)
    if "obj_articulation" in source and source["obj_articulation"] is not None:
        value = _as_array(source, "obj_articulation").astype(np.float32)
        if value.ndim > 0 and value.shape[0] == int(raw_frame_id.shape[0]):
            value = value[keep]
        payload["obj_articulation"] = value
    return payload


def save_stage2_payload(payload: dict[str, Any], output_root: Path) -> Path:
    subject_dir = output_root / str(payload["subject_id"])
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_path = subject_dir / f"{payload['seq_name']}_{payload['side']}.pkl"
    with output_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def write_stage2_meta(
    output_root: Path,
    *,
    dataset_name: str,
    processing_mode: str,
    source_root: str,
    config: dict[str, Any],
    stats: dict[str, Any],
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_name": STAGE2_SCHEMA_NAME,
        "schema_version": STAGE2_SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": dataset_name,
        "processing_mode": processing_mode,
        "source_root": str(source_root),
        "output_root": str(output_root.resolve()),
        "config": config,
        "stats": stats,
    }
    path = output_root / "meta.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)
    return path
