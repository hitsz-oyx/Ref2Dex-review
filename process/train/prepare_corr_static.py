#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle as pkl
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree


REF2DEX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANO_OPT_ROOT = REF2DEX_ROOT / "outputs" / "mano_fit"
DEFAULT_OUTPUT_ROOT = REF2DEX_ROOT / "outputs" / "train_corr_static"


for _legacy_name, _legacy_value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _legacy_name not in np.__dict__:
        setattr(np, _legacy_name, _legacy_value)


def _first_key(data: Dict, *keys: str) -> str:
    for key in keys:
        if key in data:
            return key
    raise KeyError(f"Missing keys={keys}; available={list(data.keys())}")


def _load_array(data: Dict, *keys: str) -> np.ndarray:
    return np.asarray(data[_first_key(data, *keys)])


def _points_world_to_obj(points_world: np.ndarray, obj_root_pose: np.ndarray) -> np.ndarray:
    R = obj_root_pose[..., :3, :3]
    t = obj_root_pose[..., :3, 3]
    return np.einsum("...ji,...pj->...pi", R, points_world - t[..., None, :]).astype(np.float32)


def _normals_world_to_obj(normals_world: np.ndarray, obj_root_pose: np.ndarray) -> np.ndarray:
    R = obj_root_pose[..., :3, :3]
    return np.einsum("...ji,...pj->...pi", R, normals_world).astype(np.float32)


def _face_centers_from_verts(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(verts[:, faces].mean(axis=2), dtype=np.float32)


def _face_normals_from_verts(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[:, faces[:, 0], :]
    v1 = verts[:, faces[:, 1], :]
    v2 = verts[:, faces[:, 2], :]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return (normals / np.clip(norms, 1e-10, None)).astype(np.float32)


def _load_mano_faces(mano_dir: Path) -> np.ndarray:
    with (mano_dir / "MANO_RIGHT.pkl").open("rb") as f:
        data = pkl.load(f, encoding="latin1")
    faces = np.asarray(data["f"], dtype=np.int64)
    return faces


def _compute_bidirectional_nn_per_frame(
    hand_points: np.ndarray,
    obj_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_frames = int(hand_points.shape[0])
    num_hand = int(hand_points.shape[1])
    num_obj = int(obj_points.shape[1])
    hand_to_obj_idx = np.full((num_frames, num_hand), -1, dtype=np.int32)
    hand_to_obj_dist = np.full((num_frames, num_hand), np.inf, dtype=np.float32)
    obj_to_hand_idx = np.full((num_frames, num_obj), -1, dtype=np.int32)
    obj_to_hand_dist = np.full((num_frames, num_obj), np.inf, dtype=np.float32)
    for t in range(num_frames):
        obj_tree = cKDTree(obj_points[t])
        hand_tree = cKDTree(hand_points[t])
        hand_to_obj_dist[t], hand_to_obj_idx[t] = obj_tree.query(hand_points[t], k=1)
        obj_to_hand_dist[t], obj_to_hand_idx[t] = hand_tree.query(obj_points[t], k=1)
    return (
        hand_to_obj_idx.astype(np.int32),
        hand_to_obj_dist.astype(np.float32),
        obj_to_hand_idx.astype(np.int32),
        obj_to_hand_dist.astype(np.float32),
    )


def _pack_object_crop_from_full(
    obj_points_world_full: np.ndarray,
    obj_normals_world_full: np.ndarray,
    obj_point_id_full: np.ndarray,
    obj_root_pose: np.ndarray,
    obj_keep_8cm_preopt_full: np.ndarray,
    obj_keep_8cm_postopt_full: np.ndarray,
    obj_to_hand_idx_full: np.ndarray,
    obj_to_hand_dist_full: np.ndarray,
    opt_hand_face_centers_world: np.ndarray,
    opt_hand_normals_world: np.ndarray,
) -> Dict[str, np.ndarray]:
    num_frames = int(obj_points_world_full.shape[0])
    max_crop = int(np.max(np.sum(obj_keep_8cm_preopt_full | obj_keep_8cm_postopt_full, axis=1))) if num_frames > 0 else 0
    out = {
        "obj_points_world_crop": np.zeros((num_frames, max_crop, 3), dtype=np.float32),
        "obj_normals_world_crop": np.zeros((num_frames, max_crop, 3), dtype=np.float32),
        "obj_points_obj_crop": np.zeros((num_frames, max_crop, 3), dtype=np.float32),
        "obj_normals_obj_crop": np.zeros((num_frames, max_crop, 3), dtype=np.float32),
        "obj_crop_valid_mask": np.zeros((num_frames, max_crop), dtype=bool),
        "obj_crop_raw_idx": np.full((num_frames, max_crop), -1, dtype=np.int32),
        "obj_crop_point_id": np.full((num_frames, max_crop), -1, dtype=np.int32),
        "obj_contact_region_3cm": np.zeros((num_frames, max_crop), dtype=bool),
        "obj_to_hand_nn_id": np.full((num_frames, max_crop), -1, dtype=np.int32),
        "obj_to_hand_dist": np.full((num_frames, max_crop), np.inf, dtype=np.float32),
    }
    final_keep = obj_keep_8cm_preopt_full | obj_keep_8cm_postopt_full
    for t in range(num_frames):
        keep_idx = np.flatnonzero(final_keep[t]).astype(np.int32)
        if keep_idx.size == 0:
            continue
        n_keep = int(keep_idx.size)
        obj_points_world_t = obj_points_world_full[t, keep_idx]
        obj_normals_world_t = obj_normals_world_full[t, keep_idx]
        out["obj_points_world_crop"][t, :n_keep] = obj_points_world_t
        out["obj_normals_world_crop"][t, :n_keep] = obj_normals_world_t
        out["obj_crop_valid_mask"][t, :n_keep] = True
        out["obj_crop_raw_idx"][t, :n_keep] = keep_idx
        out["obj_crop_point_id"][t, :n_keep] = obj_point_id_full[keep_idx]
        out["obj_points_obj_crop"][t, :n_keep] = _points_world_to_obj(obj_points_world_t[None], obj_root_pose[t:t + 1])[0]
        out["obj_normals_obj_crop"][t, :n_keep] = _normals_world_to_obj(obj_normals_world_t[None], obj_root_pose[t:t + 1])[0]
        out["obj_to_hand_nn_id"][t, :n_keep] = obj_to_hand_idx_full[t, keep_idx]
        out["obj_to_hand_dist"][t, :n_keep] = obj_to_hand_dist_full[t, keep_idx]
        out["obj_contact_region_3cm"][t, :n_keep] = obj_to_hand_dist_full[t, keep_idx] <= 0.03
    return out


def _load_legacy_stage2_crop(
    payload: Dict,
    pkl_path: Path,
    raw_frame_id: np.ndarray,
    side: str,
) -> Dict[str, np.ndarray]:
    preprocess_file = payload.get("source_preprocess_file") or payload.get("mano_file") or payload.get("object_file")
    if not preprocess_file:
        raise KeyError(f"{pkl_path} missing source preprocess path; cannot rebuild crop fields.")
    preprocess_path = Path(str(preprocess_file)).resolve()
    if not preprocess_path.exists():
        raise FileNotFoundError(f"Missing source preprocess file: {preprocess_path}")

    with np.load(str(preprocess_path), allow_pickle=True) as data:
        source_raw_frame_id = _load_array(data, "raw_frame_id", "frame_id").astype(np.int32)
        raw_to_pre = {int(fid): idx for idx, fid in enumerate(source_raw_frame_id.tolist())}
        selected = np.asarray([raw_to_pre[int(fid)] for fid in raw_frame_id.tolist()], dtype=np.int64)
        preprocess_frame_idx = (
            _load_array(data, "preprocess_frame_idx").astype(np.int32)[selected]
            if "preprocess_frame_idx" in data
            else selected.astype(np.int32)
        )
        obj_points_world_full = _load_array(data, "obj_points_world", "obj_points")[selected].astype(np.float32)
        obj_normals_world_full = _load_array(data, "obj_normals_world", "obj_normals")[selected].astype(np.float32)
        obj_root_pose = _load_array(data, "obj_root_pose")[selected].astype(np.float32)
        obj_point_id = _load_array(data, "obj_point_id").astype(np.int32)
        obj_keep_8cm_preopt = (
            _load_array(data, f"{side}_obj_keep_8cm")[selected].astype(bool)
            if f"{side}_obj_keep_8cm" in data
            else (_load_array(data, f"obj_to_{side}_hand_dist")[selected].astype(np.float32) <= 0.08)
        )
        hand_point_id = _load_array(data, f"{side}_hand_point_id").astype(np.int32) if f"{side}_hand_point_id" in data else None
        hand_cano_points = _load_array(data, f"{side}_hand_cano_points").astype(np.float32) if f"{side}_hand_cano_points" in data else None
        hand_finger_id = _load_array(data, f"{side}_hand_finger_id").astype(np.int32) if f"{side}_hand_finger_id" in data else None
        hand_region_id = _load_array(data, f"{side}_hand_region_id").astype(np.int32) if f"{side}_hand_region_id" in data else None

    mano_dir = (
        Path(str(payload["config"]["mano_dir"])).resolve()
        if "config" in payload and payload["config"].get("mano_dir")
        else (REF2DEX_ROOT / "assets" / "shared" / "mano").resolve()
    )
    faces = _load_mano_faces(mano_dir)
    opt_verts_world = _load_array(payload, "opt_hand_verts", "opt_hand_verts_world").astype(np.float32)
    opt_hand_face_centers_world = (
        _load_array(payload, "opt_hand_face_centers_world", "opt_hand_face_centers").astype(np.float32)
        if ("opt_hand_face_centers_world" in payload or "opt_hand_face_centers" in payload)
        else _face_centers_from_verts(opt_verts_world, faces)
    )
    opt_hand_normals_world = (
        _load_array(payload, "opt_hand_normals_world").astype(np.float32)
        if "opt_hand_normals_world" in payload
        else _face_normals_from_verts(opt_verts_world, faces)
    )
    _, _, obj_to_hand_idx_full_post, obj_to_hand_dist_full_post = _compute_bidirectional_nn_per_frame(
        opt_hand_face_centers_world,
        obj_points_world_full,
    )
    crop = _pack_object_crop_from_full(
        obj_points_world_full=obj_points_world_full,
        obj_normals_world_full=obj_normals_world_full,
        obj_point_id_full=obj_point_id,
        obj_root_pose=obj_root_pose,
        obj_keep_8cm_preopt_full=obj_keep_8cm_preopt,
        obj_keep_8cm_postopt_full=(obj_to_hand_dist_full_post <= 0.08),
        obj_to_hand_idx_full=obj_to_hand_idx_full_post,
        obj_to_hand_dist_full=obj_to_hand_dist_full_post,
        opt_hand_face_centers_world=opt_hand_face_centers_world,
        opt_hand_normals_world=opt_hand_normals_world,
    )
    crop["obj_root_pose"] = obj_root_pose
    crop["preprocess_frame_idx"] = preprocess_frame_idx
    if hand_point_id is not None:
        crop["hand_point_id"] = hand_point_id
    if hand_cano_points is not None:
        crop["hand_cano_points"] = hand_cano_points
    if hand_finger_id is not None:
        crop["hand_finger_id"] = hand_finger_id
    if hand_region_id is not None:
        crop["hand_region_id"] = hand_region_id
    return crop


def _soft_contact_label(dist: np.ndarray, d_pos: float, d_neg: float, gamma: float) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float32)
    out = np.zeros_like(dist, dtype=np.float32)
    pos_mask = dist <= float(d_pos)
    mid_mask = (dist > float(d_pos)) & (dist < float(d_neg))
    out[pos_mask] = 1.0
    if np.any(mid_mask):
        alpha = 1.0 - (dist[mid_mask] - float(d_pos)) / max(float(d_neg) - float(d_pos), 1e-8)
        out[mid_mask] = np.power(np.clip(alpha, 0.0, 1.0), float(gamma))
    return out.astype(np.float32)


def _farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    n = int(points.shape[0])
    k = min(int(k), n)
    if k <= 0:
        return np.zeros((0,), dtype=np.int32)
    if k >= n:
        return np.arange(n, dtype=np.int32)

    selected = np.zeros((k,), dtype=np.int32)
    centroid = points.mean(axis=0, keepdims=True)
    dist_to_centroid = np.linalg.norm(points - centroid, axis=1)
    selected[0] = int(np.argmax(dist_to_centroid))
    min_dist = np.linalg.norm(points - points[selected[0]], axis=1)
    for i in range(1, k):
        selected[i] = int(np.argmax(min_dist))
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[selected[i]], axis=1))
    return selected


def _balanced_obj_sample(
    points_obj: np.ndarray,
    valid_mask: np.ndarray,
    near_mask: np.ndarray,
    num_obj_points: int,
    num_near_points: int,
) -> np.ndarray:
    valid_idx = np.flatnonzero(valid_mask).astype(np.int32)
    if valid_idx.size == 0:
        return np.zeros((0,), dtype=np.int32)

    near_idx = np.flatnonzero(valid_mask & near_mask).astype(np.int32)
    near_take = min(int(num_near_points), int(num_obj_points), int(near_idx.size))
    near_sel = near_idx[_farthest_point_sample(points_obj[near_idx], near_take)] if near_take > 0 else np.zeros((0,), dtype=np.int32)

    remaining = int(num_obj_points) - int(near_sel.size)
    if remaining <= 0:
        return near_sel.astype(np.int32)

    keep_mask = np.ones((valid_idx.size,), dtype=bool)
    if near_sel.size > 0:
        keep_mask[np.isin(valid_idx, near_sel)] = False
    context_idx = valid_idx[keep_mask]
    context_take = min(int(remaining), int(context_idx.size))
    context_sel = (
        context_idx[_farthest_point_sample(points_obj[context_idx], context_take)]
        if context_take > 0 else np.zeros((0,), dtype=np.int32)
    )
    return np.concatenate([near_sel, context_sel], axis=0).astype(np.int32)


def _compute_obj_to_hand_from_geometry(
    obj_points_obj_crop: np.ndarray,
    obj_crop_valid_mask: np.ndarray,
    hand_points_obj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    T, No_crop, _ = obj_points_obj_crop.shape
    nn_id = np.full((T, No_crop), -1, dtype=np.int32)
    nn_dist = np.full((T, No_crop), np.inf, dtype=np.float32)
    for t in range(T):
        valid_idx = np.flatnonzero(obj_crop_valid_mask[t]).astype(np.int32)
        if valid_idx.size == 0:
            continue
        tree = cKDTree(hand_points_obj[t])
        dist_t, idx_t = tree.query(obj_points_obj_crop[t, valid_idx], k=1)
        nn_id[t, valid_idx] = idx_t.astype(np.int32)
        nn_dist[t, valid_idx] = dist_t.astype(np.float32)
    return nn_id, nn_dist


def _mirror_x(x: np.ndarray) -> np.ndarray:
    y = np.asarray(x).copy()
    if y.shape[-1] == 3:
        y[..., 0] *= -1.0
    return y


def _resolve_pkl_files(mano_opt_root: Path, seq_id: str | None, side: str | None) -> List[Path]:
    if seq_id is not None:
        subject_id, seq_name = seq_id.split("/", 1)
        if side is None:
            candidates = [
                mano_opt_root / subject_id / f"{seq_name}_right.pkl",
                mano_opt_root / subject_id / f"{seq_name}_left.pkl",
            ]
            return [p for p in candidates if p.exists()]
        p = mano_opt_root / subject_id / f"{seq_name}_{side}.pkl"
        return [p] if p.exists() else []

    pkls = sorted(mano_opt_root.glob("*/*.pkl"))
    if side is None:
        return pkls
    return [p for p in pkls if p.stem.endswith(f"_{side}")]


def build_train_sequence(payload: Dict, pkl_path: Path, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    seq_id = str(payload["seq_id"])
    dataset_name = str(payload.get("dataset_name", pkl_path.parent.parent.name.split("_")[0]))
    subject_id = str(payload.get("subject_id", seq_id.split("/", 1)[0]))
    seq_name = str(payload.get("seq_name", seq_id.split("/", 1)[1]))
    object_name = str(payload.get("object_name", seq_name.split("_")[0]))
    side = str(payload["side"])

    raw_frame_id = _load_array(payload, "raw_frame_id", "frame_ids").astype(np.int32)
    opt_frame_idx = np.arange(raw_frame_id.shape[0], dtype=np.int32)
    legacy_crop = None
    if (
        "obj_points_obj_crop" not in payload
        and "obj_points_world_crop" not in payload
    ):
        legacy_crop = _load_legacy_stage2_crop(payload, pkl_path, raw_frame_id, side)

    preprocess_frame_idx = (
        _load_array(payload, "preprocess_frame_idx").astype(np.int32)
        if "preprocess_frame_idx" in payload
        else (
            np.asarray(legacy_crop["preprocess_frame_idx"], dtype=np.int32)
            if legacy_crop is not None and "preprocess_frame_idx" in legacy_crop
            else np.arange(raw_frame_id.shape[0], dtype=np.int32)
        )
    )
    obj_root_pose = (
        _load_array(payload, "obj_root_pose", "object_trajectory").astype(np.float32)
        if ("obj_root_pose" in payload or "object_trajectory" in payload)
        else np.asarray(legacy_crop["obj_root_pose"], dtype=np.float32)
    )

    obj_points_obj_crop = (
        _load_array(payload, "obj_points_obj_crop").astype(np.float32)
        if "obj_points_obj_crop" in payload
        else (
            _points_world_to_obj(_load_array(payload, "obj_points_world_crop"), obj_root_pose)
            if "obj_points_world_crop" in payload
            else np.asarray(legacy_crop["obj_points_obj_crop"], dtype=np.float32)
        )
    )
    obj_normals_obj_crop = (
        _load_array(payload, "obj_normals_obj_crop").astype(np.float32)
        if "obj_normals_obj_crop" in payload
        else (
            _normals_world_to_obj(_load_array(payload, "obj_normals_world_crop"), obj_root_pose)
            if "obj_normals_world_crop" in payload
            else np.asarray(legacy_crop["obj_normals_obj_crop"], dtype=np.float32)
        )
    )
    obj_crop_valid_mask = (
        _load_array(payload, "obj_crop_valid_mask").astype(bool)
        if "obj_crop_valid_mask" in payload
        else (
            np.asarray(legacy_crop["obj_crop_valid_mask"], dtype=bool)
            if legacy_crop is not None
            else np.ones(obj_points_obj_crop.shape[:2], dtype=bool)
        )
    )
    obj_crop_raw_idx = (
        _load_array(payload, "obj_crop_raw_idx").astype(np.int32)
        if "obj_crop_raw_idx" in payload
        else (
            np.asarray(legacy_crop["obj_crop_raw_idx"], dtype=np.int32)
            if legacy_crop is not None
            else np.where(
                obj_crop_valid_mask,
                np.tile(np.arange(obj_points_obj_crop.shape[1], dtype=np.int32)[None, :], (obj_points_obj_crop.shape[0], 1)),
                -1,
            )
        )
    )
    obj_crop_point_id = (
        _load_array(payload, "obj_crop_point_id").astype(np.int32)
        if "obj_crop_point_id" in payload
        else obj_crop_raw_idx.copy()
    )
    obj_contact_region_3cm = (
        _load_array(payload, "obj_contact_region_3cm").astype(bool)
        if "obj_contact_region_3cm" in payload
        else (
            _load_array(payload, "obj_to_hand_dist").astype(np.float32) <= 0.03
            if "obj_to_hand_dist" in payload
            else np.zeros(obj_crop_valid_mask.shape, dtype=bool)
        )
    )

    if "opt_hand_face_centers_obj" in payload:
        hand_points_obj = _load_array(payload, "opt_hand_face_centers_obj").astype(np.float32)
    else:
        hand_points_world = _load_array(payload, "opt_hand_face_centers_world", "opt_hand_face_centers").astype(np.float32)
        hand_points_obj = _points_world_to_obj(hand_points_world, obj_root_pose)

    if "opt_hand_normals_obj" in payload:
        hand_normals_obj = _load_array(payload, "opt_hand_normals_obj").astype(np.float32)
    elif "opt_hand_normals_world" in payload:
        hand_normals_obj = _normals_world_to_obj(_load_array(payload, "opt_hand_normals_world"), obj_root_pose)
    else:
        hand_normals_obj = np.zeros_like(hand_points_obj, dtype=np.float32)

    hand_point_id = (
        _load_array(payload, "hand_point_id").astype(np.int32)
        if "hand_point_id" in payload
        else (
            np.asarray(legacy_crop["hand_point_id"], dtype=np.int32)
            if legacy_crop is not None and "hand_point_id" in legacy_crop
            else np.arange(hand_points_obj.shape[1], dtype=np.int32)
        )
    )
    if int(hand_points_obj.shape[1]) != int(args.num_hand_points):
        raise ValueError(
            f"Expected {args.num_hand_points} hand points, got {hand_points_obj.shape[1]} in {pkl_path}"
        )
    hand_cano_points = (
        _load_array(payload, "hand_cano_points").astype(np.float32)
        if "hand_cano_points" in payload
        else (
            np.asarray(legacy_crop["hand_cano_points"], dtype=np.float32)
            if legacy_crop is not None and "hand_cano_points" in legacy_crop
            else np.zeros((hand_points_obj.shape[1], 3), dtype=np.float32)
        )
    )
    hand_finger_id = (
        _load_array(payload, "hand_finger_id").astype(np.int32)
        if "hand_finger_id" in payload
        else (
            np.asarray(legacy_crop["hand_finger_id"], dtype=np.int32)
            if legacy_crop is not None and "hand_finger_id" in legacy_crop
            else np.zeros((hand_points_obj.shape[1],), dtype=np.int32)
        )
    )
    hand_region_id = (
        _load_array(payload, "hand_region_id").astype(np.int32)
        if "hand_region_id" in payload
        else (
            np.asarray(legacy_crop["hand_region_id"], dtype=np.int32)
            if legacy_crop is not None and "hand_region_id" in legacy_crop
            else np.zeros((hand_points_obj.shape[1],), dtype=np.int32)
        )
    )

    if args.mirror_left_to_right and side == "left":
        obj_points_obj_crop = _mirror_x(obj_points_obj_crop)
        obj_normals_obj_crop = _mirror_x(obj_normals_obj_crop)
        hand_points_obj = _mirror_x(hand_points_obj)
        hand_normals_obj = _mirror_x(hand_normals_obj)
        hand_cano_points = _mirror_x(hand_cano_points)

    if "obj_to_hand_nn_id" in payload and "obj_to_hand_dist" in payload:
        obj_to_hand_nn_id_crop = _load_array(payload, "obj_to_hand_nn_id").astype(np.int32)
        obj_to_hand_dist_crop = _load_array(payload, "obj_to_hand_dist").astype(np.float32)
    elif legacy_crop is not None:
        obj_to_hand_nn_id_crop = np.asarray(legacy_crop["obj_to_hand_nn_id"], dtype=np.int32)
        obj_to_hand_dist_crop = np.asarray(legacy_crop["obj_to_hand_dist"], dtype=np.float32)
    else:
        obj_to_hand_nn_id_crop, obj_to_hand_dist_crop = _compute_obj_to_hand_from_geometry(
            obj_points_obj_crop,
            obj_crop_valid_mask,
            hand_points_obj,
        )

    T = int(raw_frame_id.shape[0])
    No_train = int(args.num_obj_points)
    Nh_train = int(hand_points_obj.shape[1])
    N_total = No_train + Nh_train
    sample_id = np.asarray(
        [f"{subject_id}__{seq_name}__{side}__{int(fid):06d}" for fid in raw_frame_id],
        dtype=f"<U{max(32, len(subject_id) + len(seq_name) + 32)}",
    )
    points = np.zeros((T, N_total, 3), dtype=np.float32)
    normals = np.zeros((T, N_total, 3), dtype=np.float32)
    point_valid_mask = np.zeros((T, N_total), dtype=bool)
    source_point_id = np.full((T, N_total), -1, dtype=np.int32)
    sampled_obj_raw_idx = np.full((T, No_train), -1, dtype=np.int32)
    sampled_obj_point_id = np.full((T, No_train), -1, dtype=np.int32)
    obj_contact_label = np.zeros((T, No_train), dtype=np.float32)
    obj_to_hand_nn_id = np.full((T, No_train), -1, dtype=np.int32)
    obj_to_hand_point_id = np.full((T, No_train), -1, dtype=np.int32)
    obj_to_hand_cano_points = np.zeros((T, No_train, 3), dtype=np.float32)
    obj_to_hand_finger_id = np.full((T, No_train), -1, dtype=np.int32)
    obj_to_hand_region_id = np.full((T, No_train), -1, dtype=np.int32)
    obj_label_valid_mask = np.zeros((T, No_train), dtype=bool)
    obj_corr_valid_mask = np.zeros((T, No_train), dtype=bool)

    point_type_id = np.concatenate(
        [
            np.zeros((No_train,), dtype=np.int32),
            np.ones((Nh_train,), dtype=np.int32),
        ],
        axis=0,
    )
    finger_id = np.concatenate(
        [
            -np.ones((No_train,), dtype=np.int32),
            hand_finger_id.astype(np.int32),
        ],
        axis=0,
    )
    unified_hand_region_id = np.concatenate(
        [
            -np.ones((No_train,), dtype=np.int32),
            hand_region_id.astype(np.int32),
        ],
        axis=0,
    )
    unified_hand_cano_points = np.zeros((N_total, 3), dtype=np.float32)
    unified_hand_cano_points[No_train:] = hand_cano_points

    for t in range(T):
        select_idx = _balanced_obj_sample(
            points_obj=obj_points_obj_crop[t],
            valid_mask=obj_crop_valid_mask[t],
            near_mask=obj_contact_region_3cm[t] & obj_crop_valid_mask[t],
            num_obj_points=No_train,
            num_near_points=args.num_near_points,
        )
        valid_count = int(min(select_idx.size, No_train))
        if valid_count > 0:
            obj_sel = select_idx[:valid_count]
            sampled_obj_raw_idx[t, :valid_count] = obj_crop_raw_idx[t, obj_sel]
            sampled_obj_point_id[t, :valid_count] = obj_crop_point_id[t, obj_sel]
            points[t, :valid_count] = obj_points_obj_crop[t, obj_sel]
            normals[t, :valid_count] = obj_normals_obj_crop[t, obj_sel]
            point_valid_mask[t, :valid_count] = True
            source_point_id[t, :valid_count] = obj_crop_point_id[t, obj_sel]
            obj_label_valid_mask[t, :valid_count] = True
            obj_nn_t = obj_to_hand_nn_id_crop[t, obj_sel]
            obj_dist_t = obj_to_hand_dist_crop[t, obj_sel]
            obj_to_hand_nn_id[t, :valid_count] = obj_nn_t
            obj_contact_label[t, :valid_count] = _soft_contact_label(
                obj_dist_t,
                d_pos=args.d_pos,
                d_neg=args.d_neg,
                gamma=args.gamma,
            )

            valid_obj_nn = obj_nn_t >= 0
            obj_corr_valid_mask[t, :valid_count] = valid_obj_nn & (obj_dist_t < float(args.d_neg))
            if np.any(valid_obj_nn):
                obj_to_hand_point_id[t, :valid_count][valid_obj_nn] = hand_point_id[obj_nn_t[valid_obj_nn]]
                obj_to_hand_cano_points[t, :valid_count][valid_obj_nn] = hand_cano_points[obj_nn_t[valid_obj_nn]]
                obj_to_hand_finger_id[t, :valid_count][valid_obj_nn] = hand_finger_id[obj_nn_t[valid_obj_nn]]
                obj_to_hand_region_id[t, :valid_count][valid_obj_nn] = hand_region_id[obj_nn_t[valid_obj_nn]]

        points[t, No_train:] = hand_points_obj[t]
        normals[t, No_train:] = hand_normals_obj[t]
        point_valid_mask[t, No_train:] = True
        source_point_id[t, No_train:] = hand_point_id

    return {
        "sample_id": sample_id,
        "seq_id": np.asarray(seq_id),
        "dataset_name": np.asarray(dataset_name),
        "subject_id": np.asarray(subject_id),
        "seq_name": np.asarray(seq_name),
        "object_name": np.asarray(object_name),
        "side": np.asarray(side),
        "raw_frame_id": raw_frame_id.astype(np.int32),
        "preprocess_frame_idx": preprocess_frame_idx.astype(np.int32),
        "opt_frame_idx": opt_frame_idx.astype(np.int32),
        "source_mano_opt_file": np.asarray(str(pkl_path)),
        "points": points,
        "normals": normals,
        "point_type_id": point_type_id,
        "point_valid_mask": point_valid_mask,
        "source_point_id": source_point_id,
        "num_obj_points": np.asarray(No_train, dtype=np.int32),
        "num_hand_points": np.asarray(Nh_train, dtype=np.int32),
        "obj_point_start": np.asarray(0, dtype=np.int32),
        "obj_point_count": np.asarray(No_train, dtype=np.int32),
        "hand_point_start": np.asarray(No_train, dtype=np.int32),
        "hand_point_count": np.asarray(Nh_train, dtype=np.int32),
        "sampled_obj_raw_idx": sampled_obj_raw_idx,
        "sampled_obj_point_id": sampled_obj_point_id,
        "finger_id": finger_id,
        "hand_region_id": unified_hand_region_id,
        "hand_cano_points": unified_hand_cano_points,
        "obj_contact_label": obj_contact_label,
        "obj_to_hand_nn_id": obj_to_hand_nn_id,
        "obj_to_hand_point_id": obj_to_hand_point_id,
        "obj_to_hand_cano_points": obj_to_hand_cano_points,
        "obj_to_hand_finger_id": obj_to_hand_finger_id,
        "obj_to_hand_region_id": obj_to_hand_region_id,
        "obj_label_valid_mask": obj_label_valid_mask,
        "obj_corr_valid_mask": obj_corr_valid_mask,
        "T_world_from_obj": obj_root_pose.astype(np.float32),
    }


def _write_train_meta(output_root: Path, dataset_name: str, args: argparse.Namespace, stats: Dict[str, float]) -> None:
    dataset_root = output_root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "schema_name": "train_corr_static",
        "schema_version": "0.3.0",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_mano_opt_root": str(Path(args.mano_opt_root).resolve()),
        "output_root": str(dataset_root.resolve()),
        "dataset_name": dataset_name,
        "sample_unit": "single_frame_single_hand",
        "side_policy": {
            "single_hand": True,
            "preserve_side_metadata": True,
            "use_side_as_network_input": False,
            "use_side_in_labels": False,
            "mirror_left_to_right": bool(args.mirror_left_to_right),
        },
        "coordinate_policy": {
            "network_input_frame": "object",
            "use_world_coords_as_input": False,
            "keep_T_world_from_obj_for_debug": True,
            "debug_transform_field": "T_world_from_obj",
            "normal_frame": "object",
        },
        "point_counts": {
            "num_obj_points_train": int(args.num_obj_points),
            "num_hand_points_train": int(args.num_hand_points),
            "num_total_points": int(args.num_obj_points + args.num_hand_points),
        },
        "object_sampling": {
            "source": "stage2_obj_crop_8cm",
            "obj_crop_thresh": 0.08,
            "num_obj_points_train": int(args.num_obj_points),
            "sampling_strategy": "near_context_balanced",
            "near_region_thresh": 0.03,
            "num_near_points": int(args.num_near_points),
            "num_context_points": int(args.num_obj_points - args.num_near_points),
            "pad_if_insufficient": True,
            "padding_value_float": 0.0,
            "padding_value_int": -1,
        },
        "contact_label": {
            "label_type": "soft_distance_decay",
            "distance_source": "postopt_obj_to_hand_dist",
            "store_distance_in_train_file": False,
            "d_pos": float(args.d_pos),
            "d_neg": float(args.d_neg),
            "gamma": float(args.gamma),
            "formula": "1 if d<=d_pos else ((1-(d-d_pos)/(d_neg-d_pos))**gamma) if d<d_neg else 0",
            "use_1cm_spring_mask_as_label": False,
        },
        "toch_correspondence": {
            "enabled": True,
            "object_centric": True,
            "target": "nearest_hand_canonical_point",
            "canonical_hand_field": "obj_to_hand_cano_points",
            "nearest_hand_id_field": "obj_to_hand_nn_id",
            "valid_radius": float(args.d_neg),
            "loss_valid_mask": "obj_corr_valid_mask",
        },
        "runtime_knn": {
            "enabled": True,
            "object_centric": True,
            "cross_direction": "obj_to_hand",
            "compute_stage": "dataloader_or_forward",
            "store_knn_in_train_file": False,
            "store_knn_in_mano_opt": False,
            "store_edge_geometry": False,
            "recompute_after_hand_perturb": True,
            "recompute_after_global_augment": True,
            "current_runtime_field": "obj_to_hand_knn_idx",
        },
        "stats": stats,
    }
    with (dataset_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Stage 3 static train correspondence samples.")
    parser.add_argument("--mano-opt-root", type=str, default=str(DEFAULT_MANO_OPT_ROOT))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seq-id", type=str, default=None)
    parser.add_argument("--side", type=str, default=None, choices=["left", "right"])
    parser.add_argument("--num-obj-points", type=int, default=512)
    parser.add_argument("--num-hand-points", type=int, default=1538)
    parser.add_argument("--num-near-points", type=int, default=256)
    parser.add_argument(
        "--k-obj-local",
        type=int,
        default=16,
        help="Deprecated compatibility arg. Local KNN is no longer stored in Stage 3 files.",
    )
    parser.add_argument(
        "--k-hand-local",
        type=int,
        default=16,
        help="Deprecated compatibility arg. Local KNN is no longer stored in Stage 3 files.",
    )
    parser.add_argument(
        "--k-cross",
        type=int,
        default=32,
        help="Deprecated compatibility arg. Cross KNN is recomputed at runtime and not stored in Stage 3 files.",
    )
    parser.add_argument("--d-pos", type=float, default=0.005)
    parser.add_argument("--d-neg", type=float, default=0.03)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--mirror-left-to-right", action="store_true", default=False)
    parser.add_argument("--save-compressed", action="store_true", default=False)
    args = parser.parse_args()

    mano_opt_root = Path(args.mano_opt_root).resolve()
    output_root = Path(args.output_root).resolve()
    pkl_files = _resolve_pkl_files(mano_opt_root, args.seq_id, args.side)
    if not pkl_files:
        raise FileNotFoundError(f"No Stage 2 pkl files found under {mano_opt_root}")

    stats_by_dataset: Dict[str, Dict[str, float]] = {}
    save_fn = np.savez_compressed if args.save_compressed else np.savez

    for pkl_path in pkl_files:
        with pkl_path.open("rb") as f:
            payload = pickle.load(f)
        train_seq = build_train_sequence(payload, pkl_path, args)
        dataset_name = str(np.asarray(train_seq["dataset_name"]).item())
        subject_id = str(np.asarray(train_seq["subject_id"]).item())
        seq_name = str(np.asarray(train_seq["seq_name"]).item())
        side = str(np.asarray(train_seq["side"]).item())

        out_dir = output_root / dataset_name / subject_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{seq_name}_{side}.npz"
        save_fn(out_path, **train_seq)

        source_obj_valid = (
            np.asarray(payload["obj_crop_valid_mask"], dtype=bool)
            if "obj_crop_valid_mask" in payload
            else np.asarray(train_seq["point_valid_mask"])[:, : int(np.asarray(train_seq["num_obj_points"]).item())]
        )
        obj_valid = np.asarray(train_seq["point_valid_mask"])[:, : int(np.asarray(train_seq["num_obj_points"]).item())]
        num_samples = int(np.asarray(train_seq["raw_frame_id"]).shape[0])
        avg_obj_before_sampling = float(source_obj_valid.sum(axis=1).mean()) if num_samples > 0 else 0.0
        avg_padding = float((~obj_valid).sum(axis=1).mean()) if num_samples > 0 else 0.0

        stats = stats_by_dataset.setdefault(
            dataset_name,
            {
                "num_source_hand_sequences": 0,
                "num_train_samples": 0,
                "num_right_samples": 0,
                "num_left_samples": 0,
                "avg_obj_points_before_train_sampling_sum": 0.0,
                "avg_obj_padding_points_sum": 0.0,
            },
        )
        stats["num_source_hand_sequences"] += 1
        stats["num_train_samples"] += num_samples
        stats[f"num_{side}_samples"] += num_samples
        stats["avg_obj_points_before_train_sampling_sum"] += avg_obj_before_sampling * num_samples
        stats["avg_obj_padding_points_sum"] += avg_padding * num_samples
        print(f"[train-corr] saved {out_path} samples={num_samples}")

    for dataset_name, stats in stats_by_dataset.items():
        total_samples = max(int(stats["num_train_samples"]), 1)
        meta_stats = {
            "num_source_hand_sequences": int(stats["num_source_hand_sequences"]),
            "num_train_samples": int(stats["num_train_samples"]),
            "num_right_samples": int(stats["num_right_samples"]),
            "num_left_samples": int(stats["num_left_samples"]),
            "avg_obj_points_before_train_sampling": float(stats["avg_obj_points_before_train_sampling_sum"] / total_samples),
            "avg_obj_padding_points": float(stats["avg_obj_padding_points_sum"] / total_samples),
        }
        _write_train_meta(output_root, dataset_name, args, meta_stats)


if __name__ == "__main__":
    main()
