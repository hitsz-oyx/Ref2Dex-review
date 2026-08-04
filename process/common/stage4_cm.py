from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SHARED_SCHEMA_NAME = "ref2dex_cm_sequence_shared"
HAND_SCHEMA_NAME = "ref2dex_cm_sequence_hand"
SCHEMA_VERSION = "3.0.0"


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {value}")
    return device


def points_world_to_current_hand(points_world: np.ndarray, hand_root_pose_t: np.ndarray) -> np.ndarray:
    rotation = hand_root_pose_t[:, :3, :3]
    translation = hand_root_pose_t[:, :3, 3]
    return np.einsum("tji,tpj->tpi", rotation, points_world - translation[:, None]).astype(np.float32)


def compute_current_candidate_mask(
    obj_points_world: np.ndarray,
    hand_points_world: np.ndarray,
    hand_root_pose_world: np.ndarray,
    *,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    obj_local = points_world_to_current_hand(
        np.asarray(obj_points_world, dtype=np.float32),
        np.asarray(hand_root_pose_world, dtype=np.float32),
    )
    hand_local = points_world_to_current_hand(
        np.asarray(hand_points_world, dtype=np.float32),
        np.asarray(hand_root_pose_world, dtype=np.float32),
    )
    num_frames, num_obj, _ = obj_local.shape
    candidate_mask = np.empty((num_frames, num_obj), dtype=bool)
    batch_size = max(1, int(frame_batch_size))
    for start in range(0, num_frames, batch_size):
        end = min(start + batch_size, num_frames)
        obj = torch.from_numpy(obj_local[start:end]).to(device=device, dtype=torch.float32)
        hand = torch.from_numpy(hand_local[start:end]).to(device=device, dtype=torch.float32)
        dist = torch.cdist(obj, hand)
        candidate_mask[start:end] = dist.amin(dim=-1).cpu().numpy() <= float(candidate_threshold)
    return candidate_mask


def build_shared_sequence(
    source: dict[str, Any],
    *,
    source_path: Path,
    source_root: Path,
    ds_rate: int,
    source_fps: float,
) -> dict[str, np.ndarray]:
    source_rel = source_path.resolve().relative_to(source_root.resolve()).as_posix()
    return {
        "schema_name": np.asarray(SHARED_SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION),
        "source_raw_file": np.asarray(source_rel),
        "dataset_name": np.asarray(str(source["dataset_name"])),
        "seq_id": np.asarray(str(source["seq_id"])),
        "subject_id": np.asarray(str(source["subject_id"])),
        "seq_name": np.asarray(str(source["seq_name"])),
        "object_name": np.asarray(str(source["object_name"])),
        "raw_frame_id": np.asarray(source["raw_frame_id"], dtype=np.int32),
        "ds_rate": np.asarray(int(ds_rate), dtype=np.int32),
        "source_fps": np.asarray(float(source_fps), dtype=np.float32),
        "coordinate_frame": np.asarray("world"),
        "obj_points_world": np.asarray(source["obj_points_world"], dtype=np.float32),
        "obj_normals_world": np.asarray(source["obj_normals_world"], dtype=np.float32),
        "obj_point_id": np.asarray(source["obj_point_id"], dtype=np.int32),
    }


def build_hand_sequence(
    source: dict[str, Any],
    *,
    side: str,
    candidate_threshold: float,
    frame_batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    if side not in {"left", "right"}:
        raise ValueError(f"Unsupported side {side!r}")
    root_key = f"{side}_hand_root_pose"
    if root_key not in source:
        raise ValueError(f"{source['seq_id']} has no reconstructed {side}-hand trajectory")
    hand_points_key = f"{side}_hand_points_world"
    hand_normals_key = f"{side}_hand_normals_world"
    hand_point_id_key = f"{side}_hand_point_id"
    hand_cano_key = f"{side}_hand_cano_points"
    finger_id_key = f"{side}_hand_finger_id"
    region_id_key = f"{side}_hand_region_id"
    candidate_mask = compute_current_candidate_mask(
        np.asarray(source["obj_points_world"], dtype=np.float32),
        np.asarray(source[hand_points_key], dtype=np.float32),
        np.asarray(source[root_key], dtype=np.float32),
        candidate_threshold=candidate_threshold,
        frame_batch_size=frame_batch_size,
        device=device,
    )
    return {
        "schema_name": np.asarray(HAND_SCHEMA_NAME),
        "schema_version": np.asarray(SCHEMA_VERSION),
        "side": np.asarray(side),
        "hand_root_pose_world": np.asarray(source[root_key], dtype=np.float32),
        "hand_points_world": np.asarray(source[hand_points_key], dtype=np.float32),
        "hand_normals_world": np.asarray(source[hand_normals_key], dtype=np.float32),
        "hand_point_id": np.asarray(source[hand_point_id_key], dtype=np.int32),
        "hand_cano_points": np.asarray(source[hand_cano_key], dtype=np.float32),
        "hand_finger_id": np.asarray(source[finger_id_key], dtype=np.int32),
        "hand_region_id": np.asarray(source[region_id_key], dtype=np.int32),
        "obj_candidate_mask_5cm": candidate_mask,
    }


def write_meta(
    output_root: Path,
    *,
    source_description: str,
    source_root: Path,
    ds_rate: int,
    source_fps: float,
    candidate_threshold: float,
    stats: dict[str, int],
    extra: dict[str, Any] | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "shared_schema_name": SHARED_SCHEMA_NAME,
        "hand_schema_name": HAND_SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_description,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "sample_unit": "one shared sequence file plus one hand-side file",
        "coordinate_frame": "world; Dataset maps endpoints to hand_root_t",
        "future_object_policy": "not stored as a pair; Dataset creates endpoint flow at runtime",
        "num_obj_pool": 4096,
        "num_hand_points": 1538,
        "ds_rate": int(ds_rate),
        "source_fps": float(source_fps),
        "effective_fps": float(source_fps) / int(ds_rate),
        "candidate_threshold": float(candidate_threshold),
        "stats": stats,
    }
    if extra:
        payload.update(extra)
    with (output_root / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_manifest(output_root: Path) -> None:
    """Write an inventory of all currently complete hand streams in a cache."""
    output_root = output_root.resolve()
    fields = ("relative_path", "dataset", "subject", "sequence", "side", "object", "num_frames", "status")
    rows: list[dict[str, Any]] = []
    for hand_path in sorted(output_root.glob("*/*/*.npz")):
        if hand_path.name not in {"left.npz", "right.npz"}:
            continue
        shared_path = hand_path.parent / "shared.npz"
        if not shared_path.exists():
            continue
        try:
            with np.load(shared_path, allow_pickle=False) as shared, np.load(hand_path, allow_pickle=False) as hand:
                rows.append({
                    "relative_path": hand_path.relative_to(output_root).as_posix(),
                    "dataset": str(np.asarray(shared["dataset_name"]).item()),
                    "subject": str(np.asarray(shared["subject_id"]).item()),
                    "sequence": str(np.asarray(shared["seq_name"]).item()),
                    "side": str(np.asarray(hand["side"]).item()),
                    "object": str(np.asarray(shared["object_name"]).item()),
                    "num_frames": int(np.asarray(hand["hand_points_world"]).shape[0]),
                    "status": "ok",
                })
        except (KeyError, OSError, ValueError):
            continue
    with (output_root / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
