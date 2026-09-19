#!/usr/bin/env python3
"""Build a fingerprinted CmDecoderv2 view over the immutable OICM cache.

Only RL-Inspire train/val sequences receive q/wrist GT. MANO test entries are
kept as source-demonstration geometry for qualitative receding-horizon use.
The tool never edits or copies the OICM geometry cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.task.CmDecoderv2.kinematics import (
    INDEPENDENT_FINGER_NATIVE_INDICES,
    NATIVE_Q_START,
    NATIVE_TO_URDF,
    NUM_NATIVE_DOFS,
    InspireKinematics,
    extract_finger_q,
)


SCHEMA_NAME = "ref2dex_cm_decoder_v2_dexplore_view_v1"
DEFAULT_MODIFICATION_VERSION = "V1.1.1"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_info(repo: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    return commit, dirty


def _resolve(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def _load_native_q(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(payload) or payload.ndim != 2 or payload.shape[1] < NATIVE_Q_START + NUM_NATIVE_DOFS:
        raise ValueError(f"Expected Dexplore tensor [T,>={NATIVE_Q_START + NUM_NATIVE_DOFS}], got {type(payload)} {getattr(payload, 'shape', None)}")
    q = payload[:, NATIVE_Q_START:NATIVE_Q_START + NUM_NATIVE_DOFS].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(q).all():
        raise ValueError(f"Non-finite q values in {path}")
    return q


def _select_entries(source: dict[str, Any], mode: str, max_sequences: int | None) -> dict[str, list[dict[str, Any]]]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        variant = "inspire_rl" if split in {"train", "val"} else "mano"
        values = [dict(item) for item in source["sequences"][split] if item.get("variant") == variant]
        if mode == "pilot":
            values = values[: max_sequences or 1]
        elif max_sequences is not None:
            values = values[:max_sequences]
        if not values:
            raise ValueError(f"No {variant} entries for split {split}")
        selected[split] = values
    return selected


def _prepare_output(path: Path, resume: bool) -> None:
    if path.exists() and any(path.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite non-empty view: {path}; pass --resume to validate/reuse entries")
    path.mkdir(parents=True, exist_ok=True)


def build(args: argparse.Namespace) -> Path:
    if int(args.window_size) <= 0:
        raise ValueError("window_size must be positive")
    repo = Path(__file__).resolve().parents[5]
    source_index = _resolve(repo, args.index)
    output_root = _resolve(repo, args.output)
    urdf_path = _resolve(repo, args.urdf)
    work_version = str(args.work_version)
    _prepare_output(output_root, args.resume)
    source = json.loads(source_index.read_text(encoding="utf-8"))
    source_schema = source.get("schema_name")
    if source_schema not in {
        "ref2dex_object_interaction_cm_index_v1_1",
        "ref2dex_object_interaction_cm_index_v1_2",
    }:
        raise ValueError(f"Unsupported source index schema {source.get('schema_name')!r}")
    selected = _select_entries(source, args.mode, args.max_sequences)
    kinematics = InspireKinematics(urdf_path)
    implementation_path = Path(__file__).resolve()
    kinematics_path = (repo / "src/task/CmDecoderv2/kinematics.py").resolve()
    output_entries: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    converted = 0
    for split, entries in selected.items():
        for item in entries:
            geometry_root = (source_index.parent / item["path"] / "geometry").resolve()
            geometry_manifest_path = geometry_root / "manifest.json"
            geometry_manifest = json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
            frames = int(item["frame_count"])
            result = {
                **item,
                "geometry_root": str(geometry_root),
                "geometry_manifest_sha256": _sha256(geometry_manifest_path),
            }
            if split in {"train", "val"}:
                if item.get("variant") != "inspire_rl" or not str(
                    geometry_manifest.get("source_type", "")
                ).startswith("dexplore_rl_native_q"):
                    raise ValueError(f"Decoder supervision must be Dexplore RL: {item}")
                q_path = Path(geometry_manifest["rl_q"]["tensor"]).resolve()
                q_native = _load_native_q(q_path)
                if len(q_native) != frames:
                    raise ValueError(f"Frame/q mismatch for {item['id']}: {frames} != {len(q_native)}")
                wrist = np.stack([kinematics.wrist_pose_from_native(q) for q in q_native]).astype(np.float32)
                object_pose = np.load(geometry_root / "obj_pose_world.npy", mmap_mode="r")
                source_frame = np.asarray(np.load(geometry_root / "source_frame_id.npy", mmap_mode="r"), dtype=np.int64)
                if len(object_pose) != frames or len(source_frame) != frames:
                    raise ValueError(f"Object/source-frame mismatch for {item['id']}")
                expected_source_stride = int(round(float(geometry_manifest["source_fps"]) / float(geometry_manifest["effective_fps"])))
                if len(source_frame) > 1 and not np.all(np.diff(source_frame) == expected_source_stride):
                    raise ValueError(f"Non-continuous 30 Hz source frames for {item['id']}")
                finger_q = extract_finger_q(q_native).astype(np.float32)
                wrist_object = np.stack([np.linalg.inv(object_pose[frame]) @ wrist[frame] for frame in range(frames)]).astype(np.float32)
                window_count = max(0, frames - int(args.window_size))
                window_starts = np.arange(window_count, dtype=np.int32)
                offsets = np.arange(int(args.window_size) + 1, dtype=np.int32)
                window_cache_frames = window_starts[:, None] + offsets[None, :]
                window_source_frames = source_frame[window_cache_frames]
                sidecar = output_root / "sequences" / split / str(item["path"]).split("/")[-1]
                sidecar.mkdir(parents=True, exist_ok=True)
                q_out, wrist_out = sidecar / "q_native.npy", sidecar / "wrist_pose_world.npy"
                finger_out = sidecar / "q_finger_real.npy"
                wrist_object_out = sidecar / "wrist_pose_object_t.npy"
                window_cache_out = sidecar / "window_cache_frame_ids.npy"
                window_source_out = sidecar / "window_source_frame_ids.npy"
                if not args.resume or not q_out.is_file() or not wrist_out.is_file():
                    np.save(q_out, q_native)
                    np.save(wrist_out, wrist)
                np.save(finger_out, finger_q)
                np.save(wrist_object_out, wrist_object)
                np.save(window_cache_out, window_cache_frames)
                np.save(window_source_out, window_source_frames)
                stored_q = np.load(q_out, mmap_mode="r")
                stored_wrist = np.load(wrist_out, mmap_mode="r")
                if stored_q.shape != (frames, 18) or stored_wrist.shape != (frames, 4, 4):
                    raise ValueError(f"Invalid existing sidecar for {item['id']}")
                sequence_manifest = {
                    "schema_name": SCHEMA_NAME,
                    "work_version": work_version,
                    "sequence_id": item["id"],
                    "split": split,
                    "variant": "inspire_rl",
                    "frame_count": frames,
                    "effective_fps": 30.0,
                    "q_native": str(q_out),
                    "q_finger_real": str(finger_out),
                    "wrist_pose_world": str(wrist_out),
                    "wrist_pose_object_t": str(wrist_object_out),
                    "window_cache_frame_ids": str(window_cache_out),
                    "window_source_frame_ids": str(window_source_out),
                    "window_count": int(window_count),
                    "window_size": int(args.window_size),
                    "window_frame_contract": "K Cm transitions use K+1 consecutive cache frames",
                    "cache_stride": 1,
                    "source_frame_stride": expected_source_stride,
                    "source_q_tensor": str(q_path),
                    "source_q_tensor_sha256": _sha256(q_path),
                    "geometry_root": str(geometry_root),
                    "geometry_manifest_sha256": result["geometry_manifest_sha256"],
                    "source_cache_schema": geometry_manifest.get("schema_name"),
                    "point_flow_target_file": (
                        "knn_hand_points_world.npy"
                        if (geometry_root / "knn_hand_points_world.npy").is_file()
                        else "hand_points_world.npy"
                    ),
                    "point_flow_hand_points": int(
                        geometry_manifest.get("knn_hand_points", geometry_manifest.get("decoder_hand_points", 1538))
                    ),
                    "point_flow_sampling": (
                        geometry_manifest.get("surface_sampling", {}).get("method", "legacy_cache")
                    ),
                    "urdf": str(urdf_path),
                    "urdf_sha256": _sha256(urdf_path),
                    "native_q_slice": [NATIVE_Q_START, NATIVE_Q_START + NUM_NATIVE_DOFS],
                    "independent_finger_native_indices": INDEPENDENT_FINGER_NATIVE_INDICES.tolist(),
                    "native_to_urdf": NATIVE_TO_URDF.tolist(),
                    "mimic_mapping": {"7": "1.05*q6", "9": "1.05*q8", "11": "1.05*q10", "13": "1.05*q12", "16": "0.6*q15", "17": "0.8*q15"},
                    "wrist_link": "hand_base_link",
                    "implementation_sha256": {"builder": _sha256(implementation_path), "kinematics": _sha256(kinematics_path)},
                }
                manifest_path = sidecar / "manifest.json"
                manifest_path.write_text(json.dumps(sequence_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                result.update({
                    "q_native": str(q_out),
                    "q_finger_real": str(finger_out),
                    "wrist_pose_world": str(wrist_out),
                    "wrist_pose_object_t": str(wrist_object_out),
                    "window_cache_frame_ids": str(window_cache_out),
                    "window_source_frame_ids": str(window_source_out),
                    "window_count": int(window_count),
                    "view_manifest": str(manifest_path),
                    "source_q_tensor_sha256": sequence_manifest["source_q_tensor_sha256"],
                })
                converted += 1
            else:
                if item.get("variant") != "mano":
                    raise ValueError(f"Qualitative test must be MANO-only: {item}")
                result["qualitative_only"] = True
            output_entries[split].append(result)
    commit, dirty = _git_info(repo)
    index_payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": "1.0.0",
        "work_version": work_version,
        "created_at": _now(),
        "source_index": str(source_index),
        "source_index_sha256": _sha256(source_index),
        "source_index_schema": source_schema,
        "source_cache_schema": source.get("experiment_schema", source_schema),
        "urdf": str(urdf_path),
        "urdf_sha256": _sha256(urdf_path),
        "window_size": int(args.window_size),
        "cache_stride": 1,
        "effective_fps": 30.0,
        "state_perturbation_contract": {
            "smoke_enabled": False,
            "formal_train_enabled": True,
            "distribution": "zero_mean_gaussian_componentwise_clipped",
            "translation_std_m": 0.005,
            "translation_clip_m": 0.015,
            "rotation_std_deg": 5.0,
            "rotation_clip_deg": 15.0,
            "finger_q_std_rad": 0.05,
            "finger_q_clip_rad": 0.15,
        },
        "implementation_sha256": {"builder": _sha256(implementation_path), "kinematics": _sha256(kinematics_path)},
        "split_contract": {"train": "inspire_rl", "val": "inspire_rl", "test": "mano_qualitative_only"},
        "cm_input_contract": {
            "hand_stream_mode": "unique_knn_edges" if source_schema.endswith("v1_2") else "decoder",
            "knn_k": int(source.get("knn_k", 8)),
            "interaction_radius_m": float(source.get("interaction_radius_m", 0.05)),
            "hand_supervision_radius_m": float(source.get("hand_supervision_radius_m", 0.03)),
            "distance_storage": source.get("distance_storage", "runtime"),
        },
        "point_flow_supervision": {
            "target_file": "knn_hand_points_world.npy"
            if source_schema.endswith("v1_2")
            else "hand_points_world.npy",
            "hand_points": int(
                source.get(
                    "knn_hand_points_per_stream",
                    source.get("decoder_hand_points_per_stream", 1538),
                )
                if isinstance(source.get("knn_hand_points_per_stream"), int)
                else source.get("knn_hand_points_per_stream", {}).get("inspire_f1", 1538)
                if isinstance(source.get("knn_hand_points_per_stream"), dict)
                else 1538
            ),
            "all_points": True,
            "mask": "none",
            "coordinate_frame": "object_pose_t",
        },
        "sequences": output_entries,
        "counts": {split: len(values) for split, values in output_entries.items()},
    }
    index_out = output_root / "index.json"
    index_out.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cache_manifest_out = output_root / "manifest.json"
    cache_manifest_out.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_id = f"cmdecoderv2-view-{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "CmDecoderv2",
        "activity_id": args.activity_id,
        "work_version": work_version,
        "operation_category": ["data", "operation"],
        "operation": "build_dexplore_rl_decoder_view",
        "run_id": run_id,
        "run_status": "COMPLETED",
        "created_at": _now(),
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "inputs": {"source_index": str(source_index), "source_index_sha256": _sha256(source_index), "urdf": str(urdf_path), "urdf_sha256": _sha256(urdf_path)},
        "parameters": {"mode": args.mode, "max_sequences": args.max_sequences, "window_size": args.window_size, "cache_stride": 1, "right_hand_only": True, "effective_fps": 30.0, "resume": args.resume, "work_version": work_version},
        "counts": {**index_payload["counts"], "rl_sidecars": converted, "windows": {split: sum(int(item.get("window_count", 0)) for item in output_entries[split]) for split in ("train", "val")}, "discarded_tail_frames": {split: len(output_entries[split]) * int(args.window_size) for split in ("train", "val")}},
        "outputs": {"root": str(output_root), "manifest": str(cache_manifest_out), "index": str(index_out)},
        "conclusion": "SUPPORTED",
    }
    (output_root / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_status": "COMPLETED", "output": str(output_root), "counts": index_payload["counts"]}, ensure_ascii=False))
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--index", default="data/processed_data/object_interaction_cm_dexplore_rl_v1/index.json")
    parser.add_argument("--urdf", default="src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf")
    parser.add_argument("--output", default="data/processed_data/cm_decoder_v2/dexplore_rl_v1_1_pilot")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--work-version", default=DEFAULT_MODIFICATION_VERSION)
    parser.add_argument("--activity-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
