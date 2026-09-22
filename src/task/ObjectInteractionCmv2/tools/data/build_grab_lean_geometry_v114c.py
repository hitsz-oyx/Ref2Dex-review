#!/usr/bin/env python3
"""Build the V1.14c local GRAB lean geometry cache for MANO or Inspire."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from process.GRAB.raw import GRABRawAdapter
from src.task.ObjectInteractionCm.tools.data import build_bilateral_mano_v1_4_cache as mano_cache
from src.task.ObjectInteractionCmv2.multi_domain import (
    LEAN_CACHE_MANIFEST_SCHEMA,
    LEAN_GEOMETRY_SCHEMA,
    V114A_HAND_POINTS,
    V114A_HAND_SAMPLING_CONTRACT,
    fixed_bilateral_hand_indices,
    hand_index_sha256,
)
from src.task.ObjectInteractionCmv2.tools.data.build_stage4_inspire_highres_v1_4 import (
    INSPIRE_PER_SIDE,
    InspireConverter,
    TIP_IDS,
)


WORK_VERSION = "V1.14"
OBJECT_POINTS = 4096
POINTS_PER_SIDE = 2048
RADIUS_M = 0.02
REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class Entry:
    split: str
    relative: Path
    source: Path
    target: Path

    @property
    def sequence_id(self) -> str:
        return f"grab/{self.relative.as_posix()}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entries(split_root: Path, source_root: Path, output_root: Path, variant: str) -> list[Entry]:
    result: list[Entry] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        sides: dict[str, set[str]] = {}
        for raw in (split_root / f"{split}.txt").read_text(encoding="utf-8").splitlines():
            path = Path(raw.strip())
            if not path.parts:
                continue
            if path.parts[0] != "grab" or path.name not in ("left.npz", "right.npz"):
                raise ValueError(f"invalid GRAB split entry: {raw!r}")
            relative = Path(*path.parts[1:-1])
            sides.setdefault(relative.as_posix(), set()).add(path.stem)
        for key in sorted(sides):
            if sides[key] != {"left", "right"}:
                raise ValueError(f"{split}/{key}: split does not contain both hands")
            if key in seen:
                raise ValueError(f"duplicate sequence across splits: {key}")
            seen.add(key)
            relative = Path(key)
            source = source_root / relative
            if not all((source / name).is_file() for name in ("shared.npz", "left.npz", "right.npz")):
                raise FileNotFoundError(source)
            result.append(Entry(split, relative, source,
                                output_root / "sequences" / split / variant / "grab" / relative))
    counts = {split: sum(entry.split == split for entry in result) for split in ("train", "val", "test")}
    if counts != {"train": 1068, "val": 134, "test": 133}:
        raise ValueError(f"frozen GRAB split mismatch: {counts}")
    return result


def _candidate_mask(obj: np.ndarray, hand: np.ndarray, device: torch.device,
                    frame_batch: int, object_chunk: int) -> np.ndarray:
    result = np.zeros((len(obj), OBJECT_POINTS), dtype=bool)
    with torch.inference_mode():
        for first in range(0, len(obj), max(1, frame_batch)):
            last = min(len(obj), first + max(1, frame_batch))
            objects = torch.as_tensor(np.array(obj[first:last], copy=True), device=device)
            hands = torch.as_tensor(np.array(hand[first:last], copy=True), device=device)
            for start in range(0, OBJECT_POINTS, max(32, object_chunk)):
                stop = min(OBJECT_POINTS, start + max(32, object_chunk))
                nearest = torch.cdist(objects[:, start:stop], hands).amin(dim=-1)
                result[first:last, start:stop] = (nearest <= RADIUS_M).cpu().numpy()
    return result


def _common_arrays(shared: Any, geometry: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = np.asarray(shared["obj_points_world"], dtype=np.float32)
    normals = np.asarray(shared["obj_normals_world"], dtype=np.float32)
    ids = np.asarray(shared["raw_frame_id"], dtype=np.int32)
    pose = (np.asarray(shared["obj_pose_world"], dtype=np.float32)
            if "obj_pose_world" in shared.files else
            np.broadcast_to(np.eye(4, dtype=np.float32), (len(obj), 4, 4)).copy())
    if obj.shape[1:] != (OBJECT_POINTS, 3) or normals.shape != obj.shape or pose.shape != (len(obj), 4, 4):
        raise ValueError("invalid local Stage4 object geometry")
    source_fps = float(np.asarray(shared["source_fps"]).item())
    for name, value in (
        ("obj_points_pool_world", obj), ("obj_normals_pool_world", normals),
        ("obj_pose_world", pose), ("source_frame_id", ids),
        ("frame_time", ids.astype(np.float32) / source_fps),
        ("obj_point_id", np.asarray(shared["obj_point_id"], dtype=np.int32)),
    ):
        np.save(geometry / f"{name}.npy", value)
    return obj, normals, ids, pose


def _mano(entry: Entry, geometry: Path, correspondences: dict[str, dict[str, np.ndarray]],
          raw_adapter: GRABRawAdapter, raw_root: Path, device: torch.device,
          geometry_batch: int, frame_batch: int, object_chunk: int) -> int:
    with np.load(entry.source / "shared.npz", allow_pickle=False) as shared:
        obj, _, ids, _ = _common_arrays(shared, geometry)
    raw_path = raw_root / "grab" / entry.relative.with_suffix(".npz")
    raw = raw_adapter.process_sequence(str(raw_path))
    raw_ids = np.asarray(raw["raw_frame_id"], dtype=np.int32)
    selected_frames = np.searchsorted(raw_ids, ids)
    if (np.any(selected_frames >= len(raw_ids))
            or not np.array_equal(raw_ids[selected_frames], ids)):
        raise ValueError(f"{entry.sequence_id}: Stage4 frame IDs are absent from raw reconstruction")
    points_out = np.lib.format.open_memmap(
        geometry / "knn_hand_points_world.npy", mode="w+", dtype=np.float32,
        shape=(len(obj), V114A_HAND_POINTS, 3))
    normals_out = np.lib.format.open_memmap(
        geometry / "knn_hand_normals_world.npy", mode="w+", dtype=np.float32,
        shape=(len(obj), V114A_HAND_POINTS, 3))
    try:
        for side_index, side in enumerate(("left", "right")):
            vertices = np.asarray(raw[f"{side}_hand_mesh_vertices_world"], dtype=np.float32)[selected_frames]
            faces = np.asarray(raw[f"{side}_hand_mesh_faces"], dtype=np.int64)
            expected_faces = np.asarray(correspondences[side]["reference_faces"], dtype=np.int64)
            if not np.array_equal(faces, expected_faces):
                raise ValueError(f"{entry.sequence_id}/{side}: MANO topology mismatch")
            start_out = side_index * POINTS_PER_SIDE
            stop_out = start_out + POINTS_PER_SIDE
            for start in range(0, len(obj), max(1, geometry_batch)):
                stop = min(len(obj), start + max(1, geometry_batch))
                points, normals = mano_cache._sample_surface_chunk(
                    vertices[start:stop], faces, correspondences[side])
                points_out[start:stop, start_out:stop_out] = points
                normals_out[start:stop, start_out:stop_out] = normals
    finally:
        points_out.flush(); normals_out.flush()
        del points_out, normals_out
    hand = np.load(geometry / "knn_hand_points_world.npy", mmap_mode="r")
    np.save(geometry / "obj_candidate_mask_2cm.npy",
            _candidate_mask(obj, hand, device, frame_batch, object_chunk))
    return len(obj)


def _inspire(entry: Entry, geometry: Path, converter: InspireConverter, device: torch.device,
             frame_batch: int, object_chunk: int) -> int:
    with np.load(entry.source / "shared.npz", allow_pickle=False) as shared:
        obj, _, _, pose = _common_arrays(shared, geometry)
    points_out = np.lib.format.open_memmap(
        geometry / "knn_hand_points_world.npy", mode="w+", dtype=np.float32,
        shape=(len(obj), V114A_HAND_POINTS, 3))
    normals_out = np.lib.format.open_memmap(
        geometry / "knn_hand_normals_world.npy", mode="w+", dtype=np.float32,
        shape=(len(obj), V114A_HAND_POINTS, 3))
    mask = np.zeros((len(obj), OBJECT_POINTS), dtype=bool)
    fixed = fixed_bilateral_hand_indices("inspire_f1", POINTS_PER_SIDE, 42)
    try:
        for side_index, side in enumerate(("left", "right")):
            with np.load(entry.source / f"{side}.npz", allow_pickle=False) as value:
                if ("hand_mesh_vertices_world" in value.files
                        and value["hand_mesh_vertices_world"].shape[1] >= 778):
                    vertices = np.asarray(value["hand_mesh_vertices_world"], dtype=np.float32)
                    tips = (vertices - pose[:, None, :3, 3]) @ pose[:, :3, :3]
                    tips = tips[:, TIP_IDS[side]]
                else:
                    canonical = np.asarray(value["hand_cano_points"], dtype=np.float32)
                    labels = np.asarray(value["hand_finger_id"])
                    hand = np.asarray(value["hand_points_world"], dtype=np.float32)
                    center = canonical.mean(axis=0)
                    selected = []
                    for finger in range(1, 6):
                        candidates = np.flatnonzero(labels == finger)
                        if not len(candidates):
                            raise ValueError(f"{entry.sequence_id}/{side}: missing finger {finger}")
                        selected.append(int(candidates[np.linalg.norm(
                            canonical[candidates] - center, axis=1).argmax()]))
                    tips = (hand[:, selected] - pose[:, None, :3, 3]) @ pose[:, :3, :3]
            local_points, local_normals, _ = converter.convert(tips, side)
            world_points = np.einsum("bij,bpj->bpi", pose[:, :3, :3], local_points) + pose[:, None, :3, 3]
            world_normals = np.einsum("bij,bpj->bpi", pose[:, :3, :3], local_normals)
            mask |= _candidate_mask(obj, world_points, device, frame_batch, object_chunk)
            side_indices = fixed[side_index * POINTS_PER_SIDE:(side_index + 1) * POINTS_PER_SIDE] - side_index * INSPIRE_PER_SIDE
            start = side_index * POINTS_PER_SIDE
            stop = start + POINTS_PER_SIDE
            points_out[:, start:stop] = world_points[:, side_indices]
            normals_out[:, start:stop] = world_normals[:, side_indices]
    finally:
        points_out.flush(); normals_out.flush()
        del points_out, normals_out
    np.save(geometry / "obj_candidate_mask_2cm.npy", mask)
    return len(obj)


def _validate(entry: Entry, variant: str) -> dict[str, Any]:
    geometry = entry.target / "geometry"
    manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8"))
    frames = int(manifest["frame_count"])
    expected = {
        "obj_points_pool_world.npy": ((frames, OBJECT_POINTS, 3), np.float32),
        "obj_normals_pool_world.npy": ((frames, OBJECT_POINTS, 3), np.float32),
        "obj_pose_world.npy": ((frames, 4, 4), np.float32),
        "source_frame_id.npy": ((frames,), np.int32),
        "frame_time.npy": ((frames,), np.float32),
        "obj_point_id.npy": ((OBJECT_POINTS,), np.int32),
        "knn_hand_points_world.npy": ((frames, V114A_HAND_POINTS, 3), np.float32),
        "knn_hand_normals_world.npy": ((frames, V114A_HAND_POINTS, 3), np.float32),
        "obj_candidate_mask_2cm.npy": ((frames, OBJECT_POINTS), np.bool_),
    }
    for name, (shape, dtype) in expected.items():
        array = np.load(geometry / name, mmap_mode="r")
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise ValueError(f"{geometry / name}: expected {shape}/{dtype}, got {array.shape}/{array.dtype}")
        if array.dtype.kind == "f" and not np.isfinite(np.asarray(array)).all():
            raise ValueError(f"{geometry / name}: non-finite")
    forbidden = ("obj_knn_indices.npy", "hand_points_world.npy", "hand_normals_world.npy",
                 "obj_candidate_mask_5cm.npy", "hand_supervision_mask_2cm.npy",
                 "hand_min_object_distance_m.npy")
    if any((geometry / name).exists() for name in forbidden):
        raise ValueError(f"{geometry}: legacy-only arrays leaked into lean cache")
    active = int(np.asarray(np.load(geometry / "obj_candidate_mask_2cm.npy", mmap_mode="r")).any(axis=1).sum())
    return {"id": entry.sequence_id, "path": str(entry.target.resolve()), "dataset": "grab",
            "source": variant, "hand_variant": variant, "split": entry.split,
            "frame_count": frames, "active_frames": active}


def run(args: argparse.Namespace) -> int:
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    entries = _entries(args.split_root.resolve(), args.source_root.resolve(), output, args.variant)
    if args.sequence:
        wanted = set(args.sequence)
        entries = [entry for entry in entries if entry.sequence_id in wanted]
        if wanted != {entry.sequence_id for entry in entries}:
            raise ValueError("requested sequence absent from frozen split")
    if args.limit:
        entries = entries[:args.limit]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V1.14c cache generation requires CUDA")
    torch.cuda.set_device(device)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    if dirty and not args.allow_dirty:
        raise ValueError("full lean cache build requires a clean committed worktree")
    split_sha = {name: _sha256(args.split_root.resolve() / name)
                 for name in ("split.json", "train.txt", "val.txt", "test.txt")}
    manifest_path = output / f"run_manifest_{args.run_id}.json"
    manifest = {"schema_name": "ref2dex_data_run_manifest_v1", "task": "ObjectInteractionCmv2",
                "work_version": WORK_VERSION, "run_id": args.run_id, "run_status": "RUNNING",
                "operation": "v114c_grab_lean_geometry_build", "started_at": _now(),
                "git_commit": commit, "worktree_dirty": dirty, "variant": args.variant,
                "device": str(device), "expected_sequences": len(entries), "completed_sequences": 0,
                "completed_frames": 0, "failures": [], "split_sha256": split_sha,
                "inputs": {"source_root": str(args.source_root.resolve()),
                           "split_root": str(args.split_root.resolve())},
                "outputs": {"root": str(output)}, "conclusion": "INCONCLUSIVE"}
    _write_json(manifest_path, manifest)
    converter = InspireConverter(args.dex_root.resolve()) if args.variant == "inspire_f1" else None
    correspondences = (mano_cache._correspondences(args.mano_model_dir.resolve())
                       if args.variant == "mano" else None)
    raw_adapter = (GRABRawAdapter(
        num_obj_points=OBJECT_POINTS, device=str(device), grab_root=str(args.grab_raw_root.resolve()),
        mano_path=str(args.mano_model_dir.resolve()), ds_rate=4, nn_batch_size=8)
        if args.variant == "mano" else None)
    records: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(entries, 1):
        started = time.perf_counter()
        try:
            if (entry.target / "geometry" / "manifest.json").is_file():
                if not args.resume:
                    raise FileExistsError(entry.target)
                value = _validate(entry, args.variant)
            else:
                partial = entry.target.with_name(entry.target.name + ".partial")
                if partial.exists() or entry.target.exists():
                    raise FileExistsError(f"incomplete target requires review: {entry.target}")
                geometry = partial / "geometry"
                geometry.mkdir(parents=True)
                try:
                    frames = (_mano(entry, geometry, correspondences, raw_adapter,
                                    args.grab_raw_root.resolve(), device, args.geometry_batch,
                                    args.frame_batch, args.object_chunk)
                              if args.variant == "mano" else
                              _inspire(entry, geometry, converter, device, args.frame_batch,
                                       args.object_chunk))
                    source_points = V114A_HAND_POINTS if args.variant == "mano" else 2 * INSPIRE_PER_SIDE
                    _write_json(geometry / "manifest.json", {
                        "schema_name": LEAN_GEOMETRY_SCHEMA, "schema_version": "1.0.0",
                        "work_version": WORK_VERSION, "sequence_id": entry.sequence_id,
                        "dataset": "grab", "source_dataset": "grab", "source": args.variant,
                        "hand_variant": args.variant, "split": entry.split,
                        "source_path": str(entry.source.resolve()), "coordinate_frame": "object_pose_t",
                        "hand_side": "bilateral_merged_left_then_right", "frame_count": frames,
                        "object_pool_points": OBJECT_POINTS, "knn_hand_points": V114A_HAND_POINTS,
                        "knn_points_per_side": POINTS_PER_SIDE, "source_knn_hand_points": source_points,
                        "hand_sampling_contract": V114A_HAND_SAMPLING_CONTRACT,
                        "hand_sampling_seed": 42,
                        "hand_index_sha256": hand_index_sha256(fixed_bilateral_hand_indices(args.variant)),
                        "candidate_mask_source_points": source_points, "interaction_radius_m": RADIUS_M,
                        "effective_fps": 30.0, "object_representation": "rigid_se3",
                    })
                    entry.target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(partial, entry.target)
                except Exception:
                    if partial.exists():
                        shutil.rmtree(partial)
                    raise
                value = _validate(entry, args.variant)
            value["elapsed_s"] = time.perf_counter() - started
            records.append(value)
            manifest["completed_sequences"] = len(records)
            manifest["completed_frames"] = sum(int(row["frame_count"]) for row in records)
            _write_json(manifest_path, manifest)
            print(json.dumps({"status": "COMPLETED", "index": ordinal, "total": len(entries), **value}), flush=True)
        except Exception as error:
            manifest["failures"].append({"sequence_id": entry.sequence_id,
                                         "error": f"{type(error).__name__}: {error}"})
            _write_json(manifest_path, manifest)
            traceback.print_exc()
    if manifest["failures"]:
        manifest.update(run_status="FAILED", finished_at=_now(), conclusion="INVALID_IMPLEMENTATION")
        _write_json(manifest_path, manifest)
        return 1
    grouped = {split: [row for row in records if row["split"] == split]
               for split in ("train", "val", "test")}
    counts = {key: len(value) for key, value in grouped.items()}
    frame_counts = {key: sum(int(row["frame_count"]) for row in value) for key, value in grouped.items()}
    _write_json(output / "index.json", {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_2", "schema_version": "1.14c",
        "work_version": WORK_VERSION, "created_at": _now(), "object_pool_points": OBJECT_POINTS,
        "model_object_points": 1024, "knn_hand_points_per_stream": {args.variant: V114A_HAND_POINTS},
        "source_probability": {args.variant: 1.0}, "split_policy": {"grab": "full_grab_v1 exact"},
        "counts": counts, "frame_counts": frame_counts, "sequences": grouped,
    })
    _write_json(output / "cache_manifest.json", {
        "schema_name": LEAN_CACHE_MANIFEST_SCHEMA, "schema_version": "1.0.0",
        "work_version": WORK_VERSION, "dataset": "grab", "variant": args.variant,
        "total_sequences": len(records), "total_frames": sum(frame_counts.values()),
        "knn_hand_points_per_stream": {args.variant: V114A_HAND_POINTS},
        "hand_sampling_contract": V114A_HAND_SAMPLING_CONTRACT, "hand_sampling_seed": 42,
        "hand_index_sha256": hand_index_sha256(fixed_bilateral_hand_indices(args.variant)),
        "split_sha256": split_sha, "validation": {"bad_count": 0, "bad_examples": []},
        "counts": counts, "frame_counts": frame_counts,
    })
    manifest.update(run_status="COMPLETED", finished_at=_now(), conclusion="N/A",
                    completed_sequences=len(records), completed_frames=sum(frame_counts.values()))
    _write_json(manifest_path, manifest)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("mano", "inspire_f1"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--grab-raw-root", type=Path, default=REPO_ROOT / "dataset/GRAB/data")
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dex-root", type=Path, default=REPO_ROOT.parent / "dex-retargeting")
    parser.add_argument("--mano-model-dir", type=Path,
                        default=REPO_ROOT / "dataset/arctic/data/body_models/mano")
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--geometry-batch", type=int, default=64)
    parser.add_argument("--frame-batch", type=int, default=2)
    parser.add_argument("--object-chunk", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Only for bounded pilot validation before the implementation commit.")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
