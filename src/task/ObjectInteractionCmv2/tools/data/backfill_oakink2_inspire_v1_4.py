#!/usr/bin/env python3
"""Backfill OakInk2 Inspire geometry from raw annotation frames.

The previous export wrote 1656 complete sequences and left 193 sequences as
``*.partial`` because Stage3 object files did not cover every selected frame.
This tool leaves those outputs untouched.  It validates and reuses complete
sequences from the failed export, then writes only the missing sequences into a
new root.  Missing object poses come from the raw annotation ``obj_transf``;
static points and normals come from any available Stage3 file for the same
object part.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from src.task.ObjectInteractionCm.tools.data import export_oakink2_inspire_v1_4 as legacy


MODIFICATION_VERSION = "V1.4.3"
INSPIRE_POINTS_PER_SIDE = 10135
INSPIRE_HAND_POINTS = INSPIRE_POINTS_PER_SIDE * 2
DECODER_POINTS_PER_SIDE = 1538
DECODER_HAND_POINTS = DECODER_POINTS_PER_SIDE * 2


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_id(selection_id: str) -> str:
    suffix = selection_id[len("oakink2/"):] if selection_id.startswith("oakink2/") else selection_id
    return legacy._safe_id(suffix)


class RawAnnotationGeometryStore:
    """Stage3 static geometry plus annotation poses for all selected frames."""

    def __init__(self, mapping: Mapping[tuple[str, str, str], Path], sequence: str,
                 annotation: Mapping[str, Any], canonical_root: Path | None = None) -> None:
        self.mapping = mapping
        self.sequence = str(sequence)
        self.annotation = annotation
        self.canonical_root = canonical_root
        self.data: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def _static(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        for side in ("right", "left"):
            key = (self.sequence, str(object_id), side)
            path = self.mapping.get(key)
            if path is None or not path.is_file():
                continue
            cache_key = (str(object_id), side)
            if cache_key not in self.data:
                with np.load(path, allow_pickle=False) as value:
                    points = np.asarray(value["obj_points"], dtype=np.float32)
                    normals = np.asarray(value["obj_normals"], dtype=np.float32)
                if points.shape != (legacy.OBJECT_POINTS, 3) or normals.shape != points.shape:
                    raise ValueError(f"invalid Stage3 static object geometry: {path}")
                self.data[cache_key] = (points, normals)
            return self.data[cache_key]
        if self.canonical_root is not None:
            canonical = self.canonical_root / f"{str(object_id).replace('@', '_')}.npz"
            if canonical.is_file():
                # The canonical query cache contains points only.  Refuse to
                # invent normals; all selected objects are expected to have a
                # Stage3 file, and this branch is retained as an auditable
                # diagnostic rather than silently changing the normal contract.
                raise FileNotFoundError(
                    f"{self.sequence}/{object_id}: canonical points exist but Stage3 normals are missing"
                )
        raise FileNotFoundError(f"no static Stage3 geometry for {self.sequence}/{object_id}")

    def parts(self, object_ids, frame_ids):
        wanted = np.asarray(frame_ids, dtype=np.int64)
        local_points, local_normals, poses, part_ids = [], [], [], []
        obj_transf = self.annotation.get("obj_transf", {})
        for part_index, object_id in enumerate(str(value) for value in object_ids):
            if object_id not in obj_transf:
                raise KeyError(f"annotation has no obj_transf for {self.sequence}/{object_id}")
            static_points, static_normals = self._static(object_id)
            frame_map = obj_transf[object_id]
            part_poses = []
            for frame_id in wanted.tolist():
                if int(frame_id) not in frame_map:
                    raise KeyError(f"annotation has no {object_id} pose at frame {frame_id}")
                pose = np.asarray(frame_map[int(frame_id)], dtype=np.float32)
                if not legacy._is_se3(pose):
                    raise ValueError(f"invalid annotation SE(3): {self.sequence}/{object_id}/{frame_id}")
                part_poses.append(pose)
            part_pose_array = np.stack(part_poses, axis=0)
            # Confirm the raw pose agrees with Stage3 wherever that object file
            # contains the selected frame.  This catches coordinate or unit
            # drift before any new cache is written.
            for side in ("right", "left"):
                path = self.mapping.get((self.sequence, object_id, side))
                if path is None or not path.is_file():
                    continue
                with np.load(path, allow_pickle=False) as value:
                    stage_ids = np.asarray(value["raw_frame_id"], dtype=np.int64)
                    stage_pose = np.asarray(value["obj_root_pose_world"], dtype=np.float32)
                positions = np.searchsorted(stage_ids, wanted)
                valid = (positions < len(stage_ids)) & (
                    stage_ids[np.minimum(positions, len(stage_ids) - 1)] == wanted)
                if valid.any():
                    expected = np.stack([part_poses[int(i)] for i in np.flatnonzero(valid)], axis=0)
                    observed = stage_pose[positions[valid]]
                    if not np.allclose(expected, observed, atol=2e-4, rtol=0):
                        raise ValueError(f"Stage3/raw pose mismatch: {self.sequence}/{object_id}")
                break
            local_points.append(np.broadcast_to(static_points[None], (len(wanted), legacy.OBJECT_POINTS, 3)).copy())
            local_normals.append(np.broadcast_to(static_normals[None], (len(wanted), legacy.OBJECT_POINTS, 3)).copy())
            poses.append(part_pose_array)
            part_ids.append(np.full((legacy.OBJECT_POINTS,), part_index, dtype=np.int32))
        if not local_points:
            raise ValueError(f"{self.sequence}: selected segment has no object parts")
        return (
            np.concatenate(local_points, axis=1),
            np.concatenate(local_normals, axis=1),
            np.stack(poses, axis=1),
            np.concatenate(part_ids, axis=0),
        )


def _output_destination(root: Path, selection_id: str) -> Path:
    return root / "sequences" / "train" / "inspire_f1" / "oakink2" / _safe_id(selection_id)


def _set_legacy_highres_contract() -> None:
    """Make the legacy retargeter produce the approved Inspire KNN stream.

    The old exporter used 1538 points per side for both files.  Its retargeting
    and KNN routines read these module constants at call time, so setting them
    here keeps the raw reconstruction route while preventing a 3076-point
    output from being mistaken for a V1.4 training cache.
    """
    legacy.HAND_POINTS_PER_SIDE = INSPIRE_POINTS_PER_SIDE
    legacy.HAND_POINTS = INSPIRE_HAND_POINTS


def _write_decoder_compatibility_stream(path: Path) -> None:
    """Derive the optional 3076-point decoder stream from the high-res stream."""
    geometry = path / "geometry"
    points = np.load(geometry / "knn_hand_points_world.npy", mmap_mode="r")
    normals = np.load(geometry / "knn_hand_normals_world.npy", mmap_mode="r")
    if points.shape[1:] != (INSPIRE_HAND_POINTS, 3) or normals.shape != points.shape:
        raise ValueError(f"{path}: expected Inspire high-res stream {INSPIRE_HAND_POINTS}")
    side = np.linspace(0, INSPIRE_POINTS_PER_SIDE - 1, DECODER_POINTS_PER_SIDE).round().astype(np.int64)
    indices = np.concatenate((side, INSPIRE_POINTS_PER_SIDE + side))
    np.save(geometry / "hand_points_world.npy", np.asarray(points[:, indices], dtype=np.float32))
    np.save(geometry / "hand_normals_world.npy", np.asarray(normals[:, indices], dtype=np.float32))
    supervision_path = geometry / "hand_supervision_mask_2cm.npy"
    if supervision_path.is_file():
        supervision = np.load(supervision_path, mmap_mode="r")
        if supervision.shape[1:] != (INSPIRE_HAND_POINTS,):
            raise ValueError(f"{path}: invalid high-res supervision mask {supervision.shape}")
        np.save(supervision_path, np.asarray(supervision[:, indices], dtype=np.bool_))


def _validate_v1_4_geometry(path: Path) -> dict[str, Any]:
    geometry = path / "geometry"
    manifest_path = geometry / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("hand_variant") != "inspire_f1":
        raise ValueError(f"{path}: geometry manifest must declare hand_variant=inspire_f1")
    frame_count = int(value.get("frame_count", 0))
    expected = {
        "obj_points_pool_world.npy": (frame_count, legacy.OBJECT_POINTS, 3),
        "obj_normals_pool_world.npy": (frame_count, legacy.OBJECT_POINTS, 3),
        "obj_pose_world.npy": (frame_count, 4, 4),
        "source_frame_id.npy": (frame_count,),
        "frame_time.npy": (frame_count,),
        "hand_points_world.npy": (frame_count, DECODER_HAND_POINTS, 3),
        "hand_normals_world.npy": (frame_count, DECODER_HAND_POINTS, 3),
        "knn_hand_points_world.npy": (frame_count, INSPIRE_HAND_POINTS, 3),
        "knn_hand_normals_world.npy": (frame_count, INSPIRE_HAND_POINTS, 3),
        "obj_knn_indices.npy": (frame_count, legacy.OBJECT_POINTS, legacy.KNN_K),
        "obj_candidate_mask_2cm.npy": (frame_count, legacy.OBJECT_POINTS),
        "hand_supervision_mask_2cm.npy": (frame_count, DECODER_HAND_POINTS),
    }
    for name, shape in expected.items():
        file_path = geometry / name
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        array = np.load(file_path, mmap_mode="r")
        if array.shape != shape:
            raise ValueError(f"{file_path}: expected {shape}, got {array.shape}")
        if array.dtype.kind == "f" and not np.isfinite(np.asarray(array[: min(len(array), 2)])).all():
            raise ValueError(f"{file_path}: non-finite values")
    max_index = int(np.asarray(np.load(geometry / "obj_knn_indices.npy", mmap_mode="r")).max())
    if max_index >= INSPIRE_HAND_POINTS:
        raise ValueError(f"{path}: KNN index {max_index} exceeds {INSPIRE_HAND_POINTS}")
    return {"id": value.get("sequence_id", path.name), "frames": frame_count,
            "path": str(path.resolve()), "knn_hand_points": INSPIRE_HAND_POINTS}


def _cache_entry(path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    value = _validate_v1_4_geometry(path)
    return {
        "source": "inspire_f1",
        "hand_variant": "inspire_f1",
        "id": str(row["id"]),
        "path": value["path"],
        "dataset": "oakink2",
        "split": "train",
        "frame_count": int(value["frames"]),
        "knn_hand_points": INSPIRE_HAND_POINTS,
    }


def backfill(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = args.selection_index.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_name") != legacy.SELECTION_SCHEMA:
        raise ValueError(f"expected {legacy.SELECTION_SCHEMA}")
    rows = list(selection.get("segments", []))
    if args.limit is not None:
        rows = rows[int(args.offset):int(args.offset) + int(args.limit)]
    elif args.offset:
        rows = rows[int(args.offset):]
    existing_root = args.existing_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _set_legacy_highres_contract()
    # A resume must not overwrite the manifest of the interrupted baseline
    # run.  Keep each invocation independently auditable while sharing the
    # sequence root whose complete high-resolution entries are reusable.
    run_manifest_path = output_root / (
        "run_manifest.json" if not (output_root / "run_manifest.json").exists()
        else f"run_manifest_{args.run_id or output_root.name}.json"
    )
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCmv2",
        "operation": "oakink2_v1_4_raw_frame_backfill",
        "run_id": args.run_id or output_root.name,
        "run_status": "STARTED",
        "created_at": _now(),
        "modification_version": MODIFICATION_VERSION,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selection_index": str(selection_path),
        "existing_root": str(existing_root),
        "output_root": str(output_root),
        "knn_frame_batch": int(args.knn_frame_batch),
        "knn_object_chunk": int(args.knn_object_chunk),
        "mano_batch_size": int(args.mano_batch_size),
        "expected_segments": len(rows),
        "reused_segments": 0,
        "backfilled_segments": 0,
        "completed_frames": 0,
        "failures": [],
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(run_manifest_path, manifest)
    stage3_map = legacy._load_stage3_map(args.stage3_root.resolve())
    torch_device = torch.device(args.device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for OakInk2 backfill but unavailable")
        torch.cuda.set_device(torch_device)
    mano = legacy._ManoReconstructor(args.mano_root.resolve(), torch_device, args.mano_batch_size)
    inspire = legacy._InspireRetargeter(args.dex_root.resolve(), legacy.SURFACE_SEED)
    export_args = SimpleNamespace(
        resume=True,
        mano_batch_size=args.mano_batch_size,
        knn_frame_batch=args.knn_frame_batch,
        knn_object_chunk=args.knn_object_chunk,
    )
    records, failures = [], []
    for ordinal, row in enumerate(rows, 1):
        selection_id = str(row["id"])
        destination = _output_destination(output_root, selection_id)
        existing = _output_destination(existing_root, selection_id)
        try:
            reusable = False
            # Prefer complete entries already written in this high-resolution
            # root.  The original `existing_root` contains legacy 3076-point
            # entries and remains only a read-only audit source.
            for candidate in (destination, existing):
                if not (candidate / "geometry" / "manifest.json").is_file():
                    continue
                # Only an already corrected high-resolution cache may be reused.
                # The old 3076-point export is retained for audit but is never a
                # valid V1.4 training input; an invalid old entry is rebuilt below.
                try:
                    record = _cache_entry(candidate, row)
                    reusable = True
                    break
                except (FileNotFoundError, ValueError):
                    continue
            if reusable:
                manifest["reused_segments"] += 1
            else:
                with (args.annotation_root.resolve() / f"{row['sequence']}.pkl").open("rb") as stream:
                    annotation = pickle.load(stream)
                stage3 = RawAnnotationGeometryStore(stage3_map, row["sequence"], annotation,
                                                    args.canonical_root.resolve() if args.canonical_root else None)
                record_value = legacy._export_one(
                    row, annotation=annotation, stage3=stage3, reconstructor=mano,
                    retargeter=inspire, output_root=output_root, device=torch_device, args=export_args,
                )
                destination = Path(record_value["path"])
                _write_decoder_compatibility_stream(destination)
                geometry_manifest_path = destination / "geometry" / "manifest.json"
                geometry_manifest = json.loads(geometry_manifest_path.read_text(encoding="utf-8"))
                geometry_manifest["modification_version"] = MODIFICATION_VERSION
                geometry_manifest["producer_task"] = "ObjectInteractionCmv2"
                geometry_manifest["hand_variant"] = "inspire_f1"
                geometry_manifest["hand_points"] = DECODER_HAND_POINTS
                geometry_manifest["decoder_hand_points"] = DECODER_HAND_POINTS
                geometry_manifest["decoder_points_per_side"] = DECODER_POINTS_PER_SIDE
                geometry_manifest["knn_hand_points"] = INSPIRE_HAND_POINTS
                geometry_manifest["knn_points_per_side"] = INSPIRE_POINTS_PER_SIDE
                geometry_manifest["decoder_stream"] = "uniform_subsample_of_high_resolution_inspire_stream"
                _write_json(geometry_manifest_path, geometry_manifest)
                record = _cache_entry(destination, row)
                manifest["backfilled_segments"] += 1
            records.append(record)
            manifest["completed_frames"] += int(record["frame_count"])
            if ordinal % 10 == 0 or ordinal == len(rows):
                _write_json(run_manifest_path, manifest)
                print(json.dumps({"segments": ordinal, "reused": manifest["reused_segments"],
                                  "backfilled": manifest["backfilled_segments"],
                                  "failures": len(failures)}, ensure_ascii=False), flush=True)
        except Exception as error:
            failure = {"selection_id": selection_id, "error": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            manifest["failures"] = failures
            _write_json(run_manifest_path, manifest)
            print(json.dumps({"status": "FAILED", **failure}, ensure_ascii=False), file=sys.stderr, flush=True)
    if failures:
        manifest.update({"run_status": "FAILED", "finished_at": _now(), "conclusion": "INCONCLUSIVE"})
        _write_json(run_manifest_path, manifest)
        raise RuntimeError(f"OakInk2 backfill failed for {len(failures)} segments")
    if args.limit is not None:
        # A bounded resume smoke validates reuse and the revised batching
        # without publishing a misleading partial formal index/manifest.
        manifest.update({
            "run_status": "COMPLETED", "finished_at": _now(),
            "outputs": {"sample_root": str(output_root)},
            "conclusion": "SUPPORTED",
            "limited_smoke": True,
        })
        _write_json(run_manifest_path, manifest)
        return manifest
    records.sort(key=lambda value: str(value["id"]))
    entries = [dict(value) for value in records]
    index = {
        "schema_name": "ref2dex_object_interaction_cm_oakink2_index_v1_4",
        "schema_version": "1.1.0",
        "created_at": _now(),
        "modification_version": MODIFICATION_VERSION,
        "source": "OakInk2 V1.4 independent Inspire cache with raw annotation backfill",
        "source_probability": {"oakink2": 1.0},
        "object_pool_points": legacy.OBJECT_POINTS,
        "model_object_points": 1024,
        "decoder_hand_points_per_stream": DECODER_HAND_POINTS,
        "knn_hand_points_per_stream": {"inspire_f1": INSPIRE_HAND_POINTS},
        "max_knn_hand_points": INSPIRE_HAND_POINTS,
        "knn_k": legacy.KNN_K,
        "split_policy": {"oakink2": "existing selected train segments; raw frame backfill"},
        "sequences": {"train": entries, "val": [], "test": []},
        "counts": {"train": len(entries), "val": 0, "test": 0,
                   "frames": sum(int(item["frame_count"]) for item in entries)},
        "selection_index": str(selection_path),
        "existing_root": str(existing_root),
        "backfill_root": str(output_root),
    }
    cache_manifest = {
        "schema_name": "ref2dex_object_interaction_cm_oakink2_inspire_cache_manifest",
        "schema_version": "1.1.0",
        "modification_version": MODIFICATION_VERSION,
        "created_at": _now(),
        "storage": "NAS" if "/mnt/ugreen_nas/" in str(output_root) else "unknown",
        "roots": [str(existing_root), str(output_root)],
        "sequence_counts": {"oakink2": len(entries)},
        "total_sequences": len(entries),
        "total_frames": int(index["counts"]["frames"]),
        "hand_contract": "decoder bilateral 3076 compatibility points; Inspire KNN bilateral 20270 points (10135 per side)",
        "object_contract": "4096 point pool; raw annotation pose backfill",
        "effective_fps": legacy.TARGET_FPS,
        "selection_index": str(selection_path),
        "validation": {"bad_count": 0, "bad_examples": []},
        "backfill": {"reused_segments": manifest["reused_segments"],
                     "backfilled_segments": manifest["backfilled_segments"]},
    }
    _write_json(output_root / "index.json", index)
    _write_json(output_root / "cache_manifest.json", cache_manifest)
    manifest.update({
        "run_status": "COMPLETED", "finished_at": _now(),
        "outputs": {"index": str((output_root / "index.json").resolve()),
                    "cache_manifest": str((output_root / "cache_manifest.json").resolve())},
        "conclusion": "SUPPORTED",
    })
    _write_json(run_manifest_path, manifest)
    print(json.dumps({"segments": len(entries), "frames": index["counts"]["frames"],
                      "backfilled": manifest["backfilled_segments"], "output": str(output_root)},
                     ensure_ascii=False), flush=True)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-index", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path)
    parser.add_argument("--existing-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--dex-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--mano-batch-size", type=int, default=128)
    parser.add_argument("--knn-frame-batch", type=int, default=2)
    parser.add_argument("--knn-object-chunk", type=int, default=512)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    backfill(args)


if __name__ == "__main__":
    main()
