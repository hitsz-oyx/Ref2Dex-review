#!/usr/bin/env python3
"""Freeze part-local OakInk2 geometry for the approved V1.11h mixed training."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import pickle
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.task.ObjectInteractionCm.tools.data import export_oakink2_inspire_v1_4 as legacy
from src.task.ObjectInteractionCmv2.oakink2_parts import PART_ADAPTER_SCHEMA
from src.task.ObjectInteractionCmv2.tools.data.backfill_oakink2_inspire_v1_4 import RawAnnotationGeometryStore
from src.task.ObjectInteractionCmv2.multi_domain import sha256_file


WORK_VERSION = "V1.11.1"
REPO_ROOT = Path(__file__).resolve().parents[5]
WORKER_CONTEXT: dict[str, Any] = {}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def safe_sequence_path(sequence_id: str) -> str:
    return hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()


def load_oakink_records(index_path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = {}
    for row in payload.get("sequences", {}).get("train", []):
        if str(row.get("dataset")) != "oakink2":
            continue
        sequence_id = str(row["id"])
        if sequence_id in records:
            raise ValueError(f"duplicate OakInk2 cache ID: {sequence_id}")
        records[sequence_id] = row
    if not records:
        raise ValueError(f"{index_path}: no OakInk2 train records")
    return records


def source_arrays(path: str | Path) -> dict[str, np.ndarray]:
    geometry = Path(path) / "geometry"
    names = ("source_frame_id", "frame_time", "obj_points_pool_world", "obj_normals_pool_world", "obj_point_id")
    return {name: np.load(geometry / f"{name}.npy", mmap_mode="r") for name in names}


def validate_base_sequence(record: Mapping[str, Any], selection: Mapping[str, Any], expected_ids: np.ndarray,
                           expected_time: np.ndarray, local_points: list[np.ndarray], local_normals: list[np.ndarray],
                           poses: list[np.ndarray]) -> dict[str, float]:
    path = Path(record["path"]).resolve()
    manifest = json.loads((path / "geometry" / "manifest.json").read_text(encoding="utf-8"))
    object_ids = [str(value) for value in selection["selected_object_ids"]]
    if str(manifest.get("sequence_id")) != str(selection["id"]):
        raise ValueError(f"{path}: sequence ID mismatch")
    if [str(value) for value in manifest.get("selected_object_ids", [])] != object_ids:
        raise ValueError(f"{path}: selected object IDs mismatch")
    arrays = source_arrays(path)
    if not np.array_equal(np.asarray(arrays["source_frame_id"], dtype=np.int64), expected_ids):
        raise ValueError(f"{path}: source frame IDs differ from selection")
    if not np.allclose(np.asarray(arrays["frame_time"], dtype=np.float64), expected_time, atol=1e-4, rtol=0):
        raise ValueError(f"{path}: frame times differ from selected official timeline")
    point_ids = np.asarray(arrays["obj_point_id"], dtype=np.int64)
    if point_ids.shape != (4096,) or point_ids.min() < 0 or point_ids.max() >= 4096 * len(object_ids):
        raise ValueError(f"{path}: invalid component point IDs")
    part_indices = point_ids // 4096
    if set(part_indices.tolist()) != set(range(len(object_ids))):
        raise ValueError(f"{path}: cache point pool omits a selected component")
    point_local = np.concatenate(local_points, axis=0)[point_ids]
    normal_local = np.concatenate(local_normals, axis=0)[point_ids]
    frame_indices = sorted({0, len(expected_ids) // 2, len(expected_ids) - 1})
    point_residual = normal_residual = 0.0
    for frame in frame_indices:
        frame_poses = np.stack([value[frame] for value in poses], axis=0)
        rotations, translations = frame_poses[:, :3, :3], frame_poses[:, :3, 3]
        expected_points = np.empty((4096, 3), dtype=np.float32)
        expected_normals = np.empty((4096, 3), dtype=np.float32)
        for part_index in range(len(object_ids)):
            mask = part_indices == part_index
            expected_points[mask] = point_local[mask] @ rotations[part_index].T + translations[part_index]
            expected_normals[mask] = normal_local[mask] @ rotations[part_index].T
        observed_points = np.asarray(arrays["obj_points_pool_world"][frame], dtype=np.float32)
        observed_normals = np.asarray(arrays["obj_normals_pool_world"][frame], dtype=np.float32)
        point_residual = max(point_residual, float(np.max(np.linalg.norm(expected_points - observed_points, axis=-1))))
        normal_residual = max(normal_residual, float(np.max(np.linalg.norm(expected_normals - observed_normals, axis=-1))))
    if point_residual > 2e-4 or normal_residual > 2e-4:
        raise ValueError(f"{path}: raw part reconstruction mismatch")
    return {"point_replay_max_m": point_residual, "normal_replay_max": normal_residual}


def component_geometry(store: RawAnnotationGeometryStore, annotation: Mapping[str, Any], object_id: str,
                       frame_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    local_points, local_normals = store._static(object_id)
    frame_map = annotation.get("obj_transf", {}).get(object_id)
    if frame_map is None:
        raise KeyError(f"{store.sequence}/{object_id}: raw annotation lacks obj_transf")
    poses = []
    for frame_id in frame_ids.tolist():
        if int(frame_id) not in frame_map:
            raise KeyError(f"{store.sequence}/{object_id}: missing pose for frame {frame_id}")
        pose = np.asarray(frame_map[int(frame_id)], dtype=np.float32)
        if not legacy._is_se3(pose):
            raise ValueError(f"{store.sequence}/{object_id}: invalid raw component SE(3)")
        poses.append(pose)
    pose_array = np.stack(poses, axis=0)
    for side in ("right", "left"):
        path = store.mapping.get((store.sequence, object_id, side))
        if path is None or not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as stage3:
            stage_ids = np.asarray(stage3["raw_frame_id"], dtype=np.int64)
            stage_poses = np.asarray(stage3["obj_root_pose_world"], dtype=np.float32)
        positions = np.searchsorted(stage_ids, frame_ids)
        valid = (positions < len(stage_ids)) & (stage_ids[np.minimum(positions, len(stage_ids) - 1)] == frame_ids)
        if valid.any() and not np.allclose(pose_array[valid], stage_poses[positions[valid]], atol=2e-4, rtol=0):
            raise ValueError(f"{store.sequence}/{object_id}: Stage3/raw pose mismatch")
        break
    return np.asarray(local_points, dtype=np.float32), np.asarray(local_normals, dtype=np.float32), pose_array


def write_part(root: Path, sequence_id: str, part_index: int, object_id: str, points: np.ndarray,
               normals: np.ndarray, poses: np.ndarray, frame_ids: np.ndarray) -> str:
    relative = Path("sequences") / safe_sequence_path(sequence_id) / f"part_{part_index:02d}"
    destination = root / relative
    partial = destination.with_name(destination.name + ".partial")
    if destination.exists() or partial.exists():
        raise FileExistsError(destination)
    partial.mkdir(parents=True)
    try:
        np.save(partial / "object_points_local.npy", points)
        np.save(partial / "object_normals_local.npy", normals)
        np.save(partial / "obj_pose_world.npy", poses)
        np.save(partial / "source_frame_id.npy", frame_ids.astype(np.int32))
        write_json(partial / "manifest.json", {
            "schema_name": PART_ADAPTER_SCHEMA,
            "work_version": WORK_VERSION,
            "sequence_id": sequence_id,
            "object_id": object_id,
            "part_index": part_index,
            "frame_count": len(frame_ids),
            "object_pool_points": 4096,
            "coordinate_frame": "component_pose_t",
            "units": "metres",
        })
        os.replace(partial, destination)
    except BaseException:
        if partial.exists():
            for child in partial.iterdir():
                child.unlink()
            partial.rmdir()
        raise
    return str(relative)


def initialize_worker(selection_index: str, annotation_root: str, stage3_root: str, mano_index: str,
                      inspire_index: str, output_root: str) -> None:
    global WORKER_CONTEXT
    WORKER_CONTEXT = {
        "annotation_root": Path(annotation_root),
        "output_root": Path(output_root),
        "mano": load_oakink_records(Path(mano_index)),
        "inspire": load_oakink_records(Path(inspire_index)),
        "stage3_map": legacy._load_stage3_map(Path(stage3_root)),
    }


def build_record(item: tuple[int, Mapping[str, Any]]) -> dict[str, Any]:
    ordinal, row = item
    try:
        sequence_id, sequence = str(row["id"]), str(row["sequence"])
        object_ids = [str(value) for value in row["selected_object_ids"]]
        frame_ids = np.asarray(row["selected_frame_ids"], dtype=np.int64)
        expected_time = np.asarray(row["selected_timeline_positions"], dtype=np.float64) / 30.0
        if not len(frame_ids) or len(frame_ids) != len(expected_time) or np.any(np.diff(frame_ids) <= 0):
            raise ValueError("invalid selection frame timeline")
        with (WORKER_CONTEXT["annotation_root"] / f"{sequence}.pkl").open("rb") as stream:
            annotation = pickle.load(stream)
        store = RawAnnotationGeometryStore(WORKER_CONTEXT["stage3_map"], sequence, annotation)
        local_points, local_normals, poses = [], [], []
        for object_id in object_ids:
            points, normals, part_poses = component_geometry(store, annotation, object_id, frame_ids)
            local_points.append(points)
            local_normals.append(normals)
            poses.append(part_poses)
        mano_check = validate_base_sequence(WORKER_CONTEXT["mano"][sequence_id], row, frame_ids, expected_time,
                                            local_points, local_normals, poses)
        inspire_check = validate_base_sequence(WORKER_CONTEXT["inspire"][sequence_id], row, frame_ids, expected_time,
                                               local_points, local_normals, poses)
        parts = []
        for part_index, object_id in enumerate(object_ids):
            path = write_part(WORKER_CONTEXT["output_root"], sequence_id, part_index, object_id, local_points[part_index],
                              local_normals[part_index], poses[part_index], frame_ids)
            parts.append({"part_index": part_index, "object_id": object_id, "path": path})
        return {"ordinal": ordinal, "record": {"id": sequence_id, "sequence": sequence, "selected_object_ids": object_ids,
                "frame_count": len(frame_ids), "parts": parts,
                "mano_point_replay_max_m": mano_check["point_replay_max_m"],
                "inspire_point_replay_max_m": inspire_check["point_replay_max_m"]}}
    except BaseException as error:
        return {"ordinal": ordinal, "failure": {"id": str(row.get("id")), "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc()}}


def input_sha256(arguments: argparse.Namespace) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in (
        arguments.selection_index, arguments.mano_index, arguments.mano_manifest,
        arguments.inspire_index, arguments.inspire_manifest,
    )}


def source_sha256() -> dict[str, str]:
    paths = (
        Path(__file__),
        REPO_ROOT / "src/task/ObjectInteractionCmv2/oakink2_parts.py",
        REPO_ROOT / "src/task/ObjectInteractionCmv2/tools/data/backfill_oakink2_inspire_v1_4.py",
        REPO_ROOT / "src/task/ObjectInteractionCm/tools/data/export_oakink2_inspire_v1_4.py",
        REPO_ROOT / "src/task/ObjectInteractionCm/tools/data/split_oakink2_active_tool.py",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def prepare(arguments: argparse.Namespace) -> Path:
    output = arguments.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    selection = json.loads(arguments.selection_index.resolve().read_text(encoding="utf-8"))
    if selection.get("schema_name") != legacy.SELECTION_SCHEMA or not selection.get("segments"):
        raise ValueError("unexpected or empty OakInk2 selection")
    mano = load_oakink_records(arguments.mano_index.resolve())
    inspire = load_oakink_records(arguments.inspire_index.resolve())
    selection_ids = {str(row["id"]) for row in selection["segments"]}
    if set(mano) != selection_ids or set(inspire) != selection_ids:
        raise ValueError("OakInk2 cache/selection IDs do not match exactly")
    output.mkdir(parents=True)
    manifest = {
        "schema_name": "ref2dex_data_run_manifest_v1",
        "task": "ObjectInteractionCmv2",
        "operation": "oakink2_part_adapter_build",
        "run_id": arguments.run_id,
        "run_status": "STARTED",
        "created_at": now(),
        "work_version": WORK_VERSION,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "implementation_identity": "base commit plus frozen source SHA256 set",
        "source_sha256": source_sha256(),
        "input_sha256": input_sha256(arguments),
        "inputs": {"selection_index": str(arguments.selection_index.resolve()), "annotation_root": str(arguments.annotation_root.resolve()),
                   "stage3_root": str(arguments.stage3_root.resolve()), "mano_index": str(arguments.mano_index.resolve()),
                   "inspire_index": str(arguments.inspire_index.resolve())},
        "expected_segments": len(selection_ids),
        "completed_segments": 0,
        "completed_parts": 0,
        "failures": [],
        "outputs": {"root": str(output), "index": "PENDING", "cache_manifest": "PENDING"},
        "conclusion": "INCONCLUSIVE",
    }
    write_json(output / "run_manifest.json", manifest)
    return output


def run(arguments: argparse.Namespace) -> None:
    output = arguments.output_root.resolve()
    manifest_path = output / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"prepare the adapter before running it: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "STARTED" or manifest.get("run_id") != arguments.run_id:
        raise ValueError("adapter run state or ID mismatch")
    if manifest.get("input_sha256") != input_sha256(arguments) or manifest.get("source_sha256") != source_sha256():
        raise ValueError("adapter inputs or implementation changed after preparation")
    manifest.update(run_status="RUNNING", started_at=now(), command=[sys.executable, *sys.argv])
    if os.environ.get("REF2DEX_SERVICE_UNIT"):
        manifest["service_unit"] = os.environ["REF2DEX_SERVICE_UNIT"]
    write_json(manifest_path, manifest)
    selection = json.loads(arguments.selection_index.resolve().read_text(encoding="utf-8"))
    rows = list(selection["segments"])
    if not 1 <= arguments.workers <= 8:
        raise ValueError("approved adapter worker count must be in [1, 8]")
    manifest["workers"] = arguments.workers
    write_json(manifest_path, manifest)
    index_rows, failures = {}, []
    initargs = (str(arguments.selection_index.resolve()), str(arguments.annotation_root.resolve()), str(arguments.stage3_root.resolve()),
                str(arguments.mano_index.resolve()), str(arguments.inspire_index.resolve()), str(output))
    with ProcessPoolExecutor(max_workers=arguments.workers, initializer=initialize_worker, initargs=initargs) as pool:
        futures = [pool.submit(build_record, item) for item in enumerate(rows, 1)]
        for future in as_completed(futures):
            value = future.result()
            ordinal = value["ordinal"]
            if "failure" in value:
                failures.append(value["failure"])
                manifest["failures"] = failures
                print(json.dumps({"status": "FAILED", "index": ordinal, **value["failure"]}), flush=True)
            else:
                record = value["record"]
                index_rows[ordinal] = record
                manifest["completed_segments"] = len(index_rows)
                manifest["completed_parts"] += len(record["parts"])
                print(json.dumps({"status": "COMPLETED", "index": ordinal, "total": len(rows), "id": record["id"], "parts": len(record["parts"])}), flush=True)
            write_json(manifest_path, manifest)
    if failures:
        manifest.update(run_status="FAILED", finished_at=now(), conclusion="INVALID_IMPLEMENTATION")
        write_json(manifest_path, manifest)
        raise RuntimeError(f"OakInk2 part adapter failed for {len(failures)} segments")
    ordered_rows = [index_rows[ordinal] for ordinal in range(1, len(rows) + 1)]
    index = {"schema_name": PART_ADAPTER_SCHEMA, "work_version": WORK_VERSION, "created_at": now(),
             "selection_index": str(arguments.selection_index.resolve()), "sequences": ordered_rows,
             "counts": {"sequences": len(ordered_rows), "parts": sum(len(row["parts"]) for row in ordered_rows)}}
    write_json(output / "index.json", index)
    write_json(output / "cache_manifest.json", {"schema_name": PART_ADAPTER_SCHEMA, "work_version": WORK_VERSION,
               "validation": {"bad_count": 0}, "source_sha256": manifest["input_sha256"],
               "coordinate_frame": "component_pose_t", "units": "metres", "object_pool_points": 4096,
               "counts": index["counts"]})
    manifest.update(run_status="COMPLETED", finished_at=now(), outputs={"root": str(output), "index": str(output / "index.json"),
                    "cache_manifest": str(output / "cache_manifest.json")}, conclusion="N/A")
    write_json(manifest_path, manifest)


def abort(arguments: argparse.Namespace) -> None:
    manifest_path = arguments.output_root.resolve() / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "STARTED":
        raise ValueError("only an unstarted adapter preparation may be aborted")
    manifest.update(run_status="FAILED", finished_at=now(), error=arguments.reason,
                    conclusion="N/A")
    write_json(manifest_path, manifest)


def mark_stopped(arguments: argparse.Namespace) -> None:
    manifest_path = arguments.output_root.resolve() / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "RUNNING":
        raise ValueError("only a running adapter may be marked stopped")
    manifest.update(run_status="STOPPED", finished_at=now(), stop_reason=arguments.reason,
                    conclusion="N/A")
    write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "abort", "mark-stopped"))
    parser.add_argument("--selection-index", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--stage3-root", type=Path, required=True)
    parser.add_argument("--mano-index", type=Path, required=True)
    parser.add_argument("--mano-manifest", type=Path, required=True)
    parser.add_argument("--inspire-index", type=Path, required=True)
    parser.add_argument("--inspire-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        print(json.dumps({"output": str(prepare(arguments))}))
    elif arguments.action == "run":
        run(arguments)
    elif arguments.action == "abort":
        abort(arguments)
    else:
        mark_stopped(arguments)


if __name__ == "__main__":
    main()
