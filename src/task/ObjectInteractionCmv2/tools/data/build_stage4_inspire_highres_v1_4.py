#!/usr/bin/env python3
"""Produce Cmv2 20270-point Inspire/KNN32 cache from bilateral Stage4 MANO.

This is deliberately a new producer: historical Stage4 Inspire exports use
3076 points and KNN=8, and are never read or modified here.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dex_retargeting.retargeting_config import RetargetingConfig
from src.task.ObjectInteractionCm.research.hand_region_sampling.run import _build_inspire_pool, _sample_uniform_surface
from src.task.ObjectInteractionCm.tools.data import build_bilateral_mano_v1_4_cache as mano_cache
from src.task.ObjectInteractionCm.tools.data import export_oakink2_inspire_v1_4 as legacy
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel, _fk_surface
from src.task.ObjectInteractionCm.tools.data.retarget_stage4_bilateral_inspire import canonical_cloud_to_visual_local


WORK_VERSION = "V1.11.1"
OBJECT_POINTS = 4096
DECODER_PER_SIDE = 1538
DECODER_POINTS = 3076
INSPIRE_PER_SIDE = 10135
INSPIRE_POINTS = 20270
KNN_K = 32
RADIUS_M = 0.02
TIP_IDS = {"right": np.asarray([744, 320, 443, 554, 671]), "left": np.asarray([744, 320, 444, 554, 671])}
REPO_ROOT = Path(__file__).resolve().parents[5]


@dataclass(frozen=True)
class Entry:
    sequence_id: str
    relative: Path
    split: str
    domain: str
    source: Path
    target: Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    return commit, dirty


def _entries(index_path: Path, source_root: Path, output_root: Path, domain: str,
             requested: list[str]) -> list[Entry]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries: list[Entry] = []
    for split in ("train", "val", "test"):
        for row in payload.get("sequences", {}).get(split, []):
            if row.get("dataset") != domain:
                continue
            sequence_id = str(row["id"])
            prefix = f"{domain}/"
            if not sequence_id.startswith(prefix):
                raise ValueError(f"malformed {domain} id: {sequence_id}")
            relative = Path(sequence_id[len(prefix):])
            source = source_root / relative
            if not all((source / name).is_file() for name in ("shared.npz", "left.npz", "right.npz")):
                raise FileNotFoundError(f"incomplete Stage4 source: {source}")
            entries.append(Entry(sequence_id, relative, split, domain, source,
                                 output_root / "sequences" / split / "inspire_f1" / domain / relative))
    if requested:
        wanted = set(requested)
        entries = [entry for entry in entries if entry.sequence_id in wanted]
        missing = wanted - {entry.sequence_id for entry in entries}
        if missing:
            raise ValueError(f"requested ids absent from index: {sorted(missing)}")
    if not entries:
        raise ValueError(f"no {domain} entries selected")
    return entries


class InspireConverter:
    def __init__(self, dex_root: Path) -> None:
        dex_root = dex_root.resolve()
        source_urdf = REPO_ROOT.parent / "dexplore" / "dexplore" / "data" / "assets" / "inspire_hand_new" / "inspire_hand_right.urdf"
        RetargetingConfig.set_default_urdf_dir(source_urdf.parent)
        config_roots = (dex_root / "dex_retargeting" / "configs" / "offline",
                        dex_root / "configs" / "offline")
        tree = ET.parse(str(source_urdf)); active: list[str] = []
        for joint in tree.getroot().findall("joint"):
            if joint.get("type") != "fixed": active.append(str(joint.get("name")))
            if joint.get("type") == "continuous":
                joint.set("type", "revolute"); limit = joint.find("limit") or ET.SubElement(joint, "limit")
                limit.set("lower", "-1000000"); limit.set("upper", "1000000")
                limit.set("effort", limit.get("effort", "1000")); limit.set("velocity", limit.get("velocity", "3.14"))
        for mesh in tree.getroot().iter("mesh"):
            filename = mesh.get("filename")
            if filename and not Path(filename).is_absolute(): mesh.set("filename", str((source_urdf.parent / filename).resolve()))
        self.temp = tempfile.TemporaryDirectory(prefix="cmv2-inspire-")
        patched = Path(self.temp.name) / source_urdf.name; tree.write(str(patched))
        self.inv_native = np.argsort(np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9]))
        self.models: dict[str, InspireUrdfModel] = {}; self.samplings: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}; self.retargeters = {}
        for side in ("left", "right"):
            urdf = source_urdf.parent / f"inspire_hand_{side}.urdf"
            try: model = InspireUrdfModel(urdf)
            except ValueError:
                if side != "left": raise
                model = InspireUrdfModel(source_urdf); urdf = source_urdf
            pool, _ = _build_inspire_pool(urdf, 0.20)
            cloud = _sample_uniform_surface(pool, 2024, INSPIRE_PER_SIDE)
            points, normals = canonical_cloud_to_visual_local(model, cloud.points, cloud.normals, cloud.source_visual_ids)
            self.models[side] = model; self.samplings[side] = (points, normals, cloud.source_visual_ids)
            config_path = next((root / f"inspire_hand_{side}.yml" for root in config_roots
                                if (root / f"inspire_hand_{side}.yml").is_file()), None)
            if config_path is None:
                raise FileNotFoundError(f"no Inspire {side} retarget config below {dex_root}")
            self.retargeters[side] = RetargetingConfig.load_from_file(
                config_path,
                override={"urdf_path": str(patched), "add_dummy_free_joint": False,
                          "target_joint_names": active, "ignore_mimic_joint": True},
            ).build()

    def convert(self, tips: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, float]:
        retargeter = self.retargeters[side]; retargeter.reset()
        q_urdf = np.asarray([retargeter.retarget(value) for value in tips], dtype=np.float32)
        q_native = q_urdf[:, self.inv_native]
        points, normals = _fk_surface(self.models[side], q_native, *self.samplings[side])
        if not np.isfinite(points).all() or not np.isfinite(normals).all(): raise ValueError(f"non-finite Inspire {side} surface")
        return points.astype(np.float32), normals.astype(np.float32), float(np.max(np.abs(q_native)))


def _decoder_from_highres(points: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selection = np.linspace(0, INSPIRE_PER_SIDE - 1, DECODER_PER_SIDE).round().astype(np.int64)
    selection = np.concatenate((selection, INSPIRE_PER_SIDE + selection))
    return np.ascontiguousarray(points[:, selection]), np.ascontiguousarray(normals[:, selection])


def _validate(path: Path) -> dict[str, Any]:
    geometry = path / "geometry"; manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8")); frames = int(manifest["frame_count"])
    expected = {"obj_points_pool_world.npy": (frames, OBJECT_POINTS, 3), "obj_normals_pool_world.npy": (frames, OBJECT_POINTS, 3),
                "hand_points_world.npy": (frames, DECODER_POINTS, 3), "hand_normals_world.npy": (frames, DECODER_POINTS, 3),
                "knn_hand_points_world.npy": (frames, INSPIRE_POINTS, 3), "knn_hand_normals_world.npy": (frames, INSPIRE_POINTS, 3),
                "obj_knn_indices.npy": (frames, OBJECT_POINTS, KNN_K), "obj_candidate_mask_2cm.npy": (frames, OBJECT_POINTS),
                "obj_pose_world.npy": (frames, 4, 4), "frame_time.npy": (frames,), "source_frame_id.npy": (frames,)}
    for name, shape in expected.items():
        value = np.load(geometry / name, mmap_mode="r")
        if value.shape != shape: raise ValueError(f"{path}/{name}: expected {shape}, got {value.shape}")
        if value.dtype.kind == "f" and not np.isfinite(np.asarray(value[: min(frames, 2)])).all(): raise ValueError(f"{path}/{name}: non-finite")
    maximum = int(np.asarray(np.load(geometry / "obj_knn_indices.npy", mmap_mode="r")).max())
    if maximum >= INSPIRE_POINTS: raise ValueError(f"{path}: KNN index {maximum} outside {INSPIRE_POINTS}")
    ids = np.asarray(np.load(geometry / "source_frame_id.npy", mmap_mode="r"))
    if np.any(np.diff(ids) <= 0) or abs(float(manifest["effective_fps"]) - 30.0) > 1e-4: raise ValueError(f"{path}: not increasing 30 Hz stream")
    poses = np.asarray(np.load(geometry / "obj_pose_world.npy", mmap_mode="r"))
    if not all(legacy._is_se3(value) for value in poses[:: max(1, len(poses) // 16)]): raise ValueError(f"{path}: invalid object SE(3)")
    return {"id": manifest["sequence_id"], "path": str(path.resolve()), "frames": frames, "knn_index_max": maximum}


def _process(entry: Entry, converter: InspireConverter, device: torch.device, frame_batch: int,
             object_chunk: int, resume: bool) -> dict[str, Any]:
    if (entry.target / "geometry" / "manifest.json").is_file():
        if not resume: raise FileExistsError(entry.target)
        return _validate(entry.target)
    partial = entry.target.with_name(entry.target.name + ".partial")
    if partial.exists() or entry.target.exists(): raise FileExistsError(f"incomplete/existing target: {entry.target}")
    geometry = partial / "geometry"; geometry.mkdir(parents=True)
    try:
        with np.load(entry.source / "shared.npz", allow_pickle=False) as shared:
            obj = np.asarray(shared["obj_points_world"], dtype=np.float32); obj_normals = np.asarray(shared["obj_normals_world"], dtype=np.float32)
            ids = np.asarray(shared["raw_frame_id"], dtype=np.int32); pose = np.asarray(shared.get("obj_pose_world", np.broadcast_to(np.eye(4, dtype=np.float32), (len(obj), 4, 4))), dtype=np.float32)
            if obj.shape[1:] != (OBJECT_POINTS, 3) or obj_normals.shape != obj.shape or pose.shape != (len(obj), 4, 4): raise ValueError(f"invalid Stage4 object stream: {entry.source}")
            all_points: list[np.ndarray] = []; all_normals: list[np.ndarray] = []; masks: list[np.ndarray] = []; qmax = {}
            for side in ("left", "right"):
                with np.load(entry.source / f"{side}.npz", allow_pickle=False) as value:
                    data = {key: np.asarray(value[key]) for key in value.files}
                if "hand_mesh_vertices_world" in data and data["hand_mesh_vertices_world"].shape[1] >= 778:
                    vertices = np.asarray(data["hand_mesh_vertices_world"], dtype=np.float32)
                    local = (vertices - pose[:, None, :3, 3]) @ pose[:, :3, :3]
                    tips = local[:, TIP_IDS[side]]
                else:
                    canonical = np.asarray(data["hand_cano_points"], dtype=np.float32); labels = np.asarray(data["hand_finger_id"]); hand = np.asarray(data["hand_points_world"], dtype=np.float32)
                    center = canonical.mean(axis=0); selected = []
                    for finger in range(1, 6):
                        candidates = np.flatnonzero(labels == finger)
                        if not len(candidates): raise ValueError(f"{entry.sequence_id}/{side}: missing finger {finger}")
                        selected.append(int(candidates[np.linalg.norm(canonical[candidates] - center, axis=1).argmax()]))
                    tips = (hand[:, selected] - pose[:, None, :3, 3]) @ pose[:, :3, :3]
                local_points, local_normals, qmax[side] = converter.convert(tips, side)
                all_points.append(np.einsum("bij,bpj->bpi", pose[:, :3, :3], local_points) + pose[:, None, :3, 3])
                all_normals.append(np.einsum("bij,bpj->bpi", pose[:, :3, :3], local_normals))
                if "obj_candidate_mask_5cm" in data: masks.append(np.asarray(data["obj_candidate_mask_5cm"], dtype=bool))
            high_points = np.concatenate(all_points, axis=1).astype(np.float32); high_normals = np.concatenate(all_normals, axis=1).astype(np.float32)
            decoder, decoder_normals = _decoder_from_highres(high_points, high_normals)
            for name, value in (("obj_points_pool_world", obj), ("obj_normals_pool_world", obj_normals), ("obj_pose_world", pose),
                                ("source_frame_id", ids), ("frame_time", ids.astype(np.float32) / (120.0 if entry.domain == "grab" else 30.0)),
                                ("knn_hand_points_world", high_points), ("knn_hand_normals_world", high_normals),
                                ("hand_points_world", decoder), ("hand_normals_world", decoder_normals),
                                ("obj_candidate_mask_5cm", np.logical_or.reduce(masks) if masks else np.ones((len(obj), OBJECT_POINTS), dtype=bool))): np.save(geometry / f"{name}.npy", value)
        mano_cache._build_knn(geometry, device=device, frame_batch_size=frame_batch, object_chunk=object_chunk)
        _write_json(geometry / "manifest.json", {"schema_name": "ref2dex_object_interaction_cmv2_stage4_inspire_v1_4", "schema_version": "1.0.0", "work_version": WORK_VERSION,
            "sequence_id": entry.sequence_id, "dataset": entry.domain, "source_dataset": entry.domain, "source": "inspire_f1", "hand_variant": "inspire_f1", "split": entry.split,
            "source_path": str(entry.source.resolve()), "coordinate_frame": "object_pose_t", "hand_side": "bilateral_merged_left_then_right", "frame_count": len(ids),
            "object_pool_points": OBJECT_POINTS, "decoder_hand_points": DECODER_POINTS, "decoder_points_per_side": DECODER_PER_SIDE, "knn_hand_points": INSPIRE_POINTS,
            "knn_points_per_side": INSPIRE_PER_SIDE, "knn_k": KNN_K, "knn_index_dtype": "uint16", "interaction_radius_m": RADIUS_M, "hand_supervision_radius_m": RADIUS_M,
            "effective_fps": 30.0, "source_fps": 120.0 if entry.domain == "grab" else 30.0, "source_type": "stage4_mano_to_inspire_position_retarget",
            "surface_sampling": {"method": "global Inspire visual triangle area uniform", "seed": 2024, "points_per_side": INSPIRE_PER_SIDE, "space": "visual_mesh_local_then_single_fk"},
            "retarget_qpos_max_abs": qmax})
        entry.target.parent.mkdir(parents=True, exist_ok=True); os.replace(partial, entry.target)
        return _validate(entry.target)
    except Exception:
        if partial.exists(): shutil.rmtree(partial)
        raise


def run(args: argparse.Namespace) -> int:
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True); device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError(f"CUDA is required, got {device}")
    torch.cuda.set_device(device)
    entries = _entries(args.source_index.resolve(), args.source_root.resolve(), output, args.domain, args.sequence)
    if args.limit: entries = entries[:args.limit]
    commit, dirty = _git_state(); manifest_path = output / f"run_manifest_{args.run_id}.json"
    manifest = {"schema_name": "ref2dex_data_run_manifest_v1", "task": "ObjectInteractionCmv2", "operation": "stage4_inspire_highres_export", "run_id": args.run_id, "run_status": "STARTED", "started_at": _now(), "work_version": WORK_VERSION, "base_commit": commit, "worktree_dirty": dirty, "device": str(device), "expected_sequences": len(entries), "completed_sequences": 0, "completed_frames": 0, "failures": [], "inputs": {"source_index": str(args.source_index.resolve()), "source_root": str(args.source_root.resolve())}, "outputs": {"root": str(output)}, "conclusion": "INCONCLUSIVE"}
    _write_json(manifest_path, manifest); converter = InspireConverter(args.dex_root.resolve()); records = []
    for ordinal, entry in enumerate(entries, 1):
        try:
            value = _process(entry, converter, device, args.knn_frame_batch, args.knn_object_chunk, args.resume)
            records.append({"source": "inspire_f1", "hand_variant": "inspire_f1", "id": entry.sequence_id, "path": value["path"], "dataset": entry.domain, "split": entry.split, "frame_count": value["frames"]})
            manifest["completed_sequences"] = len(records); manifest["completed_frames"] += int(value["frames"]); _write_json(manifest_path, manifest)
            print(json.dumps({"status": "COMPLETED", "index": ordinal, "total": len(entries), **value}), flush=True)
        except Exception as exc:
            manifest["failures"].append({"sequence_id": entry.sequence_id, "error": f"{type(exc).__name__}: {exc}"}); _write_json(manifest_path, manifest); traceback.print_exc()
    if manifest["failures"]:
        manifest.update({"run_status": "FAILED", "finished_at": _now(), "conclusion": "INVALID_IMPLEMENTATION"}); _write_json(manifest_path, manifest); return 1
    split_rows = {split: [record for record in records if record["split"] == split] for split in ("train", "val", "test")}
    index = {"schema_name": "ref2dex_object_interaction_cm_index_v1_2", "schema_version": "1.4.0", "work_version": WORK_VERSION, "created_at": _now(), "object_pool_points": OBJECT_POINTS, "model_object_points": 1024, "decoder_hand_points_per_stream": DECODER_POINTS, "knn_hand_points_per_stream": {"inspire_f1": INSPIRE_POINTS}, "max_knn_hand_points": INSPIRE_POINTS, "knn_k": KNN_K, "source_probability": {"inspire_f1": 1.0}, "split_policy": {args.domain: "preserved from source MANO index"}, "sequences": split_rows, "counts": {key: len(value) for key, value in split_rows.items()}, "frame_counts": {key: sum(int(row["frame_count"]) for row in value) for key, value in split_rows.items()}}
    _write_json(output / "index.json", index); _write_json(output / "cache_manifest.json", {"schema_name": "ref2dex_cmv2_highres_cache_manifest_v1", "work_version": WORK_VERSION, "dataset": args.domain, "variant": "inspire_f1", "total_sequences": len(records), "total_frames": sum(int(row["frame_count"]) for row in records), "hand_contract": "decoder bilateral 3076 compatibility points; Inspire KNN bilateral 20270 points", "effective_fps": 30.0, "knn_k": KNN_K, "validation": {"bad_count": 0, "bad_examples": []}})
    manifest.update({"run_status": "COMPLETED", "finished_at": _now(), "outputs": {"root": str(output), "index": str((output / "index.json").resolve()), "cache_manifest": str((output / "cache_manifest.json").resolve())}, "conclusion": "SUPPORTED"}); _write_json(manifest_path, manifest); print(json.dumps({"status": "COMPLETED", "sequences": len(records), "frames": manifest["completed_frames"]}), flush=True); return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, required=True); parser.add_argument("--source-root", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domain", choices=("grab", "arctic"), required=True); parser.add_argument("--dex-root", type=Path, required=True); parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", required=True); parser.add_argument("--sequence", action="append", default=[]); parser.add_argument("--limit", type=int, default=0); parser.add_argument("--knn-frame-batch", type=int, default=2); parser.add_argument("--knn-object-chunk", type=int, default=512); parser.add_argument("--resume", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__": main()
