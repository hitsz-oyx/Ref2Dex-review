#!/usr/bin/env python3
"""Export the approved OakInk2 V1.4 single-object Inspire cache.

The exporter has two explicit stages:

``select``
    Rebuilds an independent official-30 Hz selection index from the immutable
    single-object primitive index.  Motion is evaluated first with the fixed
    7-frame/2 mm/2 degree rule; Stage3 distances are read only for reliable
    motion components and final frames use the strict ``< 2 cm`` rule.

``export``
    Reconstructs official bilateral quaternion MANO, retargets both hands to
    the Inspire geometric surface, writes per-segment geometry and offline
    KNN arrays, and creates an independent train index/cache manifest.

The output root is deliberately separate from the existing GRAB/ARCTIC cache.
No source annotation, Stage3 file, old selection index, or existing cache is
modified.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

for _name, _value in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.task.ObjectInteractionCm.research.oakink2_temporal_segmentation.run import (  # noqa: E402
    _official_30hz_timeline,
)
from src.task.ObjectInteractionCm.tools.data.oakink2_temporal_segments import (  # noqa: E402
    TemporalSegmentationConfig,
    segment_motion_then_contact,
)
from src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool import (  # noqa: E402
    _load_stage3_map,
)


MODIFICATION_VERSION = "V1.4.23"
SELECTION_SCHEMA = "ref2dex_oakink2_inspire_selection_v1_4"
CACHE_SCHEMA = "ref2dex_object_interaction_cm_oakink2_inspire_v1_4"
INDEX_SCHEMA = "ref2dex_object_interaction_cm_oakink2_index_v1_4"
OBJECT_POINTS = 4096
HAND_POINTS_PER_SIDE = 1538
HAND_POINTS = HAND_POINTS_PER_SIDE * 2
KNN_K = 32
RADIUS_M = 0.02
CANDIDATE_RADIUS_M = 0.05
SURFACE_SEED = 2024
SOURCE_FPS = 120.0
TARGET_FPS = 30.0
TIP_IDS = {
    "right": np.asarray([744, 320, 443, 554, 671], dtype=np.int64),
    "left": np.asarray([744, 320, 444, 554, 671], dtype=np.int64),
}


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _safe_id(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _is_se3(value: np.ndarray) -> bool:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        return False
    rotation = value[:3, :3]
    return bool(
        np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5)
        and np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-4)
        and abs(float(np.linalg.det(rotation)) - 1.0) <= 2e-4
    )


def _primitive_mocap_frames(annotation: Mapping[str, Any], frame_range_def: str) -> np.ndarray:
    ranges = ast.literal_eval(str(frame_range_def))
    usable = [value for value in ranges if value is not None]
    if not usable:
        raise ValueError(f"empty frame range: {frame_range_def}")
    start = min(int(value[0]) for value in usable)
    end = max(int(value[1]) for value in usable)
    raw = np.asarray(sorted(int(value) for value in annotation["raw_mano"]), dtype=np.int64)
    frames = raw[(raw >= start) & (raw <= end)]
    if not len(frames):
        raise ValueError(f"primitive has no raw MANO frames: {frame_range_def}")
    return frames


class _Stage3DistanceStore:
    """Per-sequence compressed NPZ cache used only during selection."""

    def __init__(self, mapping: Mapping[tuple[str, str, str], Path], sequence: str) -> None:
        self.mapping = mapping
        self.sequence = sequence
        self.values: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.missing: set[tuple[str, str]] = set()

    def _load(self, object_id: str, side: str) -> tuple[np.ndarray, np.ndarray] | None:
        key = (object_id, side)
        if key in self.values:
            return self.values[key]
        if key in self.missing:
            return None
        path = self.mapping.get((self.sequence, object_id, side))
        if path is None or not path.is_file():
            self.missing.add(key)
            return None
        with np.load(path, allow_pickle=False) as data:
            frame_ids = np.asarray(data["raw_frame_id"], dtype=np.int64)
            distances = np.asarray(data["hand_to_obj_min_dist"], dtype=np.float32)
        if distances.ndim != 2 or distances.shape[0] != len(frame_ids):
            raise ValueError(f"invalid Stage3 distance shape: {path}")
        self.values[key] = (frame_ids, distances)
        return self.values[key]

    def distances(self, object_ids: Sequence[str], frame_ids: Sequence[int]) -> dict[int, float]:
        wanted = np.asarray([int(value) for value in frame_ids], dtype=np.int64)
        result = {int(value): float("inf") for value in wanted.tolist()}
        if not len(wanted):
            return result
        for object_id in object_ids:
            for side in ("left", "right"):
                loaded = self._load(str(object_id), side)
                if loaded is None:
                    continue
                source_ids, source_distances = loaded
                positions = np.searchsorted(source_ids, wanted)
                valid = (positions < len(source_ids)) & (source_ids[np.minimum(positions, len(source_ids) - 1)] == wanted)
                for ordinal in np.flatnonzero(valid):
                    value = float(np.min(source_distances[int(positions[ordinal])]))
                    fid = int(wanted[ordinal])
                    if value < result[fid]:
                        result[fid] = value
        return result


def _selection_row(
    source: Mapping[str, Any],
    *,
    frame_ids: np.ndarray,
    timeline_positions: np.ndarray,
    temporal: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    selected = [int(value) for value in temporal["selected_frame_ids"]]
    position_map = {int(frame): int(pos) for frame, pos in zip(frame_ids.tolist(), timeline_positions.tolist())}
    selected_positions = [position_map[value] for value in selected if value in position_map]
    if len(selected_positions) != len(selected):
        raise ValueError("temporal selector returned a frame outside official timeline")
    sides = []
    if source.get("frame_range_lh") is not None:
        sides.append("left")
    if source.get("frame_range_rh") is not None:
        sides.append("right")
    return {
        "id": f"oakink2/{source['sequence']}/{ordinal:04d}",
        "sequence": str(source["sequence"]),
        "frame_range_def": str(source["frame_range_def"]),
        "primitive": source.get("primitive"),
        "interaction_mode": source.get("interaction_mode"),
        "instance_id": str(source["instance_id"]),
        "object_ids": [str(value) for value in source.get("object_ids", [])],
        "selected_object_ids": [str(value) for value in source.get("object_ids", [])],
        "selected_root_id": str(source["instance_id"]),
        "selected_hand_sides": sides,
        "frame_range": [int(source["frame_range"][0]), int(source["frame_range"][1])],
        "primitive_mocap_frame_count": int(len(frame_ids)),
        "primitive_frame_range": [int(frame_ids[0]), int(frame_ids[-1])],
        "primitive_frame_count": int(len(frame_ids)),
        "official_timeline_positions": timeline_positions.tolist(),
        "selected_frame_ids": selected,
        "selected_timeline_positions": selected_positions,
        "selected_frame_count": len(selected),
        "motion_frame_ranges": [
            [int(item["start_frame"]), int(item["end_frame"])]
            for item in temporal.get("motion_components", [])
        ],
        "temporal": temporal,
    }


def build_selection(args: argparse.Namespace) -> dict[str, Any]:
    source_index = args.single_index.resolve()
    payload = json.loads(source_index.read_text(encoding="utf-8"))
    expected_schema = "ref2dex_oakink2_single_object_segment_index_v1"
    if payload.get("schema_name") != expected_schema:
        raise ValueError(f"expected {expected_schema}, got {payload.get('schema_name')!r}")
    source_rows = list(payload.get("segments", []))
    if args.limit is not None:
        source_rows = source_rows[: int(args.limit)]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        groups[str(row["sequence"])].append(row)

    stage3_map = _load_stage3_map(args.stage3_root.resolve())
    config = TemporalSegmentationConfig(
        window_radius=3,
        window_translation_m=0.002,
        window_rotation_deg=2.0,
        part_consensus_ratio=0.60,
        motion_max_gap=1,
        motion_min_evidence_frames=3,
        contact_threshold_m=0.02,
        contact_max_gap=2,
    )
    config.validate()
    selected: list[dict[str, Any]] = []
    rejected = Counter()
    failures: list[dict[str, str]] = []
    total_distance_reads = 0
    started = _now()
    for group_index, (sequence, rows) in enumerate(sorted(groups.items()), 1):
        annotation_path = args.annotation_root.resolve() / f"{sequence}.pkl"
        try:
            with annotation_path.open("rb") as stream:
                annotation = pickle.load(stream)
            distance_store = _Stage3DistanceStore(stage3_map, sequence)
            for source in rows:
                try:
                    primitive_mocap = _primitive_mocap_frames(annotation, source["frame_range_def"])
                    frame_ids, timeline_positions = _official_30hz_timeline(
                        primitive_mocap, annotation["frame_id_list"]
                    )
                    object_ids = [str(value) for value in source.get("object_ids", [])]
                    pose_by_part = {
                        object_id: {
                            int(frame): np.asarray(annotation["obj_transf"][object_id][int(frame)], dtype=np.float32)
                            for frame in frame_ids.tolist()
                            if object_id in annotation.get("obj_transf", {})
                            and int(frame) in annotation["obj_transf"][object_id]
                        }
                        for object_id in object_ids
                    }

                    distance_calls = 0

                    def provide_distances(wanted: Sequence[int]) -> Mapping[int, float]:
                        nonlocal distance_calls, total_distance_reads
                        distance_calls += 1
                        total_distance_reads += len(wanted)
                        return distance_store.distances(object_ids, wanted)

                    temporal = segment_motion_then_contact(
                        pose_by_part,
                        frame_ids.tolist(),
                        provide_distances,
                        config,
                        timeline_positions=timeline_positions.tolist(),
                    )
                    if temporal["status"] != "selected":
                        rejected[str(temporal["status"])] += 1
                        continue
                    if int(temporal["selected_frame_count"]) < 2:
                        rejected["less_than_two_selected_frames"] += 1
                        continue
                    row = _selection_row(
                        source,
                        frame_ids=frame_ids,
                        timeline_positions=timeline_positions,
                        temporal={**temporal, "distance_provider_call_count": distance_calls},
                        ordinal=len(selected),
                    )
                    selected.append(row)
                except Exception as exc:  # keep a complete failure inventory
                    failures.append({
                        "sequence": sequence,
                        "frame_range_def": str(source.get("frame_range_def")),
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            if group_index % 10 == 0 or group_index == len(groups):
                print(
                    json.dumps({"selection_sequences": group_index, "total_sequences": len(groups),
                                "selected_segments": len(selected), "failures": len(failures)},
                               ensure_ascii=False),
                    flush=True,
                )
        except Exception as exc:
            failures.append({"sequence": sequence, "error": f"{type(exc).__name__}: {exc}"})

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection_payload = {
        "schema_name": SELECTION_SCHEMA,
        "schema_version": "1.0.0",
        "source": "OakInk2 single-object primitive index + official 30 Hz frame_id_list + Stage3 distances",
        "created_at": started,
        "modification_version": MODIFICATION_VERSION,
        "source_index": _snapshot(source_index),
        "annotation_root": str(args.annotation_root.resolve()),
        "stage3_root": str(args.stage3_root.resolve()),
        "selection_policy": {
            "object_part_tree": "input index already contains exactly one physical root per primitive",
            "motion_first": True,
            "official_timeline": "annotation frame_id_list intersection with primitive raw MANO IDs",
            "source_mocap_fps": SOURCE_FPS,
            "target_fps": TARGET_FPS,
            "window_radius_frames": 3,
            "window_translation_m": 0.002,
            "window_rotation_deg": 2.0,
            "part_consensus_ratio": 0.60,
            "motion_max_gap": 1,
            "motion_min_evidence_frames": 3,
            "hand_object_distance_strict_lt_m": 0.02,
            "contact_max_gap": 2,
            "hands": "left+right Stage3 union; no side ID in model input",
        },
        "input_segment_count": len(source_rows),
        "segment_count": len(selected),
        "selected_frame_count": int(sum(int(row["selected_frame_count"]) for row in selected)),
        "rejected_counts": dict(rejected),
        "distance_frame_queries": int(total_distance_reads),
        "failures": failures,
        "segments": selected,
        "conclusion": "SUPPORTED" if not failures else "INCONCLUSIVE",
    }
    _write_json(output / "index.json", selection_payload)
    _write_json(output / "manifest.json", {key: selection_payload[key] for key in (
        "schema_name", "schema_version", "source", "selection_policy", "input_segment_count",
        "segment_count", "selected_frame_count", "rejected_counts", "failures", "conclusion",
    )})
    commit, dirty = _git_state()
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "oakink2_v1_4_formal_selection",
        "run_id": output.name,
        "run_status": "COMPLETED" if not failures else "FAILED",
        "started_at": started,
        "completed_at": _now(),
        "modification_version": MODIFICATION_VERSION,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "inputs": {
            "single_index": selection_payload["source_index"],
            "annotation_root": str(args.annotation_root.resolve()),
            "stage3_root": str(args.stage3_root.resolve()),
        },
        "parameters": selection_payload["selection_policy"],
        "outputs": {"index": str((output / "index.json").resolve()), "manifest": str((output / "manifest.json").resolve())},
        "counts": {
            "input_segments": len(source_rows),
            "selected_segments": len(selected),
            "selected_frames": selection_payload["selected_frame_count"],
            "failures": len(failures),
        },
        "conclusion": selection_payload["conclusion"],
    }
    _write_json(output / "run_manifest.json", run_manifest)
    print(json.dumps(run_manifest["counts"], ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError(f"selection completed with {len(failures)} failures")
    return selection_payload


class _ManoReconstructor:
    def __init__(self, mano_root: Path, device: torch.device, batch_size: int = 128) -> None:
        from manotorch.manolayer import ManoLayer

        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._staging = tempfile.TemporaryDirectory(prefix="oakink2_mano_export_")
        (Path(self._staging.name) / "models").symlink_to(mano_root.resolve(), target_is_directory=True)
        self.layers = {
            side: ManoLayer(
                mano_assets_root=self._staging.name,
                rot_mode="quat",
                side=side,
                center_idx=0,
                use_pca=False,
                flat_hand_mean=True,
            ).to(device).eval()
            for side in ("left", "right")
        }
        self.faces = {
            side: np.asarray(layer.th_faces.detach().cpu().numpy(), dtype=np.int32)
            for side, layer in self.layers.items()
        }

    def reconstruct(self, raw_mano: Mapping[int, Mapping[str, Any]], frame_ids: Sequence[int], side: str) -> tuple[np.ndarray, np.ndarray]:
        prefix = "lh" if side == "left" else "rh"
        poses, betas, translations = [], [], []
        for frame_id in frame_ids:
            row = raw_mano[int(frame_id)]
            pose = np.asarray(row[f"{prefix}__pose_coeffs"], dtype=np.float32)
            beta = np.asarray(row[f"{prefix}__betas"], dtype=np.float32)
            translation = np.asarray(row[f"{prefix}__tsl"], dtype=np.float32)
            if pose.shape != (1, 16, 4) or beta.shape != (1, 10) or translation.shape != (1, 3):
                raise ValueError(f"invalid {side} MANO parameters at frame {frame_id}")
            poses.append(pose)
            betas.append(beta)
            translations.append(translation)
        pose_tensor = torch.from_numpy(np.concatenate(poses)).to(self.device)
        beta_tensor = torch.from_numpy(np.concatenate(betas)).to(self.device)
        translation_tensor = torch.from_numpy(np.concatenate(translations)).to(self.device)
        if not np.allclose(np.linalg.norm(pose_tensor.detach().cpu().numpy(), axis=-1), 1.0, atol=1e-3):
            raise ValueError(f"{side} MANO quaternion norm outside tolerance")
        vertices, joints = [], []
        layer = self.layers[side]
        with torch.inference_mode():
            for start in range(0, len(frame_ids), self.batch_size):
                stop = min(len(frame_ids), start + self.batch_size)
                output = layer(
                    pose_coeffs=pose_tensor[start:stop],
                    betas=beta_tensor[start:stop],
                )
                vertices.append((output.verts + translation_tensor[start:stop, None]).detach().cpu().numpy())
                joints.append((output.joints + translation_tensor[start:stop, None]).detach().cpu().numpy())
        verts = np.ascontiguousarray(np.concatenate(vertices).astype(np.float32))
        jnts = np.ascontiguousarray(np.concatenate(joints).astype(np.float32))
        if verts.shape != (len(frame_ids), 778, 3) or jnts.ndim != 3 or not np.isfinite(verts).all() or not np.isfinite(jnts).all():
            raise ValueError(f"invalid reconstructed {side} MANO arrays: {verts.shape}/{jnts.shape}")
        return verts, jnts


class _InspireRetargeter:
    def __init__(self, dex_root: Path, surface_seed: int = SURFACE_SEED) -> None:
        import xml.etree.ElementTree as ET

        from dex_retargeting.retargeting_config import RetargetingConfig
        from src.task.ObjectInteractionCm.research.hand_region_sampling.run import (
            _build_inspire_pool,
            _sample_uniform_surface,
        )
        from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel
        from src.task.ObjectInteractionCm.tools.data.retarget_stage4_bilateral_inspire import (
            canonical_cloud_to_visual_local,
        )

        self._fk_surface = __import__(
            "src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache",
            fromlist=["_fk_surface"],
        )._fk_surface
        self.dex_root = dex_root.resolve()
        source_urdf = REPO_ROOT.parent / "dexplore" / "dexplore" / "data" / "assets" / "inspire_hand_new" / "inspire_hand_right.urdf"
        if not source_urdf.is_file():
            raise FileNotFoundError(f"Inspire surface URDF unavailable: {source_urdf}")
        tree = ET.parse(str(source_urdf))
        active = []
        for joint in tree.getroot().findall("joint"):
            if joint.get("type") != "fixed":
                active.append(joint.get("name"))
            if joint.get("type") == "continuous":
                joint.set("type", "revolute")
                limit = joint.find("limit") or ET.SubElement(joint, "limit")
                limit.set("lower", "-1000000")
                limit.set("upper", "1000000")
                limit.set("effort", limit.get("effort", "1000"))
                limit.set("velocity", limit.get("velocity", "3.14"))
        for mesh in tree.getroot().iter("mesh"):
            filename = mesh.get("filename")
            if filename and not Path(filename).is_absolute():
                mesh.set("filename", str((source_urdf.parent / filename).resolve()))
        self._temp = tempfile.TemporaryDirectory(prefix="oakink2_inspire_urdf_")
        patched = Path(self._temp.name) / source_urdf.name
        tree.write(str(patched))
        self._inv_native = np.argsort(np.asarray([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 16, 17, 14, 15, 6, 7, 8, 9]))
        RetargetingConfig.set_default_urdf_dir(self.dex_root / "assets" / "robots" / "hands")
        self.models = {}
        self.samplings = {}
        self.configs = {}
        for side in ("left", "right"):
            urdf = REPO_ROOT.parent / "dexplore" / "dexplore" / "data" / "assets" / "inspire_hand_new" / f"inspire_hand_{side}.urdf"
            try:
                model = InspireUrdfModel(urdf)
            except ValueError:
                if side != "left":
                    raise
                model = InspireUrdfModel(source_urdf)
                urdf = source_urdf
            pool, _ = _build_inspire_pool(urdf, 0.20)
            cloud = _sample_uniform_surface(pool, int(surface_seed), HAND_POINTS_PER_SIDE)
            local_points, local_normals = canonical_cloud_to_visual_local(
                model, cloud.points, cloud.normals, cloud.source_visual_ids
            )
            self.models[side] = model
            self.samplings[side] = (local_points, local_normals, cloud.source_visual_ids)
            self.configs[side] = (RetargetingConfig.load_from_file(
                self.dex_root / "dex_retargeting" / "configs" / "offline" / f"inspire_hand_{side}.yml",
                override={
                    "urdf_path": str(patched),
                    "add_dummy_free_joint": False,
                    "target_joint_names": active,
                    "ignore_mimic_joint": True,
                },
            ).build())

    def convert(self, vertices: np.ndarray, side: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        retargeter = self.configs[side]
        retargeter.reset()
        tip_targets = np.asarray(vertices[:, TIP_IDS[side]], dtype=np.float32)
        q_urdf = np.asarray([retargeter.retarget(value) for value in tip_targets], dtype=np.float32)
        q_native = q_urdf[:, self._inv_native]
        points, normals = self._fk_surface(self.models[side], q_native, *self.samplings[side])
        predicted = np.asarray([self.models[side].link_transforms(self.models[side].qpos_to_urdf_order(q)) for q in q_native])
        # The target link order comes from dex-retargeting; using qpos output
        # for the engineering summary avoids making an unverified joint map.
        error = np.linalg.norm(tip_targets[:, None] - tip_targets[:, None], axis=-1)
        del predicted, error
        if not np.isfinite(points).all() or not np.isfinite(normals).all():
            raise ValueError(f"non-finite Inspire surface for {side}")
        return points.astype(np.float32), normals.astype(np.float32), {
            "frames": float(len(points)),
            "qpos_max_abs": float(np.max(np.abs(q_native))),
        }


class _Stage3GeometryStore:
    def __init__(self, mapping: Mapping[tuple[str, str, str], Path], sequence: str) -> None:
        self.mapping = mapping
        self.sequence = sequence
        self.data: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    def _load(self, object_id: str, side: str) -> dict[str, np.ndarray] | None:
        key = (object_id, side)
        if key in self.data:
            return self.data[key]
        path = self.mapping.get((self.sequence, object_id, side))
        if path is None or not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as value:
            data = {name: np.asarray(value[name]) for name in (
                "raw_frame_id", "obj_points", "obj_normals", "obj_root_pose_world"
            )}
        self.data[key] = data
        return data

    def parts(self, object_ids: Sequence[str], frame_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local_points, local_normals, poses, part_ids = [], [], [], []
        for part_index, object_id in enumerate(object_ids):
            candidates = [
                value for value in (self._load(str(object_id), "right"), self._load(str(object_id), "left"))
                if value is not None
            ]
            wanted = np.asarray(frame_ids, dtype=np.int64)
            data = None
            selected = None
            for candidate in candidates:
                source_ids = np.asarray(candidate["raw_frame_id"], dtype=np.int64)
                positions = np.searchsorted(source_ids, wanted)
                valid = (positions < len(source_ids)) & (
                    source_ids[np.minimum(positions, len(source_ids) - 1)] == wanted
                )
                if np.all(valid):
                    data, selected = candidate, positions.astype(np.int64)
                    break
            if data is None or selected is None:
                raise FileNotFoundError(f"no Stage3 geometry for {self.sequence}/{object_id}")
            object_points = np.asarray(data["obj_points"], dtype=np.float32)
            object_normals = np.asarray(data["obj_normals"], dtype=np.float32)
            if object_points.shape != (OBJECT_POINTS, 3) or object_normals.shape != object_points.shape:
                raise ValueError(f"invalid static Stage3 object pool: {self.sequence}/{object_id}")
            local_points.append(np.broadcast_to(object_points[None], (len(frame_ids), OBJECT_POINTS, 3)).copy())
            local_normals.append(np.broadcast_to(object_normals[None], (len(frame_ids), OBJECT_POINTS, 3)).copy())
            poses.append(np.asarray(data["obj_root_pose_world"][selected], dtype=np.float32))
            part_ids.append(np.full((OBJECT_POINTS,), part_index, dtype=np.int32))
        if not local_points:
            raise ValueError("selected segment has no object parts")
        return (
            np.concatenate(local_points, axis=1),
            np.concatenate(local_normals, axis=1),
            np.stack(poses, axis=1),
            np.concatenate(part_ids, axis=0),
        )


def _sample_object_pool(
    local_points: np.ndarray,
    local_normals: np.ndarray,
    poses: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if local_points.ndim != 3 or local_points.shape != local_normals.shape:
        raise ValueError(f"invalid Stage3 object arrays: {local_points.shape}/{local_normals.shape}")
    frame_count, point_count, _ = local_points.shape
    if poses.shape != (frame_count, point_count // OBJECT_POINTS, 4, 4):
        raise ValueError(f"invalid Stage3 object poses: {poses.shape}")
    world_parts, normal_parts = [], []
    for part_index in range(poses.shape[1]):
        pose = poses[:, part_index]
        if not np.all([_is_se3(value) for value in pose[:: max(1, len(pose) // 16)]]):
            raise ValueError("invalid object SE(3) in Stage3")
        start = part_index * OBJECT_POINTS
        stop = start + OBJECT_POINTS
        rotation = pose[:, :3, :3]
        translation = pose[:, :3, 3]
        world_parts.append(np.einsum("tij,tpj->tpi", rotation, local_points[:, start:stop]) + translation[:, None, :])
        normal_parts.append(np.einsum("tij,tpj->tpi", rotation, local_normals[:, start:stop]))
    world = np.concatenate(world_parts, axis=1)
    normals = np.concatenate(normal_parts, axis=1)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(world.shape[1], size=OBJECT_POINTS, replace=world.shape[1] < OBJECT_POINTS)
    root_pose = np.asarray(poses[:, 0], dtype=np.float32)
    return (
        np.ascontiguousarray(world[:, indices].astype(np.float32)),
        np.ascontiguousarray(normals[:, indices].astype(np.float32)),
        root_pose,
        np.asarray(indices, dtype=np.int32),
    )


def _write_knn(geometry: Path, device: torch.device, frame_batch: int, object_chunk: int) -> None:
    objects = np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r")
    hands = np.load(geometry / "hand_points_world.npy", mmap_mode="r")
    frames = len(objects)
    indices = np.lib.format.open_memmap(
        geometry / "obj_knn_indices.npy", mode="w+", dtype=np.uint16,
        shape=(frames, OBJECT_POINTS, KNN_K),
    )
    candidate_2cm = np.lib.format.open_memmap(
        geometry / "obj_candidate_mask_2cm.npy", mode="w+", dtype=np.bool_,
        shape=(frames, OBJECT_POINTS),
    )
    candidate_5cm = np.lib.format.open_memmap(
        geometry / "obj_candidate_mask_5cm.npy", mode="w+", dtype=np.bool_,
        shape=(frames, OBJECT_POINTS),
    )
    supervision = np.lib.format.open_memmap(
        geometry / "hand_supervision_mask_2cm.npy", mode="w+", dtype=np.bool_,
        shape=(frames, HAND_POINTS),
    )
    minimum = np.lib.format.open_memmap(
        geometry / "hand_min_object_distance_m.npy", mode="w+", dtype=np.float32,
        shape=(frames,),
    )
    try:
        with torch.inference_mode():
            for start in range(0, frames, max(1, int(frame_batch))):
                stop = min(frames, start + max(1, int(frame_batch)))
                object_tensor = torch.as_tensor(np.array(objects[start:stop], dtype=np.float32, copy=True), device=device)
                hand_tensor = torch.as_tensor(np.array(hands[start:stop], dtype=np.float32, copy=True), device=device)
                frame_min = torch.full((stop - start,), float("inf"), device=device)
                for point_start in range(0, OBJECT_POINTS, max(KNN_K, int(object_chunk))):
                    point_stop = min(OBJECT_POINTS, point_start + max(KNN_K, int(object_chunk)))
                    distance = torch.cdist(object_tensor[:, point_start:point_stop], hand_tensor)
                    values, ids = torch.topk(distance, k=KNN_K, dim=-1, largest=False, sorted=True)
                    indices[start:stop, point_start:point_stop] = ids.cpu().numpy().astype(np.uint16)
                    candidate_2cm[start:stop, point_start:point_stop] = (values[..., 0] < RADIUS_M).cpu().numpy()
                    candidate_5cm[start:stop, point_start:point_stop] = (values[..., 0] < CANDIDATE_RADIUS_M).cpu().numpy()
                    frame_min = torch.minimum(frame_min, values[..., 0].amin(dim=-1))
                for hand_start in range(0, HAND_POINTS, max(KNN_K, int(object_chunk))):
                    hand_stop = min(HAND_POINTS, hand_start + max(KNN_K, int(object_chunk)))
                    distance = torch.cdist(hand_tensor[:, hand_start:hand_stop], object_tensor).amin(dim=-1)
                    supervision[start:stop, hand_start:hand_stop] = (distance < RADIUS_M).cpu().numpy()
                    frame_min = torch.minimum(frame_min, distance.amin(dim=-1))
                minimum[start:stop] = frame_min.cpu().numpy().astype(np.float32)
    finally:
        for value in (indices, candidate_2cm, candidate_5cm, supervision, minimum):
            value.flush()
        del indices, candidate_2cm, candidate_5cm, supervision, minimum


def _validate_geometry(sequence: Path) -> dict[str, Any]:
    geometry = sequence / "geometry"
    manifest = json.loads((geometry / "manifest.json").read_text(encoding="utf-8"))
    frames = int(manifest["frame_count"])
    expected = {
        "obj_points_pool_world.npy": (frames, OBJECT_POINTS, 3),
        "obj_normals_pool_world.npy": (frames, OBJECT_POINTS, 3),
        "obj_pose_world.npy": (frames, 4, 4),
        "source_frame_id.npy": (frames,),
        "frame_time.npy": (frames,),
        "hand_points_world.npy": (frames, HAND_POINTS, 3),
        "hand_normals_world.npy": (frames, HAND_POINTS, 3),
        "knn_hand_points_world.npy": (frames, HAND_POINTS, 3),
        "knn_hand_normals_world.npy": (frames, HAND_POINTS, 3),
        "obj_knn_indices.npy": (frames, OBJECT_POINTS, KNN_K),
        "obj_candidate_mask_2cm.npy": (frames, OBJECT_POINTS),
        "obj_candidate_mask_5cm.npy": (frames, OBJECT_POINTS),
        "hand_supervision_mask_2cm.npy": (frames, HAND_POINTS),
        "hand_min_object_distance_m.npy": (frames,),
        "obj_point_id.npy": (OBJECT_POINTS,),
    }
    for name, shape in expected.items():
        path = geometry / name
        if not path.is_file():
            raise FileNotFoundError(path)
        value = np.load(path, mmap_mode="r")
        if value.shape != shape:
            raise ValueError(f"{path}: expected {shape}, got {value.shape}")
        if not np.isfinite(np.asarray(value[: min(len(value), 2)])).all() and value.dtype.kind == "f":
            raise ValueError(f"non-finite geometry: {path}")
    max_index = int(np.asarray(np.load(geometry / "obj_knn_indices.npy", mmap_mode="r")).max())
    if max_index >= HAND_POINTS:
        raise ValueError(f"out-of-range KNN index: {max_index}")
    return {"id": manifest["sequence_id"], "frames": frames, "path": str(sequence.resolve()), "min_knn_index": max_index}


def _export_one(
    row: Mapping[str, Any],
    *,
    annotation: Mapping[str, Any],
    stage3: _Stage3GeometryStore,
    reconstructor: _ManoReconstructor,
    retargeter: _InspireRetargeter,
    output_root: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sequence_id = str(row["id"])
    sequence_suffix = sequence_id[len("oakink2/"):] if sequence_id.startswith("oakink2/") else sequence_id
    relative = Path("sequences") / "train" / "inspire_f1" / "oakink2" / _safe_id(sequence_suffix)
    destination = output_root / relative
    manifest_path = destination / "geometry" / "manifest.json"
    if manifest_path.is_file() and args.resume:
        return _validate_geometry(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing sequence: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"incomplete output exists; inspect before rerun: {partial}")
    geometry = partial / "geometry"
    geometry.mkdir(parents=True, exist_ok=False)
    frame_ids = np.asarray(row["selected_frame_ids"], dtype=np.int64)
    timeline_positions = np.asarray(row["selected_timeline_positions"], dtype=np.int64)
    object_ids = [str(value) for value in row["selected_object_ids"]]
    local_points, local_normals, poses, _ = stage3.parts(object_ids, frame_ids)
    seed = int.from_bytes(hashlib.blake2b(sequence_id.encode(), digest_size=8).digest(), "little") & 0xFFFFFFFF
    objects, normals, object_pose, point_ids = _sample_object_pool(local_points, local_normals, poses, seed=seed)
    np.save(geometry / "obj_points_pool_world.npy", objects)
    np.save(geometry / "obj_normals_pool_world.npy", normals)
    np.save(geometry / "obj_pose_world.npy", object_pose)
    np.save(geometry / "obj_point_id.npy", point_ids)
    np.save(geometry / "source_frame_id.npy", frame_ids.astype(np.int32))
    np.save(geometry / "frame_time.npy", (timeline_positions.astype(np.float32) / TARGET_FPS))

    hands, hand_normals, retarget_stats = [], [], {}
    for side in ("left", "right"):
        vertices, _ = reconstructor.reconstruct(annotation["raw_mano"], frame_ids.tolist(), side)
        points, side_normals, stats = retargeter.convert(vertices, side)
        hands.append(points)
        hand_normals.append(side_normals)
        retarget_stats[side] = stats
    hand_points = np.ascontiguousarray(np.concatenate(hands, axis=1).astype(np.float32))
    hand_normals_world = np.ascontiguousarray(np.concatenate(hand_normals, axis=1).astype(np.float32))
    np.save(geometry / "hand_points_world.npy", hand_points)
    np.save(geometry / "hand_normals_world.npy", hand_normals_world)
    np.save(geometry / "knn_hand_points_world.npy", hand_points)
    np.save(geometry / "knn_hand_normals_world.npy", hand_normals_world)
    _write_knn(geometry, device, args.knn_frame_batch, args.knn_object_chunk)
    manifest = {
        "schema_name": CACHE_SCHEMA,
        "schema_version": "1.0.0",
        "modification_version": MODIFICATION_VERSION,
        "sequence_id": sequence_id,
        "dataset": "oakink2",
        "source": "inspire_f1",
        "source_type": "oakink2_official_quaternion_mano_to_inspire_geometric",
        "split": "train",
        "coordinate_frame": "object_pose_t",
        "object_representation": "single_root_articulated_world_points_with_reference_pose",
        "hand_side": "bilateral_merged_left_then_right",
        "merged_hand_sides": True,
        "frame_count": int(len(frame_ids)),
        "object_pool_points": OBJECT_POINTS,
        "hand_points": HAND_POINTS,
        "effective_fps": TARGET_FPS,
        "source_fps": SOURCE_FPS,
        "source_frame_id": "official 30 Hz annotation.frame_id_list raw mocap IDs",
        "frame_time": "official RGB timeline ordinal / 30",
        "selected_root_id": row["selected_root_id"],
        "selected_object_ids": object_ids,
        "primitive": row.get("primitive"),
        "motion_frame_ranges": row.get("motion_frame_ranges", []),
        "selection_index_id": sequence_id,
        "surface_sampling": {
            "method": "global Inspire visual triangle area uniform",
            "seed": SURFACE_SEED,
            "points_per_side": HAND_POINTS_PER_SIDE,
            "space": "visual_mesh_local_then_single_fk",
        },
        "offline_knn": {
            "k": KNN_K,
            "hand_points": HAND_POINTS,
            "distance_radius_m": RADIUS_M,
            "candidate_radius_m": CANDIDATE_RADIUS_M,
            "index_dtype": "uint16",
        },
        "retarget": retarget_stats,
    }
    _write_json(geometry / "manifest.json", manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, destination)
    return _validate_geometry(destination)


def export_cache(args: argparse.Namespace) -> dict[str, Any]:
    selection_path = args.selection_index.resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_name") != SELECTION_SCHEMA:
        raise ValueError(f"expected {SELECTION_SCHEMA}, got {selection.get('schema_name')!r}")
    rows = list(selection.get("segments", []))
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False) if not output_root.exists() else None
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    commit, dirty = _git_state()
    run_id = args.run_id or output_root.name
    run_manifest_path = output_root / "run_manifest.json"
    run_manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "oakink2_v1_4_inspire_cache_export",
        "run_id": run_id,
        "run_status": "STARTED",
        "started_at": _now(),
        "modification_version": MODIFICATION_VERSION,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": [sys.executable, *sys.argv],
        "device": str(device),
        "inputs": {"selection_index": _snapshot(selection_path)},
        "outputs": {"root": str(output_root), "index": "PENDING", "cache_manifest": "PENDING"},
        "expected": {"segments": len(rows)},
        "completed_segments": 0,
        "completed_frames": 0,
        "failures": [],
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(run_manifest_path, run_manifest)
    stage3_map = _load_stage3_map(args.stage3_root.resolve())
    mano = _ManoReconstructor(args.mano_root.resolve(), device, args.mano_batch_size)
    inspire = _InspireRetargeter(args.dex_root.resolve(), SURFACE_SEED)
    records, failures = [], []
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["sequence"])].append(row)
    started_time = time.time()
    for seq_index, (sequence, seq_rows) in enumerate(sorted(grouped.items()), 1):
        try:
            with (args.annotation_root.resolve() / f"{sequence}.pkl").open("rb") as stream:
                annotation = pickle.load(stream)
            stage3 = _Stage3GeometryStore(stage3_map, sequence)
            for row in seq_rows:
                try:
                    result = _export_one(
                        row, annotation=annotation, stage3=stage3, reconstructor=mano,
                        retargeter=inspire, output_root=output_root, device=device, args=args,
                    )
                    records.append({**result, "selection_id": row["id"], "dataset": "oakink2", "split": "train"})
                    run_manifest["completed_segments"] = len(records)
                    run_manifest["completed_frames"] += int(result["frames"])
                    _write_json(run_manifest_path, run_manifest)
                except Exception as exc:
                    failures.append({"selection_id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"})
                    run_manifest["failures"] = failures
                    _write_json(run_manifest_path, run_manifest)
                    print(json.dumps({"status": "FAILED", "selection_id": row.get("id"), "error": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
            print(json.dumps({"export_sequences": seq_index, "total_sequences": len(grouped), "completed_segments": len(records), "failures": len(failures), "elapsed_s": round(time.time() - started_time, 1)}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failures.append({"sequence": sequence, "error": f"{type(exc).__name__}: {exc}"})
            run_manifest["failures"] = failures
            _write_json(run_manifest_path, run_manifest)

    if failures:
        run_manifest.update({"run_status": "FAILED", "completed_at": _now(), "conclusion": "INCONCLUSIVE"})
        _write_json(run_manifest_path, run_manifest)
        raise RuntimeError(f"OakInk2 export failed for {len(failures)} records")
    records.sort(key=lambda value: str(value["selection_id"]))
    index_entries = [{
        "source": "inspire_f1",
        "id": record["selection_id"],
        "path": record["path"],
        "dataset": "oakink2",
        "split": "train",
        "frame_count": int(record["frames"]),
    } for record in records]
    index = {
        "schema_name": INDEX_SCHEMA,
        "schema_version": "1.0.0",
        "created_at": _now(),
        "modification_version": MODIFICATION_VERSION,
        "source": "OakInk2 V1.4 independent Inspire cache",
        "source_probability": {"inspire_f1": 1.0},
        "object_pool_points": OBJECT_POINTS,
        "model_object_points": 1024,
        "hand_points_per_stream": HAND_POINTS,
        "max_union_hand_points": HAND_POINTS,
        "knn_k": KNN_K,
        "split_policy": {"oakink2": "single-object primitive after official motion/contact filtering, train only"},
        "sequences": {"train": index_entries, "val": [], "test": []},
        "counts": {"train": len(index_entries), "val": 0, "test": 0, "frames": sum(int(value["frame_count"]) for value in index_entries)},
        "selection_index": str(selection_path),
    }
    _write_json(output_root / "index.json", index)
    cache_manifest = {
        "schema_name": "ref2dex_object_interaction_cm_oakink2_inspire_cache_manifest",
        "schema_version": "1.0.0",
        "modification_version": MODIFICATION_VERSION,
        "created_at": _now(),
        "storage": "NAS" if "/mnt/ugreen_nas/" in str(output_root) else "unknown",
        "roots": [str(output_root.resolve())],
        "sequence_counts": {"oakink2": len(index_entries)},
        "total_sequences": len(index_entries),
        "total_frames": int(index["counts"]["frames"]),
        "hand_contract": "bilateral merged left_then_right, 3076 points, Inspire geometric surface",
        "object_contract": "4096 point pool; one object_part_tree root per primitive",
        "effective_fps": TARGET_FPS,
        "selection_index": str(selection_path),
        "validation": {"bad_count": 0, "bad_examples": []},
    }
    _write_json(output_root / "cache_manifest.json", cache_manifest)
    run_manifest.update({
        "run_status": "COMPLETED",
        "completed_at": _now(),
        "outputs": {
            "root": str(output_root),
            "index": str((output_root / "index.json").resolve()),
            "cache_manifest": str((output_root / "cache_manifest.json").resolve()),
        },
        "conclusion": "SUPPORTED",
    })
    _write_json(run_manifest_path, run_manifest)
    print(json.dumps({"segments": len(index_entries), "frames": index["counts"]["frames"], "output": str(output_root)}, ensure_ascii=False), flush=True)
    return {"index": index, "cache_manifest": cache_manifest, "run_manifest": run_manifest}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    select = sub.add_parser("select")
    select.add_argument("--single-index", type=Path, required=True)
    select.add_argument("--annotation-root", type=Path, required=True)
    select.add_argument("--stage3-root", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)
    select.add_argument("--limit", type=int, default=None)
    export = sub.add_parser("export")
    export.add_argument("--selection-index", type=Path, required=True)
    export.add_argument("--annotation-root", type=Path, required=True)
    export.add_argument("--stage3-root", type=Path, required=True)
    export.add_argument("--mano-root", type=Path, required=True)
    export.add_argument("--dex-root", type=Path, required=True)
    export.add_argument("--output-root", type=Path, required=True)
    export.add_argument("--device", default="cuda:2")
    export.add_argument("--mano-batch-size", type=int, default=128)
    export.add_argument("--knn-frame-batch", type=int, default=2)
    export.add_argument("--knn-object-chunk", type=int, default=512)
    export.add_argument("--limit", type=int, default=None)
    export.add_argument("--run-id", default=None)
    export.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "select":
        build_selection(args)
    else:
        export_cache(args)


if __name__ == "__main__":
    main()
