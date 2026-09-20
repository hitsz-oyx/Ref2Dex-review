"""Interactive viewer for OakInk2 active-tool segment selections.

The viewer reads the compact v1.1 selection index lazily.  Human hands are
reconstructed from the original bilateral MANO annotations, while object
samples are read from Stage3 and transformed back to OakInk2 native world
coordinates with the original per-frame object poses.  No source index,
annotation, Stage3 array, cache, or split is modified.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool import (
    _load_stage3_map,
    _resolve_active_roots,
    _roots,
)


REPO = Path(__file__).resolve().parents[5]
WORK_VERSION = "V1.4.4"
INDEX_SCHEMA = "ref2dex_oakink2_active_tool_frame_selection_v1"
INDEX_VERSION = "1.1.0"

CATEGORY_NAMES = {
    "new": "本次新增",
    "bilateral": "双主物体拆分",
    "override": "语义修复",
    "static": "静止排除对照",
    "all": "全部保留",
}
FRAME_MODE_NAMES = {"selected": "最终保留帧", "primitive": "primitive 全帧"}

SELECTED_OBJECT_COLOR = np.asarray([245, 150, 55], dtype=np.uint8)
CONTEXT_OBJECT_COLOR = np.asarray([150, 155, 165], dtype=np.uint8)
LEFT_HAND_COLOR = np.asarray([70, 145, 245], dtype=np.uint8)
RIGHT_HAND_COLOR = np.asarray([65, 205, 125], dtype=np.uint8)
NON_SELECTED_HAND_COLOR = np.asarray([175, 180, 190], dtype=np.uint8)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot(path: Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "sha256": _sha256(path)}


def _git_state() -> Tuple[str, bool]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip())
    return commit, dirty


def _bounds_from_frame_def(value: str) -> Tuple[int, int]:
    ranges = ast.literal_eval(value)
    bounds = [item for item in ranges if item is not None]
    if not bounds:
        raise ValueError(f"frame_range_def has no bounds: {value!r}")
    return min(int(item[0]) for item in bounds), max(int(item[1]) for item in bounds)


def _transform_points(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    pose = np.asarray(pose, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or pose.shape != (4, 4):
        raise ValueError(f"invalid point/pose shapes: {points.shape}, {pose.shape}")
    return np.ascontiguousarray(points @ pose[:3, :3].T + pose[:3, 3])


def _sample_rows(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) <= count:
        return points
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return points[indices]


def _short_sequence(value: str) -> str:
    prefix, _, suffix = value.partition("++seq__")
    token = suffix.split("__", 1)[0][:8] if suffix else value[-8:]
    return f"{prefix}/{token}"


def _segment_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return str(row["sequence"]), str(row["frame_range_def"]), str(row["selected_root_id"])


@dataclass(frozen=True)
class SegmentRecord:
    ordinal: int
    row: dict
    status: str
    new_in_v1_1: bool

    @property
    def id(self) -> str:
        return f"{self.status}:{self.ordinal:04d}"

    @property
    def label(self) -> str:
        tag = "静止" if self.status == "rejected_static" else ("新增" if self.new_in_v1_1 else "保留")
        count = int(self.row.get("selected_frame_count", 0))
        primitive = str(self.row.get("primitive") or "unknown")
        obj = str(self.row.get("selected_object_name") or self.row.get("selected_root_id"))
        return f"[{tag} {self.ordinal:04d}] {_short_sequence(str(self.row['sequence']))} | {primitive} | {obj} | {count} 帧"


def _load_catalog(index_path: Path, previous_index_path: Path) -> Tuple[dict, List[SegmentRecord]]:
    index = _read_json(index_path)
    previous = _read_json(previous_index_path)
    if index.get("schema_name") != INDEX_SCHEMA or index.get("schema_version") != INDEX_VERSION:
        raise ValueError(
            f"expected {INDEX_SCHEMA}@{INDEX_VERSION}, got "
            f"{index.get('schema_name')}@{index.get('schema_version')}"
        )
    if int(index.get("uncertain_count", -1)) != 0:
        raise ValueError("the current visualization index must have uncertain_count=0")
    old_keys = {_segment_key(row) for row in previous.get("segments", [])}
    current_keys = {_segment_key(row) for row in index.get("segments", [])}
    records = [
        SegmentRecord(i, dict(row), "selected", _segment_key(row) not in old_keys)
        for i, row in enumerate(index.get("segments", []))
    ]

    object_root = Path(index["object_root"])
    tree = _read_json(object_root / "object_affordance/object_part_tree.json")
    root_of, members = _roots(tree)
    descriptions = _read_json(object_root / "object_raw/obj_desc.json")
    names = {str(key): str(value.get("obj_name", key)) for key, value in descriptions.items()}
    static_rows: List[dict] = []
    for uncertain in previous.get("uncertain", []):
        sequence = str(uncertain["sequence"])
        frame_def = str(uncertain["frame_range_def"])
        program = _read_json(object_root / "program/program_info" / f"{sequence}.json")
        item = program[frame_def]
        resolution = _resolve_active_roots(
            sequence=sequence,
            frame_def=frame_def,
            item=item,
            root_of=root_of,
            members=members,
            names=names,
        )
        for selected_root in resolution["selected_roots"]:
            key = (sequence, frame_def, selected_root)
            if key in current_keys:
                continue
            selected_ids = sorted(
                object_id for object_id in resolution["object_ids"]
                if root_of.get(object_id, object_id) == selected_root
            )
            static_rows.append({
                "sequence": sequence,
                "frame_range_def": frame_def,
                "primitive": item.get("primitive"),
                "interaction_mode": item.get("interaction_mode"),
                "selected_root_id": selected_root,
                "selected_object_ids": selected_ids,
                "selected_object_name": names.get(selected_root, selected_root),
                "selection_reason": resolution["selection_reason"],
                "resolution_reason": "rejected_by_existing_motion_threshold",
                "original_object_ids": resolution["original_object_ids"],
                "selected_hand_sides": [
                    side for side, roots_for_side in resolution["side_roots"].items()
                    if selected_root in roots_for_side
                ],
                "motion_frame_range": None,
                "selected_frame_ids": [],
                "selected_frame_count": 0,
            })
    start = len(records)
    records.extend(
        SegmentRecord(start + i, row, "rejected_static", True)
        for i, row in enumerate(static_rows)
    )
    return index, records


def _records_for_category(records: Sequence[SegmentRecord], category: str) -> List[SegmentRecord]:
    if category == "new":
        selected = [record for record in records if record.status == "selected" and record.new_in_v1_1]
    elif category == "bilateral":
        selected = [record for record in records if record.status == "selected" and record.new_in_v1_1
                    and record.row.get("selection_reason") == "bilateral_multi_active_split"]
    elif category == "override":
        selected = [record for record in records if record.status == "selected" and record.new_in_v1_1
                    and record.row.get("selection_reason") == "semantic_object_override"]
    elif category == "static":
        selected = [record for record in records if record.status == "rejected_static"]
    elif category == "all":
        selected = [record for record in records if record.status == "selected"]
    else:
        raise ValueError(f"unknown category: {category}")
    if not selected:
        raise ValueError(f"category {category!r} has no segments")
    return selected


class Stage3ObjectCatalog:
    """Lazy fixed object samples from the existing Stage3 outputs."""

    def __init__(self, stage3_root: Path) -> None:
        self.paths = _load_stage3_map(Path(stage3_root))
        self._cache: Dict[Tuple[str, str], np.ndarray] = {}

    def points(self, sequence: str, object_id: str) -> Optional[np.ndarray]:
        key = (sequence, object_id)
        if key in self._cache:
            return self._cache[key]
        path = self.paths.get((sequence, object_id, "right")) or self.paths.get((sequence, object_id, "left"))
        if path is None:
            return None
        with np.load(path, allow_pickle=False) as data:
            points = np.asarray(data["obj_points"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError(f"invalid Stage3 object points: {path} {points.shape}")
        self._cache[key] = points
        return points


class ManoReconstructor:
    """OakInk2 official bilateral quaternion-MANO reconstruction."""

    def __init__(self, mano_root: Path, threads: int) -> None:
        for key, value in {
            "bool": bool, "int": int, "float": float, "complex": complex,
            "object": object, "unicode": str, "str": str,
        }.items():
            if key not in np.__dict__:
                setattr(np, key, value)
        from manotorch.manolayer import ManoLayer

        torch.set_num_threads(int(threads))
        self._staging = tempfile.TemporaryDirectory(prefix="oakink2_segment_mano_")
        (Path(self._staging.name) / "models").symlink_to(Path(mano_root).resolve(), target_is_directory=True)
        self.layers = {
            side: ManoLayer(
                mano_assets_root=self._staging.name,
                rot_mode="quat",
                side=side,
                center_idx=0,
                use_pca=False,
                flat_hand_mean=True,
            ).eval()
            for side in ("left", "right")
        }
        self.faces = {side: np.asarray(layer.th_faces, dtype=np.int32) for side, layer in self.layers.items()}

    def reconstruct(self, raw_mano: Mapping[int, dict], frame_ids: np.ndarray, side: str) -> np.ndarray:
        prefix = "lh" if side == "left" else "rh"
        poses, betas, translations = [], [], []
        for frame_id in frame_ids.tolist():
            row = raw_mano[int(frame_id)]
            pose = np.asarray(row[f"{prefix}__pose_coeffs"], dtype=np.float32)
            beta = np.asarray(row[f"{prefix}__betas"], dtype=np.float32)
            translation = np.asarray(row[f"{prefix}__tsl"], dtype=np.float32)
            if pose.shape != (1, 16, 4) or beta.shape != (1, 10) or translation.shape != (1, 3):
                raise ValueError(f"frame {frame_id} has invalid {side} MANO shapes")
            poses.append(pose)
            betas.append(beta)
            translations.append(translation)
        pose_tensor = torch.from_numpy(np.concatenate(poses))
        beta_tensor = torch.from_numpy(np.concatenate(betas))
        translation_tensor = torch.from_numpy(np.concatenate(translations))
        norms = np.linalg.norm(pose_tensor.numpy(), axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(f"{side} MANO quaternion norm outside tolerance")
        vertices = []
        with torch.no_grad():
            for start in range(0, len(frame_ids), 128):
                output = self.layers[side](
                    pose_coeffs=pose_tensor[start:start + 128],
                    betas=beta_tensor[start:start + 128],
                )
                value = output.verts + translation_tensor[start:start + 128, None]
                vertices.append(value.numpy())
        result = np.ascontiguousarray(np.concatenate(vertices).astype(np.float32))
        if result.shape != (len(frame_ids), 778, 3) or not np.isfinite(result).all():
            raise ValueError(f"invalid reconstructed {side} MANO vertices: {result.shape}")
        return result


@dataclass
class SegmentTrajectory:
    record: SegmentRecord
    frame_ids: np.ndarray
    selected_frame_ids: frozenset
    hands: Dict[str, np.ndarray]
    hand_faces: Dict[str, np.ndarray]
    object_points: Dict[str, np.ndarray]
    object_poses: Mapping[str, Mapping[int, np.ndarray]]
    missing_object_points: Tuple[str, ...]
    mode: str

    @property
    def frame_count(self) -> int:
        return int(len(self.frame_ids))

    def object_world(self, frame_index: int, *, selected: bool, points_per_part: int) -> np.ndarray:
        frame_id = int(self.frame_ids[int(frame_index)])
        selected_ids = set(self.record.row["selected_object_ids"])
        values = []
        for object_id, points in self.object_points.items():
            if (object_id in selected_ids) != bool(selected):
                continue
            poses = self.object_poses.get(object_id, {})
            pose = poses.get(frame_id)
            if pose is None:
                continue
            values.append(_transform_points(_sample_rows(points, points_per_part), np.asarray(pose)))
        if not values:
            return np.empty((0, 3), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(values))


def _load_annotation(annotation_root: Path, sequence: str) -> dict:
    path = Path(annotation_root) / f"{sequence}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def _frame_ids_for_mode(record: SegmentRecord, raw_frame_ids: Iterable[int], mode: str) -> np.ndarray:
    raw = np.asarray(sorted(int(value) for value in raw_frame_ids), dtype=np.int64)
    start, end = _bounds_from_frame_def(str(record.row["frame_range_def"]))
    primitive = raw[(raw >= start) & (raw <= end)]
    if not len(primitive):
        raise ValueError(f"primitive contains no MANO frames: {record.id}")
    if mode == "selected" and record.status == "selected":
        selected = np.asarray(record.row["selected_frame_ids"], dtype=np.int64)
        missing = selected[~np.isin(selected, primitive)]
        if len(missing):
            raise ValueError(f"selected frames outside primitive/raw MANO range: {missing[:5].tolist()}")
        return selected
    if mode != "primitive" and record.status != "rejected_static":
        raise ValueError(f"unsupported frame mode: {mode}")
    return primitive


def _load_trajectory(
    record: SegmentRecord,
    mode: str,
    *,
    annotation_root: Path,
    object_catalog: Stage3ObjectCatalog,
    mano: ManoReconstructor,
    frame_limit: Optional[int] = None,
    frame_ids_override: Optional[Sequence[int]] = None,
) -> SegmentTrajectory:
    annotation = _load_annotation(annotation_root, str(record.row["sequence"]))
    if frame_ids_override is None:
        frame_ids = _frame_ids_for_mode(record, annotation["raw_mano"].keys(), mode)
    else:
        frame_ids = np.asarray([int(value) for value in frame_ids_override], dtype=np.int64)
        primitive = _frame_ids_for_mode(record, annotation["raw_mano"].keys(), "primitive")
        if not len(frame_ids) or np.any(np.diff(frame_ids) <= 0):
            raise ValueError("frame_ids_override must be non-empty and strictly increasing")
        missing = frame_ids[~np.isin(frame_ids, primitive)]
        if len(missing):
            raise ValueError(f"frame_ids_override outside primitive/raw MANO range: {missing[:5].tolist()}")
    if frame_limit is not None and len(frame_ids) > frame_limit:
        picks = np.linspace(0, len(frame_ids) - 1, frame_limit, dtype=np.int64)
        frame_ids = frame_ids[picks]
    hands = {
        side: mano.reconstruct(annotation["raw_mano"], frame_ids, side)
        for side in ("left", "right")
    }
    object_points, missing = {}, []
    for object_id in annotation.get("obj_list", []):
        points = object_catalog.points(str(record.row["sequence"]), str(object_id))
        if points is None:
            missing.append(str(object_id))
        else:
            object_points[str(object_id)] = points
    for object_id in record.row["selected_object_ids"]:
        if object_id not in object_points:
            points = object_catalog.points(str(record.row["sequence"]), str(object_id))
            if points is None:
                missing.append(str(object_id))
            else:
                object_points[str(object_id)] = points
    selected_set = frozenset(int(value) for value in record.row.get("selected_frame_ids", []))
    trajectory = SegmentTrajectory(
        record=record,
        frame_ids=frame_ids,
        selected_frame_ids=selected_set,
        hands=hands,
        hand_faces=mano.faces,
        object_points=object_points,
        object_poses=annotation["obj_transf"],
        missing_object_points=tuple(sorted(set(missing))),
        mode=mode if record.status == "selected" else "primitive",
    )
    if not len(trajectory.object_world(0, selected=True, points_per_part=4096)):
        raise ValueError(f"selected object geometry/pose unavailable for {record.id}")
    return trajectory


def _category_groups(records: Sequence[SegmentRecord]) -> Dict[str, List[SegmentRecord]]:
    return {key: _records_for_category(records, key) for key in CATEGORY_NAMES}


def _choose_record(records: Sequence[SegmentRecord], segment_id: Optional[str], index: int) -> SegmentRecord:
    if segment_id:
        matches = [record for record in records if record.id == segment_id]
        if len(matches) != 1:
            raise ValueError(f"segment id {segment_id!r} is absent or ambiguous")
        return matches[0]
    return records[int(index) % len(records)]


def run_check(args: argparse.Namespace) -> None:
    index, all_records = _load_catalog(args.index.resolve(), args.previous_index.resolve())
    groups = _category_groups(all_records)
    record = _choose_record(groups[args.category], args.segment, args.segment_index)
    object_catalog = Stage3ObjectCatalog(Path(index["stage3_root"]))
    mano = ManoReconstructor(args.mano_root, args.threads)
    trajectory = _load_trajectory(
        record,
        args.frame_mode,
        annotation_root=Path(index["annotation_root"]),
        object_catalog=object_catalog,
        mano=mano,
        frame_limit=3,
    )
    selected_points = trajectory.object_world(0, selected=True, points_per_part=4096)
    context_points = trajectory.object_world(0, selected=False, points_per_part=args.context_points_per_part)
    result = {
        "index": str(args.index.resolve()),
        "schema": f"{index['schema_name']}@{index['schema_version']}",
        "counts": {key: len(value) for key, value in groups.items()},
        "segment_id": record.id,
        "segment_label": record.label,
        "status": record.status,
        "sequence": record.row["sequence"],
        "primitive": record.row.get("primitive"),
        "selected_object": record.row.get("selected_object_name"),
        "selected_object_ids": record.row["selected_object_ids"],
        "selected_hand_sides": record.row.get("selected_hand_sides", []),
        "frame_mode": trajectory.mode,
        "checked_source_frame_ids": trajectory.frame_ids.tolist(),
        "selected_object_points": int(len(selected_points)),
        "context_object_points": int(len(context_points)),
        "left_mano_shape": list(trajectory.hands["left"].shape),
        "right_mano_shape": list(trajectory.hands["right"].shape),
        "missing_context_object_ids": list(trajectory.missing_object_points),
        "coordinate_frame": "OakInk2 native world",
        "finite": bool(
            np.isfinite(selected_points).all()
            and np.isfinite(context_points).all()
            and all(np.isfinite(value).all() for value in trajectory.hands.values())
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _prepare_run(args: argparse.Namespace, index: dict, groups: Mapping[str, Sequence[SegmentRecord]]) -> Tuple[Path, dict]:
    if args.output is None:
        raise ValueError("--output is required when starting the interactive viewer")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    commit, dirty = _git_state()
    config = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    _write_json(output / "config.json", config)
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "ObjectInteractionCm",
        "operation": "oakink2_active_tool_segment_visualization",
        "run_id": output.name,
        "run_status": "RUNNING",
        "started_at": _now(),
        "work_version": WORK_VERSION,
        "base_commit": commit,
        "worktree_dirty": dirty,
        "command": shlex.join([sys.executable, *sys.argv]),
        "config": "config.json",
        "metadata_snapshot": None,
        "seed": None,
        "checkpoint": None,
        "inputs": {
            "selection_index": _snapshot(args.index),
            "previous_selection_index": _snapshot(args.previous_index),
            "annotation_root": str(Path(index["annotation_root"]).resolve()),
            "stage3_root": str(Path(index["stage3_root"]).resolve()),
            "object_root": str(Path(index["object_root"]).resolve()),
            "mano_root": str(args.mano_root.resolve()),
        },
        "counts": {key: len(value) for key, value in groups.items()},
        "viewer": f"http://{args.host}:{args.port}",
        "output": str(output),
        "log": "viewer.log",
        "source": _snapshot(Path(__file__)),
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(output / "run_manifest.json", manifest)
    (output / "viewer.log").write_text("", encoding="utf-8")
    return output, manifest


def run_server(args: argparse.Namespace) -> None:
    try:
        import viser
    except ImportError as exc:
        raise RuntimeError("viser is required to start the interactive viewer") from exc

    index, all_records = _load_catalog(args.index.resolve(), args.previous_index.resolve())
    groups = _category_groups(all_records)
    category = args.category
    records = groups[category]
    record = _choose_record(records, args.segment, args.segment_index)
    output, manifest = _prepare_run(args, index, groups)

    def emit(message: str) -> None:
        line = f"{_now()} {message}"
        with (output / "viewer.log").open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    object_catalog = Stage3ObjectCatalog(Path(index["stage3_root"]))
    mano = ManoReconstructor(args.mano_root, args.threads)
    state: Dict[str, Any] = {
        "category": category,
        "records": records,
        "record": record,
        "trajectory": _load_trajectory(
            record,
            args.frame_mode,
            annotation_root=Path(index["annotation_root"]),
            object_catalog=object_catalog,
            mano=mano,
        ),
        "playing": False,
        "suppress": False,
    }
    lock = threading.RLock()
    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")

    category_labels = {
        f"{CATEGORY_NAMES[key]}（{len(value)}）": key for key, value in groups.items()
    }
    initial_category_label = next(label for label, key in category_labels.items() if key == category)
    with server.gui.add_folder("切分与轨迹"):
        category_gui = server.gui.add_dropdown(
            "筛选", options=tuple(category_labels), initial_value=initial_category_label
        )
        segment_gui = server.gui.add_dropdown(
            "轨迹", options=tuple(item.label for item in records), initial_value=record.label
        )
        previous_segment_gui = server.gui.add_button("上一条轨迹")
        next_segment_gui = server.gui.add_button("下一条轨迹")
        frame_mode_gui = server.gui.add_dropdown(
            "帧范围",
            options=tuple(FRAME_MODE_NAMES.values()),
            initial_value=FRAME_MODE_NAMES[state["trajectory"].mode],
        )

    with server.gui.add_folder("播放"):
        frame_gui = server.gui.add_slider(
            "帧", min=0, max=state["trajectory"].frame_count - 1,
            step=1, initial_value=int(args.frame) % state["trajectory"].frame_count,
        )
        previous_frame_gui = server.gui.add_button("上一帧")
        next_frame_gui = server.gui.add_button("下一帧")
        play_gui = server.gui.add_button("播放 / 暂停")
        fps_gui = server.gui.add_slider(
            "播放 FPS", min=1.0, max=30.0, step=1.0, initial_value=float(args.fps)
        )

    with server.gui.add_folder("显示"):
        context_gui = server.gui.add_checkbox("显示场景其他物体", initial_value=True)
        hand_mesh_gui = server.gui.add_checkbox("显示双手 mesh", initial_value=True)
        hand_points_gui = server.gui.add_checkbox("显示双手顶点", initial_value=False)
        selected_size_gui = server.gui.add_slider(
            "选中物体点大小", min=0.0005, max=0.02, step=0.0005, initial_value=0.004
        )
        context_size_gui = server.gui.add_slider(
            "其他物体点大小", min=0.0005, max=0.02, step=0.0005, initial_value=0.002
        )
        hand_size_gui = server.gui.add_slider(
            "手顶点大小", min=0.0005, max=0.02, step=0.0005, initial_value=0.003
        )
        hand_opacity_gui = server.gui.add_slider(
            "手 mesh 透明度", min=0.05, max=1.0, step=0.05, initial_value=0.65
        )

    status_gui = server.gui.add_markdown("")
    server.gui.add_markdown(
        "橙色为本条切分选中的对象，灰色为同场景其他对象；左手蓝色、右手绿色。"
        "在 `primitive 全帧` 模式中，状态栏会明确标记当前原始帧是否进入最终 v1.1 index。"
        "静止对照没有保留帧，只显示原 primitive 时间窗，且不会写回数据。"
    )

    trajectory: SegmentTrajectory = state["trajectory"]
    initial_selected = trajectory.object_world(0, selected=True, points_per_part=4096)
    initial_context = trajectory.object_world(
        0, selected=False, points_per_part=args.context_points_per_part
    )
    selected_handle = server.scene.add_point_cloud(
        "/objects/selected", points=initial_selected,
        colors=np.broadcast_to(SELECTED_OBJECT_COLOR, (len(initial_selected), 3)),
        point_size=float(selected_size_gui.value), point_shape="circle",
        precision="float32", point_shading="flat",
    )
    context_seed = initial_context if len(initial_context) else np.zeros((1, 3), dtype=np.float32)
    context_handle = server.scene.add_point_cloud(
        "/objects/context", points=context_seed,
        colors=np.broadcast_to(CONTEXT_OBJECT_COLOR, (len(context_seed), 3)),
        point_size=float(context_size_gui.value), point_shape="circle",
        precision="float32", point_shading="flat",
    )
    hand_point_handles = {}
    hand_mesh_handles = {}
    for side, color in (("left", LEFT_HAND_COLOR), ("right", RIGHT_HAND_COLOR)):
        vertices = trajectory.hands[side][0]
        hand_point_handles[side] = server.scene.add_point_cloud(
            f"/hands/{side}/points", points=vertices,
            colors=np.broadcast_to(color, (len(vertices), 3)),
            point_size=float(hand_size_gui.value), point_shape="circle",
            precision="float32", point_shading="flat",
        )
        hand_mesh_handles[side] = server.scene.add_mesh_simple(
            f"/hands/{side}/mesh", vertices=vertices, faces=trajectory.hand_faces[side],
            color=tuple(int(value) for value in color), opacity=float(hand_opacity_gui.value),
        )

    def render() -> None:
        with lock:
            trajectory = state["trajectory"]
            frame_index = int(frame_gui.value) % trajectory.frame_count
            frame_id = int(trajectory.frame_ids[frame_index])
            selected_points = trajectory.object_world(frame_index, selected=True, points_per_part=4096)
            context_points = trajectory.object_world(
                frame_index, selected=False, points_per_part=args.context_points_per_part
            )
            selected_handle.points = selected_points
            selected_handle.colors = np.broadcast_to(SELECTED_OBJECT_COLOR, (len(selected_points), 3))
            selected_handle.point_size = float(selected_size_gui.value)
            if len(context_points):
                context_handle.points = context_points
                context_handle.colors = np.broadcast_to(CONTEXT_OBJECT_COLOR, (len(context_points), 3))
            context_handle.point_size = float(context_size_gui.value)
            context_handle.visible = bool(context_gui.value) and bool(len(context_points))

            active_sides = set(trajectory.record.row.get("selected_hand_sides", []))
            for side, base_color in (("left", LEFT_HAND_COLOR), ("right", RIGHT_HAND_COLOR)):
                vertices = trajectory.hands[side][frame_index]
                color = base_color if not active_sides or side in active_sides else NON_SELECTED_HAND_COLOR
                points_handle = hand_point_handles[side]
                points_handle.points = vertices
                points_handle.colors = np.broadcast_to(color, (len(vertices), 3))
                points_handle.point_size = float(hand_size_gui.value)
                points_handle.visible = bool(hand_points_gui.value)
                mesh_handle = hand_mesh_handles[side]
                mesh_handle.vertices = vertices
                mesh_handle.opacity = float(hand_opacity_gui.value)
                mesh_handle.visible = bool(hand_mesh_gui.value)

            row = trajectory.record.row
            kept = frame_id in trajectory.selected_frame_ids
            kept_text = "是" if kept else "否"
            status = "静止阈值排除" if trajectory.record.status == "rejected_static" else "有效切分"
            status_gui.content = (
                f"**segment** `{trajectory.record.id}`  |  **状态** `{status}`  |  "
                f"**类别** `{'新增' if trajectory.record.new_in_v1_1 else '既有'}`  \n"
                f"**sequence** `{row['sequence']}`  \n"
                f"**primitive** `{row.get('primitive')}`  |  **interaction_mode** `{row.get('interaction_mode')}`  |  "
                f"**选择原因** `{row.get('selection_reason')}`  \n"
                f"**选中对象** `{row.get('selected_object_name')}`  |  **对象 ID** `{', '.join(row['selected_object_ids'])}`  |  "
                f"**主操作手** `{', '.join(row.get('selected_hand_sides', [])) or '未限定'}`  \n"
                f"**显示帧** `{frame_index + 1}/{trajectory.frame_count}`  |  **原始 frame ID** `{frame_id}`  |  "
                f"**进入最终 index** `{kept_text}`  |  **帧模式** `{FRAME_MODE_NAMES[trajectory.mode]}`  \n"
                f"**最终保留帧数** `{row.get('selected_frame_count', 0)}`  |  "
                f"**primitive 定义** `{row['frame_range_def']}`  |  **motion span** `{row.get('motion_frame_range')}`  \n"
                f"**坐标系** `OakInk2 native world`  |  **缺失上下文点云 ID** `{list(trajectory.missing_object_points)}`"
            )

    def load_record(record_to_load: SegmentRecord, mode: str, *, reset_frame: bool = True) -> None:
        with lock:
            effective_mode = "primitive" if record_to_load.status == "rejected_static" else mode
            emit(f"loading segment={record_to_load.id} mode={effective_mode}")
            loaded = _load_trajectory(
                record_to_load,
                effective_mode,
                annotation_root=Path(index["annotation_root"]),
                object_catalog=object_catalog,
                mano=mano,
            )
            state["record"] = record_to_load
            state["trajectory"] = loaded
            state["suppress"] = True
            try:
                segment_gui.value = record_to_load.label
                frame_mode_gui.value = FRAME_MODE_NAMES[loaded.mode]
                frame_gui.max = loaded.frame_count - 1
                frame_gui.value = 0 if reset_frame else min(int(frame_gui.value), loaded.frame_count - 1)
            finally:
                state["suppress"] = False
            emit(f"loaded segment={record_to_load.id} frames={loaded.frame_count}")
            render()

    @category_gui.on_update
    def _(_: Any) -> None:
        if state["suppress"]:
            return
        key = category_labels[str(category_gui.value)]
        selected_records = groups[key]
        state["category"] = key
        state["records"] = selected_records
        state["suppress"] = True
        try:
            segment_gui.options = tuple(item.label for item in selected_records)
        finally:
            state["suppress"] = False
        load_record(selected_records[0], "selected")

    @segment_gui.on_update
    def _(_: Any) -> None:
        if state["suppress"]:
            return
        match = next(item for item in state["records"] if item.label == segment_gui.value)
        requested_mode = next(key for key, label in FRAME_MODE_NAMES.items() if label == frame_mode_gui.value)
        load_record(match, requested_mode)

    def shift_segment(delta: int) -> None:
        records_now = state["records"]
        current = records_now.index(state["record"])
        requested_mode = next(key for key, label in FRAME_MODE_NAMES.items() if label == frame_mode_gui.value)
        load_record(records_now[(current + delta) % len(records_now)], requested_mode)

    @previous_segment_gui.on_click
    def _(_: Any) -> None:
        shift_segment(-1)

    @next_segment_gui.on_click
    def _(_: Any) -> None:
        shift_segment(1)

    @frame_mode_gui.on_update
    def _(_: Any) -> None:
        if state["suppress"]:
            return
        requested_mode = next(key for key, label in FRAME_MODE_NAMES.items() if label == frame_mode_gui.value)
        load_record(state["record"], requested_mode)

    @frame_gui.on_update
    def _(_: Any) -> None:
        if not state["suppress"]:
            render()

    @previous_frame_gui.on_click
    def _(_: Any) -> None:
        frame_gui.value = (int(frame_gui.value) - 1) % state["trajectory"].frame_count

    @next_frame_gui.on_click
    def _(_: Any) -> None:
        frame_gui.value = (int(frame_gui.value) + 1) % state["trajectory"].frame_count

    @play_gui.on_click
    def _(_: Any) -> None:
        state["playing"] = not bool(state["playing"])

    for control in (
        context_gui, hand_mesh_gui, hand_points_gui, selected_size_gui,
        context_size_gui, hand_size_gui, hand_opacity_gui,
    ):
        control.on_update(lambda _: render())

    render()
    emit(f"viewer=http://{args.host}:{args.port} initial_segment={record.id}")
    print(json.dumps({
        "viewer": f"http://{args.host}:{args.port}",
        "run_id": output.name,
        "run_status": "RUNNING",
        "counts": {key: len(value) for key, value in groups.items()},
        "initial_segment": record.id,
    }, ensure_ascii=False), flush=True)

    def stop_handler(_: int, __: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    last_tick = time.monotonic()
    try:
        while True:
            time.sleep(0.01)
            if not state["playing"]:
                last_tick = time.monotonic()
                continue
            interval = 1.0 / max(float(fps_gui.value), 1.0)
            current = time.monotonic()
            if current - last_tick < interval:
                continue
            last_tick = current
            state["suppress"] = True
            try:
                frame_gui.value = (int(frame_gui.value) + 1) % state["trajectory"].frame_count
            finally:
                state["suppress"] = False
            render()
    except KeyboardInterrupt:
        manifest["run_status"] = "STOPPED"
        manifest["completed_at"] = _now()
        manifest["conclusion"] = "INCONCLUSIVE"
        _write_json(output / "run_manifest.json", manifest)
        emit("viewer stopped")
    except Exception:
        (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        manifest["run_status"] = "FAILED"
        manifest["completed_at"] = _now()
        manifest["error"] = "error.txt"
        manifest["conclusion"] = "INVALID_IMPLEMENTATION"
        _write_json(output / "run_manifest.json", manifest)
        emit("viewer failed; see error.txt")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1_1/index.json"),
    )
    parser.add_argument(
        "--previous-index", type=Path,
        default=Path("data/processed_data/oicm_v1_4_raw/oakink2_active_tool_segments_v1/index.json"),
    )
    parser.add_argument(
        "--mano-root", type=Path,
        default=Path("dataset/arctic/data/body_models/mano"),
    )
    parser.add_argument("--category", choices=tuple(CATEGORY_NAMES), default="new")
    parser.add_argument("--segment", help="Exact generated segment id, for example selected:0000")
    parser.add_argument("--segment-index", type=int, default=0)
    parser.add_argument("--frame-mode", choices=tuple(FRAME_MODE_NAMES), default="selected")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--context-points-per-part", type=int, default=512)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8142)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.context_points_per_part < 1 or args.threads < 1 or args.fps <= 0:
        raise ValueError("context-points-per-part, threads and fps must be positive")
    for path in (args.index, args.previous_index, args.mano_root):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.check_only:
        run_check(args)
    else:
        run_server(args)


if __name__ == "__main__":
    main()
