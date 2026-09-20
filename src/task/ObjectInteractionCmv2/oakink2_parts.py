"""Part-local OakInk2 transition view for the approved V1.11h mixed run."""
from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .articulated import _rotation_log
from .multi_domain import ThreeDomainTransitions, _is_se3, _normal_world_to_frame, _stable_seed, _world_to_frame


PART_ADAPTER_SCHEMA = "ref2dex_cmv2_oakink2_part_adapter_v1"


@lru_cache(maxsize=128)
def _part_arrays(part_root: str) -> dict[str, np.ndarray]:
    root = Path(part_root)
    return {
        "object_points_local": np.load(root / "object_points_local.npy", mmap_mode="r"),
        "object_normals_local": np.load(root / "object_normals_local.npy", mmap_mode="r"),
        "obj_pose_world": np.load(root / "obj_pose_world.npy", mmap_mode="r"),
        "source_frame_id": np.load(root / "source_frame_id.npy", mmap_mode="r"),
    }


def _frame_id_digest(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(canonical.tobytes()).hexdigest()


@lru_cache(maxsize=8192)
def _part_summary(part_root: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], str, bool]:
    root = Path(part_root)
    arrays = {
        "object_points_local": np.load(root / "object_points_local.npy", mmap_mode="r"),
        "object_normals_local": np.load(root / "object_normals_local.npy", mmap_mode="r"),
        "obj_pose_world": np.load(root / "obj_pose_world.npy", mmap_mode="r"),
        "source_frame_id": np.load(root / "source_frame_id.npy", mmap_mode="r"),
    }
    stride = max(1, len(arrays["obj_pose_world"]) // 16)
    return (
        tuple(arrays["object_points_local"].shape),
        tuple(arrays["object_normals_local"].shape),
        tuple(arrays["obj_pose_world"].shape),
        _frame_id_digest(arrays["source_frame_id"]),
        all(_is_se3(pose) for pose in arrays["obj_pose_world"][::stride]),
    )


class OakInkPartTransitions:
    """Draw a rigid component only after a base OakInk2 transition is selected."""

    def __init__(self, base: ThreeDomainTransitions, adapter_root: str | Path) -> None:
        if any(sequence.domain != "oakink2" for sequence in base.sequences):
            raise ValueError("OakInk part adapter only accepts OakInk2 base sequences")
        self.base = base
        self.rows = base.rows
        self.dropped_timeline_transitions = base.dropped_timeline_transitions
        self.adapter_root = Path(adapter_root).resolve()
        manifest_path = self.adapter_root / "cache_manifest.json"
        index_path = self.adapter_root / "index.json"
        if not manifest_path.is_file() or not index_path.is_file():
            raise FileNotFoundError(f"missing OakInk2 part adapter under {self.adapter_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if (manifest.get("schema_name") != PART_ADAPTER_SCHEMA or index.get("schema_name") != PART_ADAPTER_SCHEMA
                or int(manifest.get("validation", {}).get("bad_count", 1) or 0) != 0):
            raise ValueError("invalid OakInk2 part adapter manifest")
        entries = {str(entry["id"]): entry for entry in index.get("sequences", [])}
        self.parts_by_sequence: list[list[dict[str, Any]]] = []
        for sequence, entry in zip(base.sequences, base.sequence_entries):
            sequence_id = str(entry["id"])
            adapter_entry = entries.get(sequence_id)
            if adapter_entry is None:
                raise ValueError(f"part adapter lacks {sequence_id}")
            parts = list(adapter_entry.get("parts", []))
            if not parts:
                raise ValueError(f"part adapter has no parts for {sequence_id}")
            expected_ids = np.asarray(sequence.arrays["source_frame_id"], dtype=np.int64)
            expected_digest = _frame_id_digest(expected_ids)
            loaded = []
            for part in parts:
                part_root = (self.adapter_root / part["path"]).resolve()
                point_shape, normal_shape, pose_shape, frame_digest, pose_valid = _part_summary(str(part_root))
                if point_shape != (4096, 3) or normal_shape != (4096, 3):
                    raise ValueError(f"{part_root}: invalid 4096-point canonical pool")
                if pose_shape != (sequence.frame_count, 4, 4):
                    raise ValueError(f"{part_root}: pose/frame count mismatch")
                if frame_digest != expected_digest:
                    raise ValueError(f"{part_root}: source frame IDs differ from hand stream")
                if not pose_valid:
                    raise ValueError(f"{part_root}: invalid component SE(3)")
                loaded.append({**part, "root": str(part_root)})
            self.parts_by_sequence.append(loaded)

    def __len__(self) -> int:
        return len(self.rows)

    def component_count(self, transition_index: int) -> int:
        sequence_index, _ = self.rows[transition_index]
        return len(self.parts_by_sequence[sequence_index])

    def sample_component(self, transition_index: int, component_index: int) -> dict[str, Any]:
        sequence_index, current = self.rows[transition_index]
        sequence = self.base.sequences[sequence_index]
        entry = self.base.sequence_entries[sequence_index]
        parts = self.parts_by_sequence[sequence_index]
        if not 0 <= component_index < len(parts):
            raise IndexError(f"component {component_index} outside {len(parts)} for {entry['id']}")
        source_ids = np.asarray(sequence.arrays["source_frame_id"], dtype=np.int64)
        raw_current = int(source_ids[current])
        stride = self.base._selected_stride(sequence, current)
        future = current + stride
        if future >= sequence.frame_count or not self.base._timeline_is_continuous(sequence, current, stride):
            raise ValueError(f"invalid OakInk2 component transition: {entry['id']}")
        part = parts[component_index]
        arrays = _part_arrays(part["root"])
        pose = np.asarray(arrays["obj_pose_world"][current], dtype=np.float32)
        next_pose = np.asarray(arrays["obj_pose_world"][future], dtype=np.float32)
        if not _is_se3(pose) or not _is_se3(next_pose):
            raise ValueError(f"invalid component pose: {entry['id']}/{part['object_id']}")
        rotation, translation = pose[:3, :3], pose[:3, 3]
        next_rotation, next_translation = next_pose[:3, :3], next_pose[:3, 3]
        delta_rotation = rotation.T @ next_rotation
        delta_translation = (next_translation - translation) @ rotation
        seed = _stable_seed(self.base.base_seed, sequence.path, raw_current, self.base.split, part["object_id"])
        selected = np.random.default_rng(seed ^ 0xC15).choice(4096, size=self.base.num_obj_points, replace=False)
        object_local = np.asarray(arrays["object_points_local"][selected], dtype=np.float32)
        normals_local = np.asarray(arrays["object_normals_local"][selected], dtype=np.float32)
        future_world = object_local @ next_rotation.T + next_translation
        hand_world = np.asarray(sequence.arrays["knn_hand_points_world"][current], dtype=np.float32)
        hand_future_world = np.asarray(sequence.arrays["knn_hand_points_world"][future], dtype=np.float32)
        hand_normals_world = np.asarray(sequence.arrays["knn_hand_normals_world"][current], dtype=np.float32)
        hand_points = _world_to_frame(hand_world, pose)
        hand_future = _world_to_frame(hand_future_world, pose)
        values = {
            "obj_points": object_local,
            "obj_normals": normals_local,
            "hand_points": hand_points,
            "hand_normals": _normal_world_to_frame(hand_normals_world, pose),
            "hand_flow": hand_future - hand_points,
            "obj_flow_gt": _world_to_frame(future_world, pose) - object_local,
        }
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError(f"non-finite component transition: {entry['id']}/{part['object_id']}")
        frame_time = np.asarray(sequence.arrays["frame_time"], dtype=np.float64)
        return {
            **{key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32)) for key, value in values.items()},
            "obj_link_id": torch.zeros(self.base.num_obj_points, dtype=torch.long),
            "link_valid_mask": torch.ones(1, dtype=torch.bool),
            "joint_parent": torch.empty(0, dtype=torch.long),
            "joint_child": torch.empty(0, dtype=torch.long),
            "joint_axis_root": torch.empty(0, 3),
            "joint_origin_root": torch.empty(0, 3),
            "joint_q_t": torch.empty(0),
            "joint_valid_mask": torch.empty(0, dtype=torch.bool),
            "delta_q_gt": torch.empty(0),
            "delta_xi_root_gt": torch.from_numpy(np.concatenate((delta_translation, _rotation_log(delta_rotation))).astype(np.float32)),
            "delta_time_s": torch.tensor(float(frame_time[future] - frame_time[current]), dtype=torch.float32),
            "stride": torch.tensor(stride, dtype=torch.int64),
            "hand_valid_mask": torch.ones(sequence.hand_points, dtype=torch.bool),
            "source": "oakink2",
            "sequence_id": f"{entry['id']}#part={part['object_id']}",
        }

    def __getitem__(self, transition_index: int) -> dict[str, Any]:
        sequence_index, current = self.rows[transition_index]
        sequence = self.base.sequences[sequence_index]
        raw_current = int(np.asarray(sequence.arrays["source_frame_id"])[current])
        component = _stable_seed(self.base.base_seed, sequence.path, raw_current, self.base.split, "component") % self.component_count(transition_index)
        return self.sample_component(transition_index, int(component))
