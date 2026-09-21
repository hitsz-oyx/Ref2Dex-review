"""V1.12 whole-object transition adapters with direct per-part pose targets."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .articulated import ArticulatedSequenceView
from .multi_domain import (
    _is_se3,
    _normal_world_to_frame,
    _stable_seed,
    _world_to_frame,
    fixed_bilateral_hand_indices,
)
from .oakink2_parts import OakInkPartTransitions, _part_arrays


def stratified_part_indices(part_ids: np.ndarray, count: int, *, minimum_per_part: int = 64,
                            seed: int = 42) -> np.ndarray:
    """Deterministic proportional sampling with a per-part floor and no replacement."""
    ids = np.asarray(part_ids, dtype=np.int64)
    if ids.ndim != 1 or count <= 0 or count > len(ids) or minimum_per_part <= 0:
        raise ValueError("invalid stratified part sampling contract")
    parts, counts = np.unique(ids, return_counts=True)
    if not np.array_equal(parts, np.arange(len(parts))):
        raise ValueError("part IDs must be contiguous from zero")
    base = np.minimum(counts, minimum_per_part)
    if int(base.sum()) > count:
        raise ValueError("sample count cannot satisfy the per-part minimum")
    quota = base.astype(np.int64)
    remaining = int(count - quota.sum())
    capacity = counts - quota
    if remaining:
        raw = remaining * counts.astype(np.float64) / counts.sum()
        extra = np.minimum(np.floor(raw).astype(np.int64), capacity)
        quota += extra
        capacity -= extra
        remaining -= int(extra.sum())
        fractional = raw - np.floor(raw)
        order = np.lexsort((parts, -fractional))
        while remaining:
            progressed = False
            for part in order:
                if capacity[part] <= 0:
                    continue
                quota[part] += 1
                capacity[part] -= 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
            if not progressed:
                raise RuntimeError("stratified quota allocation exhausted capacity")
    rng = np.random.default_rng(int(seed))
    selected = [rng.choice(np.flatnonzero(ids == part), size=int(quota[part]), replace=False)
                for part in parts]
    return rng.permutation(np.concatenate(selected)).astype(np.int64, copy=False)


def _rigid_inverse(pose: np.ndarray) -> np.ndarray:
    rotation, translation = pose[:3, :3], pose[:3, 3]
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ translation
    return result


def delta_in_reference(current_pose: np.ndarray, future_pose: np.ndarray,
                       reference_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map current component-frame points to future points in a fixed current reference."""
    for value in (current_pose, future_pose, reference_pose):
        if not _is_se3(value):
            raise ValueError("direct part GT requires valid SE(3) poses")
    delta = _rigid_inverse(reference_pose) @ future_pose @ _rigid_inverse(current_pose) @ reference_pose
    return delta[:3, :3].astype(np.float32), delta[:3, 3].astype(np.float32)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    return rotation.astype(np.float32)


def _transition_values(sequence, current: int, future: int, reference: np.ndarray,
                       selected: np.ndarray,
                       hand_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    object_world = np.asarray(sequence.arrays["obj_points_pool_world"][current, selected], dtype=np.float32)
    future_world = np.asarray(sequence.arrays["obj_points_pool_world"][future, selected], dtype=np.float32)
    hand_world = np.asarray(sequence.arrays["knn_hand_points_world"][current], dtype=np.float32)
    hand_future_world = np.asarray(sequence.arrays["knn_hand_points_world"][future], dtype=np.float32)
    hand_normals_world = np.asarray(
        sequence.arrays["knn_hand_normals_world"][current], dtype=np.float32)
    if hand_indices is not None:
        hand_world = hand_world[hand_indices]
        hand_future_world = hand_future_world[hand_indices]
        hand_normals_world = hand_normals_world[hand_indices]
    values = {
        "obj_points": _world_to_frame(object_world, reference),
        "obj_normals": _normal_world_to_frame(
            np.asarray(sequence.arrays["obj_normals_pool_world"][current, selected]), reference),
        "hand_points": _world_to_frame(hand_world, reference),
        "hand_normals": _normal_world_to_frame(hand_normals_world, reference),
        "obj_future": _world_to_frame(future_world, reference),
    }
    values["hand_flow"] = _world_to_frame(hand_future_world, reference) - values["hand_points"]
    values["obj_flow_gt"] = values["obj_future"] - values["obj_points"]
    return values


def _sample_payload(values: Mapping[str, np.ndarray], part_ids: np.ndarray,
                    rotations: np.ndarray, translations: np.ndarray, object_point_ids: np.ndarray, *, delta_time: float,
                    stride: int, hand_points: int, source: str, sequence_id: str,
                    source_frame_id: int, next_source_frame_id: int) -> dict[str, Any]:
    reconstructed = np.empty_like(values["obj_points"])
    for part in range(len(rotations)):
        mask = part_ids == part
        reconstructed[mask] = values["obj_points"][mask] @ rotations[part].T + translations[part]
    replay = float(np.max(np.linalg.norm(reconstructed - values["obj_future"], axis=-1)))
    if not np.isfinite(replay) or replay > 2e-4:
        raise ValueError(f"{sequence_id}: direct part replay residual {replay} exceeds 0.2 mm")
    tensors = {key: value for key, value in values.items() if key != "obj_future"}
    return {
        **{key: torch.from_numpy(np.ascontiguousarray(value, dtype=np.float32))
           for key, value in tensors.items()},
        "obj_point_id": torch.from_numpy(np.ascontiguousarray(object_point_ids, dtype=np.int64)),
        "obj_part_id": torch.from_numpy(np.ascontiguousarray(part_ids, dtype=np.int64)),
        "part_valid_mask": torch.ones((len(rotations),), dtype=torch.bool),
        "delta_translation_part_gt": torch.from_numpy(np.ascontiguousarray(translations, dtype=np.float32)),
        "delta_rotation_part_gt": torch.from_numpy(np.ascontiguousarray(rotations, dtype=np.float32)),
        "delta_time_s": torch.tensor(delta_time, dtype=torch.float32),
        "stride": torch.tensor(stride, dtype=torch.int64),
        "hand_valid_mask": torch.ones((hand_points,), dtype=torch.bool),
        "pose_flow_residual_max_m": torch.tensor(replay, dtype=torch.float32),
        "source_frame_id": torch.tensor(source_frame_id, dtype=torch.int64),
        "next_source_frame_id": torch.tensor(next_source_frame_id, dtype=torch.int64),
        "source": source,
        "sequence_id": sequence_id,
    }


class RigidArticulatedPartTransitions(Dataset):
    """Whole-object GRAB/ARCTIC transitions with one direct SE(3) per link."""

    def __init__(self, sequence_specs: Sequence[Mapping[str, Any]], split: str, *,
                 num_obj_points: int = 1024, stride_values: Sequence[int] = (1, 2, 3),
                 base_seed: int = 42, hand_points_per_side: int | None = None,
                 hand_sampling_seed: int = 42) -> None:
        if num_obj_points != 1024 or not stride_values or any(int(value) <= 0 for value in stride_values):
            raise ValueError("V1.12 fixes 1024 object points and positive strides")
        self.split, self.num_obj_points = split, int(num_obj_points)
        self.stride_values, self.base_seed = tuple(int(value) for value in stride_values), int(base_seed)
        self.hand_points_per_side = (
            None if hand_points_per_side is None else int(hand_points_per_side))
        self.hand_sampling_seed = int(hand_sampling_seed)
        self.sequences, self.entries = [], []
        for item in sequence_specs:
            domain = str(item.get("name", item.get("domain", "")))
            view = ArticulatedSequenceView(item["path"], domain, split, str(item["hand_variant"]), item["articulation"])
            self.sequences.append(view)
            self.entries.append(item)
        self.rows = [(sequence_index, frame) for sequence_index, view in enumerate(self.sequences)
                     for frame in range(view.frame_count - max(self.stride_values))]
        if not self.rows:
            raise ValueError("V1.12 has no GRAB/ARCTIC transitions")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, current = self.rows[index]
        view = self.sequences[sequence_index]
        source_ids = np.asarray(view.base.arrays["source_frame_id"], dtype=np.int64)
        seed = _stable_seed(self.base_seed, view.path, int(source_ids[current]), self.split)
        stride = self.stride_values[seed % len(self.stride_values)]
        future = current + stride
        frame_time = np.asarray(view.base.arrays["frame_time"], dtype=np.float64)
        if source_ids[future] <= source_ids[current] or frame_time[future] <= frame_time[current]:
            raise ValueError(f"{view.path}: invalid transition timeline")
        reference = np.asarray(view.arrays[view.root_pose_key][current], dtype=np.float32)
        root_future = np.asarray(view.arrays[view.root_pose_key][future], dtype=np.float32)
        part_pool = view.part_ids(np.arange(4096, dtype=np.int64))
        selected = stratified_part_indices(part_pool, 1024, seed=seed ^ 0xB15)
        part_ids = part_pool[selected]
        hand_indices = (None if self.hand_points_per_side is None else fixed_bilateral_hand_indices(
            view.base.hand_variant, self.hand_points_per_side, self.hand_sampling_seed))
        values = _transition_values(
            view.base, current, future, reference, selected, hand_indices)
        root_rotation, root_translation = delta_in_reference(reference, root_future, reference)
        links = view.kinematics["num_links"]
        rotations = np.tile(root_rotation[None], (links, 1, 1))
        translations = np.tile(root_translation[None], (links, 1))
        if view.domain == "arctic":
            joint = view.kinematics["joints"][0]
            delta_q = float(view.q(future)[0] - view.q(current)[0])
            joint_rotation = _axis_angle_rotation(joint["axis_root"], delta_q)
            origin = np.asarray(joint["origin_root"], dtype=np.float32)
            child = int(joint["child"])
            rotations[child] = root_rotation @ joint_rotation
            translations[child] = ((origin - origin @ joint_rotation.T) @ root_rotation.T
                                   + root_translation)
        return _sample_payload(
            values, part_ids, rotations, translations, _object_point_ids(str(view.path))[selected],
            delta_time=float(frame_time[future] - frame_time[current]), stride=stride,
            hand_points=values["hand_points"].shape[0], source=view.domain,
            sequence_id=str(self.entries[sequence_index].get("id", view.path)),
            source_frame_id=int(source_ids[current]), next_source_frame_id=int(source_ids[future]))


@lru_cache(maxsize=128)
def _object_point_ids(sequence_path: str) -> np.ndarray:
    path = Path(sequence_path) / "geometry" / "obj_point_id.npy"
    if not path.is_file():
        return np.arange(4096, dtype=np.int64)
    values = np.load(path, mmap_mode="r")
    if values.shape != (4096,):
        raise ValueError(f"{sequence_path}: obj_point_id must be [4096]")
    return values


class OakInkWholePartTransitions(Dataset):
    """Replace V1.11 component sampling with one whole multi-part OakInk2 sample."""

    def __init__(self, base, adapter_root: str | Path) -> None:
        self.adapter = OakInkPartTransitions(base, adapter_root)
        self.base, self.rows = base, base.rows
        self.dropped_timeline_transitions = base.dropped_timeline_transitions

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, current = self.rows[index]
        sequence = self.base.sequences[sequence_index]
        entry = self.base.sequence_entries[sequence_index]
        parts = self.adapter.parts_by_sequence[sequence_index]
        if [int(part.get("part_index", -1)) for part in parts] != list(range(len(parts))):
            raise ValueError(f"{entry['id']}: adapter part order must match point-ID groups")
        source_ids = np.asarray(sequence.arrays["source_frame_id"], dtype=np.int64)
        raw_current = int(source_ids[current])
        stride = self.base._selected_stride(sequence, current)
        future = current + stride
        if future >= sequence.frame_count or not self.base._timeline_is_continuous(sequence, current, stride):
            raise ValueError(f"invalid OakInk2 whole-object transition: {entry['id']}")
        reference = np.asarray(sequence.arrays["obj_pose_world"][current], dtype=np.float32)
        raw_ids = np.asarray(_object_point_ids(str(sequence.path)), dtype=np.int64)
        part_pool = raw_ids // 4096
        if set(part_pool.tolist()) != set(range(len(parts))):
            raise ValueError(f"{entry['id']}: object point IDs disagree with adapter parts")
        seed = _stable_seed(self.base.base_seed, sequence.path, raw_current, self.base.split)
        selected = stratified_part_indices(part_pool, 1024, seed=seed ^ 0xC15)
        part_ids = part_pool[selected]
        values = _transition_values(
            sequence, current, future, reference, selected,
            self.base.hand_indices[sequence_index])
        rotations, translations = [], []
        for part in parts:
            arrays = _part_arrays(part["root"])
            rotation, translation = delta_in_reference(
                np.asarray(arrays["obj_pose_world"][current], dtype=np.float32),
                np.asarray(arrays["obj_pose_world"][future], dtype=np.float32), reference)
            rotations.append(rotation)
            translations.append(translation)
        frame_time = np.asarray(sequence.arrays["frame_time"], dtype=np.float64)
        return _sample_payload(
            values, part_ids, np.stack(rotations), np.stack(translations), raw_ids[selected],
            delta_time=float(frame_time[future] - frame_time[current]), stride=stride,
            hand_points=values["hand_points"].shape[0], source="oakink2", sequence_id=str(entry["id"]),
            source_frame_id=raw_current, next_source_frame_id=int(source_ids[future]))
