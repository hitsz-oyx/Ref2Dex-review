"""Runtime construction of fixed 8-step InteractionDynamics chunks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed
from src.task.InteractionDynamics.uni3d import patchify


SHARED_SCHEMA_NAME = "ref2dex_interaction_dynamics_shared"
HAND_SCHEMA_NAME = "ref2dex_interaction_dynamics_hand"
CHUNK_LEN = 8
TEMPORAL_STRIDE = 1


def transform_points(points: np.ndarray, pose_frame_to_world: np.ndarray) -> np.ndarray:
    """Transform world points to the frame described by a frame-to-world pose."""
    rotation = np.asarray(pose_frame_to_world[:3, :3], np.float32)
    translation = np.asarray(pose_frame_to_world[:3, 3], np.float32)
    return ((np.asarray(points, np.float32) - translation) @ rotation).astype(np.float32)


def transform_normals(normals: np.ndarray, pose_frame_to_world: np.ndarray) -> np.ndarray:
    """Rotate world normals into a frame; translation is deliberately ignored."""
    rotation = np.asarray(pose_frame_to_world[:3, :3], np.float32)
    result = np.asarray(normals, np.float32) @ rotation
    return (result / np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def inverse_transform_points(points: np.ndarray, pose_frame_to_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose_frame_to_world[:3, :3], np.float32)
    translation = np.asarray(pose_frame_to_world[:3, 3], np.float32)
    return (np.asarray(points, np.float32) @ rotation.T + translation).astype(np.float32)


def _manifest_samples(path: str | Path, data_root: Path) -> dict[Path, list[int]]:
    manifest = Path(path)
    if not manifest.is_absolute() and not manifest.exists():
        manifest = data_root / manifest
    grouped: dict[Path, list[int]] = {}
    for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("decision", "keep") != "keep":
            continue
        if int(row["stride"]) != CHUNK_LEN:
            raise ValueError(f"{manifest}:{line_no}: V1 manifest stride must be 8")
        hand_path = Path(row["hand_path"])
        hand_path = hand_path.resolve() if hand_path.is_absolute() else (data_root / hand_path).resolve()
        if row.get("dominant_side", hand_path.stem) != hand_path.stem:
            raise ValueError(f"{manifest}:{line_no}: dominant side and hand_path disagree")
        grouped.setdefault(hand_path, []).append(int(row["current_frame"]))
    if not grouped:
        raise ValueError(f"No kept V1 samples in {manifest}")
    return grouped


class InteractionDynamicsDataset(Dataset):
    """A current hand/object world plus observed hand and target object chunks."""

    def __init__(self, data_path: str | Path, *, file_list: Sequence[str | Path] | None = None,
                 dominant_hand_manifest: str | Path | None = None, num_effect_points: int = 512,
                 chunk_len: int = 8, temporal_stride: int = 1, base_seed: int = 42,
                 max_samples: int | None = None, max_samples_per_sequence: int | None = None,
                 min_object_effect_norm: float = 0.0, hand_side: str | None = None) -> None:
        if (chunk_len, temporal_stride) != (CHUNK_LEN, TEMPORAL_STRIDE):
            raise ValueError("InteractionDynamics V1 requires chunk_len=8 and temporal_stride=1")
        self.data_root = Path(data_path)
        self.num_effect_points = int(num_effect_points)
        self.base_seed = int(base_seed)
        paths = ([Path(p) for p in file_list] if file_list is not None else
                 sorted(self.data_root.rglob("*.npz")))
        self.file_paths = [p for p in paths if p.name in {"left.npz", "right.npz"}]
        if hand_side is not None:
            if hand_side not in {"left", "right"}:
                raise ValueError(f"Unsupported hand_side: {hand_side}")
            self.file_paths = [path for path in self.file_paths if path.stem == hand_side]
        selected = None if dominant_hand_manifest is None else _manifest_samples(dominant_hand_manifest, self.data_root)
        if selected is not None:
            self.file_paths = [p for p in self.file_paths if p.resolve() in selected]
        if not self.file_paths:
            raise ValueError(f"No InteractionDynamics hand files found under {self.data_root}")
        self._samples: list[tuple[Path, int]] = []
        for hand_path in self.file_paths:
            shared_path = hand_path.parent / "shared.npz"
            with np.load(shared_path, allow_pickle=False) as shared, np.load(hand_path, allow_pickle=False) as hand:
                self._validate_sequence(shared_path, hand_path, shared, hand)
                count = len(shared["raw_frame_id"])
                currents = selected[hand_path.resolve()] if selected is not None else range(count - CHUNK_LEN)
                object_points = (np.asarray(shared["obj_points_world"], dtype=np.float32)
                                 if min_object_effect_norm > 0 else None)
                candidate_masks = (np.asarray(hand["obj_candidate_mask_5cm"], dtype=bool)
                                   if min_object_effect_norm > 0 else None)
                for current in currents:
                    if current < 0 or current + CHUNK_LEN >= count:
                        raise ValueError(f"{hand_path}: chunk at {current} is out of bounds")
                    if min_object_effect_norm > 0:
                        assert object_points is not None and candidate_masks is not None
                        candidate = candidate_masks[current]
                        endpoint_motion = np.linalg.norm(
                            object_points[current + CHUNK_LEN, candidate]
                            - object_points[current, candidate], axis=-1,
                        ).mean() if candidate.any() else 0.0
                        if float(endpoint_motion) < float(min_object_effect_norm):
                            continue
                    self._samples.append((hand_path, current))
        self._samples.sort(key=lambda item: (str(item[0]), item[1]))
        if max_samples_per_sequence is not None:
            by_sequence: dict[Path, list[tuple[Path, int]]] = {}
            for sample in self._samples:
                by_sequence.setdefault(sample[0].parent.resolve(), []).append(sample)
            limited: list[tuple[Path, int]] = []
            limit = int(max_samples_per_sequence)
            for sequence_samples in by_sequence.values():
                sequence_samples.sort(key=lambda item: (item[1], str(item[0])))
                if len(sequence_samples) > limit:
                    indices = np.rint(np.linspace(0, len(sequence_samples) - 1, limit)).astype(np.int64)
                    sequence_samples = [sequence_samples[index] for index in indices]
                limited.extend(sequence_samples)
            self._samples = sorted(limited, key=lambda item: (str(item[0]), item[1]))
        if max_samples is not None:
            self._samples = self._samples[:int(max_samples)]
        self._cache_path: Path | None = None
        self._cache: dict[str, np.ndarray] | None = None
        self._action_atlas: dict[str, np.ndarray] = {}

    @staticmethod
    def _validate_sequence(shared_path: Path, hand_path: Path, shared: np.lib.npyio.NpzFile,
                           hand: np.lib.npyio.NpzFile) -> None:
        if str(shared["schema_name"].item()) != SHARED_SCHEMA_NAME:
            raise ValueError(f"{shared_path}: invalid schema")
        if str(hand["schema_name"].item()) != HAND_SCHEMA_NAME:
            raise ValueError(f"{hand_path}: invalid schema")
        if str(shared["coordinate_frame"].item()) != "world":
            raise ValueError(f"{shared_path}: expected world coordinates")
        t = len(shared["raw_frame_id"])
        expected = {
            "obj_points_world": (t, 4096, 3), "obj_normals_world": (t, 4096, 3),
            "obj_root_pose_world": (t, 4, 4), "hand_points_world": (t, 1538, 3),
            "hand_normals_world": (t, 1538, 3), "hand_root_pose_world": (t, 4, 4),
            "obj_candidate_mask_5cm": (t, 4096),
        }
        for key, shape in expected.items():
            array = hand[key] if key in hand.files else shared[key]
            if array.shape != shape:
                raise ValueError(f"{hand_path if key in hand.files else shared_path}: {key} has {array.shape}, expected {shape}")
        if int(shared["ds_rate"].item()) != 4 or float(shared["source_fps"].item()) != 120.0:
            raise ValueError(f"{shared_path}: V1 requires 120 Hz source and ds_rate=4")

    def __len__(self) -> int:
        return len(self._samples)

    def sample_location(self, index: int) -> tuple[Path, int]:
        return self._samples[index]

    def _load(self, hand_path: Path) -> dict[str, np.ndarray]:
        if hand_path != self._cache_path:
            with np.load(hand_path.parent / "shared.npz", allow_pickle=False) as shared, \
                 np.load(hand_path, allow_pickle=False) as hand:
                self._cache = {key: np.asarray(shared[key]) for key in shared.files}
                self._cache.update({key: np.asarray(hand[key]) for key in hand.files})
            self._cache_path = hand_path
        assert self._cache is not None
        return self._cache

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        hand_path, current = self._samples[index]
        data = self._load(hand_path)
        future = np.arange(current + 1, current + CHUNK_LEN + 1)
        hand_pose = data["hand_root_pose_world"][current]
        obj_pose = data["obj_root_pose_world"][current]
        hand_world_0 = data["hand_points_world"][current]
        obj_world_0 = data["obj_points_world"][current]

        world_hand = transform_points(hand_world_0, obj_pose)
        world_obj = transform_points(obj_world_0, obj_pose)
        action_hand = transform_points(hand_world_0, hand_pose)
        hand_future = transform_points(data["hand_points_world"][future], hand_pose)
        hand_poses = data["hand_root_pose_world"]
        hand_local = np.stack([
            transform_points(data["hand_points_world"][frame], hand_poses[frame])
            for frame in np.concatenate([[current], future])
        ])
        hand_articulation_increment = np.diff(hand_local, axis=0)
        hand_previous_pose = hand_poses[np.concatenate([[current], future[:-1]])]
        hand_root_increment = np.linalg.inv(hand_previous_pose) @ hand_poses[future]
        hand_future_object = transform_points(data["hand_points_world"][future], obj_pose)
        object_frame_ids = np.concatenate([[current], future])
        hand_object_sequence = np.stack([
            transform_points(data["hand_points_world"][frame], data["obj_root_pose_world"][frame])
            for frame in object_frame_ids
        ])
        obj_future = transform_points(data["obj_points_world"][future], obj_pose)
        obj_normals_future = transform_normals(data["obj_normals_world"][future], obj_pose)
        object_poses = data["obj_root_pose_world"]
        previous_pose = object_poses[np.concatenate([[current], future[:-1]])]
        object_increment = np.linalg.inv(previous_pose) @ object_poses[future]
        canonical = np.asarray(data["hand_cano_points"], dtype=np.float32)
        atlas_key = str(data["side"].item())
        if atlas_key not in self._action_atlas:
            _, _, atlas = patchify(torch.from_numpy(canonical)[None], 64, 32)
            self._action_atlas[atlas_key] = atlas[0].numpy()

        raw_frame = int(data["raw_frame_id"][current])
        seed = stable_frame_seed(base_seed=self.base_seed, seq_id=str(data["seq_id"].item()),
                                 side=str(data["side"].item()), raw_frame_id=raw_frame,
                                 epoch=0, namespace="interaction-effect-sampling")
        selected, valid = sample_object_indices(data["obj_candidate_mask_5cm"][current].astype(bool),
                                                num_samples=self.num_effect_points, seed=seed)
        safe = np.maximum(selected, 0)
        dense_obj = transform_points(obj_world_0[safe], hand_pose)
        effect_obj = world_obj[safe]
        effect_normals = transform_normals(data["obj_normals_world"][current, safe], obj_pose)
        effect_disp = obj_future[:, safe] - effect_obj[None]
        obj_disp = obj_future - world_obj[None]
        dense_normals = transform_normals(data["obj_normals_world"][current, safe], hand_pose)
        dense_obj[~valid] = dense_normals[~valid] = effect_obj[~valid] = effect_normals[~valid] = 0
        effect_disp[:, ~valid] = 0

        result = {
            "world_hand_points_object": world_hand,
            "world_hand_normals_object": transform_normals(data["hand_normals_world"][current], obj_pose),
            "world_obj_points_object": world_obj,
            "world_obj_normals_object": transform_normals(data["obj_normals_world"][current], obj_pose),
            "action_hand_points_hand": action_hand,
            "action_hand_normals_hand": transform_normals(data["hand_normals_world"][current], hand_pose),
            "action_hand_points_local_sequence": hand_local.astype(np.float32),
            # Y-Teacher V1 uses each frame's own dynamic object frame. This is
            # deliberately distinct from current-object-frame effect targets.
            "action_hand_points_object_sequence": hand_object_sequence.astype(np.float32),
            "future_hand_points_object_endpoint": hand_future_object[-1].astype(np.float32),
            "action_hand_cano_points": canonical,
            "action_patch_knn_idx": self._action_atlas[atlas_key],
            "action_hand_root_increment_pose": hand_root_increment.astype(np.float32),
            "hand_disp_chunk": hand_future - action_hand[None],
            # V12-A targets remove wrist SE(3) before differencing stable MANO vertices.
            "hand_articulation_increment_gt": hand_articulation_increment.astype(np.float32),
            "hand_root_increment_pose_gt": hand_root_increment.astype(np.float32),
            "action_hand_disp_chunk_object": hand_future_object - world_hand[None],
            # Runner-side mechanism diagnostics/supervision use this target in the
            # current object frame. It is deliberately absent from model.forward.
            "hand_disp_chunk_object_gt": hand_future_object - world_hand[None],
            "dense_obj_points_hand": dense_obj,
            "dense_obj_normals_hand": dense_normals,
            "dense_hand_points_hand": action_hand,
            "dense_hand_normals_hand": transform_normals(data["hand_normals_world"][current], hand_pose),
            "effect_obj_points_object": effect_obj,
            "effect_obj_normals_object": effect_normals,
            "effect_obj_disp_gt": effect_disp,
            # Full stable-index object target is used only by Runner-side patch
            # supervision; it is deliberately absent from model.forward inputs.
            "obj_disp_chunk_gt": obj_disp,
            "obj_normals_chunk_object_gt": obj_normals_future,
            "obj_increment_pose_gt": object_increment.astype(np.float32),
            "effect_obj_valid_mask": valid,
            "effect_obj_idx": selected.astype(np.int64),
        }
        tensors = {key: torch.from_numpy(np.asarray(value)) for key, value in result.items()}
        tensors.update({
            "raw_frame_id": torch.tensor(raw_frame, dtype=torch.int64),
            "future_raw_frame_ids": torch.from_numpy(data["raw_frame_id"][future].astype(np.int64)),
            "frame_dt_s": torch.tensor(1.0 / 30.0),
            "chunk_duration_s": torch.tensor(CHUNK_LEN / 30.0),
            "temporal_stride": torch.tensor(TEMPORAL_STRIDE),
        })
        return tensors


def sequence_group_key(path: Path) -> str:
    return path.parent.as_posix()


def ensure_sequence_disjoint(train: Sequence[Path], val: Sequence[Path], test: Sequence[Path]) -> None:
    groups = [{p.resolve().parent for p in split} for split in (train, val, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("InteractionDynamics splits overlap by sequence")
