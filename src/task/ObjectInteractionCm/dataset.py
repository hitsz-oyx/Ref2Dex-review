"""Dual-hand object-interaction Dataset for existing Ref2Dex caches."""
from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, make_default_train_sampler


def _stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def _world_to_frame(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_world, dtype=np.float32)
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return ((np.asarray(points, dtype=np.float32) - translation) @ rotation).astype(np.float32)


def _normal_world_to_frame(normals: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose_world, dtype=np.float32)[:3, :3]
    result = np.asarray(normals, dtype=np.float32) @ rotation
    return (result / np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _nearest_distances(points: np.ndarray, object_points: np.ndarray) -> np.ndarray:
    """Nearest object distance with a scipy KD-tree and a numpy fallback."""
    try:
        from scipy.spatial import cKDTree

        return np.asarray(cKDTree(np.asarray(object_points, dtype=np.float32)).query(
            np.asarray(points, dtype=np.float32), k=1, workers=1
        )[0], dtype=np.float32)
    except (ImportError, TypeError):
        result = np.empty((len(points),), dtype=np.float32)
        for start in range(0, len(points), 512):
            chunk = np.asarray(points[start:start + 512], dtype=np.float32)
            delta = chunk[:, None, :] - np.asarray(object_points, dtype=np.float32)[None, :, :]
            result[start:start + len(chunk)] = np.sqrt((delta * delta).sum(axis=-1).min(axis=-1))
        return result


class _SequenceView:
    """Lazy sequence-level view supporting mmap and shared/side NPZ layouts."""

    def __init__(self, path: Path, *, expected_hand_points: int) -> None:
        self.path = Path(path)
        self.expected_hand_points = int(expected_hand_points)
        self._arrays: dict[str, Any] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        shared_dir = self.path / "shared"
        if (shared_dir / "meta.json").is_file():
            self.kind = "mmap"
            self._meta = json.loads((shared_dir / "meta.json").read_text(encoding="utf-8"))
            self.dataset_id = str(self._meta.get("dataset_name") or self._meta.get("source") or "unknown")
            self.source_name = "grab"
            self._shared_root = shared_dir
            self._side_root = self.path
            self._fields = {
                "obj": shared_dir / "obj_points_world.npy",
                "obj_normals": shared_dir / "obj_normals_world.npy",
                "pose": shared_dir / "obj_pose_world.npy",
                "raw": shared_dir / "raw_frame_id.npy",
            }
            if not self._fields["pose"].is_file():
                raise ValueError(f"{self.path}: object_pose_t requires shared/obj_pose_world.npy")
            self._load_mmap_streams()
        elif (self.path / "shared.npz").is_file():
            self.kind = "npz"
            with np.load(self.path / "shared.npz", allow_pickle=False) as shared:
                self._shared_data = {key: np.asarray(shared[key]) for key in shared.files}
            self._meta = self._shared_data
            self.dataset_id = str(_scalar(self._shared_data.get("dataset_name"), "unknown"))
            self.source_name = "grab"
            pose_key = "obj_pose_world" if "obj_pose_world" in self._shared_data else "obj_root_pose_world"
            if pose_key not in self._shared_data:
                raise ValueError(f"{self.path}: object_pose_t requires obj_pose_world or obj_root_pose_world")
            self._fields = {"pose_key": pose_key}
            self._load_npz_streams()
        elif (self.path / "geometry" / "manifest.json").is_file():
            self.kind = "inspire"
            geometry_dir = self.path / "geometry"
            self._meta = json.loads((geometry_dir / "manifest.json").read_text(encoding="utf-8"))
            self.dataset_id = "inspire_f1"
            self.source_name = "inspire_f1"
            self._fields = {
                "obj": geometry_dir / "obj_points_pool_world.npy",
                "obj_normals": geometry_dir / "obj_normals_pool_world.npy",
                "pose": geometry_dir / "obj_pose_world.npy",
                "raw": geometry_dir / "source_frame_id.npy",
                "frame_time": geometry_dir / "frame_time.npy",
            }
            required = tuple(self._fields.values())
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ValueError(f"{self.path}: Inspire geometry missing {missing}")
            active_path = geometry_dir / "obj_candidate_mask_5cm_recomputed.npy"
            if not active_path.is_file():
                active_path = geometry_dir / "obj_candidate_mask_5cm.npy"
            if not active_path.is_file():
                raise ValueError(f"{self.path}: Inspire geometry missing object candidate mask")
            self._inspire_active = np.asarray(np.load(active_path, mmap_mode="r"), dtype=bool)
            if self._inspire_active.ndim == 2:
                self._inspire_active = self._inspire_active.any(axis=1)
            self._load_inspire_stream()
        else:
            raise ValueError(f"Unsupported ObjectInteractionCm sequence layout: {self.path}")
        self._finalize_metadata()

    def _finalize_metadata(self) -> None:
        obj = self.array("obj")
        normals = self.array("obj_normals")
        pose = self.array("pose")
        raw = self.array("raw")
        if obj.ndim != 3 or obj.shape[1:] != (4096, 3):
            raise ValueError(f"{self.path}: expected object pool [T,4096,3], got {obj.shape}")
        if normals.shape != obj.shape or pose.shape != (obj.shape[0], 4, 4) or len(raw) != obj.shape[0]:
            raise ValueError(f"{self.path}: shared object/pose/raw shape mismatch")
        frame_count = obj.shape[0]
        for stream_name, arrays in self.streams.items():
            if arrays["hand"].shape[0] != frame_count or len(arrays["active"]) != frame_count:
                raise ValueError(f"{self.path}/{stream_name}: frame count mismatch")
        raw_values = np.asarray(raw)
        if not np.isfinite(raw_values).all():
            raise ValueError(f"{self.path}: invalid raw_frame_id")
        self.frame_count = frame_count
        self.effective_fps = float(_scalar(self._meta.get("effective_fps"), 0.0) or 0.0)
        if self.effective_fps <= 0.0 and self.kind == "inspire":
            frame_time = np.asarray(self.array("frame_time"), dtype=np.float64)
            deltas = np.diff(frame_time)
            positive = deltas[np.isfinite(deltas) & (deltas > 1e-6)]
            if positive.size:
                self.effective_fps = float(1.0 / np.median(positive))
        if self.effective_fps <= 0.0:
            fps = float(_scalar(self._meta.get("source_fps"), 30.0) or 30.0)
            ds_rate = int(_scalar(self._meta.get("ds_rate"), 1) or 1)
            self.effective_fps = fps / max(ds_rate, 1)

    def _load_mmap_streams(self) -> None:
        self.streams: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            root = self.path / side
            required = ["hand_points_world.npy", "hand_normals_world.npy"]
            if not root.is_dir() or any(not (root / name).is_file() for name in required):
                raise ValueError(f"{self.path}: both left/right hand streams are required")
            arrays = {
                "hand": np.load(root / "hand_points_world.npy", mmap_mode="r"),
                "hand_normals": np.load(root / "hand_normals_world.npy", mmap_mode="r"),
            }
            active_path = root / "candidate_active_5cm.npy"
            if active_path.is_file():
                arrays["active"] = np.load(active_path, mmap_mode="r").astype(bool)
            elif (root / "candidate_offsets.npy").is_file():
                offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
                arrays["active"] = offsets[1:] > offsets[:-1]
            else:
                raise ValueError(f"{self.path}/{side}: missing candidate activity metadata")
            if arrays["hand"].ndim != 3 or arrays["hand"].shape[1:] != (self.expected_hand_points, 3):
                raise ValueError(f"{self.path}/{side}: expected hand_points [T,{self.expected_hand_points},3]")
            if arrays["hand_normals"].shape != arrays["hand"].shape:
                raise ValueError(f"{self.path}/{side}: hand normals shape mismatch")
            self.streams[side] = arrays

    def _load_npz_streams(self) -> None:
        self.streams = {}
        for side in ("left", "right"):
            side_path = self.path / f"{side}.npz"
            if not side_path.is_file():
                raise ValueError(f"{self.path}: both left.npz and right.npz are required")
            with np.load(side_path, allow_pickle=False) as side_data:
                side_arrays = {key: np.asarray(side_data[key]) for key in side_data.files}
            if "obj_candidate_mask_5cm" not in side_arrays:
                raise ValueError(f"{side_path}: missing obj_candidate_mask_5cm")
            arrays = {
                "hand": side_arrays["hand_points_world"],
                "hand_normals": side_arrays["hand_normals_world"],
                "active": np.asarray(side_arrays["obj_candidate_mask_5cm"], dtype=bool).any(axis=1),
            }
            if arrays["hand"].ndim != 3 or arrays["hand"].shape[1:] != (self.expected_hand_points, 3):
                raise ValueError(f"{self.path}/{side}: expected hand_points [T,{self.expected_hand_points},3]")
            if arrays["hand_normals"].shape != arrays["hand"].shape:
                raise ValueError(f"{self.path}/{side}: hand normals shape mismatch")
            self.streams[side] = arrays

    def _load_inspire_stream(self) -> None:
        geometry_dir = self.path / "geometry"
        hand = np.load(geometry_dir / "hand_points_world.npy", mmap_mode="r")
        normals = np.load(geometry_dir / "hand_normals_world.npy", mmap_mode="r")
        if hand.ndim != 3 or hand.shape[1:] != (self.expected_hand_points, 3):
            raise ValueError(f"{self.path}: expected Inspire hand_points [T,{self.expected_hand_points},3]")
        if normals.shape != hand.shape:
            raise ValueError(f"{self.path}: Inspire hand normals shape mismatch")
        self.streams = {"inspire_f1": {"hand": hand, "hand_normals": normals, "active": self._inspire_active}}

    @property
    def sides(self) -> dict[str, dict[str, Any]]:
        """Backward-compatible alias for the available hand streams."""
        return self.streams

    def array(self, name: str) -> np.ndarray:
        if name in self._arrays:
            return self._arrays[name]
        if self.kind in {"mmap", "inspire"}:
            value = np.load(self._fields[name], mmap_mode="r")
        else:
            key = self._fields["pose_key"] if name == "pose" else {
                "obj": "obj_points_world", "obj_normals": "obj_normals_world", "raw": "raw_frame_id"
            }[name]
            value = self._shared_data[key]
        self._arrays[name] = value
        return value

    def candidate_active(self, frame: int) -> bool:
        return any(bool(stream["active"][frame]) for stream in self.streams.values())

    def hand(self, side: str, frame: int) -> tuple[np.ndarray, np.ndarray]:
        arrays = self.streams[side]
        return np.asarray(arrays["hand"][frame], dtype=np.float32), np.asarray(arrays["hand_normals"][frame], dtype=np.float32)

    def hands(self, frame: int) -> list[tuple[str, np.ndarray, np.ndarray]]:
        return [
            (name, np.asarray(arrays["hand"][frame], dtype=np.float32),
             np.asarray(arrays["hand_normals"][frame], dtype=np.float32))
            for name, arrays in self.streams.items()
        ]


class ObjectInteractionCmDataset(Dataset):
    """One dual-hand sample per sequence/current frame with runtime stride."""

    def __init__(
        self,
        roots: str | Path | Sequence[str | Path],
        *,
        num_obj_points: int = 1024,
        num_hand_points: int = 1538,
        max_hand_points: int | None = None,
        base_seed: int = 42,
        min_stride: int = 1,
        max_stride: int = 10,
        fixed_stride: int | None = None,
        active_only: bool = True,
        sequence_paths: Sequence[str | Path] | None = None,
        sequence_entries: Sequence[Mapping[str, Any]] | None = None,
        source_stride_values: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(root) for root in roots]
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.max_hand_points = int(max_hand_points or (self.num_hand_points * 2))
        self.base_seed = int(base_seed)
        self.min_stride = int(min_stride)
        self.max_stride = int(max_stride)
        self.fixed_stride = None if fixed_stride is None else int(fixed_stride)
        self.active_only = bool(active_only)
        self.source_stride_values = {
            str(key): tuple(int(value) for value in values)
            for key, values in (source_stride_values or {}).items()
        }
        if self.num_obj_points <= 0 or self.num_obj_points > 4096:
            raise ValueError("num_obj_points must lie in [1,4096]")
        if self.max_hand_points <= 0 or self.max_hand_points < self.num_hand_points:
            raise ValueError("max_hand_points must be >= num_hand_points and positive")
        if self.min_stride <= 0 or self.max_stride < self.min_stride:
            raise ValueError("Require 0 < min_stride <= max_stride")
        if self.fixed_stride is not None and not self.min_stride <= self.fixed_stride <= self.max_stride:
            raise ValueError("fixed_stride must lie in [min_stride,max_stride]")
        for source, values in self.source_stride_values.items():
            if not values or any(value <= 0 for value in values):
                raise ValueError(f"{source}: source stride values must be positive")
        if sequence_entries is not None:
            specs = [(Path(item["path"]), str(item.get("source", ""))) for item in sequence_entries]
        else:
            paths = [Path(path) for path in sequence_paths] if sequence_paths is not None else self._discover_sequences()
            specs = [(Path(path), "") for path in paths]
        self.sequences = []
        for path, source_name in specs:
            view = _SequenceView(path, expected_hand_points=self.num_hand_points)
            if source_name:
                view.source_name = source_name
            self.sequences.append(view)
        self.rows: list[tuple[int, int]] = []
        for sequence_index, sequence in enumerate(self.sequences):
            stride_values = self._stride_values(sequence.source_name)
            sequence_max_stride = max(stride_values) if stride_values else self.max_stride
            last_current = sequence.frame_count - sequence_max_stride - 1
            for frame in range(max(0, last_current + 1)):
                if not self.active_only or sequence.candidate_active(frame):
                    self.rows.append((sequence_index, frame))
        if not self.rows:
            raise ValueError("No valid dual-hand frames in ObjectInteractionCm cache")
        self._epoch = 0

    def _stride_values(self, source_name: str) -> tuple[int, ...]:
        if self.fixed_stride is not None:
            return (self.fixed_stride,)
        values = self.source_stride_values.get(str(source_name))
        if values:
            return values
        return tuple(range(self.min_stride, self.max_stride + 1))

    def _discover_sequences(self) -> list[Path]:
        paths: list[Path] = []
        for root in self.roots:
            if (root / "shared" / "meta.json").is_file() or (root / "shared.npz").is_file():
                paths.append(root)
                continue
            paths.extend(sorted(path.parent.parent for path in root.glob("**/shared/meta.json")))
            paths.extend(sorted(path.parent for path in root.glob("**/shared.npz")))
        unique = sorted({path.resolve() for path in paths})
        if not unique:
            raise FileNotFoundError(f"No ObjectInteractionCm sequences under {self.roots}")
        return unique

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sequence_index, current = self.rows[index]
        sequence = self.sequences[sequence_index]
        raw = int(np.asarray(sequence.array("raw"))[current])
        seed = _stable_seed(self.base_seed, sequence.path, raw, self._epoch)
        stride_values = self._stride_values(sequence.source_name)
        stride = self.fixed_stride or int(np.random.default_rng(seed).choice(stride_values))
        future = current + stride
        pose = sequence.array("pose")[current]
        object_world = sequence.array("obj")[current]
        object_future_world = sequence.array("obj")[future]
        object_normals_world = sequence.array("obj_normals")[current]
        selected = np.random.default_rng(seed ^ 0xA17).choice(4096, size=self.num_obj_points, replace=False)
        object_points = _world_to_frame(object_world[selected], pose)
        object_future = _world_to_frame(object_future_world[selected], pose)
        object_normals = _normal_world_to_frame(object_normals_world[selected], pose)
        hand_points_parts: list[np.ndarray] = []
        hand_normals_parts: list[np.ndarray] = []
        hand_future_parts: list[np.ndarray] = []
        stream_names: list[str] = []
        for stream_name, _, _ in sequence.hands(current):
            current_world, current_normals_world = sequence.hand(stream_name, current)
            future_world, _ = sequence.hand(stream_name, future)
            hand_points_parts.append(_world_to_frame(current_world, pose))
            hand_normals_parts.append(_normal_world_to_frame(current_normals_world, pose))
            hand_future_parts.append(_world_to_frame(future_world, pose))
            stream_names.append(stream_name)
        hand_points_real = np.concatenate(hand_points_parts, axis=0)
        hand_normals_real = np.concatenate(hand_normals_parts, axis=0)
        hand_future_real = np.concatenate(hand_future_parts, axis=0)
        real_hand_count = int(hand_points_real.shape[0])
        if real_hand_count > self.max_hand_points:
            raise ValueError(
                f"{sequence.path}: available hand points {real_hand_count} exceed max_hand_points {self.max_hand_points}"
            )
        hand_points = np.zeros((self.max_hand_points, 3), dtype=np.float32)
        hand_normals = np.zeros_like(hand_points)
        hand_future = np.zeros_like(hand_points)
        hand_points[:real_hand_count] = hand_points_real
        hand_normals[:real_hand_count] = hand_normals_real
        hand_future[:real_hand_count] = hand_future_real
        hand_valid = np.zeros((self.max_hand_points,), dtype=bool)
        hand_valid[:real_hand_count] = True
        full_object_points = _world_to_frame(object_world, pose)
        hand_distances = _nearest_distances(hand_points_real, full_object_points)
        hand_mask = np.zeros((self.max_hand_points,), dtype=bool)
        hand_mask[:real_hand_count] = hand_distances < 0.03
        return {
            "obj_points": torch.from_numpy(object_points),
            "obj_normals": torch.from_numpy(object_normals),
            "obj_flow_gt": torch.from_numpy(object_future - object_points),
            "obj_valid_mask": torch.ones((self.num_obj_points,), dtype=torch.bool),
            "hand_points": torch.from_numpy(hand_points),
            "hand_normals": torch.from_numpy(hand_normals),
            "hand_flow": torch.from_numpy(hand_future - hand_points),
            "hand_valid_mask": torch.from_numpy(hand_valid),
            "hand_supervision_mask": torch.from_numpy(hand_mask),
            "min_hand_object_distance_mm": torch.tensor(float(hand_distances.min() * 1000.0), dtype=torch.float32),
            "raw_frame_id": torch.tensor(raw, dtype=torch.int64),
            "next_raw_frame_id": torch.tensor(int(np.asarray(sequence.array("raw"))[future]), dtype=torch.int64),
            "stride": torch.tensor(stride, dtype=torch.int64),
            "delta_time_s": torch.tensor(float(stride) / sequence.effective_fps, dtype=torch.float32),
            "dataset_id": sequence.dataset_id,
            "source": sequence.source_name,
            "hand_valid_points": torch.tensor(real_hand_count, dtype=torch.int64),
        }


def _read_split_sequences(split_json: Path, root: Path, key: str) -> list[Path]:
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    value = payload.get(key)
    if not value:
        return []
    path = Path(value)
    path = path if path.is_absolute() else split_json.parent / path
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item:
            resolved = (root / item).resolve()
            if not ((resolved / "shared" / "meta.json").is_file() or (resolved / "shared.npz").is_file()):
                raise FileNotFoundError(f"Split references missing sequence: {resolved}")
            result.append(resolved)
    return result


def _split_sequences(paths: list[Path], *, seed: int, val_fraction: float, test_fraction: float):
    rng = np.random.default_rng(int(seed))
    paths = list(paths)
    rng.shuffle(paths)
    n = len(paths)
    n_test = max(1, int(round(n * test_fraction))) if n >= 3 and test_fraction > 0 else 0
    n_val = max(1, int(round(n * val_fraction))) if n - n_test >= 3 and val_fraction > 0 else 0
    return sorted(paths[n_test + n_val:]), sorted(paths[n_test:n_test + n_val]), sorted(paths[:n_test])


class SourceBalancedSampler(Sampler[int]):
    """Replacement sampler with a fixed probability for each data source."""

    def __init__(self, sources: Sequence[str], probabilities: Mapping[str, float], *, seed: int) -> None:
        self.sources = tuple(str(source) for source in sources)
        counts: dict[str, int] = {}
        for source in self.sources:
            counts[source] = counts.get(source, 0) + 1
        requested = {str(key): float(value) for key, value in probabilities.items() if float(value) > 0.0}
        active = {source: value for source, value in requested.items() if source in counts}
        if not active:
            active = {source: 1.0 for source in counts}
        total = sum(active.values())
        self._weights = torch.tensor(
            [active[source] / total / counts[source] for source in self.sources], dtype=torch.double
        )
        self.num_samples = len(self.sources)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self._weights, self.num_samples, replacement=True, generator=generator)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


def _plain_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {str(key): getattr(value, key) for key in vars(value)} if hasattr(value, "__dict__") else {}


def _resolve_index_entries(index_path: Path, split: str) -> list[dict[str, str]]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_name") != "ref2dex_object_interaction_cm_index_v1_1":
        raise ValueError(f"Unsupported ObjectInteractionCm index schema: {payload.get('schema_name')!r}")
    entries = payload.get("sequences", {}).get(split, [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"ObjectInteractionCm index has no {split!r} sequences: {index_path}")
    result: list[dict[str, str]] = []
    for item in entries:
        if not isinstance(item, Mapping) or "path" not in item or "source" not in item:
            raise ValueError(f"Malformed ObjectInteractionCm index entry: {item!r}")
        raw_path = Path(str(item["path"]))
        path = raw_path if raw_path.is_absolute() else (index_path.parent / raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"ObjectInteractionCm index references missing sequence: {path}")
        result.append({"path": str(path), "source": str(item["source"])})
    return result


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any | None = None):
    index_value = str(getattr(data_cfg, "index_path", "") or "").strip()
    if index_value:
        index_path = Path(index_value)
        if not index_path.is_absolute():
            index_path = (Path.cwd() / index_path).resolve()
        train_entries = _resolve_index_entries(index_path, "train")
        val_entries = _resolve_index_entries(index_path, "val")
        test_entries = _resolve_index_entries(index_path, "test")
    else:
        root_value = str(getattr(data_cfg, "root", "") or getattr(data_cfg, "train_path", "")).strip()
        if not root_value:
            raise ValueError("ObjectInteractionCm requires data.index_path or data.root/data.train_path")
        root = Path(root_value).resolve()
        all_paths = ObjectInteractionCmDataset(
            root, num_obj_points=int(meta_cfg.num_obj_points), num_hand_points=int(meta_cfg.num_hand_points),
            min_stride=int(data_cfg.min_stride), max_stride=int(data_cfg.max_stride), active_only=False
        )._discover_sequences()
        split_json = getattr(data_cfg, "split_json_path", None)
        if split_json:
            split_path = Path(split_json)
            if not split_path.is_absolute():
                split_path = (Path.cwd() / split_path).resolve()
            train_paths = _read_split_sequences(split_path, root, "train_split")
            val_paths = _read_split_sequences(split_path, root, "val_split")
            test_paths = _read_split_sequences(split_path, root, "test_split")
        else:
            train_paths, val_paths, test_paths = _split_sequences(
                all_paths, seed=seed, val_fraction=float(getattr(data_cfg, "val_split", 0.1)),
                test_fraction=float(getattr(data_cfg, "test_fraction", 0.1)),
            )
        train_entries = [{"path": str(path), "source": ""} for path in train_paths]
        val_entries = [{"path": str(path), "source": ""} for path in val_paths]
        test_entries = [{"path": str(path), "source": ""} for path in test_paths]

    source_stride_values = {
        "grab": tuple(int(value) for value in getattr(data_cfg, "grab_stride_values", tuple(range(1, 11)))),
        "inspire_f1": tuple(int(value) for value in getattr(data_cfg, "inspire_stride_values", tuple(range(2, 21, 2)))),
    }
    common = dict(
        num_obj_points=int(meta_cfg.num_obj_points), num_hand_points=int(meta_cfg.num_hand_points),
        max_hand_points=int(getattr(meta_cfg, "max_hand_points", int(meta_cfg.num_hand_points) * 2)),
        base_seed=int(seed), min_stride=int(data_cfg.min_stride), max_stride=int(data_cfg.max_stride),
        active_only=bool(getattr(data_cfg, "active_only", True)), source_stride_values=source_stride_values,
    )
    eval_stride = int(getattr(data_cfg, "eval_stride", 2))
    train_dataset = ObjectInteractionCmDataset(Path("."), sequence_entries=train_entries, **common)
    val_dataset = ObjectInteractionCmDataset(Path("."), sequence_entries=val_entries, fixed_stride=eval_stride, **common) if val_entries else None
    test_dataset = ObjectInteractionCmDataset(Path("."), sequence_entries=test_entries, fixed_stride=eval_stride, **common) if test_entries else None

    source_probabilities = _plain_mapping(getattr(data_cfg, "source_probabilities", {}))
    sampler: Sampler[int] | None = None
    if source_probabilities:
        sampler = SourceBalancedSampler(
            [train_dataset.sequences[sequence_index].source_name for sequence_index, _ in train_dataset.rows],
            source_probabilities, seed=seed,
        )
    if distributed is not None and getattr(distributed, "enabled", False):
        from src.base.distributed import shard_sampler_for_distributed
        if sampler is None:
            sampler = make_default_train_sampler(train_dataset, shuffle=True, seed=seed, distributed=distributed, drop_last=False)
        else:
            sampler = shard_sampler_for_distributed(sampler, distributed=distributed, drop_last=False, pad=True)
    kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
    kwargs.update(batch_size=int(data_cfg.batch_size), shuffle=sampler is None, sampler=sampler)
    train_loader = DataLoader(train_dataset, **kwargs)

    def eval_loader(dataset):
        if dataset is None:
            return None
        eval_kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
        eval_kwargs.update(batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size), shuffle=False,
                           sampler=None if distributed is None else make_default_eval_sampler(dataset, distributed=distributed))
        return DataLoader(dataset, **eval_kwargs)

    val_loader, test_loader = eval_loader(val_dataset), eval_loader(test_dataset)
    val_loaders: dict[str, Any] = {"val/": val_loader} if val_loader is not None else {}
    test_loaders: dict[str, Any] = {"test/": test_loader} if test_loader is not None else {}
    if val_entries:
        for source in ("grab", "inspire_f1"):
            entries = [item for item in val_entries if item.get("source") == source]
            if entries:
                source_view = ObjectInteractionCmDataset(Path("."), sequence_entries=entries, fixed_stride=eval_stride, **common)
                val_loaders[f"val/{source}/"] = eval_loader(source_view)
    if test_entries:
        for source in ("grab", "inspire_f1"):
            entries = [item for item in test_entries if item.get("source") == source]
            if entries:
                source_view = ObjectInteractionCmDataset(Path("."), sequence_entries=entries, fixed_stride=eval_stride, **common)
                test_loaders[f"test/{source}/"] = eval_loader(source_view)
    metadata = {
        "schema_name": "ref2dex_object_interaction_cm_v1_1",
        "coordinate_frame": "object_pose_t",
        "num_obj_pool": 4096,
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "max_hand_points": int(getattr(meta_cfg, "max_hand_points", int(meta_cfg.num_hand_points) * 2)),
        "knn_k": int(getattr(meta_cfg, "knn_k", 8)),
        "interaction_radius_m": float(getattr(meta_cfg, "interaction_radius_m", 0.05)),
        "frame_filter_distance_m": float(getattr(meta_cfg, "frame_filter_distance_m", 0.05)),
        "hand_supervision_radius_m": float(getattr(meta_cfg, "hand_supervision_radius_m", 0.03)),
        "source_probabilities": source_probabilities,
        "dataset_split": {"train_sequences": len(train_entries), "val_sequences": len(val_entries), "test_sequences": len(test_entries)},
        "source_split": {
            source: {split: sum(1 for item in entries if item.get("source") == source) for split, entries in (("train", train_entries), ("val", val_entries), ("test", test_entries))}
            for source in ("grab", "inspire_f1")
        },
        "kept_frame_ratio": float(
            len(train_dataset)
            / max(sum(max(0, s.frame_count - max(train_dataset._stride_values(s.source_name))) for s in train_dataset.sequences), 1)
        ),
    }
    return train_loader, val_loader, test_loader, metadata, val_loaders, test_loaders
