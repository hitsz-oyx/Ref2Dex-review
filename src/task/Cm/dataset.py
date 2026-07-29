"""Runtime-stride Cm samples from cached complete GRAB sequences."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base import make_file_split_dataloaders
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed


SEQUENCE_SCHEMA_NAME = "ref2dex_cm_sequence"
REQUIRED_FIELDS = {
    "schema_name", "raw_frame_id", "obj_points_world", "obj_normals_world", "obj_point_id",
    "hand_points_world", "hand_normals_world", "hand_root_pose_world",
    "obj_to_hand_min_dist", "obj_candidate_mask_5cm",
}


def scalar_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
    value = data.get(key)
    if value is None:
        return default
    value = np.asarray(value)
    return str(value.item()) if value.size == 1 else default


def _world_to_hand(points: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    """Map world points to the current ``hand_root_t`` frame."""
    rotation = np.asarray(pose_world[:3, :3], dtype=np.float32)
    translation = np.asarray(pose_world[:3, 3], dtype=np.float32)
    return ((np.asarray(points, dtype=np.float32) - translation) @ rotation).astype(np.float32)


def _normal_world_to_hand(normals: np.ndarray, pose_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(pose_world[:3, :3], dtype=np.float32)
    result = np.asarray(normals, dtype=np.float32) @ rotation
    return (result / np.clip(np.linalg.norm(result, axis=-1, keepdims=True), 1e-8, None)).astype(np.float32)


class Stage4CmDataset(Dataset):
    """One active current frame, with endpoint stride chosen at runtime.

    The cached NPZ contains states only.  Future geometry is read solely to
    construct supervision after both endpoints are represented in H_t.
    """

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        base_seed: int = 42,
        active_only: bool = True,
        min_stride: int = 1,
        max_stride: int = 12,
        fixed_stride: int | None = None,
        max_samples: int | None = None,
        coordinate_frame: str = "hand_root_t",
    ) -> None:
        if min_stride <= 0 or max_stride < min_stride:
            raise ValueError("Require 0 < min_stride <= max_stride.")
        if fixed_stride is not None and not min_stride <= fixed_stride <= max_stride:
            raise ValueError("fixed_stride must lie in [min_stride, max_stride].")
        self.data_path = Path(data_path)
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.base_seed = int(base_seed)
        # DataLoader persistent workers own Dataset replicas.  A shared value
        # makes BaseRunner.set_epoch visible to every replica.
        self._epoch = mp.Value("q", 0, lock=True)
        self.active_only = bool(active_only)
        self.min_stride, self.max_stride = int(min_stride), int(max_stride)
        self.fixed_stride = None if fixed_stride is None else int(fixed_stride)
        self.coordinate_frame = str(coordinate_frame)
        self.file_paths = (sorted(Path(path) for path in file_list) if file_list is not None else
                           (sorted(self.data_path.glob("**/*.npz")) if self.data_path.is_dir() else [self.data_path]))
        self.file_paths = [path for path in self.file_paths if path.name != "meta.npz"]
        if not self.file_paths:
            raise ValueError(f"No cached Cm sequence NPZ files found in {self.data_path}")
        self._cached_path: Path | None = None
        self._cached_data: dict[str, np.ndarray] | None = None
        self._samples: list[tuple[Path, int]] = []
        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                missing = REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing cached-sequence fields {sorted(missing)}")
                if str(np.asarray(data["schema_name"]).item()) != SEQUENCE_SCHEMA_NAME:
                    raise ValueError(f"{path}: expected schema {SEQUENCE_SCHEMA_NAME!r}")
                if str(np.asarray(data.get("coordinate_frame", "world")).item()) != "world":
                    raise ValueError(f"{path}: cached sequence coordinates must be world-frame")
                if data["obj_points_world"].shape[1:] != (4096, 3):
                    raise ValueError(f"{path}: expected obj_points_world [T,4096,3]")
                if data["hand_points_world"].shape[1:] != (self.num_hand_points, 3):
                    raise ValueError(f"{path}: unexpected hand point shape")
                candidate = np.asarray(data["obj_candidate_mask_5cm"], dtype=bool)
                frame_count = int(data["raw_frame_id"].shape[0])
                for current in range(0, frame_count - self.max_stride):
                    if not self.active_only or candidate[current].any():
                        self._samples.append((path, current))
        if max_samples is not None and int(max_samples) > 0:
            self._samples = self._samples[:int(max_samples)]
        if not self._samples:
            raise ValueError("No active frames with the configured maximum future stride.")

    def __len__(self) -> int:
        return len(self._samples)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        if self._cached_path != path or self._cached_data is None:
            with np.load(path, allow_pickle=False) as data:
                self._cached_data = {key: np.asarray(data[key]) for key in data.files}
            self._cached_path = path
        return self._cached_data

    def sample_location(self, index: int) -> tuple[Path, int]:
        return self._samples[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, current = self._samples[index]
        data = self._load_file(path)
        raw_frame = int(data["raw_frame_id"][current])
        seed_args = dict(base_seed=self.base_seed, seq_id=scalar_string(data, "seq_id", path.stem),
                         side=scalar_string(data, "side", ""), raw_frame_id=raw_frame, epoch=self.epoch)
        stride_seed = stable_frame_seed(**seed_args, namespace="cm-stride")
        object_seed = stable_frame_seed(**seed_args, namespace="cm-object-sampling")
        stride = self.fixed_stride if self.fixed_stride is not None else int(
            np.random.default_rng(stride_seed).integers(self.min_stride, self.max_stride + 1)
        )
        future = current + stride
        pose = data["hand_root_pose_world"][current]
        candidate = np.asarray(data["obj_candidate_mask_5cm"][current], dtype=bool)
        selected_idx, valid = sample_object_indices(candidate, num_samples=self.num_obj_points, seed=object_seed)
        safe = np.maximum(selected_idx, 0)
        obj_current = _world_to_hand(data["obj_points_world"][current, safe], pose)
        obj_future = _world_to_hand(data["obj_points_world"][future, safe], pose)
        obj_normals = _normal_world_to_hand(data["obj_normals_world"][current, safe], pose)
        obj_current[~valid] = obj_future[~valid] = obj_normals[~valid] = 0.0
        hand_current = _world_to_hand(data["hand_points_world"][current], pose)
        hand_future = _world_to_hand(data["hand_points_world"][future], pose)
        return {
            "obj_points": torch.from_numpy(obj_current), "obj_normals": torch.from_numpy(obj_normals),
            "obj_flow_gt": torch.from_numpy(obj_future - obj_current),
            "hand_points": torch.from_numpy(hand_current),
            "hand_normals": torch.from_numpy(_normal_world_to_hand(data["hand_normals_world"][current], pose)),
            "hand_flow": torch.from_numpy(hand_future - hand_current),
            "obj_valid_mask": torch.from_numpy(valid), "selected_obj_idx": torch.from_numpy(selected_idx.astype(np.int64)),
            "raw_frame_id": torch.tensor(raw_frame), "next_raw_frame_id": torch.tensor(int(data["raw_frame_id"][future])),
            "stride": torch.tensor(stride),
        }


def _sequence_group_key(path: Path) -> str:
    stem = path.stem.rsplit("_", 1)[0] if path.stem.endswith(("_left", "_right")) else path.stem
    return f"{path.parent.as_posix()}/{stem}"


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any | None = None):
    common = {"num_obj_points": int(meta_cfg.num_obj_points), "num_hand_points": int(meta_cfg.num_hand_points),
              "base_seed": int(seed), "active_only": bool(getattr(data_cfg, "active_only", True)),
              "min_stride": int(getattr(data_cfg, "min_stride", 1)), "max_stride": int(getattr(data_cfg, "max_stride", 12)),
              "coordinate_frame": str(meta_cfg.coordinate_frame)}
    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg, seed, dataset_cls=Stage4CmDataset, file_pattern="**/*.npz",
        train_dataset_kwargs={**common, "max_samples": getattr(data_cfg, "max_train_samples", None)},
        val_dataset_kwargs={**common, "fixed_stride": int(getattr(data_cfg, "val_stride", 1)),
                            "max_samples": getattr(data_cfg, "max_val_samples", None)},
        split_group_fn=_sequence_group_key if bool(getattr(data_cfg, "group_val_by_sequence", True)) else None,
        distributed=distributed,
    )
    first = train_loader.dataset.file_paths[0]
    with np.load(first, allow_pickle=False) as data:
        metadata.update({"schema_name": str(np.asarray(data["schema_name"]).item()), "coordinate_frame": "hand_root_t",
                         "num_obj_pool": int(data["obj_points_world"].shape[1]), "num_obj_points": int(meta_cfg.num_obj_points),
                         "num_hand_points": int(data["hand_points_world"].shape[1]), "min_stride": common["min_stride"],
                         "max_stride": common["max_stride"]})
    val_loaders: dict[str, DataLoader] = {}
    if val_loader is not None:
        # Evaluation is deterministic and reports each fixed horizon separately.
        # Constructing views from the held-out file list preserves the original
        # sequence-level split without materializing pair files.
        val_paths = val_loader.dataset.file_paths
        for stride in tuple(getattr(data_cfg, "val_strides", (1, 2, 3, 4, 5))):
            stride = int(stride)
            if not common["min_stride"] <= stride <= common["max_stride"]:
                continue
            dataset = Stage4CmDataset(
                data_cfg.val_path or data_cfg.train_path, file_list=val_paths,
                fixed_stride=stride, max_samples=getattr(data_cfg, "max_val_samples", None), **common,
            )
            loader_kwargs = {"batch_size": int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                             "shuffle": False, "num_workers": int(getattr(data_cfg, "num_workers", 0)),
                             "pin_memory": bool(getattr(data_cfg, "pin_memory", False))}
            if loader_kwargs["num_workers"] > 0:
                loader_kwargs["persistent_workers"] = bool(getattr(data_cfg, "persistent_workers", False))
                prefetch = getattr(data_cfg, "prefetch_factor", None)
                if prefetch is not None:
                    loader_kwargs["prefetch_factor"] = int(prefetch)
            val_loaders[f"val/stride_{stride}/"] = DataLoader(dataset, **loader_kwargs)
    if val_loader is not None and not val_loaders:
        val_loaders["val/stride_1/"] = val_loader
    metadata["val_loader_names"] = sorted(val_loaders)
    return train_loader, val_loader, metadata, val_loaders
