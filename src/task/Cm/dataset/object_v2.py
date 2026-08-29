"""Unified object-only Cm dataset for GRAB and ARCTIC Stage4 caches."""
from __future__ import annotations

import json
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, make_default_train_sampler
from src.task.Cm.dataset import Stage4CmDataset, _normal_world_to_hand, _world_to_hand
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed


class _MmapSequenceDataset(Dataset):
    def __init__(self, sequence_dirs: List[Path], *, num_obj_points: int = 512, num_hand_points: int = 1538,
                 base_seed: int = 42, min_stride: int = 1, max_stride: int = 10, fixed_stride: Optional[int] = None,
                 active_only: bool = True, sampling_bank_size: int = 4, fixed_eval_bank: int = 0):
        self.num_obj_points, self.num_hand_points = num_obj_points, num_hand_points
        self.base_seed, self.min_stride, self.max_stride = base_seed, min_stride, max_stride
        self.fixed_stride, self.active_only = fixed_stride, active_only
        self._epoch = mp.Value("q", 0, lock=True)
        self.sampling_bank_size, self.fixed_eval_bank = max(1, int(sampling_bank_size)), int(fixed_eval_bank)
        self.rows: List[Tuple[Path, str, int]] = []
        self._cache: "OrderedDict[Path, Dict[str, np.ndarray]]" = OrderedDict()
        self._open_sequence_limit = 16
        for sequence in sequence_dirs:
            shared = sequence / "shared"
            for side in ("left", "right"):
                side_dir = sequence / side
                if not (side_dir / "candidate_offsets.npy").exists():
                    continue
                offsets = np.load(side_dir / "candidate_offsets.npy", mmap_mode="r")
                limit = len(offsets) - 1 - max_stride
                for frame in range(max(0, limit)):
                    if not active_only or offsets[frame + 1] > offsets[frame]:
                        self.rows.append((sequence, side, frame))
        if not self.rows:
            raise ValueError("No valid samples in object V2 mmap root")

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, sequence: Path, side: str) -> Dict[str, np.ndarray]:
        key = sequence / side
        if key not in self._cache:
            shared = sequence / "shared"
            self._cache[key] = {
                "obj": np.load(shared / "obj_points_world.npy", mmap_mode="r"),
                "obj_normals": np.load(shared / "obj_normals_world.npy", mmap_mode="r"),
                "raw": np.load(shared / "raw_frame_id.npy", mmap_mode="r"),
                "hand": np.load(key / "hand_points_world.npy", mmap_mode="r"),
                "hand_normals": np.load(key / "hand_normals_world.npy", mmap_mode="r"),
                "pose": np.load(key / "hand_root_pose_world.npy", mmap_mode="r"),
                "offsets": np.load(key / "candidate_offsets.npy", mmap_mode="r"),
                "indices": np.load(key / "candidate_indices.npy", mmap_mode="r"),
                "sampling": np.load(key / "sampling_indices.npy", mmap_mode="r") if (key / "sampling_indices.npy").exists() else None,
            }
        else:
            self._cache.move_to_end(key)
        while len(self._cache) > self._open_sequence_limit:
            self._cache.popitem(last=False)
        return self._cache[key]

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, str]]:
        sequence, side, current = self.rows[index]
        data = self._load(sequence, side)
        raw = int(data["raw"][current])
        seed = stable_frame_seed(base_seed=self.base_seed, seq_id=str(sequence), side=side, raw_frame_id=raw, epoch=self.epoch, namespace="cm-object-v2")
        stride = self.fixed_stride or int(np.random.default_rng(seed).integers(self.min_stride, self.max_stride + 1))
        future = current + stride
        candidate_mask = np.zeros(data["obj"].shape[1], dtype=bool)
        candidate = np.asarray(data["indices"][data["offsets"][current]:data["offsets"][current + 1]], dtype=np.uint32)
        candidate_mask[candidate] = True
        bank = data["sampling"]
        if bank is not None and bank.shape[1] >= self.sampling_bank_size and bank.shape[2] == self.num_obj_points:
            bank_index = self.fixed_eval_bank if self.fixed_stride is not None else seed % self.sampling_bank_size
            selected = np.asarray(bank[current, bank_index], dtype=np.int64)
            valid = candidate_mask[selected]
        else:
            selected, valid = sample_object_indices(candidate_mask, num_samples=self.num_obj_points, seed=seed)
        safe = np.maximum(selected, 0)
        pose = np.asarray(data["pose"][current])
        obj_now = _world_to_hand(data["obj"][current, safe], pose)
        obj_future = _world_to_hand(data["obj"][future, safe], pose)
        normals = _normal_world_to_hand(data["obj_normals"][current, safe], pose)
        hand_now = _world_to_hand(data["hand"][current], pose)
        hand_future = _world_to_hand(data["hand"][future], pose)
        obj_now[~valid] = obj_future[~valid] = normals[~valid] = 0
        return {"obj_points": torch.from_numpy(obj_now), "obj_normals": torch.from_numpy(normals),
                "obj_flow_gt": torch.from_numpy(obj_future - obj_now), "hand_points": torch.from_numpy(hand_now),
                "hand_normals": torch.from_numpy(_normal_world_to_hand(data["hand_normals"][current], pose)),
                "hand_flow": torch.from_numpy(hand_future - hand_now), "obj_valid_mask": torch.from_numpy(valid),
                "selected_obj_idx": torch.from_numpy(selected.astype(np.int64)), "raw_frame_id": torch.tensor(raw),
                "next_raw_frame_id": torch.tensor(int(data["raw"][future])), "stride": torch.tensor(stride),
                "delta_time_s": torch.tensor(float(stride) / 30.0), "dataset_id": self._dataset_id(sequence)}

    @staticmethod
    def _dataset_id(sequence: Path) -> str:
        try:
            return json.loads((sequence / "shared" / "meta.json").read_text(encoding="utf-8")).get("dataset_name", "unknown")
        except (OSError, json.JSONDecodeError):
            return "unknown"


class CmObjectV2Dataset(Dataset):
    """Read one or more existing Stage4 roots without exposing dataset identity to Cm."""

    def __init__(self, roots: Union[str, Path, Sequence[Union[str, Path]]], **kwargs) -> None:
        self.roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(root) for root in roots]
        self.datasets: List[Dataset] = []
        self._locations: List[Tuple[int, int]] = []
        for root in self.roots:
            files = sorted(root.glob("**/left.npz")) + sorted(root.glob("**/right.npz"))
            sequence_meta = sorted(root.glob("**/shared/meta.json"))
            root_meta = (root / "meta.json").read_text(encoding="utf-8") if (root / "meta.json").is_file() else ""
            if "ref2dex_cm_object_v2" in root_meta or sequence_meta:
                # Accept both a single object-v2 root and a combined root with
                # ``grab/`` and ``arctic/`` children.  The latter is the
                # canonical V1.2 layout used by the mixed config and E1 stats.
                sequences = [path.parent.parent for path in sequence_meta]
                if not sequences:
                    raise ValueError(f"No object-v2 sequence caches under {root}")
                dataset = _MmapSequenceDataset(sequences, **kwargs)
            else:
                if not files:
                    raise ValueError(f"No Stage4 hand streams under {root}")
                dataset = Stage4CmDataset(root, file_list=files, **kwargs)
            index = len(self.datasets)
            self.datasets.append(dataset)
            self._locations.extend((index, row) for row in range(len(dataset)))

    def __len__(self) -> int:
        return len(self._locations)

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            dataset.set_epoch(epoch)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        dataset_index, row = self._locations[index]
        sample = dict(self.datasets[dataset_index][row])
        if isinstance(self.datasets[dataset_index], _MmapSequenceDataset):
            return sample
        path, _ = self.datasets[dataset_index].sample_location(row)
        with np.load(path, allow_pickle=False) as hand:
            sample["dataset_id"] = self._dataset_name(path)
        return sample

    @staticmethod
    def _dataset_name(path: Path) -> str:
        meta = path.parent / "shared.npz"
        try:
            with np.load(meta, allow_pickle=False) as data:
                return str(np.asarray(data["dataset_name"]).item())
        except (OSError, KeyError, ValueError):
            return "unknown"


def dataset_statistics(dataset: Dataset) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        name = str(sample.get("dataset_id", "unknown"))
        row = stats.setdefault(name, {"samples": 0.0, "candidate_points": 0.0, "flow_sq_sum": 0.0, "flow_points": 0.0})
        valid = sample["obj_valid_mask"].bool()
        flow = sample["obj_flow_gt"]
        row["samples"] += 1.0
        row["candidate_points"] += float(valid.sum())
        row["flow_sq_sum"] += float((flow[valid] ** 2).sum())
        row["flow_points"] += float(valid.sum())
    for row in stats.values():
        row["flow_rms_m"] = float(np.sqrt(row["flow_sq_sum"] / max(row["flow_points"] * 3.0, 1.0)))
    return stats


def write_statistics(dataset: Dataset, path: Union[str, Path]) -> Dict:
    result = dataset_statistics(dataset)
    payload = {"schema_name": "ref2dex_cm_object_v2_statistics", "sampling_rule": "p_d proportional to sqrt(N_d)", "datasets": result}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _sequence_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    result = [path.parent.parent for path in sorted(root.glob("**/shared/meta.json"))]
    if not result:
        raise FileNotFoundError(f"No object-v2 sequence caches under {root}")
    return result


def _split_by_dataset(sequence_dirs: List[Path], *, seed: int, val_fraction: float, test_fraction: float) -> Tuple[List[Path], List[Path], List[Path]]:
    grouped: Dict[str, List[Path]] = {}
    for sequence in sequence_dirs:
        try:
            name = str(json.loads((sequence / "shared" / "meta.json").read_text(encoding="utf-8")).get("dataset_name", "unknown"))
        except (OSError, json.JSONDecodeError):
            name = "unknown"
        grouped.setdefault(name, []).append(sequence)
    train: List[Path] = []
    val: List[Path] = []
    test: List[Path] = []
    rng = np.random.default_rng(int(seed))
    for name, values in sorted(grouped.items()):
        values = list(values)
        rng.shuffle(values)
        n = len(values)
        n_test = max(1, int(round(n * test_fraction))) if n >= 3 and test_fraction > 0 else 0
        n_val = max(1, int(round(n * val_fraction))) if n - n_test >= 3 and val_fraction > 0 else 0
        test.extend(values[:n_test])
        val.extend(values[n_test:n_test + n_val])
        train.extend(values[n_test + n_val:])
    if not train:
        raise ValueError("Object-v2 sequence split produced an empty train set")
    return sorted(train), sorted(val), sorted(test)


def _read_sequence_split(split_path: Path, root: Path) -> List[Path]:
    entries = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    for entry in entries:
        path = (root / entry).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Object-v2 split entry escapes root: {entry}") from exc
        if not (path / "shared" / "meta.json").is_file():
            raise FileNotFoundError(f"Object-v2 split references missing sequence: {path}")
        result.append(path)
    if len(set(result)) != len(result):
        raise ValueError(f"Duplicate object-v2 sequence in split {split_path}")
    return result


def _resolve_splits(data_cfg, root: Path, sequence_dirs: List[Path], seed: int):
    descriptor = getattr(data_cfg, "split_json_path", None)
    if descriptor:
        descriptor_path = Path(descriptor)
        if not descriptor_path.is_absolute():
            descriptor_path = (Path.cwd() / descriptor_path).resolve()
        payload = json.loads(descriptor_path.read_text(encoding="utf-8"))
        def read(name: str, required: bool):
            value = payload.get(name)
            if not value:
                if required:
                    raise ValueError(f"Split descriptor missing {name}")
                return []
            path = Path(value)
            if not path.is_absolute():
                path = descriptor_path.parent / path
            return _read_sequence_split(path.resolve(), root)
        return read("train_split", True), read("val_split", False), read("test_split", False)
    return _split_by_dataset(sequence_dirs, seed=seed,
                             val_fraction=float(getattr(data_cfg, "val_split", 0.1) or 0.1),
                             test_fraction=float(getattr(data_cfg, "test_fraction", 0.1) or 0.1))


def make_dataloaders(data_cfg, seed: int, *, meta_cfg, distributed=None):
    """Build sequence-disjoint object-v2 train/val/test loaders.

    The loader keeps ``dataset_id`` in the batch for diagnostics; Cm's model
    and loss consume the same keys as the legacy Stage4 dataset.
    """
    root_value = str(getattr(data_cfg, "root", "") or getattr(data_cfg, "train_path", "")).strip()
    root = Path(root_value).resolve()
    train_dirs, val_dirs, test_dirs = _resolve_splits(data_cfg, root, _sequence_dirs(root), seed)
    common = dict(num_obj_points=int(meta_cfg.num_obj_points), num_hand_points=int(meta_cfg.num_hand_points),
                 base_seed=int(seed), min_stride=int(getattr(data_cfg, "min_stride", 1)),
                 max_stride=int(getattr(data_cfg, "max_stride", 10)), active_only=bool(getattr(data_cfg, "active_only", True)),
                 sampling_bank_size=int(getattr(data_cfg, "sampling_bank_size", 4)),
                 fixed_eval_bank=int(getattr(data_cfg, "fixed_eval_bank", 0)))
    train_dataset = _MmapSequenceDataset(train_dirs, **common)
    val_dataset = _MmapSequenceDataset(val_dirs, fixed_stride=int(getattr(data_cfg, "val_stride", 1)), **common) if val_dirs else None
    test_dataset = _MmapSequenceDataset(test_dirs, fixed_stride=int(getattr(data_cfg, "test_stride", 1)), **common) if test_dirs else None
    train_kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
    train_kwargs.update(batch_size=int(data_cfg.batch_size),
                        shuffle=distributed is None,
                        sampler=None if distributed is None else make_default_train_sampler(
                            train_dataset, shuffle=True, seed=seed, distributed=distributed, drop_last=False))
    train_loader = DataLoader(train_dataset, **train_kwargs)
    def eval_loader(dataset, batch_size):
        if dataset is None:
            return None
        kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
        kwargs.update(batch_size=int(batch_size or data_cfg.batch_size), shuffle=False,
                      sampler=None if distributed is None else make_default_eval_sampler(dataset, distributed=distributed))
        return DataLoader(dataset, **kwargs)
    val_loader = eval_loader(val_dataset, getattr(data_cfg, "val_batch_size", None))
    test_loader = eval_loader(test_dataset, getattr(data_cfg, "test_batch_size", None))
    val_loaders = {}
    test_loaders = {}
    for stride in tuple(getattr(data_cfg, "val_strides", (1, 5, 10))):
        if not val_dirs:
            break
        view = _MmapSequenceDataset(val_dirs, fixed_stride=int(stride), **common)
        val_loaders[f"val/stride_{int(stride)}/"] = eval_loader(view, getattr(data_cfg, "val_batch_size", None))
    metadata = {"schema_name": "ref2dex_cm_object_v2", "coordinate_frame": "hand_root_t",
                "num_obj_pool": 4096, "num_obj_points": int(meta_cfg.num_obj_points),
                "num_hand_points": int(meta_cfg.num_hand_points), "dataset_split": {
                "train_sequences": len(train_dirs), "val_sequences": len(val_dirs), "test_sequences": len(test_dirs)}}
    metadata_path = root / "metadata.json"
    if metadata_path.is_file():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            metadata.update(payload)
    return train_loader, val_loader, test_loader, metadata, val_loaders, test_loaders
