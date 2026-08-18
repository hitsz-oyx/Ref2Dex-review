"""Unified object-only Cm dataset for GRAB and ARCTIC Stage4 caches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset

from src.task.Cm.dataset import Stage4CmDataset, _normal_world_to_hand, _world_to_hand
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed


class _MmapSequenceDataset(Dataset):
    def __init__(self, sequence_dirs: list[Path], *, num_obj_points: int = 512, num_hand_points: int = 1538,
                 base_seed: int = 42, min_stride: int = 1, max_stride: int = 10, fixed_stride: int | None = None,
                 active_only: bool = True):
        self.num_obj_points, self.num_hand_points = num_obj_points, num_hand_points
        self.base_seed, self.min_stride, self.max_stride = base_seed, min_stride, max_stride
        self.fixed_stride, self.active_only, self.epoch = fixed_stride, active_only, 0
        self.rows: list[tuple[Path, str, int]] = []
        self._cache: dict[Path, dict[str, np.ndarray]] = {}
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
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, sequence: Path, side: str) -> dict[str, np.ndarray]:
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
            }
        return self._cache[key]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sequence, side, current = self.rows[index]
        data = self._load(sequence, side)
        raw = int(data["raw"][current])
        seed = stable_frame_seed(base_seed=self.base_seed, seq_id=str(sequence), side=side, raw_frame_id=raw, epoch=self.epoch, namespace="cm-object-v2")
        stride = self.fixed_stride or int(np.random.default_rng(seed).integers(self.min_stride, self.max_stride + 1))
        future = current + stride
        candidate = np.asarray(data["indices"][data["offsets"][current]:data["offsets"][current + 1]], dtype=np.uint32)
        candidate_mask = np.zeros(data["obj"].shape[1], dtype=bool)
        candidate_mask[candidate] = True
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

    def __init__(self, roots: str | Path | Sequence[str | Path], **kwargs) -> None:
        self.roots = [Path(roots)] if isinstance(roots, (str, Path)) else [Path(root) for root in roots]
        self.datasets: list[Stage4CmDataset] = []
        self._locations: list[tuple[int, int]] = []
        for root in self.roots:
            files = sorted(root.glob("**/left.npz")) + sorted(root.glob("**/right.npz"))
            if (root / "meta.json").is_file() and "ref2dex_cm_object_v2" in (root / "meta.json").read_text(encoding="utf-8"):
                sequences = [path.parent.parent for path in sorted(root.glob("**/shared/meta.json"))]
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

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
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


def dataset_statistics(dataset: Dataset) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
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


def write_statistics(dataset: Dataset, path: str | Path) -> dict:
    result = dataset_statistics(dataset)
    payload = {"schema_name": "ref2dex_cm_object_v2_statistics", "sampling_rule": "p_d proportional to sqrt(N_d)", "datasets": result}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
