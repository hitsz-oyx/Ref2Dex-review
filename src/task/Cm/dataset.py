"""Stage 4 temporal-pair dataset and dense-token-compatible object sampling."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base import make_file_split_dataloaders
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed


STAGE4_SCHEMA_NAME = "ref2dex_cm_stage4"
REQUIRED_FIELDS = {
    "schema_name",
    "raw_frame_id",
    "next_raw_frame_id",
    "obj_points",
    "obj_normals",
    "obj_flow_gt",
    "hand_points",
    "hand_normals",
    "hand_flow",
    "wrist_delta",
    "hand_to_obj_min_dist",
    "obj_candidate_mask_5cm",
}


def scalar_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
    value = data.get(key)
    if value is None:
        return default
    value = np.asarray(value)
    return str(value.item()) if value.size == 1 else default


class Stage4CmDataset(Dataset):
    """One item is a consecutive Stage 4 pair with 512 checkpoint-compatible points."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        base_seed: int = 42,
        active_only: bool = False,
        min_object_flow_norm: float = 0.0,
        max_samples: int | None = None,
        coordinate_frame: str = "hand_root_t",
    ) -> None:
        self.data_path = Path(data_path)
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.base_seed = int(base_seed)
        self.active_only = bool(active_only)
        self.min_object_flow_norm = float(min_object_flow_norm)
        if self.min_object_flow_norm < 0.0:
            raise ValueError("min_object_flow_norm must be non-negative")
        self.coordinate_frame = str(coordinate_frame)
        self.file_paths = (
            sorted(Path(path) for path in file_list)
            if file_list is not None
            else (
                sorted(self.data_path.glob("**/*.npz"))
                if self.data_path.is_dir()
                else [self.data_path]
            )
        )
        self.file_paths = [path for path in self.file_paths if path.name != "meta.npz"]
        if not self.file_paths:
            raise ValueError(f"No Stage 4 NPZ files found in {self.data_path}")
        self._cached_path: Path | None = None
        self._cached_data: dict[str, np.ndarray] | None = None
        self._samples: list[tuple[Path, int]] = []

        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                missing = REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing Stage 4 fields {sorted(missing)}")
                schema = str(np.asarray(data["schema_name"]).item())
                if schema != STAGE4_SCHEMA_NAME:
                    raise ValueError(f"{path}: expected schema {STAGE4_SCHEMA_NAME!r}, got {schema!r}")
                frame = str(np.asarray(data["coordinate_frame"]).item())
                if frame != self.coordinate_frame:
                    raise ValueError(
                        f"{path}: expected coordinate_frame={self.coordinate_frame!r}, got {frame!r}"
                    )
                if data["obj_points"].shape[1:] != (4096, 3):
                    raise ValueError(f"{path}: expected obj_points shape [T, 4096, 3], got {data['obj_points'].shape}")
                if data["hand_points"].shape[1:] != (self.num_hand_points, 3):
                    raise ValueError(
                        f"{path}: expected hand_points shape [T, {self.num_hand_points}, 3], got {data['hand_points'].shape}"
                    )
                candidate_mask = np.asarray(data["obj_candidate_mask_5cm"], dtype=bool)
                pair_indices = np.arange(int(data["raw_frame_id"].shape[0]), dtype=np.int64)
                if self.active_only:
                    pair_indices = pair_indices[candidate_mask.any(axis=1)]
                if self.min_object_flow_norm > 0.0:
                    flow_norm = np.linalg.norm(np.asarray(data["obj_flow_gt"], dtype=np.float32), axis=-1)
                    candidate_count = candidate_mask.sum(axis=1)
                    candidate_flow_sum = (flow_norm * candidate_mask).sum(axis=1)
                    mean_candidate_flow = candidate_flow_sum / np.clip(candidate_count, 1, None)
                    pair_indices = pair_indices[
                        mean_candidate_flow[pair_indices] >= self.min_object_flow_norm
                    ]
                self._samples.extend((path, int(pair_idx)) for pair_idx in pair_indices.tolist())
        if max_samples is not None and int(max_samples) > 0:
            self._samples = self._samples[: int(max_samples)]
        if not self._samples:
            state = "after active-pair filtering" if self.active_only else ""
            raise ValueError(f"No Stage 4 temporal pairs found {state} in {self.data_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        if self._cached_path == path and self._cached_data is not None:
            return self._cached_data
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        self._cached_path = path
        self._cached_data = payload
        return payload

    def sample_location(self, index: int) -> tuple[Path, int]:
        return self._samples[index]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, pair_idx = self._samples[index]
        data = self._load_file(path)
        seq_id = scalar_string(data, "seq_id", path.stem)
        side = scalar_string(data, "side", "")
        raw_frame_id = int(np.asarray(data["raw_frame_id"])[pair_idx])
        seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=0,
        )
        selected_idx, obj_valid_mask = sample_object_indices(
            np.asarray(data["obj_candidate_mask_5cm"])[pair_idx],
            num_samples=self.num_obj_points,
            seed=seed,
        )
        safe_idx = np.maximum(selected_idx, 0)

        def selected(name: str) -> np.ndarray:
            value = np.asarray(data[name][pair_idx, safe_idx], dtype=np.float32).copy()
            value[~obj_valid_mask] = 0.0
            return value

        return {
            "obj_points": torch.from_numpy(selected("obj_points")),
            "obj_normals": torch.from_numpy(selected("obj_normals")),
            "obj_flow_gt": torch.from_numpy(selected("obj_flow_gt")),
            "hand_points": torch.from_numpy(np.asarray(data["hand_points"][pair_idx], dtype=np.float32).copy()),
            "hand_normals": torch.from_numpy(np.asarray(data["hand_normals"][pair_idx], dtype=np.float32).copy()),
            "hand_flow": torch.from_numpy(np.asarray(data["hand_flow"][pair_idx], dtype=np.float32).copy()),
            "wrist_delta": torch.from_numpy(np.asarray(data["wrist_delta"][pair_idx], dtype=np.float32).copy()),
            "hand_to_obj_min_dist": torch.from_numpy(
                np.asarray(data["hand_to_obj_min_dist"][pair_idx], dtype=np.float32).copy()
            ),
            "obj_valid_mask": torch.from_numpy(obj_valid_mask),
            "selected_obj_idx": torch.from_numpy(selected_idx.astype(np.int64)),
            "raw_frame_id": torch.tensor(raw_frame_id, dtype=torch.long),
            "next_raw_frame_id": torch.tensor(int(np.asarray(data["next_raw_frame_id"])[pair_idx]), dtype=torch.long),
            "pair_index": torch.tensor(pair_idx, dtype=torch.long),
        }


def _sequence_group_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_left") or stem.endswith("_right"):
        stem = stem.rsplit("_", 1)[0]
    parent = path.parent.as_posix()
    return stem if parent in {"", "."} else f"{parent}/{stem}"


def make_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    meta_cfg: Any,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any], dict[str, DataLoader]]:
    """Create BaseRunner-compatible Stage 4 loaders.

    Stage 4 itself retains every temporal pair.  ``active_only`` is a training
    sampler policy: a pair without a 5cm candidate has no object token for the
    frozen correspondence encoder, therefore it cannot contribute a point-flow
    loss in this first Cm bootstrap stage.
    """
    common = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "base_seed": int(seed),
        "active_only": bool(getattr(data_cfg, "active_only", True)),
        "min_object_flow_norm": float(getattr(data_cfg, "min_object_flow_norm", 0.0)),
        "coordinate_frame": str(meta_cfg.coordinate_frame),
    }
    train_kwargs = {
        **common,
        "max_samples": getattr(data_cfg, "max_train_samples", None),
    }
    val_kwargs = {
        **common,
        "max_samples": getattr(data_cfg, "max_val_samples", None),
    }
    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg,
        seed,
        dataset_cls=Stage4CmDataset,
        file_pattern="**/*.npz",
        train_dataset_kwargs=train_kwargs,
        val_dataset_kwargs=val_kwargs,
        split_group_fn=_sequence_group_key if bool(getattr(data_cfg, "group_val_by_sequence", True)) else None,
        distributed=distributed,
    )
    first_path = train_loader.dataset.file_paths[0]
    with np.load(first_path, allow_pickle=False) as data:
        metadata.update(
            {
                "schema_name": str(np.asarray(data["schema_name"]).item()),
                "coordinate_frame": str(np.asarray(data["coordinate_frame"]).item()),
                "num_obj_pool": int(data["obj_points"].shape[1]),
                "num_obj_points": int(meta_cfg.num_obj_points),
                "num_hand_points": int(data["hand_points"].shape[1]),
                "active_only": bool(common["active_only"]),
            }
        )
    val_loaders = {} if val_loader is None else {"val/": val_loader}
    metadata["val_loader_names"] = sorted(val_loaders)
    return train_loader, val_loader, metadata, val_loaders
