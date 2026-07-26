"""Stage 5 dataset for current-frame object effect and hand-motion targets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.base import make_file_split_dataloaders
from src.task.correspondence_ptv3_v2.sampling import sample_object_indices, stable_frame_seed

STAGE5_SCHEMA = "ref2dex_cp_stage5"
REQUIRED_FIELDS = {
    "schema_name", "seq_id", "side", "raw_frame_id", "obj_points", "obj_normals",
    "obj_flow_gt", "obj_contact_gt", "hand_points", "hand_normals", "hand_flow", "wrist_delta",
    "hand_to_obj_min_dist",
    "obj_candidate_mask_5cm",
}


class Stage5CpDataset(Dataset):
    """One deterministic 512-object-point sample from a Stage 5 temporal pair."""

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        base_seed: int = 42,
        active_only: bool = False,
        selected_pair_index: int | None = None,
        max_samples: int | None = None,
        **_: Any,
    ) -> None:
        self.data_path = Path(data_path)
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.base_seed = int(base_seed)
        self.active_only = bool(active_only)
        self.selected_pair_index = None if selected_pair_index is None else int(selected_pair_index)
        self._cached_path: Path | None = None
        self._cached_data: dict[str, np.ndarray] | None = None
        self.file_paths = self._resolve_paths(file_list)
        self.samples: list[tuple[Path, int]] = []

        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                missing = REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing Stage 5 fields {sorted(missing)}")
                schema = str(np.asarray(data["schema_name"]).item())
                if schema != STAGE5_SCHEMA:
                    raise ValueError(f"{path}: expected {STAGE5_SCHEMA!r}, got {schema!r}")
                pair_indices = np.arange(len(data["raw_frame_id"]), dtype=np.int64)
                if self.active_only:
                    candidate = np.asarray(data["obj_candidate_mask_5cm"], dtype=bool)
                    pair_indices = pair_indices[candidate.any(axis=1)]
                if self.selected_pair_index is not None:
                    pair_indices = pair_indices[pair_indices == self.selected_pair_index]
                self.samples.extend((path, int(index)) for index in pair_indices)

        if max_samples is not None and int(max_samples) > 0:
            self.samples = self.samples[:int(max_samples)]
        if not self.samples:
            raise ValueError(f"No Stage 5 samples found in {data_path}")

    def _resolve_paths(self, file_list: list[str | Path] | None) -> list[Path]:
        if file_list is not None:
            return sorted(Path(path) for path in file_list)
        if self.data_path.is_dir():
            return sorted(self.data_path.glob("**/*.npz"))
        return [self.data_path]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        if path != self._cached_path:
            with np.load(path, allow_pickle=False) as data:
                self._cached_data = {key: np.asarray(data[key]) for key in data.files}
            self._cached_path = path
        assert self._cached_data is not None
        return self._cached_data

    @staticmethod
    def _scalar_string(data: dict[str, np.ndarray], key: str) -> str:
        return str(np.asarray(data[key]).item())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, pair_index = self.samples[index]
        data = self._load_file(path)
        raw_frame_id = int(data["raw_frame_id"][pair_index])
        selected_idx, valid_mask = sample_object_indices(
            data["obj_candidate_mask_5cm"][pair_index],
            num_samples=self.num_obj_points,
            seed=stable_frame_seed(
                base_seed=self.base_seed,
                seq_id=self._scalar_string(data, "seq_id"),
                side=self._scalar_string(data, "side"),
                raw_frame_id=raw_frame_id,
                epoch=0,
            ),
        )
        safe_idx = np.maximum(selected_idx, 0)

        def selected_object_field(key: str) -> torch.Tensor:
            value = np.asarray(data[key][pair_index, safe_idx], dtype=np.float32).copy()
            value[~valid_mask] = 0.0
            return torch.from_numpy(value)

        return {
            "obj_points": selected_object_field("obj_points"),
            "obj_normals": selected_object_field("obj_normals"),
            "obj_flow_gt": selected_object_field("obj_flow_gt"),
            "obj_contact_gt": selected_object_field("obj_contact_gt"),
            "obj_valid_mask": torch.from_numpy(valid_mask),
            "hand_points": torch.from_numpy(np.asarray(data["hand_points"][pair_index], dtype=np.float32).copy()),
            "hand_normals": torch.from_numpy(np.asarray(data["hand_normals"][pair_index], dtype=np.float32).copy()),
            "hand_flow": torch.from_numpy(np.asarray(data["hand_flow"][pair_index], dtype=np.float32).copy()),
            "wrist_delta": torch.from_numpy(np.asarray(data["wrist_delta"][pair_index], dtype=np.float32).copy()),
            "hand_to_obj_min_dist": torch.from_numpy(
                np.asarray(data["hand_to_obj_min_dist"][pair_index], dtype=np.float32).copy()
            ),
            "raw_frame_id": torch.tensor(raw_frame_id, dtype=torch.long),
            "pair_index": torch.tensor(pair_index, dtype=torch.long),
        }


def _sequence_group(path: Path) -> str:
    return f"{path.parent.as_posix()}/{path.stem.rsplit('_', 1)[0]}"


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any | None = None):
    common = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "base_seed": int(seed),
        "active_only": bool(getattr(data_cfg, "active_only", False)),
        "selected_pair_index": getattr(data_cfg, "selected_pair_index", None),
    }
    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg,
        seed,
        dataset_cls=Stage5CpDataset,
        file_pattern="**/*.npz",
        train_dataset_kwargs={**common, "max_samples": getattr(data_cfg, "max_train_samples", None)},
        val_dataset_kwargs={**common, "max_samples": getattr(data_cfg, "max_val_samples", None)},
        split_group_fn=_sequence_group if bool(getattr(data_cfg, "group_val_by_sequence", True)) else None,
        distributed=distributed,
    )
    metadata.update({"num_obj_points": int(meta_cfg.num_obj_points), "num_hand_points": int(meta_cfg.num_hand_points), "coordinate_frame": "hand_root_t"})
    return train_loader, val_loader, metadata, ({} if val_loader is None else {"val/": val_loader})
