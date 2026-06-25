from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.base import as_tensor, make_file_split_dataloaders
from src.utils.correspondence import soft_contact_label


class CorrStaticDataset(Dataset):
    """Stage 3 static correspondence dataset with noisy-hand / clean-GT separation."""

    FIELD_ALIASES: dict[str, tuple[str, ...]] = {
        "points": ("points",),
        "normals": ("normals",),
        "point_type_id": ("point_type_id",),
        "point_valid_mask": ("point_valid_mask",),
        "finger_id": ("finger_id",),
        "hand_region_id": ("hand_region_id",),
        "hand_cano_points": ("hand_cano_points",),
        "obj_contact_label": ("obj_contact_label",),
        "obj_to_hand_nn_id": ("obj_to_hand_nn_id",),
        "obj_to_hand_cano_points": ("obj_to_hand_cano_points",),
        "obj_to_hand_finger_id": ("obj_to_hand_finger_id",),
        "obj_to_hand_region_id": ("obj_to_hand_region_id",),
        "obj_to_hand_knn_idx": ("obj_to_hand_knn_idx",),
        "obj_to_hand_knn_valid_mask": ("obj_to_hand_knn_valid_mask",),
        "obj_label_valid_mask": ("obj_label_valid_mask",),
        "obj_corr_valid_mask": ("obj_corr_valid_mask",),
        "obj_local_knn_idx": ("obj_local_knn_idx",),
        "obj_local_knn_valid_mask": ("obj_local_knn_valid_mask",),
        "hand_local_knn_idx": ("hand_local_knn_idx",),
        "hand_local_knn_valid_mask": ("hand_local_knn_valid_mask",),
        "hand_to_obj_knn_idx": ("hand_to_obj_knn_idx",),
        "hand_to_obj_knn_valid_mask": ("hand_to_obj_knn_valid_mask",),
        "num_obj_points": ("num_obj_points", "obj_point_count"),
        "num_hand_points": ("num_hand_points", "hand_point_count"),
    }

    PER_SEQUENCE_FIELDS: set[str] = {
        "point_type_id",
        "finger_id",
        "hand_region_id",
        "hand_cano_points",
    }

    INDEX_FIELDS: set[str] = {
        "point_type_id",
        "finger_id",
        "hand_region_id",
        "obj_to_hand_nn_id",
        "obj_to_hand_finger_id",
        "obj_to_hand_region_id",
        "obj_to_hand_knn_idx",
        "obj_local_knn_idx",
        "hand_local_knn_idx",
        "hand_to_obj_knn_idx",
        "num_obj_points",
        "num_hand_points",
    }

    BOOL_FIELDS: set[str] = {
        "point_valid_mask",
        "obj_label_valid_mask",
        "obj_corr_valid_mask",
        "obj_to_hand_knn_valid_mask",
        "obj_local_knn_valid_mask",
        "hand_local_knn_valid_mask",
        "hand_to_obj_knn_valid_mask",
    }

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[Path] | None = None,
        augment: bool = True,
        apply_hand_perturb: bool = True,
        deterministic_hand_perturb: bool = False,
        perturb_seed: int = 0,
        augment_rotation: bool = True,
        augment_translation: bool = True,
        augment_scale: bool = False,
        rotation_range: float = 180.0,
        translation_range: float = 0.1,
        scale_range: tuple[float, float] = (1.0, 1.0),
        d_pos: float = 0.005,
        d_neg: float = 0.03,
        gamma: float = 2.0,
        k_cross: int = 32,
        corr_contact_label_min: float = 0.0,
        hand_rot_std_deg: float = 10.0,
        hand_trans_std: float = 0.01,
        hand_perturb_prob: float = 1.0,
        recompute_input_cross_knn: bool = True,
        blacklist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.augment = bool(augment)
        self.apply_hand_perturb = bool(apply_hand_perturb)
        self.deterministic_hand_perturb = bool(deterministic_hand_perturb)
        self.perturb_seed = int(perturb_seed)
        self.augment_rotation = bool(augment_rotation)
        self.augment_translation = bool(augment_translation)
        self.augment_scale = bool(augment_scale)
        self.rotation_range = float(rotation_range)
        self.translation_range = float(translation_range)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.d_pos = float(d_pos)
        self.d_neg = float(d_neg)
        self.gamma = float(gamma)
        self.k_cross = int(k_cross)
        self.corr_contact_label_min = float(corr_contact_label_min)
        self.hand_rot_std_deg = float(hand_rot_std_deg)
        self.hand_trans_std = float(hand_trans_std)
        self.hand_perturb_prob = float(hand_perturb_prob)
        self.recompute_input_cross_knn = bool(recompute_input_cross_knn)

        self.data_path = Path(data_path)
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.blacklist = _load_blacklist(blacklist_path)

        if file_list is not None:
            file_paths = sorted(Path(path) for path in file_list)
        elif self.data_path.is_dir():
            file_paths = sorted(self.data_path.glob("**/*.npz"))
        else:
            file_paths = [self.data_path]

        self.file_paths = [path for path in file_paths if not self._is_blacklisted(path)]
        if not self.file_paths:
            raise ValueError(f"No Stage 3 .npz files found in {self.data_path}")

        self._samples: list[tuple[Path, int]] = []
        self._cached_file_path: Path | None = None
        self._cached_file_data: dict[str, np.ndarray] | None = None
        self._build_index()

    def _build_index(self) -> None:
        for file_path in self.file_paths:
            with np.load(file_path, allow_pickle=False) as data:
                num_frames = self._infer_num_frames(data)
            self._samples.extend((file_path, frame_idx) for frame_idx in range(num_frames))

    def _infer_num_frames(self, data: Any) -> int:
        for key in ("points", "obj_contact_label", "raw_frame_id"):
            if key in data:
                return int(np.asarray(data[key]).shape[0])
        raise ValueError("Cannot infer number of frames from data.")

    def _is_blacklisted(self, file_path: Path) -> bool:
        if not self.blacklist:
            return False
        keys = {file_path.name, file_path.stem, str(file_path)}
        try:
            keys.add(str(file_path.relative_to(self.data_root)))
        except ValueError:
            pass
        return any(key in self.blacklist for key in keys)

    def _get_file_data(self, file_path: Path) -> dict[str, np.ndarray]:
        if self._cached_file_path == file_path and self._cached_file_data is not None:
            return self._cached_file_data

        file_data: dict[str, np.ndarray] = {}
        with np.load(file_path, allow_pickle=False) as data:
            for key, aliases in self.FIELD_ALIASES.items():
                array = self._select_array(data, aliases)
                if array is not None:
                    file_data[key] = np.asarray(array)
        self._cached_file_path = file_path
        self._cached_file_data = file_data
        return file_data

    def _select_array(self, data: Any, aliases: tuple[str, ...]) -> np.ndarray | None:
        for alias in aliases:
            if alias in data:
                return data[alias]
        return None

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        file_path, frame_idx = self._samples[index]
        file_data = self._get_file_data(file_path)
        sample: dict[str, torch.Tensor] = {}

        for key, value in file_data.items():
            tensor: torch.Tensor
            if key in self.PER_SEQUENCE_FIELDS or value.ndim == 0:
                tensor = as_tensor(value)
            else:
                tensor = as_tensor(value[frame_idx])
            sample[key] = tensor.clone()

        sample = self._cast_types(sample)
        sample = self._ensure_gt_geometry(sample)

        if self.apply_hand_perturb:
            self._maybe_perturb_input_hand(sample, file_path=file_path, frame_idx=frame_idx)
        if self.augment:
            self._apply_shared_global_aug(sample)

        sample = self._recompute_clean_obj_contact_label(sample)
        if self.recompute_input_cross_knn:
            sample = self._recompute_input_cross_knn(sample)
        sample = self._refresh_corr_valid_mask(sample)
        return sample

    def _cast_types(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        for key in self.INDEX_FIELDS:
            if key in sample:
                sample[key] = sample[key].long()
        for key in self.BOOL_FIELDS:
            if key in sample:
                sample[key] = sample[key].bool()
        return sample

    def _ensure_gt_geometry(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        sample["gt_points"] = sample["points"].clone()
        sample["gt_normals"] = sample["normals"].clone()
        return sample

    def _refresh_corr_valid_mask(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "obj_label_valid_mask" not in sample or "obj_contact_label" not in sample:
            return sample

        corr_valid = sample["obj_label_valid_mask"].bool()
        if "obj_to_hand_nn_id" in sample:
            corr_valid = corr_valid & (sample["obj_to_hand_nn_id"] >= 0)
        corr_valid = corr_valid & (sample["obj_contact_label"] > self.corr_contact_label_min)
        sample["obj_corr_valid_mask"] = corr_valid
        return sample

    def _maybe_perturb_input_hand(
        self,
        sample: dict[str, torch.Tensor],
        *,
        file_path: Path,
        frame_idx: int,
    ) -> bool:
        if self.hand_perturb_prob <= 0:
            return False
        if self.hand_rot_std_deg <= 0 and self.hand_trans_std <= 0:
            return False
        generator = self._make_hand_perturb_generator(file_path=file_path, frame_idx=frame_idx)
        if torch.rand((), device=sample["points"].device, generator=generator).item() > self.hand_perturb_prob:
            return False

        num_obj_points, num_hand_points = self._infer_point_counts(sample)
        if num_hand_points <= 0:
            return False

        hand_slice = slice(num_obj_points, num_obj_points + num_hand_points)
        points = sample["points"]
        normals = sample["normals"]
        hand_valid_mask = self._hand_valid_mask(sample, num_obj_points, num_hand_points)
        if not hand_valid_mask.any():
            return False

        rot = _sample_small_rotation_matrix(
            device=points.device,
            std_deg=self.hand_rot_std_deg,
            dtype=points.dtype,
            generator=generator,
        )
        trans = torch.randn(3, device=points.device, dtype=points.dtype, generator=generator) * self.hand_trans_std

        hand_points = points[hand_slice].clone()
        hand_normals = normals[hand_slice].clone()
        center = hand_points[hand_valid_mask].mean(dim=0, keepdim=True)

        hand_points[hand_valid_mask] = (hand_points[hand_valid_mask] - center) @ rot.T + center + trans
        hand_normals[hand_valid_mask] = hand_normals[hand_valid_mask] @ rot.T

        points[hand_slice] = hand_points
        normals[hand_slice] = hand_normals
        sample["points"] = points
        sample["normals"] = normals
        return True

    def _make_hand_perturb_generator(self, *, file_path: Path, frame_idx: int) -> torch.Generator | None:
        if not self.deterministic_hand_perturb:
            return None

        try:
            stable_path = str(file_path.relative_to(self.data_root))
        except ValueError:
            stable_path = str(file_path)
        payload = f"{self.perturb_seed}|{stable_path}|{frame_idx}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _apply_shared_global_aug(self, sample: dict[str, torch.Tensor]) -> None:
        points = sample["points"]
        normals = sample["normals"]
        gt_points = sample["gt_points"]
        gt_normals = sample["gt_normals"]

        if self.augment_rotation and self.rotation_range > 0:
            rot = _random_rotation_matrix(points.device, self.rotation_range, dtype=points.dtype)
            points = points @ rot.T
            normals = normals @ rot.T
            gt_points = gt_points @ rot.T
            gt_normals = gt_normals @ rot.T

        if self.augment_translation and self.translation_range > 0:
            trans = (torch.rand(3, device=points.device, dtype=points.dtype) * 2.0 - 1.0) * self.translation_range
            points = points + trans
            gt_points = gt_points + trans

        if self.augment_scale and (self.scale_range[0] != 1.0 or self.scale_range[1] != 1.0):
            scale = torch.empty(1, device=points.device, dtype=points.dtype).uniform_(*self.scale_range)
            points = points * scale
            gt_points = gt_points * scale

        sample["points"] = points
        sample["normals"] = normals
        sample["gt_points"] = gt_points
        sample["gt_normals"] = gt_normals

    def _recompute_clean_obj_contact_label(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required_keys = {"gt_points", "obj_to_hand_nn_id", "obj_label_valid_mask"}
        if not required_keys.issubset(sample):
            return sample

        num_obj_points, num_hand_points = self._infer_point_counts(sample)
        if num_hand_points <= 0:
            sample["obj_contact_label"] = torch.zeros(num_obj_points, dtype=torch.float32)
            return sample

        obj_points = sample["gt_points"][:num_obj_points]
        hand_points = sample["gt_points"][num_obj_points : num_obj_points + num_hand_points]
        obj_nn = sample["obj_to_hand_nn_id"].long()
        valid_mask = sample["obj_label_valid_mask"].bool() & (obj_nn >= 0)

        safe_nn = obj_nn.clamp(min=0, max=max(num_hand_points - 1, 0))
        nn_hand_points = hand_points[safe_nn]
        dist = torch.norm(nn_hand_points - obj_points, dim=-1)
        labels = soft_contact_label(
            dist,
            d_pos=self.d_pos,
            d_neg=self.d_neg,
            gamma=self.gamma,
        )
        sample["obj_contact_label"] = torch.where(valid_mask, labels, torch.zeros_like(labels))
        return sample

    def _recompute_input_cross_knn(self, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        num_obj_points, num_hand_points = self._infer_point_counts(sample)
        if num_obj_points <= 0:
            sample["obj_to_hand_knn_idx"] = torch.full((0, self.k_cross), -1, dtype=torch.long)
            sample["obj_to_hand_knn_valid_mask"] = torch.zeros((0, self.k_cross), dtype=torch.bool)
            return sample

        k_cross = self.k_cross
        obj_points = sample["points"][:num_obj_points]
        hand_points = sample["points"][num_obj_points : num_obj_points + num_hand_points]
        obj_valid_mask = self._obj_valid_mask(sample, num_obj_points)
        hand_valid_mask = self._hand_valid_mask(sample, num_obj_points, num_hand_points)

        knn_idx = torch.full((num_obj_points, k_cross), -1, dtype=torch.long)
        knn_valid_mask = torch.zeros((num_obj_points, k_cross), dtype=torch.bool)

        valid_obj_idx = torch.nonzero(obj_valid_mask, as_tuple=False).squeeze(-1)
        valid_hand_idx = torch.nonzero(hand_valid_mask, as_tuple=False).squeeze(-1)
        if valid_obj_idx.numel() == 0 or valid_hand_idx.numel() == 0:
            sample["obj_to_hand_knn_idx"] = knn_idx
            sample["obj_to_hand_knn_valid_mask"] = knn_valid_mask
            return sample

        topk = min(k_cross, int(valid_hand_idx.numel()))
        obj_points_valid = obj_points[valid_obj_idx].float()
        hand_points_valid = hand_points[valid_hand_idx].float()
        dist = torch.cdist(obj_points_valid, hand_points_valid)
        neighbor_rank = torch.topk(dist, k=topk, dim=-1, largest=False).indices
        mapped_idx = valid_hand_idx[neighbor_rank]

        knn_idx[valid_obj_idx, :topk] = mapped_idx.long()
        knn_valid_mask[valid_obj_idx, :topk] = True
        sample["obj_to_hand_knn_idx"] = knn_idx
        sample["obj_to_hand_knn_valid_mask"] = knn_valid_mask
        return sample

    def _infer_point_counts(self, sample: dict[str, torch.Tensor]) -> tuple[int, int]:
        num_obj_points = self._scalar_or_none(sample.get("num_obj_points"))
        if num_obj_points is None and "obj_contact_label" in sample:
            num_obj_points = int(sample["obj_contact_label"].shape[0])
        if num_obj_points is None and "obj_label_valid_mask" in sample:
            num_obj_points = int(sample["obj_label_valid_mask"].shape[0])
        if num_obj_points is None:
            raise ValueError("Cannot infer num_obj_points from sample.")

        num_hand_points = self._scalar_or_none(sample.get("num_hand_points"))
        if num_hand_points is None and "obj_to_hand_knn_idx" in sample:
            num_hand_points = int(sample["points"].shape[0]) - int(num_obj_points)
        if num_hand_points is None and "hand_to_obj_knn_idx" in sample:
            num_hand_points = int(sample["hand_to_obj_knn_idx"].shape[0])
        if num_hand_points is None and "points" in sample:
            num_hand_points = int(sample["points"].shape[0]) - int(num_obj_points)
        if num_hand_points is None:
            raise ValueError("Cannot infer num_hand_points from sample.")

        return int(num_obj_points), int(num_hand_points)

    def _obj_valid_mask(self, sample: dict[str, torch.Tensor], num_obj_points: int) -> torch.Tensor:
        if "obj_label_valid_mask" in sample:
            return sample["obj_label_valid_mask"].bool()
        if "point_valid_mask" in sample:
            return sample["point_valid_mask"][:num_obj_points].bool()
        return torch.ones(num_obj_points, dtype=torch.bool)

    def _hand_valid_mask(self, sample: dict[str, torch.Tensor], num_obj_points: int, num_hand_points: int) -> torch.Tensor:
        if "point_valid_mask" in sample:
            mask = sample["point_valid_mask"][num_obj_points : num_obj_points + num_hand_points].bool()
            if mask.numel() == num_hand_points:
                return mask
        return torch.ones(num_hand_points, dtype=torch.bool)

    def _scalar_or_none(self, value: torch.Tensor | None) -> int | None:
        if value is None:
            return None
        if not torch.is_tensor(value):
            return int(value)
        if value.numel() != 1:
            return None
        return int(value.item())


def _load_blacklist(path: str | None) -> set[str]:
    if path in {None, ""}:
        return set()

    blacklist_path = Path(path)
    if not blacklist_path.exists():
        raise FileNotFoundError(blacklist_path)

    if blacklist_path.suffix.lower() == ".json":
        payload = json.loads(blacklist_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("items", [])
        if not isinstance(payload, list):
            raise TypeError(f"Expected blacklist JSON to contain a list, got {type(payload).__name__}")
        return {str(item) for item in payload}

    entries: set[str] = set()
    for line in blacklist_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item and not item.startswith("#"):
            entries.add(item)
    return entries


def _sample_small_rotation_matrix(
    device: torch.device,
    std_deg: float,
    *,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if std_deg <= 0:
        return torch.eye(3, device=device, dtype=dtype)

    axis = torch.randn(3, device=device, dtype=dtype, generator=generator)
    axis = axis / axis.norm().clamp(min=1e-8)
    angle = torch.randn(1, device=device, dtype=dtype, generator=generator) * (std_deg * (3.1415926535 / 180.0))
    return _axis_angle_to_matrix(axis, angle, dtype=dtype)


def _random_rotation_matrix(device: torch.device, max_angle_deg: float, *, dtype: torch.dtype) -> torch.Tensor:
    max_angle_rad = max_angle_deg * (3.1415926535 / 180.0)
    axis = torch.randn(3, device=device, dtype=dtype)
    axis = axis / axis.norm().clamp(min=1e-8)
    angle = torch.rand(1, device=device, dtype=dtype) * max_angle_rad
    return _axis_angle_to_matrix(axis, angle, dtype=dtype)


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    K = torch.zeros((3, 3), device=axis.device, dtype=dtype)
    K[0, 1] = -axis[2]
    K[0, 2] = axis[1]
    K[1, 0] = axis[2]
    K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]
    K[2, 1] = axis[0]

    I = torch.eye(3, device=axis.device, dtype=dtype)
    return I + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


def make_dataloaders(
    data_cfg: Any,
    *,
    meta_cfg: Any,
    seed: int = 1000,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    """Create dataloaders for PTv3 static correspondence samples."""
    train_path = str(getattr(data_cfg, "train_path", "")).strip()
    if not train_path:
        raise ValueError("data.train_path must point to a Stage 3 train_corr_static directory or file.")

    common_dataset_kwargs = {
        "augment_rotation": bool(getattr(meta_cfg, "augment_rotation", True)),
        "augment_translation": bool(getattr(meta_cfg, "augment_translation", True)),
        "augment_scale": bool(getattr(meta_cfg, "augment_scale", False)),
        "rotation_range": float(getattr(meta_cfg, "rotation_range", 180.0)),
        "translation_range": float(getattr(meta_cfg, "translation_range", 0.1)),
        "scale_range": tuple(getattr(meta_cfg, "scale_range", (1.0, 1.0))),
        "d_pos": float(getattr(meta_cfg, "d_pos", 0.005)),
        "d_neg": float(getattr(meta_cfg, "d_neg", 0.03)),
        "gamma": float(getattr(meta_cfg, "gamma", 2.0)),
        "k_cross": int(getattr(meta_cfg, "k_cross", 32)),
        "corr_contact_label_min": float(getattr(meta_cfg, "corr_contact_label_min", 0.0)),
        "hand_rot_std_deg": float(getattr(meta_cfg, "hand_rot_std_deg", 10.0)),
        "hand_trans_std": float(getattr(meta_cfg, "hand_trans_std", 0.01)),
        "hand_perturb_prob": float(getattr(meta_cfg, "hand_perturb_prob", 1.0)),
        "recompute_input_cross_knn": bool(getattr(meta_cfg, "recompute_input_cross_knn", True)),
        "blacklist_path": getattr(data_cfg, "blacklist_path", None),
        "perturb_seed": int(seed),
    }

    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg=data_cfg,
        seed=seed,
        dataset_cls=CorrStaticDataset,
        root=None,
        file_pattern="**/*.npz",
        train_dataset_kwargs={
            **common_dataset_kwargs,
            "augment": True,
            "apply_hand_perturb": True,
            "deterministic_hand_perturb": False,
        },
        val_dataset_kwargs={
            **common_dataset_kwargs,
            "augment": bool(getattr(meta_cfg, "val_augment", True)),
            "apply_hand_perturb": True,
            "deterministic_hand_perturb": True,
        },
    )

    if len(train_loader.dataset) > 0:
        probe_dataset = CorrStaticDataset(
            train_loader.dataset.data_path,
            file_list=list(train_loader.dataset.file_paths),
            augment=False,
            apply_hand_perturb=False,
            **common_dataset_kwargs,
        )
        first_sample = probe_dataset[0]
        num_obj_points, num_hand_points = probe_dataset._infer_point_counts(first_sample)
        metadata["num_obj_points"] = num_obj_points
        metadata["num_hand_points"] = num_hand_points
        if "obj_local_knn_idx" in first_sample:
            metadata["k_obj_local"] = int(first_sample["obj_local_knn_idx"].shape[1])
        if "hand_local_knn_idx" in first_sample:
            metadata["k_hand_local"] = int(first_sample["hand_local_knn_idx"].shape[1])
        metadata["k_cross"] = int(first_sample["obj_to_hand_knn_idx"].shape[1])

        finger_ids = first_sample["finger_id"][first_sample["finger_id"] >= 0]
        region_ids = first_sample["hand_region_id"][first_sample["hand_region_id"] >= 0]
        metadata["num_fingers"] = int(finger_ids.max().item() + 1) if finger_ids.numel() > 0 else 0
        metadata["num_regions"] = int(region_ids.max().item() + 1) if region_ids.numel() > 0 else 0

    return train_loader, val_loader, metadata


__all__ = ["CorrStaticDataset", "make_dataloaders"]
