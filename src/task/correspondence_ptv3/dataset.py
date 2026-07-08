from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.base import make_file_split_dataloaders
from src.base.data import make_dataloader_kwargs
from src.base.distributed import shard_sampler_for_distributed
from src.task.correspondence_ptv3.sampling import (
    augment_geometry,
    sample_object_indices,
    stable_frame_seed,
)
from src.utils.correspondence import soft_contact_label


class CorrStaticDataset(Dataset):
    """Stage 3 point-pool dataset with deterministic per-epoch object sampling."""

    REQUIRED_FIELDS = {
        "raw_frame_id",
        "obj_points",
        "obj_normals",
        "obj_point_id",
        "hand_points",
        "hand_normals",
        "hand_point_id",
        "obj_to_hand_min_dist",
        "obj_candidate_mask_5cm",
        "gt_obj_to_hand_knn_idx",
    }

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        k_cross: int = 32,
        k_ctx: int = 32,
        k_logit: int = 64,
        k_logit_hard_neg: int = 16,
        ctx_radius: float = 0.04,
        logit_neg_radius: float = 0.06,
        logit_far_weight: float = 0.5,
        base_seed: int = 42,
        augment: bool = True,
        apply_hand_perturb: bool = True,
        augment_rotation: bool = True,
        augment_translation: bool = False,
        augment_scale: bool = False,
        rotation_range: float = 180.0,
        translation_range: float = 0.1,
        scale_range: tuple[float, float] = (0.9, 1.1),
        d_pos: float = 0.005,
        d_neg: float = 0.03,
        gamma: float = 2.0,
        hand_rot_std_deg: float = 10.0,
        hand_trans_std: float = 0.01,
        hand_perturb_prob: float = 1.0,
        blacklist_path: str | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.data_path = Path(data_path)
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.k_gt = int(k_cross)
        self.k_ctx = int(k_ctx)
        self.k_logit = int(k_logit)
        self.k_logit_hard_neg = int(k_logit_hard_neg)
        self.ctx_radius = float(ctx_radius)
        self.logit_neg_radius = float(logit_neg_radius)
        self.logit_far_weight = float(logit_far_weight)
        if self.k_ctx <= 0:
            raise ValueError("k_ctx must be positive.")
        if self.k_logit < self.k_ctx:
            raise ValueError("k_logit must be >= k_ctx.")
        if self.k_logit_hard_neg < 0:
            raise ValueError("k_logit_hard_neg must be non-negative.")
        if self.k_ctx + self.k_logit_hard_neg > self.k_logit:
            raise ValueError("k_ctx + k_logit_hard_neg must be <= k_logit.")
        if self.logit_neg_radius < self.ctx_radius:
            raise ValueError("logit_neg_radius must be >= ctx_radius.")
        if not 0.0 < self.logit_far_weight <= 1.0:
            raise ValueError("logit_far_weight must be in (0, 1].")
        self.base_seed = int(base_seed)
        self.augment = bool(augment)
        self.apply_hand_perturb = bool(apply_hand_perturb)
        self.augment_rotation = bool(augment_rotation)
        self.augment_translation = bool(augment_translation)
        self.augment_scale = bool(augment_scale)
        self.rotation_range = float(rotation_range)
        self.translation_range = float(translation_range)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.d_pos = float(d_pos)
        self.d_neg = float(d_neg)
        self.gamma = float(gamma)
        self.hand_rot_std_deg = float(hand_rot_std_deg)
        self.hand_trans_std = float(hand_trans_std)
        self.hand_perturb_prob = float(hand_perturb_prob)
        self._epoch = mp.Value("q", 0, lock=True)
        self._cached_path: Path | None = None
        self._cached_data: dict[str, np.ndarray] | None = None

        paths = (
            sorted(Path(path) for path in file_list)
            if file_list is not None
            else (
                sorted(self.data_path.glob("**/*.npz"))
                if self.data_path.is_dir()
                else [self.data_path]
            )
        )
        blacklist = _load_blacklist(blacklist_path)
        self.file_paths = [path for path in paths if not _is_blacklisted(path, self.data_root, blacklist)]
        if not self.file_paths:
            raise ValueError(f"No Stage 3 npz files found in {self.data_path}")

        self._samples: list[tuple[Path, int]] = []
        self.file_sample_ranges: list[tuple[int, int]] = []
        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                missing = self.REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing Stage 3 fields {sorted(missing)}")
                num_frames = int(data["raw_frame_id"].shape[0])
            start = len(self._samples)
            self._samples.extend((path, frame_idx) for frame_idx in range(num_frames))
            self.file_sample_ranges.append((start, len(self._samples)))

    @property
    def epoch(self) -> int:
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

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

    @staticmethod
    def _scalar_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
        value = data.get(key)
        if value is None:
            return default
        array = np.asarray(value)
        return str(array.item()) if array.size == 1 else default

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path, frame_idx = self._samples[index]
        data = self._load_file(path)
        seq_id = self._scalar_string(data, "seq_id", path.stem)
        side = self._scalar_string(data, "side", "")
        raw_frame_id = int(np.asarray(data["raw_frame_id"])[frame_idx])
        epoch = self.epoch
        sample_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=epoch,
        )
        selected_idx, obj_valid = sample_object_indices(
            data["obj_candidate_mask_5cm"][frame_idx],
            num_samples=self.num_obj_points,
            seed=sample_seed,
        )
        safe_idx = np.maximum(selected_idx, 0)

        obj_points = np.asarray(data["obj_points"][frame_idx, safe_idx], dtype=np.float32).copy()
        obj_normals = np.asarray(data["obj_normals"][frame_idx, safe_idx], dtype=np.float32).copy()
        obj_point_id = np.asarray(data["obj_point_id"][safe_idx], dtype=np.int64).copy()
        obj_min_dist = np.asarray(
            data["obj_to_hand_min_dist"][frame_idx, safe_idx],
            dtype=np.float32,
        ).copy()
        clean_knn_idx = np.asarray(
            data["gt_obj_to_hand_knn_idx"][frame_idx, safe_idx],
            dtype=np.int64,
        ).copy()

        obj_points[~obj_valid] = 0
        obj_normals[~obj_valid] = 0
        obj_point_id[~obj_valid] = -1
        obj_min_dist[~obj_valid] = 0
        clean_knn_idx[~obj_valid] = -1

        hand_points = np.asarray(data["hand_points"][frame_idx], dtype=np.float32)
        hand_normals = np.asarray(data["hand_normals"][frame_idx], dtype=np.float32)
        aug_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=epoch,
            namespace="augmentation",
        )
        geometry = augment_geometry(
            obj_points=obj_points,
            obj_normals=obj_normals,
            hand_points=hand_points,
            hand_normals=hand_normals,
            seed=aug_seed,
            apply_hand_perturb=self.apply_hand_perturb,
            hand_rot_std_deg=self.hand_rot_std_deg,
            hand_trans_std=self.hand_trans_std,
            hand_perturb_prob=self.hand_perturb_prob,
            apply_global_aug=self.augment,
            augment_rotation=self.augment_rotation,
            rotation_range_deg=self.rotation_range,
            augment_translation=self.augment_translation,
            translation_range=self.translation_range,
            augment_scale=self.augment_scale,
            scale_range=self.scale_range,
        )
        obj_min_dist *= float(geometry.distance_scale)

        (
            input_ctx_idx,
            input_ctx_valid,
            input_logit_idx,
            input_logit_valid,
            input_logit_weight,
        ) = _compute_runtime_hand_neighbors(
            geometry.input_obj_points,
            geometry.input_hand_points,
            obj_valid,
            k_ctx=self.k_ctx,
            k_logit=self.k_logit,
            k_logit_hard_neg=self.k_logit_hard_neg,
            ctx_radius=self.ctx_radius,
            logit_neg_radius=self.logit_neg_radius,
            far_weight=self.logit_far_weight,
            seed=stable_frame_seed(
                base_seed=self.base_seed,
                seq_id=seq_id,
                side=side,
                raw_frame_id=raw_frame_id,
                epoch=epoch,
                namespace="logit-neighbors",
            ),
        )
        input_points = np.concatenate(
            [geometry.input_obj_points, geometry.input_hand_points],
            axis=0,
        )
        input_normals = np.concatenate(
            [geometry.input_obj_normals, geometry.input_hand_normals],
            axis=0,
        )
        gt_points = np.concatenate(
            [geometry.gt_obj_points, geometry.gt_hand_points],
            axis=0,
        )
        gt_normals = np.concatenate(
            [geometry.gt_obj_normals, geometry.gt_hand_normals],
            axis=0,
        )
        point_valid_mask = np.concatenate(
            [obj_valid, np.ones((self.num_hand_points,), dtype=bool)],
            axis=0,
        )
        contact_label = soft_contact_label(
            torch.from_numpy(obj_min_dist),
            d_pos=self.d_pos,
            d_neg=self.d_neg,
            gamma=self.gamma,
        )

        return {
            "points": torch.from_numpy(input_points).float(),
            "normals": torch.from_numpy(input_normals).float(),
            "gt_points": torch.from_numpy(gt_points).float(),
            "gt_normals": torch.from_numpy(gt_normals).float(),
            "point_valid_mask": torch.from_numpy(point_valid_mask),
            "runtime_obj_valid_mask": torch.from_numpy(obj_valid),
            "selected_obj_idx": torch.from_numpy(selected_idx).long(),
            "selected_obj_point_id": torch.from_numpy(obj_point_id).long(),
            "selected_obj_min_dist": torch.from_numpy(obj_min_dist).float(),
            "obj_contact_label": contact_label.float(),
            "gt_obj_to_hand_knn_idx": torch.from_numpy(clean_knn_idx).long(),
            "input_obj_to_hand_ctx_idx": torch.from_numpy(input_ctx_idx).long(),
            "input_obj_to_hand_ctx_valid_mask": torch.from_numpy(input_ctx_valid),
            "input_obj_to_hand_logit_idx": torch.from_numpy(input_logit_idx).long(),
            "input_obj_to_hand_logit_valid_mask": torch.from_numpy(input_logit_valid),
            "input_obj_to_hand_logit_loss_weight": torch.from_numpy(input_logit_weight).float(),
            "hand_cano_points": torch.from_numpy(
                np.asarray(
                    data.get(
                        "hand_cano_points",
                        np.zeros((self.num_hand_points, 3), dtype=np.float32),
                    )
                )
            ).float(),
            "hand_finger_id": torch.from_numpy(
                np.asarray(
                    data.get(
                        "hand_finger_id",
                        np.full((self.num_hand_points,), -1, dtype=np.int64),
                    )
                )
            ).long(),
            "hand_region_id": torch.from_numpy(
                np.asarray(
                    data.get(
                        "hand_region_id",
                        np.full((self.num_hand_points,), -1, dtype=np.int64),
                    )
                )
            ).long(),
            "num_obj_points": torch.tensor(self.num_obj_points, dtype=torch.long),
            "num_hand_points": torch.tensor(self.num_hand_points, dtype=torch.long),
        }


def _compute_runtime_hand_neighbors(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_ctx: int,
    k_logit: int,
    k_logit_hard_neg: int,
    ctx_radius: float,
    logit_neg_radius: float,
    far_weight: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_obj = obj_points.shape[0]
    ctx_idx = np.full((num_obj, k_ctx), -1, dtype=np.int64)
    ctx_valid = np.zeros((num_obj, k_ctx), dtype=bool)
    logit_idx = np.full((num_obj, k_logit), -1, dtype=np.int64)
    logit_valid = np.zeros((num_obj, k_logit), dtype=bool)
    logit_weight = np.zeros((num_obj, k_logit), dtype=np.float32)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return ctx_idx, ctx_valid, logit_idx, logit_valid, logit_weight
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    distance = torch.cdist(obj, hand).numpy()
    rng = np.random.default_rng(seed)
    extra_slots = int(k_logit - k_ctx)
    hard_neg_slots = min(max(int(k_logit_hard_neg), 0), extra_slots)
    far_neg_slots = max(0, extra_slots - hard_neg_slots)
    for row, obj_idx in enumerate(valid_obj_idx.tolist()):
        dist_row = distance[row]

        ctx_candidates = np.flatnonzero(dist_row <= float(ctx_radius))
        if ctx_candidates.size > 0:
            order = np.argsort(dist_row[ctx_candidates], kind="stable")
            chosen_ctx = ctx_candidates[order[:k_ctx]]
            ctx_count = int(chosen_ctx.size)
            ctx_idx[obj_idx, :ctx_count] = chosen_ctx
            ctx_valid[obj_idx, :ctx_count] = True
            logit_idx[obj_idx, :ctx_count] = chosen_ctx
            logit_valid[obj_idx, :ctx_count] = True
            logit_weight[obj_idx, :ctx_count] = 1.0

        if extra_slots <= 0:
            continue
        hard_neg_candidates = np.flatnonzero(
            (dist_row > float(ctx_radius))
            & (dist_row <= float(logit_neg_radius))
        )
        start = int(k_ctx)
        if hard_neg_slots > 0 and hard_neg_candidates.size > 0:
            hard_neg_order = np.argsort(dist_row[hard_neg_candidates], kind="stable")
            chosen_hard_neg = hard_neg_candidates[hard_neg_order[:hard_neg_slots]]
            hard_neg_count = int(chosen_hard_neg.size)
            logit_idx[obj_idx, start : start + hard_neg_count] = chosen_hard_neg
            logit_valid[obj_idx, start : start + hard_neg_count] = True
            logit_weight[obj_idx, start : start + hard_neg_count] = 1.0
            start += hard_neg_count
        if far_neg_slots <= 0:
            continue
        far_neg_candidates = np.flatnonzero(dist_row > float(logit_neg_radius))
        if far_neg_candidates.size == 0:
            continue
        far_neg_count = min(int(far_neg_slots), int(far_neg_candidates.size))
        chosen_far_neg = rng.choice(
            far_neg_candidates,
            size=far_neg_count,
            replace=False,
        )
        logit_idx[obj_idx, start : start + far_neg_count] = chosen_far_neg
        logit_valid[obj_idx, start : start + far_neg_count] = True
        logit_weight[obj_idx, start : start + far_neg_count] = float(far_weight)
    return ctx_idx, ctx_valid, logit_idx, logit_valid, logit_weight


def _compute_input_knn(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_cross: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible helper for visualization scripts.

    This preserves the old name/signature and returns only the context
    neighborhood with a very large radius, i.e. classic top-k over all hand
    points for each valid object point.
    """
    num_obj = obj_points.shape[0]
    result = np.full((num_obj, k_cross), -1, dtype=np.int64)
    valid = np.zeros((num_obj, k_cross), dtype=bool)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return result, valid
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    topk = min(int(k_cross), int(hand.shape[0]))
    idx = torch.topk(torch.cdist(obj, hand), k=topk, dim=-1, largest=False).indices.numpy()
    result[valid_obj_idx, :topk] = idx
    valid[valid_obj_idx, :topk] = True
    return result, valid


def _load_blacklist(path: str | None) -> set[str]:
    if not path:
        return set()
    blacklist_path = Path(path)
    if not blacklist_path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")
    if blacklist_path.suffix.lower() == ".json":
        import json

        payload = json.loads(blacklist_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Blacklist JSON must contain a list")
        return {str(item) for item in payload}
    return {
        line.strip()
        for line in blacklist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _is_blacklisted(path: Path, root: Path, blacklist: set[str]) -> bool:
    if not blacklist:
        return False
    keys = {str(path), path.name, path.stem, path.as_posix()}
    try:
        keys.add(path.relative_to(root).as_posix())
    except ValueError:
        pass
    return bool(keys.intersection(blacklist))


class SequenceLocalitySampler(Sampler[int]):
    """Shuffle sequences and frames while keeping each sequence in one block.

    An npz member is extracted as a whole array. Fully random frame ordering
    would therefore reload a large sequence file for nearly every sample.
    Block-local shuffling preserves stochastic frame order but loads each
    sequence only once per worker and epoch.
    """

    def __init__(self, dataset: CorrStaticDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = np.random.default_rng(
            stable_frame_seed(
                base_seed=self.seed,
                seq_id="sequence-locality-sampler",
                side="",
                raw_frame_id=0,
                epoch=self.epoch,
                namespace="frame-order",
            )
        )
        file_order = rng.permutation(len(self.dataset.file_sample_ranges))
        for file_idx in file_order:
            start, end = self.dataset.file_sample_ranges[int(file_idx)]
            yield from rng.permutation(np.arange(start, end, dtype=np.int64)).tolist()

    def __len__(self) -> int:
        return len(self.dataset)


def make_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    meta_cfg: Any,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    train_kwargs = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "k_cross": int(meta_cfg.k_cross),
        "k_ctx": int(getattr(meta_cfg, "k_ctx", meta_cfg.k_cross)),
        "k_logit": int(getattr(meta_cfg, "k_logit", getattr(meta_cfg, "k_ctx", meta_cfg.k_cross))),
        "k_logit_hard_neg": int(getattr(meta_cfg, "k_logit_hard_neg", 16)),
        "ctx_radius": float(getattr(meta_cfg, "ctx_radius", 0.04)),
        "logit_neg_radius": float(getattr(meta_cfg, "logit_neg_radius", 0.06)),
        "logit_far_weight": float(getattr(meta_cfg, "loss_cross_edge_far_weight", 0.5)),
        "base_seed": int(seed),
        "augment": True,
        "apply_hand_perturb": True,
        "augment_rotation": bool(meta_cfg.augment_rotation),
        "augment_translation": bool(meta_cfg.augment_translation),
        "augment_scale": bool(meta_cfg.augment_scale),
        "rotation_range": float(meta_cfg.rotation_range),
        "translation_range": float(meta_cfg.translation_range),
        "scale_range": tuple(meta_cfg.scale_range),
        "d_pos": float(meta_cfg.d_pos),
        "d_neg": float(meta_cfg.d_neg),
        "gamma": float(meta_cfg.gamma),
        "hand_rot_std_deg": float(meta_cfg.hand_rot_std_deg),
        "hand_trans_std": float(meta_cfg.hand_trans_std),
        "hand_perturb_prob": float(meta_cfg.hand_perturb_prob),
        "blacklist_path": getattr(data_cfg, "blacklist_path", None),
    }
    val_kwargs = {
        **train_kwargs,
        "augment": False,
        "apply_hand_perturb": bool(getattr(meta_cfg, "val_augment", False)),
    }
    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg,
        seed,
        dataset_cls=CorrStaticDataset,
        file_pattern="**/*.npz",
        train_dataset_kwargs=train_kwargs,
        val_dataset_kwargs=val_kwargs,
        distributed=distributed,
    )
    if bool(data_cfg.shuffle) and bool(
        getattr(data_cfg, "sequence_locality_shuffle", True)
    ):
        train_sampler = SequenceLocalitySampler(train_loader.dataset, seed)
        train_sampler = shard_sampler_for_distributed(
            train_sampler,
            distributed=distributed,
            drop_last=bool(data_cfg.drop_last),
            pad=True,
        )
        loader_seed = int(seed) + int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(data_cfg.batch_size),
            shuffle=False,
            sampler=train_sampler,
            **make_dataloader_kwargs(data_cfg, loader_seed),
        )
    first_path = train_loader.dataset.file_paths[0]
    with np.load(first_path, allow_pickle=False) as data:
        hand_finger_id = np.asarray(
            data.get("hand_finger_id", np.full((int(data["hand_points"].shape[1]),), -1, dtype=np.int64))
        )
        hand_region_id = np.asarray(
            data.get("hand_region_id", np.full((int(data["hand_points"].shape[1]),), -1, dtype=np.int64))
        )
        metadata.update(
            {
                "num_obj_pool": int(data["obj_points"].shape[1]),
                "num_obj_points": int(meta_cfg.num_obj_points),
                "num_hand_points": int(data["hand_points"].shape[1]),
                "k_cross": int(data["gt_obj_to_hand_knn_idx"].shape[2]),
                "k_ctx": int(getattr(meta_cfg, "k_ctx", meta_cfg.k_cross)),
                "k_logit": int(getattr(meta_cfg, "k_logit", getattr(meta_cfg, "k_ctx", meta_cfg.k_cross))),
                "k_logit_hard_neg": int(getattr(meta_cfg, "k_logit_hard_neg", 16)),
                "num_fingers": int(np.max(hand_finger_id)) + 1 if hand_finger_id.size > 0 else 0,
                "num_regions": int(np.max(hand_region_id)) + 1 if hand_region_id.size > 0 else 0,
            }
        )
    if val_loader is not None:
        val_loader.dataset.set_epoch(0)
    return train_loader, val_loader, metadata
