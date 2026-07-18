from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .distributed import make_default_eval_sampler, make_default_train_sampler


T = TypeVar("T")


def split_items(items: list[Any], val_split: float, seed: int) -> tuple[list[Any], list[Any]]:
    if val_split <= 0:
        return list(items), []
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if not items:
        return [], []

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(items))
    val_size = max(1, int(round(len(items) * val_split)))
    val_idx = set(perm[:val_size].tolist())
    train_items = [item for idx, item in enumerate(items) if idx not in val_idx]
    val_items = [item for idx, item in enumerate(items) if idx in val_idx]
    return train_items, val_items


def split_grouped_items(
    items: Sequence[T],
    *,
    val_split: float,
    seed: int,
    group_fn: Callable[[T], str],
) -> tuple[list[T], list[T], dict[str, Any]]:
    if val_split <= 0:
        group_keys = sorted({str(group_fn(item)) for item in items})
        return list(items), [], {
            "num_groups": len(group_keys),
            "num_train_groups": len(group_keys),
            "num_val_groups": 0,
        }
    if not 0 < val_split < 1:
        raise ValueError(f"val_split must be in [0, 1), got {val_split}")
    if not items:
        return [], [], {
            "num_groups": 0,
            "num_train_groups": 0,
            "num_val_groups": 0,
        }

    groups: dict[str, list[T]] = {}
    for item in items:
        key = str(group_fn(item))
        groups.setdefault(key, []).append(item)
    group_keys = sorted(groups)
    if len(group_keys) < 2:
        raise ValueError(
            f"Need at least two groups for train/val split, got {len(group_keys)}."
        )

    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(group_keys))
    num_val_groups = max(1, int(round(len(group_keys) * val_split)))
    num_val_groups = min(num_val_groups, len(group_keys) - 1)
    val_group_keys = {
        group_keys[int(index)]
        for index in permutation[:num_val_groups].tolist()
    }
    train_items = [
        item for item in items if str(group_fn(item)) not in val_group_keys
    ]
    val_items = [
        item for item in items if str(group_fn(item)) in val_group_keys
    ]
    return train_items, val_items, {
        "num_groups": len(group_keys),
        "num_train_groups": len(group_keys) - len(val_group_keys),
        "num_val_groups": len(val_group_keys),
    }


def make_worker_init_fn(seed: int):
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def make_dataloader_kwargs(cfg: Any, seed: int, *, drop_last: bool | None = None) -> dict[str, Any]:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    num_workers = int(cfg.num_workers)
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "drop_last": bool(cfg.drop_last if drop_last is None else drop_last),
        "generator": generator,
        "worker_init_fn": make_worker_init_fn(seed),
        "pin_memory": bool(cfg.pin_memory),
    }
    if num_workers > 0:
        if hasattr(cfg, "persistent_workers"):
            kwargs["persistent_workers"] = bool(cfg.persistent_workers)
        if hasattr(cfg, "prefetch_factor"):
            kwargs["prefetch_factor"] = int(cfg.prefetch_factor)
    return kwargs


def resolve_data_path(path: str | Path, root: str | Path | None = None) -> Path:
    raw = Path(path)
    if raw.is_absolute() or raw.exists():
        return raw
    if root is not None:
        candidate = Path(root) / raw
        if candidate.exists():
            return candidate
    return raw


def make_file_split_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    dataset_cls: type[Dataset],
    dataloader_cls: type[DataLoader] = DataLoader,
    resolve_data_dir=None,
    root: str | Path | None = None,
    file_pattern: str = "*.npz",
    train_dataset_kwargs: dict[str, Any] | None = None,
    val_dataset_kwargs: dict[str, Any] | None = None,
    split_group_fn: Callable[[Path], str] | None = None,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    """Build deterministic file-level train/validation loaders.

    Ref2Dex tasks own their sample schema and Dataset implementation.  The
    base layer only owns the shared file split, worker seeding, and DDP sampler
    policy.
    """
    train_path = resolve_data_path(data_cfg.train_path, root=root)
    val_path = None if data_cfg.val_path in {None, ""} else resolve_data_path(data_cfg.val_path, root=root)
    val_split = float(data_cfg.val_split)
    resolve_data_dir = resolve_data_dir or (lambda path: path)
    train_dataset_kwargs = dict(train_dataset_kwargs or {})
    val_dataset_kwargs = dict(val_dataset_kwargs or {})
    split_metadata: dict[str, Any] = {}

    def build_dataset(path, *, file_list=None, kwargs: dict[str, Any]):
        payload = dict(kwargs)
        if file_list is not None:
            payload["file_list"] = file_list
        return dataset_cls(path, **payload)

    if val_path is not None:
        train_dataset = build_dataset(train_path, kwargs=train_dataset_kwargs)
        val_dataset = build_dataset(val_path, kwargs=val_dataset_kwargs)
    else:
        data_dir = resolve_data_dir(train_path)
        file_list = sorted(Path(data_dir).glob(file_pattern))
        if not file_list:
            raise ValueError(f"No files matching {file_pattern!r} found in {data_dir}")

        if 0.0 < val_split < 1.0:
            if split_group_fn is not None:
                train_files, val_files, split_metadata = split_grouped_items(
                    file_list,
                    val_split=val_split,
                    seed=seed,
                    group_fn=split_group_fn,
                )
            else:
                train_files, val_files = split_items(file_list, val_split, seed)
            if not train_files:
                raise ValueError("Training set size must be positive")
        else:
            train_files = file_list
            val_files = []

        train_dataset = build_dataset(data_dir, file_list=train_files, kwargs=train_dataset_kwargs)
        val_dataset = (
            build_dataset(data_dir, file_list=val_files, kwargs=val_dataset_kwargs)
            if val_files
            else None
        )

    loader_seed = int(seed) + int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
    train_sampler = make_default_train_sampler(
        train_dataset,
        shuffle=bool(data_cfg.shuffle),
        seed=seed,
        distributed=distributed,
        drop_last=bool(data_cfg.drop_last),
    )
    train_loader = dataloader_cls(
        train_dataset,
        batch_size=int(data_cfg.batch_size),
        shuffle=bool(data_cfg.shuffle) and train_sampler is None,
        sampler=train_sampler,
        **make_dataloader_kwargs(data_cfg, loader_seed),
    )
    val_loader = None
    if val_dataset is not None:
        val_batch_size = getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size
        val_loader = dataloader_cls(
            val_dataset,
            batch_size=int(val_batch_size),
            shuffle=False,
            sampler=make_default_eval_sampler(val_dataset, distributed=distributed),
            **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
        )

    metadata = {
        "train_path": str(train_path),
        "val_path": None if val_path is None else str(val_path),
        "num_train_samples": len(train_dataset),
        "num_val_samples": 0 if val_dataset is None else len(val_dataset),
    }
    metadata.update(split_metadata)
    return train_loader, val_loader, metadata
