from __future__ import annotations

from pathlib import Path
import random
import json
from typing import Any, Callable, Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .distributed import make_default_eval_sampler, make_default_train_sampler


T = TypeVar("T")


def ensure_disjoint_splits(
    train_files: Sequence[Path],
    val_files: Sequence[Path],
    test_files: Sequence[Path],
) -> None:
    """Reject file-level leakage between explicitly supplied split lists."""
    named_sets = {
        "train": {path.resolve() for path in train_files},
        "val": {path.resolve() for path in val_files},
        "test": {path.resolve() for path in test_files},
    }
    overlaps = {
        "train_val": named_sets["train"] & named_sets["val"],
        "train_test": named_sets["train"] & named_sets["test"],
        "val_test": named_sets["val"] & named_sets["test"],
    }
    invalid = {name: paths for name, paths in overlaps.items() if paths}
    if invalid:
        details = {
            name: [str(path) for path in sorted(paths)[:5]]
            for name, paths in invalid.items()
        }
        raise ValueError(f"Data splits overlap: {details}")


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


def read_split_json(path: str | Path, *, root: str | Path | None = None) -> tuple[Path, Path, Path | None, Path | None]:
    """Read the canonical split descriptor and resolve its three list paths."""
    json_path = resolve_data_path(path, root=root).resolve()
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Split descriptor {json_path} is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Split descriptor {json_path} must contain a JSON object.")

    def resolve_entry(key: str, *, required: bool) -> Path | None:
        value = payload.get(key)
        if value in {None, ""}:
            if required:
                raise ValueError(f"Split descriptor {json_path} is missing required key {key!r}.")
            return None
        candidate = Path(str(value))
        return candidate if candidate.is_absolute() else (json_path.parent / candidate)

    return (
        json_path,
        resolve_entry("train_split", required=True),
        resolve_entry("val_split", required=False),
        resolve_entry("test_split", required=False),
    )


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
    test_dataset_kwargs: dict[str, Any] | None = None,
    split_group_fn: Callable[[Path], str] | None = None,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, dict[str, Any]]:
    """Build deterministic train/validation/test file loaders.

    Ref2Dex tasks own their sample schema and Dataset implementation.  The
    base layer only owns the shared file split, worker seeding, and DDP sampler
    policy.

    中文：构造 train/val/test DataLoader。
    - 各 task 自己拥有 sample schema 和 Dataset 实现；
    - base 层只负责共享的文件切分、worker seed 注入和 DDP sampler 策略。
    - ``test_path`` 必须显式提供，绝不从训练目录切分，避免测试集泄漏。
    """
    def read_split_file(value: str | Path, *, data_root: Path) -> list[Path]:
        split_path = resolve_data_path(value, root=root)
        entries = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [data_root / entry for entry in entries]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{split_path} references missing files: {missing[:3]}")
        return paths

    split_root = str(getattr(data_cfg, "root", "") or "").strip()
    split_json_path = getattr(data_cfg, "split_json_path", None)
    direct_splits = (
        getattr(data_cfg, "train_split", None),
        getattr(data_cfg, "val_split_path", None),
        getattr(data_cfg, "test_split", None),
    )
    if split_json_path not in {None, ""} and any(value not in {None, ""} for value in direct_splits):
        raise ValueError("Set either data.split_json_path or direct split paths, not both.")
    if split_json_path not in {None, ""}:
        descriptor_path, train_split, val_split_path, test_split = read_split_json(split_json_path, root=root)
        explicit_splits = (train_split, val_split_path, test_split)
    else:
        descriptor_path = None
        explicit_splits = direct_splits
    use_explicit_splits = any(value not in {None, ""} for value in explicit_splits)
    if use_explicit_splits and not split_root:
        raise ValueError("data.root is required when using split files.")
    # 把 data_cfg 里的相对路径解析为绝对路径；root 通常是项目根，用作相对路径基准
    train_path = resolve_data_path(data_cfg.train_path, root=root)
    val_path = None if data_cfg.val_path in {None, ""} else resolve_data_path(data_cfg.val_path, root=root)
    test_path = None if getattr(data_cfg, "test_path", None) in {None, ""} else resolve_data_path(data_cfg.test_path, root=root)
    val_split = float(data_cfg.val_split)
    # resolve_data_dir 允许把一个文件路径标准化为「它所在的目录」；默认就是恒等
    resolve_data_dir = resolve_data_dir or (lambda path: path)
    # 复制 kwargs 避免外部 dict 被原地修改
    train_dataset_kwargs = dict(train_dataset_kwargs or {})
    val_dataset_kwargs = dict(val_dataset_kwargs or {})
    test_dataset_kwargs = dict(test_dataset_kwargs or val_dataset_kwargs)
    split_metadata: dict[str, Any] = {}

    def build_dataset(path, *, file_list=None, kwargs: dict[str, Any]):
        """统一构造 dataset 的小工具：把 file_list 注入到 kwargs 里再交给 dataset_cls。"""
        payload = dict(kwargs)
        if file_list is not None:
            payload["file_list"] = file_list
        return dataset_cls(path, **payload)

    if use_explicit_splits:
        data_dir = resolve_data_path(split_root, root=root)
        train_split, val_split_path, test_split = explicit_splits
        if train_split in {None, ""}:
            raise ValueError("data.train_split is required when using split files.")
        train_files = read_split_file(train_split, data_root=data_dir)
        val_files = [] if val_split_path in {None, ""} else read_split_file(val_split_path, data_root=data_dir)
        test_files = [] if test_split in {None, ""} else read_split_file(test_split, data_root=data_dir)
        ensure_disjoint_splits(train_files, val_files, test_files)
        train_dataset = build_dataset(data_dir, file_list=train_files, kwargs=train_dataset_kwargs)
        val_dataset = build_dataset(data_dir, file_list=val_files, kwargs=val_dataset_kwargs) if val_files else None
        test_dataset = build_dataset(data_dir, file_list=test_files, kwargs=test_dataset_kwargs) if test_files else None
        test_path = None
        split_metadata = {"split_root": str(data_dir), "split_json_path": None if descriptor_path is None else str(descriptor_path), "train_split": str(train_split), "val_split": None if val_split_path in {None, ""} else str(val_split_path), "test_split": None if test_split in {None, ""} else str(test_split)}
    elif val_path is not None:
        # 显式给了 val_path：train / val 各走各的目录，直接各自构造
        train_dataset = build_dataset(train_path, kwargs=train_dataset_kwargs)
        val_dataset = build_dataset(val_path, kwargs=val_dataset_kwargs)
    else:
        # 没给 val_path：需要从一个目录里按文件粒度切分
        data_dir = resolve_data_dir(train_path)
        file_list = sorted(Path(data_dir).glob(file_pattern))
        if not file_list:
            raise ValueError(f"No files matching {file_pattern!r} found in {data_dir}")

        if 0.0 < val_split < 1.0:
            # 按比例切分；如果给了 split_group_fn 就按 group 切（保证同组文件不会
            # 跨 train/val 出现），否则直接按文件切
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
            # val_split=0 或越界：全部文件进 train，val 为空
            train_files = file_list
            val_files = []

        train_dataset = build_dataset(data_dir, file_list=train_files, kwargs=train_dataset_kwargs)
        val_dataset = (
            build_dataset(data_dir, file_list=val_files, kwargs=val_dataset_kwargs)
            if val_files
            else None
        )

    test_dataset = test_dataset if use_explicit_splits else (
        build_dataset(test_path, kwargs=test_dataset_kwargs)
        if test_path is not None
        else None
    )

    # loader_seed：worker 初始化的 RNG 用；DDP 时叠加 rank 让各 rank 的随机流不冲突
    loader_seed = int(seed) + int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
    # 训练 sampler：DDP 时返回 DistributedSampler；否则按 shuffle 选 Random/Sequential
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
        # sampler 非 None 时 PyTorch 要求 shuffle=False；显式写出来防误用
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
    test_loader = None
    if test_dataset is not None:
        test_batch_size = (
            getattr(data_cfg, "test_batch_size", None)
            or getattr(data_cfg, "val_batch_size", None)
            or data_cfg.batch_size
        )
        test_loader = dataloader_cls(
            test_dataset,
            batch_size=int(test_batch_size),
            shuffle=False,
            sampler=make_default_eval_sampler(test_dataset, distributed=distributed),
            **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
        )

    # 把路径、样本数等元信息返回出去，给 logger / runner 记录
    metadata = {
        "train_path": str(data_dir) if use_explicit_splits else str(train_path),
        "val_path": None if val_path is None else str(val_path),
        "test_path": None if test_path is None else str(test_path),
        "num_train_samples": len(train_dataset),
        "num_val_samples": 0 if val_dataset is None else len(val_dataset),
        "num_test_samples": 0 if test_dataset is None else len(test_dataset),
    }
    metadata.update(split_metadata)
    return train_loader, val_loader, test_loader, metadata
