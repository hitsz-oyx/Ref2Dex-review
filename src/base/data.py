from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Sequence, TypeVar

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .distributed import make_default_eval_sampler, make_default_train_sampler

T = TypeVar("T")


@dataclass
class TensorStats:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value.float() - self.mean) / (self.std + self.eps)

    def inverse(self, value: torch.Tensor) -> torch.Tensor:
        return value.float() * (self.std + self.eps) + self.mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TensorStats":
        return cls(
            mean=torch.as_tensor(payload["mean"], dtype=torch.float32),
            std=torch.as_tensor(payload["std"], dtype=torch.float32),
            eps=float(payload.get("eps", 1e-6)),
        )

    def to(self, device: torch.device | str) -> "TensorStats":
        return TensorStats(self.mean.to(device), self.std.to(device), self.eps)


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    cfg: Any,
    seed: int = 1000,
    *,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    loader_seed = int(seed) + int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
    train_sampler = make_default_train_sampler(
        train_dataset,
        shuffle=bool(cfg.shuffle),
        seed=seed,
        distributed=distributed,
        drop_last=bool(cfg.drop_last),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=bool(cfg.shuffle) and train_sampler is None,
        sampler=train_sampler,
        **make_dataloader_kwargs(cfg, loader_seed),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=getattr(cfg, "val_batch_size", None) or cfg.batch_size,
            shuffle=False,
            sampler=make_default_eval_sampler(val_dataset, distributed=distributed),
            **make_dataloader_kwargs(cfg, loader_seed, drop_last=False),
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def dataset_metadata(
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    *,
    sample_fields: Iterable[str | int | Callable[[Any], Any]] | None = None,
) -> dict[str, Any]:
    metadata = {
        "num_train": len(train_dataset),
        "num_val": len(val_dataset) if val_dataset is not None else 0,
    }
    if sample_fields is None:
        return metadata
    if isinstance(sample_fields, (str, int)) or callable(sample_fields):
        sample_fields = [sample_fields]
    for field in sample_fields:
        field_name = getattr(field, "__name__", str(field))
        metadata[f"{field_name}_shape"] = list(sample_shape(train_dataset, field))
    return metadata


def as_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif isinstance(value, list):
        tensor = tensor_from_list(value)
    else:
        tensor = torch.as_tensor(value)

    if tensor.dtype in {torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        return tensor.float()
    return tensor


class BaseDataset(Dataset):
    """Base class for datasets that handle dictionary-based payloads with field aliasing."""

    # Subclasses should define this mapping: {canonical_key: tuple_of_aliases}
    FIELD_ALIASES: dict[str, tuple[str, ...]] = {}

    def __init__(self, payload: dict[str, Any]) -> None:
        self.fields = self.standardize_payload(payload)
        lengths = {key: int(value.shape[0]) for key, value in self.fields.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All fields must have the same leading dimension, got {lengths}")
        self.length = next(iter(lengths.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: value[index] for key, value in self.fields.items()}

    def standardize_payload(self, payload: dict[str, Any]) -> dict[str, torch.Tensor]:
        fields: dict[str, torch.Tensor] = {}
        for canonical_key in self.FIELD_ALIASES:
            try:
                fields[canonical_key] = self.select_tensor(payload, canonical_key)
            except KeyError:
                # Optional fields are handled by subclasses if needed
                continue

        self.validate_shapes(fields)
        return fields

    def select_tensor(self, payload: dict[str, Any], canonical_key: str) -> torch.Tensor:
        for key in self.FIELD_ALIASES[canonical_key]:
            if key in payload:
                return as_tensor(payload[key])
        raise KeyError(
            f"Missing required field '{canonical_key}'. "
            f"Accepted keys: {self.FIELD_ALIASES[canonical_key]}"
        )

    def has_any_key(self, payload: dict[str, Any], aliases: tuple[str, ...]) -> bool:
        return any(key in payload for key in aliases)

    def validate_shapes(self, fields: dict[str, torch.Tensor]) -> None:
        """Subclasses should implement specific shape validation here."""
        pass

    @classmethod
    def from_file(cls, path: str | Path) -> "BaseDataset":
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                return cls(dict(payload))
        if suffix in {".pt", ".pth"}:
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"Expected {path} to contain a dict, got {type(payload).__name__}")
            return cls(payload)
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use .npz, .pt, or .pth.")

    @classmethod
    def make_dataloaders(cls, cfg: Any, seed: int = 1000) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
        if not cfg.train_path:
            raise ValueError("data.train_path must be set.")

        train_dataset = cls.from_file(cfg.train_path)
        if cfg.val_path:
            val_dataset = cls.from_file(cfg.val_path)
        else:
            train_dataset, val_dataset = split_train_val(train_dataset, float(cfg.val_split), seed)

        train_loader, val_loader = create_dataloaders(train_dataset, val_dataset, cfg, seed)
        return train_loader, val_loader, cls.dataset_metadata(train_dataset, val_dataset)

    @classmethod
    def dataset_metadata(cls, train_dataset: Dataset, val_dataset: Dataset | None) -> dict[str, Any]:
        metadata = {
            "num_train": len(train_dataset),
            "num_val": len(val_dataset) if val_dataset is not None else 0,
        }
        return metadata


def tensor_from_list(values: list[Any]) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot build tensor from an empty list.")
    if all(torch.is_tensor(item) for item in values):
        return torch.stack([item.detach().cpu() for item in values])
    if all(isinstance(item, np.ndarray) for item in values):
        return torch.from_numpy(np.stack(values))
    return torch.as_tensor(values)


def split_train_val(dataset: Dataset, val_split: float, seed: int) -> tuple[Dataset, Dataset | None]:
    if val_split <= 0:
        return dataset, None
    if not 0 < val_split < 1:
        raise ValueError(f"data.val_split must be in [0, 1), got {val_split}")
    n = len(dataset)
    val_size = int(round(n * val_split))
    if val_size == 0:
        return dataset, None
    train_size = n - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_dataset, val_dataset


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


def select_items_for_split(
    items: Sequence[T],
    *,
    split: str,
    val_ratio: float,
    split_seed: int,
    split_key: str,
) -> tuple[list[T], list[T]]:
    if split == "all":
        return list(items), []
    if len(items) < 2:
        raise ValueError(f"Need at least two items for train/val split, got {len(items)} in {split_key}")

    seed_payload = f"{split_seed}:{split_key}".encode("utf-8")
    stable_seed = int.from_bytes(hashlib.sha256(seed_payload).digest()[:8], byteorder="little", signed=False)
    rng = np.random.default_rng(stable_seed)
    permutation = rng.permutation(len(items))
    num_val = int(round(len(items) * val_ratio))
    num_val = min(max(num_val, 1), len(items) - 1)
    val_indices = set(int(index) for index in permutation[:num_val])
    val_items = [item for index, item in enumerate(items) if index in val_indices]

    if split == "val":
        return val_items, val_items
    train_items = [item for index, item in enumerate(items) if index not in val_indices]
    return train_items, val_items


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


def is_readable_hdf5(
    path: str | Path,
    *,
    required_keys: Iterable[str] = ("data",),
    verbose: bool = True,
) -> bool:
    try:
        import h5py
    except ImportError:
        if verbose:
            print("h5py is not installed, skipping hdf5 check.")
        return False
    try:
        with h5py.File(path, "r") as handle:
            return all(key in handle for key in required_keys)
    except OSError:
        if verbose:
            print(f"skipping unreadable hdf5: {path}")
        return False


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


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
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any]]:
    train_path = resolve_data_path(data_cfg.train_path, root=root)
    val_path = None if data_cfg.val_path in {None, ""} else resolve_data_path(data_cfg.val_path, root=root)
    val_split = float(data_cfg.val_split)
    resolve_data_dir = resolve_data_dir or (lambda path: path)
    train_dataset_kwargs = dict(train_dataset_kwargs or {})
    val_dataset_kwargs = dict(val_dataset_kwargs or {})

    def build_dataset(path, *, file_list=None, kwargs: dict[str, Any]):
        payload = dict(kwargs)
        if file_list is not None:
            payload["file_list"] = file_list
        return dataset_cls(
            path,
            **payload,
        )

    if val_path is not None:
        train_dataset = build_dataset(
            train_path,
            kwargs=train_dataset_kwargs,
        )
        val_dataset = build_dataset(
            val_path,
            kwargs=val_dataset_kwargs,
        )
    else:
        data_dir = resolve_data_dir(train_path)
        file_list = sorted(Path(data_dir).glob(file_pattern))
        if not file_list:
            raise ValueError(f"No files matching {file_pattern!r} found in {data_dir}")

        if 0.0 < val_split < 1.0:
            train_files, val_files = split_items(file_list, val_split, seed)
            if not train_files:
                raise ValueError("Training set size must be positive")
        else:
            train_files = file_list
            val_files = []

        train_dataset = build_dataset(
            data_dir,
            file_list=train_files,
            kwargs=train_dataset_kwargs,
        )
        val_dataset = (
            build_dataset(
                data_dir,
                file_list=val_files,
                kwargs=val_dataset_kwargs,
            )
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
    return train_loader, val_loader, metadata


def get_sample_field(sample: Any, field: str | int | Callable[[Any], Any]) -> Any:
    if callable(field):
        return field(sample)
    if isinstance(sample, dict):
        if field in sample:
            return sample[field]
        raise KeyError(f"Field {field!r} not found in sample. Available keys: {sorted(sample.keys())}")
    if isinstance(field, int) and isinstance(sample, (tuple, list)):
        return sample[field]
    if isinstance(field, str) and hasattr(sample, field):
        return getattr(sample, field)
    raise TypeError(f"Cannot read field {field!r} from sample type {type(sample).__name__}.")


def sample_shape(dataset: Dataset, field: str | int | Callable[[Any], Any]) -> tuple[int, ...]:
    if len(dataset) == 0:
        raise ValueError("Cannot infer shape from an empty dataset.")
    value = as_tensor(get_sample_field(dataset[0], field))
    return tuple(value.shape)


def compute_tensor_stats(
    dataset: Dataset,
    field: str | int | Callable[[Any], Any],
    sample_limit: int | None = None,
) -> TensorStats:
    if isinstance(dataset, Subset):
        iterable: Iterable[int] = dataset.indices
        source = dataset.dataset
    else:
        iterable = range(len(dataset))
        source = dataset

    count = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    for idx in iterable:
        value = as_tensor(get_sample_field(source[idx], field)).float()
        if mean is None:
            mean = torch.zeros_like(value)
            m2 = torch.zeros_like(value)
        count += 1
        delta = value - mean
        mean = mean + delta / count
        delta2 = value - mean
        m2 = m2 + delta * delta2
        if sample_limit is not None and count >= sample_limit:
            break

    if count == 0 or mean is None or m2 is None:
        raise ValueError("Cannot compute statistics from an empty dataset.")
    variance = m2 / max(1, count)
    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    return TensorStats(mean=mean, std=std)
