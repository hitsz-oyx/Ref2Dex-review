"""Object-v2 MANO dataset adapter for the GRAB-trained CmDecoder probe."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from torch.utils.data import DataLoader

from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, make_default_train_sampler
from src.task.Cm.dataset.object_v2 import _MmapSequenceDataset


def _split_sequences(root: Path, split_json: Path, dataset_filter: str) -> dict[str, list[Path]]:
    payload = json.loads(split_json.read_text(encoding="utf-8"))
    result: dict[str, list[Path]] = {}
    for split, filename in (("train", payload["train_split"]),
                            ("val", payload.get("val_split")),
                            ("test", payload.get("test_split"))):
        if not filename:
            result[split] = []
            continue
        split_path = Path(filename)
        if not split_path.is_absolute():
            split_path = split_json.parent / split_path
        entries = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = []
        prefix = dataset_filter.strip().strip("/") + "/"
        for entry in entries:
            if not entry.startswith(prefix):
                continue
            sequence = (root / entry).resolve()
            try:
                sequence.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"Object-v2 split entry escapes root: {entry}") from exc
            if not (sequence / "shared" / "meta.json").is_file():
                raise FileNotFoundError(f"Missing object-v2 sequence: {sequence}")
            paths.append(sequence)
        result[split] = sorted(paths)
    if not result["train"]:
        raise ValueError(f"No {dataset_filter!r} sequences in train split {split_json}")
    return result


def make_dataloaders(data_cfg: Any, seed: int, *, meta_cfg: Any, distributed: Any = None):
    root = Path(str(data_cfg.root)).expanduser().resolve()
    split_json = Path(str(data_cfg.split_json_path)).expanduser()
    if not split_json.is_absolute():
        split_json = (Path.cwd() / split_json).resolve()
    splits = _split_sequences(root, split_json, str(data_cfg.object_v2_filter))
    common = dict(
        num_obj_points=int(meta_cfg.num_obj_points),
        num_hand_points=int(meta_cfg.num_hand_points),
        base_seed=int(seed),
        min_stride=int(getattr(data_cfg, "min_stride", 1)),
        max_stride=int(getattr(data_cfg, "max_stride", 10)),
        active_only=bool(getattr(data_cfg, "active_only", True)),
        sampling_bank_size=int(getattr(data_cfg, "sampling_bank_size", 4)),
        fixed_eval_bank=int(getattr(data_cfg, "fixed_eval_bank", 0)),
    )
    train = _MmapSequenceDataset(splits["train"], **common)
    val = _MmapSequenceDataset(
        splits["val"], fixed_stride=int(getattr(data_cfg, "val_stride", 1)), **common
    ) if splits["val"] else None
    test = _MmapSequenceDataset(
        splits["test"], fixed_stride=int(getattr(data_cfg, "test_stride", 1)), **common
    ) if splits["test"] else None
    train_kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
    sampler = None
    if distributed is not None:
        sampler = make_default_train_sampler(
            train, shuffle=True, seed=seed, distributed=distributed, drop_last=False
        )
    train_kwargs.update(
        batch_size=int(data_cfg.batch_size),
        shuffle=sampler is None,
        sampler=sampler,
    )
    train_loader = DataLoader(train, **train_kwargs)

    def eval_loader(dataset, batch_size):
        if dataset is None:
            return None
        kwargs = make_dataloader_kwargs(data_cfg, seed, drop_last=False)
        kwargs.update(
            batch_size=int(batch_size or data_cfg.batch_size),
            shuffle=False,
            sampler=(make_default_eval_sampler(dataset, distributed=distributed)
                     if distributed is not None else None),
        )
        return DataLoader(dataset, **kwargs)

    val_loader = eval_loader(val, getattr(data_cfg, "val_batch_size", None))
    test_loader = eval_loader(test, getattr(data_cfg, "val_batch_size", None))
    val_loaders = {"val/": val_loader} if val_loader is not None else {}
    test_loaders = {"test/": test_loader} if test_loader is not None else {}
    metadata = {
        "schema_name": "ref2dex_cmdecoder_object_v2_mano",
        "dataset_filter": str(data_cfg.object_v2_filter),
        "train_sequences": len(splits["train"]),
        "val_sequences": len(splits["val"]),
        "test_sequences": len(splits["test"]),
        "train_samples": len(train),
    }
    return train_loader, val_loader, test_loader, metadata, val_loaders, test_loaders
