from __future__ import annotations

import numpy as np
from pathlib import Path
from torch.utils.data import Sampler

from src.task.correspondence_ptv3.sampling import stable_frame_seed


def sequence_group_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_left") or stem.endswith("_right"):
        stem = stem.rsplit("_", 1)[0]
    parent = path.parent.as_posix()
    return stem if parent in {"", "."} else f"{parent}/{stem}"


class SequenceLocalitySampler(Sampler[int]):
    def __init__(self, dataset: object, seed: int) -> None:
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
