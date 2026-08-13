"""V20 独立 causal field cache dataset。"""
from __future__ import annotations

import bisect
from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedFieldV20Dataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.paths = sorted((Path(root) / split).glob("*.pt"))
        if not self.paths: raise ValueError(f"No V20 cache for {split}")
        self.shards = [torch.load(path, map_location="cpu") for path in self.paths]
        self.offsets = [0]
        for shard in self.shards: self.offsets.append(self.offsets[-1] + len(shard["current_y"]))

    def __len__(self) -> int: return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard = bisect.bisect_right(self.offsets, index) - 1
        local = index - self.offsets[shard]
        return {key: value[local] for key, value in self.shards[shard].items()}
