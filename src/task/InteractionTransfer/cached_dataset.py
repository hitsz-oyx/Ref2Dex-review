"""V0.8 读取全量 static cache 的 dataset。

与 ``precompute_static.py`` 生成的 per-sequence .npy 对接，惰性 mmap 打开，
__getitem__ 只拷贝单个 transition，可支撑全量数据多 epoch 训练。
"""
from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ARRAYS = ("object_points", "object_normals", "edge_idx", "edge_valid",
          "dense_edge", "geom_contact", "hand_flow", "object_flow")


def cache_file_name(sequence: str, key: str) -> str:
    return f"{sequence.replace('/', '__')}.{key}.npy"


class CachedTransitionDataset(Dataset):
    def __init__(self, cache_root, sequences):
        self.root = Path(cache_root)
        meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.config = meta["config"]
        available = meta["sequences"]
        names = sorted(sequences)
        missing = [n for n in names if n not in available]
        if missing:
            raise FileNotFoundError(f"{len(missing)} 个 sequence 不在 static cache 中，例如 {missing[:3]}")
        self.names = [n for n in names if int(available[n]) > 0]
        self.counts = [int(available[n]) for n in self.names]
        if not self.names:
            raise FileNotFoundError("指定的 sequences 在 cache 中没有可用 transition")
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)]).astype(np.int64)
        self._mmaps: dict[tuple[str, str], np.memmap] = {}

    def __len__(self):
        return int(self.offsets[-1])

    def _array(self, name: str, key: str) -> np.memmap:
        handle = (name, key)
        if handle not in self._mmaps:
            path = self.root / cache_file_name(name, key)
            if not path.is_file():
                raise FileNotFoundError(path)
            self._mmaps[handle] = np.load(path, mmap_mode="r")
        return self._mmaps[handle]

    def __getitem__(self, index):
        seq_i = bisect_right(self.offsets, index) - 1
        name, t = self.names[seq_i], int(index - self.offsets[seq_i])
        return {key: torch.from_numpy(np.ascontiguousarray(self._array(name, key)[t])) for key in ARRAYS}
