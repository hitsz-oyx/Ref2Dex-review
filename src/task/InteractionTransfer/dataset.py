from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


class GRABOneStepDataset(Dataset):
    """Independent reader for the validated GRAB transition cache."""
    def __init__(self, root: str, sequences: List[str], object_points=512, hand_points=1538,
                 gaps=(1,), max_transitions=0):
        self.items = []
        for seq in sequences:
            d = Path(root) / seq
            if not all((d / n).is_file() for n in ("shared.npz", "right.npz", "left.npz")):
                continue
            with np.load(d / "shared.npz") as s, np.load(d / "right.npz") as r, np.load(d / "left.npz") as l:
                n = len(s["obj_points_world"])
                right_active = np.asarray(r["obj_candidate_mask_5cm"], dtype=bool).any(1)
                left_active = np.asarray(l["obj_candidate_mask_5cm"], dtype=bool).any(1)
                for gap in gaps:
                    for i in range(max(0, n - int(gap))):
                        j = i + int(gap)
                        if right_active[i] and right_active[j] and not left_active[i] and not left_active[j]:
                            self.items.append((d, i, int(gap)))
        if max_transitions:
            self.items = self.items[:max_transitions]
        if not self.items:
            raise FileNotFoundError("没有可用的 one-step transition")
        self.object_points, self.hand_points = object_points, hand_points

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        d, i, gap = self.items[index]
        j = i + gap
        with np.load(d / "shared.npz") as s, np.load(d / "right.npz") as h:
            op, on = s["obj_points_world"], s["obj_normals_world"]
            hp, hn = h["hand_points_world"], h["hand_normals_world"]
            oi = np.linspace(0, op.shape[1] - 1, self.object_points).astype(int)
            hi = np.linspace(0, hp.shape[1] - 1, self.hand_points).astype(int)
            o, nxt = op[i, oi].astype("f4"), op[j, oi].astype("f4")
            center = o.mean(0, keepdims=True)
            return {"object_points": torch.from_numpy(o - center), "object_normals": torch.from_numpy(on[i, oi].astype("f4")),
                    "hand_points": torch.from_numpy(hp[i, hi].astype("f4") - center), "hand_normals": torch.from_numpy(hn[i, hi].astype("f4")),
                    "hand_flow": torch.from_numpy((hp[j, hi] - hp[i, hi]).astype("f4")), "object_flow": torch.from_numpy(nxt - o)}
