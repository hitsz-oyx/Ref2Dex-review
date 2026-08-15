from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset

from src.task.PointWorldWAM.dataset import fixed_point_indices


@dataclass(frozen=True)
class TransitionRef:
    shared_path: Path
    right_path: Path
    current: int
    gap: int
    motion: float


class GRABOneStepDataset(Dataset):
    """Independent reader for the validated GRAB transition cache."""
    def __init__(self, root: str, sequences: List[str], object_points=512, hand_points=1538,
                 gaps=(1,), max_transitions=0):
        candidates = []
        for seq in sequences:
            d = Path(root) / seq
            if not all((d / n).is_file() for n in ("shared.npz", "right.npz", "left.npz")):
                continue
            with np.load(d / "shared.npz") as s, np.load(d / "right.npz") as r, np.load(d / "left.npz") as l:
                n = len(s["obj_points_world"])
                raw_frame_id = np.asarray(s["raw_frame_id"], dtype=np.int64)
                source_fps = float(np.asarray(s["source_fps"]).item())
                if source_fps != 120.0:
                    raise ValueError(f"{d / 'shared.npz'}: source_fps={source_fps}，预期 120")
                if not np.all(np.diff(raw_frame_id) == 4):
                    raise ValueError(f"{d / 'shared.npz'}: raw_frame_id 不是 4 的等差序列")
                right_active = np.asarray(r["obj_candidate_mask_5cm"], dtype=bool).any(1)
                left_active = np.asarray(l["obj_candidate_mask_5cm"], dtype=bool).any(1)
                for gap in gaps:
                    for i in range(max(0, n - int(gap))):
                        j = i + int(gap)
                        if right_active[i] and right_active[j] and not left_active[i] and not left_active[j]:
                            motion = float(np.linalg.norm(s["obj_points_world"][j] - s["obj_points_world"][i], axis=-1).mean())
                            candidates.append(TransitionRef(d / "shared.npz", d / "right.npz", i, int(gap), motion))
        candidates.sort(key=lambda ref: (-ref.motion, str(ref.shared_path), ref.current, ref.gap))
        self.items = candidates
        if max_transitions:
            self.items = self.items[:max_transitions]
        if not self.items:
            raise FileNotFoundError("没有可用的 one-step transition")
        self.object_points, self.hand_points = object_points, hand_points
        with np.load(self.items[0].shared_path, allow_pickle=False) as s, np.load(self.items[0].right_path, allow_pickle=False) as r:
            self.object_indices = fixed_point_indices(s["obj_points_world"].shape[1], object_points)
            self.hand_indices = fixed_point_indices(r["hand_points_world"].shape[1], hand_points)

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        ref = self.items[index]
        i, gap = ref.current, ref.gap
        j = i + gap
        with np.load(ref.shared_path, allow_pickle=False) as s, np.load(ref.right_path, allow_pickle=False) as h:
            op, on = s["obj_points_world"], s["obj_normals_world"]
            hp, hn = h["hand_points_world"], h["hand_normals_world"]
            oi, hi = self.object_indices, self.hand_indices
            o, nxt = op[i, oi].astype("f4"), op[j, oi].astype("f4")
            center = o.mean(0, keepdims=True)
            return {"object_points": torch.from_numpy(o - center), "object_normals": torch.from_numpy(on[i, oi].astype("f4")),
                    "hand_points": torch.from_numpy(hp[i, hi].astype("f4") - center), "hand_normals": torch.from_numpy(hn[i, hi].astype("f4")),
                    "hand_flow": torch.from_numpy((hp[j, hi] - hp[i, hi]).astype("f4")), "object_flow": torch.from_numpy(nxt - o)}
