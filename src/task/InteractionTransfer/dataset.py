from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset

from src.task.InteractionTransfer.geometry_cache import (
    gather_surface_points,
    sample_surface_refs,
)



def fixed_point_indices(pool_size: int, count: int) -> np.ndarray:
    """与 GRAB cache 对齐的确定性跨时间点采样。"""
    if count > pool_size:
        raise ValueError(f"采样数 {count} 大于点池 {pool_size}")
    return np.linspace(0, pool_size - 1, count, dtype=np.int64)


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
        if tuple(int(gap) for gap in gaps) != (1,):
            raise ValueError("InteractionTransfer V0.4 固定使用 gap=1")
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


class GRABRandomTransitionDataset(Dataset):
    """V1.0 geometry cache + online sampling 数据流。

    每个样本在 ``__getitem__`` 内完成（指导 V1.0 §10-11）：

    1. random current frame i（候选由 (R_active ∨ L_active) 在整个窗口
       [i, i+g] 连续成立的 start 构成）；
    2. random gap g ∈ configured gaps（默认 {1,2,4,8}），j = i+g；
    3. 左右手各在线随机采样 MANO surface 点（uniform face + Dirichlet
       barycentric），t 与 t+g 复用同一组 face id + 权重，保证 ΔH 是同一
       物理表面点的运动；
    4. 输出 concat([left, right]) 手点（默认 769+769=1538）、face normal、
       Δt（normalized gap），object 侧与 V0.x 相同（512 点、object-centered）。

    ``deterministic=True``（val/test）时按 index 播种，gap 与表面采样均可
    复现；``deterministic=False``（train）时使用 worker 本地 RNG，每个
    epoch 产生新的 gap / 表面采样。
    """

    def __init__(self, stage4_root: str, geometry_root: str, sequences: List[str],
                 object_points: int = 512, hand_points_per_side: int = 769,
                 gaps=(1, 2, 4, 8), seed: int = 13, deterministic: bool = False):
        self.stage4_root, self.geometry_root = Path(stage4_root), Path(geometry_root)
        self.object_points, self.hand_points_per_side = int(object_points), int(hand_points_per_side)
        self.gaps = tuple(int(g) for g in gaps)
        self.seed, self.deterministic = int(seed), bool(deterministic)

        meta = json.loads((self.geometry_root / "meta.json").read_text(encoding="utf-8"))
        self.config = {"stage4_root": meta["stage4_root"], "geometry_root": str(self.geometry_root),
                       "object_points": self.object_points,
                       "hand_points_per_side": self.hand_points_per_side,
                       "gaps": list(self.gaps), "seed": self.seed,
                       "deterministic": self.deterministic}
        self.faces = {side: np.load(self.geometry_root / f"faces_{side}.npy") for side in ("left", "right")}

        missing = [s for s in sequences if s not in meta["sequences"]]
        if missing:
            raise FileNotFoundError(f"{len(missing)} 个 sequence 不在 geometry cache 中，例如 {missing[:3]}")

        # 每个 (sequence, start) 记录全部 valid gaps；len(dataset) = start 数。
        items: list[tuple[str, int, tuple[int, ...]]] = []
        for seq in sorted(sequences):
            if seq not in meta["sequences"]:
                continue
            active_l = self._geometry(seq, "left")[1]
            active_r = self._geometry(seq, "right")[1]
            active = active_l | active_r
            prefix = np.concatenate([[0], np.cumsum(active)])
            n = len(active)
            per_start: dict[int, list[int]] = {}
            for g in self.gaps:
                if n <= g:
                    continue
                ok = (prefix[g + 1:] - prefix[:n - g]) == (g + 1)
                for i in np.nonzero(ok)[0]:
                    per_start.setdefault(int(i), []).append(g)
            for i in sorted(per_start):
                gaps_i = tuple(sorted(per_start[i]))
                if gaps_i:
                    items.append((seq, i, gaps_i))
        if not items:
            raise FileNotFoundError("没有可用 transition（(R∨L)-active 窗口为空）")
        self.items = items
        self._rng = None  # train 模式下惰性创建（fork 后每个 worker 独立）

    def __len__(self):
        return len(self.items)

    # ---------- 文件句柄 / 数组缓存（per worker，LRU 限制内存） ----------
    @lru_cache(maxsize=8)
    def _geometry(self, seq: str, side: str):
        with np.load(self.geometry_root / seq / f"{side}.npz", allow_pickle=False) as z:
            return np.asarray(z["hand_vertices_world"]), np.asarray(z["active_mask"], dtype=bool)

    @lru_cache(maxsize=8)
    def _hand_static(self, seq: str, side: str):
        with np.load(self.stage4_root / seq / f"{side}.npz", allow_pickle=False) as z:
            return np.asarray(z["hand_normals_world"])

    @lru_cache(maxsize=8)
    def _object(self, seq: str):
        with np.load(self.stage4_root / seq / "shared.npz", allow_pickle=False) as z:
            return np.asarray(z["obj_points_world"]), np.asarray(z["obj_normals_world"])

    def _train_rng(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _item_rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng([self.seed, index])
        return self._train_rng()

    def __getitem__(self, index: int):
        seq, i, gaps_i = self.items[index]
        rng = self._item_rng(index)
        gap = int(gaps_i[int(rng.integers(len(gaps_i)))])
        j = i + gap

        obj_points, obj_normals = self._object(seq)
        oi = fixed_point_indices(obj_points.shape[1], self.object_points)
        o = obj_points[i, oi].astype("f4")
        center = o.mean(0, keepdims=True)
        nxt = obj_points[j, oi].astype("f4")

        pts, nrm, flow = [], [], []
        for side in ("left", "right"):
            vertices, _ = self._geometry(seq, side)
            face_normals = self._hand_static(seq, side)
            face_idx, bary = sample_surface_refs(self.faces[side].shape[0], self.hand_points_per_side, rng)
            p_i = gather_surface_points(vertices[i], self.faces[side], face_idx, bary)
            p_j = gather_surface_points(vertices[j], self.faces[side], face_idx, bary)
            pts.append(p_i - center)
            nrm.append(face_normals[i][face_idx].astype("f4"))
            flow.append(p_j - p_i)

        return {"object_points": torch.from_numpy(o - center),
                "object_normals": torch.from_numpy(obj_normals[i, oi].astype("f4")),
                "hand_points": torch.from_numpy(np.concatenate(pts)),
                "hand_normals": torch.from_numpy(np.concatenate(nrm)),
                "hand_flow": torch.from_numpy(np.concatenate(flow).astype("f4")),
                "object_flow": torch.from_numpy(nxt - o),
                "gap": torch.tensor(float(gap), dtype=torch.float32)}
