"""V17 full-frame、sequence-disjoint interaction residual dataset。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset
from src.task.InteractionDynamics.goal_interaction_diffusion import (
    estimate_active_interval, extract_goal, meaningful_motion_mask)
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.residual_interaction_regression import residual_target
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future
from src.task.InteractionDynamics.uni3d import gather_points, patchify


@dataclass(frozen=True)
class V17Split:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def dominant_sequences(data_root: str | Path, manifest: str | Path) -> list[Path]:
    """取 manifest 确认过 dominant hand 的 sequence 文件，但不沿用其 chunk 筛选。"""
    import json
    root = Path(data_root)
    paths = set()
    with Path(manifest).open() as stream:
        for line in stream:
            row = json.loads(line)
            relative = row.get("hand_path") or row.get("path")
            if relative:
                path = Path(relative)
                paths.add(path if path.is_absolute() else root / path)
    return sorted(path for path in paths if path.exists())


def split_sequences(paths: list[Path], seed: int = 42,
                    ratios=(.8, .1, .1)) -> V17Split:
    groups: dict[Path, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent, []).append(path)
    if len(groups) < 3 or abs(sum(ratios) - 1) > 1e-6:
        raise ValueError("V17 needs at least three sequences and ratios summing to one")
    order = sorted(groups)
    np.random.default_rng(seed).shuffle(order)
    train_end = max(1, int(len(order) * ratios[0]))
    val_end = max(train_end + 1, int(len(order) * (ratios[0] + ratios[1])))
    expand = lambda names: sorted(path for name in names for path in groups[name])
    return V17Split(expand(order[:train_end]), expand(order[train_end:val_end]),
                    expand(order[val_end:]))


class V17Dataset(Dataset):
    def __init__(self, data_root: str | Path, paths: list[Path], horizon: int = 4,
                 goal_segment: int = 4, tau_m: float = .015) -> None:
        self.base = InteractionDynamicsDataset(data_root, file_list=paths)
        self.horizon, self.goal_segment, self.tau_m = horizon, goal_segment, tau_m
        self._sequence: dict[Path, dict] = {}
        self.samples: list[tuple[Path, int]] = []
        for path in paths:
            data = self.base._load(path)
            poses = torch.from_numpy(np.asarray(data["obj_root_pose_world"], np.float32))
            meaningful = meaningful_motion_mask(poses, 3, .2, 1.).numpy()
            if not meaningful.any():
                continue
            start, end, distance = estimate_active_interval(data, meaningful)
            self._sequence[path] = {"poses": poses, "active": (start, end),
                                    "distance_cm": distance}
            self.samples.extend((path, frame) for frame in range(len(poses) - max(8, horizon)))
        self._index = {sample: index for index, sample in enumerate(self.base._samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, frame = self.samples[index]
        sample = self.base[self._index[(path, frame)]]
        meta = self._sequence[path]
        active = meta["active"][0] <= frame <= meta["active"][1]
        goal, _ = extract_goal(meta["poses"], frame, active, self.goal_segment, 3, .2, 1.)
        hand = sample["action_hand_points_object_sequence"][:self.horizon + 1]
        points, normals = sample["world_obj_points_object"], sample["world_obj_normals_object"]
        anchors, _, knn = patchify(points[None], 128, 32)
        patches = torch.cat([100 * (gather_points(points[None], knn) - anchors[:, :, None]),
                             gather_points(normals[None], knn)], -1)[0]
        y = build_interaction_y(hand, anchors[0], self.tau_m)
        y = {key: 100 * value for key, value in y.items()
             if key in {"relative_geometry", "relative_distance", "relative_motion"}}
        state, future = pack_state_future(y, self.horizon)
        residual = residual_target(state, future, self.horizon)
        return {"state": state, "future": future, "residual": residual,
                "anchors_cm": 100 * anchors[0], "object_patches": patches, "goal": goal,
                "distance_cm": torch.tensor(meta["distance_cm"][frame]),
                "sequence": str(path), "frame": frame}


class CachedV17Dataset(Dataset):
    """读取按 shard 保存的 V17 tensors，避免训练时重复解压 NPZ 和构造 Teacher。"""
    def __init__(self, root: str | Path, split: str) -> None:
        self.files = sorted((Path(root) / split).glob("*.pt"))
        if not self.files:
            raise ValueError(f"No V17 cache shards found for {split} under {root}")
        self.shards, self.offsets = [], [0]
        for path in self.files:
            shard = torch.load(path, map_location="cpu")
            self.shards.append(shard)
            self.offsets.append(self.offsets[-1] + len(shard["state"]))

    def __len__(self): return self.offsets[-1]

    def __getitem__(self, index):
        import bisect
        shard_index = bisect.bisect_right(self.offsets, index) - 1
        local = index - self.offsets[shard_index]
        return {key: value[local] for key, value in self.shards[shard_index].items()}
