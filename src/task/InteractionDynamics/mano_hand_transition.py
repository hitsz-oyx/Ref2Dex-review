"""V18.4 object-centric MANO-H deterministic transition baseline。"""
from __future__ import annotations

import bisect
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset

from src.task.InteractionDynamics.state_interaction_diffusion import AdaLNBlock


class CachedManoHDataset(Dataset):
    def __init__(self, root: str | Path, split: str) -> None:
        self.paths = sorted((Path(root) / split).glob("event_*.pt"))
        self.shards = [torch.load(path, map_location="cpu") for path in self.paths]
        if not self.shards:
            raise ValueError(f"No V18.4 MANO-H cache for {split}")
        self.offsets = [0]
        for shard in self.shards:
            self.offsets.append(self.offsets[-1] + len(shard["state"]))

    def __len__(self) -> int:
        return self.offsets[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_index = bisect.bisect_right(self.offsets, index) - 1
        local = index - self.offsets[shard_index]
        shard = self.shards[shard_index]
        keys = ("state", "future", "residual", "anchors_cm", "object_patches", "current_h",
                "future_delta_h", "frame", "grasp_frame", "frames_to_grasp",
                "betas", "raw_frame_ids", "object_rotation", "object_translation",
                "global_orient", "hand_pose", "transl")
        item = {key: shard[key][local] for key in keys}
        item.update({"side": shard["side"], "source_raw_file": shard["source_raw_file"],
                     "source_hand_cache": shard["source_hand_cache"],
                     "mano_key": str(Path(shard["source_hand_cache"]).parent.parent.name)
                     + ":" + shard["side"]})
        return item


class ManoHandTransition(nn.Module):
    def __init__(self, horizon: int = 8, dim: int = 256, heads: int = 8,
                 layers: int = 6) -> None:
        super().__init__(); self.horizon = horizon
        self.object_point = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.current_h = nn.Sequential(nn.Linear(33, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
                                  nn.GELU(), nn.Linear(dim, horizon * 30))

    def forward(self, state: torch.Tensor, anchors_cm: torch.Tensor,
                patches: torch.Tensor, current_h: torch.Tensor) -> torch.Tensor:
        condition = self.current_h(current_h)
        tokens = self.state(state) + self.anchor(anchors_cm) + self.object_point(patches).amax(2)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.head(torch.cat([tokens.mean(1), condition], -1)).reshape(-1, self.horizon, 30)
