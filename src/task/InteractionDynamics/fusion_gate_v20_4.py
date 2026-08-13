"""V20.4 只融合 velocity 的轻量 gate 与冻结预测缓存。"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset


def fusion_features(y_free: torch.Tensor, y_h: torch.Tensor) -> torch.Tensor:
    """构造每个 anchor/timestep 的 8 维预测 disagreement 特征。"""
    horizon = y_free.shape[-2]
    time = torch.linspace(1 / horizon, 1, horizon, device=y_free.device, dtype=y_free.dtype)
    time = time.reshape(*([1] * (y_free.ndim - 2)), horizon, 1).expand(*y_free.shape[:-1], 1)
    return torch.cat((
        y_free[..., 3:4], y_h[..., 3:4], (y_free[..., 3:4] - y_h[..., 3:4]).abs(),
        (y_free[..., :3] - y_h[..., :3]).square().sum(-1, keepdim=True).sqrt(),
        (y_free[..., 4:7] - y_h[..., 4:7]).square().sum(-1, keepdim=True).sqrt(),
        y_free[..., 4:7].square().sum(-1, keepdim=True).sqrt(),
        y_h[..., 4:7].square().sum(-1, keepdim=True).sqrt(), time,
    ), dim=-1)


class VelocityFusionGate(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))


def fuse_velocity(y_free: torch.Tensor, y_h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    result = y_free.clone()
    result[..., 4:7] = (1 - gate) * y_free[..., 4:7] + gate * y_h[..., 4:7]
    return result


class FusionPredictionDataset(Dataset):
    def __init__(self, path: str | Path) -> None:
        self.data = torch.load(path, map_location="cpu")
        required = {"y_free", "y_h", "future_y", "current_y"}
        missing = required - self.data.keys()
        if missing:
            raise ValueError(f"融合缓存缺少字段: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.data["y_free"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {key: self.data[key][index]
                for key in ("y_free", "y_h", "future_y", "current_y")}
