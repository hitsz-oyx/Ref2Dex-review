from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn


def import_upstream_dynamics_predictor(pointworld_root: str):
    """导入官方 DynamicsPredictor；兼容本机 Python 3.8 的仅导入路径。"""
    root = str(Path(pointworld_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    if sys.version_info < (3, 9):
        # 这三个模块只被 BaseModel 使用；官方源码中的 list[str] 注解无法由
        # Python 3.8 解析。DynamicsPredictor 本身不读取它们。
        for name in ("pointworld.norm_stats", "pointworld.losses", "pointworld.metrics"):
            sys.modules.setdefault(name, types.ModuleType(name))
    from pointworld.base import DynamicsPredictor

    return DynamicsPredictor


class AdapterMLP(nn.Module):
    def __init__(self, in_dim: int, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class GRABPointWorldForward(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        if cfg.predictor_dim % 2:
            raise ValueError("predictor_dim 必须为偶数")
        dynamics_cls = import_upstream_dynamics_predictor(cfg.pointworld_root)
        args = Namespace(
            ptv3_size=cfg.ptv3_size,
            ptv3_patch_size=cfg.ptv3_patch_size,
            grid_size=cfg.grid_size,
            dynamics_head_init_scale=cfg.dynamics_head_init_scale,
        )
        channels = cfg.predictor_dim
        self.scene_adapter = AdapterMLP(14, channels)
        self.hand_adapter = AdapterMLP(9, channels)
        self.time_embed = nn.Embedding(11, channels)
        self.hand_type_emb = nn.Parameter(torch.zeros(1, 1, 1, channels))
        nn.init.normal_(self.hand_type_emb, std=0.02)
        self.dynamics_predictor = dynamics_cls(args, channels, T=11)
        means, stds = self._load_output_stats(cfg.norm_stats_path)
        self.register_buffer("output_mean", means.view(1, 11, 1, 3))
        self.register_buffer("output_std", stds.view(1, 11, 1, 3))

    @staticmethod
    def _load_output_stats(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        stats = data["per_timestep_statistics"]["droid"]["gt_scene_flows_relative"]
        means = torch.tensor([stats[f"timestep_{step}"]["mean"] for step in range(11)])
        variances = torch.tensor([stats[f"timestep_{step}"]["variance"] for step in range(11)])
        return means.float(), variances.clamp_min(1e-6).sqrt().float()

    @staticmethod
    def raw_features(
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        object0 = object_points[:, 0]
        # [B,T,Nobj,Nhand] 仅用于 1024x256 的 V0；避免引入新的近邻依赖。
        dist = torch.stack(
            [torch.cdist(object0, hand_points[:, step]).amin(dim=-1) for step in range(11)],
            dim=-1,
        )
        scene_raw = torch.cat([object_normals[:, 0], dist], dim=-1)
        velocity = torch.zeros_like(hand_points)
        velocity[:, 1:] = hand_points[:, 1:] - hand_points[:, :-1]
        acceleration = torch.zeros_like(hand_points)
        acceleration[:, 1:] = velocity[:, 1:] - velocity[:, :-1]
        hand_raw = torch.cat([hand_normals, velocity, acceleration], dim=-1)
        return scene_raw, hand_raw

    def forward(
        self,
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        scene_raw, hand_raw = self.raw_features(
            object_points, object_normals, hand_points, hand_normals
        )
        scene_feat = self.scene_adapter(scene_raw)
        hand_feat = self.hand_adapter(hand_raw)
        time_ids = torch.arange(11, device=hand_points.device)
        hand_feat = hand_feat + self.time_embed(time_ids)[None, :, None] + self.hand_type_emb
        scene_exists = torch.ones(scene_feat.shape[:2], dtype=torch.bool, device=scene_feat.device)
        hand_exists = torch.ones(hand_points.shape[:3], dtype=torch.bool, device=hand_points.device)
        upstream = self.dynamics_predictor(
            object_points[:, 0],
            scene_feat,
            scene_exists,
            hand_points,
            hand_feat,
            hand_exists,
            training=self.training,
        )
        displacement = upstream["pred"] * self.output_std + self.output_mean
        displacement = displacement.clone()
        displacement[:, 0] = 0
        tracks = object_points[:, :1] + displacement
        return {"object_tracks": tracks, "displacement": displacement, "log_var": upstream["log_var"]}

    def load_pointworld_checkpoint(self, path: str) -> Dict[str, int]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        if not isinstance(state, dict):
            raise TypeError(f"无法识别 PointWorld checkpoint: {path}")
        own = self.state_dict()
        selected = {}
        for raw_key, value in state.items():
            key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
            marker = "dynamics_predictor."
            if marker not in key:
                continue
            key = key[key.index(marker):]
            if key in own and own[key].shape == value.shape:
                selected[key] = value
        missing, unexpected = self.load_state_dict(selected, strict=False)
        backbone_loaded = sum(key.startswith("dynamics_predictor.predictor_model.") for key in selected)
        if backbone_loaded == 0:
            raise RuntimeError("checkpoint 没有匹配到 PTv3 interaction backbone 参数")
        return {
            "loaded": len(selected),
            "backbone_loaded": backbone_loaded,
            "missing": len(missing),
            "unexpected": len(unexpected),
        }
