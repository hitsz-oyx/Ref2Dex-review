from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .mano_action import BimanualManoAction
from .pointworld_bimanual_forward import (
    build_predictor,
    load_pointworld_backbone,
    run_joint_backbone,
)
from .pointworld_forward import AdapterMLP


class ScalarTimeEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(32) * math.pi)
        self.mlp = nn.Sequential(nn.Linear(64, channels), nn.SiLU(), nn.Linear(channels, channels))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        phase = value[:, None] * self.frequencies[None]
        return self.mlp(torch.cat([phase.sin(), phase.cos()], dim=-1))


class WAMV1(nn.Module):
    def __init__(self, cfg, statistics: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        channels = cfg.predictor_dim
        self.predictor_model = build_predictor(cfg)
        self.register_buffer("_grid_size", torch.tensor([cfg.grid_size]), persistent=False)
        self.mano = BimanualManoAction(
            cfg.mano_model_dir,
            cfg.left_vtemplate,
            cfg.right_vtemplate,
            statistics,
        )
        self.object_adapter = AdapterMLP(14, channels)
        self.left_adapter = AdapterMLP(9, channels)
        self.right_adapter = AdapterMLP(9, channels)
        self.object_type_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.left_type_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.right_type_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.tau_embedding = ScalarTimeEmbedding(channels)
        self.action_embedding = nn.Sequential(
            nn.Linear(60, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        readout_channels = 5 * channels
        self.left_action_head = nn.Sequential(
            nn.Linear(readout_channels, 256), nn.GELU(), nn.Linear(256, 30)
        )
        self.right_action_head = nn.Sequential(
            nn.Linear(readout_channels, 256), nn.GELU(), nn.Linear(256, 30)
        )
        for embedding in (self.object_type_emb, self.left_type_emb, self.right_type_emb):
            nn.init.normal_(embedding, std=0.02)

    def normalized_target(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            side: self.mano.normalize(side, batch[f"{side}_action"])
            for side in ("left", "right")
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_left_action: torch.Tensor,
        noisy_right_action: torch.Tensor,
        tau: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        noisy_points = {}
        for side, action in (
            ("left", noisy_left_action),
            ("right", noisy_right_action),
        ):
            noisy_points[side] = self.mano.points_from_action(
                side,
                batch[f"{side}_global_orient"],
                batch[f"{side}_hand_pose"],
                batch[f"{side}_transl"],
                batch[f"{side}_betas"],
                action,
            )
        left_flow = noisy_points["left"] - batch["left_hand_points"]
        right_flow = noisy_points["right"] - batch["right_hand_points"]
        distance_left = torch.cdist(
            batch["object_points"], batch["left_hand_points"]
        ).amin(-1, keepdim=True)
        distance_right = torch.cdist(
            batch["object_points"], batch["right_hand_points"]
        ).amin(-1, keepdim=True)
        time_feature = self.tau_embedding(tau)[:, None]
        object_feat = self.object_adapter(
            torch.cat(
                [
                    batch["object_points"],
                    batch["object_normals"],
                    batch["prev_object_flow"],
                    batch["target_object_flow"],
                    distance_left,
                    distance_right,
                ],
                dim=-1,
            )
        ) + self.object_type_emb + time_feature
        left_feat = self.left_adapter(
            torch.cat(
                [batch["left_hand_points"], batch["left_hand_normals"], left_flow],
                dim=-1,
            )
        ) + self.left_type_emb + time_feature
        right_feat = self.right_adapter(
            torch.cat(
                [batch["right_hand_points"], batch["right_hand_normals"], right_flow],
                dim=-1,
            )
        ) + self.right_type_emb + time_feature
        object_output, left_output, right_output = run_joint_backbone(
            self.predictor_model,
            [
                batch["object_points"],
                batch["left_hand_points"],
                batch["right_hand_points"],
            ],
            [object_feat, left_feat, right_feat],
            self._grid_size,
        )
        pooled = [
            value.max(1).values for value in (object_output, left_output, right_output)
        ]
        action_feature = self.action_embedding(
            torch.cat([noisy_left_action, noisy_right_action], dim=-1)
        )
        readout = torch.cat(pooled + [action_feature, time_feature[:, 0]], dim=-1)
        return {
            "left_velocity": self.left_action_head(readout),
            "right_velocity": self.right_action_head(readout),
            "left_noisy_hand_flow": left_flow,
            "right_noisy_hand_flow": right_flow,
        }

    @torch.no_grad()
    def sample(
        self,
        batch: Dict[str, torch.Tensor],
        initial_left: torch.Tensor,
        initial_right: torch.Tensor,
        steps: int,
    ) -> Dict[str, torch.Tensor]:
        left, right = initial_left, initial_right
        step_size = 1.0 / steps
        for index in range(steps):
            tau = torch.full(
                (len(left),), index / steps, device=left.device, dtype=left.dtype
            )
            velocity = self(batch, left, right, tau)
            left = left + step_size * velocity["left_velocity"]
            right = right + step_size * velocity["right_velocity"]
        return {"left_action": left, "right_action": right}

    def load_pointworld_checkpoint(self, path: str) -> Dict[str, object]:
        return load_pointworld_backbone(self, path)
