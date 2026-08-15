from __future__ import annotations

from argparse import Namespace
from typing import Dict

import torch
import torch.nn as nn

from .pointworld_forward import AdapterMLP, import_upstream_dynamics_predictor


class GRABPointWorldOneStep(nn.Module):
    """局部 right-hand point flow -> object point flow。"""

    def __init__(self, cfg) -> None:
        super().__init__()
        channels = cfg.predictor_dim
        args = Namespace(
            ptv3_size=cfg.ptv3_size,
            ptv3_patch_size=cfg.ptv3_patch_size,
            grid_size=cfg.grid_size,
            enable_rpe=cfg.enable_rpe,
            drop_path=cfg.drop_path,
            shuffle_orders=cfg.shuffle_orders,
            dynamics_head_init_scale=1.0,
        )
        dynamics_cls = import_upstream_dynamics_predictor(cfg.pointworld_root, True)
        template = dynamics_cls(args, channels, T=11)
        self.predictor_model = template.predictor_model
        self.skip_film_gamma = template.skip_film_gamma
        self.skip_film_beta = template.skip_film_beta
        self.hand_film_gamma = template.robot_film_gamma
        self.hand_film_beta = template.robot_film_beta
        self.register_buffer("_grid_size", torch.tensor([cfg.grid_size]), persistent=False)
        self.scene_adapter = AdapterMLP(7, channels)
        self.hand_adapter = AdapterMLP(9, channels)
        self.hand_type_emb = nn.Parameter(torch.zeros(1, 1, channels))
        nn.init.normal_(self.hand_type_emb, std=0.02)
        self.one_step_dynamics_head = nn.Sequential(
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        nn.init.kaiming_normal_(self.one_step_dynamics_head[-1].weight)
        nn.init.zeros_(self.one_step_dynamics_head[-1].bias)

    def forward(
        self,
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_flow: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        distance = torch.cdist(object_points, hand_points).amin(dim=-1, keepdim=True)
        scene_feat = self.scene_adapter(
            torch.cat([object_points, object_normals, distance], dim=-1)
        )
        hand_feat = self.hand_adapter(
            torch.cat([hand_points, hand_normals, hand_flow], dim=-1)
        ) + self.hand_type_emb
        batch_size, scene_points, channels = scene_feat.shape
        hand_points_count = hand_feat.shape[1]
        coord = torch.cat([object_points, hand_points], dim=1)
        feat = torch.cat([scene_feat, hand_feat], dim=1)
        batch = torch.arange(batch_size, device=coord.device).repeat_interleave(
            scene_points + hand_points_count
        )
        is_hand = torch.cat(
            [
                torch.zeros(batch_size, scene_points, dtype=torch.bool, device=coord.device),
                torch.ones(batch_size, hand_points_count, dtype=torch.bool, device=coord.device),
            ],
            dim=1,
        ).flatten()
        point = self.predictor_model(
            {
                "coord": coord.flatten(0, 1),
                "feat": feat.flatten(0, 1),
                "batch": batch,
                "is_robot": is_hand,
                "grid_size": self._grid_size,
            }
        )
        scene_output = point.feat[~is_hand].view(batch_size, scene_points, channels)
        hand_output = point.feat[is_hand].view(batch_size, hand_points_count, channels)
        hand_summary = hand_output.max(dim=1, keepdim=True).values.expand(
            -1, scene_points, -1
        )
        fused = (
            scene_output
            + scene_feat * self.skip_film_gamma
            + self.skip_film_beta
            + hand_summary * self.hand_film_gamma
            + self.hand_film_beta
        )
        return {"object_flow": self.one_step_dynamics_head(fused)}

    def load_pointworld_checkpoint(self, path: str) -> Dict[str, object]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        own = self.state_dict()
        selected = {}
        film_mapping = {
            "skip_film_gamma": "skip_film_gamma",
            "skip_film_beta": "skip_film_beta",
            "robot_film_gamma": "hand_film_gamma",
            "robot_film_beta": "hand_film_beta",
        }
        for raw_key, value in state.items():
            key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
            marker = "dynamics_predictor."
            if marker not in key:
                continue
            suffix = key[key.index(marker) + len(marker) :]
            if suffix.startswith("predictor_model."):
                target = suffix
            elif suffix in film_mapping:
                target = film_mapping[suffix]
            else:
                continue
            if target in own and own[target].shape == value.shape:
                selected[target] = value
        missing, unexpected = self.load_state_dict(selected, strict=False)
        backbone_loaded = sum(key.startswith("predictor_model.") for key in selected)
        missing_rpe = [key for key in missing if key.startswith("predictor_model.") and ".rpe." in key]
        missing_pretrained_non_rpe = [
            key
            for key in missing
            if (
                key.startswith("predictor_model.")
                or key in film_mapping.values()
            )
            and ".rpe." not in key
        ]
        if backbone_loaded == 0 or missing_pretrained_non_rpe:
            raise RuntimeError(
                f"PointWorld backbone 加载不完整: loaded={backbone_loaded}, "
                f"missing={missing_pretrained_non_rpe}"
            )
        return {
            "loaded": len(selected),
            "backbone_loaded": backbone_loaded,
            "missing_rpe": len(missing_rpe),
            "missing_pretrained_non_rpe": missing_pretrained_non_rpe,
            "unexpected": len(unexpected),
        }
