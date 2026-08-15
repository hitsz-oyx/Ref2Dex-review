from __future__ import annotations

from argparse import Namespace
from typing import Dict, List

import torch
import torch.nn as nn

from .pointworld_forward import AdapterMLP, import_upstream_dynamics_predictor


def run_joint_backbone(
    predictor: nn.Module,
    coordinates: List[torch.Tensor],
    features: List[torch.Tensor],
    grid_size: torch.Tensor,
) -> List[torch.Tensor]:
    if len(coordinates) != len(features):
        raise ValueError("coordinates/features token group 数量不一致")
    batch_size = coordinates[0].shape[0]
    lengths = [value.shape[1] for value in coordinates]
    coord = torch.cat(coordinates, dim=1)
    feat = torch.cat(features, dim=1)
    batch = torch.arange(batch_size, device=coord.device).repeat_interleave(sum(lengths))
    is_hand = torch.cat(
        [
            torch.zeros(batch_size, lengths[0], dtype=torch.bool, device=coord.device),
            torch.ones(batch_size, sum(lengths[1:]), dtype=torch.bool, device=coord.device),
        ],
        dim=1,
    ).flatten()
    output = predictor(
        {
            "coord": coord.flatten(0, 1),
            "feat": feat.flatten(0, 1),
            "batch": batch,
            "is_robot": is_hand,
            "grid_size": grid_size,
        }
    ).feat.view(batch_size, sum(lengths), -1)
    return list(torch.split(output, lengths, dim=1))


def load_pointworld_backbone(module: nn.Module, path: str) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint
    for key in ("model", "state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
            break
    if not isinstance(state, dict):
        raise TypeError(f"无法识别 PointWorld checkpoint: {path}")
    own = module.state_dict()
    selected = {}
    marker = "dynamics_predictor.predictor_model."
    for raw_key, value in state.items():
        key = raw_key[len("module.") :] if raw_key.startswith("module.") else raw_key
        if marker not in key:
            continue
        target = "predictor_model." + key.split(marker, 1)[1]
        if target in own and own[target].shape == value.shape:
            selected[target] = value
    missing, unexpected = module.load_state_dict(selected, strict=False)
    missing_rpe = [key for key in missing if key.startswith("predictor_model.") and ".rpe." in key]
    missing_pretrained_non_rpe = [
        key
        for key in missing
        if key.startswith("predictor_model.") and ".rpe." not in key
    ]
    if not selected or missing_pretrained_non_rpe:
        raise RuntimeError(
            f"PointWorld backbone 加载不完整: loaded={len(selected)}, "
            f"missing={missing_pretrained_non_rpe}"
        )
    return {
        "backbone_loaded": len(selected),
        "missing_rpe": len(missing_rpe),
        "missing_pretrained_non_rpe": missing_pretrained_non_rpe,
        "unexpected": len(unexpected),
    }


def build_predictor(cfg):
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
    return dynamics_cls(args, cfg.predictor_dim, T=11).predictor_model


class GRABPointWorldBimanualForward(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        channels = cfg.predictor_dim
        self.predictor_model = build_predictor(cfg)
        self.register_buffer("_grid_size", torch.tensor([cfg.grid_size]), persistent=False)
        self.object_adapter = AdapterMLP(11, channels)
        self.left_hand_adapter = AdapterMLP(9, channels)
        self.right_hand_adapter = AdapterMLP(9, channels)
        self.object_type_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.left_type_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.right_type_emb = nn.Parameter(torch.zeros(1, 1, channels))
        self.left_film_gamma = nn.Parameter(torch.ones(1, 1, channels))
        self.left_film_beta = nn.Parameter(torch.zeros(1, 1, channels))
        self.right_film_gamma = nn.Parameter(torch.ones(1, 1, channels))
        self.right_film_beta = nn.Parameter(torch.zeros(1, 1, channels))
        self.object_flow_head = nn.Sequential(
            nn.Linear(channels, 128), nn.GELU(), nn.Linear(128, 3)
        )
        for embedding in (self.object_type_emb, self.left_type_emb, self.right_type_emb):
            nn.init.normal_(embedding, std=0.02)

    def forward(
        self,
        object_points: torch.Tensor,
        object_normals: torch.Tensor,
        prev_object_flow: torch.Tensor,
        left_hand_points: torch.Tensor,
        left_hand_normals: torch.Tensor,
        left_hand_flow: torch.Tensor,
        right_hand_points: torch.Tensor,
        right_hand_normals: torch.Tensor,
        right_hand_flow: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        distance_left = torch.cdist(object_points, left_hand_points).amin(-1, keepdim=True)
        distance_right = torch.cdist(object_points, right_hand_points).amin(-1, keepdim=True)
        object_feat = self.object_adapter(
            torch.cat(
                [
                    object_points,
                    object_normals,
                    distance_left,
                    distance_right,
                    prev_object_flow,
                ],
                dim=-1,
            )
        ) + self.object_type_emb
        left_feat = self.left_hand_adapter(
            torch.cat([left_hand_points, left_hand_normals, left_hand_flow], dim=-1)
        ) + self.left_type_emb
        right_feat = self.right_hand_adapter(
            torch.cat([right_hand_points, right_hand_normals, right_hand_flow], dim=-1)
        ) + self.right_type_emb
        object_output, left_output, right_output = run_joint_backbone(
            self.predictor_model,
            [object_points, left_hand_points, right_hand_points],
            [object_feat, left_feat, right_feat],
            self._grid_size,
        )
        left_summary = left_output.max(1, keepdim=True).values
        right_summary = right_output.max(1, keepdim=True).values
        fused = (
            object_output
            + object_feat
            + left_summary * self.left_film_gamma
            + self.left_film_beta
            + right_summary * self.right_film_gamma
            + self.right_film_beta
        )
        return {"object_flow": self.object_flow_head(fused)}

    def load_pointworld_checkpoint(self, path: str) -> Dict[str, object]:
        return load_pointworld_backbone(self, path)
