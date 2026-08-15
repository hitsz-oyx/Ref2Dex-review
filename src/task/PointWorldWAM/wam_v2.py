from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn

from .modules.action_readout import ActionReadout
from .modules.spatial_queries import SpatialQueryPool
from .modules.subject_mano_bank import SubjectManoBank
from .modules.world_action_transformer import WorldActionTransformer
from .pointworld_bimanual_forward import (
    build_predictor,
    load_pointworld_backbone,
    run_joint_backbone,
)
from .pointworld_forward import AdapterMLP


class ScalarEmbedding(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(32) * math.pi)
        self.mlp = nn.Sequential(
            nn.Linear(64, channels), nn.SiLU(), nn.Linear(channels, channels)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        phase = value[:, None] * self.frequencies[None]
        return self.mlp(torch.cat([phase.sin(), phase.cos()], dim=-1))


class ChunkJointWAM(nn.Module):
    """Unified chunk model; forward/inverse differ only in world/action noise time."""

    def __init__(
        self,
        cfg,
        statistics: Dict[str, torch.Tensor],
        subjects: Sequence[str],
    ) -> None:
        super().__init__()
        channels = cfg.predictor_dim
        temporal_dim = cfg.temporal_dim
        self.chunk_size = cfg.chunk_size
        self.predictor_model = build_predictor(cfg)
        self.register_buffer(
            "_grid_size", torch.tensor([cfg.grid_size]), persistent=False
        )
        for key, value in statistics.items():
            self.register_buffer(key, value.clone())
        self.mano = SubjectManoBank(
            cfg.mano_model_dir, cfg.subject_template_root, subjects
        )
        self.object_adapter = AdapterMLP(12, channels)
        self.left_adapter = AdapterMLP(12, channels)
        self.right_adapter = AdapterMLP(12, channels)
        self.object_type_embedding = nn.Parameter(torch.empty(1, 1, channels))
        self.left_type_embedding = nn.Parameter(torch.empty(1, 1, channels))
        self.right_type_embedding = nn.Parameter(torch.empty(1, 1, channels))
        self.finger_embedding = nn.Embedding(cfg.finger_classes, channels)
        self.region_embedding = nn.Embedding(cfg.region_classes, channels)
        self.physical_time_embedding = nn.Embedding(cfg.chunk_size, channels)
        self.world_noise_embedding = ScalarEmbedding(channels)
        self.action_noise_embedding = ScalarEmbedding(channels)
        self.left_state_embedding = AdapterMLP(30, channels)
        self.right_state_embedding = AdapterMLP(30, channels)
        self.left_noisy_action_embedding = AdapterMLP(30, channels)
        self.right_noisy_action_embedding = AdapterMLP(30, channels)
        self.object_queries = SpatialQueryPool(
            channels, cfg.object_queries, cfg.query_heads
        )
        self.left_queries = SpatialQueryPool(channels, cfg.hand_queries, cfg.query_heads)
        self.right_queries = SpatialQueryPool(
            channels, cfg.hand_queries, cfg.query_heads
        )
        self.temporal = WorldActionTransformer(
            channels,
            temporal_dim,
            cfg.temporal_depth,
            cfg.temporal_heads,
            cfg.chunk_size,
        )
        self.world_context = nn.Linear(temporal_dim, channels)
        self.left_action_readout = ActionReadout(temporal_dim, cfg.temporal_heads)
        self.right_action_readout = ActionReadout(temporal_dim, cfg.temporal_heads)
        self.world_head = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, 3)
        )
        action_readout_dim = temporal_dim + 4 * channels
        self.left_action_head = nn.Sequential(
            nn.Linear(action_readout_dim, 256), nn.GELU(), nn.Linear(256, 30)
        )
        self.right_action_head = nn.Sequential(
            nn.Linear(action_readout_dim, 256), nn.GELU(), nn.Linear(256, 30)
        )
        for embedding in (
            self.object_type_embedding,
            self.left_type_embedding,
            self.right_type_embedding,
        ):
            nn.init.normal_(embedding, std=0.02)

    def normalize_world(self, world: torch.Tensor) -> torch.Tensor:
        return (world - self.world_mean) / self.world_std

    def denormalize_world(self, world: torch.Tensor) -> torch.Tensor:
        return world * self.world_std + self.world_mean

    def normalize_action(self, side: str, action: torch.Tensor) -> torch.Tensor:
        return (action - getattr(self, f"{side}_mean")) / getattr(
            self, f"{side}_std"
        )

    def denormalize_action(self, side: str, action: torch.Tensor) -> torch.Tensor:
        return action * getattr(self, f"{side}_std") + getattr(
            self, f"{side}_mean"
        )

    def normalized_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "world": self.normalize_world(batch["world_chunk"]),
            "left": self.normalize_action("left", batch["left_action_chunk"]),
            "right": self.normalize_action("right", batch["right_action_chunk"]),
        }

    def hand_points_from_normalized_action(
        self, batch: Dict[str, torch.Tensor], side: str, action: torch.Tensor
    ) -> torch.Tensor:
        return self.mano.points_from_action_chunk(
            batch["subject_id"],
            side,
            batch[f"{side}_global_orient"],
            batch[f"{side}_hand_pose"],
            batch[f"{side}_transl"],
            batch[f"{side}_betas"],
            self.denormalize_action(side, action),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_world: torch.Tensor,
        noisy_left_action: torch.Tensor,
        noisy_right_action: torch.Tensor,
        tau_world: torch.Tensor,
        tau_action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, chunk_size = noisy_world.shape[:2]
        if chunk_size != self.chunk_size:
            raise ValueError(f"chunk size {chunk_size} != configured {self.chunk_size}")
        world_flow = self.denormalize_world(noisy_world)
        noisy_hand_points = {
            side: self.hand_points_from_normalized_action(batch, side, action)
            for side, action in (
                ("left", noisy_left_action),
                ("right", noisy_right_action),
            )
        }
        object_points = batch["object_points"][:, None].expand(-1, chunk_size, -1, -1)
        coordinate_scale = tau_world[:, None, None, None]
        candidate_object = object_points + coordinate_scale * world_flow
        time_indices = torch.arange(chunk_size, device=noisy_world.device)
        time_embedding = self.physical_time_embedding(time_indices)[None, :, None]
        world_noise = self.world_noise_embedding(tau_world)[:, None, None]
        action_noise = self.action_noise_embedding(tau_action)[:, None, None]
        object_normals = batch["object_normals"][:, None].expand(
            -1, chunk_size, -1, -1
        )
        previous_flow = batch["prev_object_flow"][:, None].expand(
            -1, chunk_size, -1, -1
        )
        object_features = self.object_adapter(
            torch.cat(
                [object_points, object_normals, previous_flow, world_flow], dim=-1
            )
        )
        object_features = (
            object_features
            + self.object_type_embedding
            + time_embedding
            + world_noise
        )
        hand_features = {}
        hand_conditioning = {}
        state_embeddings = {}
        noisy_embeddings = {}
        for side, adapter, type_embedding, noisy_action in (
            ("left", self.left_adapter, self.left_type_embedding, noisy_left_action),
            ("right", self.right_adapter, self.right_type_embedding, noisy_right_action),
        ):
            current = batch[f"{side}_hand_points"][:, None].expand(
                -1, chunk_size, -1, -1
            )
            canonical = batch[f"{side}_hand_cano_points"][:, None].expand(
                -1, chunk_size, -1, -1
            )
            normals = batch[f"{side}_hand_normals"][:, None].expand(
                -1, chunk_size, -1, -1
            )
            candidate = noisy_hand_points[side]
            features = adapter(
                torch.cat([current, canonical, normals, candidate - current], dim=-1)
            )
            identity = self.finger_embedding(batch[f"{side}_finger_id"]) + self.region_embedding(
                batch[f"{side}_region_id"]
            )
            features = (
                features
                + identity[:, None]
                + type_embedding
                + time_embedding
                + action_noise
            )
            state = torch.cat(
                [
                    batch[f"{side}_transl"],
                    batch[f"{side}_global_orient"],
                    batch[f"{side}_hand_pose"],
                ],
                dim=-1,
            )
            state_embedding = getattr(self, f"{side}_state_embedding")(state)
            noisy_embedding = getattr(self, f"{side}_noisy_action_embedding")(
                noisy_action.flatten(0, 1)
            ).reshape(batch_size, chunk_size, -1)
            hand_features[side] = features
            state_embeddings[side] = state_embedding
            noisy_embeddings[side] = noisy_embedding
            hand_conditioning[side] = (
                state_embedding[:, None] + noisy_embedding + action_noise[:, :, 0]
            )
        flat = lambda value: value.flatten(0, 1)
        object_output, left_output, right_output = run_joint_backbone(
            self.predictor_model,
            [
                flat(candidate_object),
                flat(noisy_hand_points["left"]),
                flat(noisy_hand_points["right"]),
            ],
            [
                flat(object_features),
                flat(hand_features["left"]),
                flat(hand_features["right"]),
            ],
            self._grid_size,
        )
        spatial_outputs = {
            "object": object_output.reshape(batch_size, chunk_size, *object_output.shape[1:]),
            "left": left_output.reshape(batch_size, chunk_size, *left_output.shape[1:]),
            "right": right_output.reshape(batch_size, chunk_size, *right_output.shape[1:]),
        }
        compact = {
            "object": self.object_queries(object_output).reshape(
                batch_size, chunk_size, -1, object_output.shape[-1]
            ),
            "left": self.left_queries(
                left_output, flat(hand_conditioning["left"])
            ).reshape(batch_size, chunk_size, -1, left_output.shape[-1]),
            "right": self.right_queries(
                right_output, flat(hand_conditioning["right"])
            ).reshape(batch_size, chunk_size, -1, right_output.shape[-1]),
        }
        temporal = self.temporal(compact["object"], compact["left"], compact["right"])
        world_context = self.world_context(temporal["object"].mean(2))[:, :, None]
        world_features = (
            spatial_outputs["object"] + object_features + world_context
        )
        world_velocity = self.world_head(world_features)
        all_temporal = torch.cat(
            [temporal["object"], temporal["left"], temporal["right"]], dim=2
        )
        action_context = {
            "left": self.left_action_readout(all_temporal),
            "right": self.right_action_readout(all_temporal),
        }
        tau_action_embedding = self.action_noise_embedding(tau_action)[:, None].expand(
            -1, chunk_size, -1
        )
        time_readout = self.physical_time_embedding(time_indices)[None].expand(
            batch_size, -1, -1
        )
        action_velocity = {}
        for side in ("left", "right"):
            readout = torch.cat(
                [
                    action_context[side],
                    state_embeddings[side][:, None].expand(-1, chunk_size, -1),
                    noisy_embeddings[side],
                    tau_action_embedding,
                    time_readout,
                ],
                dim=-1,
            )
            action_velocity[side] = getattr(self, f"{side}_action_head")(readout)
        return {
            "world_velocity": world_velocity,
            "left_action_velocity": action_velocity["left"],
            "right_action_velocity": action_velocity["right"],
            "left_noisy_hand_points": noisy_hand_points["left"],
            "right_noisy_hand_points": noisy_hand_points["right"],
        }

    def load_pointworld_checkpoint(self, path: str) -> Dict[str, object]:
        return load_pointworld_backbone(self, path)
