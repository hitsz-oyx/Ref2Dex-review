"""Temporal-D2 decoder: unordered Cm token windows to Inspire references."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.base import load_config
from src.task.ObjectInteractionCm.model import ObjectInteractionCmModel

from .pointflow import DifferentiableInspireSurface, make_relative_transform, world_to_object


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class TemporalD2Core(nn.Module):
    """Cross-attend future-step/state queries to a KxS interaction token set.

    There is intentionally no slot-index embedding: permuting slots within a
    frame leaves the result unchanged, while time embeddings preserve order.
    """

    def __init__(
        self,
        *,
        window_size: int = 4,
        num_slots: int = 16,
        cm_dim: int = 32,
        link_count: int = 18,
        link_feature_dim: int = 10,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.link_count = int(link_count)
        self.token_projection = _mlp(cm_dim + 6, hidden_dim, hidden_dim)
        self.state_projection = _mlp(15, hidden_dim, hidden_dim)
        self.link_projection = _mlp(link_feature_dim, hidden_dim, hidden_dim)
        self.time_embedding = nn.Embedding(window_size, hidden_dim)
        self.future_embedding = nn.Embedding(window_size, hidden_dim)
        self.link_embedding = nn.Embedding(link_count, hidden_dim)
        self.summary_type = nn.Parameter(torch.zeros(hidden_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim))
        self.readout = _mlp(hidden_dim * 2, hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 6)
        self.translation_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 3)
        for head in (self.q_head, self.translation_head, self.rotation_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        cm_tokens: torch.Tensor,
        anchor_pos: torch.Tensor,
        anchor_normal: torch.Tensor,
        current_state: torch.Tensor,
        current_link_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if cm_tokens.ndim != 4:
            raise ValueError(f"Expected cm_tokens [B,K,S,D], got {tuple(cm_tokens.shape)}")
        batch, window, slots, _ = cm_tokens.shape
        if (window, slots) != (self.window_size, self.num_slots):
            raise ValueError(f"Expected window/slots {(self.window_size, self.num_slots)}, got {(window, slots)}")
        if anchor_pos.shape != (*cm_tokens.shape[:-1], 3) or anchor_normal.shape != anchor_pos.shape:
            raise ValueError("Cm anchor shapes do not match tokens")
        if current_state.shape != (batch, 15):
            raise ValueError(f"Expected current_state [B,15], got {tuple(current_state.shape)}")
        if current_link_features.shape[:2] != (batch, self.link_count):
            raise ValueError(f"Expected {self.link_count} current link queries")
        time_ids = torch.arange(window, device=cm_tokens.device)
        memory = self.token_projection(torch.cat([cm_tokens, anchor_pos, anchor_normal], dim=-1))
        memory = memory + self.time_embedding(time_ids)[None, :, None, :]
        memory = memory.reshape(batch, window * slots, -1)
        future_ids = torch.arange(window, device=cm_tokens.device)
        future = self.future_embedding(future_ids)
        summary = self.state_projection(current_state)[:, None, :] + future[None, :, :] + self.summary_type
        link_ids = torch.arange(self.link_count, device=cm_tokens.device)
        links = self.link_projection(current_link_features) + self.link_embedding(link_ids)[None, :, :]
        links = links[:, None, :, :] + future[None, :, None, :]
        queries = torch.cat([summary[:, :, None, :], links], dim=2).reshape(batch, window * (self.link_count + 1), -1)
        decoded = self.decoder(tgt=queries, memory=memory).reshape(batch, window, self.link_count + 1, -1)
        horizon_feature = self.readout(torch.cat([decoded[:, :, 0], decoded[:, :, 1:].mean(dim=2)], dim=-1))
        return {
            "pred_q_delta": self.q_head(horizon_feature),
            "pred_wrist_translation": self.translation_head(horizon_feature),
            "pred_wrist_rotvec": self.rotation_head(horizon_feature),
        }


class CmDecoderV2(nn.Module):
    """Frozen ObjectInteractionCm encoder followed by Temporal-D2."""

    def __init__(self, cfg: Any, **_: Any) -> None:
        super().__init__()
        meta = cfg.meta
        self.window_size = int(meta.window_size)
        oicm_config_path = str(cfg.oicm_config)
        oicm_checkpoint_path = Path(str(cfg.oicm_checkpoint))
        if not oicm_checkpoint_path.is_absolute():
            oicm_checkpoint_path = (Path.cwd() / oicm_checkpoint_path).resolve()
        if not oicm_checkpoint_path.is_file():
            raise FileNotFoundError(f"Frozen OICM checkpoint does not exist: {oicm_checkpoint_path}")
        expected_sha256 = str(getattr(cfg, "oicm_checkpoint_sha256", "") or "").strip().lower()
        if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
            raise ValueError(
                "model.oicm_checkpoint_sha256 must be a locked 64-character SHA256; "
                "formal config intentionally remains gated until the OICM run is terminal"
            )
        digest = hashlib.sha256()
        with oicm_checkpoint_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Frozen OICM checkpoint SHA256 changed: expected {expected_sha256}, got {actual_sha256}"
            )
        oicm_cfg = load_config(oicm_config_path)
        oicm_model_cfg = copy.copy(oicm_cfg.model)
        oicm_model_cfg.meta = oicm_cfg.meta
        self.oicm = ObjectInteractionCmModel(oicm_model_cfg)
        checkpoint = torch.load(oicm_checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        self.oicm.load_state_dict(state, strict=bool(getattr(cfg, "strict_oicm", True)))
        self.oicm.requires_grad_(False)
        self.oicm.eval()
        self.oicm_checkpoint = str(oicm_checkpoint_path)
        self.oicm_checkpoint_sha256 = actual_sha256
        self.core = TemporalD2Core(
            window_size=self.window_size,
            num_slots=int(meta.num_cm_tokens),
            cm_dim=int(meta.cm_dim),
            link_count=int(meta.query_link_count),
            hidden_dim=int(cfg.hidden_dim),
            num_heads=int(cfg.num_heads),
            num_layers=int(cfg.num_layers),
            feedforward_dim=int(cfg.feedforward_dim),
            dropout=float(cfg.dropout),
        )
        self.point_surface = DifferentiableInspireSurface(
            getattr(cfg, "surface_urdf", "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"),
            sample_count=int(meta.num_hand_points),
            surface_seed=2024,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.oicm.eval()
        return self

    def _encode_cm_window(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, window = batch["obj_points"].shape[:2]
        if window != self.window_size:
            raise ValueError(f"Expected K={self.window_size}, got K={window}")
        flattened: dict[str, torch.Tensor] = {}
        for key in ("obj_points", "obj_normals", "obj_valid_mask", "hand_points", "hand_normals", "hand_flow", "hand_valid_mask"):
            value = batch[key]
            flattened[key] = value.reshape(batch_size * window, *value.shape[2:])
        with torch.no_grad():
            encoded = self.oicm(flattened)
        def restore(key: str) -> torch.Tensor:
            value = encoded[key]
            return value.reshape(batch_size, window, *value.shape[1:])
        return restore("cm_tokens"), restore("cm_anchor_pos"), restore("cm_anchor_normal"), restore("sample_valid")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cm_tokens, anchor_pos, anchor_normal, sample_valid = self._encode_cm_window(batch)
        current_state = torch.cat([
            batch["current_finger_q"].float(),
            batch["current_wrist_translation_object"].float(),
            batch["current_wrist_rotation_6d_object"].float(),
        ], dim=-1)
        output = self.core(
            cm_tokens.float(), anchor_pos.float(), anchor_normal.float(),
            current_state, batch["current_link_features"].float(),
        )
        output.update({
            "cm_tokens": cm_tokens,
            "cm_anchor_pos": anchor_pos,
            "cm_anchor_normal": anchor_normal,
            "cm_sample_valid": sample_valid,
        })
        # Training batches provide the full current wrist/object pose so the
        # q+wrist outputs can be converted into a differentiable point flow.
        # Qualitative viewers may omit these fields and keep the original
        # q/wrist-only output contract.
        required = {"current_wrist_pose_world", "object_pose_world"}
        if required.issubset(batch):
            pred_q_delta = output["pred_q_delta"]
            pred_wrist_translation = output["pred_wrist_translation"]
            pred_wrist_rotvec = output["pred_wrist_rotvec"]
            batch_size = pred_q_delta.shape[0]
            window = pred_q_delta.shape[1]
            current_finger = batch["current_finger_q"].float()
            current_wrist = batch["current_wrist_pose_world"].float()
            object_pose = batch["object_pose_world"].float()
            current_points_world = self.point_surface(current_finger, current_wrist)
            future_finger = current_finger[:, None, :] + pred_q_delta
            wrist_delta = make_relative_transform(pred_wrist_rotvec, pred_wrist_translation)
            future_wrist = current_wrist[:, None, :, :] @ wrist_delta
            future_points_world = self.point_surface(
                future_finger.reshape(batch_size * window, -1),
                future_wrist.reshape(batch_size * window, 4, 4),
            ).reshape(batch_size, window, -1, 3)
            current_points_object = world_to_object(current_points_world, object_pose)
            future_points_object = world_to_object(future_points_world, object_pose)
            output.update({
                "current_hand_points_object": current_points_object,
                "pred_hand_points_object": future_points_object,
                "pred_hand_flow": future_points_object - current_points_object[:, None],
            })
        return output
