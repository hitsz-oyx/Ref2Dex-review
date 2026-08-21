from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.base.checkpoint import load_checkpoint
from src.base.base_config import task_config_from_dict
from src.task.Cm.model import CmFlowModel


class CmDecoderModel(nn.Module):
    """Frozen GRAB-Cm encoder followed by a small q-to-next-q decoder."""

    def __init__(self, cfg: Any, *, condition_shape: Any = None, target_shape: Any = None) -> None:
        del condition_shape, target_shape
        super().__init__()
        ckpt = load_checkpoint(cfg.meta.cm_checkpoint, map_location="cpu")
        cm_cfg = task_config_from_dict(ckpt["config"])
        self.cm = CmFlowModel(cm_cfg)
        self.cm.load_state_dict(ckpt["model"], strict=True)
        self.cm.eval()
        for parameter in self.cm.parameters():
            parameter.requires_grad_(False)
        self.decoder_input = str(getattr(cfg.meta, "decoder_input", "qt_cm"))
        if self.decoder_input not in {"qt_cm", "qt_only", "cm_only"}:
            raise ValueError(f"Unsupported decoder_input: {self.decoder_input}")
        self.flow_mode = str(getattr(cfg.meta, "flow_mode", "normal"))
        if self.flow_mode not in {"normal", "shuffled"}:
            raise ValueError(f"Unsupported flow_mode: {self.flow_mode}")
        self.q_input_scale = float(getattr(cfg.meta, "q_input_scale", 1.0))
        self.q_target_scale = float(getattr(cfg.meta, "q_target_scale", 1.0))
        cm_dim = int(self.cm.num_cm_tokens * self.cm.cm_dim) if self.decoder_input != "qt_only" else 0
        input_dim = cm_dim + (6 if self.decoder_input != "cm_only" else 0)
        self.decoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 6),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.cm.eval()
        return self

    @torch.no_grad()
    def encode_cm(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "cm_tokens" in batch:
            if self.flow_mode != "normal":
                raise ValueError("flow_mode=shuffled requires online Cm encoding, not cached cm_tokens")
            return batch["cm_tokens"]
        self.cm.eval()
        cm_batch = batch
        if self.flow_mode == "shuffled":
            cm_batch = dict(batch)
            cm_batch["hand_flow"] = batch["hand_flow"][torch.randperm(batch["hand_flow"].shape[0], device=batch["hand_flow"].device)]
        return self.cm(cm_batch)["cm_tokens"]

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = []
        cm_tokens = None
        if self.decoder_input != "qt_only":
            cm_tokens = self.encode_cm(batch)
            features.append(cm_tokens.flatten(1))
        if self.decoder_input != "cm_only":
            features.append(batch["q_t"] * self.q_input_scale)
        pred_q_next_scaled = self.decoder(torch.cat(features, dim=-1))
        pred_q_next = pred_q_next_scaled / self.q_target_scale
        return {"pred_q_next": pred_q_next, "pred_q_next_scaled": pred_q_next_scaled, "cm_tokens": cm_tokens}
