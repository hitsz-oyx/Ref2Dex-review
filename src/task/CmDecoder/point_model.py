"""Per-hand-point Cm decoder used before kinematic q fitting."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.task.Cm.src.model import CmFlowModel


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
        nn.GELU(),
    )


class CmPointFlowModel(nn.Module):
    """Apply frozen Cm slots to every current hand point and predict point flow.

    The trainable path never consumes ``q_t``. Current hand geometry, frozen
    DenseToken hand features/contact, and frozen Cm action tokens are decoded
    into one 3-D flow vector for each of the 1538 corresponding hand points.
    """

    def __init__(self, cfg: Any, *, condition_shape: Any = None, target_shape: Any = None) -> None:
        del condition_shape, target_shape
        super().__init__()
        checkpoint = load_checkpoint(cfg.meta.cm_checkpoint, map_location="cpu")
        cm_cfg = task_config_from_dict(checkpoint["config"])
        self.cm = CmFlowModel(cm_cfg)
        self.cm.load_state_dict(checkpoint["model"], strict=True)
        self.cm.eval()
        for parameter in self.cm.parameters():
            parameter.requires_grad_(False)

        self.decoder_input = "hand_points_cm"
        self.flow_mode = str(getattr(cfg.meta, "flow_mode", "normal"))
        if self.flow_mode not in {"normal", "shuffled"}:
            raise ValueError(f"Unsupported flow_mode: {self.flow_mode}")
        self.point_flow_target_scale = float(getattr(cfg.meta, "point_flow_target_scale", 1.0))
        if self.point_flow_target_scale <= 0.0:
            raise ValueError("point_flow_target_scale must be positive")

        cm_dim = int(self.cm.cm_dim)
        dense_dim = int(self.cm.dense_token_dim)
        # z_hand + xyz + normal + frozen contact prior. q_t is deliberately absent.
        self.hand_context_encoder = _mlp(dense_dim + 7, cm_dim, cm_dim)
        self.edge_backbone = nn.Sequential(
            nn.Linear(cm_dim * 2, cm_dim),
            nn.GELU(),
            nn.Linear(cm_dim, max(32, cm_dim // 2)),
            nn.GELU(),
        )
        edge_dim = max(32, cm_dim // 2)
        self.edge_logit_head = nn.Linear(edge_dim, 1)
        self.edge_flow_head = nn.Linear(edge_dim, 3)
        nn.init.normal_(self.edge_logit_head.weight, std=1e-3)
        nn.init.zeros_(self.edge_logit_head.bias)
        nn.init.zeros_(self.edge_flow_head.weight)
        nn.init.zeros_(self.edge_flow_head.bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.cm.eval()
        return self

    @torch.no_grad()
    def _frozen_features(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_obj, z_hand, hand_contact = self.cm.dense_encoder(
            obj_points=batch["obj_points"],
            obj_normals=batch["obj_normals"],
            hand_points=batch["hand_points"],
            hand_normals=batch["hand_normals"],
            obj_valid_mask=batch["obj_valid_mask"],
        )
        if "cm_tokens" in batch:
            if self.flow_mode != "normal":
                raise ValueError("flow_mode=shuffled cannot reuse normal-flow cached Cm tokens")
            cm_tokens = batch["cm_tokens"]
        else:
            cm_batch = batch
            if self.flow_mode == "shuffled":
                cm_batch = dict(batch)
                permutation = torch.randperm(batch["hand_flow"].shape[0], device=batch["hand_flow"].device)
                cm_batch["hand_flow"] = batch["hand_flow"][permutation]
            # This path also computes the frozen object-flow head, but keeps the
            # Cm checkpoint as the single source of action-token semantics.
            cm_tokens = self.cm(cm_batch)["cm_tokens"]
        return z_hand, hand_contact, cm_tokens

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_hand, hand_contact, cm_tokens = self._frozen_features(batch)
        geometry_scale = float(self.cm.head.geometry_input_scale)
        hand_context = self.hand_context_encoder(
            torch.cat(
                [
                    z_hand,
                    batch["hand_points"] * geometry_scale,
                    batch["hand_normals"],
                    hand_contact.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        num_hand = hand_context.shape[1]
        num_slots = cm_tokens.shape[1]
        edge_input = torch.cat(
            [
                hand_context.unsqueeze(2).expand(-1, -1, num_slots, -1),
                cm_tokens.unsqueeze(1).expand(-1, num_hand, -1, -1),
            ],
            dim=-1,
        )
        edge_features = self.edge_backbone(edge_input)
        edge_logits = self.edge_logit_head(edge_features).squeeze(-1)
        edge_weight = torch.softmax(edge_logits, dim=2)
        candidate_flow_scaled = self.edge_flow_head(edge_features)
        pred_hand_flow_scaled = (edge_weight.unsqueeze(-1) * candidate_flow_scaled).sum(dim=2)
        pred_hand_flow = pred_hand_flow_scaled / self.point_flow_target_scale
        return {
            "pred_hand_flow": pred_hand_flow,
            "pred_hand_flow_scaled": pred_hand_flow_scaled,
            "pred_hand_points_next": batch["hand_points"] + pred_hand_flow,
            "cm_tokens": cm_tokens,
            "hand_point_slot_weight": edge_weight,
            "hand_point_candidate_flow": candidate_flow_scaled / self.point_flow_target_scale,
        }
