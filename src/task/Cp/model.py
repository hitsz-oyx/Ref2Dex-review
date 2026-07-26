"""Cp effect tokens and a closure generator in the pretrained Cm space."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.base import task_config_from_dict
from src.task.Cm.model import CmFlowModel


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SlotAttention(nn.Module):
    """Deterministic sparse bottleneck over a dense feature field."""

    def __init__(self, dim: int, num_slots: int, num_iterations: int) -> None:
        super().__init__()
        self.num_iterations = int(num_iterations)
        self.slot_init = nn.Parameter(torch.randn(1, num_slots, dim) * dim**-0.5)
        self.norm_inputs = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.to_key = nn.Linear(dim, dim, bias=False)
        self.to_value = nn.Linear(dim, dim, bias=False)
        self.to_query = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.scale = dim**-0.5

    def forward(
        self,
        inputs: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = self.norm_inputs(inputs)
        keys, values = self.to_key(normalized), self.to_value(normalized)
        slots = self.slot_init.expand(inputs.shape[0], -1, -1)
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            if valid_mask.shape != inputs.shape[:2]:
                raise ValueError(
                    f"valid_mask must have shape {tuple(inputs.shape[:2])}, got {tuple(valid_mask.shape)}."
                )
        for _ in range(self.num_iterations):
            logits = torch.einsum("bkd,bnd->bkn", self.to_query(self.norm_slots(slots)), keys)
            logits = logits * self.scale
            if valid_mask is not None:
                logits = logits.masked_fill(~valid_mask[:, None, :], torch.finfo(logits.dtype).min)
            assignment = torch.softmax(logits, dim=1)
            if valid_mask is not None:
                assignment = assignment * valid_mask[:, None, :]
            weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bkn,bnd->bkd", weights, values)
            slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1)).view_as(slots)
            slots = slots + self.mlp(slots)
        return slots, assignment, weights


class FrozenCmHandDecoder(nn.Module):
    """Frozen current-state encoder and Cm-to-hand decoder from a Cm checkpoint."""

    def __init__(self, checkpoint: str | Path, *, dense_checkpoint: str | None = None) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint).resolve()
        payload = torch.load(checkpoint_path, map_location="cpu")
        if "config" not in payload or "model" not in payload:
            raise ValueError(f"Unsupported Cm checkpoint: {checkpoint_path}")
        cm_cfg = task_config_from_dict(payload["config"])
        if dense_checkpoint:
            cm_cfg.meta.dense_checkpoint = str(Path(dense_checkpoint).resolve())
        self.cm_model = CmFlowModel(cm_cfg)
        self.cm_model.load_state_dict(payload["model"], strict=True)
        self.cm_model.requires_grad_(False)
        self.cm_model.eval()
        self.cm_dim = int(self.cm_model.cm_dim)
        self.dense_token_dim = int(self.cm_model.dense_token_dim)
        self.loss_meta = cm_cfg.meta

    @torch.no_grad()
    def encode_current_hand(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        _, z_hand, _ = self.cm_model.dense_encoder(
            obj_points=batch["obj_points"],
            obj_normals=batch["obj_normals"],
            hand_points=batch["hand_points"],
            hand_normals=batch["hand_normals"],
            obj_valid_mask=batch["obj_valid_mask"],
        )
        return z_hand

    def decode(self, cm_tokens: torch.Tensor, z_hand: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.cm_model.eval()
        return self.cm_model.head.decode_hand_from_cm(
            cm_tokens=cm_tokens,
            z_hand=z_hand,
            hand_points=batch["hand_points"],
            hand_normals=batch["hand_normals"],
        )


class CpHumanClosureModel(nn.Module):
    """Cp effect autoencoder plus trainable ``Cp + H_t -> Cm`` generator.

    The closure branch owns its hand-state encoder.  The independent frozen
    ``FrozenCmHandDecoder`` owns the pretrained Cm interpretation protocol.
    """

    def __init__(self, cfg: Any, **_: Any) -> None:
        super().__init__()
        meta = cfg.meta
        self.dim = int(meta.dim)
        self.cp_encoder = make_mlp(10, self.dim, self.dim)
        self.cp_slots = SlotAttention(self.dim, int(meta.num_cp_tokens), int(meta.slot_iters))
        # Predicting sub-millimetre object flow and a contact logit through
        # one final projection causes destructive multi-task gradients. Keep a
        # common Cp/object feature trunk but give the two physical quantities
        # independent heads. A zero flow head is also the correct prior for
        # stationary grasp frames.
        self.cp_effect_decoder = make_mlp(self.dim + 6, self.dim, self.dim)
        self.cp_mixture_head = nn.Linear(self.dim, 1)
        self.cp_contact_head = nn.Linear(self.dim, 1)
        self.cp_flow_head = nn.Linear(self.dim, 3)
        nn.init.zeros_(self.cp_flow_head.weight)
        nn.init.zeros_(self.cp_flow_head.bias)

        self.closure_hand_state = make_mlp(int(meta.cm_dense_token_dim) + 6, self.dim, self.dim)
        self.cp_to_hand = nn.MultiheadAttention(self.dim, num_heads=8, batch_first=True)
        self.closure_cm_slots = SlotAttention(self.dim, int(meta.num_cm_tokens), int(meta.slot_iters))
        self.cm_hand_decoder: FrozenCmHandDecoder | None = None
        if str(meta.stage) == "closed_loop":
            if not meta.cm_hand_checkpoint:
                raise ValueError("closed_loop requires meta.cm_hand_checkpoint.")
            self.cm_hand_decoder = FrozenCmHandDecoder(
                meta.cm_hand_checkpoint,
                dense_checkpoint=meta.cm_dense_checkpoint or None,
            )
            if self.cm_hand_decoder.cm_dim != self.dim:
                raise ValueError(
                    f"Cp dim={self.dim} does not match pretrained Cm dim={self.cm_hand_decoder.cm_dim}."
                )
            if self.cm_hand_decoder.dense_token_dim != int(meta.cm_dense_token_dim):
                raise ValueError("meta.cm_dense_token_dim does not match the Cm checkpoint.")

    def encode_cp(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        effect = torch.cat(
            [batch["obj_points"], batch["obj_normals"], batch["obj_contact_gt"].unsqueeze(-1), batch["obj_flow_gt"]], dim=-1
        )
        return self.cp_slots(self.cp_encoder(effect), valid_mask=batch["obj_valid_mask"])

    def decode_effect(self, cp_tokens: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_obj, num_slots = batch["obj_points"].shape[1], cp_tokens.shape[1]
        tokens = cp_tokens.unsqueeze(1).expand(-1, num_obj, -1, -1)
        points = batch["obj_points"].unsqueeze(2).expand(-1, -1, num_slots, -1)
        normals = batch["obj_normals"].unsqueeze(2).expand(-1, -1, num_slots, -1)
        decoded = self.cp_effect_decoder(torch.cat([tokens, points, normals], dim=-1))
        mixture_logits = self.cp_mixture_head(decoded).squeeze(-1)
        contact_per_slot = self.cp_contact_head(decoded).squeeze(-1)
        flow_per_slot = self.cp_flow_head(decoded)
        slot_weight = torch.softmax(mixture_logits, dim=2)
        contact_logits = (slot_weight * contact_per_slot).sum(dim=2)
        object_flow = (slot_weight.unsqueeze(-1) * flow_per_slot).sum(dim=2)
        return contact_logits, object_flow, slot_weight

    def generate_cm(
        self,
        cp_tokens: torch.Tensor,
        z_hand: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hand_input = torch.cat([z_hand, batch["hand_points"], batch["hand_normals"]], dim=-1)
        hand_state = self.closure_hand_state(hand_input)
        task_state, _ = self.cp_to_hand(hand_state, cp_tokens, cp_tokens, need_weights=False)
        return self.closure_cm_slots(hand_state + task_state)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cp_tokens, cp_assignment, cp_weights = self.encode_cp(batch)
        contact_logits, object_flow, effect_slot_weight = self.decode_effect(cp_tokens, batch)
        output = {
            "cp_tokens": cp_tokens,
            "cp_assignment": cp_assignment,
            "cp_slot_weights": cp_weights,
            "pred_obj_contact_logits": contact_logits,
            "pred_obj_flow": object_flow,
            "cp_effect_slot_weight": effect_slot_weight,
        }
        if self.cm_hand_decoder is not None:
            z_hand = self.cm_hand_decoder.encode_current_hand(batch)
            cm_tokens, cm_assignment, cm_weights = self.generate_cm(cp_tokens, z_hand, batch)
            hand_prediction = self.cm_hand_decoder.decode(cm_tokens, z_hand, batch)
            output.update(
                {
                    "pred_cm_tokens": cm_tokens,
                    "pred_cm_assignment": cm_assignment,
                    "pred_cm_slot_weights": cm_weights,
                    **hand_prediction,
                }
            )
        return output

    def load_cp_effect(self, path: str | Path) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        state = payload.get("model", payload)
        prefixes = (
            "cp_encoder.",
            "cp_slots.",
            "cp_effect_decoder.",
            "cp_mixture_head.",
            "cp_contact_head.",
            "cp_flow_head.",
        )
        own = self.state_dict()
        selected = {key: value for key, value in state.items() if key in own and key.startswith(prefixes)}
        required_prefixes = tuple(prefixes)
        missing_prefixes = [prefix for prefix in required_prefixes if not any(key.startswith(prefix) for key in selected)]
        if missing_prefixes:
            raise ValueError(
                f"{path} does not contain a complete Cp effect checkpoint; missing {missing_prefixes}. "
                "Retrain the Cp effect stage with the current architecture."
            )
        self.load_state_dict(selected, strict=False)
        for name, parameter in self.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(False)
