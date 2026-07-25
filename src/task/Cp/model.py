"""Future-leak-free modules for the three-stage Cp human closure."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class SlotAttention(nn.Module):
    """Deterministic sparse bottleneck over a dense point feature field."""

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

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = self.norm_inputs(inputs)
        keys, values = self.to_key(inputs), self.to_value(inputs)
        slots = self.slot_init.expand(inputs.shape[0], -1, -1)

        for _ in range(self.num_iterations):
            logits = torch.einsum("bkd,bnd->bkn", self.to_query(self.norm_slots(slots)), keys)
            assignment = torch.softmax(logits * self.scale, dim=1)
            weights = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            updates = torch.einsum("bkn,bnd->bkd", weights, values)
            slots = self.gru(updates.flatten(0, 1), slots.flatten(0, 1)).view_as(slots)
            slots = slots + self.mlp(slots)
        return slots, assignment, weights


class CpHumanClosureModel(nn.Module):
    """Cp effect model, Cm teacher/decoder, and current-hand closure generator.

    ``hand_flow`` is only used by ``teacher_cm`` during Stage A.  Neither
    hand-flow decoder nor closed-loop generator receives a future variable.
    """

    def __init__(self, cfg: Any, **_: Any) -> None:
        super().__init__()
        meta = cfg.meta
        self.dim = int(meta.dim)
        cp_slots = int(meta.num_cp_tokens)
        cm_slots = int(meta.num_cm_tokens)
        slot_iters = int(meta.slot_iters)

        # Stage B: [position, normal, contact, object flow] -> Cp -> effect.
        self.cp_encoder = make_mlp(10, self.dim, self.dim)
        self.cp_slots = SlotAttention(self.dim, cp_slots, slot_iters)
        self.cp_decode = make_mlp(self.dim + 6, self.dim, 4)

        # Stage A: only teacher construction may use GT hand flow.
        self.teacher_encoder = make_mlp(12, self.dim, self.dim)
        self.teacher_slots = SlotAttention(self.dim, cm_slots, slot_iters)
        self.hand_decoder = make_mlp(self.dim + 6, self.dim, 3)

        # Stage C: Cp plus current hand state, without future geometry.
        self.hand_state = make_mlp(6, self.dim, self.dim)
        self.cp_to_hand = nn.MultiheadAttention(self.dim, num_heads=8, batch_first=True)
        self.cm_slots = SlotAttention(self.dim, cm_slots, slot_iters)

    def encode_cp(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        effect = torch.cat(
            [
                batch["obj_points"],
                batch["obj_normals"],
                batch["obj_contact_gt"].unsqueeze(-1),
                batch["obj_flow_gt"],
            ],
            dim=-1,
        )
        return self.cp_slots(self.cp_encoder(effect))

    def decode_effect(self, cp_tokens: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        num_obj = batch["obj_points"].shape[1]
        num_slots = cp_tokens.shape[1]
        expanded_tokens = cp_tokens.unsqueeze(1).expand(-1, num_obj, -1, -1)
        expanded_points = batch["obj_points"].unsqueeze(2).expand(-1, -1, num_slots, -1)
        expanded_normals = batch["obj_normals"].unsqueeze(2).expand(-1, -1, num_slots, -1)
        decoded = self.cp_decode(torch.cat([expanded_tokens, expanded_points, expanded_normals], dim=-1)).mean(2)
        return decoded[..., 0], decoded[..., 1:]

    def teacher_cm(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zeros = torch.zeros_like(batch["hand_points"])
        teacher_input = torch.cat(
            [batch["hand_points"], batch["hand_normals"], batch["hand_flow"], zeros], dim=-1
        )
        return self.teacher_slots(self.teacher_encoder(teacher_input))

    def decode_hand(self, cm_tokens: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hand_points = batch["hand_points"]
        state = self.hand_state(torch.cat([hand_points, batch["hand_normals"]], dim=-1))
        score = torch.einsum("bhd,bkd->bhk", state, cm_tokens) / self.dim**0.5
        context = torch.einsum("bhk,bkd->bhd", score.softmax(dim=-1), cm_tokens)
        return self.hand_decoder(torch.cat([context, hand_points, batch["hand_normals"]], dim=-1))

    def closure_cm(self, cp_tokens: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.hand_state(torch.cat([batch["hand_points"], batch["hand_normals"]], dim=-1))
        attended, _ = self.cp_to_hand(state, cp_tokens, cp_tokens, need_weights=False)
        return self.cm_slots(state + attended)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        cp_tokens, cp_assignment, cp_weights = self.encode_cp(batch)
        teacher_tokens, teacher_assignment, _ = self.teacher_cm(batch)
        pred_cm_tokens, pred_cm_assignment, _ = self.closure_cm(cp_tokens, batch)
        contact_logits, pred_obj_flow = self.decode_effect(cp_tokens, batch)
        return {
            "cp_tokens": cp_tokens,
            "cp_assignment": cp_assignment,
            "cp_slot_weights": cp_weights,
            "teacher_cm_tokens": teacher_tokens,
            "teacher_cm_assignment": teacher_assignment,
            "pred_hand_flow_teacher": self.decode_hand(teacher_tokens, batch),
            "pred_obj_contact_logits": contact_logits,
            "pred_obj_flow": pred_obj_flow,
            "pred_cm_tokens": pred_cm_tokens,
            "pred_cm_assignment": pred_cm_assignment,
            "pred_hand_flow": self.decode_hand(pred_cm_tokens, batch),
        }

    def load_stage(self, path: str | Path, prefixes: tuple[str, ...]) -> None:
        payload = torch.load(Path(path), map_location="cpu")
        state = payload.get("model", payload)
        own_state = self.state_dict()
        selected = {key: value for key, value in state.items() if key in own_state and key.startswith(prefixes)}
        if not selected:
            raise ValueError(f"{path} contains no Cp parameters matching {prefixes}.")
        self.load_state_dict(selected, strict=False)
        for name, parameter in self.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(False)
