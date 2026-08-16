"""V0.x 兼容层：旧 checkpoint（单右手、8D action、无 Δt）的模型与 inverse。

V1.0 代码库加载 v08_full_20ep 等 V0.8/V0.10 权重时使用；检测逻辑见
``provider.load_model``。模型 forward 与 V0.10 model.py 完全一致
（K=16 单手 KNN、action 8D、forward_core 无 dt），仅 ``InteractionMessage``
需要显式传 ``action_dim=8``（V1.0 默认 9）。inverse 为单 twist 6D rigid。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.task.InteractionTransfer.edge_builder import build_edges
from src.task.InteractionTransfer.dense_state_encoder import FrozenDenseStateEncoder
from src.task.InteractionTransfer.inverse_optimize import fit_rigid_twist, twist_to_flow
from src.task.InteractionTransfer.metrics import epe
from src.task.InteractionTransfer.modules import (
    InteractionMessage,
    ObjectAggregator,
    ObjectEffectDecoder,
    RelationEncoder,
)


class LegacyInteractionTransfer(nn.Module):
    """V0.x forward model: (O,H,ΔH) -> relation messages -> dO（单右手）。"""

    def __init__(self, k=16, radius=0.05, dense_checkpoint=None, dense_dim=64,
                 relation_dim=64, message_dim=32):
        super().__init__()
        self.k, self.radius = int(k), float(radius)
        self.static_encoder = FrozenDenseStateEncoder(dense_checkpoint or "src/task/Cm/densetoken_ckpt/best.pt")
        dense_dim = self.static_encoder.output_dim
        self.relation = RelationEncoder(self.static_encoder.output_dim // 2, relation_dim)
        self.message = InteractionMessage(relation_dim, message_dim, action_dim=8)
        self.aggregate = ObjectAggregator(message_dim, dense_dim)
        self.effect = ObjectEffectDecoder(dense_dim)

    @torch.no_grad()
    def encode_static(self, object_points, object_normals, hand_points, hand_normals):
        edge_idx, edge_valid = build_edges(object_points, hand_points, self.k, self.radius)
        z_obj, z_hand = self.static_encoder(object_points, object_normals, hand_points, hand_normals)
        dense_edge, contact_prob = self.static_encoder.edge_features(z_obj, z_hand, edge_idx)
        obj_p = object_points.unsqueeze(2).expand(-1, -1, edge_idx.shape[-1], -1)
        obj_n = object_normals.unsqueeze(2).expand_as(obj_p)
        hand_p = torch.gather(hand_points.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                              edge_idx.unsqueeze(-1).expand_as(obj_p))
        hand_n = torch.gather(hand_normals.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                              edge_idx.unsqueeze(-1).expand_as(hand_p))
        rel = hand_p - obj_p
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        unit = rel / dist.clamp_min(1e-8)
        geom = torch.cat([rel, dist, obj_n, hand_n, (unit * obj_n).sum(-1, keepdim=True),
                          -(unit * hand_n).sum(-1, keepdim=True), (obj_n * hand_n).sum(-1, keepdim=True)], -1)
        geom_contact = torch.cat([geom, contact_prob.unsqueeze(-1)], -1)
        return {"object_points": object_points, "object_normals": object_normals,
                "edge_idx": edge_idx, "edge_valid": edge_valid,
                "dense_edge": dense_edge, "geom_contact": geom_contact}

    def forward_core(self, object_points, object_normals, edge_idx, edge_valid, dense_edge,
                     geom_contact, hand_flow):
        relation = self.relation(dense_edge, geom_contact)
        obj_p = object_points.unsqueeze(2).expand(-1, -1, edge_idx.shape[-1], -1)
        flow = torch.gather(hand_flow.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                            edge_idx.unsqueeze(-1).expand_as(obj_p))
        obj_n = object_normals.unsqueeze(2).expand_as(obj_p)
        vn = (flow * obj_n).sum(-1, keepdim=True)
        vt = flow - vn * obj_n
        action = torch.cat([flow, vn, vt, torch.linalg.vector_norm(flow, dim=-1, keepdim=True)], -1)
        messages = self.message(relation, action)
        object_field, weights = self.aggregate(messages, edge_valid)
        object_flow = self.effect(object_points, object_normals, object_field)
        return {"object_flow": object_flow, "object_field": object_field, "edge_message": messages,
                "edge_index": edge_idx, "edge_valid": edge_valid, "relation": relation,
                "attention": weights}

    def forward(self, object_points, object_normals, hand_points, hand_normals, hand_flow):
        static = self.encode_static(object_points, object_normals, hand_points, hand_normals)
        return self.forward_core(**static, hand_flow=hand_flow)


def legacy_optimize_hand_flow(
    model: LegacyInteractionTransfer,
    static: dict,
    hand_points: torch.Tensor,
    target_object_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    init_flow: torch.Tensor,
    parameterization: str = "rigid",
    target_kind: str = "effect",
    steps: int = 300,
    lr: float = 0.01,
    history_every: int = 25,
) -> dict:
    """V0.x inverse：单 twist 6D rigid、无 dt；返回结构与 V1.0 optimize_hand_flow 一致。"""
    model.requires_grad_(False)

    if target_kind == "c_obj":
        with torch.no_grad():
            target_field = model.forward_core(**static, hand_flow=gt_flow)["object_field"]

    def forward_out(flow: torch.Tensor) -> dict:
        return model.forward_core(**static, hand_flow=flow)

    with torch.no_grad():
        initial_out = forward_out(init_flow)

    if parameterization == "rigid":
        center = hand_points.mean(dim=1, keepdim=True)
        var = fit_rigid_twist(init_flow, hand_points).detach().clone().requires_grad_(True)

        def current_flow() -> torch.Tensor:
            return twist_to_flow(var, hand_points, center)
    elif parameterization == "free":
        var = init_flow.detach().clone().requires_grad_(True)

        def current_flow() -> torch.Tensor:
            return var
    else:
        raise ValueError(f"unknown parameterization: {parameterization}")

    optimizer = torch.optim.Adam([var], lr=lr)
    history = []
    for step in range(steps + 1):
        flow = current_flow()
        out = forward_out(flow)
        if target_kind == "effect":
            loss = torch.nn.functional.mse_loss(out["object_flow"], target_object_flow)
        else:
            loss = torch.nn.functional.mse_loss(out["object_field"], target_field)
        if step < steps:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if step % history_every == 0 or step == steps:
            with torch.no_grad():
                history.append({"step": step, "loss": float(loss),
                                "epe_mm": float(epe(out["object_flow"], target_object_flow)) * 1000})

    with torch.no_grad():
        optimized_flow = current_flow().detach()
        optimized_out = forward_out(optimized_flow)
        gt_out = forward_out(gt_flow)

    def summarize(out: dict, flow: torch.Tensor) -> dict:
        return {
            "flow": flow.detach().cpu(),
            "object_flow": out["object_flow"].detach().cpu(),
            "object_field": out["object_field"].detach().cpu(),
            "epe_mm": float(epe(out["object_flow"], target_object_flow)) * 1000,
            "c_obj_l2_to_gt": float(torch.linalg.vector_norm(
                out["object_field"] - gt_out["object_field"], dim=-1).mean()),
        }

    return {
        "initial": summarize(initial_out, init_flow),
        "optimized": summarize(optimized_out, optimized_flow),
        "gt": summarize(gt_out, gt_flow),
        "history": history,
        "meta": {"parameterization": parameterization, "target_kind": target_kind,
                 "steps": steps, "lr": lr, "dt": 1.0},
    }
