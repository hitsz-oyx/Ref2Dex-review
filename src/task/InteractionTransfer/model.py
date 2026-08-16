from __future__ import annotations

import torch
import torch.nn as nn

from .edge_builder import build_edges
from .dense_state_encoder import FrozenDenseStateEncoder
from .modules import RelationEncoder, InteractionMessage, ObjectAggregator, ObjectEffectDecoder


class InteractionTransfer(nn.Module):
    """V0 forward model: (O,H,dH) -> relation messages -> dO.

    V0.8 将 forward 拆为 encode_static / forward_core：static 部分只依赖
    (O_t, H_t)，可离线缓存；forward_core 只依赖缓存张量和 hand_flow。
    """
    def __init__(self, k=16, radius=0.05, dense_checkpoint=None, dense_dim=64, relation_dim=64, message_dim=32):
        super().__init__()
        self.k, self.radius = int(k), float(radius)
        self.static_encoder = FrozenDenseStateEncoder(dense_checkpoint or "src/task/Cm/densetoken_ckpt/best.pt")
        dense_dim = self.static_encoder.output_dim
        self.relation = RelationEncoder(self.static_encoder.output_dim // 2, relation_dim)
        self.message = InteractionMessage(relation_dim, message_dim)
        self.aggregate = ObjectAggregator(message_dim, dense_dim)
        self.effect = ObjectEffectDecoder(dense_dim)

    @torch.no_grad()
    def encode_static(self, object_points, object_normals, hand_points, hand_normals):
        """计算冻结的静态关系特征；输出可直接缓存复用。"""
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

    def forward_core(self, object_points, object_normals, edge_idx, edge_valid, dense_edge, geom_contact, hand_flow):
        """只跑可训练部分；static 输入可来自 encode_static 或离线 cache。"""
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
        # No hand/relation edge is exposed to the decoder.
        object_flow = self.effect(object_points, object_normals, object_field)
        return {"object_flow": object_flow, "object_field": object_field, "edge_message": messages,
                "edge_index": edge_idx, "edge_valid": edge_valid, "relation": relation,
                "attention": weights}

    def forward(self, object_points, object_normals, hand_points, hand_normals, hand_flow):
        static = self.encode_static(object_points, object_normals, hand_points, hand_normals)
        return self.forward_core(**static, hand_flow=hand_flow)
