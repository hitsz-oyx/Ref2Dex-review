from __future__ import annotations

import torch
import torch.nn as nn

from ..dense_state_encoder import FrozenDenseStateEncoder
from ..edge_builder import build_edges
from ..modules import ObjectAggregator, ObjectEffectDecoder, RelationEncoder


class DirectEdge(nn.Module):
    """与 Cm 使用相同 R/a/edge/decoder，仅移除 zero-preserving Cm gate。"""

    def __init__(self, k=16, radius=0.05, dense_checkpoint=None, relation_dim=64, message_dim=32):
        super().__init__()
        self.k, self.radius = int(k), float(radius)
        self.static_encoder = FrozenDenseStateEncoder(dense_checkpoint or "src/task/Cm/densetoken_ckpt/best.pt")
        dense_dim = self.static_encoder.output_dim
        self.relation = RelationEncoder(dense_dim // 2, relation_dim)
        self.direct_edge = nn.Sequential(nn.Linear(relation_dim + 8, message_dim), nn.GELU(),
                                         nn.Linear(message_dim, message_dim))
        self.aggregate = ObjectAggregator(message_dim, dense_dim)
        self.effect = ObjectEffectDecoder(dense_dim)

    def train(self, mode=True):
        super().train(mode)
        self.static_encoder.eval()
        return self

    def forward(self, object_points, object_normals, hand_points, hand_normals, hand_flow):
        edge_idx, edge_valid = build_edges(object_points, hand_points, self.k, self.radius)
        z_obj, z_hand = self.static_encoder(object_points, object_normals, hand_points, hand_normals)
        hand_p = torch.gather(hand_points.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                              edge_idx.unsqueeze(-1).expand(-1, object_points.shape[1], -1, 3))
        hand_n = torch.gather(hand_normals.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                              edge_idx.unsqueeze(-1).expand_as(hand_p))
        obj_p = object_points.unsqueeze(2).expand_as(hand_p)
        obj_n = object_normals.unsqueeze(2).expand_as(hand_p)
        rel = hand_p - obj_p
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        unit = rel / dist.clamp_min(1e-8)
        geom = torch.cat([rel, dist, obj_n, hand_n, (unit * obj_n).sum(-1, keepdim=True),
                          -(unit * hand_n).sum(-1, keepdim=True), (obj_n * hand_n).sum(-1, keepdim=True)], -1)
        dense_edge, contact_prob = self.static_encoder.edge_features(z_obj, z_hand, edge_idx)
        relation = self.relation(dense_edge, torch.cat([geom, contact_prob.unsqueeze(-1)], -1))
        flow = torch.gather(hand_flow.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                            edge_idx.unsqueeze(-1).expand_as(hand_p))
        vn = (flow * obj_n).sum(-1, keepdim=True)
        vt = flow - vn * obj_n
        action = torch.cat([flow, vn, vt, torch.linalg.vector_norm(flow, dim=-1, keepdim=True)], -1)
        edge_feature = self.direct_edge(torch.cat([relation, action], -1))
        object_field, weights = self.aggregate(edge_feature, edge_valid)
        return {"object_flow": self.effect(object_points, object_normals, object_field),
                "object_field": object_field, "edge_message": edge_feature,
                "edge_index": edge_idx, "edge_valid": edge_valid, "relation": relation,
                "attention": weights}
