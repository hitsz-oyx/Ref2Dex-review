from __future__ import annotations

import torch
import torch.nn as nn

from .edge_builder import build_edges
from .dense_state_encoder import FrozenDenseStateEncoder
from .modules import RelationEncoder, InteractionMessage, ObjectAggregator, ObjectEffectDecoder


class InteractionTransfer(nn.Module):
    """V0 forward model: (O,H,dH) -> relation messages -> dO."""
    def __init__(self, k=16, radius=0.05, dense_checkpoint=None, dense_dim=64, relation_dim=64, message_dim=32):
        super().__init__()
        self.k, self.radius = int(k), float(radius)
        self.static_encoder = FrozenDenseStateEncoder(dense_checkpoint or "src/task/Cm/densetoken_ckpt/best.pt")
        dense_dim = self.static_encoder.output_dim
        self.relation = RelationEncoder(dense_dim, relation_dim)
        self.message = InteractionMessage(relation_dim, message_dim)
        self.aggregate = ObjectAggregator(message_dim, dense_dim)
        self.effect = ObjectEffectDecoder(dense_dim)

    def forward(self, object_points, object_normals, hand_points, hand_normals, hand_flow):
        edge_idx, edge_valid = build_edges(object_points, hand_points, self.k, self.radius)
        z_obj, z_hand = self.static_encoder(object_points, object_normals, hand_points, hand_normals)
        hand_edge = torch.gather(z_hand.unsqueeze(1).expand(-1, z_obj.shape[1], -1, -1), 2,
                                 edge_idx.unsqueeze(-1).expand(-1, -1, -1, z_hand.shape[-1]))
        obj_edge = z_obj.unsqueeze(2).expand_as(hand_edge)
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
        relation = self.relation(torch.cat([obj_edge, hand_edge], -1), geom)
        flow = torch.gather(hand_flow.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                            edge_idx.unsqueeze(-1).expand_as(hand_p))
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
