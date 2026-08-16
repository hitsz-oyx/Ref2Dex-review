from __future__ import annotations

import torch
import torch.nn as nn

from .edge_builder import build_edges_bimanual
from .dense_state_encoder import FrozenDenseStateEncoder
from .modules import RelationEncoder, InteractionMessage, ObjectAggregator, ObjectEffectDecoder


class InteractionTransfer(nn.Module):
    """V1.0 forward model: (O,H_L,H_R,ΔH,Δt) -> relation messages -> dO.

    相对 V0.8 的变化（指导 V1.0）：
    - 双手输入：left/right 手点分别 K_L=8 / K_R=8 近邻（split KNN），总 edge
      数仍为 16；模型本身不接收 left/right side embedding，保持
      embodiment-agnostic，左右信息只用于 edge builder 的采样均衡。
    - action 9D：[Δh, v_n, v_t, |Δh|, Δt]，Δt 为 normalized gap（g），
      让模型区分 33/67/133/267 ms 的时间尺度。
    - 手点每 epoch 在线随机采样，static 部分不再离线缓存，每次 forward
      在线跑 frozen PTv3（V1.0 接受的代价）。
    """
    def __init__(self, k_l=8, k_r=8, radius=0.05, dense_checkpoint=None, dense_dim=64,
                 relation_dim=64, message_dim=32):
        super().__init__()
        self.k_l, self.k_r, self.radius = int(k_l), int(k_r), float(radius)
        self.static_encoder = FrozenDenseStateEncoder(dense_checkpoint or "src/task/Cm/densetoken_ckpt/best.pt")
        dense_dim = self.static_encoder.output_dim
        self.relation = RelationEncoder(self.static_encoder.output_dim // 2, relation_dim)
        self.message = InteractionMessage(relation_dim, message_dim, action_dim=9)
        self.aggregate = ObjectAggregator(message_dim, dense_dim)
        self.effect = ObjectEffectDecoder(dense_dim)

    @property
    def k(self):
        return self.k_l + self.k_r

    @torch.no_grad()
    def encode_static(self, object_points, object_normals, left_points, left_normals,
                      right_points, right_normals):
        """计算冻结的静态关系特征（每次 forward 在线执行）。"""
        hand_points = torch.cat([left_points, right_points], 1)
        hand_normals = torch.cat([left_normals, right_normals], 1)
        edge_idx, edge_valid = build_edges_bimanual(object_points, left_points, right_points,
                                                    self.k_l, self.k_r, self.radius)
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
                     geom_contact, hand_flow, dt):
        """可训练部分；dt 为 [B] 或 [B,1] 的 normalized gap（g）。"""
        relation = self.relation(dense_edge, geom_contact)
        obj_p = object_points.unsqueeze(2).expand(-1, -1, edge_idx.shape[-1], -1)
        flow = torch.gather(hand_flow.unsqueeze(1).expand(-1, object_points.shape[1], -1, -1), 2,
                            edge_idx.unsqueeze(-1).expand_as(obj_p))
        obj_n = object_normals.unsqueeze(2).expand_as(obj_p)
        vn = (flow * obj_n).sum(-1, keepdim=True)
        vt = flow - vn * obj_n
        dt_channel = dt.reshape(-1, 1, 1, 1).to(flow.dtype).expand(*flow.shape[:-1], 1)
        action = torch.cat([flow, vn, vt, torch.linalg.vector_norm(flow, dim=-1, keepdim=True),
                            dt_channel], -1)
        messages = self.message(relation, action)
        object_field, weights = self.aggregate(messages, edge_valid)
        # No hand/relation edge is exposed to the decoder.
        object_flow = self.effect(object_points, object_normals, object_field)
        return {"object_flow": object_flow, "object_field": object_field, "edge_message": messages,
                "edge_index": edge_idx, "edge_valid": edge_valid, "relation": relation,
                "attention": weights}

    def forward(self, object_points, object_normals, left_points, left_normals,
                right_points, right_normals, hand_flow, dt):
        static = self.encode_static(object_points, object_normals, left_points, left_normals,
                                    right_points, right_normals)
        return self.forward_core(**static, hand_flow=hand_flow, dt=dt)
