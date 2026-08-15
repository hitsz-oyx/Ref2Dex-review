from __future__ import annotations

import torch
import torch.nn as nn


class RelationEncoder(nn.Module):
    def __init__(self, dense_dim=64, relation_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dense_dim + 14, relation_dim), nn.GELU(),
                                 nn.Linear(relation_dim, relation_dim), nn.LayerNorm(relation_dim))

    def forward(self, dense_edge, geom):
        return self.net(torch.cat([dense_edge, geom], -1))


class InteractionMessage(nn.Module):
    def __init__(self, relation_dim=64, message_dim=32):
        super().__init__()
        # bias-free action path makes action=0 exactly message=0.
        self.action = nn.Sequential(nn.Linear(8, message_dim, bias=False), nn.GELU(),
                                    nn.Linear(message_dim, message_dim, bias=False))
        self.gate = nn.Sequential(nn.Linear(relation_dim, message_dim), nn.Sigmoid())
        self.message = nn.Sequential(nn.Linear(message_dim, message_dim, bias=False), nn.GELU(),
                                     nn.Linear(message_dim, message_dim, bias=False))

    def forward(self, relation, action):
        return self.message(self.gate(relation) * self.action(action))


class ObjectAggregator(nn.Module):
    def __init__(self, message_dim=32, object_dim=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(message_dim, message_dim), nn.GELU(), nn.Linear(message_dim, 1))
        self.proj = nn.Linear(message_dim, object_dim, bias=False)

    def forward(self, messages, valid):
        logits = self.score(messages).squeeze(-1).masked_fill(~valid, -1e4)
        weights = torch.softmax(logits, -1) * valid.float()
        pooled = (messages * weights.unsqueeze(-1)).sum(-2)
        return self.proj(pooled), weights


class ObjectEffectDecoder(nn.Module):
    def __init__(self, object_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(object_dim + 6 + object_dim, 128), nn.GELU(),
                                 nn.Linear(128, 3))

    def forward(self, points, normals, object_field):
        global_field = object_field.mean(1, keepdim=True).expand_as(object_field)
        return self.net(torch.cat([points, normals, object_field, global_field], -1))
