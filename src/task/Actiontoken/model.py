from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from src.task.Posetoken.model import StaticPoseEncoder


class DynamicActionEncoder(nn.Module):
    def __init__(self, dim: int = 384, global_dim: int = 128, heads: int = 6,
                 layers: int = 2, patch_size: int = 32) -> None:
        super().__init__()
        self.pair_mlp = nn.Sequential(nn.Linear(dim * 3 + global_dim * 2, dim * 2),
                                      nn.GELU(), nn.Linear(dim * 2, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           norm_first=True, activation="gelu", dropout=0.0)
        self.temporal = nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim))
        self.flow_decoder = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                          nn.Linear(dim, patch_size * 3))
        nn.init.zeros_(self.flow_decoder[-1].weight)
        nn.init.zeros_(self.flow_decoder[-1].bias)
        self.patch_size = patch_size

    def forward_pairs(self, before: torch.Tensor, after: torch.Tensor,
                      global_before: torch.Tensor, global_after: torch.Tensor
                      ) -> dict[str, torch.Tensor]:
        global_before = global_before[:, :, None].expand(-1, -1, before.shape[2], -1)
        global_after = global_after[:, :, None].expand_as(global_before)
        forward_pair = torch.cat(
            [before, after, after - before, global_before, global_after], -1)
        reverse_pair = torch.cat(
            [after, before, before - after, global_after, global_before], -1)
        action = .5 * (self.pair_mlp(forward_pair) - self.pair_mlp(reverse_pair))
        batch, steps, patches, dim = action.shape
        action = action.transpose(1, 2).reshape(batch * patches, steps, dim)
        action = .5 * (self.temporal(action) - self.temporal(-action))
        action = action.reshape(batch, patches, steps, dim).transpose(1, 2)
        flow = .5 * (self.flow_decoder(action) - self.flow_decoder(-action))
        flow = flow.reshape(batch, steps, patches, self.patch_size, 3)
        return {"action_tokens": action, "pred_dense_flow_internal": flow}

    def forward(self, local: torch.Tensor, global_pose: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_pairs(local[:, :-1], local[:, 1:],
                                  global_pose[:, :-1], global_pose[:, 1:])


class ActionTokenModel(nn.Module):
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        meta = cfg.meta
        pose_meta = SimpleNamespace(model_dim=meta.model_dim, global_dim=meta.global_dim,
                                    attention_heads=meta.attention_heads,
                                    patch_size=meta.patch_size, num_patches=meta.num_patches,
                                    num_hand_points=meta.num_hand_points)
        self.pose_encoder = StaticPoseEncoder(SimpleNamespace(meta=pose_meta))
        payload = torch.load(meta.pose_encoder_checkpoint, map_location="cpu")
        self.pose_encoder.load_state_dict(payload["pose_encoder"], strict=False)
        self.pose_encoder.requires_grad_(False).eval()
        self.dynamic = DynamicActionEncoder(meta.model_dim, meta.global_dim,
                                            meta.attention_heads, meta.temporal_layers,
                                            meta.patch_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self.pose_encoder.eval()
        return self

    def encode_pose_sequence(self, batch: dict[str, torch.Tensor]):
        points = batch["hand_points_root_sequence"]
        batch_size, frames, count, _ = points.shape
        pose_batch = {"hand_points_root": points.reshape(batch_size * frames, count, 3),
                      "hand_cano_points": batch["hand_cano_points"][:, None].expand(
                          -1, frames, -1, -1).reshape(batch_size * frames, count, 3),
                      "patch_knn_idx": batch["patch_knn_idx"][:, None].expand(
                          -1, frames, -1, -1).reshape(batch_size * frames,
                                                      *batch["patch_knn_idx"].shape[1:])}
        with torch.no_grad():
            encoded = self.pose_encoder(pose_batch)
        return (encoded["pose_tokens"].reshape(batch_size, frames, -1, encoded["pose_tokens"].shape[-1]),
                encoded["global_pose_token"].reshape(batch_size, frames, -1))

    def forward(self, batch: dict[str, torch.Tensor]):
        local, global_pose = self.encode_pose_sequence(batch)
        output = self.dynamic(local, global_pose)
        return {**output, "pose_tokens_sequence": local,
                "global_pose_token_sequence": global_pose}
