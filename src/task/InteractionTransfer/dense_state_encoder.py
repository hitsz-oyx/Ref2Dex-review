from __future__ import annotations

import importlib
import sys
from pathlib import Path
import torch
import torch.nn as nn

from src.base import task_config_from_dict


def _ptv3(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        path = Path(__file__).resolve().parents[3] / "third_party" / "PointTransformerV3"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    return importlib.import_module("PointTransformerV3.model").PointTransformerV3


class FrozenDenseStateEncoder(nn.Module):
    """Independent loader for the pretrained correspondence DenseToken."""
    def __init__(self, checkpoint: str | Path = "src/task/Cm/densetoken_ckpt/best.pt"):
        super().__init__()
        if str(checkpoint) == "synthetic":
            self.output_dim = 64
            self.num_obj_points, self.num_hand_points = 512, 1538
            self.obj = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64))
            self.hand = nn.Sequential(nn.Linear(6, 64), nn.GELU(), nn.Linear(64, 64))
            self.edge_shared = nn.Sequential(nn.Linear(128, 64), nn.GELU())
            self.cross_head = nn.Linear(64, 1)
            for p in self.parameters(): p.requires_grad_(False)
            self.eval()
            return
        checkpoint = Path(checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = Path.cwd() / checkpoint
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if "config" not in data or "model" not in data:
            raise ValueError(f"unsupported DenseToken checkpoint: {checkpoint}")
        cfg = task_config_from_dict(data["config"])
        meta = cfg.meta
        ptv3 = _ptv3(meta.ptv3_repo_path)
        self.num_obj_points = int(meta.num_obj_points)
        self.num_hand_points = int(meta.num_hand_points)
        if (self.num_obj_points, self.num_hand_points) != (512, 1538):
            raise ValueError("DenseToken bootstrap requires 512 object / 1538 hand points")
        self.grid_size = float(meta.ptv3_grid_size)
        self.output_dim = int(tuple(meta.ptv3_dec_channels)[0])
        self.backbone = ptv3(
            in_channels=int(getattr(meta, "point_feat_dim", 11)), order=tuple(meta.ptv3_order),
            stride=tuple(meta.ptv3_stride), enc_depths=tuple(meta.ptv3_enc_depths),
            enc_channels=tuple(meta.ptv3_enc_channels), enc_num_head=tuple(meta.ptv3_enc_num_head),
            enc_patch_size=tuple(meta.ptv3_enc_patch_size), dec_depths=tuple(meta.ptv3_dec_depths),
            dec_channels=tuple(meta.ptv3_dec_channels), dec_num_head=tuple(meta.ptv3_dec_num_head),
            dec_patch_size=tuple(meta.ptv3_dec_patch_size), mlp_ratio=float(meta.ptv3_mlp_ratio),
            qkv_bias=bool(meta.ptv3_qkv_bias), qk_scale=None, attn_drop=float(meta.ptv3_attn_drop),
            proj_drop=float(meta.ptv3_proj_drop), drop_path=float(meta.ptv3_drop_path),
            pre_norm=bool(meta.ptv3_pre_norm), shuffle_orders=bool(meta.ptv3_shuffle_orders),
            enable_rpe=bool(meta.ptv3_enable_rpe), enable_flash=bool(meta.ptv3_enable_flash),
            upcast_attention=bool(meta.ptv3_upcast_attention), upcast_softmax=bool(meta.ptv3_upcast_softmax),
            cls_mode=False, pdnorm_bn=False, pdnorm_ln=False)
        self.edge_shared = nn.Sequential(nn.Linear(self.output_dim * 2, self.output_dim), nn.GELU(),
                                         nn.Linear(self.output_dim, self.output_dim // 2), nn.GELU())
        self.cross_head = nn.Linear(self.output_dim // 2, 1)
        state = data["model"]
        own = self.state_dict()
        selected = {}
        for key, value in state.items():
            # Original DenseToken wraps PTv3 as ``backbone.backbone``; this
            # task vendors only the PTv3 module and therefore removes one
            # wrapper component while loading.
            target = key[len("backbone."):] if key.startswith("backbone.") else key
            if target in own and own[target].shape == value.shape:
                selected[target] = value
        missing, _ = self.load_state_dict(selected, strict=False)
        if any(k.startswith("backbone.") for k in missing):
            raise RuntimeError(f"DenseToken backbone checkpoint incomplete: {len(missing)} missing keys")
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, obj_points, obj_normals, hand_points, hand_normals):
        b, no, _ = obj_points.shape
        if hasattr(self, "obj"):
            return self.obj(torch.cat([obj_points, obj_normals], -1)), self.hand(torch.cat([hand_points, hand_normals], -1))
        points = torch.cat([obj_points, hand_points], 1).float()
        normals = torch.cat([obj_normals, hand_normals], 1).float()
        d = torch.cdist(obj_points, hand_points)
        obj_nn = d.argmin(-1)
        hand_nn = d.argmin(1)
        obj_delta = torch.gather(hand_points.unsqueeze(1).expand(-1, obj_points.shape[1], -1, -1), 2,
                                 obj_nn.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2) - obj_points
        hand_delta = torch.gather(obj_points.unsqueeze(1).expand(-1, hand_points.shape[1], -1, -1), 2,
                                  hand_nn.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3)).squeeze(2) - hand_points
        obj_dir = obj_delta / obj_delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        hand_dir = hand_delta / hand_delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        obj_extra = torch.cat([d.amin(-1, keepdim=True), (obj_dir * obj_normals).sum(-1, keepdim=True),
                               torch.zeros_like(d.amin(-1, keepdim=True))], -1)
        hand_extra = torch.cat([d.amin(1).unsqueeze(-1), (hand_dir * hand_normals).sum(-1, keepdim=True),
                                torch.zeros_like(d.amin(1).unsqueeze(-1))], -1)
        obj_type = torch.cat([torch.ones_like(obj_points[..., :1]), torch.zeros_like(obj_points[..., :1])], -1)
        hand_type = torch.cat([torch.zeros_like(hand_points[..., :1]), torch.ones_like(hand_points[..., :1])], -1)
        feat = torch.cat([torch.cat([obj_points, obj_type, obj_normals, obj_extra], -1),
                          torch.cat([hand_points, hand_type, hand_normals, hand_extra], -1)], 1)
        batch = torch.arange(b, device=points.device).repeat_interleave(points.shape[1])
        result = self.backbone({"coord": points.flatten(0, 1), "feat": feat.flatten(0, 1),
                                "batch": batch, "grid_size": self.grid_size})
        token = result.feat.view(b, points.shape[1], self.output_dim)
        return token[:, :no], token[:, no:]

    def edge_features(self, z_obj, z_hand, edge_idx):
        hand_edge = torch.gather(z_hand.unsqueeze(1).expand(-1, z_obj.shape[1], -1, -1), 2,
                                 edge_idx.unsqueeze(-1).expand(-1, -1, -1, z_hand.shape[-1]))
        return self.edge_shared(torch.cat([z_obj.unsqueeze(2).expand_as(hand_edge), hand_edge], -1)), torch.sigmoid(self.cross_head(self.edge_shared(torch.cat([z_obj.unsqueeze(2).expand_as(hand_edge), hand_edge], -1))).squeeze(-1))
