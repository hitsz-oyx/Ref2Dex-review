from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _build_stem(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    mask_f = mask.float()
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    max_logits = masked_logits.amax(dim=dim, keepdim=True)
    max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
    exp_logits = torch.exp(masked_logits - max_logits) * mask_f
    denom = exp_logits.sum(dim=dim, keepdim=True)
    return torch.where(denom > 0, exp_logits / denom.clamp(min=1e-12), torch.zeros_like(exp_logits))


def _load_ptv3_model_class(repo_path: str | Path):
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"PointTransformerV3 repo path not found: {repo_path}")
    parent = str(repo_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module = importlib.import_module("PointTransformerV3.model")
    return module.PointTransformerV3


class PTv3DenseBackbone(nn.Module):
    def __init__(self, meta: Any, in_channels: int) -> None:
        super().__init__()
        ptv3_cls = _load_ptv3_model_class(getattr(meta, "ptv3_repo_path"))
        self.grid_size = float(meta.ptv3_grid_size)
        self.output_dim = int(tuple(meta.ptv3_dec_channels)[0])
        self.shuffle_orders = bool(meta.ptv3_shuffle_orders)
        self.backbone = ptv3_cls(
            in_channels=int(in_channels),
            order=tuple(meta.ptv3_order),
            stride=tuple(meta.ptv3_stride),
            enc_depths=tuple(meta.ptv3_enc_depths),
            enc_channels=tuple(meta.ptv3_enc_channels),
            enc_num_head=tuple(meta.ptv3_enc_num_head),
            enc_patch_size=tuple(meta.ptv3_enc_patch_size),
            dec_depths=tuple(meta.ptv3_dec_depths),
            dec_channels=tuple(meta.ptv3_dec_channels),
            dec_num_head=tuple(meta.ptv3_dec_num_head),
            dec_patch_size=tuple(meta.ptv3_dec_patch_size),
            mlp_ratio=float(meta.ptv3_mlp_ratio),
            qkv_bias=bool(meta.ptv3_qkv_bias),
            qk_scale=None,
            attn_drop=float(meta.ptv3_attn_drop),
            proj_drop=float(meta.ptv3_proj_drop),
            drop_path=float(meta.ptv3_drop_path),
            pre_norm=bool(meta.ptv3_pre_norm),
            shuffle_orders=bool(meta.ptv3_shuffle_orders),
            enable_rpe=bool(meta.ptv3_enable_rpe),
            enable_flash=bool(meta.ptv3_enable_flash),
            upcast_attention=bool(meta.ptv3_upcast_attention),
            upcast_softmax=bool(meta.ptv3_upcast_softmax),
            cls_mode=False,
            pdnorm_bn=False,
            pdnorm_ln=False,
        )

    def forward(self, feat: torch.Tensor, coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        use_shuffle = self.shuffle_orders and self.training
        for module in self.backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = use_shuffle

        batch_size, num_points, _ = feat.shape
        dense_out = feat.new_zeros(batch_size, num_points, self.output_dim)

        flat_feat: list[torch.Tensor] = []
        flat_coord: list[torch.Tensor] = []
        flat_batch: list[torch.Tensor] = []
        flat_dense_idx: list[torch.Tensor] = []
        for batch_idx in range(batch_size):
            valid_idx = torch.nonzero(valid_mask[batch_idx], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            flat_feat.append(feat[batch_idx, valid_idx])
            flat_coord.append(coord[batch_idx, valid_idx])
            flat_batch.append(
                torch.full((valid_idx.numel(),), batch_idx, device=feat.device, dtype=torch.long)
            )
            flat_dense_idx.append(valid_idx + batch_idx * num_points)

        if not flat_feat:
            return dense_out

        data_dict = {
            "feat": torch.cat(flat_feat, dim=0).contiguous(),
            "coord": torch.cat(flat_coord, dim=0).contiguous(),
            "batch": torch.cat(flat_batch, dim=0).contiguous(),
            "grid_size": self.grid_size,
        }
        point = self.backbone(data_dict)
        flat_out = point.feat
        dense_out_flat = dense_out.view(batch_size * num_points, self.output_dim)
        dense_out_flat[torch.cat(flat_dense_idx, dim=0)] = flat_out.to(dtype=dense_out_flat.dtype)
        return dense_out
