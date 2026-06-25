from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.correspondence import compute_obj_to_hand_edge_features, gather_knn_features


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


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
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
    """Wrap official PointTransformerV3 to operate on dense batched tensors."""

    def __init__(self, meta: Any, in_channels: int) -> None:
        super().__init__()
        ptv3_cls = _load_ptv3_model_class(getattr(meta, "ptv3_repo_path"))
        self.grid_size = float(meta.ptv3_grid_size)
        self.output_dim = int(tuple(meta.ptv3_dec_channels)[0])
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
            flat_batch.append(torch.full((valid_idx.numel(),), batch_idx, device=feat.device, dtype=torch.long))
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
        dense_out_flat[torch.cat(flat_dense_idx, dim=0)] = flat_out
        return dense_out


class StaticHOCPTv3(nn.Module):
    """Object-centric PTv3 static correspondence model."""

    def __init__(
        self,
        cfg: Any,
        *,
        condition_shape: list[int] | tuple[int, ...] | None = None,
        target_shape: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        meta = cfg.meta

        self.stem_dim = int(getattr(meta, "hidden_dim", 96))
        self.num_obj_points = int(meta.num_obj_points)
        self.num_hand_points = int(meta.num_hand_points)
        self.num_fingers = int(meta.num_fingers)
        self.num_regions = int(meta.num_regions)
        self.use_finger_region_head = bool(getattr(meta, "use_finger_region_head", False))

        self.obj_stem = _build_stem(6, self.stem_dim)
        self.hand_stem = _build_stem(9, self.stem_dim)
        self.backbone = PTv3DenseBackbone(meta, in_channels=self.stem_dim)
        self.token_dim = int(self.backbone.output_dim)

        self.edge_geo_dim = self.token_dim // 2
        self.cano_embed_dim = self.token_dim // 2
        self.edge_geo_mlp = _build_mlp(6, self.token_dim // 2, self.edge_geo_dim)
        self.hand_cano_mlp = _build_mlp(3, self.token_dim // 2, self.cano_embed_dim)
        self.cross_q = nn.Linear(self.token_dim, self.token_dim)
        self.cross_k = nn.Linear(self.token_dim + self.edge_geo_dim, self.token_dim)
        self.cross_v = nn.Linear(self.token_dim + self.edge_geo_dim + self.cano_embed_dim, self.token_dim)
        self.cross_bias = _build_mlp(6, self.token_dim // 2, 1)
        self.cross_out = _build_mlp(self.token_dim, self.token_dim, self.token_dim)

        self.contact_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.GELU(),
            nn.Linear(self.token_dim // 2, 1),
        )

        if self.use_finger_region_head:
            if self.num_fingers <= 0 or self.num_regions <= 0:
                raise ValueError("Finger/region heads require positive num_fingers and num_regions.")
            self.finger_head = nn.Sequential(
                nn.Linear(self.token_dim, self.token_dim // 2),
                nn.GELU(),
                nn.Linear(self.token_dim // 2, self.num_fingers),
            )
            self.region_head = nn.Sequential(
                nn.Linear(self.token_dim, self.token_dim // 2),
                nn.GELU(),
                nn.Linear(self.token_dim // 2, self.num_regions),
            )

        cross_edge_input_dim = self.token_dim * 2 + 6
        self.edge_shared_dim = self.token_dim // 2
        self.edge_shared_backbone = nn.Sequential(
            nn.Linear(cross_edge_input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.edge_shared_dim),
            nn.GELU(),
        )
        self.cross_edge_head = nn.Linear(self.edge_shared_dim, 1)
        self.cano_edge_head = nn.Linear(self.edge_shared_dim, 3)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        points = batch["points"].float()
        normals = batch["normals"].float()
        batch_size, total_points, _ = points.shape
        expected_total = self.num_obj_points + self.num_hand_points
        if total_points < expected_total:
            raise ValueError(f"Expected at least {expected_total} points, got {total_points}.")

        obj_points = points[:, : self.num_obj_points]
        obj_normals = normals[:, : self.num_obj_points]
        hand_points = points[:, self.num_obj_points : expected_total]
        hand_normals = normals[:, self.num_obj_points : expected_total]

        hand_cano_points = batch["hand_cano_points"].float()
        if hand_cano_points.dim() == 2:
            hand_cano_points = hand_cano_points.unsqueeze(0).expand(batch_size, -1, -1)
        hand_cano = hand_cano_points[:, self.num_obj_points : expected_total]

        point_valid_mask = batch.get("point_valid_mask")
        if point_valid_mask is None:
            point_valid_mask = torch.ones(batch_size, expected_total, device=points.device, dtype=torch.bool)
        elif point_valid_mask.dim() == 1:
            point_valid_mask = point_valid_mask.unsqueeze(0).expand(batch_size, -1)
        point_valid_mask = point_valid_mask[:, :expected_total].bool()

        obj_raw = torch.cat([obj_points, obj_normals], dim=-1)
        hand_raw = torch.cat([hand_points, hand_normals, hand_cano], dim=-1)
        obj_feat = self.obj_stem(obj_raw)
        hand_feat = self.hand_stem(hand_raw)

        coord = torch.cat([obj_points, hand_points], dim=1)
        feat = torch.cat([obj_feat, hand_feat], dim=1)
        tokens = self.backbone(feat, coord, point_valid_mask)

        z_obj = tokens[:, : self.num_obj_points]
        z_hand = tokens[:, self.num_obj_points : expected_total]
        obj_to_hand_knn_idx = batch["obj_to_hand_knn_idx"].long()
        obj_to_hand_knn_valid_mask = batch["obj_to_hand_knn_valid_mask"].bool()
        z_obj_cross, obj_to_hand_attn, edge_geo, z_hand_neighbors = self._compute_obj_cross_context(
            z_obj=z_obj,
            z_hand=z_hand,
            obj_points=obj_points,
            hand_points=hand_points,
            obj_normals=obj_normals,
            hand_normals=hand_normals,
            hand_cano=hand_cano,
            obj_to_hand_knn_idx=obj_to_hand_knn_idx,
            obj_to_hand_knn_valid_mask=obj_to_hand_knn_valid_mask,
        )

        outputs = {
            "pred_obj_contact": self.contact_head(z_obj_cross).squeeze(-1),
            "obj_dense_tokens": z_obj_cross,
            "hand_dense_tokens": z_hand,
            "ptv3_obj_tokens": z_obj,
            "obj_to_hand_attn": obj_to_hand_attn,
        }

        edge_shared = self._compute_shared_edge_features(
            z_obj_cross=z_obj_cross,
            z_hand_neighbors=z_hand_neighbors,
            edge_geo=edge_geo,
        )
        outputs.update(
            {
                "pred_cross_cano": self._predict_cross_cano(
                    edge_shared=edge_shared,
                ),
                "pred_cross_contact": self._compute_cross_edge_predictions(
                    edge_shared=edge_shared,
                ),
            }
        )

        if self.use_finger_region_head:
            outputs["pred_obj_to_hand_finger"] = self.finger_head(z_obj_cross)
            outputs["pred_obj_to_hand_region"] = self.region_head(z_obj_cross)

        return outputs

    def _compute_obj_cross_context(
        self,
        z_obj: torch.Tensor,
        z_hand: torch.Tensor,
        obj_points: torch.Tensor,
        hand_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_cano: torch.Tensor,
        obj_to_hand_knn_idx: torch.Tensor,
        obj_to_hand_knn_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = z_obj.shape[0]
        edge_geo_list: list[torch.Tensor] = []
        z_hand_neighbors_list: list[torch.Tensor] = []
        hand_cano_neighbors_list: list[torch.Tensor] = []

        for batch_idx in range(batch_size):
            edge_feats = compute_obj_to_hand_edge_features(
                obj_points=obj_points[batch_idx],
                hand_points=hand_points[batch_idx],
                obj_normals=obj_normals[batch_idx],
                hand_normals=hand_normals[batch_idx],
                obj_to_hand_knn_idx=obj_to_hand_knn_idx[batch_idx],
                obj_to_hand_knn_valid_mask=obj_to_hand_knn_valid_mask[batch_idx],
            )
            edge_geo = torch.cat(
                [
                    edge_feats["delta"],
                    edge_feats["dist"],
                    edge_feats["signed_dist"],
                    edge_feats["normal_dot"],
                ],
                dim=-1,
            )
            edge_geo_list.append(edge_geo)
            z_hand_neighbors_list.append(
                gather_knn_features(
                    z_hand[batch_idx],
                    obj_to_hand_knn_idx[batch_idx],
                    obj_to_hand_knn_valid_mask[batch_idx],
                )
            )
            hand_cano_neighbors_list.append(
                gather_knn_features(
                    hand_cano[batch_idx],
                    obj_to_hand_knn_idx[batch_idx],
                    obj_to_hand_knn_valid_mask[batch_idx],
                )
            )

        edge_geo = torch.stack(edge_geo_list, dim=0)
        z_hand_neighbors = torch.stack(z_hand_neighbors_list, dim=0)
        hand_cano_neighbors = torch.stack(hand_cano_neighbors_list, dim=0)

        edge_geo_embed = self.edge_geo_mlp(edge_geo)
        hand_cano_embed = self.hand_cano_mlp(hand_cano_neighbors)

        q = self.cross_q(z_obj).unsqueeze(2)
        k = self.cross_k(torch.cat([z_hand_neighbors, edge_geo_embed], dim=-1))
        v = self.cross_v(torch.cat([z_hand_neighbors, edge_geo_embed, hand_cano_embed], dim=-1))
        logits = (q * k).sum(dim=-1) / (self.token_dim**0.5)
        logits = logits + self.cross_bias(edge_geo).squeeze(-1)
        attn = _masked_softmax(logits, obj_to_hand_knn_valid_mask)
        cross_ctx = torch.sum(attn.unsqueeze(-1) * v, dim=2)
        valid_obj_mask = obj_to_hand_knn_valid_mask.any(dim=-1, keepdim=True).float()
        cross_update = self.cross_out(cross_ctx) * valid_obj_mask
        z_obj_cross = z_obj + cross_update
        return z_obj_cross, attn, edge_geo, z_hand_neighbors

    def _compute_shared_edge_features(
        self,
        z_obj_cross: torch.Tensor,
        z_hand_neighbors: torch.Tensor,
        edge_geo: torch.Tensor,
    ) -> torch.Tensor:
        k_cross = z_hand_neighbors.shape[2]
        z_obj_expanded = z_obj_cross.unsqueeze(2).expand(-1, -1, k_cross, -1)
        edge_input = torch.cat([z_obj_expanded, z_hand_neighbors, edge_geo], dim=-1)
        return self.edge_shared_backbone(edge_input)

    def _predict_cross_cano(
        self,
        edge_shared: torch.Tensor,
    ) -> torch.Tensor:
        return self.cano_edge_head(edge_shared)

    def _compute_cross_edge_predictions(
        self,
        edge_shared: torch.Tensor,
    ) -> torch.Tensor:
        return self.cross_edge_head(edge_shared).squeeze(-1)
