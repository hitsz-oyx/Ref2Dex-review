from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.utils.correspondence import gather_knn_features


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
        dense_out_flat[torch.cat(flat_dense_idx, dim=0)] = flat_out.to(dtype=dense_out_flat.dtype)
        return dense_out


class StaticHOCPTv3V2(nn.Module):
    def __init__(self, cfg: Any, *, condition_shape=None, target_shape=None) -> None:
        del condition_shape, target_shape
        super().__init__()
        self.cfg = cfg
        meta = cfg.meta
        self.num_obj_points = int(meta.num_obj_points)
        self.num_hand_points = int(meta.num_hand_points)
        self.point_feat_dim = int(getattr(meta, "point_feat_dim", 11))
        self.backbone = PTv3DenseBackbone(meta, in_channels=self.point_feat_dim)
        self.token_dim = int(self.backbone.output_dim)
        self.contact_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.GELU(),
            nn.Linear(self.token_dim // 2, 1),
        )
        self.edge_shared_backbone = nn.Sequential(
            nn.Linear(self.token_dim * 2, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.GELU(),
        )
        self.cross_edge_head = nn.Linear(self.token_dim // 2, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        z_obj, z_hand = self.encode_points(batch)
        supervision_edge_idx = batch["supervision_edge_idx"].long()
        supervision_edge_valid_mask = batch["supervision_edge_valid_mask"].bool()
        z_hand_neighbors = self._gather_batched_knn_features(
            z_hand,
            supervision_edge_idx,
            supervision_edge_valid_mask,
        )

        pred_obj_contact_logits = self.contact_head(z_obj).squeeze(-1)
        edge_shared = self._compute_shared_edge_features(z_obj=z_obj, z_hand_neighbors=z_hand_neighbors)
        pred_cross_contact_logits = self.cross_edge_head(edge_shared).squeeze(-1)

        return {
            "pred_obj_contact_logits": pred_obj_contact_logits,
            "pred_obj_contact_prob": torch.sigmoid(pred_obj_contact_logits),
            "pred_cross_contact_logits": pred_cross_contact_logits,
            "pred_cross_contact_prob": torch.sigmoid(pred_cross_contact_logits),
        }

    def encode_points(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        points = batch["points"].float()
        normals = batch["normals"].float()
        point_valid_mask = batch["point_valid_mask"].bool()
        obj_valid_mask = batch["runtime_obj_valid_mask"].bool()
        if "supervision_edge_idx" not in batch or "supervision_edge_valid_mask" not in batch:
            raise KeyError("v2 batch must contain supervision_edge_idx and supervision_edge_valid_mask.")

        expected_total = self.num_obj_points + self.num_hand_points
        obj_points = points[:, : self.num_obj_points]
        obj_normals = normals[:, : self.num_obj_points]
        hand_points = points[:, self.num_obj_points : expected_total]
        hand_normals = normals[:, self.num_obj_points : expected_total]

        obj_feat, hand_feat = self._build_point_features(
            obj_points=obj_points,
            obj_normals=obj_normals,
            hand_points=hand_points,
            hand_normals=hand_normals,
            obj_valid_mask=obj_valid_mask,
        )
        coord = torch.cat([obj_points, hand_points], dim=1)
        feat = torch.cat([obj_feat, hand_feat], dim=1)
        tokens = self.backbone(feat, coord, point_valid_mask)
        z_obj = tokens[:, : self.num_obj_points]
        z_hand = tokens[:, self.num_obj_points : expected_total]
        return z_obj, z_hand

    def predict_dense_cross_for_object(
        self,
        batch: dict[str, torch.Tensor],
        obj_idx: int,
    ) -> torch.Tensor:
        z_obj, z_hand = self.encode_points(batch)
        obj_token = z_obj[:, obj_idx : obj_idx + 1]
        hand_token = z_hand
        edge_input = torch.cat(
            [
                obj_token.unsqueeze(2).expand(-1, -1, hand_token.shape[1], -1),
                hand_token.unsqueeze(1),
            ],
            dim=-1,
        )
        edge_shared = self.edge_shared_backbone(edge_input)
        dense_logits = self.cross_edge_head(edge_shared).squeeze(1).squeeze(-1)
        return torch.sigmoid(dense_logits)

    def _build_point_features(
        self,
        *,
        obj_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obj_extra = self._compute_opposite_cloud_features(
            query_points=obj_points,
            query_normals=obj_normals,
            opposite_points=hand_points,
            opposite_valid_mask=None,
            query_valid_mask=obj_valid_mask,
        )
        hand_extra = self._compute_opposite_cloud_features(
            query_points=hand_points,
            query_normals=hand_normals,
            opposite_points=obj_points,
            opposite_valid_mask=obj_valid_mask,
            query_valid_mask=None,
        )
        obj_type = obj_points.new_zeros(obj_points.shape[0], obj_points.shape[1], 2)
        obj_type[..., 0] = 1.0
        hand_type = hand_points.new_zeros(hand_points.shape[0], hand_points.shape[1], 2)
        hand_type[..., 1] = 1.0
        obj_feat = torch.cat([obj_points, obj_type, obj_normals, obj_extra], dim=-1)
        hand_feat = torch.cat([hand_points, hand_type, hand_normals, hand_extra], dim=-1)
        obj_feat = obj_feat * obj_valid_mask.unsqueeze(-1).float()
        return obj_feat, hand_feat

    @staticmethod
    def _compute_opposite_cloud_features(
        *,
        query_points: torch.Tensor,
        query_normals: torch.Tensor,
        opposite_points: torch.Tensor,
        opposite_valid_mask: torch.Tensor | None,
        query_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, num_query, _ = query_points.shape
        num_opp = opposite_points.shape[1]
        distance = torch.cdist(query_points, opposite_points)
        if opposite_valid_mask is None:
            opposite_valid_mask = torch.ones(batch_size, num_opp, device=query_points.device, dtype=torch.bool)
        masked_distance = distance.masked_fill(~opposite_valid_mask.unsqueeze(1), float("inf"))
        any_opp_valid = opposite_valid_mask.any(dim=-1)
        safe_masked_distance = torch.where(any_opp_valid.view(batch_size, 1, 1), masked_distance, torch.zeros_like(masked_distance))
        nn_idx = torch.argmin(safe_masked_distance, dim=-1)
        batch_idx = torch.arange(batch_size, device=query_points.device).view(-1, 1)
        nn_points = opposite_points[batch_idx, nn_idx]
        delta_nn = nn_points - query_points
        nn_dist = torch.norm(delta_nn, dim=-1, keepdim=True)
        dir_nn = delta_nn / nn_dist.clamp(min=1e-8)
        opp_valid_f = opposite_valid_mask.float()
        opp_count = opp_valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        opposite_centroid = (opposite_points * opp_valid_f.unsqueeze(-1)).sum(dim=1) / opp_count
        delta_cent = opposite_centroid.unsqueeze(1) - query_points
        cent_dist = torch.norm(delta_cent, dim=-1, keepdim=True)
        dir_cent = delta_cent / cent_dist.clamp(min=1e-8)
        proj_nn = torch.sum(dir_nn * query_normals, dim=-1, keepdim=True)
        proj_cent = torch.sum(dir_cent * query_normals, dim=-1, keepdim=True)
        features = torch.cat([nn_dist, proj_nn, proj_cent], dim=-1)
        if query_valid_mask is not None:
            features = features * query_valid_mask.unsqueeze(-1).float()
        return torch.where(any_opp_valid.view(batch_size, 1, 1), features, torch.zeros_like(features))

    @staticmethod
    def _gather_batched_knn_features(
        features: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        gathered: list[torch.Tensor] = []
        for batch_idx in range(features.shape[0]):
            gathered.append(
                gather_knn_features(
                    features[batch_idx],
                    knn_idx[batch_idx],
                    knn_valid_mask[batch_idx],
                )
            )
        return torch.stack(gathered, dim=0)

    def _compute_shared_edge_features(
        self,
        *,
        z_obj: torch.Tensor,
        z_hand_neighbors: torch.Tensor,
    ) -> torch.Tensor:
        num_edges = z_hand_neighbors.shape[2]
        z_obj_expanded = z_obj.unsqueeze(2).expand(-1, -1, num_edges, -1)
        edge_input = torch.cat([z_obj_expanded, z_hand_neighbors], dim=-1)
        return self.edge_shared_backbone(edge_input)
