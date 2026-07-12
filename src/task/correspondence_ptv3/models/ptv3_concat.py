from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

from src.task.correspondence_ptv3.models.common import PTv3DenseBackbone, _build_mlp, masked_softmax
from src.task.correspondence_ptv3.supervision.base import ContactSupervision
from src.utils.correspondence import compute_obj_to_hand_edge_features, gather_knn_features


class PTv3ConcatModel(nn.Module):
    backbone_cls = PTv3DenseBackbone

    def __init__(
        self,
        task_meta: Any,
        *,
        contact_supervision: ContactSupervision,
        point_feat_dim: int,
        use_cross_attn: bool,
        use_cano_head: bool,
        use_finger_region_head: bool,
        ptv3: Mapping[str, Any],
        condition_shape: list[int] | tuple[int, ...] | None = None,
        target_shape: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        del condition_shape, target_shape
        super().__init__()
        self.task_meta = task_meta

        self.num_obj_points = int(task_meta.num_obj_points)
        self.num_hand_points = int(task_meta.num_hand_points)
        self.num_fingers = int(task_meta.num_fingers)
        self.num_regions = int(task_meta.num_regions)
        self.point_feat_dim = int(point_feat_dim)
        self.contact_supervision = contact_supervision
        self.contact_supervision_mode = self.contact_supervision.name
        self.contact_output_dim = self.contact_supervision.output_dim
        self.contact_bin_decode_mode = str(getattr(self.contact_supervision, "decode_mode", "expectation"))
        self.use_cross_attn = bool(use_cross_attn)
        self.use_finger_region_head = bool(use_finger_region_head)
        self.use_cano_head = bool(use_cano_head)

        self.backbone = self.backbone_cls(ptv3, in_channels=self.point_feat_dim)
        self.token_dim = int(self.backbone.output_dim)

        if self.use_cross_attn:
            self.edge_geo_dim = self.token_dim // 2
            self.edge_geo_mlp = _build_mlp(6, self.token_dim // 2, self.edge_geo_dim)
            self.cross_q = nn.Linear(self.token_dim, self.token_dim)
            self.cross_k = nn.Linear(self.token_dim + self.edge_geo_dim, self.token_dim)
            self.cross_v = nn.Linear(self.token_dim + self.edge_geo_dim, self.token_dim)
            self.cross_bias = _build_mlp(6, self.token_dim // 2, 1)
            self.cross_out = _build_mlp(self.token_dim, self.token_dim, self.token_dim)

        self.contact_head = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim // 2),
            nn.GELU(),
            nn.Linear(self.token_dim // 2, self.contact_output_dim),
        )
        if self.use_cano_head:
            self.cano_head = nn.Sequential(
                nn.Linear(self.token_dim, self.token_dim // 2),
                nn.GELU(),
                nn.Linear(self.token_dim // 2, 3),
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

        cross_edge_input_dim = self.token_dim * 2
        self.edge_shared_dim = self.token_dim // 2
        self.edge_shared_backbone = nn.Sequential(
            nn.Linear(cross_edge_input_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, self.edge_shared_dim),
            nn.GELU(),
        )
        self.cross_edge_head = nn.Linear(self.edge_shared_dim, self.contact_output_dim)

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

        point_valid_mask = batch.get("point_valid_mask")
        if point_valid_mask is None:
            point_valid_mask = torch.ones(batch_size, expected_total, device=points.device, dtype=torch.bool)
        elif point_valid_mask.dim() == 1:
            point_valid_mask = point_valid_mask.unsqueeze(0).expand(batch_size, -1)
        point_valid_mask = point_valid_mask[:, :expected_total].bool()
        obj_valid_mask = batch.get("runtime_obj_valid_mask")
        if obj_valid_mask is None:
            obj_valid_mask = point_valid_mask[:, : self.num_obj_points]
        elif obj_valid_mask.dim() == 1:
            obj_valid_mask = obj_valid_mask.unsqueeze(0).expand(batch_size, -1)
        obj_valid_mask = obj_valid_mask.bool()

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
        ctx_idx = batch.get("input_obj_to_hand_ctx_idx")
        ctx_valid = batch.get("input_obj_to_hand_ctx_valid_mask")
        if ctx_idx is None or ctx_valid is None:
            ctx_idx = batch["input_obj_to_hand_knn_idx"]
            ctx_valid = batch["input_obj_to_hand_knn_valid_mask"]
        ctx_idx = ctx_idx.long()
        ctx_valid = ctx_valid.bool()

        logit_idx = batch.get("input_obj_to_hand_logit_idx")
        logit_valid = batch.get("input_obj_to_hand_logit_valid_mask")
        if logit_idx is None or logit_valid is None:
            logit_idx = ctx_idx
            logit_valid = ctx_valid
        logit_idx = logit_idx.long()
        logit_valid = logit_valid.bool()

        if self.use_cross_attn:
            z_obj_cross, obj_to_hand_attn = self._compute_obj_cross_context(
                z_obj=z_obj,
                z_hand=z_hand,
                obj_points=obj_points,
                hand_points=hand_points,
                obj_normals=obj_normals,
                hand_normals=hand_normals,
                obj_to_hand_knn_idx=ctx_idx,
                obj_to_hand_knn_valid_mask=ctx_valid,
            )
        else:
            z_obj_cross = z_obj
            obj_to_hand_attn = z_obj.new_zeros(
                batch_size,
                self.num_obj_points,
                ctx_idx.shape[-1],
            )

        z_hand_logit_neighbors = self._gather_batched_knn_features(
            z_hand,
            logit_idx,
            logit_valid,
        )

        pred_obj_contact_logits = self.contact_head(z_obj_cross)
        outputs = {
            "pred_obj_contact_logits": pred_obj_contact_logits,
            "obj_dense_tokens": z_obj_cross,
            "hand_dense_tokens": z_hand,
            "ptv3_obj_tokens": z_obj,
            "obj_to_hand_attn": obj_to_hand_attn,
        }
        if self.use_cano_head:
            outputs["pred_obj_cano"] = self.cano_head(z_obj_cross)

        edge_shared = self._compute_shared_edge_features(
            z_obj_cross=z_obj_cross,
            z_hand_neighbors=z_hand_logit_neighbors,
        )
        pred_cross_contact_logits = self._compute_cross_edge_predictions(
            edge_shared=edge_shared,
        )
        outputs["pred_cross_contact_logits"] = pred_cross_contact_logits
        outputs.update(
            self.contact_supervision.legacy_output_aliases(
                pred_obj_contact_logits=pred_obj_contact_logits,
                pred_cross_contact_logits=pred_cross_contact_logits,
            )
        )

        pred_obj_contact_prob = self.contact_supervision.decode(pred_obj_contact_logits)
        pred_cross_contact_prob = self.contact_supervision.decode(pred_cross_contact_logits)
        outputs["pred_obj_contact_prob"] = pred_obj_contact_prob
        outputs["pred_cross_contact_prob"] = pred_cross_contact_prob
        outputs["pred_obj_contact"] = self.contact_supervision.export_compat(
            pred_obj_contact_logits,
            pred_obj_contact_prob,
        )
        outputs["pred_cross_contact"] = self.contact_supervision.export_compat(
            pred_cross_contact_logits,
            pred_cross_contact_prob,
        )

        if self.use_finger_region_head:
            outputs["pred_obj_finger"] = self.finger_head(z_obj_cross)
            outputs["pred_obj_region"] = self.region_head(z_obj_cross)

        return outputs

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
            opposite_valid_mask = torch.ones(
                batch_size,
                num_opp,
                device=query_points.device,
                dtype=torch.bool,
            )
        else:
            opposite_valid_mask = opposite_valid_mask.bool()

        masked_distance = distance.masked_fill(
            ~opposite_valid_mask.unsqueeze(1),
            float("inf"),
        )
        any_opp_valid = opposite_valid_mask.any(dim=-1)
        safe_masked_distance = torch.where(
            any_opp_valid.view(batch_size, 1, 1),
            masked_distance,
            torch.zeros_like(masked_distance),
        )
        nn_idx = torch.argmin(safe_masked_distance, dim=-1)
        batch_idx = torch.arange(batch_size, device=query_points.device).view(-1, 1)
        nn_points = opposite_points[batch_idx, nn_idx]
        delta_nn = nn_points - query_points
        nn_dist = torch.norm(delta_nn, dim=-1, keepdim=True)
        dir_nn = delta_nn / nn_dist.clamp(min=1e-8)

        opp_valid_f = opposite_valid_mask.float()
        opp_count = opp_valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        opposite_centroid = (
            opposite_points * opp_valid_f.unsqueeze(-1)
        ).sum(dim=1) / opp_count
        delta_cent = opposite_centroid.unsqueeze(1) - query_points
        cent_dist = torch.norm(delta_cent, dim=-1, keepdim=True)
        dir_cent = delta_cent / cent_dist.clamp(min=1e-8)

        proj_nn = torch.sum(dir_nn * query_normals, dim=-1, keepdim=True)
        proj_cent = torch.sum(dir_cent * query_normals, dim=-1, keepdim=True)

        features = torch.cat([nn_dist, proj_nn, proj_cent], dim=-1)
        if query_valid_mask is not None:
            features = features * query_valid_mask.unsqueeze(-1).float()
        features = torch.where(
            any_opp_valid.view(batch_size, 1, 1),
            features,
            torch.zeros_like(features),
        )
        return features

    def _compute_obj_cross_context(
        self,
        z_obj: torch.Tensor,
        z_hand: torch.Tensor,
        obj_points: torch.Tensor,
        hand_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_normals: torch.Tensor,
        obj_to_hand_knn_idx: torch.Tensor,
        obj_to_hand_knn_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = z_obj.shape[0]
        edge_geo_list: list[torch.Tensor] = []
        z_hand_neighbors_list: list[torch.Tensor] = []

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
        edge_geo = torch.stack(edge_geo_list, dim=0)
        z_hand_neighbors = torch.stack(z_hand_neighbors_list, dim=0)

        edge_geo_embed = self.edge_geo_mlp(edge_geo)
        q = self.cross_q(z_obj).unsqueeze(2)
        k = self.cross_k(torch.cat([z_hand_neighbors, edge_geo_embed], dim=-1))
        v = self.cross_v(torch.cat([z_hand_neighbors, edge_geo_embed], dim=-1))

        logits = (q * k).sum(dim=-1) / (self.token_dim**0.5)
        logits = logits + self.cross_bias(edge_geo).squeeze(-1)
        attn = masked_softmax(logits, obj_to_hand_knn_valid_mask)
        cross_ctx = torch.sum(attn.unsqueeze(-1) * v, dim=2)
        valid_obj_mask = obj_to_hand_knn_valid_mask.any(dim=-1, keepdim=True).float()
        cross_update = self.cross_out(cross_ctx) * valid_obj_mask
        z_obj_cross = z_obj + cross_update
        return z_obj_cross, attn

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
        z_obj_cross: torch.Tensor,
        z_hand_neighbors: torch.Tensor,
    ) -> torch.Tensor:
        k_cross = z_hand_neighbors.shape[2]
        z_obj_expanded = z_obj_cross.unsqueeze(2).expand(-1, -1, k_cross, -1)
        edge_input = torch.cat([z_obj_expanded, z_hand_neighbors], dim=-1)
        return self.edge_shared_backbone(edge_input)

    def _compute_cross_edge_predictions(
        self,
        edge_shared: torch.Tensor,
    ) -> torch.Tensor:
        return self.cross_edge_head(edge_shared)


CorrespondencePTV3Model = PTv3ConcatModel
