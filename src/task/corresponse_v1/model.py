from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from src.base import MLP
from src.utils.correspondence import (
    compute_cross_edge_features,
    compute_local_edge_features,
    gather_knn_features,
)


class EdgeConvBlock(nn.Module):
    """EdgeConv block for local graph message passing.

    For each point i with neighbor j:
        message_ij = MLP([z_i, z_j - z_i, rel_xyz_ij])
        z_i = z_i + aggregate(message_ij)
    """

    def __init__(self, dim: int, mlp_ratio: int = 2, activation: str = "gelu") -> None:
        super().__init__()
        self.mlp = MLP(
            input_dim=dim * 2 + 3,  # [z_i, z_j - z_i, rel_xyz]
            output_dim=dim,
            hidden_dims=[dim * mlp_ratio],
            activation=activation,
            output_activation=None,
        )

    def forward(
        self,
        z: torch.Tensor,
        points: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: [N, D] point features.
            points: [N, 3] point coordinates.
            knn_idx: [N, K] neighbor indices.
            knn_valid_mask: [N, K] valid mask.

        Returns:
            z_out: [N, D] updated features.
        """
        knn_idx = knn_idx.long()
        N, D = z.shape
        K = knn_idx.shape[1]

        # Gather neighbor features: [N, K, D]
        z_neighbors = gather_knn_features(z, knn_idx, knn_valid_mask)

        # Relative features: [N, K, D]
        z_query = z.unsqueeze(1).expand(-1, K, -1)
        z_diff = z_neighbors - z_query

        # Relative positions: [N, K, 3]
        rel_xyz, _ = compute_local_edge_features(points, knn_idx, knn_valid_mask)

        # Edge features: [N, K, D*2 + 3]
        edge_input = torch.cat([z_query, z_diff, rel_xyz], dim=-1)

        # Messages: [N, K, D]
        messages = self.mlp(edge_input)

        # Mask invalid edges
        mask = knn_valid_mask.unsqueeze(-1).float()  # [N, K, 1]
        messages = messages * mask

        # Mean aggregation over neighbors
        neighbor_counts = mask.sum(dim=1).clamp(min=1)  # [N, 1]
        aggregated = messages.sum(dim=1) / neighbor_counts  # [N, D]

        # Residual connection
        return z + aggregated


class CrossGraphBlock(nn.Module):
    """Cross-graph message passing block: hand -> object.

    Updates object tokens by aggregating messages from nearby hand points.
    """

    def __init__(self, dim: int, mlp_ratio: int = 2, activation: str = "gelu") -> None:
        super().__init__()
        # Edge features: [z_obj, z_hand, delta, dist, signed_dist, normal_dot]
        # delta(3) + dist(1) + signed_dist(1) + normal_dot(1) = 6
        edge_feat_dim = 6
        self.mlp = MLP(
            input_dim=dim * 2 + edge_feat_dim,
            output_dim=dim,
            hidden_dims=[dim * mlp_ratio],
            activation=activation,
            output_activation=None,
        )

    def forward(
        self,
        z_obj: torch.Tensor,
        z_hand: torch.Tensor,
        obj_points: torch.Tensor,
        hand_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_to_obj_knn_idx: torch.Tensor,
        hand_to_obj_knn_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: update object tokens from hand neighbors.

        Args:
            z_obj: [No, D] object features.
            z_hand: [Nh, D] hand features.
            obj_points: [No, 3] object coordinates.
            hand_points: [Nh, 3] hand coordinates.
            obj_normals: [No, 3] object normals.
            hand_normals: [Nh, 3] hand normals.
            hand_to_obj_knn_idx: [Nh, K_cross] object neighbor indices.
            hand_to_obj_knn_valid_mask: [Nh, K_cross] valid mask.

        Returns:
            z_obj_out: [No, D] updated object features.
        """
        hand_to_obj_knn_idx = hand_to_obj_knn_idx.long()
        No, D = z_obj.shape
        Nh = z_hand.shape[0]
        K = hand_to_obj_knn_idx.shape[1]

        # Compute cross edge geometric features
        edge_feats = compute_cross_edge_features(
            hand_points=hand_points,
            obj_points=obj_points,
            hand_normals=hand_normals,
            obj_normals=obj_normals,
            hand_to_obj_knn_idx=hand_to_obj_knn_idx,
            hand_to_obj_knn_valid_mask=hand_to_obj_knn_valid_mask,
        )
        # edge_feats: delta(3), dist(1), signed_dist(1), normal_dot(1) -> [Nh, K, 6]
        edge_geo = torch.cat(
            [edge_feats["delta"], edge_feats["dist"], edge_feats["signed_dist"], edge_feats["normal_dot"]],
            dim=-1,
        )

        # Gather object features for each hand-object edge: [Nh, K, D]
        z_obj_neighbors = gather_knn_features(z_obj, hand_to_obj_knn_idx, hand_to_obj_knn_valid_mask)

        # Hand features: [Nh, 1, D]
        z_hand_expanded = z_hand.unsqueeze(1).expand(-1, K, -1)

        # Edge input: [Nh, K, D*2 + 6]
        edge_input = torch.cat([z_obj_neighbors, z_hand_expanded, edge_geo], dim=-1)

        # Messages: [Nh, K, D]
        messages = self.mlp(edge_input)

        # Mask invalid edges
        mask = hand_to_obj_knn_valid_mask.unsqueeze(-1).float()  # [Nh, K, 1]
        messages = messages * mask

        # Scatter messages back to object points
        # hand_to_obj_knn_idx: [Nh, K] -> object index for each (hand, k)
        # We need to sum messages from all hand points targeting each object point
        z_obj_update = torch.zeros(No, D, device=z_obj.device, dtype=z_obj.dtype)
        flat_idx = hand_to_obj_knn_idx.clamp(min=0).reshape(-1)  # [Nh*K]
        flat_messages = messages.reshape(-1, D)  # [Nh*K, D]
        flat_mask = mask.reshape(-1)  # [Nh*K]

        # Only valid edges contribute
        valid_mask_bool = flat_mask.squeeze(-1) > 0.5
        if valid_mask_bool.any():
            valid_idx = flat_idx[valid_mask_bool].long()
            valid_messages = flat_messages[valid_mask_bool]
            z_obj_update.scatter_add_(0, valid_idx.unsqueeze(-1).expand(-1, D), valid_messages)

            # Normalize by in-degree
            in_degree = torch.zeros(No, device=z_obj.device, dtype=z_obj.dtype)
            in_degree.scatter_add_(0, valid_idx, torch.ones_like(valid_idx, dtype=z_obj.dtype))
            in_degree = in_degree.clamp(min=1).unsqueeze(-1)
            z_obj_update = z_obj_update / in_degree

        return z_obj + z_obj_update


class StaticHOCNet(nn.Module):
    """Static Hand-Object Correspondence Network (Small version).

    Architecture:
        1. Input embedding (separate for object and hand points)
        2. Object local encoder (EdgeConv, K=16)
        3. Hand local encoder (EdgeConv, K=16)
        4. Cross interaction encoder (Hand->Object messages, K=32)
        5. Output heads:
           - obj_contact: BCE
           - obj_to_hand_cano: SmoothL1
           - obj_to_hand_finger: CrossEntropy (optional)
           - obj_to_hand_region: CrossEntropy (optional)
           - cross_edge_contact: BCE
    """

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

        self.hidden_dim = int(meta.hidden_dim)
        self.num_obj_points = int(meta.num_obj_points)
        self.num_hand_points = int(meta.num_hand_points)
        self.k_obj_local = int(meta.k_obj_local)
        self.k_hand_local = int(meta.k_hand_local)
        self.k_cross = int(meta.k_cross)
        self.num_local_layers = int(meta.num_local_layers)
        self.num_cross_layers = int(meta.num_cross_layers)
        self.mlp_ratio = int(meta.mlp_ratio)
        self.num_fingers = int(meta.num_fingers)
        self.num_regions = int(meta.num_regions)

        type_emb_dim = int(meta.type_emb_dim)
        finger_emb_dim = int(meta.finger_emb_dim)
        region_emb_dim = int(meta.region_emb_dim)

        # ---- Input embeddings ----
        # Object input: xyz(3) + normal(3) + type_emb(16) = 22
        obj_input_dim = 3 + 3 + type_emb_dim
        self.obj_type_embedding = nn.Embedding(2, type_emb_dim)  # 0=object, 1=hand
        self.obj_embed_mlp = MLP(
            input_dim=obj_input_dim,
            output_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim],
            activation="gelu",
            output_activation=None,
        )

        # Hand input: xyz(3) + normal(3) + type_emb(16) + finger_emb(16) + region_emb(16) + cano_xyz(3) = 57
        hand_input_dim = 3 + 3 + type_emb_dim + finger_emb_dim + region_emb_dim + 3
        self.hand_type_embedding = nn.Embedding(2, type_emb_dim)
        self.finger_embedding = nn.Embedding(self.num_fingers + 1, finger_emb_dim, padding_idx=0)
        self.region_embedding = nn.Embedding(self.num_regions + 1, region_emb_dim, padding_idx=0)
        self.hand_embed_mlp = MLP(
            input_dim=hand_input_dim,
            output_dim=self.hidden_dim,
            hidden_dims=[self.hidden_dim],
            activation="gelu",
            output_activation=None,
        )

        # ---- Local encoders ----
        self.obj_local_blocks = nn.ModuleList([
            EdgeConvBlock(self.hidden_dim, self.mlp_ratio, "gelu")
            for _ in range(self.num_local_layers)
        ])
        self.hand_local_blocks = nn.ModuleList([
            EdgeConvBlock(self.hidden_dim, self.mlp_ratio, "gelu")
            for _ in range(self.num_local_layers)
        ])

        # ---- Cross interaction encoder ----
        self.cross_blocks = nn.ModuleList([
            CrossGraphBlock(self.hidden_dim, self.mlp_ratio, "gelu")
            for _ in range(self.num_cross_layers)
        ])

        # ---- Output heads ----
        # Head 1: object contact probability
        self.contact_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        # Head 2: object-to-hand canonical correspondence
        self.cano_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 3),
        )

        # Head 3: object-to-hand finger classification
        self.finger_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.num_fingers),
        )

        # Head 4: object-to-hand region classification
        self.region_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.num_regions),
        )

        # Head 5: cross edge contact probability
        cross_edge_input_dim = self.hidden_dim * 2 + 6  # z_hand + z_obj + edge_geo
        self.cross_edge_head = nn.Sequential(
            nn.Linear(cross_edge_input_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def _embed_object_points(
        self,
        obj_points: torch.Tensor,
        obj_normals: torch.Tensor,
    ) -> torch.Tensor:
        """Embed object points into hidden features.

        Args:
            obj_points: [B, No, 3]
            obj_normals: [B, No, 3]

        Returns:
            z_obj: [B, No, D]
        """
        B, No, _ = obj_points.shape
        type_id = torch.zeros(B, No, dtype=torch.long, device=obj_points.device)
        type_emb = self.obj_type_embedding(type_id)  # [B, No, type_emb_dim]
        obj_input = torch.cat([obj_points, obj_normals, type_emb], dim=-1)  # [B, No, 22]
        z_obj = self.obj_embed_mlp(obj_input)  # [B, No, D]
        return z_obj

    def _embed_hand_points(
        self,
        hand_points: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_cano_points: torch.Tensor,
        finger_id: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        """Embed hand points into hidden features.

        Args:
            hand_points: [B, Nh, 3]
            hand_normals: [B, Nh, 3]
            hand_cano_points: [B, Nh, 3]
            finger_id: [B, Nh]
            region_id: [B, Nh]

        Returns:
            z_hand: [B, Nh, D]
        """
        B, Nh, _ = hand_points.shape
        type_id = torch.ones(B, Nh, dtype=torch.long, device=hand_points.device)
        type_emb = self.hand_type_embedding(type_id)  # [B, Nh, type_emb_dim]

        # Shift finger_id and region_id by +1 to use 0 as padding_idx
        finger_id_shifted = finger_id.clamp(min=-1) + 1
        region_id_shifted = region_id.clamp(min=-1) + 1
        finger_emb = self.finger_embedding(finger_id_shifted)  # [B, Nh, finger_emb_dim]
        region_emb = self.region_embedding(region_id_shifted)  # [B, Nh, region_emb_dim]

        hand_input = torch.cat(
            [hand_points, hand_normals, type_emb, finger_emb, region_emb, hand_cano_points],
            dim=-1,
        )  # [B, Nh, 57]
        z_hand = self.hand_embed_mlp(hand_input)  # [B, Nh, D]
        return z_hand

    def _apply_local_encoder(
        self,
        z: torch.Tensor,
        points: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_valid_mask: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        """Apply local encoder blocks.

        Args:
            z: [B, N, D] features.
            points: [B, N, 3] coordinates.
            knn_idx: [B, N, K] neighbor indices.
            knn_valid_mask: [B, N, K] valid mask.
            blocks: list of EdgeConvBlock.

        Returns:
            z: [B, N, D] updated features.
        """
        B = z.shape[0]
        for block in blocks:
            z_new = []
            for b in range(B):
                z_new.append(block(z[b], points[b], knn_idx[b], knn_valid_mask[b]))
            z = torch.stack(z_new, dim=0)
        return z

    def _apply_cross_encoder(
        self,
        z_obj: torch.Tensor,
        z_hand: torch.Tensor,
        obj_points: torch.Tensor,
        hand_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_to_obj_knn_idx: torch.Tensor,
        hand_to_obj_knn_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross interaction encoder blocks.

        Returns:
            z_obj: [B, No, D] updated object features.
        """
        B = z_obj.shape[0]
        for block in self.cross_blocks:
            z_obj_new = []
            for b in range(B):
                z_obj_new.append(
                    block(
                        z_obj=z_obj[b],
                        z_hand=z_hand[b],
                        obj_points=obj_points[b],
                        hand_points=hand_points[b],
                        obj_normals=obj_normals[b],
                        hand_normals=hand_normals[b],
                        hand_to_obj_knn_idx=hand_to_obj_knn_idx[b],
                        hand_to_obj_knn_valid_mask=hand_to_obj_knn_valid_mask[b],
                    )
                )
            z_obj = torch.stack(z_obj_new, dim=0)
        return z_obj

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: dict with keys:
                points: [B, N_total, 3]
                normals: [B, N_total, 3]
                point_type_id: [B, N_total] or [N_total]
                finger_id: [B, N_total] or [N_total]
                hand_region_id: [B, N_total] or [N_total]
                hand_cano_points: [B, N_total, 3] or [N_total, 3]
                obj_local_knn_idx: [B, No, K_obj_local]
                obj_local_knn_valid_mask: [B, No, K_obj_local]
                hand_local_knn_idx: [B, Nh, K_hand_local]
                hand_local_knn_valid_mask: [B, Nh, K_hand_local]
                hand_to_obj_knn_idx: [B, Nh, K_cross]
                hand_to_obj_knn_valid_mask: [B, Nh, K_cross]
        point_valid_mask: [B, N_total] (optional)

        Returns:
            dict with keys:
                pred_obj_contact: [B, No]
                pred_obj_to_hand_cano: [B, No, 3]
                pred_obj_to_hand_finger: [B, No, num_fingers]
                pred_obj_to_hand_region: [B, No, num_regions]
                pred_cross_contact: [B, Nh, K_cross]
        """
        points = batch["points"]  # [B, N_total, 3]
        normals = batch["normals"]  # [B, N_total, 3]
        B, N_total, _ = points.shape
        No = self.num_obj_points
        Nh = self.num_hand_points

        # Split into object and hand parts
        obj_points = points[:, :No]  # [B, No, 3]
        obj_normals = normals[:, :No]  # [B, No, 3]
        hand_points = points[:, No:]  # [B, Nh, 3]
        hand_normals = normals[:, No:]  # [B, Nh, 3]

        # Get hand semantics
        finger_id = batch["finger_id"]  # [B, N_total] or [N_total]
        hand_region_id = batch["hand_region_id"]  # [B, N_total] or [N_total]
        hand_cano_points = batch["hand_cano_points"]  # [B, N_total, 3] or [N_total, 3]

        # Handle 2D vs 3D tensors
        if finger_id.dim() == 1:
            finger_id = finger_id.unsqueeze(0).expand(B, -1)
        if hand_region_id.dim() == 1:
            hand_region_id = hand_region_id.unsqueeze(0).expand(B, -1)
        if hand_cano_points.dim() == 2:
            hand_cano_points = hand_cano_points.unsqueeze(0).expand(B, -1, -1)

        hand_finger_id = finger_id[:, No:]  # [B, Nh]
        hand_region_id = hand_region_id[:, No:]  # [B, Nh]
        hand_cano = hand_cano_points[:, No:]  # [B, Nh, 3]

        # ---- 1. Input embedding ----
        z_obj = self._embed_object_points(obj_points, obj_normals)  # [B, No, D]
        z_hand = self._embed_hand_points(hand_points, hand_normals, hand_cano, hand_finger_id, hand_region_id)  # [B, Nh, D]

        # ---- 2. Local encoders ----
        obj_local_knn_idx = batch["obj_local_knn_idx"].long()  # [B, No, K_obj_local]
        obj_local_knn_valid_mask = batch["obj_local_knn_valid_mask"]  # [B, No, K_obj_local]
        hand_local_knn_idx = batch["hand_local_knn_idx"].long()  # [B, Nh, K_hand_local]
        hand_local_knn_valid_mask = batch["hand_local_knn_valid_mask"]  # [B, Nh, K_hand_local]

        z_obj = self._apply_local_encoder(z_obj, obj_points, obj_local_knn_idx, obj_local_knn_valid_mask, self.obj_local_blocks)
        z_hand = self._apply_local_encoder(z_hand, hand_points, hand_local_knn_idx, hand_local_knn_valid_mask, self.hand_local_blocks)

        # ---- 3. Cross interaction encoder ----
        hand_to_obj_knn_idx = batch["hand_to_obj_knn_idx"].long()  # [B, Nh, K_cross]
        hand_to_obj_knn_valid_mask = batch["hand_to_obj_knn_valid_mask"]  # [B, Nh, K_cross]

        z_obj = self._apply_cross_encoder(
            z_obj=z_obj,
            z_hand=z_hand,
            obj_points=obj_points,
            hand_points=hand_points,
            obj_normals=obj_normals,
            hand_normals=hand_normals,
            hand_to_obj_knn_idx=hand_to_obj_knn_idx,
            hand_to_obj_knn_valid_mask=hand_to_obj_knn_valid_mask,
        )

        # ---- 4. Output heads ----
        # Head 1: obj contact probability
        pred_obj_contact = self.contact_head(z_obj).squeeze(-1)  # [B, No]

        # Head 2: obj-to-hand canonical correspondence
        pred_obj_to_hand_cano = self.cano_head(z_obj)  # [B, No, 3]

        # Head 3: obj-to-hand finger classification
        pred_obj_to_hand_finger = self.finger_head(z_obj)  # [B, No, num_fingers]

        # Head 4: obj-to-hand region classification
        pred_obj_to_hand_region = self.region_head(z_obj)  # [B, No, num_regions]

        # Head 5: cross edge contact
        pred_cross_contact = self._compute_cross_edge_predictions(
            z_obj=z_obj,
            z_hand=z_hand,
            obj_points=obj_points,
            hand_points=hand_points,
            obj_normals=obj_normals,
            hand_normals=hand_normals,
            hand_to_obj_knn_idx=hand_to_obj_knn_idx,
            hand_to_obj_knn_valid_mask=hand_to_obj_knn_valid_mask,
        )  # [B, Nh, K_cross]

        return {
            "pred_obj_contact": pred_obj_contact,
            "pred_obj_to_hand_cano": pred_obj_to_hand_cano,
            "pred_obj_to_hand_finger": pred_obj_to_hand_finger,
            "pred_obj_to_hand_region": pred_obj_to_hand_region,
            "pred_cross_contact": pred_cross_contact,
        }

    def _compute_cross_edge_predictions(
        self,
        z_obj: torch.Tensor,
        z_hand: torch.Tensor,
        obj_points: torch.Tensor,
        hand_points: torch.Tensor,
        obj_normals: torch.Tensor,
        hand_normals: torch.Tensor,
        hand_to_obj_knn_idx: torch.Tensor,
        hand_to_obj_knn_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross edge contact predictions.

        Returns:
            pred_cross_contact: [B, Nh, K_cross]
        """
        B, Nh, D = z_hand.shape
        K = hand_to_obj_knn_idx.shape[2]
        No = z_obj.shape[1]

        preds = []
        for b in range(B):
            # Compute edge features
            edge_feats = compute_cross_edge_features(
                hand_points=hand_points[b],
                obj_points=obj_points[b],
                hand_normals=hand_normals[b],
                obj_normals=obj_normals[b],
                hand_to_obj_knn_idx=hand_to_obj_knn_idx[b],
                hand_to_obj_knn_valid_mask=hand_to_obj_knn_valid_mask[b],
            )
            edge_geo = torch.cat(
                [edge_feats["delta"], edge_feats["dist"], edge_feats["signed_dist"], edge_feats["normal_dot"]],
                dim=-1,
            )  # [Nh, K, 6]

            # Gather object features
            z_obj_neighbors = gather_knn_features(z_obj[b], hand_to_obj_knn_idx[b], hand_to_obj_knn_valid_mask[b])  # [Nh, K, D]

            # Hand features
            z_hand_expanded = z_hand[b].unsqueeze(1).expand(-1, K, -1)  # [Nh, K, D]

            # Edge input
            edge_input = torch.cat([z_hand_expanded, z_obj_neighbors, edge_geo], dim=-1)  # [Nh, K, D*2+6]

            # Predict
            pred = self.cross_edge_head(edge_input).squeeze(-1)  # [Nh, K]
            preds.append(pred)

        return torch.stack(preds, dim=0)  # [B, Nh, K_cross]
