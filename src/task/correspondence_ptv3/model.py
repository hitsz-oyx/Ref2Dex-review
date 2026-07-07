# =============================================================================
# correspondence_ptv3 模型定义模块
# =============================================================================
# 本文件实现 correspondence_ptv3 任务的神经网络模型。
#
# 主要组件：
#   1. _build_stem / _build_mlp : 通用的小型 MLP 构建器（stem / 投影头）；
#   2. _masked_softmax          : 支持布尔 mask 的 softmax 实现；
#   3. _load_ptv3_model_class   : 动态加载 PointTransformerV3 官方实现；
#   4. PTv3DenseBackbone        : 把 PTv3 适配为"密集张量 (B, N, C)"的包装器；
#   5. StaticHOCPTv3            : 整个手-物对应关系模型（物体 stem + 手 stem +
#                                统一主干 + 接触/对应关系/分类头）。
#
# 设计目标：
#   - 输入：当前帧的物体点云 + 静态手部点云 + 若干 KNN 索引 + GT 标签；
#   - 输出：物体接触概率、物体级 canonical 对应点、cross-edge 接触、
#           可选 finger/region。
# =============================================================================

# 启用 Python 3.7+ 的延迟类型注解求值。
from __future__ import annotations

# importlib 用于动态导入 PTv3 模型。
import importlib
# sys 用于向 Python 路径中临时加入 PTv3 仓库路径。
import sys
# 面向对象的路径处理。
from pathlib import Path
# 通用类型提示。
from typing import Any

# PyTorch 主入口。
import torch
# PyTorch 神经网络模块。
import torch.nn as nn

# 与"对应关系"相关的几何特征计算 / 邻居特征聚合工具。
from src.utils.correspondence import (
    compute_obj_to_hand_edge_features,
    decode_contact_bin_logits,
    gather_knn_features,
)


def _build_stem(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """构造一个 "Linear -> LN -> GELU -> Linear" 的 stem 模块。

    用于把不同来源的原始输入特征投影到统一的 hidden_dim 通道。
    LayerNorm 放在第一个 Linear 之后，能稳定不同输入特征之间的尺度。
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
    )


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """构造一个 "Linear -> GELU -> Linear" 的双层 MLP。

    通用投影器，用于把某种几何/语义特征映射到目标维度
    （例如把 3D 坐标映射到 64 维 embedding）。
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """支持布尔 mask 的 softmax。

    - mask=True 的位置会参与 softmax 计算；
    - mask=False 的位置先被填成 -inf（被 exp 后变为 0）；
    - 在数值上做了 max 减法的稳定化；
    - 对"全 False 的行"做了特殊处理，输出全 0，避免 NaN。

    Args:
        logits: 任意形状的 logits 张量。
        mask:  与 logits 形状相同的 bool 张量。
        dim:   沿哪个维度做 softmax。

    Returns:
        与 logits 形状相同的概率张量。
    """
    # 把 mask 转为 float，方便在最后一步做权重清零。
    mask_f = mask.float()
    # mask=False 的位置填 -inf，使其 exp 后为 0。
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    # 取每行最大值（数值稳定性）。
    max_logits = masked_logits.amax(dim=dim, keepdim=True)
    # 如果整行都是 -inf，max 会得到 -inf，需要手动归零。
    max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
    # exp(x - max) * mask（防止 -inf 之外的极小数值干扰）。
    exp_logits = torch.exp(masked_logits - max_logits) * mask_f
    # 求和得到分母。
    denom = exp_logits.sum(dim=dim, keepdim=True)
    # 分母为 0（整行被屏蔽）时直接返回全 0。
    return torch.where(denom > 0, exp_logits / denom.clamp(min=1e-12), torch.zeros_like(exp_logits))


def _load_ptv3_model_class(repo_path: str | Path):
    """动态加载 PointTransformerV3 官方实现中的 PointTransformerV3 类。

    流程：
      1. 把 repo_path 解析为绝对路径；
      2. 检查路径是否存在；
      3. 把 repo_path 的父目录加入 sys.path（因为 PTv3 的导入名是
         "PointTransformerV3"，它需要父目录在 sys.path 中）；
      4. importlib.import_module("PointTransformerV3.model") 加载子模块；
      5. 返回 module.PointTransformerV3 类对象。
    """
    # 展开 ~ 并解析为绝对路径。
    repo_path = Path(repo_path).expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"PointTransformerV3 repo path not found: {repo_path}")

    # PTv3 的包名是 PointTransformerV3，要求其父目录在 sys.path 中。
    parent = str(repo_path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    # 动态加载 PTv3 的 model 子模块。
    module = importlib.import_module("PointTransformerV3.model")
    return module.PointTransformerV3


class PTv3DenseBackbone(nn.Module):
    """把官方 PointTransformerV3 包装为"密集张量输入"版本的 Backbone。

    PTv3 原始接口的输入是 dict 格式（"feat" / "coord" / "batch"），
    点的数量是动态变化的（基于 batch 维度 cat 起来的 flatten 表示）。
    为了与本项目"统一 (B, N, C) 张量"的风格对齐，本类做了如下包装：
      - 输入仍是 (B, N, C) 密集张量 + 有效性 mask；
      - 在内部把"有效点"按 batch 收集成 PTv3 期望的 dict；
      - 调用 PTv3 后，再把输出 scatter 回 (B, N, output_dim) 的密集张量。
    """

    def __init__(self, meta: Any, in_channels: int) -> None:
        """初始化。

        Args:
            meta: 配置对象（包含所有 ptv3_* 字段）。
            in_channels: 输入特征维度（等于 stem 输出维度 hidden_dim）。
        """
        super().__init__()
        # 动态加载 PTv3 模型类。
        ptv3_cls = _load_ptv3_model_class(getattr(meta, "ptv3_repo_path"))
        # 体素栅格粒度（PTv3 需要）。
        self.grid_size = float(meta.ptv3_grid_size)
        # Backbone 输出维度 = 解码器第一阶段的通道数。
        self.output_dim = int(tuple(meta.ptv3_dec_channels)[0])
        self.shuffle_orders = bool(meta.ptv3_shuffle_orders)
        # 构造 PTv3 模型。本任务不使用分类头、PDNorm 等高级开关。
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
            qk_scale=None,  # 让 PTv3 内部用 1/sqrt(d) 作为默认 scale。
            attn_drop=float(meta.ptv3_attn_drop),
            proj_drop=float(meta.ptv3_proj_drop),
            drop_path=float(meta.ptv3_drop_path),
            pre_norm=bool(meta.ptv3_pre_norm),
            shuffle_orders=bool(meta.ptv3_shuffle_orders),
            enable_rpe=bool(meta.ptv3_enable_rpe),
            enable_flash=bool(meta.ptv3_enable_flash),
            upcast_attention=bool(meta.ptv3_upcast_attention),
            upcast_softmax=bool(meta.ptv3_upcast_softmax),
            cls_mode=False,    # 不在最后做分类聚合。
            pdnorm_bn=False,   # 不使用 PDNorm-BN。
            pdnorm_ln=False,   # 不使用 PDNorm-LN。
        )

    def forward(self, feat: torch.Tensor, coord: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """前向推理。

        Args:
            feat: (B, N, C_in) 输入特征。
            coord: (B, N, 3) 输入坐标（点云 XYZ）。
            valid_mask: (B, N) bool 掩码，标记哪些点为有效点。

        Returns:
            (B, N, output_dim) 密集特征张量；无效点对应位置为 0。
        """
        # 官方 PTv3 在 eval() 下仍会随机 shuffle serialization order，且
        # SerializedPooling 的默认值也是 True。训练时保留这种增强，评估时
        # 统一关闭，保证同一 checkpoint / sample 的验证结果可复现。
        use_shuffle = self.shuffle_orders and self.training
        for module in self.backbone.modules():
            if hasattr(module, "shuffle_orders"):
                module.shuffle_orders = use_shuffle

        # 取出 batch 维度和点数。
        batch_size, num_points, _ = feat.shape
        # 准备输出张量（无效点位置保持为 0）。
        dense_out = feat.new_zeros(batch_size, num_points, self.output_dim)

        # 把每个 batch 内的有效点收集到 list 末尾，统一 cat 给 PTv3。
        flat_feat: list[torch.Tensor] = []
        flat_coord: list[torch.Tensor] = []
        flat_batch: list[torch.Tensor] = []
        # 用于把 PTv3 的输出 scatter 回 (B, N, C) 的稠密格式。
        flat_dense_idx: list[torch.Tensor] = []

        # 逐 batch 取出有效点。
        for batch_idx in range(batch_size):
            valid_idx = torch.nonzero(valid_mask[batch_idx], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                # 整个 batch 都没有有效点：跳过。
                continue
            flat_feat.append(feat[batch_idx, valid_idx])
            flat_coord.append(coord[batch_idx, valid_idx])
            # batch 索引：用于 PTv3 区分"哪些点属于哪个样本"。
            flat_batch.append(
                torch.full((valid_idx.numel(),), batch_idx, device=feat.device, dtype=torch.long)
            )
            # 把"batch 内索引"换算为"全局稠密索引"，方便 scatter。
            flat_dense_idx.append(valid_idx + batch_idx * num_points)

        # 所有样本都无效：直接返回全 0。
        if not flat_feat:
            return dense_out

        # 构造 PTv3 期望的输入 dict。
        data_dict = {
            "feat": torch.cat(flat_feat, dim=0).contiguous(),
            "coord": torch.cat(flat_coord, dim=0).contiguous(),
            "batch": torch.cat(flat_batch, dim=0).contiguous(),
            "grid_size": self.grid_size,
        }
        # 调用 PTv3 主干，得到"逐点特征"（已上采样回原始分辨率）。
        point = self.backbone(data_dict)
        flat_out = point.feat

        # scatter 回稠密张量（view 成 (B*N, C) 方便用一维索引）。
        dense_out_flat = dense_out.view(batch_size * num_points, self.output_dim)
        # PTv3 的部分算子会在 autocast 区域内显式返回 float32，而 stem
        # token / dense_out 是 float16。索引赋值不会自动做 dtype promotion，
        # 因此在 scatter 前对齐到目标 dtype。
        dense_out_flat[torch.cat(flat_dense_idx, dim=0)] = flat_out.to(
            dtype=dense_out_flat.dtype
        )
        return dense_out


class StaticHOCPTv3(nn.Module):
    """基于 PointTransformerV3 的静态 hand-object token 模型。

    当前默认路径：
      1. 直接使用 runtime noisy geometry 构造共享点特征；
      2. 不使用 hand/object 双 stem；
      3. cross-attn 默认关闭，但保留 config 开关；
      4. 物体点和 cross-edge 都输出 10-bin contact logits；
      5. 同时导出解码后的连续概率，供指标和可视化使用。
    """

    def __init__(
        self,
        cfg: Any,
        *,
        condition_shape: list[int] | tuple[int, ...] | None = None,
        target_shape: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """初始化。

        Args:
            cfg: 任务配置对象（应包含 meta、model 等字段）。
            condition_shape / target_shape: 框架约定接口（这里未使用，预留）。
        """
        super().__init__()
        self.cfg = cfg
        meta = cfg.meta

        self.num_obj_points = int(meta.num_obj_points)
        self.num_hand_points = int(meta.num_hand_points)
        self.num_fingers = int(meta.num_fingers)
        self.num_regions = int(meta.num_regions)
        self.point_feat_dim = int(getattr(meta, "point_feat_dim", 11))
        self.num_contact_bins = int(getattr(meta, "num_contact_bins", 10))
        self.contact_bin_decode_mode = str(
            getattr(meta, "contact_bin_decode_mode", "expectation")
        )
        self.use_cross_attn = bool(getattr(meta, "use_cross_attn", False))
        self.use_finger_region_head = bool(getattr(meta, "use_finger_region_head", False))
        self.use_cano_head = bool(getattr(meta, "use_cano_head", False))

        self.backbone = PTv3DenseBackbone(meta, in_channels=self.point_feat_dim)
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
            nn.Linear(self.token_dim // 2, self.num_contact_bins),
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
        self.cross_edge_head = nn.Linear(self.edge_shared_dim, self.num_contact_bins)

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

        pred_obj_contact_bin = self.contact_head(z_obj_cross)
        outputs = {
            "pred_obj_contact_bin": pred_obj_contact_bin,
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
        pred_cross_contact_bin = self._compute_cross_edge_predictions(
            edge_shared=edge_shared,
        )
        outputs["pred_cross_contact_bin"] = pred_cross_contact_bin

        pred_obj_contact_prob = decode_contact_bin_logits(
            pred_obj_contact_bin,
            mode=self.contact_bin_decode_mode,
        )
        pred_cross_contact_prob = decode_contact_bin_logits(
            pred_cross_contact_bin,
            mode=self.contact_bin_decode_mode,
        )
        outputs["pred_obj_contact_prob"] = pred_obj_contact_prob
        outputs["pred_cross_contact_prob"] = pred_cross_contact_prob
        outputs["pred_obj_contact"] = self._safe_logit_from_prob(pred_obj_contact_prob)
        outputs["pred_cross_contact"] = self._safe_logit_from_prob(pred_cross_contact_prob)

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
        attn = _masked_softmax(logits, obj_to_hand_knn_valid_mask)
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

    @staticmethod
    def _safe_logit_from_prob(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
        prob = prob.clamp(min=eps, max=1.0 - eps)
        return torch.log(prob) - torch.log1p(-prob)
