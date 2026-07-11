from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.base import make_file_split_dataloaders
from src.base.data import make_dataloader_kwargs
from src.base.distributed import make_default_eval_sampler, shard_sampler_for_distributed
from src.task.correspondence_ptv3.config import (
    resolve_logit_far_min_radius,
    resolve_logit_near_radius,
)
from src.task.correspondence_ptv3.sampling import (
    augment_geometry,
    sample_object_indices,
    stable_frame_seed,
)
from src.utils.correspondence import soft_contact_label


def _validate_stratified_logit_config(
    distance_edges: tuple[float, ...] | list[float],
    quotas: tuple[int, ...] | list[int],
) -> tuple[np.ndarray, np.ndarray]:
    edges = tuple(float(value) for value in distance_edges)
    quota_values = tuple(int(value) for value in quotas)
    if len(edges) != 4:
        raise ValueError("logit_stratified_distance_edges must contain exactly 4 values.")
    if len(quota_values) != len(edges) + 1:
        raise ValueError("logit_stratified_quotas must have len(distance_edges) + 1 entries.")
    if sum(quota_values) != 128:
        raise ValueError("logit_stratified_quotas must sum to 128.")
    if any(quota < 0 for quota in quota_values):
        raise ValueError("logit_stratified_quotas must be non-negative.")
    if any(edge <= 0.0 for edge in edges) or any(
        curr <= prev for prev, curr in zip(edges[:-1], edges[1:])
    ):
        raise ValueError("logit_stratified_distance_edges must be positive and strictly increasing.")
    return np.asarray(edges, dtype=np.float32), np.asarray(quota_values, dtype=np.int64)


def _stratified_distance_bucket_candidates(
    dist_row: np.ndarray,
    distance_edges: np.ndarray,
) -> list[np.ndarray]:
    edge0, edge1, edge2, edge3 = (float(value) for value in distance_edges)
    return [
        np.flatnonzero(dist_row <= edge0),
        np.flatnonzero((dist_row > edge0) & (dist_row <= edge1)),
        np.flatnonzero((dist_row > edge1) & (dist_row < edge2)),
        np.flatnonzero((dist_row >= edge2) & (dist_row <= edge3)),
        np.flatnonzero(dist_row > edge3),
    ]


class CorrStaticDataset(Dataset):
    """Stage 3 点池数据集，支持每个 epoch 确定性地从候选池中采样 object 点。

    数据流概览：
        1. 从 Stage 3 落盘的 .npz 文件加载 clean object / hand 几何及 GT 邻接表
        2. 每个 epoch 从 5cm 候选池中确定性随机采样 num_obj_points 个 object 点
        3. 对采样的 object 点应用几何增强（全局旋转/平移/缩放）以及手部扰动
        4. 在 noisy geometry 上计算 runtime context 邻域，用于 cross-attn
        5. 在 clean GT geometry 上计算 runtime logit 邻域（近点 + 远点）用于 cross-edge 监督
        6. 返回训练所需的 input/gt 点云、邻接关系、contact soft label 等字段
    """

    # Stage 3 .npz 文件必须包含的字段（缺一不可）
    REQUIRED_FIELDS = {
        "raw_frame_id",
        "obj_points",
        "obj_normals",
        "obj_point_id",
        "hand_points",
        "hand_normals",
        "hand_point_id",
        "obj_to_hand_min_dist",
        "obj_candidate_mask_5cm",
        "gt_obj_to_hand_knn_idx",
    }

    def __init__(
        self,
        data_path: str | Path,
        *,
        file_list: list[str | Path] | None = None,
        num_obj_points: int = 512,
        num_hand_points: int = 1538,
        k_cross: int = 32,
        k_ctx: int = 32,
        k_near_logit: int = 32,
        k_far_logit: int = 32,
        logit_sampling_mode: str = "balanced",
        logit_stratified_distance_edges: tuple[float, ...] = (0.005, 0.015, 0.03, 0.06),
        logit_stratified_quotas: tuple[int, ...] = (16, 32, 32, 32, 16),
        ctx_radius: float = 0.04,
        logit_near_radius: float | None = None,
        logit_far_min_radius: float | None = None,
        logit_pos_radius: float | None = None,
        logit_neg_min_radius: float | None = None,
        logit_neg_radius: float = 0.06,
        logit_far_weight: float = 0.5,
        base_seed: int = 42,
        augment: bool = True,
        apply_hand_perturb: bool = True,
        augment_rotation: bool = True,
        augment_translation: bool = False,
        augment_scale: bool = False,
        rotation_range: float = 180.0,
        translation_range: float = 0.1,
        scale_range: tuple[float, float] = (0.9, 1.1),
        d_pos: float = 0.005,
        d_neg: float = 0.03,
        gamma: float = 2.0,
        hand_rot_std_deg: float = 10.0,
        hand_trans_std: float = 0.01,
        hand_perturb_prob: float = 1.0,
        fix_overfit_seed: bool = False,
        blacklist_path: str | None = None,
        **_: Any,
    ) -> None:
        """初始化数据集，构建文件索引和样本列表。

        Args:
            data_path: 数据集根目录或单个 .npz 文件路径
            file_list: 显式指定的文件列表，None 时自动扫描目录下所有 .npz
            num_obj_points: 每个样本采样的 object 点数（默认 512）
            num_hand_points: 手部点云大小（默认 1538）
            k_cross: GT KNN 邻居数
            k_ctx: runtime context 邻域最大点数
            k_near_logit: logit 邻域中近点最大采样数
            k_far_logit: logit 邻域中远点最大采样数
            ctx_radius: context 邻域半径（米）
            logit_near_radius: 近点判定半径（米），None 时退化到 logit_pos_radius
            logit_far_min_radius: 远点判定半径（米），None 时退化到 logit_neg_min_radius
            logit_pos_radius: 旧版近点半径别名
            logit_neg_min_radius: 旧版远点半径别名
            logit_neg_radius: logit 负样本外圈半径
            logit_far_weight: 远点 loss 权重
            base_seed: 随机种子基址
            augment: 是否应用全局几何增强
            apply_hand_perturb: 是否对手部施加高斯扰动
            augment_rotation/translation/scale: 各类增强开关
            rotation_range/translation_range/scale_range: 各类增强幅度
            d_pos/d_neg/gamma: soft contact label 的参数（d_pos 内=1，d_neg 外=0，中间软过渡）
            hand_rot_std_deg: 手部旋转扰动标准差（度）
            hand_trans_std: 手部位移扰动标准差（米）
            hand_perturb_prob: 手部扰动应用概率
            blacklist_path: 可选的黑名单文件路径，用于排除某些序列
        """
        super().__init__()
        self.data_path = Path(data_path)
        # 如果 data_path 是单文件，把它的父目录作为 root；否则直接使用
        self.data_root = self.data_path if self.data_path.is_dir() else self.data_path.parent
        self.num_obj_points = int(num_obj_points)
        self.num_hand_points = int(num_hand_points)
        self.k_gt = int(k_cross)
        self.k_ctx = int(k_ctx)
        self.k_near_logit = int(k_near_logit)
        self.k_far_logit = int(k_far_logit)
        self.logit_sampling_mode = str(logit_sampling_mode).lower()
        (
            self.logit_stratified_distance_edges,
            self.logit_stratified_quotas,
        ) = _validate_stratified_logit_config(
            logit_stratified_distance_edges,
            logit_stratified_quotas,
        )
        self.ctx_radius = float(ctx_radius)
        # 兼容新旧命名：新参数（logit_near_radius / logit_far_min_radius）优先，
        # 未提供时退化到旧参数（logit_pos_radius / logit_neg_min_radius），
        # 再未提供时调用 resolve_*_radius 从 config 解析。
        resolved_logit_near_radius = (
            logit_near_radius if logit_near_radius is not None else logit_pos_radius
        )
        resolved_logit_far_min_radius = (
            logit_far_min_radius
            if logit_far_min_radius is not None
            else logit_neg_min_radius
        )
        if resolved_logit_near_radius is None:
            resolved_logit_near_radius = resolve_logit_near_radius(None)
        if resolved_logit_far_min_radius is None:
            resolved_logit_far_min_radius = resolve_logit_far_min_radius(None)
        self.logit_near_radius = float(resolved_logit_near_radius)
        self.logit_far_min_radius = float(resolved_logit_far_min_radius)
        # 同步保存旧别名，便于下游访问
        self.logit_pos_radius = self.logit_near_radius
        self.logit_neg_min_radius = self.logit_far_min_radius
        self.logit_neg_radius = float(logit_neg_radius)
        self.logit_far_weight = float(logit_far_weight)
        if self.logit_sampling_mode not in {"balanced", "dense", "stratified"}:
            raise ValueError(
                "logit_sampling_mode must be 'balanced', 'dense', or 'stratified', got "
                f"{self.logit_sampling_mode!r}."
            )
        if self.k_ctx <= 0:
            raise ValueError("k_ctx must be positive.")
        if self.logit_sampling_mode == "balanced":
            if self.k_near_logit <= 0 or self.k_far_logit <= 0:
                raise ValueError("k_near_logit and k_far_logit must be positive.")
        elif self.logit_sampling_mode == "stratified":
            if int(self.logit_stratified_quotas.sum()) <= 0:
                raise ValueError("logit_stratified_quotas must sum to a positive K.")
        elif self.k_near_logit < 0 or self.k_far_logit < 0:
            raise ValueError("k_near_logit and k_far_logit must be non-negative.")
        if self.logit_near_radius <= 0.0:
            raise ValueError("logit_near_radius must be positive.")
        if self.logit_far_min_radius < self.logit_near_radius:
            raise ValueError("logit_far_min_radius must be >= logit_near_radius.")
        self.base_seed = int(base_seed)
        self.augment = bool(augment)
        self.apply_hand_perturb = bool(apply_hand_perturb)
        self.augment_rotation = bool(augment_rotation)
        self.augment_translation = bool(augment_translation)
        self.augment_scale = bool(augment_scale)
        self.rotation_range = float(rotation_range)
        self.translation_range = float(translation_range)
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.d_pos = float(d_pos)
        self.d_neg = float(d_neg)
        self.gamma = float(gamma)
        self.hand_rot_std_deg = float(hand_rot_std_deg)
        self.hand_trans_std = float(hand_trans_std)
        self.hand_perturb_prob = float(hand_perturb_prob)
        # 纯过拟合测试：把 stable_frame_seed 中的 epoch 项强制置 0，
        # 让 object 采样 / 增强 / logit 邻居在每个 epoch 保持完全一致。
        self.fix_overfit_seed = bool(fix_overfit_seed)
        # 跨进程共享的 epoch 计数；DataLoader worker 进程会读这个值决定采样随机性
        self._epoch = mp.Value("q", 0, lock=True)
        # 简单的文件级缓存：避免同一 worker 内反复读取同一个 .npz
        self._cached_path: Path | None = None
        self._cached_data: dict[str, np.ndarray] | None = None

        # 解析输入文件列表：优先 file_list，其次扫目录，最后当作单文件
        paths = (
            sorted(Path(path) for path in file_list)
            if file_list is not None
            else (
                sorted(self.data_path.glob("**/*.npz"))
                if self.data_path.is_dir()
                else [self.data_path]
            )
        )
        # 应用黑名单过滤
        blacklist = _load_blacklist(blacklist_path)
        self.file_paths = [path for path in paths if not _is_blacklisted(path, self.data_root, blacklist)]
        if not self.file_paths:
            raise ValueError(f"No Stage 3 npz files found in {self.data_path}")

        # 构建 (file, frame_idx) 的样本级索引，并记录每个文件对应的样本区间
        # （后者用于 SequenceLocalitySampler 按文件块打乱）
        self._samples: list[tuple[Path, int]] = []
        self.file_sample_ranges: list[tuple[int, int]] = []
        for path in self.file_paths:
            with np.load(path, allow_pickle=False) as data:
                missing = self.REQUIRED_FIELDS.difference(data.files)
                if missing:
                    raise KeyError(f"{path}: missing Stage 3 fields {sorted(missing)}")
                num_frames = int(data["raw_frame_id"].shape[0])
            start = len(self._samples)
            self._samples.extend((path, frame_idx) for frame_idx in range(num_frames))
            self.file_sample_ranges.append((start, len(self._samples)))

    @property
    def epoch(self) -> int:
        """当前 epoch（线程安全读）。"""
        return int(self._epoch.value)

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，使得每个 epoch 内的 object 采样结果可复现且彼此不同。"""
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def __len__(self) -> int:
        return len(self._samples)

    def _load_file(self, path: Path) -> dict[str, np.ndarray]:
        """带缓存的 .npz 加载：同一 worker 进程内对同一 path 只读一次磁盘。"""
        if self._cached_path == path and self._cached_data is not None:
            return self._cached_data
        with np.load(path, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        self._cached_path = path
        self._cached_data = payload
        return payload

    @staticmethod
    def _scalar_string(data: dict[str, np.ndarray], key: str, default: str = "") -> str:
        """从 npz 中读取 0-d 字符串标量；不存在或形状不对时返回 default。"""
        value = data.get(key)
        if value is None:
            return default
        array = np.asarray(value)
        return str(array.item()) if array.size == 1 else default

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """取一个样本。

        Returns:
            dict, 字段含义：
            - points / normals: 拼接后的 noisy 物体+手部点云
            - gt_points / gt_normals: 拼接后的 clean GT 点云
            - point_valid_mask: 拼接后各点是否有效
            - runtime_obj_valid_mask: 当前帧采样的 object 点是否 valid
            - selected_obj_idx / selected_obj_point_id: 采样的 object 点在原 pool 中的索引和 ID
            - selected_obj_min_dist: 每个采样 object 点到最近 hand 点的距离
            - obj_contact_label: soft contact label（基于 clean GT）
            - gt_obj_to_hand_knn_idx: clean GT 的 KNN 邻居表
            - input_obj_to_hand_ctx_idx / _valid_mask: noisy 几何上的 context 邻域
            - input_obj_to_hand_logit_idx / _valid_mask / _loss_weight / _near_count / _far_count:
              clean GT 几何上的 logit 邻域（用于 cross-edge 监督）
            - hand_cano_points / hand_finger_id / hand_region_id: 手部元信息
            - num_obj_points / num_hand_points: 张量形式的几何尺寸
        """
        path, frame_idx = self._samples[index]
        data = self._load_file(path)
        seq_id = self._scalar_string(data, "seq_id", path.stem)
        side = self._scalar_string(data, "side", "")
        raw_frame_id = int(np.asarray(data["raw_frame_id"])[frame_idx])
        epoch = self.epoch
        # 纯过拟合测试：固定 object / augment / edge 邻居的 seed，
        # 让每个 epoch 喂给模型的输入完全一致。
        seed_epoch = 0 if self.fix_overfit_seed else epoch
        # 由 (seq, side, frame, epoch) 派生稳定种子：保证每个 epoch 重新采样，
        # 但不同 worker 拿到同一 index 时结果一致。
        sample_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=seed_epoch,
        )
        # 从 5cm 候选池中采样 num_obj_points 个 object 点
        selected_idx, obj_valid = sample_object_indices(
            data["obj_candidate_mask_5cm"][frame_idx],
            num_samples=self.num_obj_points,
            seed=sample_seed,
        )
        # safe_idx: 把 padding（-1）夹到 0，避免 fancy indexing 报错；
        # 后续会把这些位置用 0 / -1 / False 标记为 invalid
        safe_idx = np.maximum(selected_idx, 0)

        obj_points = np.asarray(data["obj_points"][frame_idx, safe_idx], dtype=np.float32).copy()
        obj_normals = np.asarray(data["obj_normals"][frame_idx, safe_idx], dtype=np.float32).copy()
        obj_point_id = np.asarray(data["obj_point_id"][safe_idx], dtype=np.int64).copy()
        obj_min_dist = np.asarray(
            data["obj_to_hand_min_dist"][frame_idx, safe_idx],
            dtype=np.float32,
        ).copy()
        clean_knn_idx = np.asarray(
            data["gt_obj_to_hand_knn_idx"][frame_idx, safe_idx],
            dtype=np.int64,
        ).copy()

        # 无效点用 0 / -1 占位，避免污染下游数值计算
        obj_points[~obj_valid] = 0
        obj_normals[~obj_valid] = 0
        obj_point_id[~obj_valid] = -1
        obj_min_dist[~obj_valid] = 0
        clean_knn_idx[~obj_valid] = -1

        hand_points = np.asarray(data["hand_points"][frame_idx], dtype=np.float32)
        hand_normals = np.asarray(data["hand_normals"][frame_idx], dtype=np.float32)
        # 几何增强使用独立 seed，与采样 seed 解耦，方便单独复现
        aug_seed = stable_frame_seed(
            base_seed=self.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=seed_epoch,
            namespace="augmentation",
        )
        # 几何增强：返回 input（可能含扰动）+ gt（clean）两套点云
        geometry = augment_geometry(
            obj_points=obj_points,
            obj_normals=obj_normals,
            hand_points=hand_points,
            hand_normals=hand_normals,
            seed=aug_seed,
            apply_hand_perturb=self.apply_hand_perturb,
            hand_rot_std_deg=self.hand_rot_std_deg,
            hand_trans_std=self.hand_trans_std,
            hand_perturb_prob=self.hand_perturb_prob,
            apply_global_aug=self.augment,
            augment_rotation=self.augment_rotation,
            rotation_range_deg=self.rotation_range,
            augment_translation=self.augment_translation,
            translation_range=self.translation_range,
            augment_scale=self.augment_scale,
            scale_range=self.scale_range,
        )
        # 如果全局增强做了 isotropic 缩放，把距离按缩放系数同步更新
        obj_min_dist *= float(geometry.distance_scale)

        # runtime context 邻域（noisy 几何上做 KNN，按 4cm 半径截断）
        input_ctx_idx, input_ctx_valid = _compute_runtime_context_neighbors(
            geometry.input_obj_points,
            geometry.input_hand_points,
            obj_valid,
            k_ctx=self.k_ctx,
            ctx_radius=self.ctx_radius,
        )
        # runtime logit 邻域（clean GT 几何上定义 cross-edge 监督集合）
        if self.logit_sampling_mode == "dense":
            (
                input_logit_idx,
                input_logit_valid,
                input_logit_weight,
                input_logit_near_count,
                input_logit_far_count,
            ) = _compute_dense_logit_neighbors(
                geometry.gt_obj_points,
                geometry.gt_hand_points,
                obj_valid,
                logit_near_radius=self.logit_near_radius,
                logit_far_min_radius=self.logit_far_min_radius,
            )
        elif self.logit_sampling_mode == "stratified":
            (
                input_logit_idx,
                input_logit_valid,
                input_logit_weight,
                input_logit_near_count,
                input_logit_far_count,
            ) = _compute_stratified_logit_neighbors(
                geometry.gt_obj_points,
                geometry.gt_hand_points,
                obj_valid,
                distance_edges=self.logit_stratified_distance_edges,
                quotas=self.logit_stratified_quotas,
                logit_near_radius=self.logit_near_radius,
                logit_far_min_radius=self.logit_far_min_radius,
                seed=stable_frame_seed(
                    base_seed=self.base_seed,
                    seq_id=seq_id,
                    side=side,
                    raw_frame_id=raw_frame_id,
                    epoch=seed_epoch,
                    namespace="logit-neighbors",
                ),
            )
        else:
            (
                input_logit_idx,
                input_logit_valid,
                input_logit_weight,
                input_logit_near_count,
                input_logit_far_count,
            ) = _compute_runtime_logit_neighbors(
                geometry.gt_obj_points,
                geometry.gt_hand_points,
                obj_valid,
                k_near_logit=self.k_near_logit,
                k_far_logit=self.k_far_logit,
                logit_near_radius=self.logit_near_radius,
                logit_far_min_radius=self.logit_far_min_radius,
                seed=stable_frame_seed(
                    base_seed=self.base_seed,
                    seq_id=seq_id,
                    side=side,
                    raw_frame_id=raw_frame_id,
                    epoch=seed_epoch,
                    namespace="logit-neighbors",
                ),
            )
        # 把 object 和 hand 拼成单个点云（前半 object，后半 hand）
        input_points = np.concatenate(
            [geometry.input_obj_points, geometry.input_hand_points],
            axis=0,
        )
        input_normals = np.concatenate(
            [geometry.input_obj_normals, geometry.input_hand_normals],
            axis=0,
        )
        gt_points = np.concatenate(
            [geometry.gt_obj_points, geometry.gt_hand_points],
            axis=0,
        )
        gt_normals = np.concatenate(
            [geometry.gt_obj_normals, geometry.gt_hand_normals],
            axis=0,
        )
        point_valid_mask = np.concatenate(
            [obj_valid, np.ones((self.num_hand_points,), dtype=bool)],
            axis=0,
        )
        # soft contact label：d_pos 内=1，d_neg 外=0，中段用 gamma 控制过渡斜率
        contact_label = soft_contact_label(
            torch.from_numpy(obj_min_dist),
            d_pos=self.d_pos,
            d_neg=self.d_neg,
            gamma=self.gamma,
        )

        return {
            "points": torch.from_numpy(input_points).float(),
            "normals": torch.from_numpy(input_normals).float(),
            "gt_points": torch.from_numpy(gt_points).float(),
            "gt_normals": torch.from_numpy(gt_normals).float(),
            "point_valid_mask": torch.from_numpy(point_valid_mask),
            "runtime_obj_valid_mask": torch.from_numpy(obj_valid),
            "selected_obj_idx": torch.from_numpy(selected_idx).long(),
            "selected_obj_point_id": torch.from_numpy(obj_point_id).long(),
            "selected_obj_min_dist": torch.from_numpy(obj_min_dist).float(),
            "obj_contact_label": contact_label.float(),
            "gt_obj_to_hand_knn_idx": torch.from_numpy(clean_knn_idx).long(),
            "input_obj_to_hand_ctx_idx": torch.from_numpy(input_ctx_idx).long(),
            "input_obj_to_hand_ctx_valid_mask": torch.from_numpy(input_ctx_valid),
            "input_obj_to_hand_logit_idx": torch.from_numpy(input_logit_idx).long(),
            "input_obj_to_hand_logit_valid_mask": torch.from_numpy(input_logit_valid),
            "input_obj_to_hand_logit_loss_weight": torch.from_numpy(input_logit_weight).float(),
            "input_obj_to_hand_logit_near_count": torch.from_numpy(input_logit_near_count).long(),
            "input_obj_to_hand_logit_far_count": torch.from_numpy(input_logit_far_count).long(),
            "hand_cano_points": torch.from_numpy(
                np.asarray(
                    data.get(
                        "hand_cano_points",
                        np.zeros((self.num_hand_points, 3), dtype=np.float32),
                    )
                )
            ).float(),
            "hand_finger_id": torch.from_numpy(
                np.asarray(
                    data.get(
                        "hand_finger_id",
                        np.full((self.num_hand_points,), -1, dtype=np.int64),
                    )
                )
            ).long(),
            "hand_region_id": torch.from_numpy(
                np.asarray(
                    data.get(
                        "hand_region_id",
                        np.full((self.num_hand_points,), -1, dtype=np.int64),
                    )
                )
            ).long(),
            "num_obj_points": torch.tensor(self.num_obj_points, dtype=torch.long),
            "num_hand_points": torch.tensor(self.num_hand_points, dtype=torch.long),
        }


def _compute_runtime_context_neighbors(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_ctx: int,
    ctx_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """在 noisy 几何上为每个 valid object 点计算 context 邻域。

    规则：取 ctx_radius 半径内的所有 hand 点，按距离升序取前 k_ctx 个；
    不足 k_ctx 时剩余位置用 -1 / False 填充。

    Args:
        obj_points: (N, 3) noisy object 点
        hand_points: (M, 3) noisy hand 点
        obj_valid: (N,) bool，标记每个 object 点是否有效
        k_ctx: 最大邻域大小
        ctx_radius: 邻域半径

    Returns:
        ctx_idx: (N, k_ctx) int64，hand 点的索引；-1 表示 padding
        ctx_valid: (N, k_ctx) bool
    """
    num_obj = obj_points.shape[0]
    ctx_idx = np.full((num_obj, k_ctx), -1, dtype=np.int64)
    ctx_valid = np.zeros((num_obj, k_ctx), dtype=bool)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return ctx_idx, ctx_valid
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    # 一次 cdist 拿到所有 valid obj 点到所有 hand 点的距离
    distance = torch.cdist(obj, hand).numpy()
    for row, obj_idx in enumerate(valid_obj_idx.tolist()):
        dist_row = distance[row]
        ctx_candidates = np.flatnonzero(dist_row <= float(ctx_radius))
        if ctx_candidates.size > 0:
            # 按距离升序取前 k_ctx（np.argsort + stable 保证可复现）
            order = np.argsort(dist_row[ctx_candidates], kind="stable")
            chosen_ctx = ctx_candidates[order[:k_ctx]]
            ctx_count = int(chosen_ctx.size)
            ctx_idx[obj_idx, :ctx_count] = chosen_ctx
            ctx_valid[obj_idx, :ctx_count] = True
    return ctx_idx, ctx_valid


def _compute_runtime_logit_neighbors(
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_near_logit: int,
    k_far_logit: int,
    logit_near_radius: float,
    logit_far_min_radius: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """在 clean GT 几何上为每个 valid object 点采样 logit 监督邻域。

    每个物体点独立固定采样 k_near_logit 个近点 + k_far_logit 个远点，不够就 padding。
    - 近点：到该 obj 点距离 <= logit_near_radius 的 hand 点
    - 远点：到该 obj 点距离 > logit_far_min_radius 的 hand 点
    近点/远点都在各自候选池中随机抽取；候选不足时取全部，剩余位置 padding。

    Args:
        gt_obj_points: (N, 3) clean GT object 点
        gt_hand_points: (M, 3) clean GT hand 点
        obj_valid: (N,) bool
        k_near_logit: 近点最大采样数
        k_far_logit: 远点最大采样数
        logit_near_radius: 近点判定半径（米）
        logit_far_min_radius: 远点判定半径（米）
        seed: 随机种子

    Returns:
        logit_idx: (N, k_near_logit + k_far_logit) int64
        logit_valid: (N, k_near_logit + k_far_logit) bool
        logit_weight: (N, k_near_logit + k_far_logit) float32，当前默认 valid 全 1.0
        near_count: (N,) int64，实际近点采样数
        far_count: (N,) int64，实际远点采样数
    """
    k_logit = int(k_near_logit) + int(k_far_logit)
    num_obj = gt_obj_points.shape[0]
    logit_idx = np.full((num_obj, k_logit), -1, dtype=np.int64)
    logit_valid = np.zeros((num_obj, k_logit), dtype=bool)
    logit_weight = np.zeros((num_obj, k_logit), dtype=np.float32)
    near_count = np.zeros((num_obj,), dtype=np.int64)
    far_count = np.zeros((num_obj,), dtype=np.int64)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return logit_idx, logit_valid, logit_weight, near_count, far_count
    obj = torch.from_numpy(np.asarray(gt_obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(gt_hand_points, dtype=np.float32))
    distance = torch.cdist(obj, hand).numpy()
    rng = np.random.default_rng(seed)
    for row, obj_idx in enumerate(valid_obj_idx.tolist()):
        dist_row = distance[row]
        # 近点采样：在 logit_near_radius 半径内的 hand 点里随机抽不超过 k_near_logit 个
        near_candidates = np.flatnonzero(dist_row <= float(logit_near_radius))
        if near_candidates.size > 0:
            n_near = min(int(k_near_logit), int(near_candidates.size))
            chosen = rng.choice(near_candidates, size=int(n_near), replace=False)
            logit_idx[obj_idx, :n_near] = chosen
            logit_valid[obj_idx, :n_near] = True
            logit_weight[obj_idx, :n_near] = 1.0
            near_count[obj_idx] = int(n_near)
        # 远点采样：在 logit_far_min_radius 半径之外的 hand 点里随机抽不超过 k_far_logit 个
        far_candidates = np.flatnonzero(dist_row > float(logit_far_min_radius))
        if far_candidates.size > 0:
            n_far = min(int(k_far_logit), int(far_candidates.size))
            chosen = rng.choice(far_candidates, size=int(n_far), replace=False)
            start = int(k_near_logit)
            logit_idx[obj_idx, start:start + n_far] = chosen
            logit_valid[obj_idx, start:start + n_far] = True
            logit_weight[obj_idx, start:start + n_far] = 1.0
            far_count[obj_idx] = int(n_far)
    return logit_idx, logit_valid, logit_weight, near_count, far_count


def _compute_dense_logit_neighbors(
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    logit_near_radius: float,
    logit_far_min_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use every hand point as a logit supervision edge for each valid object point."""
    num_obj = gt_obj_points.shape[0]
    num_hand = gt_hand_points.shape[0]
    all_hand_idx = np.arange(num_hand, dtype=np.int64)
    logit_idx = np.broadcast_to(all_hand_idx[None, :], (num_obj, num_hand)).copy()
    logit_valid = np.broadcast_to(
        np.asarray(obj_valid, dtype=bool)[:, None],
        (num_obj, num_hand),
    ).copy()
    logit_weight = logit_valid.astype(np.float32)
    logit_idx[~logit_valid] = -1

    near_count = np.zeros((num_obj,), dtype=np.int64)
    far_count = np.zeros((num_obj,), dtype=np.int64)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return logit_idx, logit_valid, logit_weight, near_count, far_count

    obj = torch.from_numpy(np.asarray(gt_obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(gt_hand_points, dtype=np.float32))
    distance = torch.cdist(obj, hand).numpy()
    near_count[valid_obj_idx] = np.count_nonzero(
        distance <= float(logit_near_radius),
        axis=-1,
    ).astype(np.int64)
    far_count[valid_obj_idx] = np.count_nonzero(
        distance > float(logit_far_min_radius),
        axis=-1,
    ).astype(np.int64)
    return logit_idx, logit_valid, logit_weight, near_count, far_count


def _compute_stratified_logit_neighbors(
    gt_obj_points: np.ndarray,
    gt_hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    distance_edges: np.ndarray | tuple[float, ...] | list[float],
    quotas: np.ndarray | tuple[int, ...] | list[int],
    logit_near_radius: float,
    logit_far_min_radius: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample K=128 cross-edge targets across contact-relevant distance bands."""
    edges, quota_values = _validate_stratified_logit_config(
        tuple(float(value) for value in np.asarray(distance_edges).tolist()),
        tuple(int(value) for value in np.asarray(quotas).tolist()),
    )
    k_logit = int(quota_values.sum())
    num_obj = int(gt_obj_points.shape[0])
    logit_idx = np.full((num_obj, k_logit), -1, dtype=np.int64)
    logit_valid = np.zeros((num_obj, k_logit), dtype=bool)
    logit_weight = np.zeros((num_obj, k_logit), dtype=np.float32)
    near_count = np.zeros((num_obj,), dtype=np.int64)
    far_count = np.zeros((num_obj,), dtype=np.int64)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return logit_idx, logit_valid, logit_weight, near_count, far_count

    obj = torch.from_numpy(np.asarray(gt_obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(gt_hand_points, dtype=np.float32))
    distance = torch.cdist(obj, hand).numpy()
    rng = np.random.default_rng(seed)
    for row, obj_idx in enumerate(valid_obj_idx.tolist()):
        selected: list[int] = []
        remaining_parts: list[np.ndarray] = []
        dist_row = distance[row]
        for layer_idx, candidates in enumerate(
            _stratified_distance_bucket_candidates(dist_row, edges)
        ):
            shuffled = rng.permutation(candidates)
            take_count = min(int(quota_values[layer_idx]), int(shuffled.size))
            selected.extend(int(value) for value in shuffled[:take_count])
            remaining_parts.append(np.asarray(shuffled[take_count:], dtype=np.int64))

        offsets = [0] * len(remaining_parts)

        def refill_round_robin(layer_indices: tuple[int, ...]) -> None:
            while len(selected) < k_logit:
                progress = False
                for layer_idx in layer_indices:
                    offset = offsets[layer_idx]
                    remaining = remaining_parts[layer_idx]
                    if offset >= int(remaining.size):
                        continue
                    selected.append(int(remaining[offset]))
                    offsets[layer_idx] += 1
                    progress = True
                    if len(selected) >= k_logit:
                        break
                if not progress:
                    break

        refill_round_robin((1, 2, 3))
        if len(selected) < k_logit:
            refill_round_robin((0,))
        if len(selected) < k_logit:
            refill_round_robin((4,))

        count = min(k_logit, len(selected))
        if count <= 0:
            continue
        chosen = np.asarray(selected[:count], dtype=np.int64)
        logit_idx[obj_idx, :count] = chosen
        logit_valid[obj_idx, :count] = True
        logit_weight[obj_idx, :count] = 1.0
        selected_dist = dist_row[chosen]
        near_count[obj_idx] = int(np.count_nonzero(selected_dist <= logit_near_radius))
        far_count[obj_idx] = int(np.count_nonzero(selected_dist > logit_far_min_radius))
    return logit_idx, logit_valid, logit_weight, near_count, far_count


def _compute_input_knn(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid: np.ndarray,
    *,
    k_cross: int,
) -> tuple[np.ndarray, np.ndarray]:
    """保持向后兼容的辅助函数（供可视化脚本使用）。

    行为：取每个 valid object 点到所有 hand 点的 top-k（即经典 KNN，无半径限制）。

    Args:
        obj_points: (N, 3) object 点
        hand_points: (M, 3) hand 点
        obj_valid: (N,) bool
        k_cross: top-k 数量

    Returns:
        result: (N, k_cross) int64，hand 索引
        valid: (N, k_cross) bool
    """
    num_obj = obj_points.shape[0]
    result = np.full((num_obj, k_cross), -1, dtype=np.int64)
    valid = np.zeros((num_obj, k_cross), dtype=bool)
    valid_obj_idx = np.flatnonzero(obj_valid)
    if valid_obj_idx.size == 0:
        return result, valid
    obj = torch.from_numpy(np.asarray(obj_points[valid_obj_idx], dtype=np.float32))
    hand = torch.from_numpy(np.asarray(hand_points, dtype=np.float32))
    topk = min(int(k_cross), int(hand.shape[0]))
    idx = torch.topk(torch.cdist(obj, hand), k=topk, dim=-1, largest=False).indices.numpy()
    result[valid_obj_idx, :topk] = idx
    valid[valid_obj_idx, :topk] = True
    return result, valid


def _load_blacklist(path: str | None) -> set[str]:
    """加载黑名单文件。

    支持 .json（列表）或纯文本（每行一个，支持 # 注释）。
    未指定路径时返回空集。
    """
    if not path:
        return set()
    blacklist_path = Path(path)
    if not blacklist_path.exists():
        raise FileNotFoundError(f"Blacklist file not found: {blacklist_path}")
    if blacklist_path.suffix.lower() == ".json":
        import json

        payload = json.loads(blacklist_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Blacklist JSON must contain a list")
        return {str(item) for item in payload}
    return {
        line.strip()
        for line in blacklist_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _is_blacklisted(path: Path, root: Path, blacklist: set[str]) -> bool:
    """判断文件是否在黑名单中。

    黑名单键可以匹配：绝对路径、文件名、不带后缀的文件名、
    POSIX 路径、或相对 root 的 POSIX 路径。
    """
    if not blacklist:
        return False
    keys = {str(path), path.name, path.stem, path.as_posix()}
    try:
        keys.add(path.relative_to(root).as_posix())
    except ValueError:
        pass
    return bool(keys.intersection(blacklist))


def _sequence_group_key(path: Path) -> str:
    """为 train/val 文件划分生成序列组 key。

    规则：去掉路径末尾的 _left / _right 后缀，使同序列的 left/right 进入同一组，
    这样划分时不会把同序列的左右手分散到 train 和 val。
    """
    stem = path.stem
    if stem.endswith("_left") or stem.endswith("_right"):
        stem = stem.rsplit("_", 1)[0]
    parent = path.parent.as_posix()
    return stem if parent in {"", "."} else f"{parent}/{stem}"


class SequenceLocalitySampler(Sampler[int]):
    """按文件块打乱的采样器：先随机选文件，再在文件内随机打乱 frame 顺序。

    设计动机：.npz 内的整个序列是按帧存盘的；如果完全打乱 frame 顺序，
    几乎每个 sample 都要从磁盘重读整个 sequence。
    按文件块打乱可以保留 frame 维度的随机性，但每个 sequence 文件
    在每个 worker / epoch 中只被读一次，显著降低 I/O 开销。
    """

    def __init__(self, dataset: CorrStaticDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        # 用稳定 seed 派生文件/frame 顺序，保证不同 epoch 间可复现且彼此不同
        rng = np.random.default_rng(
            stable_frame_seed(
                base_seed=self.seed,
                seq_id="sequence-locality-sampler",
                side="",
                raw_frame_id=0,
                epoch=self.epoch,
                namespace="frame-order",
            )
        )
        # 1. 随机排列文件顺序
        file_order = rng.permutation(len(self.dataset.file_sample_ranges))
        for file_idx in file_order:
            # 2. 对该文件内的 frame 区间再随机打乱
            start, end = self.dataset.file_sample_ranges[int(file_idx)]
            yield from rng.permutation(np.arange(start, end, dtype=np.int64)).tolist()

    def __len__(self) -> int:
        return len(self.dataset)


def make_dataloaders(
    data_cfg: Any,
    seed: int,
    *,
    meta_cfg: Any,
    distributed: Any | None = None,
) -> tuple[DataLoader, DataLoader | None, dict[str, Any], dict[str, DataLoader]]:
    """构建 train / val DataLoader。

    Returns:
        train_loader: 训练 DataLoader
        val_loader: 简单 val loader（val_clean），可与 train 用同一 file-split
        metadata: 数据集元信息（点数、维度等）
        val_loaders: 命名 val loader 字典（val_clean/、val_perturbed/）
    """
    train_kwargs = {
        "num_obj_points": int(meta_cfg.num_obj_points),
        "num_hand_points": int(meta_cfg.num_hand_points),
        "k_cross": int(meta_cfg.k_cross),
        "k_ctx": int(getattr(meta_cfg, "k_ctx", meta_cfg.k_cross)),
        "k_near_logit": int(getattr(meta_cfg, "k_near_logit", 32)),
        "k_far_logit": int(getattr(meta_cfg, "k_far_logit", 32)),
        "logit_sampling_mode": str(getattr(meta_cfg, "logit_sampling_mode", "balanced")),
        "logit_stratified_distance_edges": tuple(
            getattr(meta_cfg, "logit_stratified_distance_edges", (0.005, 0.015, 0.03, 0.06))
        ),
        "logit_stratified_quotas": tuple(
            getattr(meta_cfg, "logit_stratified_quotas", (16, 32, 32, 32, 16))
        ),
        "ctx_radius": float(getattr(meta_cfg, "ctx_radius", 0.04)),
        "logit_near_radius": resolve_logit_near_radius(meta_cfg),
        "logit_far_min_radius": resolve_logit_far_min_radius(meta_cfg),
        "logit_neg_radius": float(getattr(meta_cfg, "logit_neg_radius", 0.06)),
        "logit_far_weight": float(getattr(meta_cfg, "loss_cross_edge_far_weight", 0.5)),
        "base_seed": int(seed),
        "augment": bool(getattr(meta_cfg, "augment", True)),
        "apply_hand_perturb": bool(
            getattr(meta_cfg, "apply_hand_perturb", True)
        ),
        "augment_rotation": bool(meta_cfg.augment_rotation),
        "augment_translation": bool(meta_cfg.augment_translation),
        "augment_scale": bool(meta_cfg.augment_scale),
        "rotation_range": float(meta_cfg.rotation_range),
        "translation_range": float(meta_cfg.translation_range),
        "scale_range": tuple(meta_cfg.scale_range),
        "d_pos": float(meta_cfg.d_pos),
        "d_neg": float(meta_cfg.d_neg),
        "gamma": float(meta_cfg.gamma),
        "hand_rot_std_deg": float(meta_cfg.hand_rot_std_deg),
        "hand_trans_std": float(meta_cfg.hand_trans_std),
        "hand_perturb_prob": float(meta_cfg.hand_perturb_prob),
        "fix_overfit_seed": bool(getattr(meta_cfg, "fix_overfit_seed", False)),
        "blacklist_path": getattr(data_cfg, "blacklist_path", None),
    }
    # val_clean：关闭全局增强和手部扰动，保证 GT 几何不被破坏
    val_clean_kwargs = {
        **train_kwargs,
        "augment": False,
        "apply_hand_perturb": False,
        "hand_perturb_prob": 0.0,
    }
    # val_perturbed：保留手部扰动（不改变 GT 几何和 GT logit 邻居集），
    # 用于评估模型在 noisy hand 下的鲁棒性
    val_perturbed_kwargs = {
        **train_kwargs,
        "augment": False,
        "apply_hand_perturb": bool(getattr(meta_cfg, "val_augment", False)),
        "hand_perturb_prob": float(getattr(meta_cfg, "val_hand_perturb_prob", 1.0)),
    }
    train_loader, val_loader, metadata = make_file_split_dataloaders(
        data_cfg,
        seed,
        dataset_cls=CorrStaticDataset,
        file_pattern="**/*.npz",
        train_dataset_kwargs=train_kwargs,
        val_dataset_kwargs=val_clean_kwargs,
        split_group_fn=(
            _sequence_group_key
            if bool(getattr(data_cfg, "group_val_by_sequence", True))
            else None
        ),
        distributed=distributed,
    )
    # 默认开启按 sequence 块打乱，显著降低 .npz 的随机 I/O
    if bool(data_cfg.shuffle) and bool(
        getattr(data_cfg, "sequence_locality_shuffle", True)
    ):
        train_sampler = SequenceLocalitySampler(train_loader.dataset, seed)
        train_sampler = shard_sampler_for_distributed(
            train_sampler,
            distributed=distributed,
            drop_last=bool(data_cfg.drop_last),
            pad=True,
        )
        loader_seed = int(seed) + int(getattr(distributed, "rank", 0) if getattr(distributed, "enabled", False) else 0)
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=int(data_cfg.batch_size),
            shuffle=False,
            sampler=train_sampler,
            **make_dataloader_kwargs(data_cfg, loader_seed),
        )
    val_loaders: dict[str, DataLoader] = {}
    if val_loader is not None:
        val_loader.dataset.set_epoch(0)
        val_loaders["val_clean/"] = val_loader
        # 显式开启 val_augment 时才构造扰动版 val loader
        if bool(getattr(meta_cfg, "val_augment", False)):
            val_perturbed_dataset = CorrStaticDataset(
                val_loader.dataset.data_root,
                file_list=val_loader.dataset.file_paths,
                **val_perturbed_kwargs,
            )
            val_perturbed_dataset.set_epoch(0)
            loader_seed = int(seed) + int(
                getattr(distributed, "rank", 0)
                if getattr(distributed, "enabled", False)
                else 0
            )
            val_perturbed_loader = DataLoader(
                val_perturbed_dataset,
                batch_size=int(getattr(data_cfg, "val_batch_size", None) or data_cfg.batch_size),
                shuffle=False,
                sampler=make_default_eval_sampler(
                    val_perturbed_dataset,
                    distributed=distributed,
                ),
                **make_dataloader_kwargs(data_cfg, loader_seed, drop_last=False),
            )
            val_loaders["val_perturbed/"] = val_perturbed_loader
    # 从第一个 npz 文件读出 dataset 维度信息塞进 metadata，下游 model 用来构图
    first_path = train_loader.dataset.file_paths[0]
    with np.load(first_path, allow_pickle=False) as data:
        hand_finger_id = np.asarray(
            data.get("hand_finger_id", np.full((int(data["hand_points"].shape[1]),), -1, dtype=np.int64))
        )
        hand_region_id = np.asarray(
            data.get("hand_region_id", np.full((int(data["hand_points"].shape[1]),), -1, dtype=np.int64))
        )
        metadata.update(
            {
                "num_obj_pool": int(data["obj_points"].shape[1]),
                "num_obj_points": int(meta_cfg.num_obj_points),
                "num_hand_points": int(data["hand_points"].shape[1]),
                "k_cross": int(data["gt_obj_to_hand_knn_idx"].shape[2]),
                "k_ctx": int(getattr(meta_cfg, "k_ctx", meta_cfg.k_cross)),
                "k_near_logit": int(getattr(meta_cfg, "k_near_logit", 32)),
                "k_far_logit": int(getattr(meta_cfg, "k_far_logit", 32)),
                "logit_sampling_mode": str(getattr(meta_cfg, "logit_sampling_mode", "balanced")),
                "logit_stratified_distance_edges": tuple(
                    getattr(meta_cfg, "logit_stratified_distance_edges", (0.005, 0.015, 0.03, 0.06))
                ),
                "logit_stratified_quotas": tuple(
                    getattr(meta_cfg, "logit_stratified_quotas", (16, 32, 32, 32, 16))
                ),
                "k_logit": (
                    int(sum(getattr(meta_cfg, "logit_stratified_quotas", (16, 32, 32, 32, 16))))
                    if str(getattr(meta_cfg, "logit_sampling_mode", "balanced")).lower() == "stratified"
                    else (
                        int(data["hand_points"].shape[1])
                        if str(getattr(meta_cfg, "logit_sampling_mode", "balanced")).lower() == "dense"
                        else int(getattr(meta_cfg, "k_near_logit", 32)) + int(getattr(meta_cfg, "k_far_logit", 32))
                    )
                ),
                "logit_near_radius": resolve_logit_near_radius(meta_cfg),
                "logit_far_min_radius": resolve_logit_far_min_radius(meta_cfg),
                "k_logit_hard_neg": int(getattr(meta_cfg, "k_logit_hard_neg", 16)),
                "num_fingers": int(np.max(hand_finger_id)) + 1 if hand_finger_id.size > 0 else 0,
                "num_regions": int(np.max(hand_region_id)) + 1 if hand_region_id.size > 0 else 0,
                "fix_overfit_seed": bool(getattr(meta_cfg, "fix_overfit_seed", False)),
                "val_loader_names": sorted(val_loaders),
            }
        )
    return train_loader, val_loader, metadata, val_loaders
