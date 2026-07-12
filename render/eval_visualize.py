"""eval_visualize.py

对单条 Stage 3 序列运行已训练模型，并把 GT 标签与模型预测并排展示在
Open3D 窗口中。

运行示例:
    PYTHONPATH=. python render/eval_visualize.py \\
        --checkpoint outputs/train/<run>/checkpoints/latest.pt \\
        --input    processed_data/generated/stage3/<dataset>/<seq>_<side>.npz

按键说明
--------
A / D   或  ← / →
    上一帧 / 下一帧（步长由 --step 控制，默认 1）。
[ / ]
    上一 epoch / 下一 epoch
    改变对物体点 512 个采样点的稳定哈希种子，从而复现训练时不同 epoch
    的子集。所有 loss / 概率都会按新 epoch 重新计算并刷新。
G
    切换 GT / Eval 显示模式（顶部状态栏 mode）。切换不重置已选中的
    物体点、frame 或 epoch。
C
    切换 heatmap / cross 视图（顶部状态栏 view）。切换不重置已选中的
    物体点、frame 或 epoch，两个视图共享同一份 runtime 状态。
, / .
    上一 / 下一物体点
    改变 selected_rank 选取的物体点。cross 视图下手部颜色会随
    选中的物体点重新计算并刷新。被选中的物体点用一个 yellow sphere
    标记 (默认半径 3mm，由 --marker-radius 调整)，随相机缩放同步。
R
    重置视角到初始相机参数。

显示模式 (G 切换)
================
GT
    显示 Stage 3 的 ground truth 标签:
      * 物体点云按 soft contact label 着色（手部用 GT 干净手）
      * cross 视图下手部按到选中物体点的欧氏距离上色（基于 clean hand）
Eval
    显示模型推理结果:
      * 物体点云按 pred_obj_contact 概率着色（见下方 heatmap 配色）
      * 手部用 noisy hand（手 perturb 后的输入几何）
      * cross 视图下对“选中物体点 × 全部 hand 点”复用现有 edge head
        做 dense 推理，并按 pred_cross_contact 概率连续上色

视图 (C 切换)
=============
heatmap
    整片物体点云按 contact 概率/标签着色，cross 视图里没有边。
    选中的物体点仍以 yellow sphere 高亮。
cross
    普通灰色物体点云 + 1 个 yellow sphere 高亮点（被 ,/. 选中）
    + 手部颜色按模式不同:
      * GT  模式: 全部 hand 点按到选中物体点的欧氏距离上色 (cap=3cm)
      * Eval 模式: 全部 hand 点按 dense pred_cross_contact 概率上色
    按 ,/. 切换物体点会实时重新计算并刷新手部颜色。
    heatmap 视图下被点云的 prob / label 着色完全独立，切换不影响。

颜色说明
========
object 点云 (heatmap 视图)
    GT 模式:
        蓝色 (0.12, 0.36, 0.98)   = 概率 0.0  （无接触）
        白色 (1.0 , 1.0 , 1.0 )   = 概率 0.5  （过渡带）
        红色 (0.98, 0.16, 0.12)   = 概率 1.0  （强接触）
    Eval 模式 (按 sigmoid 后的概率 [0,1] 线性插值):
        蓝色 (0.12, 0.36, 0.98)   = 概率 0.0  （无接触）
        白色 (1.0 , 1.0 , 1.0 )   = 概率 0.5  （不确定）
        红色 (0.98, 0.16, 0.12)   = 概率 1.0  （确定接触）

object 点云 (cross 视图)
    灰色 (0.62, 0.62, 0.66)  全部统一色，被选中的单点用 sphere 高亮

手部点云
    heatmap 视图 / cross 视图无选中点:
        GT 模式:  橙色 (0.95, 0.58, 0.12)  = clean hand
        Eval 模式: 品红 (0.85, 0.16, 0.85)  = noisy hand
    cross 视图 + 有选中点 (GT 模式，按 soft_contact_label 概率上色):
        红色 (0.98, 0.16, 0.12)  = 概率 1.0  (更接近 / 更可能接触)
        灰色 (0.18, 0.18, 0.22)  = 概率 0.0  (更远 / 更不可能接触)
        概率由 clean hand 到选中物体点的距离经 d_pos/d_neg/gamma 映射得到
    cross 视图 + 有选中点 (Eval 模式，按 pred_cross_contact 概率上色):
        红色 (0.98, 0.16, 0.12)  = 概率 1.0  (模型更认为接触)
        灰色 (0.18, 0.18, 0.22)  = 概率 0.0  (模型更认为不接触)
        中间概率在线性插值为红灰渐变
        这里是“选中物体点 × 全部 hand 点”的 dense 推理结果，不再只限于 KNN

被选中的物体点 (任意视图)
    黄色 sphere (1.0, 0.95, 0.15)，默认半径 3mm (--marker-radius)
    缩放时按世界尺度放大；位置跟随机身实时更新。

CLI 示例
========
# 默认打开
PYTHONPATH=. python render/eval_visualize.py \\
    --checkpoint outputs/train/<run>/checkpoints/latest.pt \\
    --input    processed_data/generated/stage3/.../bowl_pass_1_right.npz

# 指定 GPU、并从第 5 帧、epoch 3 开始
PYTHONPATH=. python render/eval_visualize.py \\
    --checkpoint outputs/train/<run>/checkpoints/latest.pt \\
    --input    processed_data/generated/stage3/.../bowl_pass_1_right.npz \\
    --device cuda --frame 5 --epoch 3

# 仅作数据检查 + 单次推理（不开窗口）
PYTHONPATH=. python render/eval_visualize.py \\
    --checkpoint ... --input ... --check-only
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate

from src.base import load_checkpoint
from src.task.correspondence_ptv3.config_loader import load_correspondence_config
from src.task.correspondence_ptv3.data import _compute_runtime_context_neighbors
from src.task.correspondence_ptv3.sampling import (
    augment_geometry,
    sample_object_indices,
    stable_frame_seed,
)
from src.task.correspondence_ptv3.runner import CorrespondencePTV3Runner
from src.utils.correspondence import (
    decode_contact_logits,
    soft_contact_label,
)


os.environ.setdefault("DISPLAY", "localhost:10.0")


REQUIRED_FIELDS = {
    "schema_name",
    "raw_frame_id",
    "seq_id",
    "side",
    "obj_points",
    "obj_normals",
    "obj_point_id",
    "hand_points",
    "hand_normals",
    "hand_point_id",
    "hand_cano_points",
    "hand_finger_id",
    "hand_region_id",
    "obj_to_hand_min_dist",
    "obj_candidate_mask_5cm",
    "gt_obj_to_hand_knn_idx",
}


@dataclass
class RuntimeFrame:
    raw_frame_id: int
    sample_seed: int
    augmentation_seed: int
    hand_perturbed: bool
    distance_scale: float
    obj_points: np.ndarray
    obj_normals: np.ndarray
    obj_valid: np.ndarray
    selected_obj_idx: np.ndarray
    selected_obj_point_id: np.ndarray
    selected_obj_min_dist: np.ndarray
    obj_contact_soft: np.ndarray
    obj_contact_hard: np.ndarray
    clean_hand_points: np.ndarray
    clean_hand_normals: np.ndarray
    noisy_hand_points: np.ndarray
    noisy_hand_normals: np.ndarray
    clean_knn_idx: np.ndarray
    input_ctx_idx: np.ndarray
    input_ctx_valid: np.ndarray
    input_logit_idx: np.ndarray
    input_logit_valid: np.ndarray


@dataclass
class PredictionFrame:
    pred_obj_contact: np.ndarray
    pred_cross_contact: np.ndarray
    obj_dense_tokens: torch.Tensor
    hand_dense_tokens: torch.Tensor


def _scalar(data: Any, key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = np.asarray(data[key])
    return str(value.item()) if value.size == 1 else default


def _add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
) -> None:
    destination = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true")
    group.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: default})


def _validate(data: dict[str, np.ndarray], path: Path) -> dict[str, int]:
    missing = REQUIRED_FIELDS.difference(data.keys())
    if missing:
        raise KeyError(f"{path}: missing Stage 3 fields {sorted(missing)}")
    if _scalar(data, "schema_name") != "train_corr_static":
        raise ValueError(
            f"{path}: expected schema_name='train_corr_static', "
            f"got {_scalar(data, 'schema_name')!r}"
        )
    obj = np.asarray(data["obj_points"])
    hand = np.asarray(data["hand_points"])
    obj_normals = np.asarray(data["obj_normals"])
    hand_normals = np.asarray(data["hand_normals"])
    min_dist = np.asarray(data["obj_to_hand_min_dist"])
    candidate = np.asarray(data["obj_candidate_mask_5cm"])
    knn = np.asarray(data["gt_obj_to_hand_knn_idx"])
    if obj.ndim != 3 or obj.shape[-1] != 3:
        raise ValueError(f"obj_points must be [T,4096,3], got {obj.shape}")
    frames, pool, _ = obj.shape
    expected = {
        "obj_normals": (frames, pool, 3),
        "obj_point_id": (pool,),
        "hand_points": (frames, hand.shape[1], 3),
        "hand_normals": (frames, hand.shape[1], 3),
        "hand_point_id": (hand.shape[1],),
        "hand_cano_points": (hand.shape[1], 3),
        "hand_finger_id": (hand.shape[1],),
        "hand_region_id": (hand.shape[1],),
        "obj_to_hand_min_dist": (frames, pool),
        "obj_candidate_mask_5cm": (frames, pool),
    }
    for key, shape in expected.items():
        actual = np.asarray(data[key]).shape
        if actual != shape:
            raise ValueError(f"{key}: expected {shape}, got {actual}")
    if knn.shape[:2] != (frames, pool) or knn.ndim != 3:
        raise ValueError(f"gt_obj_to_hand_knn_idx has invalid shape {knn.shape}")
    if not np.array_equal(candidate, min_dist <= 0.05):
        mismatch = int(np.count_nonzero(candidate != (min_dist <= 0.05)))
        raise ValueError(f"5cm candidate mask disagrees with distances at {mismatch} points")
    if np.any(knn[~candidate] != -1):
        raise ValueError("Non-candidate KNN rows must be padded with -1")
    if np.any(knn[candidate] < 0) or np.any(knn[candidate] >= hand.shape[1]):
        raise ValueError("Candidate KNN rows contain an invalid hand index")
    if not all(np.isfinite(value).all() for value in (obj, obj_normals, hand, hand_normals, min_dist)):
        raise ValueError("Stage 3 geometry contains non-finite values")
    return {
        "frames": frames,
        "pool": pool,
        "hand": hand.shape[1],
        "k": knn.shape[2],
        "candidate_min": int(candidate.sum(axis=1).min()),
        "candidate_median": int(np.median(candidate.sum(axis=1))),
        "candidate_max": int(candidate.sum(axis=1).max()),
    }


def _probability_colors(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0)
    low = probability <= 0.5
    colors = np.empty((probability.shape[0], 3), dtype=np.float32)
    alpha = (probability[low] / 0.5)[:, None]
    colors[low] = (
        np.asarray([0.12, 0.36, 0.98], dtype=np.float32)[None] * (1.0 - alpha)
        + np.asarray([1.0, 1.0, 1.0], dtype=np.float32)[None] * alpha
    )
    beta = ((probability[~low] - 0.5) / 0.5)[:, None]
    colors[~low] = (
        np.asarray([1.0, 1.0, 1.0], dtype=np.float32)[None] * (1.0 - beta)
        + np.asarray([0.98, 0.16, 0.12], dtype=np.float32)[None] * beta
    )
    return colors


def _binary_contact_colors(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    colors = np.tile(
        np.asarray([0.45, 0.45, 0.48], dtype=np.float32)[None],
        (mask.shape[0], 1),
    )
    colors[mask] = np.asarray([0.98, 0.18, 0.12], dtype=np.float32)
    return colors


def _red_gray_colors(score: np.ndarray) -> np.ndarray:
    score = np.clip(np.asarray(score, dtype=np.float32), 0.0, 1.0)
    near_red = np.asarray([0.98, 0.16, 0.12], dtype=np.float32)
    far_gray = np.asarray([0.18, 0.18, 0.22], dtype=np.float32)
    return far_gray[None] * (1.0 - score[:, None]) + near_red[None] * score[:, None]


def _distance_to_contact_probability(
    distance: np.ndarray,
    *,
    d_pos: float,
    d_neg: float,
    gamma: float,
) -> np.ndarray:
    distance_tensor = torch.from_numpy(np.asarray(distance, dtype=np.float32))
    probability = soft_contact_label(
        distance_tensor,
        d_pos=float(d_pos),
        d_neg=float(d_neg),
        gamma=float(gamma),
    )
    return probability.detach().cpu().numpy().astype(np.float32)


def _probability_hand_colors(probabilities: np.ndarray) -> np.ndarray:
    return _red_gray_colors(np.asarray(probabilities, dtype=np.float32))


def _build_runtime_frame(
    data: dict[str, np.ndarray],
    frame: int,
    epoch: int,
    args: argparse.Namespace,
    *,
    edge_sampler: Any,
) -> RuntimeFrame:
    seq_id = _scalar(data, "seq_id", "unknown")
    side = _scalar(data, "side", "")
    raw_frame_id = int(np.asarray(data["raw_frame_id"])[frame])
    seed_epoch = 0 if bool(args.fix_overfit_seed) else epoch
    sample_seed = stable_frame_seed(
        base_seed=args.base_seed,
        seq_id=seq_id,
        side=side,
        raw_frame_id=raw_frame_id,
        epoch=seed_epoch,
    )
    selected_idx, obj_valid = sample_object_indices(
        np.asarray(data["obj_candidate_mask_5cm"][frame]),
        num_samples=args.num_obj_points,
        seed=sample_seed,
    )
    safe_idx = np.maximum(selected_idx, 0)
    obj_points = np.asarray(data["obj_points"][frame, safe_idx], dtype=np.float32).copy()
    obj_normals = np.asarray(data["obj_normals"][frame, safe_idx], dtype=np.float32).copy()
    obj_point_id = np.asarray(data["obj_point_id"][safe_idx], dtype=np.int64).copy()
    obj_min_dist = np.asarray(
        data["obj_to_hand_min_dist"][frame, safe_idx],
        dtype=np.float32,
    ).copy()
    clean_knn = np.asarray(
        data["gt_obj_to_hand_knn_idx"][frame, safe_idx],
        dtype=np.int64,
    ).copy()
    obj_points[~obj_valid] = 0
    obj_normals[~obj_valid] = 0
    obj_point_id[~obj_valid] = -1
    obj_min_dist[~obj_valid] = 0
    clean_knn[~obj_valid] = -1

    hand_points = np.asarray(data["hand_points"][frame], dtype=np.float32)
    hand_normals = np.asarray(data["hand_normals"][frame], dtype=np.float32)
    augmentation_seed = stable_frame_seed(
        base_seed=args.base_seed,
        seq_id=seq_id,
        side=side,
        raw_frame_id=raw_frame_id,
        epoch=seed_epoch,
        namespace="augmentation",
    )
    geometry = augment_geometry(
        obj_points=obj_points,
        obj_normals=obj_normals,
        hand_points=hand_points,
        hand_normals=hand_normals,
        seed=augmentation_seed,
        apply_hand_perturb=bool(args.hand_perturb),
        hand_rot_std_deg=float(args.hand_rot_std_deg),
        hand_trans_std=float(args.hand_trans_std),
        hand_perturb_prob=float(args.hand_perturb_prob),
        apply_global_aug=bool(args.augment),
        augment_rotation=bool(args.augment_rotation),
        rotation_range_deg=float(args.rotation_range),
        augment_translation=bool(args.augment_translation),
        translation_range=float(args.translation_range),
        augment_scale=bool(args.augment_scale),
        scale_range=tuple(args.scale_range),
    )
    obj_min_dist *= float(geometry.distance_scale)

    input_ctx_idx, input_ctx_valid = _compute_runtime_context_neighbors(
        geometry.input_obj_points,
        geometry.input_hand_points,
        obj_valid,
        k_ctx=int(args.k_ctx),
        ctx_radius=float(args.ctx_radius),
    )
    edge_sample = edge_sampler.sample(
        gt_obj_points=geometry.gt_obj_points,
        gt_hand_points=geometry.gt_hand_points,
        obj_valid=obj_valid,
        seed=stable_frame_seed(
            base_seed=args.base_seed,
            seq_id=seq_id,
            side=side,
            raw_frame_id=raw_frame_id,
            epoch=seed_epoch,
            namespace="logit-neighbors",
        ),
    )
    input_logit_idx = edge_sample.idx
    input_logit_valid = edge_sample.valid_mask
    obj_contact_soft = soft_contact_label(
        torch.from_numpy(obj_min_dist),
        d_pos=float(args.d_pos),
        d_neg=float(args.d_neg),
        gamma=float(args.gamma),
    ).numpy()
    obj_contact_hard = np.logical_and(obj_valid, obj_min_dist <= float(args.d_pos))
    return RuntimeFrame(
        raw_frame_id=raw_frame_id,
        sample_seed=sample_seed,
        augmentation_seed=augmentation_seed,
        hand_perturbed=geometry.hand_perturbed,
        distance_scale=float(geometry.distance_scale),
        obj_points=geometry.gt_obj_points,
        obj_normals=geometry.gt_obj_normals,
        obj_valid=obj_valid,
        selected_obj_idx=selected_idx,
        selected_obj_point_id=obj_point_id,
        selected_obj_min_dist=obj_min_dist,
        obj_contact_soft=obj_contact_soft,
        obj_contact_hard=obj_contact_hard,
        clean_hand_points=geometry.gt_hand_points,
        clean_hand_normals=geometry.gt_hand_normals,
        noisy_hand_points=geometry.input_hand_points,
        noisy_hand_normals=geometry.input_hand_normals,
        clean_knn_idx=clean_knn,
        input_ctx_idx=input_ctx_idx,
        input_ctx_valid=input_ctx_valid,
        input_logit_idx=input_logit_idx,
        input_logit_valid=input_logit_valid,
    )


def _build_model_batch(
    runtime: RuntimeFrame,
    data: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    points = np.concatenate([runtime.obj_points, runtime.noisy_hand_points], axis=0)
    normals = np.concatenate([runtime.obj_normals, runtime.noisy_hand_normals], axis=0)
    point_valid_mask = np.concatenate(
        [runtime.obj_valid, np.ones((runtime.noisy_hand_points.shape[0],), dtype=bool)],
        axis=0,
    )
    batch = {
        "points": torch.from_numpy(points).float().unsqueeze(0),
        "normals": torch.from_numpy(normals).float().unsqueeze(0),
        "point_valid_mask": torch.from_numpy(point_valid_mask).unsqueeze(0),
        "input_obj_to_hand_ctx_idx": torch.from_numpy(runtime.input_ctx_idx).long().unsqueeze(0),
        "input_obj_to_hand_ctx_valid_mask": torch.from_numpy(runtime.input_ctx_valid).unsqueeze(0),
        "input_obj_to_hand_logit_idx": torch.from_numpy(runtime.input_logit_idx).long().unsqueeze(0),
        "input_obj_to_hand_logit_valid_mask": torch.from_numpy(runtime.input_logit_valid).unsqueeze(0),
        "hand_cano_points": torch.from_numpy(np.asarray(data["hand_cano_points"])).float().unsqueeze(0),
    }
    return batch


def _run_inference(
    runner: Any,
    runtime: RuntimeFrame,
    data: dict[str, np.ndarray],
) -> PredictionFrame:
    batch = _build_model_batch(runtime, data)
    try:
        with torch.no_grad():
            preds = runner.inference(runner.model, batch)
    except AssertionError as exc:
        message = str(exc)
        if "implicit gemm only support cuda" in message.lower():
            raise RuntimeError(
                "This checkpoint's PTv3/spconv inference path requires CUDA. "
                "Run eval_visualize with --device cuda or --device auto on a GPU machine."
            ) from exc
        raise
    pred_obj_contact = torch.sigmoid(preds["pred_obj_contact"][0]).detach().cpu().numpy()
    pred_cross_contact = torch.sigmoid(preds["pred_cross_contact"][0]).detach().cpu().numpy()
    pred_obj_contact[~runtime.obj_valid] = 0
    pred_cross_contact[~runtime.input_logit_valid] = 0
    return PredictionFrame(
        pred_obj_contact=pred_obj_contact,
        pred_cross_contact=pred_cross_contact,
        obj_dense_tokens=preds["obj_dense_tokens"][0].detach(),
        hand_dense_tokens=preds["hand_dense_tokens"][0].detach(),
    )


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple((key, _freeze_config_value(val)) for key, val in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _edge_sampler_signature(config: dict[str, Any]) -> tuple[Any, ...]:
    return _freeze_config_value(config)


def _dense_cross_probabilities(
    model: Any,
    prediction: PredictionFrame,
    runtime: RuntimeFrame,
    selected_slot: int,
) -> np.ndarray:
    device = prediction.obj_dense_tokens.device
    with torch.no_grad():
        obj_token = prediction.obj_dense_tokens[selected_slot].to(device=device).unsqueeze(0).unsqueeze(0)
        hand_tokens = prediction.hand_dense_tokens.to(device=device).unsqueeze(0).unsqueeze(0)

        obj_point = torch.from_numpy(
            np.asarray(runtime.obj_points[selected_slot], dtype=np.float32)
        ).to(device=device)
        obj_normal = torch.from_numpy(
            np.asarray(runtime.obj_normals[selected_slot], dtype=np.float32)
        ).to(device=device)
        hand_points = torch.from_numpy(
            np.asarray(runtime.noisy_hand_points, dtype=np.float32)
        ).to(device=device)
        hand_normals = torch.from_numpy(
            np.asarray(runtime.noisy_hand_normals, dtype=np.float32)
        ).to(device=device)

        edge_shared = model._compute_shared_edge_features(
            z_obj_cross=obj_token,
            z_hand_neighbors=hand_tokens,
        )
        dense_logits = model._compute_cross_edge_predictions(edge_shared)[0, 0]
        dense_prob = decode_contact_logits(
            dense_logits,
            supervision_mode=model.contact_supervision_mode,
            mode=model.contact_bin_decode_mode,
        )
    return dense_prob.detach().cpu().numpy().astype(np.float32)


def _build_selected_edges(
    obj_point: np.ndarray,
    hand_points: np.ndarray,
    hand_indices: np.ndarray,
    edge_mask: np.ndarray,
    edge_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_idx = np.flatnonzero(edge_mask)
    if valid_idx.size == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )
    points: list[np.ndarray] = []
    lines: list[list[int]] = []
    colors: list[np.ndarray] = []
    for rank in valid_idx.tolist():
        hand_idx = int(hand_indices[rank])
        if hand_idx < 0 or hand_idx >= len(hand_points):
            continue
        start = len(points)
        points.extend((obj_point, hand_points[hand_idx]))
        lines.append([start, start + 1])
        colors.append(edge_colors[rank])
    if not lines:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0, 3), dtype=np.float64),
        )
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(lines, dtype=np.int32),
        np.asarray(colors, dtype=np.float64),
    )


class EvalViewer:
    def __init__(
        self,
        data: dict[str, np.ndarray],
        runner: Any,
        args: argparse.Namespace,
    ) -> None:
        import open3d as o3d

        self.o3d = o3d
        self.data = data
        self.runner = runner
        self.args = args
        self.frame = int(np.clip(args.frame, 0, len(data["raw_frame_id"]) - 1))
        self.epoch = max(0, int(args.epoch))
        self.step = max(1, int(args.step))
        self.show_gt = bool(args.start_gt)
        self.show_cross = bool(args.start_cross)
        self.selected_rank = 0
        self._selected_marker_position: np.ndarray | None = None
        self.geometries: dict[str, Any] = {}
        self._cache_signature: tuple[Any, ...] | None = None
        self._cache_runtime: RuntimeFrame | None = None
        self._cache_prediction: PredictionFrame | None = None
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        if not self.vis.create_window(
            window_name=f"Eval audit: {_scalar(data, 'seq_id')} {_scalar(data, 'side')}",
            width=args.width,
            height=args.height,
        ):
            raise RuntimeError(
                "Open3D window creation failed. Check DISPLAY; expected "
                f"{os.environ.get('DISPLAY')!r}."
            )
        options = self.vis.get_render_option()
        options.point_size = float(args.point_size)
        options.line_width = float(args.line_width)
        options.background_color = np.asarray([0.035, 0.035, 0.045])
        self._register_callbacks()
        self.refresh(reset_view=True)

    def _register_callbacks(self) -> None:
        for key in (262, ord("D")):
            self.vis.register_key_callback(key, lambda vis: self._move(+self.step))
        for key in (263, ord("A")):
            self.vis.register_key_callback(key, lambda vis: self._move(-self.step))
        self.vis.register_key_callback(ord("["), lambda vis: self._change_epoch(-1))
        self.vis.register_key_callback(ord("]"), lambda vis: self._change_epoch(+1))
        self.vis.register_key_callback(ord("G"), lambda vis: self._toggle_mode())
        self.vis.register_key_callback(ord("C"), lambda vis: self._toggle_view())
        self.vis.register_key_callback(ord(","), lambda vis: self._select_object(-1))
        self.vis.register_key_callback(ord("."), lambda vis: self._select_object(+1))
        self.vis.register_key_callback(ord("R"), lambda vis: self._reset_view())

    def _runtime_signature(self) -> tuple[Any, ...]:
        return (
            self.frame,
            self.epoch,
            self.args.base_seed,
            self.args.fix_overfit_seed,
            self.args.edge_sampler_signature,
            self.args.num_obj_points,
            self.args.k_cross,
            self.args.k_ctx,
            self.args.ctx_radius,
            self.args.augment,
            self.args.hand_perturb,
            self.args.augment_rotation,
            self.args.augment_translation,
            self.args.augment_scale,
            self.args.rotation_range,
            self.args.translation_range,
            tuple(self.args.scale_range),
            self.args.hand_rot_std_deg,
            self.args.hand_trans_std,
            self.args.hand_perturb_prob,
            self.args.d_pos,
            self.args.d_neg,
            self.args.gamma,
        )

    def _get_cached_state(self) -> tuple[RuntimeFrame, PredictionFrame]:
        signature = self._runtime_signature()
        if signature != self._cache_signature:
            runtime = _build_runtime_frame(
                self.data,
                self.frame,
                self.epoch,
                self.args,
                edge_sampler=self.args.edge_sampler,
            )
            prediction = _run_inference(self.runner, runtime, self.data)
            self._cache_signature = signature
            self._cache_runtime = runtime
            self._cache_prediction = prediction
        assert self._cache_runtime is not None
        assert self._cache_prediction is not None
        return self._cache_runtime, self._cache_prediction

    def _action_label(self, action: str) -> str:
        return {
            "move_next": "next_frame",
            "move_prev": "prev_frame",
            "epoch_next": "next_epoch",
            "epoch_prev": "prev_epoch",
            "toggle_mode": "toggle_mode",
            "toggle_view": "toggle_view",
            "object_next": "next_object",
            "object_prev": "prev_object",
            "reset_view": "reset_view",
        }.get(action, action)

    def _move(self, delta: int) -> bool:
        self.frame = int(np.clip(self.frame + delta, 0, len(self.data["raw_frame_id"]) - 1))
        self.refresh(action="move_next" if delta > 0 else "move_prev")
        return False

    def _change_epoch(self, delta: int) -> bool:
        self.epoch = max(0, self.epoch + delta)
        self.refresh(action="epoch_next" if delta > 0 else "epoch_prev")
        return False

    def _toggle_mode(self) -> bool:
        self.show_gt = not self.show_gt
        self.refresh(action="toggle_mode")
        return False

    def _toggle_view(self) -> bool:
        self.show_cross = not self.show_cross
        self.refresh(action="toggle_view")
        return False

    def _select_object(self, delta: int) -> bool:
        runtime, _ = self._get_cached_state()
        valid_count = int(runtime.obj_valid.sum())
        if valid_count > 0:
            self.selected_rank = int(np.clip(self.selected_rank + delta, 0, valid_count - 1))
        self.refresh(action="object_next" if delta > 0 else "object_prev")
        return False

    def _reset_view(self) -> bool:
        self.refresh(reset_view=True, action="reset_view")
        return False

    def _set_point_cloud(
        self,
        name: str,
        points: np.ndarray | None,
        color: np.ndarray,
        per_point_color: np.ndarray | None = None,
    ) -> None:
        if points is None or len(points) == 0:
            self._remove(name)
            return
        geometry = self.geometries.get(name)
        is_new = geometry is None
        if is_new:
            geometry = self.o3d.geometry.PointCloud()
            self.geometries[name] = geometry
        geometry.points = self.o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        colors = (
            np.asarray(per_point_color, dtype=np.float64)
            if per_point_color is not None
            else np.tile(np.asarray(color, dtype=np.float64)[None], (len(points), 1))
        )
        geometry.colors = self.o3d.utility.Vector3dVector(colors)
        if is_new:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            self.vis.update_geometry(geometry)

    def _set_lines(
        self,
        name: str,
        points: np.ndarray,
        lines: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        if len(lines) == 0:
            self._remove(name)
            return
        geometry = self.geometries.get(name)
        is_new = geometry is None
        if is_new:
            geometry = self.o3d.geometry.LineSet()
            self.geometries[name] = geometry
        geometry.points = self.o3d.utility.Vector3dVector(points)
        geometry.lines = self.o3d.utility.Vector2iVector(lines)
        geometry.colors = self.o3d.utility.Vector3dVector(colors)
        if is_new:
            self.vis.add_geometry(geometry, reset_bounding_box=False)
        else:
            self.vis.update_geometry(geometry)

    def _remove(self, name: str) -> None:
        geometry = self.geometries.pop(name, None)
        if geometry is not None:
            self.vis.remove_geometry(geometry, reset_bounding_box=False)
        if name == "selected_object":
            self._selected_marker_position = None

    def _set_sphere_marker(
        self,
        name: str,
        position: np.ndarray,
        color: np.ndarray,
        radius: float,
    ) -> None:
        # 用一个 sphere mesh 标记选中物体点，半径按米计，缩放时会跟着放大。
        # 通过记录上次的 world position，更新时只做相对位移，避免重置整个 mesh。
        position = np.asarray(position, dtype=np.float64)
        sphere = self.geometries.get(name)
        is_new = sphere is None
        if is_new:
            sphere = self.o3d.geometry.TriangleMesh.create_sphere(radius=float(radius))
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(np.asarray(color, dtype=np.float64))
            self.geometries[name] = sphere
            sphere.translate(position)
            self._selected_marker_position = position.copy()
        else:
            prev = self._selected_marker_position
            if prev is not None:
                sphere.translate(position - prev)
            else:
                sphere.translate(position)
            self._selected_marker_position = position.copy()
        if is_new:
            self.vis.add_geometry(sphere, reset_bounding_box=False)
        else:
            self.vis.update_geometry(sphere)

    def refresh(self, reset_view: bool = False, action: str | None = None) -> None:
        runtime, prediction = self._get_cached_state()
        valid_slots = np.flatnonzero(runtime.obj_valid)
        if valid_slots.size == 0:
            self.selected_rank = 0
            selected_slot = None
        else:
            self.selected_rank = int(np.clip(self.selected_rank, 0, valid_slots.size - 1))
            selected_slot = int(valid_slots[self.selected_rank])

        obj_points = runtime.obj_points[valid_slots]
        if self.show_cross:
            obj_colors = np.tile(
                np.asarray([0.62, 0.62, 0.66], dtype=np.float32)[None],
                (len(valid_slots), 1),
            )
        elif self.show_gt:
            obj_colors = _probability_colors(runtime.obj_contact_soft[valid_slots])
        else:
            obj_colors = _probability_colors(prediction.pred_obj_contact[valid_slots])
        self._set_point_cloud(
            "object_points",
            obj_points,
            np.asarray([0.8, 0.8, 0.8]),
            obj_colors,
        )

        hand_points = runtime.clean_hand_points if self.show_gt else runtime.noisy_hand_points
        hand_distances: np.ndarray | None = None
        dense_cross_prob: np.ndarray | None = None
        if self.show_cross and selected_slot is not None:
            selected_obj_point = runtime.obj_points[selected_slot]
            hand_distances = np.linalg.norm(
                hand_points - selected_obj_point[None],
                axis=-1,
            ).astype(np.float32)
        if self.show_cross and selected_slot is not None:
            # cross 视图: GT 用真实欧氏距离上色, Eval 用 pred_cross_contact 概率上色.
            if self.show_gt:
                # GT: clean hand 到选中物体点的距离，先映射成 soft label 概率。
                assert hand_distances is not None
                gt_cross_prob = _distance_to_contact_probability(
                    hand_distances,
                    d_pos=float(self.args.d_pos),
                    d_neg=float(self.args.d_neg),
                    gamma=float(self.args.gamma),
                )
                hand_colors = _probability_hand_colors(gt_cross_prob)
            else:
                # Eval: 对“选中物体点 × 全部 hand 点”直接跑 dense edge head。
                dense_cross_prob = _dense_cross_probabilities(
                    self.runner.model,
                    prediction,
                    runtime,
                    selected_slot,
                )
                hand_colors = _probability_hand_colors(dense_cross_prob)
            self._set_point_cloud(
                "hand_points",
                hand_points,
                np.asarray([0.0, 0.0, 0.0]),
                hand_colors,
            )
        else:
            hand_color = (
                np.asarray([0.95, 0.58, 0.12])
                if self.show_gt
                else np.asarray([0.85, 0.16, 0.85])
            )
            self._set_point_cloud("hand_points", hand_points, hand_color)

        if selected_slot is None:
            self._remove("selected_object")
            selected_pred = 0.0
            selected_pool_idx = -1
            selected_point_id = -1
            selected_min_dist = 0.0
        else:
            self._set_sphere_marker(
                "selected_object",
                runtime.obj_points[selected_slot],
                np.asarray([1.0, 0.95, 0.15]),
                radius=float(self.args.marker_radius),
            )
            selected_pred = float(prediction.pred_obj_contact[selected_slot])
            selected_pool_idx = int(runtime.selected_obj_idx[selected_slot])
            selected_point_id = int(runtime.selected_obj_point_id[selected_slot])
            selected_min_dist = float(runtime.selected_obj_min_dist[selected_slot])

        mode_name = "GT" if self.show_gt else "Eval"
        view_name = "cross" if self.show_cross else "heatmap"
        if hand_distances is not None:
            cross_min = float(hand_distances.min())
        else:
            cross_min = 0.0
        if dense_cross_prob is not None:
            cross_prob_max = float(dense_cross_prob.max())
        else:
            cross_prob_max = 0.0
        status = (
            f"[eval] frame={self.frame}/{len(self.data['raw_frame_id']) - 1} "
            f"raw={runtime.raw_frame_id} epoch={self.epoch} mode={mode_name} "
            f"view={view_name} valid={int(runtime.obj_valid.sum())} "
            f"obj={self.selected_rank + 1 if selected_slot is not None else 0}/"
            f"{int(runtime.obj_valid.sum())} pool_idx={selected_pool_idx} "
            f"point_id={selected_point_id} dist={selected_min_dist:.4f} "
            f"hand_min={cross_min:.4f} cross_max={cross_prob_max:.3f} "
            f"pred={selected_pred:.3f} "
            f"hand_noise={runtime.hand_perturbed} seed={runtime.sample_seed}"
        )
        if action is None:
            print(f"\r{status}   ", end="", flush=True)
        else:
            print(
                f"\n[{self._action_label(action)}] {status}",
                flush=True,
            )
        if reset_view:
            self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()

    def run(self) -> None:
        try:
            self.vis.run()
        finally:
            print()
            self.vis.destroy_window()

    def smoke(self, frame_count: int) -> None:
        try:
            for _ in range(max(0, int(frame_count) - 1)):
                self.frame = min(self.frame + self.step, len(self.data["raw_frame_id"]) - 1)
                self.refresh()
        finally:
            print()
            self.vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize GT vs model predictions on a single Stage 3 sequence."
    )
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or checkpoint.pt path")
    parser.add_argument("--input", required=True, help="Stage 3 .npz file")
    parser.add_argument("--config", default=None, help="Optional Hydra config name or saved config path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Optional Hydra-style override, for example edge_sampler=stratified.",
    )
    parser.add_argument("--device", default="auto", help="Device override, for example cpu or cuda:0")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=None)
    parser.add_argument("--num-obj-points", type=int, default=None)
    parser.add_argument("--k-cross", type=int, default=None)
    parser.add_argument("--k-ctx", type=int, default=None)
    parser.add_argument("--ctx-radius", type=float, default=None)
    parser.add_argument("--marker-radius", type=float, default=0.003)

    _add_bool_flag(parser, "augment", default=False)
    _add_bool_flag(parser, "hand-perturb", default=False)
    _add_bool_flag(parser, "augment-rotation", default=True)
    _add_bool_flag(parser, "augment-translation", default=False)
    _add_bool_flag(parser, "augment-scale", default=False)
    parser.add_argument("--rotation-range", type=float, default=180.0)
    parser.add_argument("--translation-range", type=float, default=0.1)
    parser.add_argument("--scale-range", type=float, nargs=2, default=(0.9, 1.1))
    parser.add_argument("--hand-rot-std-deg", type=float, default=10.0)
    parser.add_argument("--hand-trans-std", type=float, default=0.01)
    parser.add_argument("--hand-perturb-prob", type=float, default=1.0)

    parser.add_argument("--d-pos", type=float, default=None)
    parser.add_argument("--d-neg", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)

    _add_bool_flag(parser, "start-gt", default=False)
    _add_bool_flag(parser, "start-cross", default=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--smoke-frames",
        type=int,
        default=0,
        help="Open a window, update N frames, then close.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--line-width", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    with np.load(input_path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    stats = _validate(data, input_path)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    config_source = args.config if args.config is not None else checkpoint["config"]
    cfg = load_correspondence_config(
        config_source,
        overrides=list(args.set),
    )
    cfg.train.device = args.device
    runner = CorrespondencePTV3Runner(
        cfg,
        mode="eval",
        checkpoint=args.checkpoint,
        build_data=False,
    )
    runner.setup_inference(args.checkpoint)

    args.edge_sampler = instantiate(runner.cfg.edge_sampler)
    args.edge_sampler_signature = _edge_sampler_signature(runner.cfg.edge_sampler)
    args.edge_sampler_name = type(args.edge_sampler).__name__
    args.fix_overfit_seed = bool(getattr(runner.cfg.meta, "fix_overfit_seed", False))

    if args.base_seed is None:
        args.base_seed = int(runner.cfg.train.seed)
    if args.num_obj_points is None:
        args.num_obj_points = int(runner.cfg.meta.num_obj_points)
    if args.k_cross is None:
        args.k_cross = int(runner.cfg.meta.k_cross)
    if args.k_ctx is None:
        args.k_ctx = int(getattr(runner.cfg.meta, "k_ctx", runner.cfg.meta.k_cross))
    if args.ctx_radius is None:
        args.ctx_radius = float(getattr(runner.cfg.meta, "ctx_radius", 0.04))
    if args.d_pos is None:
        args.d_pos = float(runner.cfg.meta.d_pos)
    if args.d_neg is None:
        args.d_neg = float(runner.cfg.meta.d_neg)
    if args.gamma is None:
        args.gamma = float(runner.cfg.meta.gamma)
    if int(args.num_obj_points) != int(runner.cfg.meta.num_obj_points):
        raise ValueError(
            f"--num-obj-points={args.num_obj_points} does not match checkpoint "
            f"meta.num_obj_points={runner.cfg.meta.num_obj_points}."
        )
    if int(args.k_cross) != int(runner.cfg.meta.k_cross):
        raise ValueError(
            f"--k-cross={args.k_cross} does not match checkpoint "
            f"meta.k_cross={runner.cfg.meta.k_cross}."
        )
    if int(args.k_ctx) <= 0:
        raise ValueError("--k-ctx must be positive.")
    args.frame = int(np.clip(args.frame, 0, stats["frames"] - 1))

    runtime = _build_runtime_frame(
        data,
        args.frame,
        max(0, args.epoch),
        args,
        edge_sampler=args.edge_sampler,
    )
    prediction = _run_inference(runner, runtime, data)
    valid_contact = prediction.pred_obj_contact[runtime.obj_valid]
    print(
        f"[eval] {input_path}\n"
        f"  checkpoint={Path(args.checkpoint).expanduser()}\n"
        f"  seq={_scalar(data, 'seq_id')} side={_scalar(data, 'side')} "
        f"frames={stats['frames']} pool={stats['pool']} hand={stats['hand']} "
        f"K={stats['k']}\n"
        f"  candidate[min/median/max]={stats['candidate_min']}/"
        f"{stats['candidate_median']}/{stats['candidate_max']}\n"
        f"  fix_overfit_seed={str(args.fix_overfit_seed).lower()} "
        f"edge_sampler={args.edge_sampler_name} "
        f"num_obj_points={int(args.num_obj_points)} "
        f"num_hand_points={int(stats['hand'])}\n"
        f"  frame={args.frame} epoch={max(0, args.epoch)} "
        f"selected={int(runtime.obj_valid.sum())} "
        f"padding={int((~runtime.obj_valid).sum())} "
        f"pred_mean={float(valid_contact.mean()) if valid_contact.size > 0 else 0.0:.4f}"
    )
    if args.check_only:
        return
    print(
        "Keys: Left/A Right/D frames; [/] epoch; "
        "G GT/Eval; C heatmap/cross; ,/. object; R reset"
    )
    viewer = EvalViewer(data, runner, args)
    if args.smoke_frames > 0:
        viewer.smoke(args.smoke_frames)
    else:
        viewer.run()


if __name__ == "__main__":
    main()
