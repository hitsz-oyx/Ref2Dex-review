#!/usr/bin/env python3
# Stage 3 contact / correspondence heatmap 可视化工具（Open3D）。
#
# ────────────────────────────────────────────────────────────────────
# 按键说明
# ────────────────────────────────────────────────────────────────────
# 帧切换（逐帧）
#   ← / H   : 上一帧
#   → / L   : 下一帧
#
# 帧切换（10 帧）
#   ↓ / J   : 后退 10 帧
#   ↑ / K   : 前进 10 帧
#
# 模式切换
#   M       : 循环切换 heat map 显示模式
#             （依次在 label_contact / label_corr_valid / label_finger /
#              label_region / pred_obj_contact / pred_cross_contact* /
#              pred_cross_cano* 之间循环）
#   C       : 循环切换对应连线的来源
#             （auto / nn / cross_topk / cross_cano_topk / none）
#
# 退出
#   Q / Esc : 关闭窗口并退出
#
# 注意：↓/J 实际是向"前"帧跳（即 frame_idx -10），↑/K 是向"后"帧跳（+10）。
# ────────────────────────────────────────────────────────────────────
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("DISPLAY", "localhost:10.0")

import open3d as o3d
from scipy.spatial import cKDTree

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PER_SEQUENCE_FIELDS = {
    "point_type_id",
    "finger_id",
    "hand_region_id",
    "hand_cano_points",
    "num_obj_points",
    "num_hand_points",
    "obj_point_start",
    "obj_point_count",
    "hand_point_start",
    "hand_point_count",
}

HEAT_MODES = [
    "label_contact",
    "label_corr_valid",
    "label_finger",
    "label_region",
    "pred_obj_contact",
    "gt_cross_contact",
    "pred_cross_contact",
    "pred_cross_contact_max",
    "pred_cross_contact_mean",
    "gt_cross_cano",
    "pred_cross_cano",
    "pred_cross_cano_error_min",
    "pred_cross_cano_error_mean",
]

LINE_SOURCES = ["auto", "nn", "cross_topk", "cross_cano_topk", "none"]

FINGER_PALETTE = np.asarray(
    [
        [0.70, 0.70, 0.70],
        [0.95, 0.35, 0.25],
        [0.95, 0.65, 0.20],
        [0.20, 0.75, 0.35],
        [0.20, 0.55, 0.95],
        [0.70, 0.35, 0.95],
    ],
    dtype=np.float64,
)

REGION_PALETTE = np.asarray(
    [
        [0.70, 0.70, 0.70],
        [0.95, 0.30, 0.30],
        [0.95, 0.60, 0.20],
        [0.25, 0.80, 0.40],
        [0.20, 0.60, 0.95],
        [0.75, 0.35, 0.95],
    ],
    dtype=np.float64,
)

COLOR_HAND = np.asarray([0.78, 0.78, 0.78], dtype=np.float64)
COLOR_INVALID = np.asarray([0.15, 0.15, 0.15], dtype=np.float64)
COLOR_BG = np.asarray([0.15, 0.15, 0.15], dtype=np.float64)
COLOR_CANO_HAND = np.asarray([0.85, 0.85, 0.85], dtype=np.float64)

GT_PRED_TOGGLE_MODE = {
    "label_contact": "pred_obj_contact",
    "pred_obj_contact": "label_contact",
    "gt_cross_contact": "pred_cross_contact",
    "pred_cross_contact": "gt_cross_contact",
    "pred_cross_contact_max": "gt_cross_contact",
    "pred_cross_contact_mean": "gt_cross_contact",
    "gt_cross_cano": "pred_cross_cano",
    "pred_cross_cano": "gt_cross_cano",
    "pred_cross_cano_error_min": "gt_cross_cano",
    "pred_cross_cano_error_mean": "gt_cross_cano",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage 3 contact / correspondence heat maps in Open3D.")
    parser.add_argument("--input", required=True, help="Stage 3 .npz file path.")
    parser.add_argument("--pred", default=None, help="Optional prediction file (.npz/.pt/.pth/.pkl).")
    parser.add_argument("--frame", type=int, default=0, help="Initial frame index inside the Stage 3 sequence.")
    parser.add_argument("--raw-frame-id", type=int, default=None, help="Use raw_frame_id to select initial frame.")
    parser.add_argument(
        "--mode",
        default="label_contact",
        choices=HEAT_MODES,
        help=(
            "Heat map source. "
            "pred_cross_contact = max over K. "
            "pred_cross_cano = cano error on the best-contact edge."
        ),
    )
    parser.add_argument(
        "--line-source",
        default="auto",
        choices=LINE_SOURCES,
        help="Line source. auto: pred_cross_contact_* -> cross_topk, pred_cross_cano_* -> cross_cano_topk, else nn.",
    )
    parser.add_argument("--topk-lines", type=int, default=96, help="Maximum number of lines to draw.")
    parser.add_argument("--line-threshold", type=float, default=0.10, help="Minimum score for drawing lines.")
    parser.add_argument("--cross-k", type=int, default=None, help="Fallback K for runtime KNN when pred file lacks KNN.")
    parser.add_argument("--point-size", type=float, default=3.0, help="Open3D point size.")
    parser.add_argument("--width", type=int, default=1280, help="Window width.")
    parser.add_argument("--height", type=int, default=720, help="Window height.")
    parser.add_argument("--world-frame", action="store_true", help="Transform object/hand points to world frame.")
    parser.add_argument("--show-axis", action="store_true", help="Show coordinate frame.")
    parser.add_argument("--show-canonical-hand", action="store_true", help="Show canonical hand cloud when visualizing pred_cross_cano.")
    parser.add_argument(
        "--cano-offset",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Optional manual offset for canonical hand view.",
    )
    parser.add_argument("--summary", action="store_true", help="Only print summary without opening window.")
    return parser.parse_args()


def _as_numpy(value: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sigmoid_if_needed(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    if x.min() < 0.0 or x.max() > 1.0:
        return 1.0 / (1.0 + np.exp(-x))
    return x


def soft_contact_label_np(
    dist: np.ndarray,
    d_pos: float = 0.005,
    d_neg: float = 0.03,
    gamma: float = 2.0,
) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    label = np.zeros_like(dist, dtype=np.float64)
    pos_mask = dist <= float(d_pos)
    mid_mask = (dist > float(d_pos)) & (dist < float(d_neg))
    label[pos_mask] = 1.0
    if np.any(mid_mask):
        alpha = 1.0 - (dist[mid_mask] - float(d_pos)) / max(float(d_neg) - float(d_pos), 1e-12)
        label[mid_mask] = np.power(np.clip(alpha, 0.0, 1.0), float(gamma))
    return label


def _flatten_prediction_payload(payload: Any) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    def visit(node: Any, prefix: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                name = f"{prefix}.{key}" if prefix else str(key)
                visit(value, name)
            return
        try:
            arr = _as_numpy(node)
        except Exception:
            return
        if prefix:
            out[prefix] = arr
            out.setdefault(prefix.split(".")[-1], arr)

    visit(payload)
    return out


def load_prediction_file(path: str | None) -> dict[str, np.ndarray] | None:
    if path in {None, ""}:
        return None
    pred_path = Path(path)
    suffix = pred_path.suffix.lower()
    if suffix == ".npz":
        with np.load(pred_path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    if suffix in {".pt", ".pth", ".pkl"}:
        if suffix == ".pkl":
            with pred_path.open("rb") as f:
                payload = pickle.load(f)
        else:
            if torch is None:
                raise ImportError("torch is required to read .pt/.pth prediction files.")
            payload = torch.load(pred_path, map_location="cpu")
        return _flatten_prediction_payload(payload)
    raise ValueError(f"Unsupported prediction file suffix: {pred_path.suffix}")


def _infer_num_frames(data: dict[str, np.ndarray]) -> int:
    for key in ("points", "obj_contact_label", "raw_frame_id"):
        if key in data:
            return int(np.asarray(data[key]).shape[0])
    raise ValueError("Cannot infer number of frames from Stage 3 file.")


def _select_frame_array(data: dict[str, np.ndarray], key: str, frame_idx: int, num_frames: int) -> np.ndarray | None:
    if key not in data:
        return None
    arr = np.asarray(data[key])
    if arr.ndim == 0 or key in PER_SEQUENCE_FIELDS:
        return arr
    if arr.shape[0] == num_frames:
        return arr[frame_idx]
    if arr.shape[0] == 1:
        return arr[0]
    return arr


def _resolve_frame_idx(data: dict[str, np.ndarray], frame: int, raw_frame_id: int | None) -> int:
    num_frames = _infer_num_frames(data)
    if raw_frame_id is None:
        if frame < 0 or frame >= num_frames:
            raise IndexError(f"frame={frame} out of range [0, {num_frames})")
        return int(frame)
    raw = np.asarray(data["raw_frame_id"]).reshape(-1)
    matches = np.where(raw == int(raw_frame_id))[0]
    if matches.size == 0:
        raise KeyError(f"raw_frame_id={raw_frame_id} not found in file.")
    return int(matches[0])


def load_stage3_sequence(path: str) -> tuple[dict[str, np.ndarray], int]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return arrays, _infer_num_frames(arrays)


def sample_stage3_frame(data: dict[str, np.ndarray], frame_idx: int, num_frames: int) -> dict[str, np.ndarray]:
    return {key: _select_frame_array(data, key, frame_idx, num_frames) for key in data}


def _scalar_int(value: np.ndarray | None) -> int | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.ndim == 0:
        return int(arr.item())
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    return None


def infer_point_counts(sample: dict[str, np.ndarray]) -> tuple[int, int]:
    num_obj = _scalar_int(sample.get("num_obj_points"))
    if num_obj is None and sample.get("obj_contact_label") is not None:
        num_obj = int(np.asarray(sample["obj_contact_label"]).shape[0])
    if num_obj is None:
        raise ValueError("Cannot infer num_obj_points.")
    num_hand = _scalar_int(sample.get("num_hand_points"))
    if num_hand is None and sample.get("points") is not None:
        num_hand = int(np.asarray(sample["points"]).shape[0]) - int(num_obj)
    if num_hand is None:
        raise ValueError("Cannot infer num_hand_points.")
    return int(num_obj), int(num_hand)


def transform_points_world(points: np.ndarray, T_world_from_obj: np.ndarray | None) -> np.ndarray:
    if T_world_from_obj is None:
        return points
    T = np.asarray(T_world_from_obj, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def scalar_heat_colors(values: np.ndarray, valid_mask: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    colors = np.tile(COLOR_INVALID[None], (values.shape[0], 1))
    if not np.any(valid_mask):
        return colors

    valid_values = values[valid_mask]
    lo = float(np.min(valid_values) if vmin is None else vmin)
    hi = float(np.max(valid_values) if vmax is None else vmax)
    if hi <= lo:
        norm = np.zeros_like(values, dtype=np.float64)
    else:
        norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    ctrl_x = np.asarray([0.0, 0.33, 0.66, 1.0], dtype=np.float64)
    ctrl_c = np.asarray(
        [
            [0.05, 0.20, 0.90],
            [0.10, 0.85, 0.95],
            [0.98, 0.90, 0.20],
            [0.90, 0.10, 0.10],
        ],
        dtype=np.float64,
    )
    for channel in range(3):
        colors[:, channel] = np.interp(norm, ctrl_x, ctrl_c[:, channel])
    colors[~valid_mask] = COLOR_INVALID
    return colors


def indexed_palette_colors(labels: np.ndarray, valid_mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    colors = np.tile(COLOR_INVALID[None], (labels.shape[0], 1))
    safe = np.clip(labels, 0, palette.shape[0] - 1)
    colors[valid_mask] = palette[safe[valid_mask]]
    colors[~valid_mask] = COLOR_INVALID
    return colors


def resolve_prediction_array(pred_data: dict[str, np.ndarray] | None, key: str, frame_idx: int, num_frames: int) -> np.ndarray | None:
    if pred_data is None or key not in pred_data:
        return None
    arr = np.asarray(pred_data[key])
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == num_frames:
        return arr[frame_idx]
    if arr.shape[0] == 1:
        return arr[0]
    return arr


def compute_runtime_obj_to_hand_knn(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    hand_valid_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    knn_idx = np.full((obj_points.shape[0], k), -1, dtype=np.int64)
    knn_valid_mask = np.zeros((obj_points.shape[0], k), dtype=bool)
    obj_valid_idx = np.flatnonzero(obj_valid_mask).astype(np.int64)
    hand_valid_idx = np.flatnonzero(hand_valid_mask).astype(np.int64)
    if obj_valid_idx.size == 0 or hand_valid_idx.size == 0 or k <= 0:
        return knn_idx, knn_valid_mask
    tree = cKDTree(hand_points[hand_valid_idx])
    query_k = min(int(k), int(hand_valid_idx.size))
    _, nn_local = tree.query(obj_points[obj_valid_idx], k=query_k)
    nn_local = np.asarray(nn_local, dtype=np.int64)
    if nn_local.ndim == 1:
        nn_local = nn_local[:, None]
    knn_idx[obj_valid_idx, :query_k] = hand_valid_idx[nn_local]
    knn_valid_mask[obj_valid_idx, :query_k] = True
    return knn_idx, knn_valid_mask


def choose_canonical_offset(obj_points: np.ndarray, hand_points: np.ndarray) -> np.ndarray:
    if obj_points.size == 0 and hand_points.size == 0:
        return np.asarray([0.30, 0.0, 0.0], dtype=np.float64)
    if obj_points.size == 0:
        all_points = hand_points
    elif hand_points.size == 0:
        all_points = obj_points
    else:
        all_points = np.concatenate([obj_points, hand_points], axis=0)
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    extent = float(np.max(bbox_max - bbox_min))
    return np.asarray([extent * 1.8 + 0.05, 0.0, 0.0], dtype=np.float64)


def _resolve_gt_cross_bundle(
    sample: dict[str, np.ndarray],
    pred_data: dict[str, np.ndarray] | None,
    frame_idx: int,
    num_frames: int,
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    hand_valid_mask: np.ndarray,
    k_override: int | None,
    d_pos: float = 0.005,
    d_neg: float = 0.03,
    gamma: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knn_idx = sample.get("obj_to_hand_knn_idx")
    knn_valid = sample.get("obj_to_hand_knn_valid_mask")
    if knn_idx is None or knn_valid is None:
        if pred_data is not None:
            knn_idx = resolve_prediction_array(pred_data, "obj_to_hand_knn_idx", frame_idx, num_frames)
            knn_valid = resolve_prediction_array(pred_data, "obj_to_hand_knn_valid_mask", frame_idx, num_frames)
    if knn_idx is None or knn_valid is None:
        k_cross = int(k_override or 32)
        knn_idx, knn_valid = compute_runtime_obj_to_hand_knn(
            obj_points=obj_points,
            hand_points=hand_points,
            obj_valid_mask=obj_valid_mask,
            hand_valid_mask=hand_valid_mask,
            k=k_cross,
        )
    else:
        knn_idx = np.asarray(knn_idx, dtype=np.int64)
        knn_valid = np.asarray(knn_valid, dtype=bool)

    safe_idx = np.clip(knn_idx, 0, max(hand_points.shape[0] - 1, 0))
    neighbor_hand = hand_points[safe_idx]
    delta = neighbor_hand - obj_points[:, None, :]
    dist = np.linalg.norm(delta, axis=-1)
    labels = soft_contact_label_np(dist, d_pos=d_pos, d_neg=d_neg, gamma=gamma)
    labels = labels * knn_valid.astype(np.float64)
    return labels, knn_idx, knn_valid


def build_heatmap(
    sample: dict[str, np.ndarray],
    pred_data: dict[str, np.ndarray] | None,
    frame_idx: int,
    num_frames: int,
    mode: str,
    obj_valid_mask: np.ndarray,
    cross_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obj_contact = np.asarray(sample["obj_contact_label"], dtype=np.float64)
    ranking_score = obj_contact.copy()

    if mode == "label_contact":
        heat = obj_contact
        colors = scalar_heat_colors(heat, obj_valid_mask, vmin=0.0, vmax=1.0)
        return heat, colors, ranking_score

    if mode == "label_corr_valid":
        heat = np.asarray(sample["obj_corr_valid_mask"], dtype=np.float64)
        colors = scalar_heat_colors(heat, obj_valid_mask, vmin=0.0, vmax=1.0)
        return heat, colors, ranking_score

    if mode == "label_finger":
        finger = np.asarray(sample["obj_to_hand_finger_id"], dtype=np.int64)
        colors = indexed_palette_colors(finger, obj_valid_mask & (finger >= 0), FINGER_PALETTE)
        return finger.astype(np.float64), colors, ranking_score

    if mode == "label_region":
        region = np.asarray(sample["obj_to_hand_region_id"], dtype=np.int64)
        colors = indexed_palette_colors(region, obj_valid_mask & (region >= 0), REGION_PALETTE)
        return region.astype(np.float64), colors, ranking_score

    if pred_data is None:
        raise ValueError(f"mode={mode} requires --pred.")

    if mode == "pred_obj_contact":
        pred = resolve_prediction_array(pred_data, "pred_obj_contact", frame_idx, num_frames)
        if pred is None:
            raise KeyError("pred_obj_contact not found in prediction file.")
        heat = _sigmoid_if_needed(pred).reshape(-1)
        colors = scalar_heat_colors(heat, obj_valid_mask, vmin=0.0, vmax=1.0)
        return heat, colors, heat

    if mode in {"gt_cross_contact", "gt_cross_cano"}:
        num_obj, num_hand = infer_point_counts(sample)
        points = np.asarray(sample["points"], dtype=np.float64)
        point_valid_mask = np.asarray(sample.get("point_valid_mask", np.ones(points.shape[0], dtype=bool)), dtype=bool)
        obj_points = points[:num_obj]
        hand_points = points[num_obj : num_obj + num_hand]
        hand_valid_mask = point_valid_mask[num_obj : num_obj + num_hand]
        gt_cross, _, _ = _resolve_gt_cross_bundle(
            sample=sample,
            pred_data=pred_data,
            frame_idx=frame_idx,
            num_frames=num_frames,
            obj_points=obj_points,
            hand_points=hand_points,
            obj_valid_mask=obj_valid_mask,
            hand_valid_mask=hand_valid_mask,
            k_override=cross_k,
        )
        heat = gt_cross.max(axis=1)
        colors = scalar_heat_colors(heat, obj_valid_mask, vmin=0.0, vmax=1.0)
        return heat, colors, heat

    if mode in {"pred_cross_contact", "pred_cross_contact_max", "pred_cross_contact_mean"}:
        pred = resolve_prediction_array(pred_data, "pred_cross_contact", frame_idx, num_frames)
        if pred is None:
            raise KeyError("pred_cross_contact not found in prediction file.")
        pred = _sigmoid_if_needed(pred)
        if pred.ndim != 2:
            raise ValueError(f"pred_cross_contact should have shape [No, K], got {pred.shape}")
        if mode == "pred_cross_contact_mean":
            heat = pred.mean(axis=1)
        else:
            heat = pred.max(axis=1)
        colors = scalar_heat_colors(heat, obj_valid_mask, vmin=0.0, vmax=1.0)
        return heat, colors, heat

    if mode in {"pred_cross_cano", "pred_cross_cano_error_min", "pred_cross_cano_error_mean"}:
        pred = resolve_prediction_array(pred_data, "pred_cross_cano", frame_idx, num_frames)
        if pred is None:
            raise KeyError("pred_cross_cano not found in prediction file.")
        target = np.asarray(sample["obj_to_hand_cano_points"], dtype=np.float64)[:, None, :]
        if pred.ndim != 3 or pred.shape[0] != target.shape[0] or pred.shape[-1] != 3:
            raise ValueError(f"pred_cross_cano should have shape [No, K, 3], got {pred.shape}")
        err = np.linalg.norm(np.asarray(pred, dtype=np.float64) - target, axis=-1)
        if mode == "pred_cross_cano_error_mean":
            heat = err.mean(axis=1)
            ranking_score = np.exp(-heat)
        elif mode == "pred_cross_cano_error_min":
            heat = err.min(axis=1)
            ranking_score = np.exp(-heat)
        else:
            pred_cross = resolve_prediction_array(pred_data, "pred_cross_contact", frame_idx, num_frames)
            if pred_cross is None:
                best_edge_idx = err.argmin(axis=1)
                heat = err[np.arange(err.shape[0]), best_edge_idx]
                ranking_score = np.exp(-heat)
            else:
                pred_cross = _sigmoid_if_needed(pred_cross)
                if pred_cross.shape != err.shape:
                    raise ValueError(
                        f"pred_cross_contact shape {pred_cross.shape} mismatches pred_cross_cano shape {pred.shape}"
                    )
                best_edge_idx = pred_cross.argmax(axis=1)
                heat = err[np.arange(err.shape[0]), best_edge_idx]
                ranking_score = pred_cross[np.arange(pred_cross.shape[0]), best_edge_idx]
        colors = scalar_heat_colors(heat, obj_valid_mask)
        return heat, colors, ranking_score

    raise ValueError(f"Unsupported mode: {mode}")


def build_nn_lines(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_nn_id: np.ndarray,
    obj_valid_mask: np.ndarray,
    colors_obj: np.ndarray,
    ranking_score: np.ndarray,
    topk_lines: int,
    line_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = obj_valid_mask & (obj_nn_id >= 0) & (ranking_score >= float(line_threshold))
    candidate_idx = np.flatnonzero(valid)
    if candidate_idx.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3))
    order = np.argsort(-ranking_score[candidate_idx])[: int(topk_lines)]
    chosen = candidate_idx[order]
    line_points = np.concatenate([obj_points[chosen], hand_points[obj_nn_id[chosen]]], axis=0)
    line_edges = np.stack(
        [
            np.arange(chosen.size, dtype=np.int32),
            np.arange(chosen.size, dtype=np.int32) + chosen.size,
        ],
        axis=1,
    )
    return line_points, line_edges, colors_obj[chosen]


def _resolve_cross_prediction_bundle(
    pred_data: dict[str, np.ndarray] | None,
    frame_idx: int,
    num_frames: int,
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    hand_valid_mask: np.ndarray,
    k_override: int | None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if pred_data is None:
        return None, None, None
    pred_cross = resolve_prediction_array(pred_data, "pred_cross_contact", frame_idx, num_frames)
    if pred_cross is None:
        return None, None, None
    pred_cross = _sigmoid_if_needed(pred_cross)
    knn_idx = resolve_prediction_array(pred_data, "obj_to_hand_knn_idx", frame_idx, num_frames)
    knn_valid = resolve_prediction_array(pred_data, "obj_to_hand_knn_valid_mask", frame_idx, num_frames)
    if knn_idx is None or knn_valid is None:
        k_cross = int(k_override or pred_cross.shape[-1])
        knn_idx, knn_valid = compute_runtime_obj_to_hand_knn(
            obj_points=obj_points,
            hand_points=hand_points,
            obj_valid_mask=obj_valid_mask,
            hand_valid_mask=hand_valid_mask,
            k=k_cross,
        )
    else:
        knn_idx = np.asarray(knn_idx, dtype=np.int64)
        knn_valid = np.asarray(knn_valid, dtype=bool)
    if pred_cross.shape != knn_idx.shape:
        raise ValueError(f"pred_cross_contact shape {pred_cross.shape} mismatches KNN shape {knn_idx.shape}")
    return pred_cross, knn_idx, knn_valid


def build_cross_topk_lines(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    hand_valid_mask: np.ndarray,
    pred_data: dict[str, np.ndarray] | None,
    frame_idx: int,
    num_frames: int,
    topk_lines: int,
    line_threshold: float,
    k_override: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_cross, knn_idx, knn_valid = _resolve_cross_prediction_bundle(
        pred_data,
        frame_idx,
        num_frames,
        obj_points,
        hand_points,
        obj_valid_mask,
        hand_valid_mask,
        k_override,
    )
    if pred_cross is None or knn_idx is None or knn_valid is None:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3))

    valid_edges = knn_valid & (pred_cross >= float(line_threshold))
    obj_idx, edge_idx = np.where(valid_edges)
    if obj_idx.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3))

    edge_scores = pred_cross[obj_idx, edge_idx]
    order = np.argsort(-edge_scores)[: int(topk_lines)]
    obj_idx = obj_idx[order]
    edge_idx = edge_idx[order]
    hand_idx = knn_idx[obj_idx, edge_idx]

    line_points = np.concatenate([obj_points[obj_idx], hand_points[hand_idx]], axis=0)
    line_edges = np.stack(
        [
            np.arange(obj_idx.size, dtype=np.int32),
            np.arange(obj_idx.size, dtype=np.int32) + obj_idx.size,
        ],
        axis=1,
    )
    line_colors = scalar_heat_colors(edge_scores[order], np.ones(order.size, dtype=bool), vmin=0.0, vmax=1.0)
    return line_points, line_edges, line_colors


def build_cross_cano_lines(
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    hand_valid_mask: np.ndarray,
    hand_cano_points: np.ndarray,
    pred_data: dict[str, np.ndarray] | None,
    frame_idx: int,
    num_frames: int,
    topk_lines: int,
    line_threshold: float,
    k_override: int | None,
    cano_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_cross, _, knn_valid = _resolve_cross_prediction_bundle(
        pred_data,
        frame_idx,
        num_frames,
        obj_points,
        hand_points,
        obj_valid_mask,
        hand_valid_mask,
        k_override,
    )
    pred_cross_cano = resolve_prediction_array(pred_data, "pred_cross_cano", frame_idx, num_frames) if pred_data is not None else None
    if pred_cross is None or pred_cross_cano is None or knn_valid is None:
        return (
            np.zeros((0, 3)),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3)),
            hand_cano_points + cano_offset[None],
        )
    pred_cross_cano = np.asarray(pred_cross_cano, dtype=np.float64)
    if pred_cross_cano.ndim != 3 or pred_cross_cano.shape[:2] != pred_cross.shape:
        raise ValueError(
            f"pred_cross_cano shape {pred_cross_cano.shape} mismatches pred_cross_contact shape {pred_cross.shape}"
        )

    valid_edges = knn_valid & (pred_cross >= float(line_threshold))
    obj_idx, edge_idx = np.where(valid_edges)
    if obj_idx.size == 0:
        return (
            np.zeros((0, 3)),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3)),
            hand_cano_points + cano_offset[None],
        )

    edge_scores = pred_cross[obj_idx, edge_idx]
    order = np.argsort(-edge_scores)[: int(topk_lines)]
    obj_idx = obj_idx[order]
    edge_idx = edge_idx[order]
    pred_cano = pred_cross_cano[obj_idx, edge_idx] + cano_offset[None]

    line_points = np.concatenate([obj_points[obj_idx], pred_cano], axis=0)
    line_edges = np.stack(
        [
            np.arange(obj_idx.size, dtype=np.int32),
            np.arange(obj_idx.size, dtype=np.int32) + obj_idx.size,
        ],
        axis=1,
    )
    line_colors = scalar_heat_colors(edge_scores[order], np.ones(order.size, dtype=bool), vmin=0.0, vmax=1.0)
    return line_points, line_edges, line_colors, hand_cano_points + cano_offset[None]


def build_gt_cross_topk_lines(
    sample: dict[str, np.ndarray],
    pred_data: dict[str, np.ndarray] | None,
    frame_idx: int,
    num_frames: int,
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    hand_valid_mask: np.ndarray,
    topk_lines: int,
    line_threshold: float,
    k_override: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gt_cross, knn_idx, knn_valid = _resolve_gt_cross_bundle(
        sample=sample,
        pred_data=pred_data,
        frame_idx=frame_idx,
        num_frames=num_frames,
        obj_points=obj_points,
        hand_points=hand_points,
        obj_valid_mask=obj_valid_mask,
        hand_valid_mask=hand_valid_mask,
        k_override=k_override,
    )
    valid_edges = knn_valid & (gt_cross >= float(line_threshold))
    obj_idx, edge_idx = np.where(valid_edges)
    if obj_idx.size == 0:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32), np.zeros((0, 3))

    edge_scores = gt_cross[obj_idx, edge_idx]
    order = np.argsort(-edge_scores)[: int(topk_lines)]
    obj_idx = obj_idx[order]
    edge_idx = edge_idx[order]
    hand_idx = knn_idx[obj_idx, edge_idx]

    line_points = np.concatenate([obj_points[obj_idx], hand_points[hand_idx]], axis=0)
    line_edges = np.stack(
        [
            np.arange(obj_idx.size, dtype=np.int32),
            np.arange(obj_idx.size, dtype=np.int32) + obj_idx.size,
        ],
        axis=1,
    )
    line_colors = scalar_heat_colors(edge_scores[order], np.ones(order.size, dtype=bool), vmin=0.0, vmax=1.0)
    return line_points, line_edges, line_colors


def build_gt_cross_cano_lines(
    sample: dict[str, np.ndarray],
    pred_data: dict[str, np.ndarray] | None,
    frame_idx: int,
    num_frames: int,
    obj_points: np.ndarray,
    hand_points: np.ndarray,
    obj_valid_mask: np.ndarray,
    hand_valid_mask: np.ndarray,
    hand_cano_points: np.ndarray,
    topk_lines: int,
    line_threshold: float,
    k_override: int | None,
    cano_offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_cross, _, _ = _resolve_gt_cross_bundle(
        sample=sample,
        pred_data=pred_data,
        frame_idx=frame_idx,
        num_frames=num_frames,
        obj_points=obj_points,
        hand_points=hand_points,
        obj_valid_mask=obj_valid_mask,
        hand_valid_mask=hand_valid_mask,
        k_override=k_override,
    )
    if "obj_to_hand_cano_points" not in sample:
        return (
            np.zeros((0, 3)),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3)),
            hand_cano_points + cano_offset[None],
        )

    point_scores = gt_cross.max(axis=1)
    valid_points = obj_valid_mask & (point_scores >= float(line_threshold))
    obj_idx = np.flatnonzero(valid_points)
    if obj_idx.size == 0:
        return (
            np.zeros((0, 3)),
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, 3)),
            hand_cano_points + cano_offset[None],
        )

    order = np.argsort(-point_scores[obj_idx])[: int(topk_lines)]
    obj_idx = obj_idx[order]
    gt_cano = np.asarray(sample["obj_to_hand_cano_points"], dtype=np.float64)[obj_idx] + cano_offset[None]

    line_points = np.concatenate([obj_points[obj_idx], gt_cano], axis=0)
    line_edges = np.stack(
        [
            np.arange(obj_idx.size, dtype=np.int32),
            np.arange(obj_idx.size, dtype=np.int32) + obj_idx.size,
        ],
        axis=1,
    )
    line_colors = scalar_heat_colors(point_scores[obj_idx], np.ones(obj_idx.size, dtype=bool), vmin=0.0, vmax=1.0)
    return line_points, line_edges, line_colors, hand_cano_points + cano_offset[None]


def make_point_cloud() -> o3d.geometry.PointCloud:
    return o3d.geometry.PointCloud()


def make_line_set() -> o3d.geometry.LineSet:
    return o3d.geometry.LineSet()


def set_point_cloud_geometry(pcd: o3d.geometry.PointCloud, points: np.ndarray, colors: np.ndarray) -> None:
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))


def set_line_set_geometry(ls: o3d.geometry.LineSet, points: np.ndarray, edges: np.ndarray, colors: np.ndarray) -> None:
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(edges, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))


def print_summary(
    sample: dict[str, np.ndarray],
    frame_idx: int,
    num_frames: int,
    num_obj: int,
    num_hand: int,
    mode: str,
    line_source: str,
    pred_data: dict[str, np.ndarray] | None,
) -> None:
    raw_frame_id = sample.get("raw_frame_id")
    seq_name = sample.get("seq_name")
    subject_id = sample.get("subject_id")
    side = sample.get("side")
    obj_contact = np.asarray(sample["obj_contact_label"], dtype=np.float64)
    obj_valid = np.asarray(sample["obj_label_valid_mask"], dtype=bool)
    print(f"frame_idx={frame_idx}/{num_frames-1}")
    if raw_frame_id is not None:
        print(f"raw_frame_id={int(np.asarray(raw_frame_id).reshape(-1)[0])}")
    if seq_name is not None:
        subject_text = subject_id.item() if hasattr(subject_id, "item") else subject_id
        seq_text = seq_name.item() if hasattr(seq_name, "item") else seq_name
        side_text = side.item() if hasattr(side, "item") else side
        print(f"sequence={subject_text}/{seq_text}_{side_text}")
    print(f"num_obj_points={num_obj} num_hand_points={num_hand}")
    print(f"valid_obj_points={int(obj_valid.sum())}")
    print(
        "obj_contact_label: "
        f"min={obj_contact[obj_valid].min():.4f} "
        f"max={obj_contact[obj_valid].max():.4f} "
        f"mean={obj_contact[obj_valid].mean():.4f}"
    )
    print(f"mode={mode}")
    print(f"line_source={line_source}")
    if pred_data is not None:
        print("prediction_keys=", sorted(pred_data.keys()))


class ContactMapViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stage3_data, self.num_frames = load_stage3_sequence(args.input)
        self.pred_data = load_prediction_file(args.pred)
        self.frame_idx = _resolve_frame_idx(self.stage3_data, args.frame, args.raw_frame_id)
        self.mode_idx = HEAT_MODES.index(args.mode)
        self.line_source_idx = LINE_SOURCES.index(args.line_source)

        first_sample = sample_stage3_frame(self.stage3_data, self.frame_idx, self.num_frames)
        self.num_obj, self.num_hand = infer_point_counts(first_sample)

        self.obj_pcd = make_point_cloud()
        self.hand_pcd = make_point_cloud()
        self.line_set = make_line_set()
        self.cano_pcd = make_point_cloud()
        self.cano_line_set = make_line_set()
        self.axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.initialized_view = False

    @property
    def mode(self) -> str:
        return HEAT_MODES[self.mode_idx]

    @property
    def line_source(self) -> str:
        return LINE_SOURCES[self.line_source_idx]

    def _resolved_line_source(self) -> str:
        if self.line_source != "auto":
            return self.line_source
        if self.mode.startswith("gt_cross_contact"):
            return "gt_cross_topk"
        if self.mode.startswith("gt_cross_cano"):
            return "gt_cross_cano_topk"
        if self.mode.startswith("pred_cross_contact"):
            return "cross_topk"
        if self.mode.startswith("pred_cross_cano"):
            return "cross_cano_topk"
        return "nn"

    def _frame_sample(self) -> dict[str, np.ndarray]:
        return sample_stage3_frame(self.stage3_data, self.frame_idx, self.num_frames)

    def _prepare_frame(self) -> dict[str, Any]:
        sample = self._frame_sample()
        num_obj, num_hand = infer_point_counts(sample)
        points = np.asarray(sample["points"], dtype=np.float64)
        point_valid_mask = np.asarray(sample.get("point_valid_mask", np.ones(points.shape[0], dtype=bool)), dtype=bool)
        obj_points = points[:num_obj]
        hand_points = points[num_obj : num_obj + num_hand]
        obj_valid_mask = point_valid_mask[:num_obj]
        hand_valid_mask = point_valid_mask[num_obj : num_obj + num_hand]

        T_world_from_obj = None if sample.get("T_world_from_obj") is None else np.asarray(sample["T_world_from_obj"])
        if self.args.world_frame:
            obj_points = transform_points_world(obj_points, T_world_from_obj)
            hand_points = transform_points_world(hand_points, T_world_from_obj)

        heat_values, obj_colors, ranking_score = build_heatmap(
            sample=sample,
            pred_data=self.pred_data,
            frame_idx=self.frame_idx,
            num_frames=self.num_frames,
            mode=self.mode,
            obj_valid_mask=obj_valid_mask,
            cross_k=self.args.cross_k,
        )
        hand_colors = np.tile(COLOR_HAND[None], (num_hand, 1))
        hand_colors[~hand_valid_mask] = COLOR_INVALID

        line_source = self._resolved_line_source()
        line_points = np.zeros((0, 3), dtype=np.float64)
        line_edges = np.zeros((0, 2), dtype=np.int32)
        line_colors = np.zeros((0, 3), dtype=np.float64)
        cano_points = np.zeros((0, 3), dtype=np.float64)
        cano_colors = np.zeros((0, 3), dtype=np.float64)
        cano_line_points = np.zeros((0, 3), dtype=np.float64)
        cano_line_edges = np.zeros((0, 2), dtype=np.int32)
        cano_line_colors = np.zeros((0, 3), dtype=np.float64)

        if line_source == "nn" and sample.get("obj_to_hand_nn_id") is not None:
            line_points, line_edges, line_colors = build_nn_lines(
                obj_points=obj_points,
                hand_points=hand_points,
                obj_nn_id=np.asarray(sample["obj_to_hand_nn_id"], dtype=np.int64),
                obj_valid_mask=obj_valid_mask,
                colors_obj=obj_colors,
                ranking_score=ranking_score,
                topk_lines=self.args.topk_lines,
                line_threshold=self.args.line_threshold,
            )
        elif line_source == "cross_topk":
            line_points, line_edges, line_colors = build_cross_topk_lines(
                obj_points=obj_points,
                hand_points=hand_points,
                obj_valid_mask=obj_valid_mask,
                hand_valid_mask=hand_valid_mask,
                pred_data=self.pred_data,
                frame_idx=self.frame_idx,
                num_frames=self.num_frames,
                topk_lines=self.args.topk_lines,
                line_threshold=self.args.line_threshold,
                k_override=self.args.cross_k,
            )
        elif line_source == "gt_cross_topk":
            line_points, line_edges, line_colors = build_gt_cross_topk_lines(
                sample=sample,
                pred_data=self.pred_data,
                frame_idx=self.frame_idx,
                num_frames=self.num_frames,
                obj_points=obj_points,
                hand_points=hand_points,
                obj_valid_mask=obj_valid_mask,
                hand_valid_mask=hand_valid_mask,
                topk_lines=self.args.topk_lines,
                line_threshold=self.args.line_threshold,
                k_override=self.args.cross_k,
            )
        elif line_source == "cross_cano_topk":
            hand_cano_points = np.asarray(sample["hand_cano_points"], dtype=np.float64)[num_obj : num_obj + num_hand]
            obj_points_for_offset = obj_points[obj_valid_mask] if np.any(obj_valid_mask) else obj_points
            hand_points_for_offset = hand_points[hand_valid_mask] if np.any(hand_valid_mask) else hand_points
            cano_offset = (
                np.asarray(self.args.cano_offset, dtype=np.float64)
                if self.args.cano_offset is not None
                else choose_canonical_offset(obj_points_for_offset, hand_points_for_offset)
            )
            cano_line_points, cano_line_edges, cano_line_colors, cano_points = build_cross_cano_lines(
                obj_points=obj_points,
                hand_points=hand_points,
                obj_valid_mask=obj_valid_mask,
                hand_valid_mask=hand_valid_mask,
                hand_cano_points=hand_cano_points,
                pred_data=self.pred_data,
                frame_idx=self.frame_idx,
                num_frames=self.num_frames,
                topk_lines=self.args.topk_lines,
                line_threshold=self.args.line_threshold,
                k_override=self.args.cross_k,
                cano_offset=cano_offset,
            )
            if self.args.show_canonical_hand:
                cano_colors = np.tile(COLOR_CANO_HAND[None], (cano_points.shape[0], 1))
            else:
                cano_points = np.zeros((0, 3), dtype=np.float64)
                cano_colors = np.zeros((0, 3), dtype=np.float64)
        elif line_source == "gt_cross_cano_topk":
            hand_cano_points = np.asarray(sample["hand_cano_points"], dtype=np.float64)[num_obj : num_obj + num_hand]
            obj_points_for_offset = obj_points[obj_valid_mask] if np.any(obj_valid_mask) else obj_points
            hand_points_for_offset = hand_points[hand_valid_mask] if np.any(hand_valid_mask) else hand_points
            cano_offset = (
                np.asarray(self.args.cano_offset, dtype=np.float64)
                if self.args.cano_offset is not None
                else choose_canonical_offset(obj_points_for_offset, hand_points_for_offset)
            )
            cano_line_points, cano_line_edges, cano_line_colors, cano_points = build_gt_cross_cano_lines(
                sample=sample,
                pred_data=self.pred_data,
                frame_idx=self.frame_idx,
                num_frames=self.num_frames,
                obj_points=obj_points,
                hand_points=hand_points,
                obj_valid_mask=obj_valid_mask,
                hand_valid_mask=hand_valid_mask,
                hand_cano_points=hand_cano_points,
                topk_lines=self.args.topk_lines,
                line_threshold=self.args.line_threshold,
                k_override=self.args.cross_k,
                cano_offset=cano_offset,
            )
            if self.args.show_canonical_hand:
                cano_colors = np.tile(COLOR_CANO_HAND[None], (cano_points.shape[0], 1))
            else:
                cano_points = np.zeros((0, 3), dtype=np.float64)
                cano_colors = np.zeros((0, 3), dtype=np.float64)

        return {
            "sample": sample,
            "num_obj": num_obj,
            "num_hand": num_hand,
            "obj_points": obj_points,
            "hand_points": hand_points,
            "obj_valid_mask": obj_valid_mask,
            "hand_valid_mask": hand_valid_mask,
            "obj_colors": obj_colors,
            "hand_colors": hand_colors,
            "heat_values": heat_values,
            "line_points": line_points,
            "line_edges": line_edges,
            "line_colors": line_colors,
            "cano_points": cano_points,
            "cano_colors": cano_colors,
            "cano_line_points": cano_line_points,
            "cano_line_edges": cano_line_edges,
            "cano_line_colors": cano_line_colors,
            "line_source": line_source,
        }

    def _status_text(self, state: dict[str, Any]) -> str:
        sample = state["sample"]
        raw_frame_id = sample.get("raw_frame_id")
        raw_text = f" raw={int(np.asarray(raw_frame_id).reshape(-1)[0])}" if raw_frame_id is not None else ""
        return (
            f"[frame {self.frame_idx}/{self.num_frames-1}{raw_text}] "
            f"mode={self.mode} line_source={state['line_source']} "
            f"valid_obj={int(state['obj_valid_mask'].sum())}"
        )

    def _apply_state_to_geometry(self, state: dict[str, Any]) -> None:
        set_point_cloud_geometry(
            self.obj_pcd,
            state["obj_points"][state["obj_valid_mask"]],
            state["obj_colors"][state["obj_valid_mask"]],
        )
        set_point_cloud_geometry(
            self.hand_pcd,
            state["hand_points"][state["hand_valid_mask"]],
            state["hand_colors"][state["hand_valid_mask"]],
        )
        set_line_set_geometry(self.line_set, state["line_points"], state["line_edges"], state["line_colors"])
        set_point_cloud_geometry(self.cano_pcd, state["cano_points"], state["cano_colors"])
        set_line_set_geometry(
            self.cano_line_set,
            state["cano_line_points"],
            state["cano_line_edges"],
            state["cano_line_colors"],
        )

    def _set_initial_camera(self, state: dict[str, Any]) -> None:
        all_pts = [state["obj_points"][state["obj_valid_mask"]], state["hand_points"][state["hand_valid_mask"]]]
        if state["cano_points"].size > 0:
            all_pts.append(state["cano_points"])
        all_pts = [pts for pts in all_pts if pts.size > 0]
        if not all_pts:
            return

        merged = np.concatenate(all_pts, axis=0)
        bbox_min = merged.min(axis=0)
        bbox_max = merged.max(axis=0)
        center = 0.5 * (bbox_min + bbox_max)
        extent = float(np.max(bbox_max - bbox_min))
        if extent < 1e-6:
            extent = 1.0

        cam_pos = center + np.asarray([1.0, -1.0, 0.6], dtype=np.float64) * extent * 0.9
        front = center - cam_pos
        front_norm = float(np.linalg.norm(front))
        if front_norm < 1e-12:
            return

        view_ctl = self.vis.get_view_control()
        view_ctl.set_lookat(center)
        view_ctl.set_front(front / front_norm)
        view_ctl.set_up([0.0, 0.0, 1.0])
        view_ctl.set_zoom(0.55)

    def _update_frame(self, vis: o3d.visualization.Visualizer | None = None) -> None:
        state = self._prepare_frame()
        self._apply_state_to_geometry(state)

        if vis is not None:
            vis.update_geometry(self.obj_pcd)
            vis.update_geometry(self.hand_pcd)
            vis.update_geometry(self.line_set)
            vis.update_geometry(self.cano_pcd)
            vis.update_geometry(self.cano_line_set)
            vis.poll_events()
            vis.update_renderer()
        print(self._status_text(state))

    def _step_frame(self, delta: int, vis: o3d.visualization.Visualizer) -> bool:
        self.frame_idx = int(np.clip(self.frame_idx + delta, 0, self.num_frames - 1))
        self._update_frame(vis)
        return False

    def _cycle_mode(self, vis: o3d.visualization.Visualizer) -> bool:
        self.mode_idx = (self.mode_idx + 1) % len(HEAT_MODES)
        self._update_frame(vis)
        return False

    def _cycle_line_source(self, vis: o3d.visualization.Visualizer) -> bool:
        self.line_source_idx = (self.line_source_idx + 1) % len(LINE_SOURCES)
        self._update_frame(vis)
        return False

    def _toggle_gt_pred(self, vis: o3d.visualization.Visualizer) -> bool:
        target_mode = GT_PRED_TOGGLE_MODE.get(self.mode)
        if target_mode is None:
            print(f"[toggle_gt_pred] no paired GT/pred mode for {self.mode}")
            return False
        if target_mode.startswith("pred_") and self.pred_data is None:
            print(f"[toggle_gt_pred] mode={target_mode} requires --pred")
            return False
        self.mode_idx = HEAT_MODES.index(target_mode)
        self.line_source_idx = LINE_SOURCES.index("auto")
        self._update_frame(vis)
        return False

    def _print_help(self) -> None:
        print("Keyboard:")
        print("  Left / H : previous frame")
        print("  Right / L: next frame")
        print("  Down / J : -10 frames")
        print("  Up / K   : +10 frames")
        print("  M        : cycle heat mode")
        print("  G        : toggle GT / pred counterpart")
        print("  C        : cycle line source")
        print("  Q / Esc  : quit")

    def _close(self, vis: o3d.visualization.Visualizer) -> bool:
        if hasattr(vis, "close"):
            vis.close()
        else:  # pragma: no cover
            vis.destroy_window()
        return False

    def run(self) -> None:
        print(f"[visualize_contact_map] DISPLAY={os.environ.get('DISPLAY', '')}")
        self._print_help()
        initial_state = self._prepare_frame()
        self._apply_state_to_geometry(initial_state)

        self.vis.create_window(
            window_name="Correspondence Heatmap Viewer",
            width=int(self.args.width),
            height=int(self.args.height),
            left=50,
            top=50,
            visible=True,
        )
        self.vis.add_geometry(self.obj_pcd)
        self.vis.add_geometry(self.hand_pcd)
        self.vis.add_geometry(self.line_set)
        self.vis.add_geometry(self.cano_pcd)
        self.vis.add_geometry(self.cano_line_set)
        if self.args.show_axis:
            self.vis.add_geometry(self.axis)

        render_opt = self.vis.get_render_option()
        render_opt.background_color = COLOR_BG
        render_opt.point_size = float(self.args.point_size)
        render_opt.line_width = 1.5

        self.vis.register_key_callback(262, lambda vis: self._step_frame(+1, vis))  # Right
        self.vis.register_key_callback(263, lambda vis: self._step_frame(-1, vis))  # Left
        self.vis.register_key_callback(ord("L"), lambda vis: self._step_frame(+1, vis))
        self.vis.register_key_callback(ord("H"), lambda vis: self._step_frame(-1, vis))
        self.vis.register_key_callback(265, lambda vis: self._step_frame(+10, vis))  # Up
        self.vis.register_key_callback(264, lambda vis: self._step_frame(-10, vis))  # Down
        self.vis.register_key_callback(ord("K"), lambda vis: self._step_frame(+10, vis))
        self.vis.register_key_callback(ord("J"), lambda vis: self._step_frame(-10, vis))
        self.vis.register_key_callback(ord("M"), self._cycle_mode)
        self.vis.register_key_callback(ord("G"), self._toggle_gt_pred)
        self.vis.register_key_callback(ord("C"), self._cycle_line_source)
        self.vis.register_key_callback(ord("Q"), self._close)
        self.vis.register_key_callback(256, self._close)  # Esc

        self._update_frame(self.vis)
        if not self.initialized_view:
            self._set_initial_camera(initial_state)
            self.initialized_view = True
        self.vis.run()
        self.vis.destroy_window()


def main() -> None:
    args = parse_args()
    viewer = ContactMapViewer(args)
    if args.summary:
        sample = viewer._frame_sample()
        num_obj, num_hand = infer_point_counts(sample)
        print_summary(
            sample=sample,
            frame_idx=viewer.frame_idx,
            num_frames=viewer.num_frames,
            num_obj=num_obj,
            num_hand=num_hand,
            mode=viewer.mode,
            line_source=viewer._resolved_line_source(),
            pred_data=viewer.pred_data,
        )
        return
    viewer.run()


if __name__ == "__main__":
    main()
