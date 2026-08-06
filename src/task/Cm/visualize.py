# =============================================================================
# CmAction 物体点流可视化脚本 — 按键说明
# =============================================================================
# 通用操作
#   A / D  (← / →)    上一对 / 下一对 时序 pair
#   R                   重置相机视角
#
# 显示切换
#   G                   循环切换物体流显示模式: GT -> Pred -> Both
#   L                   开关流线段 (current→future 的连线,稀疏采样)
#   H                   开关"未来手"点云 (hand_future) 的显示
#   S                   切换手部 Slot Assignment 分区与 soft-anchor 球
#   T                   切换单步 / fixed-stride chunk 模式
#   P                   从当前帧开始/清除最多 10 帧的 teacher-forced chunk
#   G                   切换 GT / Pred / Both
#   W                   切换 hand-root / world 坐标系（需要 Stage 4 world pose 字段）
#
# 颜色约定
#   CURRENT_COLOR  深灰   物体当前帧点
#   GT_COLOR      绿色   物体 GT 未来位置
#   PRED_COLOR    红色   物体模型预测的未来位置
#   HAND_CURRENT  暖黄   手部当前帧
#   HAND_FUTURE   青色   手部未来帧 (H 键开关)
# =============================================================================
"""Interactive GT / prediction point-flow viewer for a trained CmAction model.

Keys:
  A / D or ← / →  previous / next temporal pair
  G              cycle GT -> Pred -> Both
  L              toggle flow line segments
  H              toggle future-hand context
  S              toggle argmax Slot Assignment colors and soft anchors
  T              toggle one-step / fixed-stride chunk view
  P              start or clear a teacher-forced fixed-stride chunk from the current pair
  W              toggle hand-root / world coordinates when Stage 4 stores poses
  R              reset camera
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.base import build_runner_from_checkpoint
from src.task.Cm.dataset import Stage4CmDataset, _normal_world_to_hand, _world_to_hand
from src.task.Cm.runner import CmActionRunner


os.environ.setdefault("DISPLAY", "localhost:10.0")

CURRENT_COLOR = np.asarray([0.38, 0.40, 0.45], dtype=np.float64)
HAND_CURRENT_COLOR = np.asarray([0.90, 0.78, 0.46], dtype=np.float64)
HAND_FUTURE_COLOR = np.asarray([0.34, 0.90, 0.90], dtype=np.float64)
GT_COLOR = np.asarray([0.15, 0.85, 0.34], dtype=np.float64)
PRED_COLOR = np.asarray([0.96, 0.24, 0.16], dtype=np.float64)
ROLLOUT_CONTACT_RADIUS_M = 0.05
SLOT_COLORS = np.asarray(
    [
        [0.894, 0.102, 0.110], [0.216, 0.494, 0.722], [0.302, 0.686, 0.290], [0.596, 0.306, 0.639],
        [1.000, 0.498, 0.000], [1.000, 1.000, 0.200], [0.651, 0.337, 0.157], [0.969, 0.506, 0.749],
        [0.600, 0.600, 0.600], [0.122, 0.471, 0.706], [0.173, 0.627, 0.173], [0.839, 0.153, 0.157],
        [0.580, 0.404, 0.741], [0.549, 0.337, 0.294], [0.890, 0.467, 0.761], [0.498, 0.498, 0.498],
    ],
    dtype=np.float64,
)


@dataclass
class ViewerState:
    pair_idx: int = 0
    mode: str = "both"
    show_lines: bool = True
    show_future_hand: bool = True
    show_slot_assignment: bool = False
    trajectory_mode: bool = False
    show_world_coordinates: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CmAction GT and predicted object point flow.")
    parser.add_argument("--checkpoint", required=True, help="CmAction BaseRunner checkpoint (.pt or run directory).")
    parser.add_argument("--input", required=True, help="One Stage 4 sequence NPZ.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pair", type=int, default=0, help="Pair index after optional active-pair filtering.")
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include pairs without a 5cm object candidate; they render no object flow slots.",
    )
    parser.add_argument("--flow-stride", type=int, default=8, help="Draw one line every N selected object points.")
    parser.add_argument("--stride", type=int, default=1, help="Fixed endpoint stride for inspection (default: 1).")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--assignment-report",
        action="store_true",
        help=(
            "Run inference on every selected pair and report whether the same MANO points "
            "change their argmax slot over time; does not create a window."
        ),
    )
    return parser.parse_args()


def _ensure_cm_runner(runner: Any) -> CmActionRunner:
    if not isinstance(runner, CmActionRunner):
        raise ValueError("Cm visualize.py only supports CmAction BaseRunner checkpoints.")
    return runner


def _prepare_single_batch(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in sample.items() if torch.is_tensor(value)}


def _predict(runner: CmActionRunner, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = runner.inference(runner.model, _prepare_single_batch(sample))
    return {key: value.detach().cpu() for key, value in output.items()}


def _mode_cycle(mode: str) -> str:
    order = ("gt", "pred", "both")
    return order[(order.index(mode) + 1) % len(order)]


def _empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def _slot_colors(num_slots: int) -> np.ndarray:
    if num_slots > len(SLOT_COLORS):
        raise ValueError(f"Viewer has {len(SLOT_COLORS)} slot colors, got {num_slots} slots.")
    return SLOT_COLORS[:num_slots]


def _flow_lines(start: np.ndarray, end: np.ndarray, *, stride: int, color: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(start) == 0:
        return _empty_points(), np.empty((0, 2), dtype=np.int32), _empty_points()
    ids = np.arange(0, len(start), max(1, int(stride)), dtype=np.int32)
    points = np.concatenate([start[ids], end[ids]], axis=0).astype(np.float64)
    lines = np.stack([np.arange(len(ids)), np.arange(len(ids)) + len(ids)], axis=1).astype(np.int32)
    colors = np.broadcast_to(color[None, :], (len(lines), 3)).copy()
    return points, lines, colors


def _assignment_report(runner: CmActionRunner, dataset: Stage4CmDataset) -> dict[str, Any]:
    """Summarize slot reassignment for one temporal sequence.

    MANO point ordering is fixed within a Stage 4 sequence, so adjacent samples
    compare the same surface points directly.  With the default active-pair
    filter, an "adjacent" transition means adjacent active pairs; raw pair
    indices are reported so that skipped inactive intervals remain explicit.
    """
    assignments: list[np.ndarray] = []
    decoder_usages: list[np.ndarray] = []
    raw_frame_ids: list[int] = []
    for dataset_idx in range(len(dataset)):
        sample = dataset[dataset_idx]
        prediction = _predict(runner, sample)
        assignments.append(prediction["cm_assignment"].squeeze(0).numpy().argmax(axis=0))
        decoder_usages.append(prediction["decoder_slot_usage"].squeeze(0).numpy())
        raw_frame_ids.append(int(sample["raw_frame_id"]))

    assignment_array = np.stack(assignments, axis=0)
    unique_slot_count = np.asarray(
        [np.unique(assignment_array[:, point_idx]).size for point_idx in range(assignment_array.shape[1])]
    )
    if len(assignment_array) > 1:
        adjacent_change_rate = float((assignment_array[1:] != assignment_array[:-1]).mean())
        reassigned_point_fraction = float((assignment_array != assignment_array[:1]).any(axis=0).mean())
    else:
        adjacent_change_rate = 0.0
        reassigned_point_fraction = 0.0
    mean_decoder_usage = np.stack(decoder_usages, axis=0).mean(axis=0)
    decoder_usage_entropy = float(
        -(np.clip(mean_decoder_usage, 1e-8, None) * np.log(np.clip(mean_decoder_usage, 1e-8, None))).sum()
    )
    return {
        "num_samples": len(dataset),
        "raw_frame_id_range": (raw_frame_ids[0], raw_frame_ids[-1]),
        "adjacent_assignment_change_rate": adjacent_change_rate,
        "reassigned_hand_point_fraction": reassigned_point_fraction,
        "mean_unique_slots_per_hand_point": float(unique_slot_count.mean()),
        "max_unique_slots_for_one_hand_point": int(unique_slot_count.max()),
        "mean_decoder_slot_usage": mean_decoder_usage.tolist(),
        "decoder_slot_usage_entropy": decoder_usage_entropy,
        "decoder_slot_usage_max": float(mean_decoder_usage.max()),
    }


@dataclass
class RolloutStep:
    """One teacher-forced fixed-stride chunk in its native hand-root frame."""

    dataset_idx: int
    obj_points: np.ndarray
    obj_normals: np.ndarray
    obj_valid_mask: np.ndarray
    gt_obj_points: np.ndarray
    gt_next_obj_points: np.ndarray
    pred_next_obj_points: np.ndarray
    prediction: dict[str, torch.Tensor]


def _transform_to_next_hand_frame(
    points: np.ndarray,
    normals: np.ndarray,
    current_pose_world: np.ndarray,
    next_pose_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map geometry between hand frames using cached world hand-root poses."""
    current = np.asarray(current_pose_world, dtype=np.float64)
    future = np.asarray(next_pose_world, dtype=np.float64)
    world_points = np.asarray(points, dtype=np.float64) @ current[:3, :3].T + current[:3, 3]
    next_points = (world_points - future[:3, 3]) @ future[:3, :3]
    world_normals = np.asarray(normals, dtype=np.float64) @ current[:3, :3].T
    next_normals = world_normals @ future[:3, :3]
    next_normals /= np.clip(np.linalg.norm(next_normals, axis=-1, keepdims=True), 1e-8, None)
    return next_points, next_normals


def _rollout_valid_mask(
    points: np.ndarray,
    hand_points: np.ndarray,
    seed_valid_mask: np.ndarray,
    *,
    radius: float = ROLLOUT_CONTACT_RADIUS_M,
) -> np.ndarray:
    """Recompute the current candidate mask from predicted object geometry only."""
    seed_valid = np.asarray(seed_valid_mask, dtype=bool)
    valid = np.zeros_like(seed_valid)
    if not seed_valid.any():
        return valid
    candidate_points = np.asarray(points, dtype=np.float64)[seed_valid]
    hand = np.asarray(hand_points, dtype=np.float64)
    # At most 512x1538 distances; keeping this local avoids using GT object
    # candidate masks during autoregressive rollout.
    squared_distance = ((candidate_points[:, None, :] - hand[None, :, :]) ** 2).sum(axis=-1)
    valid[seed_valid] = squared_distance.min(axis=1) <= float(radius) ** 2
    return valid


class InteractiveFlowViewer:
    def __init__(
        self,
        *,
        runner: CmActionRunner,
        dataset: Stage4CmDataset,
        state: ViewerState,
        flow_stride: int,
    ) -> None:
        self.runner = runner
        self.dataset = dataset
        self.state = state
        self.flow_stride = max(1, int(flow_stride))
        self.sample: dict[str, torch.Tensor] | None = None
        self.prediction: dict[str, torch.Tensor] | None = None
        self.vis = None
        self.current_obj = None
        self.gt_obj = None
        self.pred_obj = None
        self.hand_current = None
        self.hand_future = None
        self.gt_lines = None
        self.pred_lines = None
        self.slot_anchors: list[Any] = []
        self.slot_anchor_base_vertices: list[np.ndarray] = []
        self.rollout_steps: dict[int, RolloutStep] = {}
        self.rollout_order: list[int] = []
        self.rollout_start_idx: int | None = None
        self.rollout_seed_valid_mask: np.ndarray | None = None
        self.rollout_stop_reason = ""

    def _refresh_cache(self) -> None:
        self.sample = self.dataset[self.state.pair_idx]
        rollout_step = self.rollout_steps.get(self.state.pair_idx)
        self.prediction = rollout_step.prediction if rollout_step is not None else _predict(self.runner, self.sample)

    def _current_hand_to_world_pose(self) -> np.ndarray:
        """Return ``T_world<-hand`` for the displayed current temporal pair."""
        path, raw_pair_idx = self.dataset.sample_location(self.state.pair_idx)
        data = self.dataset._load_file(path)
        if "hand_root_pose_world" not in data:
            raise ValueError(
                "This Stage 4 file has no hand_root_pose_world field. Regenerate it with "
                "process/GRAB/stage4_cm.py schema 1.1+ to enable world coordinates."
            )
        pose = np.asarray(data["hand_root_pose_world"][raw_pair_idx], dtype=np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Expected hand_root_pose_world [4, 4], got {pose.shape}.")
        return pose

    def _to_display_coordinates(self, points: np.ndarray) -> np.ndarray:
        """Transform current hand-root-frame geometry only when world view is active."""
        points = np.asarray(points, dtype=np.float64)
        if not self.state.show_world_coordinates or len(points) == 0:
            return points
        pose = self._current_hand_to_world_pose()
        return points @ pose[:3, :3].T + pose[:3, 3][None, :]

    def _fixed_object_fields(
        self,
        dataset_idx: int,
        selected_obj_idx: np.ndarray,
        seed_valid_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch one fixed source-point set and its GT one-step endpoint.

        Stage 4 preserves a stable 4096-point object ordering across its
        sequence.  The rollout therefore follows exactly the start-frame
        source points rather than re-sampling GT candidate points at each
        later frame.
        """
        path, raw_pair_idx = self.dataset.sample_location(dataset_idx)
        data = self.dataset._load_file(path)
        safe_idx = np.maximum(np.asarray(selected_obj_idx, dtype=np.int64), 0)
        valid = np.asarray(seed_valid_mask, dtype=bool)
        sample = self.dataset[dataset_idx]
        future_idx = raw_pair_idx + int(sample["stride"])
        pose = np.asarray(data["hand_root_pose_world"][raw_pair_idx], dtype=np.float32)
        current = _world_to_hand(data["obj_points_world"][raw_pair_idx, safe_idx], pose).astype(np.float64)
        normals = _normal_world_to_hand(data["obj_normals_world"][raw_pair_idx, safe_idx], pose).astype(np.float64)
        endpoint = _world_to_hand(data["obj_points_world"][future_idx, safe_idx], pose).astype(np.float64)
        current[~valid] = 0.0
        normals[~valid] = 0.0
        endpoint[~valid] = 0.0
        return current, normals, endpoint

    @staticmethod
    def _rollout_input(
        sample: dict[str, torch.Tensor],
        obj_points: np.ndarray,
        obj_normals: np.ndarray,
        obj_valid_mask: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        prepared = dict(sample)
        prepared["obj_points"] = torch.from_numpy(np.asarray(obj_points, dtype=np.float32).copy())
        prepared["obj_normals"] = torch.from_numpy(np.asarray(obj_normals, dtype=np.float32).copy())
        prepared["obj_valid_mask"] = torch.from_numpy(np.asarray(obj_valid_mask, dtype=bool).copy())
        return prepared

    def _start_rollout(self) -> None:
        """Build complete fixed-stride chunks from GT inputs, up to a 10-frame span.

        Every chunk is inferred from its own cached current object state.  This
        deliberately avoids feeding a previous prediction into the next chunk:
        D/A therefore inspect a sequence of independent, teacher-forced chunks
        rather than an autoregressive rollout.  A partial final chunk is never
        created when the stride does not divide 10.
        """
        start_idx = self.state.pair_idx
        start_sample = self.dataset[start_idx]
        fixed_stride = int(start_sample["stride"])
        if fixed_stride <= 0:
            raise ValueError(f"Expected a positive stride, got {fixed_stride}.")
        # Keep every displayed chunk at the stride selected when P was pressed.
        # Without this, a dataset created without --stride would sample a new
        # deterministic-but-different horizon for each later raw frame.
        self.dataset.fixed_stride = fixed_stride
        start_sample = self.dataset[start_idx]
        selected_obj_idx = start_sample["selected_obj_idx"].numpy().astype(np.int64)
        seed_valid_mask = start_sample["obj_valid_mask"].numpy().astype(bool)
        if not seed_valid_mask.any():
            raise ValueError("The current pair has no valid object points; choose an active pair before chunk view.")

        raw_to_dataset_idx = {
            int(self.dataset[dataset_idx]["raw_frame_id"]): dataset_idx
            for dataset_idx in range(len(self.dataset))
        }
        steps: dict[int, RolloutStep] = {}
        order: list[int] = []
        start_raw = int(start_sample["raw_frame_id"])
        max_raw = start_raw + 10
        current_idx = start_idx
        stop_reason = "reached the 10-frame fixed-stride chunk limit"

        while True:
            current_raw = int(self.dataset[current_idx]["raw_frame_id"])
            target_raw = int(self.dataset[current_idx]["next_raw_frame_id"])
            raw_step = target_raw - current_raw
            if target_raw > max_raw:
                break
            current_sample = self.dataset[current_idx]
            # Use the same 4096-pool identities selected at P time for every
            # chunk.  Their positions/normals come from this chunk's actual GT
            # current frame; only the visualization sampling is fixed.
            current_points, current_normals, gt_next_points = self._fixed_object_fields(
                current_idx, selected_obj_idx, seed_valid_mask
            )
            current_valid_mask = seed_valid_mask.copy()
            prediction = _predict(
                self.runner,
                self._rollout_input(current_sample, current_points, current_normals, current_valid_mask),
            )
            pred_flow = prediction["pred_obj_flow"].squeeze(0).numpy().astype(np.float64)
            pred_next_points = current_points + pred_flow
            steps[current_idx] = RolloutStep(
                dataset_idx=current_idx,
                obj_points=current_points.copy(),
                obj_normals=current_normals.copy(),
                obj_valid_mask=current_valid_mask.copy(),
                gt_obj_points=current_points.copy(),
                gt_next_obj_points=gt_next_points,
                pred_next_obj_points=pred_next_points,
                prediction=prediction,
            )
            order.append(current_idx)

            # next_raw_frame_id / raw_frame_id live in source-frame units
            # (raw_frame_id steps equal ds_rate); fixed_stride is in NPZ-index
            # units, so chain bookkeeping must use raw_frame_id-space values.
            if target_raw + raw_step > max_raw:
                stop_reason = "next chunk would exceed the 10-frame raw-frame window"
                break
            next_idx = raw_to_dataset_idx.get(target_raw)
            if next_idx is None:
                stop_reason = (
                    f"no selected pair starts at raw frame {target_raw}; "
                    "cannot continue the fixed-stride chunk sequence"
                )
                break
            if next_idx <= current_idx:
                stop_reason = f"invalid non-forward temporal chain at raw frame {target_raw}"
                break
            current_idx = next_idx

        self.rollout_steps = steps
        self.rollout_order = order
        self.rollout_start_idx = start_idx
        self.rollout_seed_valid_mask = seed_valid_mask
        self.rollout_stop_reason = stop_reason
        end_raw = int(self.dataset[order[-1]]["raw_frame_id"])
        print(
            f"fixed-stride chunks: start_raw={start_raw} end_raw={end_raw} stride={fixed_stride} "
            f"chunks={len(order)}; {stop_reason}"
        )

    def _clear_rollout(self) -> None:
        self.rollout_steps = {}
        self.rollout_order = []
        self.rollout_start_idx = None
        self.rollout_seed_valid_mask = None
        self.rollout_stop_reason = ""

    def _scene_arrays(self) -> dict[str, np.ndarray]:
        assert self.sample is not None and self.prediction is not None
        rollout_step = self.rollout_steps.get(self.state.pair_idx)
        if self.state.trajectory_mode and rollout_step is not None:
            return self._chunk_scene_arrays(rollout_step)
        valid = self.sample["obj_valid_mask"].numpy().astype(bool)
        current = self.sample["obj_points"].numpy()[valid].astype(np.float64)
        gt_future = current + self.sample["obj_flow_gt"].numpy()[valid].astype(np.float64)
        pred_future = current + self.prediction["pred_obj_flow"].squeeze(0).numpy()[valid].astype(np.float64)
        hand_current = self.sample["hand_points"].numpy().astype(np.float64)
        hand_future = hand_current + self.sample["hand_flow"].numpy().astype(np.float64)
        cm_assignment = self.prediction["cm_assignment"].squeeze(0).numpy()
        slot_ids = cm_assignment.argmax(axis=0)
        palette = _slot_colors(cm_assignment.shape[0])
        return {
            "current": current,
            "gt_future": gt_future,
            "pred_future": pred_future,
            "hand_current": hand_current,
            "hand_future": hand_future,
            "hand_slot_colors": palette[slot_ids],
            "slot_colors": palette,
            "slot_anchor_pos": self.prediction["cm_anchor_pos"].squeeze(0).numpy().astype(np.float64),
            "decoder_slot_usage": self.prediction["decoder_slot_usage"].squeeze(0).numpy(),
        }

    def _chunk_scene_arrays(self, chunk_step: RolloutStep) -> dict[str, np.ndarray]:
        """Render a fixed-identity, teacher-forced chunk in the normal view."""
        assert self.sample is not None and self.prediction is not None
        valid = chunk_step.obj_valid_mask
        hand_current = self.sample["hand_points"].numpy().astype(np.float64)
        hand_future = hand_current + self.sample["hand_flow"].numpy().astype(np.float64)
        cm_assignment = self.prediction["cm_assignment"].squeeze(0).numpy()
        slot_ids = cm_assignment.argmax(axis=0)
        palette = _slot_colors(cm_assignment.shape[0])
        return {
            "current": chunk_step.obj_points[valid],
            "gt_future": chunk_step.gt_next_obj_points[valid],
            "pred_future": chunk_step.pred_next_obj_points[valid],
            "hand_current": hand_current,
            "hand_future": hand_future,
            "hand_slot_colors": palette[slot_ids],
            "slot_colors": palette,
            "slot_anchor_pos": self.prediction["cm_anchor_pos"].squeeze(0).numpy().astype(np.float64),
            "decoder_slot_usage": self.prediction["decoder_slot_usage"].squeeze(0).numpy(),
        }

    def _trajectory_gt_scene_arrays(self) -> dict[str, np.ndarray]:
        """Trajectory-mode baseline: the actual GT state at the selected time."""
        assert self.sample is not None and self.prediction is not None
        valid = self.sample["obj_valid_mask"].numpy().astype(bool)
        current = self.sample["obj_points"].numpy()[valid].astype(np.float64)
        gt_next = current + self.sample["obj_flow_gt"].numpy()[valid].astype(np.float64)
        hand_current = self.sample["hand_points"].numpy().astype(np.float64)
        hand_future = hand_current + self.sample["hand_flow"].numpy().astype(np.float64)
        cm_assignment = self.prediction["cm_assignment"].squeeze(0).numpy()
        slot_ids = cm_assignment.argmax(axis=0)
        palette = _slot_colors(cm_assignment.shape[0])
        return {
            "trajectory": np.asarray(True),
            "trajectory_gt_baseline": np.asarray(True),
            "gt_current": current,
            "eval_current": _empty_points(),
            "gt_next": gt_next,
            "eval_next": _empty_points(),
            "hand_current": hand_current,
            "hand_future": hand_future,
            "hand_slot_colors": palette[slot_ids],
            "slot_colors": palette,
            "slot_anchor_pos": self.prediction["cm_anchor_pos"].squeeze(0).numpy().astype(np.float64),
            "decoder_slot_usage": self.prediction["decoder_slot_usage"].squeeze(0).numpy(),
            "rollout_valid_count": np.asarray(int(valid.sum())),
        }

    def _rollout_scene_arrays(self, rollout_step: RolloutStep) -> dict[str, np.ndarray]:
        assert self.sample is not None and self.prediction is not None
        assert self.rollout_seed_valid_mask is not None
        visible = self.rollout_seed_valid_mask
        hand_current = self.sample["hand_points"].numpy().astype(np.float64)
        hand_future = hand_current + self.sample["hand_flow"].numpy().astype(np.float64)
        cm_assignment = self.prediction["cm_assignment"].squeeze(0).numpy()
        slot_ids = cm_assignment.argmax(axis=0)
        palette = _slot_colors(cm_assignment.shape[0])
        return {
            "trajectory": np.asarray(True),
            "gt_current": rollout_step.gt_obj_points[visible],
            "eval_current": rollout_step.obj_points[visible],
            "gt_next": rollout_step.gt_next_obj_points[visible],
            "eval_next": rollout_step.pred_next_obj_points[visible],
            "hand_current": hand_current,
            "hand_future": hand_future,
            "hand_slot_colors": palette[slot_ids],
            "slot_colors": palette,
            "slot_anchor_pos": self.prediction["cm_anchor_pos"].squeeze(0).numpy().astype(np.float64),
            "decoder_slot_usage": self.prediction["decoder_slot_usage"].squeeze(0).numpy(),
            "rollout_valid_count": np.asarray(int(rollout_step.obj_valid_mask.sum())),
        }

    @staticmethod
    def _set_point_cloud(point_cloud: Any, points: np.ndarray, color: np.ndarray) -> None:
        import open3d as o3d

        point_cloud.points = o3d.utility.Vector3dVector(points)
        color = np.asarray(color, dtype=np.float64)
        if len(points) == 0:
            colors = _empty_points()
        elif color.ndim == 1:
            colors = np.broadcast_to(color[None, :], (len(points), 3)).copy()
        elif color.shape == (len(points), 3):
            colors = color
        else:
            raise ValueError(f"Expected one RGB color or [{len(points)}, 3], got {tuple(color.shape)}")
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    @staticmethod
    def _set_line_set(line_set: Any, points: np.ndarray, lines: np.ndarray, colors: np.ndarray) -> None:
        import open3d as o3d

        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

    @staticmethod
    def _set_slot_anchor(
        sphere: Any,
        base_vertices: np.ndarray,
        position: np.ndarray,
        color: np.ndarray,
        *,
        visible: bool,
    ) -> None:
        import open3d as o3d

        if visible:
            sphere.vertices = o3d.utility.Vector3dVector(base_vertices + position[None, :])
            sphere.vertex_colors = o3d.utility.Vector3dVector(
                np.broadcast_to(color[None, :], base_vertices.shape).copy()
            )
        else:
            sphere.vertices = o3d.utility.Vector3dVector(_empty_points())
            sphere.vertex_colors = o3d.utility.Vector3dVector(_empty_points())

    def _update_geometry(self, *, reset_view: bool = False) -> None:
        arrays = self._scene_arrays()
        show_gt = self.state.mode in {"gt", "both"}
        show_pred = self.state.mode in {"pred", "both"}
        is_trajectory = bool(arrays.get("trajectory", False))
        if is_trajectory:
            self._set_point_cloud(self.current_obj, _empty_points(), CURRENT_COLOR)
            self._set_point_cloud(
                self.gt_obj, self._to_display_coordinates(arrays["gt_current"]) if show_gt else _empty_points(), GT_COLOR
            )
            self._set_point_cloud(
                self.pred_obj, self._to_display_coordinates(arrays["eval_current"]) if show_pred else _empty_points(), PRED_COLOR
            )
        else:
            self._set_point_cloud(self.current_obj, self._to_display_coordinates(arrays["current"]), CURRENT_COLOR)
            self._set_point_cloud(
                self.gt_obj, self._to_display_coordinates(arrays["gt_future"]) if show_gt else _empty_points(), GT_COLOR
            )
            self._set_point_cloud(
                self.pred_obj, self._to_display_coordinates(arrays["pred_future"]) if show_pred else _empty_points(), PRED_COLOR
            )
        hand_color = arrays["hand_slot_colors"] if self.state.show_slot_assignment else HAND_CURRENT_COLOR
        self._set_point_cloud(self.hand_current, self._to_display_coordinates(arrays["hand_current"]), hand_color)
        self._set_point_cloud(
            self.hand_future,
            self._to_display_coordinates(arrays["hand_future"]) if self.state.show_future_hand else _empty_points(),
            HAND_FUTURE_COLOR,
        )
        if is_trajectory:
            gt_line_start, gt_line_end = arrays["gt_current"], arrays["gt_next"]
            pred_line_start, pred_line_end = arrays["eval_current"], arrays["eval_next"]
        else:
            gt_line_start, gt_line_end = arrays["current"], arrays["gt_future"]
            pred_line_start, pred_line_end = arrays["current"], arrays["pred_future"]
        if self.state.show_lines and show_gt:
            points, lines, colors = _flow_lines(
                self._to_display_coordinates(gt_line_start),
                self._to_display_coordinates(gt_line_end),
                stride=self.flow_stride,
                color=GT_COLOR,
            )
        else:
            points, lines, colors = _flow_lines(_empty_points(), _empty_points(), stride=1, color=GT_COLOR)
        self._set_line_set(self.gt_lines, points, lines, colors)
        if self.state.show_lines and show_pred:
            points, lines, colors = _flow_lines(
                self._to_display_coordinates(pred_line_start),
                self._to_display_coordinates(pred_line_end),
                stride=self.flow_stride,
                color=PRED_COLOR,
            )
        else:
            points, lines, colors = _flow_lines(_empty_points(), _empty_points(), stride=1, color=PRED_COLOR)
        self._set_line_set(self.pred_lines, points, lines, colors)
        for slot_idx, sphere in enumerate(self.slot_anchors):
            self._set_slot_anchor(
                sphere,
                self.slot_anchor_base_vertices[slot_idx],
                self._to_display_coordinates(arrays["slot_anchor_pos"][slot_idx : slot_idx + 1])[0],
                arrays["slot_colors"][slot_idx],
                visible=self.state.show_slot_assignment,
            )
        for geometry in (
            self.current_obj,
            self.gt_obj,
            self.pred_obj,
            self.hand_current,
            self.hand_future,
            self.gt_lines,
            self.pred_lines,
            *self.slot_anchors,
        ):
            self.vis.update_geometry(geometry)
        if reset_view:
            self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()
        self._print_status(arrays)

    def _print_status(self, arrays: dict[str, np.ndarray]) -> None:
        assert self.sample is not None
        is_trajectory = bool(arrays.get("trajectory", False))
        raw_frame = int(self.sample["raw_frame_id"].item())
        next_frame = int(self.sample["next_raw_frame_id"].item())
        if self.state.trajectory_mode and self.rollout_start_idx is None:
            print(
                f"chunk=ready raw={raw_frame}->{next_frame} valid_obj={len(arrays['current'])} "
                "press P to start fixed-stride teacher-forced chunks"
            )
            return
        flow_error = arrays["pred_future"] - arrays["gt_future"]
        mse = float(np.mean(flow_error**2)) if len(flow_error) else float("nan")
        mae = float(np.mean(np.abs(flow_error))) if len(flow_error) else float("nan")
        decoder_usage = arrays["decoder_slot_usage"]
        top_slot = int(np.argmax(decoder_usage))
        display_name = {"gt": "GT", "pred": "Pred", "both": "Both"}[self.state.mode]
        static_display_name = {"gt": "GT", "pred": "Pred", "both": "Both"}[self.state.mode]
        if self.state.trajectory_mode and self.rollout_order:
            trajectory_step = self.rollout_order.index(self.state.pair_idx)
            start_raw = int(self.dataset[self.rollout_start_idx]["raw_frame_id"])
            print(
                f"chunk={trajectory_step}/{len(self.rollout_order) - 1} start_raw={start_raw} "
                f"raw={raw_frame}->{next_frame} show={display_name} "
                f"flow_mse={mse:.8f} flow_mae={mae:.8f} "
                f"lines={'on' if self.state.show_lines else 'off'} "
                f"future_hand={'on' if self.state.show_future_hand else 'off'} "
                f"slots={'on' if self.state.show_slot_assignment else 'off'} "
                f"frame={'world' if self.state.show_world_coordinates else 'hand'} "
                f"decoder_top=slot_{top_slot:02d}:{float(decoder_usage[top_slot]):.3f}"
            )
            return
        print(
            f"pair={self.state.pair_idx}/{len(self.dataset) - 1} raw={raw_frame}->{next_frame} "
            f"mode={static_display_name} "
            f"valid_obj={len(arrays['current'])} "
            f"flow_mse={mse:.8f} flow_mae={mae:.8f} "
            f"lines={'on' if self.state.show_lines else 'off'} "
            f"future_hand={'on' if self.state.show_future_hand else 'off'} "
            f"slots={'on' if self.state.show_slot_assignment else 'off'} "
            f"frame={'world' if self.state.show_world_coordinates else 'hand'} "
            f"decoder_top=slot_{top_slot:02d}:{float(decoder_usage[top_slot]):.3f}"
        )

    def _refresh(self, *, reset_view: bool = False) -> bool:
        self._refresh_cache()
        self._update_geometry(reset_view=reset_view)
        return False

    def _change_pair(self, delta: int):
        def callback(_vis):
            if self.state.trajectory_mode and self.rollout_order:
                current_pos = self.rollout_order.index(self.state.pair_idx)
                next_pos = int(np.clip(current_pos + delta, 0, len(self.rollout_order) - 1))
                self.state.pair_idx = self.rollout_order[next_pos]
            else:
                self.state.pair_idx = int(np.clip(self.state.pair_idx + delta, 0, len(self.dataset) - 1))
            return self._refresh()

        return callback

    def _cycle_mode(self, _vis):
        self.state.mode = _mode_cycle(self.state.mode)
        return self._refresh()

    def _toggle_lines(self, _vis):
        self.state.show_lines = not self.state.show_lines
        return self._refresh()

    def _toggle_future_hand(self, _vis):
        self.state.show_future_hand = not self.state.show_future_hand
        return self._refresh()

    def _toggle_slot_assignment(self, _vis):
        self.state.show_slot_assignment = not self.state.show_slot_assignment
        return self._refresh()

    def _toggle_trajectory_mode(self, _vis):
        self.state.trajectory_mode = not self.state.trajectory_mode
        if not self.state.trajectory_mode:
            self._clear_rollout()
            self.state.mode = "both"
            print("trajectory mode: off (teacher-forced one-step view)")
        else:
            self.state.mode = "gt"
            print("chunk mode: ready; press P to start fixed-stride teacher-forced chunks")
        return self._refresh()

    def _toggle_rollout(self, _vis):
        if not self.state.trajectory_mode:
            self.state.trajectory_mode = True
        if self.rollout_order:
            self._clear_rollout()
            self.state.mode = "gt"
            print("chunk view: cleared; now showing the current one-step pair")
        else:
            try:
                self._start_rollout()
            except ValueError as exc:
                print(f"fixed-stride chunk view unavailable: {exc}")
                return False
            self.state.mode = "both"
        return self._refresh()

    def _toggle_world_coordinates(self, _vis):
        if not self.state.show_world_coordinates:
            try:
                self._current_hand_to_world_pose()
            except ValueError as exc:
                print(f"world-coordinate view unavailable: {exc}")
                return False
        self.state.show_world_coordinates = not self.state.show_world_coordinates
        print(f"coordinate frame: {'world' if self.state.show_world_coordinates else 'hand-root'}")
        return self._refresh(reset_view=True)

    def _reset_camera(self, _vis):
        return self._refresh(reset_view=True)

    def run(self) -> None:
        import open3d as o3d

        self._refresh_cache()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name="CmAction point flow", width=1400, height=920):
            raise RuntimeError(
                "Open3D window creation failed. Check DISPLAY/OpenGL; "
                f"DISPLAY={os.environ.get('DISPLAY')!r}."
            )
        self.vis = vis
        option = vis.get_render_option()
        option.point_size = 4.0
        option.line_width = 1.0
        option.background_color = np.asarray([0.025, 0.025, 0.035])
        self.current_obj = o3d.geometry.PointCloud()
        self.gt_obj = o3d.geometry.PointCloud()
        self.pred_obj = o3d.geometry.PointCloud()
        self.hand_current = o3d.geometry.PointCloud()
        self.hand_future = o3d.geometry.PointCloud()
        self.gt_lines = o3d.geometry.LineSet()
        self.pred_lines = o3d.geometry.LineSet()
        num_slots = int(self.prediction["cm_tokens"].shape[1])
        for _ in range(num_slots):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.004, resolution=12)
            sphere.compute_vertex_normals()
            self.slot_anchors.append(sphere)
            self.slot_anchor_base_vertices.append(np.asarray(sphere.vertices).copy())
        for geometry in (
            self.current_obj,
            self.gt_obj,
            self.pred_obj,
            self.hand_current,
            self.hand_future,
            self.gt_lines,
            self.pred_lines,
            *self.slot_anchors,
        ):
            vis.add_geometry(geometry)
        self._update_geometry(reset_view=True)
        keymap = {
            ord("A"): self._change_pair(-1),
            ord("D"): self._change_pair(+1),
            263: self._change_pair(-1),
            262: self._change_pair(+1),
            ord("G"): self._cycle_mode,
            ord("L"): self._toggle_lines,
            ord("H"): self._toggle_future_hand,
            ord("S"): self._toggle_slot_assignment,
            ord("T"): self._toggle_trajectory_mode,
            ord("P"): self._toggle_rollout,
            ord("W"): self._toggle_world_coordinates,
            ord("R"): self._reset_camera,
        }
        for key, callback in keymap.items():
            vis.register_key_callback(key, callback)
        print(
            "A/D: pair or chunk step, G: GT/Pred/Both, L: flow lines, H: future hand, "
            "S: slot colors/anchors, T: chunk mode, P: start/clear fixed-stride chunks, W: hand/world, R: reset"
        )
        vis.run()
        vis.destroy_window()


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(
        args.checkpoint,
        mode="eval",
        device=args.device,
        build_data=False,
    )
    runner.setup_inference(args.checkpoint)
    runner = _ensure_cm_runner(runner)
    dataset = Stage4CmDataset(
        Path(args.input),
        num_obj_points=int(runner.cfg.meta.num_obj_points),
        num_hand_points=int(runner.cfg.meta.num_hand_points),
        base_seed=int(runner.cfg.train.seed),
        active_only=not args.include_inactive,
        min_stride=int(runner.cfg.data.min_stride),
        max_stride=int(runner.cfg.data.max_stride),
        fixed_stride=args.stride,
        coordinate_frame=str(runner.cfg.meta.coordinate_frame),
    )
    if args.assignment_report:
        print(_assignment_report(runner, dataset))
        return
    if args.pair < 0 or args.pair >= len(dataset):
        raise ValueError(f"--pair must be in [0, {len(dataset) - 1}], got {args.pair}")
    state = ViewerState(pair_idx=args.pair)
    sample = dataset[state.pair_idx]
    prediction = _predict(runner, sample)
    valid = sample["obj_valid_mask"].bool()
    gt = sample["obj_flow_gt"][valid]
    pred = prediction["pred_obj_flow"].squeeze(0)[valid]
    if args.check_only:
        print(
            {
                "pair": state.pair_idx,
                "raw_frame_id": int(sample["raw_frame_id"]),
                "next_raw_frame_id": int(sample["next_raw_frame_id"]),
                "valid_object_count": int(valid.sum()),
                "cm_tokens_shape": tuple(prediction["cm_tokens"].shape),
                "pred_obj_flow_shape": tuple(prediction["pred_obj_flow"].shape),
                "decoder_slot_usage": prediction["decoder_slot_usage"].squeeze(0).tolist(),
                "flow_mse": float(torch.mean((pred - gt).square())) if len(gt) else None,
                "flow_mae": float(torch.mean((pred - gt).abs())) if len(gt) else None,
            }
        )
        return
    try:
        InteractiveFlowViewer(
            runner=runner,
            dataset=dataset,
            state=state,
            flow_stride=args.flow_stride,
        ).run()
    except ImportError as exc:
        raise RuntimeError("open3d is required for interactive visualization. Use --check-only for validation.") from exc


if __name__ == "__main__":
    main()
