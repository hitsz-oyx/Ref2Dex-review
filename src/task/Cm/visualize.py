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
from src.task.Cm.dataset import Stage4CmDataset
from src.task.Cm.runner import CmActionRunner


os.environ.setdefault("DISPLAY", "localhost:10.0")

CURRENT_COLOR = np.asarray([0.38, 0.40, 0.45], dtype=np.float64)
HAND_CURRENT_COLOR = np.asarray([0.90, 0.78, 0.46], dtype=np.float64)
HAND_FUTURE_COLOR = np.asarray([0.34, 0.90, 0.90], dtype=np.float64)
GT_COLOR = np.asarray([0.15, 0.85, 0.34], dtype=np.float64)
PRED_COLOR = np.asarray([0.96, 0.24, 0.16], dtype=np.float64)


@dataclass
class ViewerState:
    pair_idx: int = 0
    mode: str = "both"
    show_lines: bool = True
    show_future_hand: bool = True


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
    parser.add_argument("--check-only", action="store_true")
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


def _flow_lines(start: np.ndarray, end: np.ndarray, *, stride: int, color: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(start) == 0:
        return _empty_points(), np.empty((0, 2), dtype=np.int32), _empty_points()
    ids = np.arange(0, len(start), max(1, int(stride)), dtype=np.int32)
    points = np.concatenate([start[ids], end[ids]], axis=0).astype(np.float64)
    lines = np.stack([np.arange(len(ids)), np.arange(len(ids)) + len(ids)], axis=1).astype(np.int32)
    colors = np.broadcast_to(color[None, :], (len(lines), 3)).copy()
    return points, lines, colors


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

    def _refresh_cache(self) -> None:
        self.sample = self.dataset[self.state.pair_idx]
        self.prediction = _predict(self.runner, self.sample)

    def _scene_arrays(self) -> dict[str, np.ndarray]:
        assert self.sample is not None and self.prediction is not None
        valid = self.sample["obj_valid_mask"].numpy().astype(bool)
        current = self.sample["obj_points"].numpy()[valid].astype(np.float64)
        gt_future = current + self.sample["obj_flow_gt"].numpy()[valid].astype(np.float64)
        pred_future = current + self.prediction["pred_obj_flow"].squeeze(0).numpy()[valid].astype(np.float64)
        hand_current = self.sample["hand_points"].numpy().astype(np.float64)
        hand_future = hand_current + self.sample["hand_flow"].numpy().astype(np.float64)
        return {
            "current": current,
            "gt_future": gt_future,
            "pred_future": pred_future,
            "hand_current": hand_current,
            "hand_future": hand_future,
        }

    @staticmethod
    def _set_point_cloud(point_cloud: Any, points: np.ndarray, color: np.ndarray) -> None:
        import open3d as o3d

        point_cloud.points = o3d.utility.Vector3dVector(points)
        colors = np.broadcast_to(color[None, :], (len(points), 3)).copy()
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

    @staticmethod
    def _set_line_set(line_set: Any, points: np.ndarray, lines: np.ndarray, colors: np.ndarray) -> None:
        import open3d as o3d

        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)

    def _update_geometry(self, *, reset_view: bool = False) -> None:
        arrays = self._scene_arrays()
        show_gt = self.state.mode in {"gt", "both"}
        show_pred = self.state.mode in {"pred", "both"}
        self._set_point_cloud(self.current_obj, arrays["current"], CURRENT_COLOR)
        self._set_point_cloud(self.gt_obj, arrays["gt_future"] if show_gt else _empty_points(), GT_COLOR)
        self._set_point_cloud(self.pred_obj, arrays["pred_future"] if show_pred else _empty_points(), PRED_COLOR)
        self._set_point_cloud(self.hand_current, arrays["hand_current"], HAND_CURRENT_COLOR)
        self._set_point_cloud(
            self.hand_future,
            arrays["hand_future"] if self.state.show_future_hand else _empty_points(),
            HAND_FUTURE_COLOR,
        )
        if self.state.show_lines and show_gt:
            points, lines, colors = _flow_lines(
                arrays["current"], arrays["gt_future"], stride=self.flow_stride, color=GT_COLOR
            )
        else:
            points, lines, colors = _flow_lines(_empty_points(), _empty_points(), stride=1, color=GT_COLOR)
        self._set_line_set(self.gt_lines, points, lines, colors)
        if self.state.show_lines and show_pred:
            points, lines, colors = _flow_lines(
                arrays["current"], arrays["pred_future"], stride=self.flow_stride, color=PRED_COLOR
            )
        else:
            points, lines, colors = _flow_lines(_empty_points(), _empty_points(), stride=1, color=PRED_COLOR)
        self._set_line_set(self.pred_lines, points, lines, colors)
        for geometry in (
            self.current_obj,
            self.gt_obj,
            self.pred_obj,
            self.hand_current,
            self.hand_future,
            self.gt_lines,
            self.pred_lines,
        ):
            self.vis.update_geometry(geometry)
        if reset_view:
            self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()
        self._print_status(arrays)

    def _print_status(self, arrays: dict[str, np.ndarray]) -> None:
        assert self.sample is not None
        flow_error = arrays["pred_future"] - arrays["gt_future"]
        mse = float(np.mean(flow_error**2)) if len(flow_error) else float("nan")
        mae = float(np.mean(np.abs(flow_error))) if len(flow_error) else float("nan")
        raw_frame = int(self.sample["raw_frame_id"].item())
        next_frame = int(self.sample["next_raw_frame_id"].item())
        print(
            f"pair={self.state.pair_idx}/{len(self.dataset) - 1} raw={raw_frame}->{next_frame} "
            f"mode={self.state.mode} valid_obj={len(arrays['current'])} "
            f"flow_mse={mse:.8f} flow_mae={mae:.8f} "
            f"lines={'on' if self.state.show_lines else 'off'} "
            f"future_hand={'on' if self.state.show_future_hand else 'off'}"
        )

    def _refresh(self, *, reset_view: bool = False) -> bool:
        self._refresh_cache()
        self._update_geometry(reset_view=reset_view)
        return False

    def _change_pair(self, delta: int):
        def callback(_vis):
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
        for geometry in (
            self.current_obj,
            self.gt_obj,
            self.pred_obj,
            self.hand_current,
            self.hand_future,
            self.gt_lines,
            self.pred_lines,
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
            ord("R"): self._reset_camera,
        }
        for key, callback in keymap.items():
            vis.register_key_callback(key, callback)
        print("A/D or arrows: pair, G: GT/Pred/Both, L: flow lines, H: future hand, R: reset camera")
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
        coordinate_frame=str(runner.cfg.meta.coordinate_frame),
    )
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
