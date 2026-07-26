"""Interactive Cp object-effect and human-closure viewer.

Keys:
  A / D or left / right  previous / next temporal pair
  V                    cycle Human -> Object flow -> Both
  G                    cycle object flow GT -> Pred -> Both
  C                    cycle object contact colors Off -> GT -> Pred
  L                    toggle sparse current-to-future object-flow lines
  H                    toggle GT / predicted future hand point clouds
  R                    reset camera
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("DISPLAY", "localhost:10.0")

import numpy as np
import torch

from src.base import build_runner_from_checkpoint

from .dataset import Stage5CpDataset


CURRENT_COLOR = np.asarray([0.38, 0.40, 0.45], dtype=np.float64)
GT_COLOR = np.asarray([0.15, 0.85, 0.34], dtype=np.float64)
PRED_COLOR = np.asarray([0.96, 0.24, 0.16], dtype=np.float64)
HAND_CURRENT_COLOR = np.asarray([0.90, 0.78, 0.46], dtype=np.float64)
HAND_FUTURE_COLOR = np.asarray([0.34, 0.90, 0.90], dtype=np.float64)
HAND_PRED_COLOR = PRED_COLOR


@dataclass
class ViewerState:
    pair_idx: int = 0
    scene: str = "both"
    object_mode: str = "both"
    contact_mode: str = "off"
    show_lines: bool = True
    show_future_hand: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Cp object effect and human closure.")
    parser.add_argument("--checkpoint", required=True, help="Cp BaseRunner checkpoint (.pt or run directory).")
    parser.add_argument("--input", required=True, help="One Stage 5 sequence NPZ.")
    parser.add_argument("--pair", type=int, default=0, help="Temporal-pair index in the input file.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--flow-stride", type=int, default=8, help="Draw one object flow line every N valid points.")
    parser.add_argument("--check-only", action="store_true", help="Run one forward pass without opening Open3D.")
    parser.add_argument("--save", default=None, help="Optional static hand-closure PNG output path.")
    return parser.parse_args()


def _empty_points() -> np.ndarray:
    return np.empty((0, 3), dtype=np.float64)


def _prepare_batch(sample: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0).to(device) for key, value in sample.items() if torch.is_tensor(value)}


def _predict(runner: Any, sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        output = runner.inference(runner.model, _prepare_batch(sample, runner.device))
    return {key: value.detach().cpu() for key, value in output.items()}


def _cycle(value: str, options: tuple[str, ...]) -> str:
    return options[(options.index(value) + 1) % len(options)]


def _flow_lines(start: np.ndarray, end: np.ndarray, *, stride: int, color: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(start) == 0:
        return _empty_points(), np.empty((0, 2), dtype=np.int32), _empty_points()
    indices = np.arange(0, len(start), max(1, int(stride)), dtype=np.int32)
    points = np.concatenate([start[indices], end[indices]], axis=0).astype(np.float64)
    lines = np.stack([np.arange(len(indices)), np.arange(len(indices)) + len(indices)], axis=1).astype(np.int32)
    colors = np.broadcast_to(color[None, :], (len(lines), 3)).copy()
    return points, lines, colors


def _contact_colors(contact: np.ndarray, color: np.ndarray) -> np.ndarray:
    """Blend a neutral object color into a contact color using a [0, 1] field."""
    weight = np.clip(np.asarray(contact, dtype=np.float64), 0.0, 1.0)[:, None]
    return (1.0 - weight) * CURRENT_COLOR[None, :] + weight * color[None, :]


def _save_snapshot(path: Path, current: np.ndarray, future_gt: np.ndarray, future_pred: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(9, 8), dpi=180)
    axes = figure.add_subplot(111, projection="3d")
    stride = 3
    axes.scatter(*current[::stride].T, s=1.5, c=HAND_CURRENT_COLOR, label="Current hand")
    axes.scatter(*future_gt[::stride].T, s=1.5, c=HAND_FUTURE_COLOR, label="GT future hand")
    axes.scatter(*future_pred[::stride].T, s=1.5, c=HAND_PRED_COLOR, label="Cp→Cm→Fm prediction")
    axes.set_title("Cp human closure")
    axes.set_axis_off()
    axes.legend(loc="upper right", markerscale=4)
    all_points = np.concatenate([current, future_gt, future_pred], axis=0)
    center = all_points.mean(axis=0)
    radius = max(np.ptp(all_points, axis=0).max() * 0.5, 1e-4)
    axes.set_xlim(center[0] - radius, center[0] + radius)
    axes.set_ylim(center[1] - radius, center[1] + radius)
    axes.set_zlim(center[2] - radius, center[2] + radius)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


class InteractiveCpViewer:
    """Open3D viewer matching Cm's GT/pred point-flow interaction pattern."""

    def __init__(self, runner: Any, dataset: Stage5CpDataset, state: ViewerState, flow_stride: int) -> None:
        self.runner = runner
        self.dataset = dataset
        self.state = state
        self.flow_stride = max(1, int(flow_stride))
        self.sample: dict[str, torch.Tensor] | None = None
        self.prediction: dict[str, torch.Tensor] | None = None
        self.vis: Any = None
        self.current_obj: Any = None
        self.gt_obj: Any = None
        self.pred_obj: Any = None
        self.hand_current: Any = None
        self.hand_future: Any = None
        self.hand_pred_future: Any = None
        self.gt_lines: Any = None
        self.pred_lines: Any = None

    def _refresh_cache(self) -> None:
        self.sample = self.dataset[self.state.pair_idx]
        self.prediction = _predict(self.runner, self.sample)

    def _scene_arrays(self) -> dict[str, np.ndarray]:
        assert self.sample is not None and self.prediction is not None
        valid = self.sample["obj_valid_mask"].numpy().astype(bool)
        current_obj = self.sample["obj_points"].numpy()[valid]
        gt_flow = self.sample["obj_flow_gt"].numpy()[valid]
        pred_flow = self.prediction["pred_obj_flow"].squeeze(0).numpy()[valid]
        hand_current = self.sample["hand_points"].numpy()
        hand_gt_future = hand_current + self.sample["hand_flow"].numpy()
        hand_pred_future = _empty_points()
        if "pred_hand_flow" in self.prediction:
            hand_pred_future = hand_current + self.prediction["pred_hand_flow"].squeeze(0).numpy()
        return {
            "obj_current": current_obj,
            "obj_gt_future": current_obj + gt_flow,
            "obj_pred_future": current_obj + pred_flow,
            "obj_contact_gt": self.sample["obj_contact_gt"].numpy()[valid],
            "obj_contact_pred": torch.sigmoid(self.prediction["pred_obj_contact_logits"].squeeze(0)).numpy()[valid],
            "hand_current": hand_current,
            "hand_gt_future": hand_gt_future,
            "hand_pred_future": hand_pred_future,
        }

    @staticmethod
    def _set_point_cloud(point_cloud: Any, points: np.ndarray, color: np.ndarray) -> None:
        import open3d as o3d

        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
        if len(points) == 0:
            point_cloud.colors = o3d.utility.Vector3dVector(_empty_points())
            return
        if color.shape == (3,):
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

    def _update_geometry(self, *, reset_view: bool = False) -> None:
        arrays = self._scene_arrays()
        show_object = self.state.scene in {"object", "both"}
        show_hand = self.state.scene in {"human", "both"}
        show_gt = self.state.object_mode in {"gt", "both"}
        show_pred = self.state.object_mode in {"pred", "both"}
        current_color = CURRENT_COLOR
        if self.state.contact_mode == "gt":
            current_color = _contact_colors(arrays["obj_contact_gt"], GT_COLOR)
        elif self.state.contact_mode == "pred":
            current_color = _contact_colors(arrays["obj_contact_pred"], PRED_COLOR)
        self._set_point_cloud(self.current_obj, arrays["obj_current"] if show_object else _empty_points(), current_color)
        self._set_point_cloud(
            self.gt_obj,
            arrays["obj_gt_future"] if show_object and show_gt else _empty_points(),
            GT_COLOR,
        )
        self._set_point_cloud(
            self.pred_obj,
            arrays["obj_pred_future"] if show_object and show_pred else _empty_points(),
            PRED_COLOR,
        )
        self._set_point_cloud(self.hand_current, arrays["hand_current"] if show_hand else _empty_points(), HAND_CURRENT_COLOR)
        self._set_point_cloud(
            self.hand_future,
            arrays["hand_gt_future"] if show_hand and self.state.show_future_hand else _empty_points(),
            HAND_FUTURE_COLOR,
        )
        self._set_point_cloud(
            self.hand_pred_future,
            arrays["hand_pred_future"] if show_hand and self.state.show_future_hand else _empty_points(),
            HAND_PRED_COLOR,
        )
        if show_object and self.state.show_lines and show_gt:
            line_points, lines, colors = _flow_lines(
                arrays["obj_current"], arrays["obj_gt_future"], stride=self.flow_stride, color=GT_COLOR
            )
        else:
            line_points, lines, colors = _flow_lines(_empty_points(), _empty_points(), stride=1, color=GT_COLOR)
        self._set_line_set(self.gt_lines, line_points, lines, colors)
        if show_object and self.state.show_lines and show_pred:
            line_points, lines, colors = _flow_lines(
                arrays["obj_current"], arrays["obj_pred_future"], stride=self.flow_stride, color=PRED_COLOR
            )
        else:
            line_points, lines, colors = _flow_lines(_empty_points(), _empty_points(), stride=1, color=PRED_COLOR)
        self._set_line_set(self.pred_lines, line_points, lines, colors)
        for geometry in (
            self.current_obj,
            self.gt_obj,
            self.pred_obj,
            self.hand_current,
            self.hand_future,
            self.hand_pred_future,
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
        object_error = arrays["obj_pred_future"] - arrays["obj_gt_future"]
        object_epe_mm = float(np.linalg.norm(object_error, axis=-1).mean() * 1000.0) if len(object_error) else float("nan")
        message = (
            f"pair={self.state.pair_idx}/{len(self.dataset) - 1} raw={int(self.sample['raw_frame_id'])} "
            f"scene={self.state.scene} object={self.state.object_mode} contact={self.state.contact_mode} "
            f"object_epe={object_epe_mm:.3f}mm lines={'on' if self.state.show_lines else 'off'}"
        )
        if len(arrays["hand_pred_future"]):
            hand_error = arrays["hand_pred_future"] - arrays["hand_gt_future"]
            message += f" hand_epe={np.linalg.norm(hand_error, axis=-1).mean() * 1000.0:.3f}mm"
        print(message)

    def _refresh(self, *, reset_view: bool = False) -> bool:
        self._refresh_cache()
        self._update_geometry(reset_view=reset_view)
        return False

    def _change_pair(self, delta: int):
        def callback(_vis: Any) -> bool:
            self.state.pair_idx = int(np.clip(self.state.pair_idx + delta, 0, len(self.dataset) - 1))
            return self._refresh()

        return callback

    def _cycle_scene(self, _vis: Any) -> bool:
        self.state.scene = _cycle(self.state.scene, ("human", "object", "both"))
        return self._refresh()

    def _cycle_object_mode(self, _vis: Any) -> bool:
        self.state.object_mode = _cycle(self.state.object_mode, ("gt", "pred", "both"))
        return self._refresh()

    def _cycle_contact_mode(self, _vis: Any) -> bool:
        self.state.contact_mode = _cycle(self.state.contact_mode, ("off", "gt", "pred"))
        return self._refresh()

    def _toggle_lines(self, _vis: Any) -> bool:
        self.state.show_lines = not self.state.show_lines
        return self._refresh()

    def _toggle_future_hand(self, _vis: Any) -> bool:
        self.state.show_future_hand = not self.state.show_future_hand
        return self._refresh()

    def _reset_camera(self, _vis: Any) -> bool:
        return self._refresh(reset_view=True)

    def run(self) -> None:
        import open3d as o3d

        self._refresh_cache()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name="Cp object effect and human closure", width=1400, height=920):
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
        self.hand_pred_future = o3d.geometry.PointCloud()
        self.gt_lines = o3d.geometry.LineSet()
        self.pred_lines = o3d.geometry.LineSet()
        for geometry in (
            self.current_obj,
            self.gt_obj,
            self.pred_obj,
            self.hand_current,
            self.hand_future,
            self.hand_pred_future,
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
            ord("V"): self._cycle_scene,
            ord("G"): self._cycle_object_mode,
            ord("C"): self._cycle_contact_mode,
            ord("L"): self._toggle_lines,
            ord("H"): self._toggle_future_hand,
            ord("R"): self._reset_camera,
        }
        for key, callback in keymap.items():
            vis.register_key_callback(key, callback)
        print(
            "A/D: pair, V: Human/Object/Both, G: object GT/Pred/Both, "
            "C: contact Off/GT/Pred, L: flow lines, H: future hand, R: reset"
        )
        vis.run()
        vis.destroy_window()


def main() -> None:
    args = parse_args()
    runner = build_runner_from_checkpoint(args.checkpoint, mode="eval", device=args.device, build_data=False)
    runner.setup_inference(args.checkpoint)
    dataset = Stage5CpDataset(
        Path(args.input),
        num_obj_points=int(runner.cfg.meta.num_obj_points),
        num_hand_points=int(runner.cfg.meta.num_hand_points),
        base_seed=int(runner.cfg.train.seed),
    )
    if args.pair < 0 or args.pair >= len(dataset):
        raise ValueError(f"--pair must be in [0, {len(dataset) - 1}], got {args.pair}")
    state = ViewerState(pair_idx=args.pair)
    sample = dataset[state.pair_idx]
    prediction = _predict(runner, sample)
    valid = sample["obj_valid_mask"].bool()
    object_epe_mm = torch.linalg.norm(
        prediction["pred_obj_flow"].squeeze(0)[valid] - sample["obj_flow_gt"][valid], dim=-1
    ).mean() * 1000.0
    report: dict[str, Any] = {
        "pair": state.pair_idx,
        "raw_frame_id": int(sample["raw_frame_id"]),
        "valid_object_count": int(valid.sum()),
        "object_epe_mm": float(object_epe_mm),
        "pred_obj_flow_shape": tuple(prediction["pred_obj_flow"].shape),
    }
    if "pred_hand_flow" in prediction:
        report["hand_epe_mm"] = float(
            torch.linalg.norm(prediction["pred_hand_flow"].squeeze(0) - sample["hand_flow"], dim=-1).mean() * 1000.0
        )
    print(report)
    if args.save:
        if "pred_hand_flow" not in prediction:
            raise ValueError("--save requires a closed_loop Cp checkpoint with pred_hand_flow.")
        current_hand = sample["hand_points"].numpy()
        _save_snapshot(
            Path(args.save),
            current_hand,
            current_hand + sample["hand_flow"].numpy(),
            current_hand + prediction["pred_hand_flow"].squeeze(0).numpy(),
        )
        print(f"saved snapshot: {args.save}")
    if args.check_only:
        return
    try:
        InteractiveCpViewer(runner, dataset, state, args.flow_stride).run()
    except ImportError as exc:
        raise RuntimeError("open3d is required for interactive visualization. Use --check-only for validation.") from exc


if __name__ == "__main__":
    main()
