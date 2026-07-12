from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.base import build_runner_from_checkpoint
from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2
from src.task.correspondence_ptv3_v2.losses import contact_target_from_distance
from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner


OBJ_BLUE = np.asarray([0.12, 0.36, 0.98], dtype=np.float64)
OBJ_WHITE = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
OBJ_RED = np.asarray([0.98, 0.16, 0.12], dtype=np.float64)
OBJ_GRAY = np.asarray([0.62, 0.62, 0.66], dtype=np.float64)
HAND_GT = np.asarray([0.95, 0.58, 0.12], dtype=np.float64)
HAND_EVAL = np.asarray([0.85, 0.16, 0.85], dtype=np.float64)
HAND_CROSS_LOW = np.asarray([0.18, 0.18, 0.22], dtype=np.float64)
HAND_CROSS_HIGH = np.asarray([0.98, 0.16, 0.12], dtype=np.float64)
MARKER_COLOR = np.asarray([1.0, 0.95, 0.15], dtype=np.float64)


@dataclass
class ViewerState:
    frame_idx: int = 0
    epoch: int = 0
    selected_rank: int = 0
    show_gt: bool = True
    show_cross: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize correspondence_ptv3_v2 predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--marker-radius", type=float, default=0.003)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _ensure_v2_runner(runner: Any) -> None:
    if not isinstance(runner, CorrespondencePTV3V2Runner):
        raise ValueError("visualize.py only supports correspondence_ptv3_v2 checkpoints.")


def _prepare_single_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prepared: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        if value.dim() == 0:
            prepared[key] = value
        else:
            prepared[key] = value.unsqueeze(0)
    return prepared


def _load_sequence(path: Path, runner: CorrespondencePTV3V2Runner) -> CorrStaticDatasetV2:
    meta = runner.cfg.meta
    return CorrStaticDatasetV2(
        path,
        num_obj_points=int(meta.num_obj_points),
        num_hand_points=int(meta.num_hand_points),
        k_ctx=int(meta.k_ctx),
        ctx_radius=float(meta.ctx_radius),
        num_supervision_edges=int(meta.num_supervision_edges),
        contact_radius=float(meta.contact_radius),
        base_seed=int(runner.cfg.train.seed),
        augment=False,
        apply_hand_perturb=False,
        eval_sampling_epoch=None,
    )


def _sample(dataset: CorrStaticDatasetV2, frame_idx: int, epoch: int) -> dict[str, torch.Tensor]:
    dataset.set_epoch(epoch)
    return dataset[frame_idx]


def _select_valid_obj(batch: dict[str, torch.Tensor], rank: int) -> int:
    valid_idx = torch.nonzero(batch["runtime_obj_valid_mask"].bool(), as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return 0
    return int(valid_idx[int(rank) % int(valid_idx.numel())].item())


def _prob_to_diverging_colors(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(prob.astype(np.float64), 0.0, 1.0)
    colors = np.empty((prob.shape[0], 3), dtype=np.float64)
    low_mask = prob <= 0.5
    if np.any(low_mask):
        alpha = (prob[low_mask] / 0.5)[:, None]
        colors[low_mask] = OBJ_BLUE[None, :] * (1.0 - alpha) + OBJ_WHITE[None, :] * alpha
    high_mask = ~low_mask
    if np.any(high_mask):
        alpha = ((prob[high_mask] - 0.5) / 0.5)[:, None]
        colors[high_mask] = OBJ_WHITE[None, :] * (1.0 - alpha) + OBJ_RED[None, :] * alpha
    return colors


def _prob_to_cross_colors(prob: np.ndarray) -> np.ndarray:
    alpha = np.clip(prob.astype(np.float64), 0.0, 1.0)[:, None]
    return HAND_CROSS_LOW[None, :] * (1.0 - alpha) + HAND_CROSS_HIGH[None, :] * alpha


def _gt_cross_prob(batch: dict[str, torch.Tensor], obj_idx: int, contact_radius: float) -> torch.Tensor:
    gt_points = batch["gt_points"].float()
    num_obj = int(batch["num_obj_points"])
    obj_point = gt_points[obj_idx]
    hand_points = gt_points[num_obj : num_obj + int(batch["num_hand_points"])]
    distance = torch.norm(hand_points - obj_point.unsqueeze(0), dim=-1)
    return contact_target_from_distance(distance, contact_radius=contact_radius).cpu()


def _predict_batch(
    runner: CorrespondencePTV3V2Runner,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    prepared = runner.prepare_batch(_prepare_single_batch(batch))
    with torch.no_grad():
        pred = runner.inference(runner.model, prepared)
    return {key: value.detach().cpu() for key, value in pred.items()}


def _dense_cross_prob(
    runner: CorrespondencePTV3V2Runner,
    batch: dict[str, torch.Tensor],
    obj_idx: int,
) -> torch.Tensor:
    prepared = runner.prepare_batch(_prepare_single_batch(batch))
    model = runner.model
    if model is None:
        raise RuntimeError("Runner model is not initialized.")
    with torch.no_grad():
        return model.predict_dense_cross_for_object(prepared, obj_idx).squeeze(0).detach().cpu()


class InteractiveViewer:
    def __init__(
        self,
        *,
        runner: CorrespondencePTV3V2Runner,
        dataset: CorrStaticDatasetV2,
        state: ViewerState,
        marker_radius: float,
    ) -> None:
        self.runner = runner
        self.dataset = dataset
        self.state = state
        self.marker_radius = float(marker_radius)
        self.current_batch: dict[str, torch.Tensor] | None = None
        self.current_pred: dict[str, torch.Tensor] | None = None
        self.selected_obj_idx = 0
        self.initial_camera = None
        self.vis = None
        self.obj_pcd = None
        self.hand_pcd = None
        self.marker = None

    def refresh_cache(self) -> None:
        self.current_batch = _sample(self.dataset, self.state.frame_idx, self.state.epoch)
        self.current_pred = _predict_batch(self.runner, self.current_batch)
        self.selected_obj_idx = _select_valid_obj(self.current_batch, self.state.selected_rank)

    def _build_scene_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.current_batch is not None and self.current_pred is not None
        batch = self.current_batch
        pred = self.current_pred
        num_obj = int(batch["num_obj_points"])
        num_hand = int(batch["num_hand_points"])
        gt_points = batch["gt_points"].numpy()
        noisy_points = batch["points"].numpy()
        obj_valid = batch["runtime_obj_valid_mask"].numpy().astype(bool)

        if self.state.show_cross:
            obj_points = gt_points[:num_obj]
            obj_colors = np.broadcast_to(OBJ_GRAY[None, :], (num_obj, 3)).copy()
            if self.state.show_gt:
                hand_points = gt_points[num_obj : num_obj + num_hand]
                hand_prob = _gt_cross_prob(batch, self.selected_obj_idx, float(self.runner.cfg.meta.contact_radius)).numpy()
            else:
                hand_points = noisy_points[num_obj : num_obj + num_hand]
                hand_prob = _dense_cross_prob(self.runner, batch, self.selected_obj_idx).numpy()
            hand_colors = _prob_to_cross_colors(hand_prob)
        else:
            if self.state.show_gt:
                obj_points = gt_points[:num_obj]
                obj_prob = batch["contact_target"].numpy()
                hand_points = gt_points[num_obj : num_obj + num_hand]
                hand_colors = np.broadcast_to(HAND_GT[None, :], (num_hand, 3)).copy()
            else:
                obj_points = noisy_points[:num_obj]
                obj_prob = pred["pred_obj_contact_prob"].squeeze(0).numpy()
                hand_points = noisy_points[num_obj : num_obj + num_hand]
                hand_colors = np.broadcast_to(HAND_EVAL[None, :], (num_hand, 3)).copy()
            obj_colors = _prob_to_diverging_colors(obj_prob)

        obj_points = obj_points.copy()
        obj_points[~obj_valid] = 0.0
        obj_colors = obj_colors.copy()
        obj_colors[~obj_valid] = 0.0
        marker_center = obj_points[self.selected_obj_idx]
        return obj_points, obj_colors, hand_points, hand_colors, marker_center

    def _update_geometry(self) -> None:
        import open3d as o3d

        obj_points, obj_colors, hand_points, hand_colors, marker_center = self._build_scene_arrays()
        self.obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
        self.obj_pcd.colors = o3d.utility.Vector3dVector(obj_colors)
        self.hand_pcd.points = o3d.utility.Vector3dVector(hand_points)
        self.hand_pcd.colors = o3d.utility.Vector3dVector(hand_colors)
        marker_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=self.marker_radius)
        marker_mesh.paint_uniform_color(MARKER_COLOR)
        marker_mesh.compute_vertex_normals()
        marker_mesh.translate(marker_center)
        self.marker.vertices = marker_mesh.vertices
        self.marker.triangles = marker_mesh.triangles
        self.marker.vertex_colors = marker_mesh.vertex_colors
        self.marker.vertex_normals = marker_mesh.vertex_normals
        self.vis.update_geometry(self.obj_pcd)
        self.vis.update_geometry(self.hand_pcd)
        self.vis.update_geometry(self.marker)
        self.vis.poll_events()
        self.vis.update_renderer()
        self._print_status()

    def _print_status(self) -> None:
        mode = "GT" if self.state.show_gt else "Eval"
        view = "cross" if self.state.show_cross else "heatmap"
        print(
            f"frame={self.state.frame_idx} epoch={self.state.epoch} mode={mode} "
            f"view={view} selected_obj={self.selected_obj_idx}"
        )

    def _refresh_scene(self) -> bool:
        self.refresh_cache()
        self._update_geometry()
        return False

    def _change_frame(self, delta: int):
        def callback(_vis):
            self.state.frame_idx = int(np.clip(self.state.frame_idx + delta, 0, len(self.dataset) - 1))
            return self._refresh_scene()

        return callback

    def _change_epoch(self, delta: int):
        def callback(_vis):
            self.state.epoch = max(0, self.state.epoch + delta)
            return self._refresh_scene()

        return callback

    def _change_selected(self, delta: int):
        def callback(_vis):
            self.state.selected_rank = max(0, self.state.selected_rank + delta)
            return self._refresh_scene()

        return callback

    def _toggle_gt_eval(self, _vis):
        self.state.show_gt = not self.state.show_gt
        return self._refresh_scene()

    def _toggle_view(self, _vis):
        self.state.show_cross = not self.state.show_cross
        return self._refresh_scene()

    def _reset_camera(self, _vis):
        if self.initial_camera is not None:
            _vis.get_view_control().convert_from_pinhole_camera_parameters(self.initial_camera, allow_arbitrary=True)
        return False

    def run(self) -> None:
        import open3d as o3d

        self.refresh_cache()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="correspondence_ptv3_v2")
        self.vis = vis
        self.obj_pcd = o3d.geometry.PointCloud()
        self.hand_pcd = o3d.geometry.PointCloud()
        self.marker = o3d.geometry.TriangleMesh()
        vis.add_geometry(self.obj_pcd)
        vis.add_geometry(self.hand_pcd)
        vis.add_geometry(self.marker)
        self._update_geometry()
        self.initial_camera = vis.get_view_control().convert_to_pinhole_camera_parameters()

        keymap = {
            ord("A"): self._change_frame(-1),
            ord("D"): self._change_frame(+1),
            263: self._change_frame(-1),
            262: self._change_frame(+1),
            ord("["): self._change_epoch(-1),
            ord("]"): self._change_epoch(+1),
            ord("G"): self._toggle_gt_eval,
            ord("C"): self._toggle_view,
            ord(","): self._change_selected(-1),
            ord("."): self._change_selected(+1),
            ord("R"): self._reset_camera,
        }
        for key, callback in keymap.items():
            vis.register_key_callback(key, callback)

        print("A/D or arrows: frame, [/]: epoch, G: GT/Eval, C: heatmap/cross, ,/.: object, R: reset")
        self._print_status()
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
    _ensure_v2_runner(runner)
    dataset = _load_sequence(Path(args.input), runner)
    state = ViewerState(frame_idx=args.frame, epoch=args.epoch)
    sample = _sample(dataset, state.frame_idx, state.epoch)
    selected_obj = _select_valid_obj(sample, state.selected_rank)
    gt_obj_heat = sample["contact_target"].cpu()
    pred = _predict_batch(runner, sample)
    eval_obj_heat = pred["pred_obj_contact_prob"].squeeze(0).cpu()
    gt_cross = _gt_cross_prob(sample, selected_obj, float(runner.cfg.meta.contact_radius))
    eval_cross = _dense_cross_prob(runner, sample, selected_obj)
    if args.check_only:
        print(
            {
                "frame": state.frame_idx,
                "epoch": state.epoch,
                "selected_obj": selected_obj,
                "gt_obj_heat_shape": tuple(gt_obj_heat.shape),
                "eval_obj_heat_shape": tuple(eval_obj_heat.shape),
                "gt_cross_shape": tuple(gt_cross.shape),
                "eval_cross_shape": tuple(eval_cross.shape),
            }
        )
        return
    try:
        viewer = InteractiveViewer(
            runner=runner,
            dataset=dataset,
            state=state,
            marker_radius=args.marker_radius,
        )
        viewer.run()
    except ImportError as exc:
        raise RuntimeError("open3d is required for interactive visualization. Use --check-only for validation.") from exc


if __name__ == "__main__":
    main()
