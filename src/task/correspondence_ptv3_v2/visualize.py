from __future__ import annotations

# =============================================================================
# 交互式可视化脚本 — 按键说明
# =============================================================================
# 通用操作
#   A / D               上一帧 / 下一帧  (← / → 方向键亦可)
#   [ / ]               上一 epoch / 下一 epoch
#   H                   切换 CrossEdge / HandHeatmap 主模式
#   G                   在当前主模式内切换 GT 与模型 Eval 概率的着色
#   ,  / .              切换选中的物体 (仅 CrossEdge 着色使用)
#   R                   重置相机视角
#
# 扰动模式 (Object Perturbation)
#   P                   开启 / 关闭扰动模式 (开启后物体位姿/姿态会按
#                       perturbation 平移旋转渲染)
#   0                   清零扰动 (translation/rotation 归零并关闭扰动)
#   F                   在扰动模式下,切换"用参考 GT(原始未扰动)着
#                       色手部"和"用当前伪 GT(已扰动)着色手部"
#
# 平移扰动 (扰动开启时有效,单位: 米/按键)
#   J / L               物体沿 X 轴 -/+  (左/右)
#   I / K               物体沿 Y 轴 +/-   (上/下)
#   U / O               物体沿 Z 轴 +/-   (前/后)
#
# 旋转扰动 (扰动开启时有效,单位: 度/按键)
#   Z / X               物体绕 X 轴 -/+
#   C / V               物体绕 Y 轴 -/+
#   B / N               物体绕 Z 轴 -/+
# =============================================================================

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.base import build_runner_from_checkpoint
from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2
from src.task.correspondence_ptv3_v2.losses import contact_target_from_distance
from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner


os.environ.setdefault("DISPLAY", "localhost:10.0")


OBJ_GRAY = np.asarray([0.62, 0.62, 0.66], dtype=np.float64)
HAND_CROSS_LOW = np.asarray([0.18, 0.18, 0.22], dtype=np.float64)
HAND_CROSS_HIGH = np.asarray([0.98, 0.16, 0.12], dtype=np.float64)
HAND_HEATMAP_LOW = np.asarray([0.07, 0.13, 0.24], dtype=np.float64)
HAND_HEATMAP_HIGH = np.asarray([0.72, 0.28, 0.96], dtype=np.float64)
MARKER_COLOR = np.asarray([1.0, 0.95, 0.15], dtype=np.float64)


@dataclass
class ViewerState:
    frame_idx: int = 0
    epoch: int = 0
    selected_rank: int = 0
    show_gt: bool = True
    display_mode: str = "cross_edge"


@dataclass
class PerturbationState:
    enabled: bool = False
    translation: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    rotation_deg_xyz: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize correspondence_ptv3_v2 CrossEdge and HandHeatmap predictions."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--marker-radius", type=float, default=0.003)
    parser.add_argument("--perturb-translation-step", type=float, default=0.005)
    parser.add_argument("--perturb-rotation-step-deg", type=float, default=5.0)
    parser.add_argument(
        "--vis-contact-radius",
        type=float,
        default=None,
        help="Override GT visualization radius in meters. Defaults to the checkpoint contact_radius.",
    )
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


def _clone_tensor_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _load_sequence(path: Path, runner: CorrespondencePTV3V2Runner) -> CorrStaticDatasetV2:
    meta = runner.cfg.meta
    return CorrStaticDatasetV2(
        path,
        num_obj_points=int(meta.num_obj_points),
        num_hand_points=int(meta.num_hand_points),
        num_supervision_edges=int(meta.num_supervision_edges),
        contact_radius=float(meta.contact_radius),
        base_seed=int(runner.cfg.train.seed),
        augment=False,
        apply_obj_perturb=False,
        eval_sampling_epoch=None,
        coordinate_frame=str(getattr(meta, "coordinate_frame", "hand_root")),
    )


def _sample(dataset: CorrStaticDatasetV2, frame_idx: int, epoch: int) -> dict[str, torch.Tensor]:
    dataset.set_epoch(epoch)
    return dataset[frame_idx]


def _select_valid_obj(batch: dict[str, torch.Tensor], rank: int) -> int:
    valid_idx = torch.nonzero(batch["runtime_obj_valid_mask"].bool(), as_tuple=False).squeeze(-1)
    if valid_idx.numel() == 0:
        return 0
    return int(valid_idx[int(rank) % int(valid_idx.numel())].item())


def _prob_to_colors(prob: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    alpha = np.clip(prob.astype(np.float64), 0.0, 1.0)[:, None]
    return low[None, :] * (1.0 - alpha) + high[None, :] * alpha


def _prob_to_cross_colors(prob: np.ndarray) -> np.ndarray:
    return _prob_to_colors(prob, HAND_CROSS_LOW, HAND_CROSS_HIGH)


def _prob_to_hand_heatmap_colors(prob: np.ndarray) -> np.ndarray:
    return _prob_to_colors(prob, HAND_HEATMAP_LOW, HAND_HEATMAP_HIGH)


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


def _reference_hand_heatmap_prob(
    batch: dict[str, torch.Tensor],
    contact_radius: float,
) -> torch.Tensor:
    """Return the dense clean hand-contact target used by the training loss."""
    return contact_target_from_distance(
        batch["hand_min_dist"].float(), contact_radius=contact_radius
    ).cpu()


def _current_pseudo_hand_heatmap_prob(
    batch: dict[str, torch.Tensor],
    contact_radius: float,
) -> torch.Tensor:
    """Build the current-scene pseudo target from the visualizer input points.

    This intentionally uses the perturbed object points that the model receives.
    Unlike ``hand_min_dist`` (computed from the full clean object pool), it is a
    sampled-object pseudo target and is only used to inspect perturbation behavior.
    """
    num_obj = int(batch["num_obj_points"])
    num_hand = int(batch["num_hand_points"])
    points = batch["points"].float()
    valid = batch["runtime_obj_valid_mask"].bool()
    hand_points = points[num_obj : num_obj + num_hand]
    obj_points = points[:num_obj][valid]
    if obj_points.numel() == 0:
        return torch.zeros((num_hand,), dtype=hand_points.dtype)
    min_dist = torch.cdist(hand_points, obj_points).amin(dim=-1)
    return contact_target_from_distance(min_dist, contact_radius=contact_radius).cpu()


def _predict_hand_heatmap_prob(
    runner: CorrespondencePTV3V2Runner,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return only the dedicated dense hand-contact head prediction."""
    prediction = _predict_batch(runner, batch)
    if "pred_hand_contact_prob" not in prediction:
        raise RuntimeError(
            "This checkpoint has no dedicated hand-contact head. "
            "Use a checkpoint trained with meta.loss_hand_contact_weight > 0."
        )
    probability = prediction["pred_hand_contact_prob"]
    if probability.ndim != 2 or probability.shape[0] != 1:
        raise ValueError(
            "Expected pred_hand_contact_prob with shape [1, num_hand_points], "
            f"got {tuple(probability.shape)}."
        )
    return probability.squeeze(0)


def _axis_rotation_matrix(axis: int, angle_rad: float) -> torch.Tensor:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    if axis == 0:
        matrix = ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    elif axis == 1:
        matrix = ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    elif axis == 2:
        matrix = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    else:
        raise ValueError(f"axis must be 0/1/2, got {axis}.")
    return torch.tensor(matrix, dtype=torch.float32)


def _compose_rotation_deg_xyz(rotation_deg_xyz: np.ndarray) -> torch.Tensor:
    rotation = torch.eye(3, dtype=torch.float32)
    for axis, angle_deg in enumerate(np.asarray(rotation_deg_xyz, dtype=np.float32)):
        angle_rad = float(np.deg2rad(float(angle_deg)))
        rotation = _axis_rotation_matrix(axis, angle_rad) @ rotation
    return rotation


def _apply_object_perturbation(
    batch: dict[str, torch.Tensor],
    perturbation: PerturbationState,
) -> dict[str, torch.Tensor]:
    if not perturbation.enabled:
        return batch

    translation = torch.from_numpy(np.asarray(perturbation.translation, dtype=np.float32))
    rotation = _compose_rotation_deg_xyz(perturbation.rotation_deg_xyz)
    num_obj = int(batch["num_obj_points"])
    out = _clone_tensor_batch(batch)
    for point_key, normal_key in (("points", "normals"), ("gt_points", "gt_normals")):
        points = out[point_key].float().clone()
        normals = out[normal_key].float().clone()
        points[:num_obj] = points[:num_obj] @ rotation.T + translation
        normals[:num_obj] = normals[:num_obj] @ rotation.T
        norms = torch.linalg.norm(normals[:num_obj], dim=-1, keepdim=True).clamp_min(1e-8)
        normals[:num_obj] = normals[:num_obj] / norms
        out[point_key] = points
        out[normal_key] = normals
    return out


class InteractiveViewer:
    def __init__(
        self,
        *,
        runner: CorrespondencePTV3V2Runner,
        dataset: CorrStaticDatasetV2,
        state: ViewerState,
        marker_radius: float,
        vis_contact_radius: float,
        perturb_translation_step: float,
        perturb_rotation_step_deg: float,
    ) -> None:
        self.runner = runner
        self.dataset = dataset
        self.state = state
        self.marker_radius = float(marker_radius)
        self.vis_contact_radius = float(vis_contact_radius)
        self.perturb_translation_step = float(perturb_translation_step)
        self.perturb_rotation_step_deg = float(perturb_rotation_step_deg)
        self.perturbation = PerturbationState()
        self.show_reference_gt = False
        self.base_batch: dict[str, torch.Tensor] | None = None
        self.current_batch: dict[str, torch.Tensor] | None = None
        self.selected_obj_idx = 0
        self.vis = None
        self.obj_pcd = None
        self.hand_pcd = None
        self.marker = None

    def refresh_cache(self) -> None:
        self.base_batch = _sample(self.dataset, self.state.frame_idx, self.state.epoch)
        self.current_batch = _apply_object_perturbation(self.base_batch, self.perturbation)
        self.selected_obj_idx = _select_valid_obj(self.current_batch, self.state.selected_rank)

    def _mode_label(self) -> str:
        if self.state.display_mode == "cross_edge":
            return "CrossEdge"
        if self.state.display_mode == "hand_heatmap":
            return "HandHeatmap"
        raise ValueError(f"Unknown display mode: {self.state.display_mode!r}.")

    def _has_hand_heatmap_head(self) -> bool:
        model = self.runner.model
        return bool(model is not None and getattr(model, "use_hand_contact_head", False))

    def _build_scene_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.current_batch is not None
        assert self.base_batch is not None
        batch = self.current_batch
        num_obj = int(batch["num_obj_points"])
        num_hand = int(batch["num_hand_points"])
        gt_points = batch["gt_points"].numpy()
        noisy_points = batch["points"].numpy()
        obj_valid = batch["runtime_obj_valid_mask"].numpy().astype(bool)

        obj_points = gt_points[:num_obj]
        obj_colors = np.broadcast_to(OBJ_GRAY[None, :], (num_obj, 3)).copy()
        if self.state.display_mode == "cross_edge":
            if self.state.show_gt:
                hand_points = gt_points[num_obj : num_obj + num_hand]
                gt_color_batch = (
                    self.base_batch
                    if (self.perturbation.enabled and self.show_reference_gt)
                    else batch
                )
                hand_prob = _gt_cross_prob(
                    gt_color_batch, self.selected_obj_idx, self.vis_contact_radius
                ).numpy()
            else:
                hand_points = noisy_points[num_obj : num_obj + num_hand]
                hand_prob = _dense_cross_prob(self.runner, batch, self.selected_obj_idx).numpy()
            hand_colors = _prob_to_cross_colors(hand_prob)
        else:
            if self.state.show_gt:
                hand_points = gt_points[num_obj : num_obj + num_hand]
                if self.perturbation.enabled and not self.show_reference_gt:
                    hand_prob = _current_pseudo_hand_heatmap_prob(
                        batch, self.vis_contact_radius
                    ).numpy()
                else:
                    hand_prob = _reference_hand_heatmap_prob(
                        self.base_batch, self.vis_contact_radius
                    ).numpy()
            else:
                hand_points = noisy_points[num_obj : num_obj + num_hand]
                hand_prob = _predict_hand_heatmap_prob(self.runner, batch).numpy()
            hand_colors = _prob_to_hand_heatmap_colors(hand_prob)
        obj_points = obj_points.copy()
        obj_points[~obj_valid] = 0.0
        obj_colors = obj_colors.copy()
        obj_colors[~obj_valid] = 0.0
        marker_center = obj_points[self.selected_obj_idx]
        return obj_points, obj_colors, hand_points, hand_colors, marker_center

    def _update_geometry(self, *, reset_view: bool = False) -> None:
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
        if reset_view:
            self.vis.reset_view_point(True)
        self.vis.poll_events()
        self.vis.update_renderer()
        self._print_status()

    def _print_status(self) -> None:
        source = "GT" if self.state.show_gt else "Eval"
        perturb_mode = "Perturb" if self.perturbation.enabled else "Base"
        if not self.state.show_gt:
            gt_color_mode = "N/A"
        elif not self.perturbation.enabled:
            gt_color_mode = "CleanGT"
        else:
            gt_color_mode = (
                "RefGT"
                if (self.perturbation.enabled and self.show_reference_gt)
                else "CurrentPseudo"
            )
        translation = np.asarray(self.perturbation.translation, dtype=np.float32)
        rotation = np.asarray(self.perturbation.rotation_deg_xyz, dtype=np.float32)
        selected = (
            str(self.selected_obj_idx)
            if self.state.display_mode == "cross_edge"
            else "N/A"
        )
        print(
            f"frame={self.state.frame_idx} epoch={self.state.epoch} "
            f"display={self._mode_label()} source={source} "
            f"selected_obj={selected} scene={perturb_mode} "
            f"gt_color={gt_color_mode} "
            f"t(mm)=({translation[0] * 1e3:+.1f},{translation[1] * 1e3:+.1f},{translation[2] * 1e3:+.1f}) "
            f"r(deg)=({rotation[0]:+.1f},{rotation[1]:+.1f},{rotation[2]:+.1f})"
        )

    def _refresh_scene(self, *, reset_view: bool = False) -> bool:
        self.refresh_cache()
        self._update_geometry(reset_view=reset_view)
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

    def _toggle_display_mode(self, _vis):
        if self.state.display_mode == "cross_edge":
            if not self._has_hand_heatmap_head():
                print(
                    "HandHeatmap is unavailable: this checkpoint has no dedicated "
                    "hand-contact head."
                )
                return False
            self.state.display_mode = "hand_heatmap"
        else:
            self.state.display_mode = "cross_edge"
        return self._refresh_scene()

    def _reset_camera(self, _vis):
        return self._refresh_scene(reset_view=True)

    def _toggle_perturbation(self, _vis):
        self.perturbation.enabled = not self.perturbation.enabled
        if not self.perturbation.enabled:
            self.show_reference_gt = False
        return self._refresh_scene()

    def _reset_perturbation(self, _vis):
        self.perturbation.translation[:] = 0.0
        self.perturbation.rotation_deg_xyz[:] = 0.0
        self.perturbation.enabled = False
        self.show_reference_gt = False
        return self._refresh_scene()

    def _toggle_reference_gt(self, _vis):
        if not self.perturbation.enabled:
            print("Reference GT is only available in perturbation mode. Press P to enable it.")
            return False
        self.show_reference_gt = not self.show_reference_gt
        return self._refresh_scene()

    def _translate_perturbation(self, axis: int, delta: float):
        def callback(_vis):
            if not self.perturbation.enabled:
                print("Perturbation mode is off. Press P to enable it.")
                return False
            self.perturbation.translation[axis] += float(delta)
            return self._refresh_scene()

        return callback

    def _rotate_perturbation(self, axis: int, delta_deg: float):
        def callback(_vis):
            if not self.perturbation.enabled:
                print("Perturbation mode is off. Press P to enable it.")
                return False
            self.perturbation.rotation_deg_xyz[axis] += float(delta_deg)
            return self._refresh_scene()

        return callback

    def run(self) -> None:
        import open3d as o3d

        self.refresh_cache()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        if not vis.create_window(window_name="correspondence_ptv3_v2", width=1280, height=900):
            raise RuntimeError(
                "Open3D window creation failed. Check DISPLAY and OpenGL availability; "
                f"DISPLAY={__import__('os').environ.get('DISPLAY')!r}."
            )
        self.vis = vis
        render_option = vis.get_render_option()
        if render_option is None:
            raise RuntimeError("Open3D render option is unavailable after window creation.")
        render_option.point_size = 4.0
        render_option.line_width = 1.0
        render_option.background_color = np.asarray([0.035, 0.035, 0.045])
        self.obj_pcd = o3d.geometry.PointCloud()
        self.hand_pcd = o3d.geometry.PointCloud()
        self.marker = o3d.geometry.TriangleMesh()
        vis.add_geometry(self.obj_pcd)
        vis.add_geometry(self.hand_pcd)
        vis.add_geometry(self.marker)
        self._update_geometry(reset_view=True)

        keymap = {
            ord("A"): self._change_frame(-1),
            ord("D"): self._change_frame(+1),
            263: self._change_frame(-1),
            262: self._change_frame(+1),
            ord("["): self._change_epoch(-1),
            ord("]"): self._change_epoch(+1),
            ord("H"): self._toggle_display_mode,
            ord("G"): self._toggle_gt_eval,
            ord(","): self._change_selected(-1),
            ord("."): self._change_selected(+1),
            ord("P"): self._toggle_perturbation,
            ord("F"): self._toggle_reference_gt,
            ord("0"): self._reset_perturbation,
            ord("J"): self._translate_perturbation(0, -self.perturb_translation_step),
            ord("L"): self._translate_perturbation(0, +self.perturb_translation_step),
            ord("I"): self._translate_perturbation(1, +self.perturb_translation_step),
            ord("K"): self._translate_perturbation(1, -self.perturb_translation_step),
            ord("U"): self._translate_perturbation(2, +self.perturb_translation_step),
            ord("O"): self._translate_perturbation(2, -self.perturb_translation_step),
            ord("Z"): self._rotate_perturbation(0, -self.perturb_rotation_step_deg),
            ord("X"): self._rotate_perturbation(0, +self.perturb_rotation_step_deg),
            ord("C"): self._rotate_perturbation(1, -self.perturb_rotation_step_deg),
            ord("V"): self._rotate_perturbation(1, +self.perturb_rotation_step_deg),
            ord("B"): self._rotate_perturbation(2, -self.perturb_rotation_step_deg),
            ord("N"): self._rotate_perturbation(2, +self.perturb_rotation_step_deg),
            ord("R"): self._reset_camera,
        }
        for key, callback in keymap.items():
            vis.register_key_callback(key, callback)

        print(
            "A/D or arrows: frame, [/]: epoch, H: CrossEdge/HandHeatmap, "
            "G: GT/Eval, ,/.: CrossEdge object, "
            "P: perturb on/off, F: ref GT colors, 0: clear perturb, "
            "J/L: tx, I/K: ty, U/O: tz, "
            "Z/X: rx, C/V: ry, B/N: rz, R: reset camera"
        )
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
    runner.setup_inference(args.checkpoint)
    _ensure_v2_runner(runner)
    dataset = _load_sequence(Path(args.input), runner)
    state = ViewerState(frame_idx=args.frame, epoch=args.epoch)
    sample = _sample(dataset, state.frame_idx, state.epoch)
    selected_obj = _select_valid_obj(sample, state.selected_rank)
    vis_contact_radius = float(
        args.vis_contact_radius
        if args.vis_contact_radius is not None
        else runner.cfg.meta.contact_radius
    )
    gt_cross = _gt_cross_prob(sample, selected_obj, vis_contact_radius)
    eval_cross = _dense_cross_prob(runner, sample, selected_obj)
    if args.check_only:
        has_hand_heatmap_head = bool(
            runner.model is not None and getattr(runner.model, "use_hand_contact_head", False)
        )
        hand_heatmap_info: dict[str, Any] = {"hand_heatmap_available": has_hand_heatmap_head}
        if has_hand_heatmap_head:
            hand_heatmap_info.update(
                {
                    "gt_hand_heatmap_shape": tuple(
                        _reference_hand_heatmap_prob(sample, vis_contact_radius).shape
                    ),
                    "eval_hand_heatmap_shape": tuple(
                        _predict_hand_heatmap_prob(runner, sample).shape
                    ),
                }
            )
        print(
            {
                "frame": state.frame_idx,
                "epoch": state.epoch,
                "selected_obj": selected_obj,
                "vis_contact_radius": vis_contact_radius,
                "gt_cross_shape": tuple(gt_cross.shape),
                "eval_cross_shape": tuple(eval_cross.shape),
                **hand_heatmap_info,
            }
        )
        return
    try:
        viewer = InteractiveViewer(
            runner=runner,
            dataset=dataset,
            state=state,
            marker_radius=args.marker_radius,
            vis_contact_radius=vis_contact_radius,
            perturb_translation_step=args.perturb_translation_step,
            perturb_rotation_step_deg=args.perturb_rotation_step_deg,
        )
        viewer.run()
    except ImportError as exc:
        raise RuntimeError("open3d is required for interactive visualization. Use --check-only for validation.") from exc


if __name__ == "__main__":
    main()
