from __future__ import annotations

# =============================================================================
# 交互式可视化脚本 — HTML / Three.js 鼠标控制
# =============================================================================
# 通用操作
#   上一帧 / 下一帧      按钮 [◀ Prev] / [Next ▶]  (← / → 方向键亦可)
#   上一 epoch / 下一    按钮 [◀ Epoch] / [Epoch ▶]  + 输入框直接设置
#   主模式切换            [CrossEdge] / [HandHeatmap] 按钮
#   着色源                [GT] / [Eval] 按钮
#   选中物体              [◀ Prev] / [Next ▶] 按钮 (CrossEdge 着色用)
#   重置相机              [Reset Cam] 按钮  (R 键亦可)
#
# 扰动模式
#   开启 / 关闭           [Enable] 按钮 (P 键亦可)
#   清零扰动              [Reset (0)] 按钮 (0 键亦可)
#   Ref GT / Pseudo 切换  [Reference GT] 按钮 (F 键亦可, 仅扰动开启)
#
# 平移扰动 (扰动开启时可用,单位 米)
#   X / Y / Z             滑块或 [X-] [X+] 等步进按钮 (步长见右上标签)
#
# 旋转扰动 (扰动开启时可用,单位 度)
#   Rx / Ry / Rz          滑块或 [Rx-] [Rx+] 等步进按钮 (步长见右上标签)
# =============================================================================

import argparse
import copy
import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.base import build_runner_from_checkpoint, load_config
from src.task.correspondence_ptv3_v2.dataset import CorrStaticDatasetV2
from src.task.correspondence_ptv3_v2.losses import contact_target_from_distance
from src.task.correspondence_ptv3_v2.runner import CorrespondencePTV3V2Runner


OBJ_GRAY = np.asarray([0.62, 0.62, 0.66], dtype=np.float64)
HAND_CROSS_LOW = np.asarray([0.18, 0.18, 0.22], dtype=np.float64)
HAND_CROSS_HIGH = np.asarray([0.98, 0.16, 0.12], dtype=np.float64)
HAND_HEATMAP_LOW = np.asarray([0.07, 0.13, 0.24], dtype=np.float64)
HAND_HEATMAP_HIGH = np.asarray([0.72, 0.28, 0.96], dtype=np.float64)
MARKER_COLOR = np.asarray([1.0, 0.95, 0.15], dtype=np.float64)


@ dataclass
class ViewerState:
    frame_idx: int = 0
    epoch: int = 0
    selected_rank: int = 0
    show_gt: bool = True
    display_mode: str = "cross_edge"


@ dataclass
class PerturbationState:
    enabled: bool = False
    translation: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))
    rotation_deg_xyz: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=np.float32))


@ dataclass
class HandPerturbationState:
    enabled: bool = False
    strength: float = 1.0
    variant: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize correspondence_ptv3_v2 CrossEdge and HandHeatmap predictions in the browser."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional model checkpoint. Omit it for GT-only dataset visualization.",
    )
    parser.add_argument(
        "--config",
        default="src.task.correspondence_ptv3_v2.config:Config",
        help="Task config or YAML used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Stage 3 file/root. Defaults to data.val_path or data.train_path from the config.",
    )
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
        help="Override GT visualization radius in meters. Defaults to meta.contact_radius.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP host to bind the visualizer to. Default 127.0.0.1 (loopback).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port to bind the visualizer to. Default 8765.",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _ensure_v2_runner(runner: Any) -> None:
    if not isinstance(runner, CorrespondencePTV3V2Runner):
        raise ValueError("visualize.py only supports correspondence_ptv3_v2 checkpoints.")


def _prepare_single_batch(batch: dict[str, Any]) -> dict[str, Any]:
    prepared: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            prepared[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            prepared[key] = [value]
        else:
            prepared[key] = value
    return prepared


def _clone_tensor_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cloned: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        cloned[key] = value.clone() if torch.is_tensor(value) else value
    return cloned


def _load_sequence(path: Path, cfg: Any) -> CorrStaticDatasetV2:
    meta = cfg.meta
    return CorrStaticDatasetV2(
        path,
        num_obj_points=int(meta.num_obj_points),
        num_hand_points=int(meta.num_hand_points),
        num_supervision_edges=int(meta.num_supervision_edges),
        contact_radius=float(meta.contact_radius),
        base_seed=int(cfg.train.seed),
        augment=False,
        apply_obj_perturb=False,
        apply_hand_perturb=False,
        runtime_resample_object=False,
        use_mano_reconstruction=False,
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


def _gt_cross_prob(
    batch: dict[str, torch.Tensor],
    obj_idx: int,
    contact_radius: float,
    *,
    point_key: str = "gt_points",
) -> torch.Tensor:
    gt_points = batch[point_key].float()
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


def _sample_has_mano(batch: dict[str, Any]) -> bool:
    value = batch.get("has_mano")
    if torch.is_tensor(value):
        return bool(value.numel() and value.bool().all().item())
    return bool(value)


def _mano_representation(batch: dict[str, Any]) -> tuple[str | None, int]:
    if not _sample_has_mano(batch):
        return None, 0
    use_pca = batch.get("mano_use_pca", True)
    if torch.is_tensor(use_pca):
        use_pca = bool(use_pca.reshape(-1)[0].item())
    pose_dim = batch.get(
        "mano_num_pca_comps" if use_pca else "mano_pose_dim",
        24 if use_pca else 45,
    )
    if torch.is_tensor(pose_dim):
        pose_dim = int(pose_dim.reshape(-1)[0].item())
    return ("pca" if bool(use_pca) else "axis_angle"), int(pose_dim)


def _apply_hand_perturbation(
    batch: dict[str, Any],
    perturbation: HandPerturbationState,
    mano_runner: CorrespondencePTV3V2Runner | None,
    *,
    base_pca_std: float,
    base_pca_scale: float,
    base_axis_angle_std_rad: float,
    base_geometry_noise_scale: float = 1.0,
) -> dict[str, Any]:
    if (
        not perturbation.enabled
        or perturbation.strength <= 0.0
        or mano_runner is None
        or not _sample_has_mano(batch)
    ):
        return batch

    prepared = _prepare_single_batch(_clone_tensor_batch(batch))
    prepared["apply_hand_perturb"] = torch.ones((1,), dtype=torch.bool)
    seed = prepared.get("hand_perturb_seed")
    if torch.is_tensor(seed):
        # Variant zero exactly matches the dataset seed. Other variants are
        # deterministic but independent without changing frame/epoch sampling.
        offset = int(perturbation.variant) * 0x1E3779B97F4A7C15
        prepared["hand_perturb_seed"] = (seed.long() + offset) & ((1 << 63) - 1)

    meta = mano_runner.cfg.meta
    original = {
        "apply_hand_perturb": getattr(meta, "apply_hand_perturb", False),
        "hand_perturb_prob": getattr(meta, "hand_perturb_prob", 1.0),
        "hand_pca_std": getattr(meta, "hand_pca_std", 0.0),
        "hand_pca_noise_scale": getattr(meta, "hand_pca_noise_scale", 0.0),
        "hand_axis_angle_std_rad": getattr(meta, "hand_axis_angle_std_rad", 0.0),
        "hand_geometry_noise_scale": getattr(meta, "hand_geometry_noise_scale", 1.0),
    }
    try:
        meta.apply_hand_perturb = True
        # The UI toggle is the gate. Once enabled, always draw a perturbation;
        # the YAML probability remains visible as training metadata.
        meta.hand_perturb_prob = 1.0
        meta.hand_pca_std = float(base_pca_std) * float(perturbation.strength)
        meta.hand_pca_noise_scale = float(base_pca_scale) * float(perturbation.strength)
        meta.hand_axis_angle_std_rad = (
            float(base_axis_angle_std_rad) * float(perturbation.strength)
        )
        meta.hand_geometry_noise_scale = (
            float(base_geometry_noise_scale) * float(perturbation.strength)
        )
        mano_runner._reconstruct_hand_from_mano(
            prepared,
            side=prepared.get("__mano_side__"),
        )
    finally:
        for key, value in original.items():
            setattr(meta, key, value)

    out = _clone_tensor_batch(batch)
    out["points"] = prepared["points"].squeeze(0).detach().cpu()
    out["normals"] = prepared["normals"].squeeze(0).detach().cpu()
    return out


class HtmlVisualizer:
    """Backend for the HTML / Three.js visualizer.

    This class owns the same state machine the Open3D ``InteractiveViewer``
    used (frame, epoch, perturbation, etc.) and serves it over an HTTP
    API.  All NumPy arrays are converted to plain Python lists at the
    boundary so the JSON encoder can serialise them.
    """

    def __init__(
        self,
        *,
        runner: CorrespondencePTV3V2Runner | None,
        dataset: CorrStaticDatasetV2,
        state: ViewerState,
        marker_radius: float,
        vis_contact_radius: float,
        perturb_translation_step: float,
        perturb_rotation_step_deg: float,
        mano_runner: CorrespondencePTV3V2Runner | None = None,
        hand_pca_std: float = 0.0,
        hand_pca_noise_scale: float = 0.0,
        hand_pca_noise_clip: float = 0.0,
        hand_axis_angle_std_rad: float = 0.0,
        hand_axis_angle_clip_rad: float = 0.0,
        hand_perturb_prob: float = 0.0,
        hand_pca_uses_per_dim_std: bool = False,
        hand_geometry_noise_scale: float = 1.0,
    ) -> None:
        self.runner = runner
        self.dataset = dataset
        self.state = state
        self.marker_radius = float(marker_radius)
        self.vis_contact_radius = float(vis_contact_radius)
        self.perturb_translation_step = float(perturb_translation_step)
        self.perturb_rotation_step_deg = float(perturb_rotation_step_deg)
        self.perturbation = PerturbationState()
        self.hand_perturbation = HandPerturbationState()
        self.mano_runner = mano_runner
        self.hand_pca_std = float(hand_pca_std)
        self.hand_pca_noise_scale = float(hand_pca_noise_scale)
        self.hand_pca_noise_clip = float(hand_pca_noise_clip)
        self.hand_axis_angle_std_rad = float(hand_axis_angle_std_rad)
        self.hand_axis_angle_clip_rad = float(hand_axis_angle_clip_rad)
        self.hand_perturb_prob = float(hand_perturb_prob)
        self.hand_pca_uses_per_dim_std = bool(hand_pca_uses_per_dim_std)
        self.hand_geometry_noise_scale = float(hand_geometry_noise_scale)
        self.show_reference_gt = False
        self.base_batch: dict[str, torch.Tensor] | None = None
        self.current_batch: dict[str, torch.Tensor] | None = None
        self.selected_obj_idx = 0
        self._lock = threading.Lock()
        self.refresh_cache()

    # ---------- state helpers ----------

    def refresh_cache(self) -> None:
        self.base_batch = _sample(self.dataset, self.state.frame_idx, self.state.epoch)
        self.current_batch = _apply_object_perturbation(self.base_batch, self.perturbation)
        self.current_batch = _apply_hand_perturbation(
            self.current_batch,
            self.hand_perturbation,
            self.mano_runner,
            base_pca_std=self.hand_pca_std,
            base_pca_scale=self.hand_pca_noise_scale,
            base_axis_angle_std_rad=self.hand_axis_angle_std_rad,
            base_geometry_noise_scale=self.hand_geometry_noise_scale,
        )
        self.selected_obj_idx = _select_valid_obj(self.current_batch, self.state.selected_rank)

    def _has_mano(self) -> bool:
        return bool(
            self.mano_runner is not None
            and self.base_batch is not None
            and _sample_has_mano(self.base_batch)
        )

    def _any_perturbation_enabled(self) -> bool:
        return self.perturbation.enabled or (
            self.hand_perturbation.enabled and self._has_mano()
        )

    def _has_hand_heatmap_head(self) -> bool:
        if self.runner is None:
            return False
        model = self.runner.model
        return bool(model is not None and getattr(model, "use_hand_contact_head", False))

    def _has_model(self) -> bool:
        return self.runner is not None and self.runner.model is not None

    def _mode_available(self, mode: str) -> bool:
        if mode == "cross_edge":
            return self.state.show_gt or self._has_model()
        if mode == "hand_heatmap":
            return self.state.show_gt or self._has_hand_heatmap_head()
        return False

    def get_state_dict(self) -> dict[str, Any]:
        hand_repr, hand_pose_dim = _mano_representation(self.base_batch or {})
        strength = float(self.hand_perturbation.strength)
        return {
            "frame_idx": self.state.frame_idx,
            "total_frames": len(self.dataset),
            "epoch": self.state.epoch,
            "selected_rank": self.state.selected_rank,
            "show_gt": self.state.show_gt,
            "display_mode": self.state.display_mode,
            "perturbation_enabled": self.perturbation.enabled,
            "show_reference_gt": self.show_reference_gt,
            "translation": [float(v) for v in self.perturbation.translation.tolist()],
            "rotation_deg_xyz": [float(v) for v in self.perturbation.rotation_deg_xyz.tolist()],
            "hand_perturbation_enabled": self.hand_perturbation.enabled,
            "hand_perturbation_available": self._has_mano(),
            "hand_perturbation_representation": hand_repr,
            "hand_pose_dim": hand_pose_dim,
            "hand_perturbation_strength": strength,
            "hand_perturbation_variant": self.hand_perturbation.variant,
            "hand_pca_std_config": self.hand_pca_std,
            "hand_pca_noise_scale_config": self.hand_pca_noise_scale,
            "hand_pca_noise_clip": self.hand_pca_noise_clip,
            "hand_axis_angle_std_rad_config": self.hand_axis_angle_std_rad,
            "hand_axis_angle_clip_rad": self.hand_axis_angle_clip_rad,
            "hand_perturb_prob_config": self.hand_perturb_prob,
            "hand_pca_uses_per_dim_std": self.hand_pca_uses_per_dim_std,
            "hand_geometry_noise_profile_active": bool(
                self.mano_runner is not None
                and getattr(self.mano_runner, "_hand_geometry_profiles", None)
            ),
            "hand_geometry_noise_scale_effective": (
                self.hand_geometry_noise_scale * strength
            ),
            "hand_pca_std_effective": self.hand_pca_std * strength,
            "hand_pca_noise_scale_effective": self.hand_pca_noise_scale * strength,
            "hand_axis_angle_std_rad_effective": self.hand_axis_angle_std_rad * strength,
            "perturb_translation_step": self.perturb_translation_step,
            "perturb_rotation_step_deg": self.perturb_rotation_step_deg,
            "marker_radius": self.marker_radius,
            "vis_contact_radius": self.vis_contact_radius,
            "has_hand_heatmap_head": self._has_hand_heatmap_head(),
            "has_model": self._has_model(),
        }

    def _build_scene_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
                hand_points = (
                    noisy_points[num_obj : num_obj + num_hand]
                    if self.hand_perturbation.enabled and self._has_mano()
                    else gt_points[num_obj : num_obj + num_hand]
                )
                gt_color_batch = (
                    self.base_batch
                    if (self._any_perturbation_enabled() and self.show_reference_gt)
                    else batch
                )
                hand_prob = _gt_cross_prob(
                    gt_color_batch,
                    self.selected_obj_idx,
                    self.vis_contact_radius,
                    point_key=(
                        "points"
                        if self._any_perturbation_enabled() and not self.show_reference_gt
                        else "gt_points"
                    ),
                ).numpy()
            else:
                assert self.runner is not None
                hand_points = noisy_points[num_obj : num_obj + num_hand]
                hand_prob = _dense_cross_prob(self.runner, batch, self.selected_obj_idx).numpy()
            hand_colors = _prob_to_cross_colors(hand_prob)
        else:
            if self.state.show_gt:
                hand_points = (
                    noisy_points[num_obj : num_obj + num_hand]
                    if self.hand_perturbation.enabled and self._has_mano()
                    else gt_points[num_obj : num_obj + num_hand]
                )
                if self._any_perturbation_enabled() and not self.show_reference_gt:
                    hand_prob = _current_pseudo_hand_heatmap_prob(
                        batch, self.vis_contact_radius
                    ).numpy()
                else:
                    hand_prob = _reference_hand_heatmap_prob(
                        self.base_batch, self.vis_contact_radius
                    ).numpy()
            else:
                assert self.runner is not None
                hand_points = noisy_points[num_obj : num_obj + num_hand]
                hand_prob = _predict_hand_heatmap_prob(self.runner, batch).numpy()
            hand_colors = _prob_to_hand_heatmap_colors(hand_prob)
        marker_center = obj_points[self.selected_obj_idx]
        return (
            obj_points[obj_valid].copy(),
            obj_colors[obj_valid].copy(),
            hand_points,
            hand_colors,
            marker_center,
        )

    def get_scene_dict(self) -> dict[str, Any]:
        with self._lock:
            obj_points, obj_colors, hand_points, hand_colors, marker_center = (
                self._build_scene_arrays()
            )
            return {
                "obj_points": obj_points.tolist(),
                "obj_colors": obj_colors.tolist(),
                "hand_points": hand_points.tolist(),
                "hand_colors": hand_colors.tolist(),
                "marker_center": [float(v) for v in marker_center.tolist()],
                "marker_radius": self.marker_radius,
            }

    # ---------- action handling ----------

    def apply_action(self, action: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            n_frames = len(self.dataset)
            if action == "prev_frame":
                self.state.frame_idx = max(0, self.state.frame_idx - 1)
            elif action == "next_frame":
                self.state.frame_idx = min(n_frames - 1, self.state.frame_idx + 1)
            elif action == "set_frame":
                self.state.frame_idx = int(
                    np.clip(int(params.get("frame_idx", 0)), 0, n_frames - 1)
                )
            elif action == "prev_epoch":
                self.state.epoch = max(0, self.state.epoch - 1)
            elif action == "next_epoch":
                self.state.epoch = max(0, self.state.epoch + 1)
            elif action == "set_epoch":
                self.state.epoch = max(0, int(params.get("epoch", 0)))
            elif action == "set_mode":
                mode = str(params.get("mode", "cross_edge"))
                if self._mode_available(mode):
                    self.state.display_mode = mode
            elif action == "set_show_gt":
                requested = bool(params.get("show_gt", True))
                if requested or self._has_model():
                    previous = self.state.show_gt
                    self.state.show_gt = requested
                    if not self._mode_available(self.state.display_mode):
                        self.state.show_gt = previous
            elif action == "toggle_gt":
                if self._has_model():
                    previous = self.state.show_gt
                    self.state.show_gt = not self.state.show_gt
                    if not self._mode_available(self.state.display_mode):
                        self.state.show_gt = previous
            elif action == "toggle_mode":
                if self.state.display_mode == "cross_edge":
                    if self._mode_available("hand_heatmap"):
                        self.state.display_mode = "hand_heatmap"
                else:
                    self.state.display_mode = "cross_edge"
            elif action == "prev_obj":
                self.state.selected_rank = max(0, self.state.selected_rank - 1)
            elif action == "next_obj":
                self.state.selected_rank = max(0, self.state.selected_rank + 1)
            elif action == "set_obj":
                self.state.selected_rank = max(0, int(params.get("rank", 0)))
            elif action == "toggle_perturb":
                self.perturbation.enabled = not self.perturbation.enabled
                if not self.perturbation.enabled:
                    self.show_reference_gt = False
            elif action == "toggle_hand_perturb":
                if self._has_mano():
                    self.hand_perturbation.enabled = not self.hand_perturbation.enabled
                    if not self._any_perturbation_enabled():
                        self.show_reference_gt = False
            elif action == "reset_hand_perturb":
                self.hand_perturbation.enabled = False
                self.hand_perturbation.strength = 1.0
                self.hand_perturbation.variant = 0
                if not self._any_perturbation_enabled():
                    self.show_reference_gt = False
            elif action == "set_hand_perturb_strength":
                self.hand_perturbation.strength = float(
                    np.clip(float(params.get("strength", 1.0)), 0.0, 5.0)
                )
            elif action == "prev_hand_variant":
                self.hand_perturbation.variant = max(0, self.hand_perturbation.variant - 1)
            elif action == "next_hand_variant":
                self.hand_perturbation.variant += 1
            elif action == "set_hand_variant":
                self.hand_perturbation.variant = max(0, int(params.get("variant", 0)))
            elif action == "reset_perturb":
                self.perturbation.translation[:] = 0.0
                self.perturbation.rotation_deg_xyz[:] = 0.0
                self.perturbation.enabled = False
                self.show_reference_gt = False
            elif action == "toggle_refgt":
                if not self._any_perturbation_enabled():
                    pass
                else:
                    self.show_reference_gt = not self.show_reference_gt
            elif action == "set_translation":
                vec = params.get("translation")
                if isinstance(vec, (list, tuple)) and len(vec) == 3:
                    self.perturbation.translation[:] = [float(v) for v in vec]
            elif action == "set_rotation":
                vec = params.get("rotation_deg_xyz")
                if isinstance(vec, (list, tuple)) and len(vec) == 3:
                    self.perturbation.rotation_deg_xyz[:] = [float(v) for v in vec]
            elif action == "translate":
                axis = int(params.get("axis", 0))
                delta = float(params.get("delta", 0.0))
                if 0 <= axis < 3:
                    self.perturbation.translation[axis] += delta
            elif action == "rotate":
                axis = int(params.get("axis", 0))
                delta = float(params.get("delta_deg", 0.0))
                if 0 <= axis < 3:
                    self.perturbation.rotation_deg_xyz[axis] += delta
            else:
                raise ValueError(f"Unknown action: {action!r}")
            self.refresh_cache()
            return self.get_state_dict()


class _Handler(BaseHTTPRequestHandler):
    server_version = "CorrespondencePTV3V2Visualizer/1.0"
    visualizer: HtmlVisualizer | None = None
    html_path: Path | None = None

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default stderr noise; the visualizer prints its own
        # status line.  Uncomment to debug HTTP traffic.
        return

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        assert self.html_path is not None
        with self.html_path.open("rb") as f:
            body = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_404(self) -> None:
        body = b"Not Found"
        self.send_response(HTTPStatus.NOT_FOUND)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON body: {e}") from e

    def do_GET(self) -> None:
        assert self.visualizer is not None
        if self.path in ("/", "/index.html"):
            self._send_html()
        elif self.path == "/api/state":
            self._send_json(self.visualizer.get_state_dict())
        elif self.path == "/api/scene":
            self._send_json(self.visualizer.get_scene_dict())
        elif self.path == "/api/health":
            self._send_json({"ok": True})
        else:
            self._send_404()

    def do_POST(self) -> None:
        assert self.visualizer is not None
        if self.path != "/api/action":
            self._send_404()
            return
        try:
            body = self._read_json_body()
        except ValueError as e:
            self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
            return
        action = body.pop("action", None)
        if not isinstance(action, str) or not action:
            self._send_json(
                {"error": "Missing 'action' field in JSON body."},
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            new_state = self.visualizer.apply_action(action, body)
        except Exception as e:  # noqa: BLE001 — surface to client
            self._send_json({"error": str(e)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json({"ok": True, "state": new_state})


def main() -> None:
    args = parse_args()
    runner: CorrespondencePTV3V2Runner | None = None
    if args.checkpoint:
        loaded_runner = build_runner_from_checkpoint(
            args.checkpoint,
            mode="eval",
            device=args.device,
            build_data=False,
        )
        loaded_runner.setup_inference(args.checkpoint)
        _ensure_v2_runner(loaded_runner)
        runner = loaded_runner
        cfg = runner.cfg
    else:
        cfg = load_config(args.config)

    input_value = args.input
    if not input_value:
        input_value = getattr(cfg.data, "val_path", None) or getattr(
            cfg.data, "train_path", None
        )
    if not input_value:
        raise ValueError(
            "No Stage 3 input configured. Pass --input or set data.val_path/data.train_path."
        )
    input_path = Path(str(input_value))
    dataset = _load_sequence(input_path, cfg)
    state = ViewerState(frame_idx=args.frame, epoch=args.epoch)
    sample = _sample(dataset, state.frame_idx, state.epoch)
    selected_obj = _select_valid_obj(sample, state.selected_rank)
    vis_contact_radius = float(
        args.vis_contact_radius
        if args.vis_contact_radius is not None
        else cfg.meta.contact_radius
    )
    gt_cross = _gt_cross_prob(sample, selected_obj, vis_contact_radius)
    eval_cross = _dense_cross_prob(runner, sample, selected_obj) if runner is not None else None
    if args.check_only:
        has_hand_heatmap_head = bool(
            runner is not None
            and runner.model is not None
            and getattr(runner.model, "use_hand_contact_head", False)
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
                "eval_cross_shape": tuple(eval_cross.shape) if eval_cross is not None else None,
                "has_mano": _sample_has_mano(sample),
                "mano_representation": _mano_representation(sample)[0],
                **hand_heatmap_info,
            }
        )
        return

    mano_runner: CorrespondencePTV3V2Runner | None = None
    if runner is None and _sample_has_mano(sample):
        mano_cfg = copy.deepcopy(cfg)
        mano_runner = CorrespondencePTV3V2Runner(
            mano_cfg,
            mode="eval",
            device="cpu",
            build_data=False,
        )

    visualizer = HtmlVisualizer(
        runner=runner,
        dataset=dataset,
        state=state,
        marker_radius=args.marker_radius,
        vis_contact_radius=vis_contact_radius,
        perturb_translation_step=args.perturb_translation_step,
        perturb_rotation_step_deg=args.perturb_rotation_step_deg,
        mano_runner=mano_runner,
        hand_pca_std=float(getattr(cfg.meta, "hand_pca_std", 0.0)),
        hand_pca_noise_scale=float(getattr(cfg.meta, "hand_pca_noise_scale", 0.0)),
        hand_pca_noise_clip=float(getattr(cfg.meta, "hand_pca_noise_clip", 0.0)),
        hand_axis_angle_std_rad=float(getattr(cfg.meta, "hand_axis_angle_std_rad", 0.0)),
        hand_axis_angle_clip_rad=float(getattr(cfg.meta, "hand_axis_angle_clip_rad", 0.0)),
        hand_perturb_prob=float(getattr(cfg.meta, "hand_perturb_prob", 0.0)),
        hand_pca_uses_per_dim_std=(
            getattr(cfg.meta, "hand_pca_train_std_per_dim", None) is not None
        ),
        hand_geometry_noise_scale=float(
            getattr(cfg.meta, "hand_geometry_noise_scale", 1.0)
        ),
    )
    html_path = Path(__file__).resolve().parent / "visualize.html"
    _Handler.visualizer = visualizer
    _Handler.html_path = html_path
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"[visualize] serving on http://{args.host}:{args.port}\n"
        f"  checkpoint: {args.checkpoint or 'none (GT-only)'}\n"
        f"  input:      {input_path}\n"
        f"  frames:     {len(dataset)}\n"
        f"  mode:       {state.display_mode} (HandHeatmap available: "
        f"{visualizer._has_hand_heatmap_head()})\n"
        f"  MANO hand:  {visualizer.get_state_dict()['hand_perturbation_representation'] or 'unavailable'}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[visualize] shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
