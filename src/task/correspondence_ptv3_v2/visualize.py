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
import json
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.base import build_runner_from_checkpoint
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize correspondence_ptv3_v2 CrossEdge and HandHeatmap predictions in the browser."
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
        self._lock = threading.Lock()
        self.refresh_cache()

    # ---------- state helpers ----------

    def refresh_cache(self) -> None:
        self.base_batch = _sample(self.dataset, self.state.frame_idx, self.state.epoch)
        self.current_batch = _apply_object_perturbation(self.base_batch, self.perturbation)
        self.selected_obj_idx = _select_valid_obj(self.current_batch, self.state.selected_rank)

    def _has_hand_heatmap_head(self) -> bool:
        model = self.runner.model
        return bool(model is not None and getattr(model, "use_hand_contact_head", False))

    def get_state_dict(self) -> dict[str, Any]:
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
            "perturb_translation_step": self.perturb_translation_step,
            "perturb_rotation_step_deg": self.perturb_rotation_step_deg,
            "marker_radius": self.marker_radius,
            "vis_contact_radius": self.vis_contact_radius,
            "has_hand_heatmap_head": self._has_hand_heatmap_head(),
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
                if mode == "hand_heatmap" and not self._has_hand_heatmap_head():
                    pass  # silently no-op; the button is disabled in the UI
                else:
                    self.state.display_mode = mode
            elif action == "set_show_gt":
                self.state.show_gt = bool(params.get("show_gt", True))
            elif action == "toggle_gt":
                self.state.show_gt = not self.state.show_gt
            elif action == "toggle_mode":
                if self.state.display_mode == "cross_edge":
                    if self._has_hand_heatmap_head():
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
            elif action == "reset_perturb":
                self.perturbation.translation[:] = 0.0
                self.perturbation.rotation_deg_xyz[:] = 0.0
                self.perturbation.enabled = False
                self.show_reference_gt = False
            elif action == "toggle_refgt":
                if not self.perturbation.enabled:
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

    visualizer = HtmlVisualizer(
        runner=runner,
        dataset=dataset,
        state=state,
        marker_radius=args.marker_radius,
        vis_contact_radius=vis_contact_radius,
        perturb_translation_step=args.perturb_translation_step,
        perturb_rotation_step_deg=args.perturb_rotation_step_deg,
    )
    html_path = Path(__file__).resolve().parent / "visualize.html"
    _Handler.visualizer = visualizer
    _Handler.html_path = html_path
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(
        f"[visualize] serving on http://{args.host}:{args.port}\n"
        f"  checkpoint: {args.checkpoint}\n"
        f"  input:      {args.input}\n"
        f"  frames:     {len(dataset)}\n"
        f"  mode:       {state.display_mode} (HandHeatmap available: "
        f"{visualizer._has_hand_heatmap_head()})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[visualize] shutting down...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
