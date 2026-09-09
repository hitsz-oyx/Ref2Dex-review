"""Diagnose OICM object effect induced by a pure-Inspire decoder rollout.

For every rollout transition ``state_t -> state_{t+1}``, this diagnostic builds
the predicted Inspire hand point flow and forwards it through the frozen OICM.
It records OICM's own ``pred_obj_flow``.  The actual object point flow from
``t`` to ``t+1`` is computed separately for display-only comparison; it is
never passed to the decoder or OICM.  The optional Viser view supports both
teacher-forced and recursive rollout modes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.base import load_config
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel

from ...visualize_inspire_test import (
    InspireRolloutEngine,
    InspireTestSequence,
    _load_model,
    _load_sequence,
    _normals_world_to_frame,
    _points_world_to_frame,
    _resolve,
)


MODIFICATION_VERSION = "V1.1.4"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _surface_state(
    *,
    urdf_model: InspireUrdfModel,
    kinematics: Any,
    finger_q: np.ndarray,
    wrist_pose: np.ndarray,
    sampled_points: np.ndarray,
    sampled_normals: np.ndarray,
    sampled_visual_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """FK the same 1538 sampled surface points used by the Inspire cache."""
    links = kinematics.link_transforms_from_state(finger_q, wrist_pose)
    points = np.empty_like(sampled_points, dtype=np.float32)
    normals = np.empty_like(sampled_points, dtype=np.float32)
    for visual_id in np.unique(sampled_visual_ids):
        mask = sampled_visual_ids == visual_id
        visual = urdf_model.visuals[int(visual_id)]
        transform = links[visual.link] @ visual.local_transform
        points[mask] = sampled_points[mask] @ transform[:3, :3].T + transform[:3, 3]
        normals[mask] = sampled_normals[mask] @ transform[:3, :3].T
    normals /= np.clip(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8, None)
    return points, normals


def _effect_colors(magnitude_mm: np.ndarray, upper_mm: float) -> np.ndarray:
    """Small blue-yellow-red map without adding a plotting dependency."""
    ratio = np.clip(np.asarray(magnitude_mm, dtype=np.float32) / max(float(upper_mm), 1e-6), 0.0, 1.0)
    low = np.asarray([40.0, 110.0, 235.0], dtype=np.float32)
    mid = np.asarray([245.0, 220.0, 45.0], dtype=np.float32)
    high = np.asarray([230.0, 45.0, 35.0], dtype=np.float32)
    colors = np.empty((len(ratio), 3), dtype=np.float32)
    first = ratio <= 0.5
    alpha = ratio[first, None] * 2.0
    colors[first] = low[None] * (1.0 - alpha) + mid[None] * alpha
    alpha = (ratio[~first, None] - 0.5) * 2.0
    colors[~first] = mid[None] * (1.0 - alpha) + high[None] * alpha
    return colors.astype(np.uint8)


def _effective_object_flow(raw_flow: np.ndarray, sample_valid: bool) -> np.ndarray:
    """Apply OICM's semantic validity contract to its always-finite raw head."""
    raw_flow = np.asarray(raw_flow, dtype=np.float32)
    return raw_flow.copy() if bool(sample_valid) else np.zeros_like(raw_flow)


class EffectDiagnostic:
    """Roll out CmDecoderv2 and run its generated hand flow through OICM."""

    def __init__(
        self,
        *,
        sequence: InspireTestSequence,
        decoder: torch.nn.Module,
        kinematics: Any,
        render_surface: Any,
        device: torch.device,
        cfg: Any,
    ) -> None:
        self.sequence = sequence
        self.decoder = decoder
        self.oicm = decoder.oicm
        self.kinematics = kinematics
        self.device = device
        self.window_size = int(cfg.meta.window_size)
        self.num_obj_points = int(cfg.meta.num_obj_points)
        self.surface_model = InspireUrdfModel(_resolve(str(cfg.data.urdf_path)))
        self.sampled_points, self.sampled_normals, self.sampled_visual_ids = self.surface_model.surface_samples(
            int(cfg.meta.num_hand_points), 2024
        )
        self.rollout = InspireRolloutEngine(
            sequence=sequence,
            model=decoder,
            kinematics=kinematics,
            surface=render_surface,
            device=device,
            window_size=self.window_size,
            num_obj_points=self.num_obj_points,
        )
        self.teacher_cache: dict[int, dict[str, Any]] = {}

    def _object_selection(self, frame: int) -> np.ndarray:
        rng = np.random.default_rng(2024 + int(frame) * 131)
        return rng.choice(self.sequence.object_points.shape[1], size=self.num_obj_points, replace=False)

    def _object_input(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Use only current object geometry; future object pose is never read."""
        pose = np.asarray(self.sequence.object_pose[frame], dtype=np.float32)
        selected = self._object_selection(frame)
        points_world = np.asarray(self.sequence.object_points[frame, selected], dtype=np.float32)
        normals_world = np.asarray(self.sequence.object_normals[frame, selected], dtype=np.float32)
        points_object = _points_world_to_frame(points_world, pose)
        normals_object = _normals_world_to_frame(normals_world, pose)
        return points_world, normals_world, points_object, normals_object, selected

    def _gt_object_flow(self, frame: int, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the real object point flow for display-only comparison.

        The decoder/OICM never receives this tensor.  Corresponding points are
        selected with the same deterministic indices at frame ``t`` and then
        transformed by the actual Dexplore object trajectory from ``t`` to
        ``t+1``.  World vectors are also expressed in the current object frame
        so they share the OICM prediction coordinate contract.
        """
        current = np.asarray(self.sequence.object_points[frame, selected], dtype=np.float32)
        future = np.asarray(self.sequence.object_points[frame + 1, selected], dtype=np.float32)
        flow_world = future - current
        pose = np.asarray(self.sequence.object_pose[frame], dtype=np.float32)
        flow_object = flow_world @ pose[:3, :3]
        return flow_world, flow_object

    def _forward_effect(self, frame: int, current: dict[str, Any], future: dict[str, Any]) -> dict[str, Any]:
        object_world, _, object_points, object_normals, selected = self._object_input(frame)
        gt_object_flow_world, gt_object_flow_object = self._gt_object_flow(frame, selected)
        current_points_world, current_normals_world = _surface_state(
            urdf_model=self.surface_model,
            kinematics=self.kinematics,
            finger_q=np.asarray(current["finger_q"], dtype=np.float64),
            wrist_pose=np.asarray(current["wrist"], dtype=np.float64),
            sampled_points=self.sampled_points,
            sampled_normals=self.sampled_normals,
            sampled_visual_ids=self.sampled_visual_ids,
        )
        future_points_world, _ = _surface_state(
            urdf_model=self.surface_model,
            kinematics=self.kinematics,
            finger_q=np.asarray(future["finger_q"], dtype=np.float64),
            wrist_pose=np.asarray(future["wrist"], dtype=np.float64),
            sampled_points=self.sampled_points,
            sampled_normals=self.sampled_normals,
            sampled_visual_ids=self.sampled_visual_ids,
        )
        pose = np.asarray(self.sequence.object_pose[frame], dtype=np.float32)
        hand_points = _points_world_to_frame(current_points_world, pose)
        hand_normals = _normals_world_to_frame(current_normals_world, pose)
        future_points = _points_world_to_frame(future_points_world, pose)
        hand_flow = future_points - hand_points
        batch = {
            "obj_points": torch.from_numpy(object_points).unsqueeze(0).to(self.device),
            "obj_normals": torch.from_numpy(object_normals).unsqueeze(0).to(self.device),
            "obj_valid_mask": torch.ones((1, self.num_obj_points), dtype=torch.bool, device=self.device),
            "hand_points": torch.from_numpy(hand_points).unsqueeze(0).to(self.device),
            "hand_normals": torch.from_numpy(hand_normals).unsqueeze(0).to(self.device),
            "hand_flow": torch.from_numpy(hand_flow).unsqueeze(0).to(self.device),
            "hand_valid_mask": torch.ones((1, hand_points.shape[0]), dtype=torch.bool, device=self.device),
        }
        with torch.inference_mode():
            output = self.oicm(batch)
        predicted_object_flow_object = output["pred_obj_flow"][0].detach().cpu().numpy().astype(np.float32)
        oicm_sample_valid = bool(output["sample_valid"][0].item())
        effective_object_flow_object = _effective_object_flow(predicted_object_flow_object, oicm_sample_valid)
        rotation = pose[:3, :3]
        predicted_object_flow_world = predicted_object_flow_object @ rotation.T
        effective_object_flow_world = effective_object_flow_object @ rotation.T
        effect_magnitude_mm = np.linalg.norm(predicted_object_flow_object, axis=-1) * 1000.0
        effective_magnitude_mm = np.linalg.norm(effective_object_flow_object, axis=-1) * 1000.0
        nearest_mm = float(cKDTree(np.asarray(self.sequence.object_points[frame], dtype=np.float32)).query(current_points_world, k=1)[0].min() * 1000.0)
        hand_flow_magnitude_mm = np.linalg.norm(hand_flow, axis=-1) * 1000.0
        decoder_valid = current.get("cm_valid")
        return {
            "object_points_world": object_world,
            "object_pose_world": pose,
            "hand_points_world": current_points_world,
            "hand_flow_world": future_points_world - current_points_world,
            "pred_obj_flow_object": predicted_object_flow_object,
            "pred_obj_flow_world": predicted_object_flow_world,
            "effective_obj_flow_object": effective_object_flow_object,
            "effective_obj_flow_world": effective_object_flow_world,
            "gt_obj_flow_object": gt_object_flow_object,
            "gt_obj_flow_world": gt_object_flow_world,
            "cm_tokens": output["cm_tokens"][0].detach().cpu().numpy().astype(np.float32),
            "cm_anchor_pos_object": output["cm_anchor_pos"][0].detach().cpu().numpy().astype(np.float32),
            "oicm_sample_valid": oicm_sample_valid,
            "oicm_sampled_active_count": int(output["sampled_active_count"][0].item()),
            "min_hand_object_distance_mm": nearest_mm,
            "hand_flow_rms_mm": float(np.sqrt(np.mean(hand_flow_magnitude_mm ** 2))),
            "effect_rms_mm": float(np.sqrt(np.mean(effect_magnitude_mm ** 2))),
            "effect_mean_mm": float(np.mean(effect_magnitude_mm)),
            "effect_max_mm": float(np.max(effect_magnitude_mm)),
            "effective_effect_rms_mm": float(np.sqrt(np.mean(effective_magnitude_mm ** 2))),
            "gt_effect_rms_mm": float(np.sqrt(np.mean(np.linalg.norm(gt_object_flow_object, axis=-1) ** 2)) * 1000.0),
            "pred_gt_effect_epe_mm": float(np.mean(np.linalg.norm(effective_object_flow_object - gt_object_flow_object, axis=-1)) * 1000.0),
            "decoder_cm_valid": np.asarray(decoder_valid if decoder_valid is not None else np.zeros(self.window_size), dtype=bool),
        }

    def run(self, max_steps: int | None = None) -> dict[str, np.ndarray]:
        max_frame = self.sequence.frame_count - self.window_size
        step_count = max(0, max_frame)
        if max_steps is not None:
            step_count = min(step_count, int(max_steps))
        self.rollout.start(0)
        records: list[dict[str, Any]] = []
        for frame in range(step_count):
            current = self.rollout.ensure(frame)
            future = self.rollout.ensure(frame + 1)
            if current is None or future is None:
                break
            records.append(self._forward_effect(frame, current, future))
        if not records:
            raise RuntimeError("No rollout transitions were generated")
        return {
            "frame": np.asarray(list(range(len(records))), dtype=np.int32),
            "source_frame_id": np.asarray([self.sequence.source_frame[index] for index in range(len(records))], dtype=np.int64),
            "object_points_world": np.stack([r["object_points_world"] for r in records]),
            "object_pose_world": np.stack([r["object_pose_world"] for r in records]),
            "hand_points_world": np.stack([r["hand_points_world"] for r in records]),
            "hand_flow_world": np.stack([r["hand_flow_world"] for r in records]),
            "pred_obj_flow_object": np.stack([r["pred_obj_flow_object"] for r in records]),
            "pred_obj_flow_world": np.stack([r["pred_obj_flow_world"] for r in records]),
            "effective_obj_flow_object": np.stack([r["effective_obj_flow_object"] for r in records]),
            "effective_obj_flow_world": np.stack([r["effective_obj_flow_world"] for r in records]),
            "gt_obj_flow_object": np.stack([r["gt_obj_flow_object"] for r in records]),
            "gt_obj_flow_world": np.stack([r["gt_obj_flow_world"] for r in records]),
            "cm_tokens": np.stack([r["cm_tokens"] for r in records]),
            "cm_anchor_pos_object": np.stack([r["cm_anchor_pos_object"] for r in records]),
            "oicm_sample_valid": np.asarray([r["oicm_sample_valid"] for r in records], dtype=bool),
            "oicm_sampled_active_count": np.asarray([r["oicm_sampled_active_count"] for r in records], dtype=np.int32),
            "min_hand_object_distance_mm": np.asarray([r["min_hand_object_distance_mm"] for r in records], dtype=np.float32),
            "hand_flow_rms_mm": np.asarray([r["hand_flow_rms_mm"] for r in records], dtype=np.float32),
            "effect_rms_mm": np.asarray([r["effect_rms_mm"] for r in records], dtype=np.float32),
            "effect_mean_mm": np.asarray([r["effect_mean_mm"] for r in records], dtype=np.float32),
            "effect_max_mm": np.asarray([r["effect_max_mm"] for r in records], dtype=np.float32),
            "effective_effect_rms_mm": np.asarray([r["effective_effect_rms_mm"] for r in records], dtype=np.float32),
            "gt_effect_rms_mm": np.asarray([r["gt_effect_rms_mm"] for r in records], dtype=np.float32),
            "pred_gt_effect_epe_mm": np.asarray([r["pred_gt_effect_epe_mm"] for r in records], dtype=np.float32),
            "decoder_cm_valid": np.stack([r["decoder_cm_valid"] for r in records]),
        }

    def start_rollout(self, frame: int) -> None:
        frame = int(frame)
        max_frame = self.sequence.frame_count - self.window_size - 1
        if frame < 0 or frame > max_frame:
            raise ValueError(f"rollout 起始帧必须在 [0, {max_frame}]，收到 {frame}")
        self.rollout.start(frame)

    def teacher_record(self, frame: int) -> dict[str, Any] | None:
        """Predict one step with the ground-truth Inspire state at frame t."""
        frame = int(frame)
        max_frame = self.sequence.frame_count - self.window_size - 1
        if frame < 0 or frame > max_frame:
            return None
        if frame not in self.teacher_cache:
            # InspireRolloutEngine.start uses the GT Inspire state as handoff;
            # one ensure step therefore implements teacher forcing for t -> t+1.
            self.rollout.start(frame)
            current = self.rollout.ensure(frame)
            future = self.rollout.ensure(frame + 1)
            if current is None or future is None:
                return None
            self.teacher_cache[frame] = self._forward_effect(frame, current, future)
        return self.teacher_cache[frame]

    def rollout_record(self, frame: int) -> dict[str, Any] | None:
        """Get one step from the currently selected recursive rollout handoff."""
        frame = int(frame)
        handoff = self.rollout.handoff_frame
        if handoff is None or frame < int(handoff):
            return None
        current = self.rollout.ensure(frame)
        future = self.rollout.ensure(frame + 1)
        if current is None or future is None:
            return None
        return self._forward_effect(frame, current, future)

    def interactive_record(self, frame: int, mode: str) -> dict[str, Any] | None:
        if mode == "teacherforced":
            return self.teacher_record(frame)
        return self.rollout_record(frame)


def _summary(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    distance = arrays["min_hand_object_distance_mm"]
    effect = arrays["effect_rms_mm"]
    effective = arrays["effective_effect_rms_mm"]
    valid = arrays["oicm_sample_valid"]
    gt_effect = arrays.get("gt_effect_rms_mm")
    pred_gt_epe = arrays.get("pred_gt_effect_epe_mm")
    far = distance > 50.0
    near = ~far

    def group(mask: np.ndarray) -> dict[str, Any]:
        values = effect[mask]
        effective_values = effective[mask]
        return {
            "frames": int(mask.sum()),
            "valid_ratio": float(valid[mask].mean()) if mask.any() else None,
            "effect_rms_mean_mm": float(values.mean()) if values.size else None,
            "effect_rms_median_mm": float(np.median(values)) if values.size else None,
            "effect_rms_max_mm": float(values.max()) if values.size else None,
            "effective_effect_rms_mean_mm": float(effective_values.mean()) if effective_values.size else None,
            "effective_effect_rms_max_mm": float(effective_values.max()) if effective_values.size else None,
            "distance_min_mm": float(distance[mask].min()) if mask.any() else None,
            "distance_max_mm": float(distance[mask].max()) if mask.any() else None,
        }

    summary = {
        "frames": int(len(effect)),
        "far_gt_50mm": group(far),
        "near_le_50mm": group(near),
        "overall_effect_rms_mean_mm": float(effect.mean()),
        "overall_effect_rms_median_mm": float(np.median(effect)),
        "overall_effect_rms_max_mm": float(effect.max()),
        "overall_effective_effect_rms_mean_mm": float(effective.mean()),
        "overall_effective_effect_rms_max_mm": float(effective.max()),
        "oicm_valid_ratio": float(valid.mean()),
    }
    if gt_effect is not None:
        summary.update({
            "overall_gt_effect_rms_mean_mm": float(gt_effect.mean()),
            "overall_gt_effect_rms_max_mm": float(gt_effect.max()),
            "gt_object_flow_display_only": True,
        })
    if pred_gt_epe is not None:
        summary["overall_pred_gt_effect_epe_mm"] = float(pred_gt_epe.mean())
    return summary


def _serve(arrays: dict[str, np.ndarray], *, port: int, fps: float) -> None:
    import viser

    server = viser.ViserServer(host="0.0.0.0", port=int(port))
    frame = server.gui.add_slider("Frame", min=0, max=len(arrays["frame"]) - 1, step=1, initial_value=0)
    frame_step = server.gui.add_slider("Jump size (frames)", min=1, max=30, step=1, initial_value=1)
    point_size = server.gui.add_slider("Point size (m)", min=0.001, max=0.020, step=0.001, initial_value=0.004)
    effect_scale = server.gui.add_slider("Effect color scale (mm)", min=1.0, max=50.0, step=1.0, initial_value=10.0)
    vector_scale = server.gui.add_slider("Vector scale", min=0.0, max=10.0, step=0.5, initial_value=3.0)
    effect_mode = server.gui.add_dropdown("Effect display", options=("effective", "raw"), initial_value="effective")
    previous = server.gui.add_button("Previous jump")
    following = server.gui.add_button("Next jump")
    play = server.gui.add_button("Play")
    stop = server.gui.add_button("Stop")
    status = server.gui.add_markdown("")
    state = {"playing": False}
    handles: list[Any] = []

    def render() -> None:
        nonlocal handles
        index = int(frame.value)
        points = arrays["object_points_world"][index]
        flow_key = "effective_obj_flow_world" if str(effect_mode.value) == "effective" else "pred_obj_flow_world"
        flow = arrays[flow_key][index]
        magnitude = np.linalg.norm(flow, axis=-1) * 1000.0
        colors = _effect_colors(magnitude, float(effect_scale.value))
        hand = arrays["hand_points_world"][index]
        hand_flow = arrays["hand_flow_world"][index]
        size = float(point_size.value)
        scale = float(vector_scale.value)
        object_indices = np.arange(0, len(points), 4)
        hand_indices = np.arange(0, len(hand), 12)
        object_segments = np.stack(
            [points[object_indices], points[object_indices] + flow[object_indices] * scale], axis=1
        )
        object_segment_colors = np.repeat(colors[object_indices, None, :], 2, axis=1)
        hand_segments = np.stack(
            [hand[hand_indices], hand[hand_indices] + hand_flow[hand_indices] * scale], axis=1
        )
        with server.atomic():
            for handle in handles:
                handle.remove()
            handles = [
                server.scene.add_point_cloud("/world/object_effect", points, colors=colors, point_size=size),
                server.scene.add_point_cloud("/world/inspire_rollout", hand, colors=(235, 65, 55), point_size=size),
                server.scene.add_line_segments(
                    "/world/object_effect_vectors",
                    points=object_segments,
                    colors=object_segment_colors,
                    line_width=2.0,
                ),
                server.scene.add_line_segments(
                    "/world/inspire_rollout_flow",
                    points=hand_segments,
                    colors=(210, 60, 220),
                    line_width=1.5,
                ),
            ]
        valid = bool(arrays["oicm_sample_valid"][index])
        status.content = (
            f"**rollout effect diagnostic**  \nframe: **{index}/{len(arrays['frame']) - 1}**, "
            f"distance: **{arrays['min_hand_object_distance_mm'][index]:.1f} mm**, "
            f"hand flow RMS: **{arrays['hand_flow_rms_mm'][index]:.2f} mm**  \n"
            f"OICM valid: **{valid}**, sampled active: **{int(arrays['oicm_sampled_active_count'][index])}**, "
            f"raw effect RMS: **{arrays['effect_rms_mm'][index]:.3f} mm**, "
            f"effective effect RMS: **{arrays['effective_effect_rms_mm'][index]:.3f} mm**, "
            f"max: **{arrays['effect_max_mm'][index]:.3f} mm**  \n"
            f"colors: **{effect_mode.value}** effect (blue → yellow → red); no future object GT is used."
        )

    def jump(direction: int) -> None:
        state["playing"] = False
        frame.value = max(0, min(len(arrays["frame"]) - 1, int(frame.value) + int(direction) * int(frame_step.value)))

    frame.on_update(lambda _: render())
    frame_step.on_update(lambda _: render())
    point_size.on_update(lambda _: render())
    effect_scale.on_update(lambda _: render())
    vector_scale.on_update(lambda _: render())
    effect_mode.on_update(lambda _: render())
    previous.on_click(lambda _: jump(-1))
    following.on_click(lambda _: jump(1))
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    render()
    print(f"Viser Inspire rollout effect viewer: http://localhost:{port}", flush=True)
    try:
        while True:
            if state["playing"]:
                frame.value = (int(frame.value) + 1) % len(arrays["frame"])
            time.sleep(1.0 / max(float(fps), 1e-3))
    except KeyboardInterrupt:
        state["playing"] = False
        raise


def _serve_interactive(diagnostic: EffectDiagnostic, *, port: int, fps: float) -> None:
    """Serve a Chinese interactive teacher-forcing/rollout effect viewer."""
    import viser

    sequence = diagnostic.sequence
    max_frame = sequence.frame_count - diagnostic.window_size - 1
    server = viser.ViserServer(host="0.0.0.0", port=int(port))
    mode = server.gui.add_dropdown("执行模式", options=("教师强制", "递归Rollout"), initial_value="递归Rollout")
    effect_display = server.gui.add_dropdown("Effect显示", options=("预测+GT", "仅预测", "仅GT"), initial_value="预测+GT")
    prediction_type = server.gui.add_dropdown("预测类型", options=("合同有效预测", "Raw预测"), initial_value="合同有效预测")
    frame = server.gui.add_slider("帧", min=0, max=max_frame, step=1, initial_value=0)
    frame_step = server.gui.add_slider("跳帧步长", min=1, max=30, step=1, initial_value=1)
    point_size = server.gui.add_slider("点大小（米）", min=0.001, max=0.020, step=0.001, initial_value=0.004)
    effect_scale = server.gui.add_slider("颜色上限（毫米）", min=1.0, max=100.0, step=1.0, initial_value=10.0)
    vector_scale = server.gui.add_slider("向量显示倍率", min=0.0, max=10.0, step=0.5, initial_value=3.0)
    restart = server.gui.add_button("从当前帧开始 Rollout")
    previous = server.gui.add_button("上一跳")
    following = server.gui.add_button("下一跳")
    play = server.gui.add_button("播放")
    stop = server.gui.add_button("停止")
    status = server.gui.add_markdown("")
    server.gui.add_markdown(
        "### 说明\n"
        "**教师强制**：每帧用真实 Inspire 状态预测下一步。  \n"
        "**递归 Rollout**：点击“从当前帧开始 Rollout”后，当前帧只初始化一次，后续递归使用预测状态。  \n"
        "红色/橙色线 = 预测 object effect；绿色线 = 真实轨迹 GT effect；灰色点 = 当前物体点。  \n"
        "GT effect 只用于界面对照，不会输入 decoder 或 OICM。"
    )
    state: dict[str, Any] = {"mode": "rollout", "playing": False}
    handles: list[Any] = []
    lock = threading.RLock()

    def _mode_value() -> str:
        return "teacherforced" if str(mode.value) == "教师强制" else "rollout"

    def _current_record(index: int) -> dict[str, Any] | None:
        return diagnostic.interactive_record(index, _mode_value())

    def render() -> None:
        nonlocal handles
        index = int(frame.value)
        record = _current_record(index)
        selected = diagnostic._object_selection(index)
        object_points = np.asarray(sequence.object_points[index, selected], dtype=np.float32)
        gt_flow_world, _ = diagnostic._gt_object_flow(index, selected)
        size = float(point_size.value)
        scale = float(vector_scale.value)
        display_mode = str(effect_display.value)
        pred_key = "pred_obj_flow_world" if str(prediction_type.value) == "Raw预测" else "effective_obj_flow_world"
        pred_flow = None if record is None else np.asarray(record[pred_key], dtype=np.float32)
        gt_flow = np.asarray(gt_flow_world, dtype=np.float32)
        object_indices = np.arange(0, len(object_points), 4)
        gt_segments = np.stack(
            [object_points[object_indices], object_points[object_indices] + gt_flow[object_indices] * scale], axis=1
        )
        pred_segments = None
        if pred_flow is not None:
            pred_segments = np.stack(
                [object_points[object_indices], object_points[object_indices] + pred_flow[object_indices] * scale], axis=1
            )
        hand = np.asarray(sequence.hand_points[index], dtype=np.float32) if record is None else np.asarray(record["hand_points_world"], dtype=np.float32)
        hand_indices = np.arange(0, len(hand), 12)
        hand_flow = None if record is None else np.asarray(record["hand_flow_world"], dtype=np.float32)
        with server.atomic():
            for handle in handles:
                handle.remove()
            clouds: list[Any] = [
                server.scene.add_point_cloud("/world/object_points", object_points, colors=(155, 155, 155), point_size=size),
                server.scene.add_point_cloud("/world/inspire_rollout", hand, colors=(235, 65, 55), point_size=size),
            ]
            if hand_flow is not None:
                hand_segments = np.stack([hand[hand_indices], hand[hand_indices] + hand_flow[hand_indices] * scale], axis=1)
                clouds.append(server.scene.add_line_segments("/world/inspire_rollout_flow", points=hand_segments, colors=(210, 60, 220), line_width=1.5))
            if display_mode in {"预测+GT", "仅预测"} and pred_segments is not None:
                pred_magnitude = np.linalg.norm(pred_flow, axis=-1) * 1000.0
                pred_colors = _effect_colors(pred_magnitude, float(effect_scale.value))
                pred_color = np.repeat(pred_colors[object_indices, None, :], 2, axis=1)
                clouds.append(server.scene.add_line_segments("/world/predicted_effect", points=pred_segments, colors=pred_color, line_width=2.5))
            if display_mode in {"预测+GT", "仅GT"}:
                gt_magnitude = np.linalg.norm(gt_flow, axis=-1) * 1000.0
                gt_color = np.repeat(_effect_colors(gt_magnitude, float(effect_scale.value))[object_indices, None, :], 2, axis=1)
                clouds.append(server.scene.add_line_segments("/world/gt_effect", points=gt_segments, colors=gt_color, line_width=2.0))
            handles = clouds

        distance = float(cKDTree(np.asarray(sequence.object_points[index], dtype=np.float32)).query(hand, k=1)[0].min() * 1000.0)
        gt_rms = float(np.sqrt(np.mean(np.linalg.norm(gt_flow, axis=-1) ** 2)) * 1000.0)
        hand_text = "当前帧尚未产生 decoder 预测"
        pred_text = "预测 effect：—"
        valid_text = "OICM valid：—"
        if record is not None:
            valid = bool(record["oicm_sample_valid"])
            pred_rms = float(record["effective_effect_rms_mm"] if str(prediction_type.value) != "Raw预测" else record["effect_rms_mm"])
            pred_text = f"预测 effect RMS：**{pred_rms:.3f} mm**；预测-GT EPE：**{record['pred_gt_effect_epe_mm']:.3f} mm**"
            valid_text = f"OICM valid：**{valid}**；有效采样点：**{int(record['oicm_sampled_active_count'])}**"
            hand_text = f"hand flow RMS：**{record['hand_flow_rms_mm']:.3f} mm**"
        handoff = diagnostic.rollout.handoff_frame
        handoff_text = "—" if handoff is None else str(handoff)
        status.content = (
            f"**{sequence.id}**  \n模式：**{mode.value}**；当前帧：**{index}/{max_frame}**；Rollout 起始帧：**{handoff_text}**  \n"
            f"物体距离：**{distance:.2f} mm**；{hand_text}  \n"
            f"{pred_text}  \nGT effect RMS：**{gt_rms:.3f} mm**；{valid_text}  \n"
            "图例：**红/橙 = 预测 effect，绿色/蓝绿色 = GT effect，紫色 = Inspire hand flow**。"
        )

    def render_locked() -> None:
        with lock:
            render()

    def on_mode_change(_: Any) -> None:
        with lock:
            state["mode"] = _mode_value()
            if state["mode"] == "rollout":
                try:
                    diagnostic.start_rollout(int(frame.value))
                except ValueError:
                    pass
            render()

    def on_restart(_: Any) -> None:
        with lock:
            state["mode"] = "rollout"
            if str(mode.value) != "递归Rollout":
                mode.value = "递归Rollout"
            diagnostic.start_rollout(int(frame.value))
            render()

    def on_jump(direction: int) -> None:
        with lock:
            state["playing"] = False
            target = max(0, min(max_frame, int(frame.value) + int(direction) * int(frame_step.value)))
            frame.value = target

    mode.on_update(on_mode_change)
    frame.on_update(lambda _: render_locked())
    frame_step.on_update(lambda _: render_locked())
    point_size.on_update(lambda _: render_locked())
    effect_scale.on_update(lambda _: render_locked())
    vector_scale.on_update(lambda _: render_locked())
    effect_display.on_update(lambda _: render_locked())
    prediction_type.on_update(lambda _: render_locked())
    restart.on_click(on_restart)
    previous.on_click(lambda _: on_jump(-1))
    following.on_click(lambda _: on_jump(1))
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    diagnostic.start_rollout(0)
    render_locked()
    print(f"Viser Inspire effect viewer: http://localhost:{port}", flush=True)
    try:
        while True:
            if state["playing"]:
                frame.value = (int(frame.value) + 1) % (max_frame + 1)
            time.sleep(1.0 / max(float(fps), 1e-3))
    except KeyboardInterrupt:
        state["playing"] = False
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml")
    parser.add_argument("--checkpoint", default="outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_203046/checkpoints/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", default="s1/camera_takepicture_3_Retake")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--rl-root", default="data/processed_data/inspire_rl")
    parser.add_argument("--parent-root", default="data/processed_data/cm_object_v2_surface512_object_pose_20260830")
    parser.add_argument(
        "--output-root",
        default="src/task/CmDecoderv2/research/inspire_rollout_effect/output",
    )
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    sequence = _load_sequence(
        cfg,
        args.sequence,
        int(args.sequence_index),
        rl_root=_resolve(args.rl_root),
        parent_root=_resolve(args.parent_root),
    )
    checkpoint_path = _resolve(args.checkpoint)
    decoder, kinematics, render_surface = _load_model(cfg, checkpoint_path, device)
    diagnostic = EffectDiagnostic(
        sequence=sequence,
        decoder=decoder,
        kinematics=kinematics,
        render_surface=render_surface,
        device=device,
        cfg=cfg,
    )
    arrays = diagnostic.run(max_steps=args.max_steps)

    run_id = f"cmdecoderv2-inspire-effect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output = _resolve(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "effect.npz", **arrays)
    summary = _summary(arrays)
    (output / "effect_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_name": "ref2dex_cmdecoderv2_inspire_rollout_effect_v1",
        "task": "CmDecoderv2",
        "activity_id": args.activity_id,
        "run_id": run_id,
        "run_status": "RUNNING" if args.serve else "COMPLETED",
        "conclusion": "INCONCLUSIVE",
        "conclusion_scope": "rollout/teacher-forced Inspire hand flow forwarded through frozen OICM; GT object flow is display-only",
        "modification_version": MODIFICATION_VERSION,
        "operation_category": ["diagnostic", "experiment", "operation"],
        "created_at": _now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "frozen_oicm_checkpoint": str(decoder.oicm_checkpoint),
        "frozen_oicm_checkpoint_sha256": str(decoder.oicm_checkpoint_sha256),
        "sequence_id": sequence.id,
        "frame_count": int(len(arrays["frame"])),
        "window_size": int(cfg.meta.window_size),
        "coordinate_frame": "object_pose_t",
        "effect_definition": "raw=OICM pred_obj_flow from decoder-generated Inspire hand flow; effective=raw masked to zero when OICM sample_valid=false; gt=actual corresponding object point flow from t to t+1, display-only",
        "gt_object_flow_display_only": True,
        "outputs": {
            "effect": str(output / "effect.npz"),
            "summary": str(output / "effect_summary.json"),
            "run_manifest": str(output / "run_manifest.json"),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_status": manifest["run_status"], "output": str(output), "summary": summary}, ensure_ascii=False), flush=True)
    if args.serve:
        try:
            _serve_interactive(diagnostic, port=int(args.port), fps=float(args.fps))
        except KeyboardInterrupt:
            # Keep the manifest truthful when the user stops the interactive
            # viewer with Ctrl-C.  External hard kills cannot run this block,
            # so callers should reconcile those runs as STOPPED separately.
            manifest["run_status"] = "STOPPED"
            manifest["stopped_at"] = _now()
            manifest["stop_reason"] = "KeyboardInterrupt"
            (output / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            raise


if __name__ == "__main__":
    main()
