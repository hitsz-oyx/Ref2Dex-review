"""Forward MANO-source decoder rollout hand flow through frozen OICM.

The rollout starts from the formal MANO qualitative-only source using the
existing viewer contract: zero Inspire finger-q plus the MANO wrist at the
handoff frame, followed by recursive decoder predictions.  For each predicted
Inspire transition, the generated hand flow is sent to OICM and its object
effect is recorded without reading future object state or object-flow GT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.base import load_config
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel

from ...visualize_mano import (
    GrabManoSequence,
    RolloutEngine,
    _load_model,
    _load_sequence,
    _resolve,
)
from ...dataset import _normals_world_to_frame, _points_world_to_frame
from ..inspire_rollout_effect.run import (
    _effect_colors,
    _effective_object_flow,
    _summary,
    _surface_state,
)


WORK_VERSION = "V1.1.5"


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ManoRolloutEffectDiagnostic:
    """Run recursive MANO-source rollout and evaluate its generated effect."""

    def __init__(
        self,
        *,
        sequence: GrabManoSequence,
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
        self.rollout = RolloutEngine(
            sequence=sequence,
            model=decoder,
            kinematics=kinematics,
            surface=render_surface,
            device=device,
            window_size=self.window_size,
            num_obj_points=self.num_obj_points,
        )

    def _object_input(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Use only the current MANO/GRAB object geometry and pose."""
        pose = np.asarray(self.sequence.object_pose[frame], dtype=np.float32)
        rng = np.random.default_rng(2024 + int(frame) * 131)
        selected = rng.choice(self.sequence.object_points.shape[1], size=self.num_obj_points, replace=False)
        points_world = np.asarray(self.sequence.object_points[frame, selected], dtype=np.float32)
        normals_world = np.asarray(self.sequence.object_normals[frame, selected], dtype=np.float32)
        points_object = _points_world_to_frame(points_world, pose)
        normals_object = _normals_world_to_frame(normals_world, pose)
        return points_world, normals_world, points_object, normals_object

    def _forward_effect(self, frame: int, current: dict[str, Any], future: dict[str, Any]) -> dict[str, Any]:
        object_world, _, object_points, object_normals = self._object_input(frame)
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
        raw_object_flow = output["pred_obj_flow"][0].detach().cpu().numpy().astype(np.float32)
        sample_valid = bool(output["sample_valid"][0].item())
        effective_object_flow = _effective_object_flow(raw_object_flow, sample_valid)
        rotation = pose[:3, :3]
        raw_object_flow_world = raw_object_flow @ rotation.T
        effective_object_flow_world = effective_object_flow @ rotation.T
        raw_magnitude_mm = np.linalg.norm(raw_object_flow, axis=-1) * 1000.0
        effective_magnitude_mm = np.linalg.norm(effective_object_flow, axis=-1) * 1000.0
        nearest_mm = float(
            cKDTree(np.asarray(self.sequence.object_points[frame], dtype=np.float32))
            .query(current_points_world, k=1)[0]
            .min()
            * 1000.0
        )
        hand_flow_magnitude_mm = np.linalg.norm(hand_flow, axis=-1) * 1000.0
        decoder_cm_valid = future.get("cm_valid")
        if decoder_cm_valid is None:
            decoder_cm_valid = np.zeros(self.window_size, dtype=bool)
        return {
            "object_points_world": object_world,
            "object_pose_world": pose,
            "mano_hand_points_world": np.asarray(self.sequence.hand_points[frame], dtype=np.float32),
            "hand_points_world": current_points_world,
            "hand_flow_world": future_points_world - current_points_world,
            "pred_obj_flow_object": raw_object_flow,
            "pred_obj_flow_world": raw_object_flow_world,
            "effective_obj_flow_object": effective_object_flow,
            "effective_obj_flow_world": effective_object_flow_world,
            "cm_tokens": output["cm_tokens"][0].detach().cpu().numpy().astype(np.float32),
            "cm_anchor_pos_object": output["cm_anchor_pos"][0].detach().cpu().numpy().astype(np.float32),
            "oicm_sample_valid": sample_valid,
            "oicm_sampled_active_count": int(output["sampled_active_count"][0].item()),
            "min_hand_object_distance_mm": nearest_mm,
            "hand_flow_rms_mm": float(np.sqrt(np.mean(hand_flow_magnitude_mm**2))),
            "effect_rms_mm": float(np.sqrt(np.mean(raw_magnitude_mm**2))),
            "effect_mean_mm": float(np.mean(raw_magnitude_mm)),
            "effect_max_mm": float(np.max(raw_magnitude_mm)),
            "effective_effect_rms_mm": float(np.sqrt(np.mean(effective_magnitude_mm**2))),
            "decoder_cm_valid": np.asarray(decoder_cm_valid, dtype=bool),
            "rollout_finger_q": np.asarray(current["finger_q"], dtype=np.float32),
            "rollout_wrist_pose_world": np.asarray(current["wrist"], dtype=np.float32),
        }

    def run(self, max_steps: int | None = None) -> dict[str, np.ndarray]:
        step_count = max(0, self.sequence.frame_count - self.window_size)
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
            raise RuntimeError("No MANO rollout transitions were generated")
        return {
            "frame": np.arange(len(records), dtype=np.int32),
            "source_frame_id": np.asarray([self.sequence.source_frame[i] for i in range(len(records))], dtype=np.int64),
            "object_points_world": np.stack([record["object_points_world"] for record in records]),
            "object_pose_world": np.stack([record["object_pose_world"] for record in records]),
            "mano_hand_points_world": np.stack([record["mano_hand_points_world"] for record in records]),
            "hand_points_world": np.stack([record["hand_points_world"] for record in records]),
            "hand_flow_world": np.stack([record["hand_flow_world"] for record in records]),
            "pred_obj_flow_object": np.stack([record["pred_obj_flow_object"] for record in records]),
            "pred_obj_flow_world": np.stack([record["pred_obj_flow_world"] for record in records]),
            "effective_obj_flow_object": np.stack([record["effective_obj_flow_object"] for record in records]),
            "effective_obj_flow_world": np.stack([record["effective_obj_flow_world"] for record in records]),
            "cm_tokens": np.stack([record["cm_tokens"] for record in records]),
            "cm_anchor_pos_object": np.stack([record["cm_anchor_pos_object"] for record in records]),
            "oicm_sample_valid": np.asarray([record["oicm_sample_valid"] for record in records], dtype=bool),
            "oicm_sampled_active_count": np.asarray([record["oicm_sampled_active_count"] for record in records], dtype=np.int32),
            "min_hand_object_distance_mm": np.asarray([record["min_hand_object_distance_mm"] for record in records], dtype=np.float32),
            "hand_flow_rms_mm": np.asarray([record["hand_flow_rms_mm"] for record in records], dtype=np.float32),
            "effect_rms_mm": np.asarray([record["effect_rms_mm"] for record in records], dtype=np.float32),
            "effect_mean_mm": np.asarray([record["effect_mean_mm"] for record in records], dtype=np.float32),
            "effect_max_mm": np.asarray([record["effect_max_mm"] for record in records], dtype=np.float32),
            "effective_effect_rms_mm": np.asarray([record["effective_effect_rms_mm"] for record in records], dtype=np.float32),
            "decoder_cm_valid": np.stack([record["decoder_cm_valid"] for record in records]),
            "rollout_finger_q": np.stack([record["rollout_finger_q"] for record in records]),
            "rollout_wrist_pose_world": np.stack([record["rollout_wrist_pose_world"] for record in records]),
        }


def _serve(arrays: dict[str, np.ndarray], *, port: int, fps: float) -> None:
    import viser

    server = viser.ViserServer(host="0.0.0.0", port=int(port))
    frame = server.gui.add_slider("Frame", min=0, max=len(arrays["frame"]) - 1, step=1, initial_value=0)
    frame_step = server.gui.add_slider("Jump size (frames)", min=1, max=30, step=1, initial_value=1)
    point_size = server.gui.add_slider("Point size (m)", min=0.001, max=0.020, step=0.001, initial_value=0.004)
    effect_scale = server.gui.add_slider("Effect color scale (mm)", min=1.0, max=50.0, step=1.0, initial_value=10.0)
    vector_scale = server.gui.add_slider("Vector scale", min=0.0, max=10.0, step=0.5, initial_value=3.0)
    hand_display = server.gui.add_dropdown("Hand display", options=("MANO GT", "rollout Inspire", "both"), initial_value="both")
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
        pred_hand = arrays["hand_points_world"][index]
        mano_hand = arrays["mano_hand_points_world"][index]
        hand_flow = arrays["hand_flow_world"][index]
        size = float(point_size.value)
        scale = float(vector_scale.value)
        object_indices = np.arange(0, len(points), 4)
        hand_indices = np.arange(0, len(pred_hand), 12)
        object_segments = np.stack([points[object_indices], points[object_indices] + flow[object_indices] * scale], axis=1)
        object_segment_colors = np.repeat(colors[object_indices, None, :], 2, axis=1)
        hand_segments = np.stack([pred_hand[hand_indices], pred_hand[hand_indices] + hand_flow[hand_indices] * scale], axis=1)
        with server.atomic():
            for handle in handles:
                handle.remove()
            handles = [
                server.scene.add_point_cloud("/world/object_effect", points, colors=colors, point_size=size),
                server.scene.add_line_segments("/world/object_effect_vectors", points=object_segments, colors=object_segment_colors, line_width=2.0),
                server.scene.add_line_segments("/world/inspire_rollout_flow", points=hand_segments, colors=(210, 60, 220), line_width=1.5),
            ]
            selected = str(hand_display.value)
            if selected in {"MANO GT", "both"}:
                handles.append(server.scene.add_point_cloud("/world/mano_gt", mano_hand, colors=(60, 130, 240), point_size=size))
            if selected in {"rollout Inspire", "both"}:
                handles.append(server.scene.add_point_cloud("/world/rollout_inspire", pred_hand, colors=(235, 65, 55), point_size=size))
        valid = bool(arrays["oicm_sample_valid"][index])
        decoder_valid = int(arrays["decoder_cm_valid"][index].sum())
        decoder_total = int(arrays["decoder_cm_valid"][index].size)
        status.content = (
            f"**MANO rollout effect diagnostic**  \nframe: **{index}/{len(arrays['frame']) - 1}**, "
            f"distance: **{arrays['min_hand_object_distance_mm'][index]:.1f} mm**, "
            f"hand flow RMS: **{arrays['hand_flow_rms_mm'][index]:.2f} mm**  \n"
            f"decoder Cm valid: **{decoder_valid}/{decoder_total}**, OICM valid: **{valid}**, "
            f"sampled active: **{int(arrays['oicm_sampled_active_count'][index])}**  \n"
            f"raw effect RMS: **{arrays['effect_rms_mm'][index]:.3f} mm**, "
            f"effective effect RMS: **{arrays['effective_effect_rms_mm'][index]:.3f} mm**, "
            f"max: **{arrays['effect_max_mm'][index]:.3f} mm**  \n"
            f"colors: **{effect_mode.value}** effect; MANO blue, rollout Inspire red; no future object GT is used."
        )

    def jump(direction: int) -> None:
        state["playing"] = False
        frame.value = max(0, min(len(arrays["frame"]) - 1, int(frame.value) + int(direction) * int(frame_step.value)))

    frame.on_update(lambda _: render())
    frame_step.on_update(lambda _: render())
    point_size.on_update(lambda _: render())
    effect_scale.on_update(lambda _: render())
    vector_scale.on_update(lambda _: render())
    hand_display.on_update(lambda _: render())
    effect_mode.on_update(lambda _: render())
    previous.on_click(lambda _: jump(-1))
    following.on_click(lambda _: jump(1))
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    render()
    print(f"Viser MANO rollout effect viewer: http://localhost:{port}", flush=True)
    try:
        while True:
            if state["playing"]:
                frame.value = (int(frame.value) + 1) % len(arrays["frame"])
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
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-root", default="src/task/CmDecoderv2/research/mano_rollout_effect/output")
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--port", type=int, default=8103)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    sequence_entry, index_path = _load_sequence(cfg, args.sequence, int(args.sequence_index))
    sequence = GrabManoSequence(sequence_entry)
    checkpoint_path = _resolve(args.checkpoint)
    decoder, kinematics, render_surface = _load_model(cfg, checkpoint_path, device)
    diagnostic = ManoRolloutEffectDiagnostic(
        sequence=sequence,
        decoder=decoder,
        kinematics=kinematics,
        render_surface=render_surface,
        device=device,
        cfg=cfg,
    )
    arrays = diagnostic.run(max_steps=args.max_steps)

    run_id = f"cmdecoderv2-mano-effect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output = _resolve(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "effect.npz", **arrays)
    summary = _summary(arrays)
    (output / "effect_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_name": "ref2dex_cmdecoderv2_mano_rollout_effect_v1",
        "task": "CmDecoderv2",
        "activity_id": args.activity_id,
        "run_id": run_id,
        "run_status": "RUNNING" if args.serve else "COMPLETED",
        "conclusion": "INCONCLUSIVE",
        "conclusion_scope": "MANO-source rollout-generated Inspire hand flow forwarded through frozen OICM; no future object GT comparison",
        "work_version": WORK_VERSION,
        "operation_category": ["diagnostic", "experiment", "operation"],
        "created_at": _now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "frozen_oicm_checkpoint": str(decoder.oicm_checkpoint),
        "frozen_oicm_checkpoint_sha256": str(decoder.oicm_checkpoint_sha256),
        "decoder_view_index": str(index_path),
        "sequence_id": sequence.id,
        "frame_count": int(len(arrays["frame"])),
        "window_size": int(cfg.meta.window_size),
        "coordinate_frame": "object_pose_t",
        "source": {
            "hand": "original_grab_mano_parent_cache",
            "object": "original_grab_object_parent_cache",
            "paired_inspire_test_gt": False,
            "initialization": "zero Inspire finger q plus MANO wrist at handoff frame 0",
        },
        "effect_definition": "raw=OICM pred_obj_flow from recursive MANO-source rollout hand flow; effective=raw masked to zero when OICM sample_valid=false; no future object pose/flow is read",
        "outputs": {
            "effect": str(output / "effect.npz"),
            "summary": str(output / "effect_summary.json"),
            "run_manifest": str(output / "run_manifest.json"),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_status": manifest["run_status"], "output": str(output), "summary": summary}, ensure_ascii=False), flush=True)
    if args.serve:
        _serve(arrays, port=int(args.port), fps=float(args.fps))


if __name__ == "__main__":
    main()
