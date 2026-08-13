"""Viser 通用 object/reference trajectory 与 hand trajectory viewer。"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np
import viser

from src.task.InteractionDynamics.viewer_v2.prediction_v18_5 import V18_5PredictionProvider


def metric(value, suffix="") -> str:
    return "N/A" if value is None else f"{value}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--host", default="0.0.0.0"); parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda"); parser.add_argument("--fps", type=float, default=6.)
    parser.add_argument("--post-latch-frames", type=int, default=60)
    args = parser.parse_args()
    provider = V18_5PredictionProvider(args.checkpoint, args.cache, args.split, device=args.device,
                                       post_latch_frames=args.post_latch_frames)
    server = viser.ViserServer(host=args.host, port=args.port)
    state = {"sample": provider(0), "playing": False}; lock = threading.Lock()
    hand_model = server.gui.add_dropdown("Hand Model", ("MANO",), initial_value="MANO")
    sample_slider = server.gui.add_slider("Sample", min=0, max=len(provider)-1, step=1, initial_value=0)
    frame_slider = server.gui.add_slider("Frame", min=0,
        max=len(state["sample"].pred_hand) - 1, step=1, initial_value=0)
    play = server.gui.add_button("Play"); stop = server.gui.add_button("Stop")
    show_object_path = server.gui.add_checkbox("Show Reference Object Trajectory", True)
    show_prediction = server.gui.add_checkbox("Show Prediction", True)
    show_gt = server.gui.add_checkbox("Show Ground Truth", True)
    status = server.gui.add_markdown("")
    object_handle = pred_handle = gt_handle = path_handle = None

    def render() -> None:
        nonlocal object_handle, pred_handle, gt_handle, path_handle
        sample = state["sample"]; frame = min(int(frame_slider.value), len(sample.pred_hand) - 1)
        for handle in (object_handle, pred_handle, gt_handle, path_handle):
            if handle is not None: handle.remove()
        obj = sample.object_trajectory
        object_handle = server.scene.add_mesh_simple("/world/object/current",
            obj.world_vertices(frame), obj.faces, color=(180, 180, 180))
        pred = sample.pred_hand[frame]
        pred_handle = server.scene.add_mesh_simple("/world/hand/pred", pred.vertices, pred.faces,
            color=(80, 160, 255), visible=show_prediction.value)
        gt_handle = None
        if sample.gt_hand is not None:
            gt = sample.gt_hand[frame]
            gt_handle = server.scene.add_mesh_simple("/world/hand/gt", gt.vertices, gt.faces,
                color=(80, 220, 120), opacity=.35, wireframe=False, visible=show_gt.value)
        centers = obj.poses[:, :3, 3]
        path_handle = server.scene.add_spline_catmull_rom("/world/object/reference_trajectory",
            centers, color=(255, 170, 40), line_width=2., visible=show_object_path.value)
        stable = sample.stable_states[min(frame, len(sample.stable_states) - 1)] if sample.stable_states else None
        phase = "OBJECT FOLLOW" if sample.latch_frame is not None and frame > sample.latch_frame else "GRASP FORMATION"
        latch = "NO STABLE LATCH WITHIN HORIZON" if sample.latch_frame is None else f"frame {sample.latch_frame}"
        status.content = (f"**{sample.label}**  \nPhase: **{phase}**  \nPredicted latch: **{latch}**  \n"
            f"Contact anchors: **{metric(None if stable is None else stable.contact_count)}**  \n"
            f"u RMS: **{metric(None if stable is None else round(stable.u_rms_cm, 3), ' cm/frame')}**  \n"
            f"Stable consecutive: **{metric(None if stable is None else stable.stable_count)}**  \n"
            f"Grasp proxy: **{'PASS' if sample.grasp_proxy else 'FAIL'}**  \n"
            f"Collision: **{metric(sample.collision)}**  \n"
            f"Pred penetration max/mean: **{metric(sample.max_penetration_mm, ' mm')} / "
            f"{metric(sample.mean_penetration_mm, ' mm')}**  \n"
            f"GT penetration max/mean: **{metric(sample.gt_max_penetration_mm, ' mm')} / "
            f"{metric(sample.gt_mean_penetration_mm, ' mm')}**")

    @frame_slider.on_update
    def _(_):
        with lock: render()
    @sample_slider.on_update
    def _(_):
        with lock:
            state["sample"] = provider(int(sample_slider.value))
            frame_slider.max = len(state["sample"].pred_hand) - 1
            frame_slider.value = 0; render()
    @play.on_click
    def _(_): state["playing"] = True
    @stop.on_click
    def _(_): state["playing"] = False
    @show_prediction.on_update
    def _(_):
        if pred_handle is not None: pred_handle.visible = show_prediction.value
    @show_gt.on_update
    def _(_):
        if gt_handle is not None: gt_handle.visible = show_gt.value
    @show_object_path.on_update
    def _(_):
        if path_handle is not None: path_handle.visible = show_object_path.value
    render()
    print(f"Trajectory Grasp Viewer: http://localhost:{args.port}", flush=True)
    print(f"samples={len(provider)} hand_backend={hand_model.value}", flush=True)
    while True:
        if state["playing"]:
            frame_slider.value = (int(frame_slider.value) + 1) % len(state["sample"].pred_hand)
        time.sleep(1 / args.fps)


if __name__ == "__main__": main()
