"""GT MANO / GT-Y robot / Direct-H MANO 的 object-centric 三路 Viser viewer。"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import viser

from src.task.InteractionDynamics.viewer_gty.provider import GTYComparisonProvider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--review", type=Path, default=Path("outputs/gty_generalization/manual_review.json"))
    parser.add_argument("--host", default="0.0.0.0"); parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--fps", type=float, default=6.); args = parser.parse_args()
    provider = GTYComparisonProvider(args.cache); server = viser.ViserServer(host=args.host, port=args.port)
    state = {"trajectory": 0, "sample": provider.load(0, provider.robots[0]), "playing": False}
    lock = threading.Lock(); handles = []
    trajectory = server.gui.add_dropdown("Trajectory", provider.names, initial_value=provider.names[0])
    previous_trajectory = server.gui.add_button("Previous Trajectory"); next_trajectory = server.gui.add_button("Next Trajectory")
    robot = server.gui.add_dropdown("Robot Hand", provider.robots, initial_value=provider.robots[0])
    frame = server.gui.add_slider("Frame", min=0, max=state["sample"].frames - 1, step=1, initial_value=0)
    previous_frame = server.gui.add_button("< Frame"); next_frame = server.gui.add_button("Frame >")
    play = server.gui.add_button("Play"); stop = server.gui.add_button("Stop")
    jump_contact = server.gui.add_button("Jump Contact"); jump_stable = server.gui.add_button("Jump Stable")
    show_gt = server.gui.add_checkbox("Show GT", True); show_robot = server.gui.add_checkbox("Show GT-Y Optimize", True)
    show_pred = server.gui.add_checkbox("Show Direct-H", True); show_object = server.gui.add_checkbox("Show Object", True)
    show_contacts = server.gui.add_checkbox("Show Contacts", False)
    good = server.gui.add_button("Mark Good"); bad = server.gui.add_button("Mark Bad"); unsure = server.gui.add_button("Mark Unsure")
    status = server.gui.add_markdown("")

    def offsets(sample):
        diameter = np.linalg.norm(np.ptp(sample.object_vertices, axis=0))
        spacing = max(.15, 1.5 * diameter)
        return (-spacing * np.array([1, 0, 0]), np.zeros(3), spacing * np.array([1, 0, 0]))

    def render() -> None:
        nonlocal handles
        for handle in handles: handle.remove()
        handles = []; sample = state["sample"]; index = min(int(frame.value), sample.frames - 1)
        panel_offsets = offsets(sample)
        for name, visible, offset in (("gt", show_gt.value, panel_offsets[0]),
                                      ("gty_opt", show_robot.value, panel_offsets[1]),
                                      ("pred_h", show_pred.value, panel_offsets[2])):
            if show_object.value and visible:
                handles.append(server.scene.add_mesh_simple(f"/world/{name}/object",
                    sample.object_vertices + offset, sample.object_faces, color=(180, 180, 180)))
        handles.append(server.scene.add_mesh_simple("/world/gt/hand", sample.gt_vertices[index] + panel_offsets[0],
            sample.mano_faces, color=(80, 220, 120), visible=show_gt.value))
        handles.append(server.scene.add_mesh_simple("/world/pred_h/hand", sample.pred_vertices[index] + panel_offsets[2],
            sample.mano_faces, color=(80, 160, 255), visible=show_pred.value))
        robot_ok = bool(sample.robot_available[index])
        if robot_ok:
            handles.append(server.scene.add_mesh_simple("/world/gty_opt/hand",
                sample.robot_vertices[index] + panel_offsets[1], sample.robot_faces,
                color=(255, 165, 65), visible=show_robot.value))
        if show_contacts.value:
            for panel, offset, y in (("gt", panel_offsets[0], sample.gt_y),
                                     ("pred_h", panel_offsets[2], sample.pred_y)):
                if index > 0:
                    points = sample.anchors[y[:, index - 1, 3] < 2] + offset
                    if len(points): handles.append(server.scene.add_point_cloud(
                        f"/world/{panel}/contacts", points, colors=(255, 40, 40), point_size=.006))
        metrics = sample.robot_metrics; residual = metrics["y_residual"][index]
        pred_error = None if index == 0 else float(np.sqrt(np.mean((sample.pred_y[:, index-1] - sample.gt_y[:, index-1]) ** 2)))
        status.content = (f"**{sample.source_raw_file}**  \nObject: **{sample.object_name}** | Subject: **{sample.subject}**  \n"
            f"Frame: **{index}/{sample.frames-1}** | raw: **{sample.raw_frame_ids[index]}** | Robot: **{sample.robot_name}**  \n"
            f"GT-Y Optimize: **{'available' if robot_ok else 'Not optimized'}**  \n"
            f"Y residual: **{residual if residual is not None else 'N/A'} cm** | penetration: **{metrics['penetration_mm'][index]} mm**  \n"
            f"joint margin: **{metrics['joint_limit_margin'][index]:.4f} rad** | contacts: **{metrics['contact_count'][index]}**  \n"
            f"Direct-H field error: **{pred_error if pred_error is not None else 'N/A'} cm** | penetration: **{sample.pred_metrics['penetration_mm'][index]} mm**  \n"
            f"Direct-H contacts: **{None if index == 0 else int((sample.pred_y[:,index-1,3]<2).sum())}**")

    def reload(index: int | None = None) -> None:
        if index is not None: state["trajectory"] = index % len(provider)
        trajectory.value = provider.names[state["trajectory"]]
        state["sample"] = provider.load(state["trajectory"], robot.value)
        frame.max = state["sample"].frames - 1; frame.value = 0; render()

    @trajectory.on_update
    def _(_): reload(provider.names.index(trajectory.value))
    @previous_trajectory.on_click
    def _(_): reload(state["trajectory"] - 1)
    @next_trajectory.on_click
    def _(_): reload(state["trajectory"] + 1)
    @robot.on_update
    def _(_): reload()
    @frame.on_update
    def _(_):
        with lock: render()
    @previous_frame.on_click
    def _(_): frame.value = max(0, int(frame.value) - 1)
    @next_frame.on_click
    def _(_): frame.value = min(frame.max, int(frame.value) + 1)
    @play.on_click
    def _(_): state["playing"] = True
    @stop.on_click
    def _(_): state["playing"] = False
    @jump_contact.on_click
    def _(_):
        indices = np.nonzero((state["sample"].gt_y[..., 3] < 2).sum(0) >= 4)[0]
        if len(indices): frame.value = int(indices[0]) + 1
    @jump_stable.on_click
    def _(_): frame.value = int(state["sample"].robot_metrics["representative_frames"][-1])
    for control in (show_gt, show_robot, show_pred, show_object, show_contacts): control.on_update(lambda _: render())

    def mark(label: str) -> None:
        args.review.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(args.review.read_text()) if args.review.exists() else {}
        sample = state["sample"]; key = f"{sample.source_raw_file}|{sample.robot_name}|{int(frame.value)}"
        data[key] = {"trajectory": sample.source_raw_file, "robot": sample.robot_name,
                     "frame": int(frame.value), "label": label}
        args.review.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    @good.on_click
    def _(_): mark("good")
    @bad.on_click
    def _(_): mark("bad")
    @unsure.on_click
    def _(_): mark("unsure")
    render(); print(f"GT-Y Generalization Viewer: http://localhost:{args.port}", flush=True)
    print(f"trajectories={len(provider)} robots={provider.robots}", flush=True)
    while True:
        if state["playing"]: frame.value = (int(frame.value) + 1) % (frame.max + 1)
        time.sleep(1 / args.fps)


if __name__ == "__main__":
    main()
