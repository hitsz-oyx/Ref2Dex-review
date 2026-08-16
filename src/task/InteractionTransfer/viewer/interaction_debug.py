"""InteractionTransfer V0.10.1 中间表示 Debug Viewer。

分析 ΔH → M_ij → C_i^O → ΔO 链路：
- 点云 + M edge（Off / Top20 / Top50 / Selected Object Point）；
- 点击 object 点选择后只显示它的 K 条 edge 与邻域表；
- C_obj 四种显示：Off / Magnitude / GT-vs-Current diff / GT-vs-Optimized diff；
- Free Point Flow 优化（不冒充合法 MANO mesh，只显示移动后的点云）。

用法::

    python -m src.task.InteractionTransfer.viewer.interaction_debug \\
        --root data/processed_data/stage4/data/grab \\
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \\
        --checkpoint outputs/InteractionTransfer/v08_full_20ep/best.pt \\
        --device cuda:0 --port 8081
"""
from __future__ import annotations

import argparse
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
import viser

from src.task.InteractionTransfer.inverse_optimize import optimize_hand_flow
from src.task.InteractionTransfer.viewer.mesh_provider import flow_to_mesh_twist, twist_transform
from src.task.InteractionTransfer.viewer.provider import TrajectoryProvider, load_model
from src.task.InteractionTransfer.viewer.scene_renderer import SceneBatch, norm01, viridis

COLOR_OBJ = (170, 170, 170)
COLOR_HAND = (200, 200, 210)
COLOR_EDGE = (255, 40, 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--grab-root", default=None)
    parser.add_argument("--mano-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--sequence", default=None)
    return parser.parse_args()


class DebugViewer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model, self.epoch = load_model(args.checkpoint, args.dense_checkpoint, self.device)
        self.provider = TrajectoryProvider(args.root, args.split, args.grab_root,
                                           args.mano_path, str(self.device))
        self.bundle = None
        self.valid_list = []
        self.result = None          # inverse 结果（rigid 或 free）
        self.result_key = None
        self.selected = -1          # 选中的 object point 下标
        self.lock = threading.RLock()

        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.batch = SceneBatch(self.server)
        self._build_gui()
        self._load_sequence(args.sequence or self.provider.sequences[0])

    # ---------- GUI ----------
    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown("# InteractionTransfer Debug")
        self.info = gui.add_markdown("")

        self.seq_gui = gui.add_dropdown("Sequence", self.provider.sequences)
        self.seq_gui.on_update(lambda _: self._load_sequence(self.seq_gui.value))
        self.tr_gui = gui.add_slider("Valid Transition", min=0, max=1, step=1, initial_value=0)
        self.tr_gui.on_update(lambda _: self.render())

        gui.add_markdown("## Action")
        self.action_gui = gui.add_button_group("Action", ("GT", "Zero", "Reverse", "Cross"))
        self.action_gui.on_click(lambda _: self.render())

        gui.add_markdown("## Layers")
        self.show_hand_flow = gui.add_checkbox("Hand Flow Arrows", True)
        self.show_hand_flow.on_update(lambda _: self.render())
        self.flow_stride = gui.add_slider("Hand Flow Stride", min=1, max=16, step=1, initial_value=8)
        self.flow_stride.on_update(lambda _: self.render())
        self.show_obj_flow = gui.add_checkbox("Object Flow Arrows", False)
        self.show_obj_flow.on_update(lambda _: self.render())
        self.point_size = gui.add_slider("Point Size", min=0.002, max=0.012, step=0.001,
                                         initial_value=0.005)
        self.point_size.on_update(lambda _: self.render())

        gui.add_markdown("## Edge View")
        self.edge_gui = gui.add_button_group("Edges", ("Off", "Top 20", "Top 50", "Selected"))
        self.edge_gui.on_click(lambda _: self.render())

        gui.add_markdown("## C_obj Display")
        self.cobj_gui = gui.add_button_group("C_obj", ("Off", "Magnitude", "GT-Current", "GT-Opt"))
        self.cobj_gui.on_click(lambda _: self.render())

        self.selected_info = gui.add_markdown("点击灰色 object 点云选择一个 object point。")

        with gui.add_folder("Inverse Optimization", expand_by_default=False):
            self.inverse_info = gui.add_markdown("")
            self.param_gui = gui.add_button_group("Parameterization", ("Rigid SE(3)", "Free Flow"))
            self.param_gui.on_click(lambda _: self._invalidate_result())
            self.init_gui = gui.add_button_group("Initialization", ("Zero", "GT"))
            self.init_gui.on_click(lambda _: self._invalidate_result())
            self.target_gui = gui.add_button_group("Target", ("Effect", "C_obj"))
            self.target_gui.on_click(lambda _: self._invalidate_result())
            self.steps_gui = gui.add_slider("Steps", min=50, max=600, step=50, initial_value=300)
            self.steps_gui.on_update(lambda _: self._invalidate_result())
            self.optimize_btn = gui.add_button("Optimize")
            self.reset_btn = gui.add_button("Reset")
            self.optimize_btn.on_click(lambda _: self._optimize())
            self.reset_btn.on_click(lambda _: self._invalidate_result())

    # ---------- 数据 ----------
    def _load_sequence(self, name: str):
        with self.lock:
            self.bundle = self.provider.load(name)
            self.valid_list = self.bundle.valid_frames()
            self.selected = -1
            self._invalidate_result()

    def _invalidate_result(self):
        self.result = None
        self.result_key = None
        if hasattr(self, "inverse_info"):
            self.inverse_info.content = ""
        self.render()

    def _frame(self) -> int:
        if not self.valid_list:
            return 0
        return self.valid_list[min(int(self.tr_gui.value), len(self.valid_list) - 1)]

    def _current_inputs(self):
        i = self._frame()
        inputs = self.bundle.model_inputs(i, self.device)
        with torch.no_grad():
            static = self.model.encode_static(inputs["object_points"], inputs["object_normals"],
                                              inputs["hand_points"], inputs["hand_normals"])
            out = self.model.forward_core(hand_flow=inputs["hand_flow"], **static)
        return i, inputs, static, out

    def _cross_flow(self, inputs, i: int) -> torch.Tensor:
        """cross-sample action：同序列下一个 valid transition 的 hand flow。"""
        nxt = [f for f in self.valid_list if f > i]
        if not nxt:
            return torch.zeros_like(inputs["hand_flow"])
        cross = self.bundle.model_inputs(nxt[0], self.device)
        return cross["hand_flow"]

    # ---------- 渲染 ----------
    def render(self):
        with self.lock:
            try:
                self.batch.clear()
                self._render()
            except Exception:
                self.info.content = f"```\n{traceback.format_exc()[-1200:]}\n```"

    def _render(self):
        b = self.bundle
        if not self.valid_list:
            self.info.content = "该序列没有 valid transition"
            return
        i, inputs, static, gt_out = self._current_inputs()
        center = inputs["center"]
        op = self.bundle.obj_points_world[i, self.bundle.object_indices].astype(np.float32) - center
        hp = self.bundle.hand_points_world[i].astype(np.float32) - center
        action = self.action_gui.value
        if action == "GT":
            out = gt_out
            flow = inputs["hand_flow"]
        elif action == "Cross":
            flow = self._cross_flow(inputs, i)
            with torch.no_grad():
                out = self.model.forward_core(hand_flow=flow, **static)
        else:
            flow = {"Zero": torch.zeros_like(inputs["hand_flow"]),
                    "Reverse": -inputs["hand_flow"]}[action]
            with torch.no_grad():
                out = self.model.forward_core(hand_flow=flow, **static)

        obj_colors = self._object_colors(inputs, out, gt_out, op)
        obj_handle = self.batch.cloud("/object", op, obj_colors, float(self.point_size.value))
        obj_handle.on_click(lambda event: self._on_object_click(event))
        self.batch.cloud("/hand", hp, np.tile(np.array(COLOR_HAND, np.uint8), (len(hp), 1)),
                         float(self.point_size.value) * 0.8)

        if self.show_hand_flow.value:
            self.batch.arrows("/hand_flow", hp, flow.squeeze(0).cpu().numpy(),
                              (120, 150, 255), int(self.flow_stride.value))
        if self.show_obj_flow.value:
            self.batch.arrows("/obj_flow_pred", op, out["object_flow"].squeeze(0).cpu().numpy(),
                              (255, 140, 0), 1)
            self.batch.arrows("/obj_flow_gt", op, inputs["object_flow"].squeeze(0).cpu().numpy(),
                              (0, 200, 120), 1)

        self._render_edges(out, op, hp)
        self._render_free_flow(inputs, hp)
        self._status(i, action, out, inputs)

    def _object_colors(self, inputs, out, gt_out, op) -> np.ndarray:
        mode = self.cobj_gui.value
        if mode == "Off":
            return np.tile(np.array(COLOR_OBJ, np.uint8), (len(op), 1))
        if mode == "Magnitude":
            values = np.linalg.norm(out["object_field"].squeeze(0).cpu().numpy(), axis=1)
        elif mode == "GT-Current":
            delta = (out["object_field"] - gt_out["object_field"]).squeeze(0).cpu().numpy()
            values = np.linalg.norm(delta, axis=1)
        else:  # GT-Opt
            if self.result is None:
                self.inverse_info.content = "GT-Opt diff 需要先 Optimize"
                return np.tile(np.array(COLOR_OBJ, np.uint8), (len(op), 1))
            opt = torch.from_numpy(np.asarray(self.result["optimized"]["object_field"])).to(self.device)
            delta = (opt - gt_out["object_field"]).squeeze(0).cpu().numpy()
            values = np.linalg.norm(delta, axis=1)
        return viridis(norm01(values))

    def _render_edges(self, out, op, hp):
        mode = self.edge_gui.value
        if mode == "Off":
            return
        valid = out["edge_valid"].astype(bool)
        edge_idx = out["edge_index"]
        n_obj, k_nn = edge_idx.shape
        mags = np.linalg.norm(out["edge_message"], axis=-1)
        attn = out["attention"]

        if mode == "Selected":
            if not (0 <= self.selected < n_obj):
                return
            row = self.selected
            neighbors = edge_idx[row]
            mask = valid[row]
            segs = np.stack([np.full((int(mask.sum()), 3), op[row]),
                             hp[neighbors[mask]]], axis=1)
            strength = norm01(mags[row][mask])
            colors = (strength[:, None] * np.array([[255, 40, 40]])).astype(np.uint8)
            colors = np.repeat(colors[:, None, :], 2, axis=1)
            self.batch._keep(self.server.scene.add_line_segments(
                "/edges_selected", points=segs.astype(np.float32), colors=colors, line_width=2.0))
            self._selected_table(row, neighbors[mask], mags[row][mask], attn[row][mask])
            self.batch._keep(self.server.scene.add_point_cloud(
                "/selected_point", points=op[row][None],
                colors=np.array([[255, 215, 0]], np.uint8),
                point_size=float(self.point_size.value) * 3.0))
            return

        k = int(mode.split()[-1])
        metric = np.where(valid, mags, -np.inf).reshape(-1)
        top = np.argsort(metric)[::-1][:k]
        obj_i, hand_j = top // k_nn, edge_idx.reshape(-1)[top]
        segs = np.stack([op[obj_i], hp[hand_j]], axis=1)
        strength = norm01(mags.reshape(-1)[top])
        colors = (strength[:, None] * np.array([[255, 40, 40]])).astype(np.uint8)
        colors = np.repeat(colors[:, None, :], 2, axis=1)
        self.batch._keep(self.server.scene.add_line_segments(
            "/edges", points=segs.astype(np.float32), colors=colors, line_width=2.0))

    def _selected_table(self, row, neighbors, mags, attn):
        order = np.argsort(mags)[::-1]
        lines = [f"**Selected Object Point {row}**（按 ||M|| 降序）", "",
                 "| Neighbor | \\|\\|M\\|\\| | Attention |", "|---|---|---|"]
        for rank in order:
            lines.append(f"| {int(neighbors[rank])} | {mags[rank]:.4f} | {attn[rank]:.4f} |")
        self.selected_info.content = "\n".join(lines)

    def _render_free_flow(self, inputs, hp):
        if self.result is None or self.result["meta"]["parameterization"] != "free":
            return
        flow = np.asarray(self.result["optimized"]["flow"]).squeeze(0)
        moved = hp + flow
        colors = viridis(norm01(np.linalg.norm(flow, axis=1)))
        self.batch.cloud("/free_moved", moved, colors, float(self.point_size.value) * 0.8)

    def _on_object_click(self, event):
        inner = getattr(event, "event", event)
        origin = getattr(inner, "ray_origin", None)
        direction = getattr(inner, "ray_direction", None)
        if origin is None or direction is None:
            return
        origin, direction = np.asarray(origin, np.float32), np.asarray(direction, np.float32)
        with self.lock:
            i = self._frame()
            center = self.bundle.model_inputs(i, self.device)["center"]
            op = self.bundle.obj_points_world[i, self.bundle.object_indices] - center
            v = op - origin
            t = v @ direction
            perp = np.linalg.norm(v - np.outer(t, direction), axis=1)
            perp[t <= 0] = np.inf
            self.selected = int(np.argmin(perp))
            self.render()

    def _status(self, i, action, out, inputs):
        from src.task.InteractionTransfer.metrics import epe

        epe_mm = float(epe(out["object_flow"], inputs["object_flow"])) * 1000
        b = self.bundle
        self.info.content = (
            f"**{b.name}** frame {i}（raw {int(b.raw_frame_id[i])}） | valid "
            f"{int(self.tr_gui.value) + 1}/{len(self.valid_list)} | action **{action}** | "
            f"EPE **{epe_mm:.2f} mm**"
            + (f" | mano parity {b.hand_parity_mm:.2f} mm" if b.hand_parity_mm > 1.0 else ""))

    # ---------- Inverse ----------
    def _optimize(self):
        with self.lock:
            i = self._frame()
            param = "rigid" if self.param_gui.value == "Rigid SE(3)" else "free"
            key = (i, param, self.init_gui.value, self.target_gui.value, int(self.steps_gui.value))
            if self.result is not None and self.result_key == key:
                return
            inputs = self.bundle.model_inputs(i, self.device)
            with torch.no_grad():
                static = self.model.encode_static(inputs["object_points"], inputs["object_normals"],
                                                  inputs["hand_points"], inputs["hand_normals"])
            init_flow = (inputs["hand_flow"] if self.init_gui.value == "GT"
                         else torch.zeros_like(inputs["hand_flow"]))
            self.inverse_info.content = "optimizing..."
            result = optimize_hand_flow(
                self.model, static, inputs["hand_points"], inputs["object_flow"],
                gt_flow=inputs["hand_flow"], init_flow=init_flow, parameterization=param,
                target_kind=self.target_gui.value.lower(), steps=int(self.steps_gui.value), lr=0.01)
            self.result, self.result_key = result, key
            history = result["history"]
            tail = " → ".join(f"{h['epe_mm']:.2f}" for h in history[::max(1, len(history) // 6)])
            self.inverse_info.content = (
                f"{param} | init={self.init_gui.value} | target={self.target_gui.value}  \n"
                f"EPE initial **{result['initial']['epe_mm']:.2f}** → optimized "
                f"**{result['optimized']['epe_mm']:.2f}** | GT action **{result['gt']['epe_mm']:.2f}** mm  \n"
                f"history: {tail} mm")
            self.render()


def main() -> None:
    DebugViewer(parse_args())
    print("debug viewer running", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
