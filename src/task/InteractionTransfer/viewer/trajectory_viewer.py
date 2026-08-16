"""InteractionTransfer V1.0 正式主 Viewer。

与 legacy ``visualize_intermediate.py`` 的区别：
- 真实 MANO 双手 mesh（geometry cache 778 顶点/手）+ GRAB object mesh；
- V1.0 数据流：双手 769+769 在线表面采样（按 (i, gap) 播种）、Gap 下拉
  （{1,2,4,8}，限当前帧 valid gaps）、9D action（含 Δt）；
- Forward：Current / Pred Next / GT Next 的"下一状态"mesh 对比；
- Inverse：Initial / Optimized / GT 三路并排；rigid 12D（左右手各自
  SE(3)）优化结果分别提升回两只手的 MANO mesh；
- ``server.atomic()`` 批量重建，播放不闪烁；panel 间距按 object diameter 自适应。

用法::

    python -m src.task.InteractionTransfer.viewer.trajectory_viewer \\
        --root data/processed_data/stage4/data/grab \\
        --geometry-root data/processed_data/stage4/interactiontransfer_geometry_cache \\
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \\
        --checkpoint outputs/InteractionTransfer/v10_full_20ep/best.pt \\
        --device cuda:0 --port 8080
"""
from __future__ import annotations

import argparse
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import viser

from src.task.InteractionTransfer.inverse_optimize import optimize_hand_flow
from src.task.InteractionTransfer.metrics import epe
from src.task.InteractionTransfer.viewer.legacy_model import legacy_optimize_hand_flow
from src.task.InteractionTransfer.viewer.mesh_provider import flow_to_mesh_twist, twist_transform
from src.task.InteractionTransfer.viewer.provider import GAPS, TrajectoryProvider, load_model
from src.task.InteractionTransfer.viewer.scene_renderer import (
    SceneBatch, auto_spacing, panel_offsets,
)

COLOR_CURRENT_OBJ = (175, 175, 175)
COLOR_CURRENT_HAND = (120, 140, 180)
COLOR_PRED = (255, 140, 0)
COLOR_GT = (0, 200, 120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="GRAB stage4 cache 目录")
    parser.add_argument("--geometry-root", required=True, help="geometry cache 目录")
    parser.add_argument("--split", type=Path, required=True, help="split txt（每行一个序列）")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--grab-root", default=None, help="raw GRAB 根目录（默认 process/GRAB 约定）")
    parser.add_argument("--mano-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sequence", default=None, help="初始序列，默认 split 第一个")
    return parser.parse_args()


def split_hands(inputs: dict):
    """concat 手点 -> (left, right) 两块。"""
    n = inputs["hand_points"].shape[1] // 2
    return (inputs["hand_points"][:, :n], inputs["hand_normals"][:, :n],
            inputs["hand_points"][:, n:], inputs["hand_normals"][:, n:])


def encode_static_split(model, inputs):
    lp, ln, rp, rn = split_hands(inputs)
    return model.encode_static(inputs["object_points"], inputs["object_normals"], lp, ln, rp, rn)


class TrajectoryViewer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model, self.epoch = load_model(args.checkpoint, args.dense_checkpoint, self.device)
        self.legacy = bool(getattr(self.model, "is_legacy", False))
        self.provider = TrajectoryProvider(args.root, args.split, args.geometry_root,
                                           args.grab_root, args.mano_path, str(self.device))
        self.bundle = None
        self.result = None            # 当前帧的 inverse 结果
        self.result_frame = -1
        self.gap = 1
        self.playing = False
        self.lock = threading.RLock()

        self.server = viser.ViserServer(host=args.host, port=args.port)
        self.batch = SceneBatch(self.server)
        self._build_gui()
        self._load_sequence(args.sequence or self.provider.sequences[0])

    # ---------- GUI ----------
    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown("# InteractionTransfer (V1.0)" if not self.legacy
                         else "# InteractionTransfer (V0.x legacy ckpt)")
        self.info = gui.add_markdown("")

        self.seq_gui = gui.add_dropdown("Sequence", self.provider.sequences)
        self.seq_gui.on_update(lambda _: self._load_sequence(self.seq_gui.value))

        gui.add_markdown("## Frame")
        self.frame_gui = gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
        self.frame_gui.on_update(lambda _: self._on_frame())
        self.gap_gui = gui.add_dropdown("Gap (Δt)", tuple(str(g) for g in GAPS), initial_value="1")
        self.gap_gui.on_update(lambda _: self._on_gap())
        if self.legacy:
            self.gap_gui.visible = False  # V0.x 只支持 one-step
        self.prev_btn = gui.add_button("<", visible=True)
        self.play_btn = gui.add_button("Play")
        self.stop_btn = gui.add_button("Stop")
        self.next_btn = gui.add_button(">")
        self.jump_btn = gui.add_button("Jump Valid")
        self.fps_gui = gui.add_slider("Playback FPS", min=1, max=30, step=1, initial_value=6)
        self.prev_btn.on_click(lambda _: self._step_frame(-1))
        self.next_btn.on_click(lambda _: self._step_frame(+1))
        self.play_btn.on_click(lambda _: setattr(self, "playing", True))
        self.stop_btn.on_click(lambda _: setattr(self, "playing", False))
        self.jump_btn.on_click(lambda _: self._jump_valid())

        gui.add_markdown("## View")
        self.view_gui = gui.add_button_group("Mode", ("Forward", "Inverse"))
        self.view_gui.on_click(lambda _: self.render())

        gui.add_markdown("## Scene")
        self.show_hand = gui.add_checkbox("Current Hands", True)
        self.show_object = gui.add_checkbox("Current Object", True)
        self.show_pred = gui.add_checkbox("Predicted Next", True)
        self.show_gt = gui.add_checkbox("Ground Truth Next", True)
        for box in (self.show_hand, self.show_object, self.show_pred, self.show_gt):
            box.on_update(lambda _: self.render())
        self.compare_gui = gui.add_button_group("Comparison", ("Overlay", "Side-by-side"))
        self.compare_gui.on_click(lambda _: self.render())

        with gui.add_folder("Advanced", expand_by_default=False):
            self.show_flow = gui.add_checkbox("Sparse Flow Arrows", False)
            self.show_flow.on_update(lambda _: self.render())
            self.flow_stride = gui.add_slider("Flow Stride", min=1, max=32, step=1, initial_value=16)
            self.flow_stride.on_update(lambda _: self.render())
            self.next_opacity = gui.add_slider("Next-state Opacity", min=0.1, max=0.9,
                                               step=0.05, initial_value=0.45)
            self.next_opacity.on_update(lambda _: self.render())

        with gui.add_folder("Inverse Optimization", expand_by_default=False):
            self.inverse_info = gui.add_markdown("")
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

    # ---------- 状态 ----------
    def _load_sequence(self, name: str):
        with self.lock:
            self.bundle = self.provider.load(name)
            self._invalidate_result()
            frames = self._valid_frame_list()
            frame = frames[0] if frames else 0
            self.frame_gui.max = self.bundle.frames - 1
            self.frame_gui.value = frame
            self.render()

    def _on_gap(self):
        with self.lock:
            if self.legacy:
                self.gap = 1
                return
            self.gap = int(self.gap_gui.value)
            self._invalidate_result()

    def _invalidate_result(self):
        self.result = None
        self.result_frame = -1
        if hasattr(self, "inverse_info"):
            self.inverse_info.content = ""
        self.render()

    def _on_frame(self):
        with self.lock:
            if self.result is not None and self.result_frame != int(self.frame_gui.value):
                self.result = None
                self.inverse_info.content = ""
            self.render()

    def _step_frame(self, delta: int):
        value = int(self.frame_gui.value) + delta
        self.frame_gui.value = max(0, min(self.frame_gui.max, value))

    def _jump_valid(self):
        b = self.bundle
        current = int(self.frame_gui.value)
        valid = self._valid_frame_list()
        ahead = [i for i in valid if i > current]
        self.frame_gui.value = ahead[0] if ahead else (valid[0] if valid else 0)

    # ---------- V0.x / V1.0 兼容 ----------
    def _valid(self, i: int) -> bool:
        return self.bundle.valid_transition_v010(i) if self.legacy \
            else self.bundle.valid_transition(i, self.gap)

    def _valid_frame_list(self) -> list[int]:
        return self.bundle.valid_frames_v010() if self.legacy \
            else self.bundle.valid_frames(self.gap)

    def _model_inputs(self, i: int) -> dict:
        return self.bundle.model_inputs_v010(i, self.device) if self.legacy \
            else self.bundle.model_inputs(i, self.device, self.gap)

    def _encode(self, inputs: dict) -> dict:
        if self.legacy:
            return self.model.encode_static(inputs["object_points"], inputs["object_normals"],
                                            inputs["hand_points"], inputs["hand_normals"])
        return encode_static_split(self.model, inputs)

    def _forward(self, inputs: dict, flow: torch.Tensor = None) -> dict:
        hand_flow = inputs["hand_flow"] if flow is None else flow
        if self.legacy:
            return self.model.forward_core(hand_flow=hand_flow, **self._encode(inputs))
        return self.model.forward_core(hand_flow=hand_flow, dt=inputs["gap"], **self._encode(inputs))

    # ---------- Forward ----------
    def _forward_outputs(self, i: int):
        """encode_static + forward_core(GT action)。仅对 valid transition 调用。"""
        inputs = self._model_inputs(i)
        with torch.no_grad():
            out = self._forward(inputs)
        return inputs, out

    def _center(self, i: int) -> np.ndarray:
        oi = self.bundle.object_indices
        return self.bundle.obj_points_world[i, oi].mean(0)

    def _hand_meshes(self, i: int, center: np.ndarray):
        """当前帧双手 mesh（world 减 object center）。"""
        return {side: self.bundle.hand_verts[side][i] - center for side in ("left", "right")}

    def _next_object_mesh(self, object_flow: torch.Tensor, points: torch.Tensor,
                          mesh_verts: torch.Tensor) -> np.ndarray:
        """512 点 flow -> 刚体 twist -> 整个 object mesh 的下一状态。"""
        twist, _ = flow_to_mesh_twist(object_flow, points)
        center = points.mean(dim=1, keepdim=True)
        return twist_transform(twist, mesh_verts, center).squeeze(0).cpu().numpy()

    def _hand_meshes_from_flow(self, flow: torch.Tensor, hp: torch.Tensor, hand_meshes: dict) -> dict:
        """优化得到的 ΔH 分别拟合 twist 并提升回手部 mesh。

        V1.0：concat 双手各拟合一块；V0.x：单右手块，左手 mesh 保持当前帧。
        """
        if self.legacy:
            twist, center = flow_to_mesh_twist(flow, hp)
            mesh = torch.from_numpy(hand_meshes["right"]).to(self.device).unsqueeze(0)
            return {"right": twist_transform(twist, mesh, center).squeeze(0).cpu().numpy(),
                    "left": hand_meshes["left"]}
        n = hp.shape[1] // 2
        out = {}
        for side, sl in (("left", slice(0, n)), ("right", slice(n, None))):
            block_pts = hp[:, sl]
            block_flow = flow[:, sl]
            twist, center = flow_to_mesh_twist(block_flow, block_pts)
            mesh = torch.from_numpy(hand_meshes[side]).to(self.device).unsqueeze(0)
            out[side] = twist_transform(twist, mesh, center).squeeze(0).cpu().numpy()
        return out

    def render(self):
        with self.lock:
            try:
                self.batch.clear()
                if self.view_gui.value == "Inverse":
                    self._render_inverse()
                else:
                    self._render_forward()
            except Exception:
                self.info.content = f"```\n{traceback.format_exc()[-1200:]}\n```"

    def _render_forward(self):
        b, i, gap = self.bundle, int(self.frame_gui.value), self.gap
        valid = self._valid(i)
        center = self._center(i)

        obj_mesh = b.obj_world[i] - center
        hand_meshes = self._hand_meshes(i, center)
        spacing = auto_spacing(obj_mesh)

        if not valid:
            self._panel_meshes("/current", np.zeros(3), obj_mesh, hand_meshes, None, None)
            self._status(i, valid=False)
            return

        inputs, out = self._forward_outputs(i)
        pred_flow = out["object_flow"]
        gt_flow = inputs["object_flow"]
        pred_epe = float(epe(pred_flow, gt_flow)) * 1000
        pred_mesh = self._next_object_mesh(pred_flow, inputs["object_points"],
                                           torch.from_numpy(obj_mesh).to(self.device))
        gt_mesh = b.obj_world[i + gap] - center

        if self.compare_gui.value == "Overlay":
            self._panel_meshes("/scene", np.zeros(3), obj_mesh, hand_meshes,
                               pred_mesh if self.show_pred.value else None,
                               gt_mesh if self.show_gt.value else None)
            top = self._label_height(obj_mesh, gt_mesh, pred_mesh)
            if self.show_pred.value:
                self.batch.label("/scene/pred_label", (0.0, top, 0.0), f"Pred  EPE {pred_epe:.2f} mm")
            elif self.show_gt.value:
                self.batch.label("/scene/gt_label", (0.0, top, 0.0), "GT next")
        else:
            offsets = panel_offsets(2, spacing)
            self._panel_meshes("/pred", offsets[0], obj_mesh, hand_meshes,
                               pred_mesh if self.show_pred.value else None, None)
            self._panel_meshes("/gt", offsets[1], obj_mesh, hand_meshes,
                               None, gt_mesh if self.show_gt.value else None)
            top = self._label_height(obj_mesh, gt_mesh, pred_mesh)
            self.batch.label("/pred/label", (float(offsets[0][0]), top, 0.0),
                             f"Pred next  EPE {pred_epe:.2f} mm")
            self.batch.label("/gt/label", (float(offsets[1][0]), top, 0.0), "GT next")

        if self.show_flow.value:
            self._sparse_flows("/scene", np.zeros(3), inputs, pred_flow)

        self._status(i, valid=True, pred_epe=pred_epe)

    def _render_inverse(self):
        b, i, gap = self.bundle, int(self.frame_gui.value), self.gap
        valid = self._valid(i)
        center = self._center(i)
        obj_mesh = b.obj_world[i] - center
        hand_meshes = self._hand_meshes(i, center)

        if not valid:
            self._panel_meshes("/current", np.zeros(3), obj_mesh, hand_meshes, None, None)
            self._status(i, valid=False)
            return

        if self.result is None:
            self.inverse_info.content = ("尚未优化：点击 Optimize（Rigid 6D 单右手）"
                                         if self.legacy else "尚未优化：点击 Optimize（Rigid 12D 双手）")
            self._render_inverse_placeholder(i, obj_mesh, hand_meshes, center)
            return

        spacing = auto_spacing(obj_mesh)
        offsets = panel_offsets(3, spacing)
        inputs = self._model_inputs(i)
        hp = inputs["hand_points"]

        obj_model = torch.from_numpy(obj_mesh).to(self.device).unsqueeze(0)
        op = inputs["object_points"]
        columns = {}
        for name in ("initial", "optimized", "gt"):
            row = self.result[name]
            flow = torch.from_numpy(np.asarray(row["flow"])).to(self.device)
            obj_flow = torch.from_numpy(np.asarray(row["object_flow"])).to(self.device)
            columns[name] = {
                "hand": ({side: b.hand_verts[side][i + gap] - center for side in ("left", "right")}
                         if name == "gt" else self._hand_meshes_from_flow(flow, hp, hand_meshes)),
                "next_obj": (b.obj_world[i + gap] - center) if name == "gt"
                else self._next_object_mesh(obj_flow, op, obj_model),
                "epe": row["epe_mm"],
            }

        titles = {"initial": "Initial", "optimized": "Optimized", "gt": "GT action"}
        colors = {"initial": COLOR_CURRENT_HAND, "optimized": COLOR_PRED, "gt": COLOR_GT}
        gt_mesh_next = columns["gt"]["next_obj"]
        top = self._label_height(obj_mesh, gt_mesh_next, columns["optimized"]["next_obj"])
        for idx, name in enumerate(("initial", "optimized", "gt")):
            col = columns[name]
            x = np.array([float(offsets[idx]), 0.0, 0.0])
            self._panel_meshes(f"/{name}", x, obj_mesh, col["hand"],
                               col["next_obj"] if name != "gt" else None,
                               col["next_obj"] if name == "gt" else None,
                               hand_color=colors[name])
            self.batch.label(f"/{name}/label", (float(offsets[idx]), top, 0.0),
                             f"{titles[name]}  EPE {col['epe']:.2f} mm")
        history = self.result["history"]
        tail = " → ".join(f"{h['epe_mm']:.2f}" for h in history[::max(1, len(history) // 6)])
        rigid_desc = "rigid 6D (right)" if self.legacy else "rigid 12D (L+R)"
        self.inverse_info.content = (
            f"{rigid_desc} | init={self.init_gui.value} | target={self.target_gui.value}  \n"
            f"history EPE: {tail} mm")
        self._status(i, valid=True, inverse=True)

    def _render_inverse_placeholder(self, i, obj_mesh, hand_meshes, center):
        b, gap = self.bundle, self.gap
        spacing = auto_spacing(obj_mesh)
        offsets = panel_offsets(2, spacing)
        inputs = self._model_inputs(i)
        hp = inputs["hand_points"]
        init_flow = (inputs["hand_flow"] if self.init_gui.value == "GT"
                     else torch.zeros_like(inputs["hand_flow"]))
        init_hand = self._hand_meshes_from_flow(init_flow, hp, hand_meshes)
        with torch.no_grad():
            init_out = self._forward(inputs, init_flow)
        init_epe = float(epe(init_out["object_flow"], inputs["object_flow"])) * 1000
        init_next = self._next_object_mesh(init_out["object_flow"], inputs["object_points"],
                                           torch.from_numpy(obj_mesh).to(self.device))
        gt_hand = {side: b.hand_verts[side][i + gap] - center for side in ("left", "right")}
        gt_next = b.obj_world[i + gap] - center
        top = self._label_height(obj_mesh, gt_next, init_next)
        self._panel_meshes("/initial", np.array([float(offsets[0]), 0, 0]), obj_mesh,
                           init_hand, init_next, None)
        self._panel_meshes("/gt", np.array([float(offsets[1]), 0, 0]), obj_mesh,
                           gt_hand, None, gt_next, hand_color=COLOR_GT)
        self.batch.label("/initial/label", (float(offsets[0]), top, 0.0),
                         f"Initial  EPE {init_epe:.2f} mm")
        self.batch.label("/gt/label", (float(offsets[1]), top, 0.0), "GT action")
        self._status(i, valid=True, inverse=True, init_epe=init_epe)

    # ---------- 渲染元件 ----------
    def _panel_meshes(self, prefix: str, offset: np.ndarray, obj_mesh, hand_meshes,
                      pred_next, gt_next, hand_color=COLOR_CURRENT_HAND):
        opacity = float(self.next_opacity.value)
        if self.show_object.value and obj_mesh is not None:
            self.batch.mesh(f"{prefix}/object", obj_mesh + offset, self.bundle.obj_faces,
                            COLOR_CURRENT_OBJ)
        if self.show_hand.value and hand_meshes is not None:
            for side in ("left", "right"):
                mesh = hand_meshes.get(side) if isinstance(hand_meshes, dict) else None
                if mesh is not None:
                    self.batch.mesh(f"{prefix}/hand_{side}", mesh + offset,
                                    self.bundle.hand_faces[side], hand_color)
        if pred_next is not None:
            self.batch.mesh(f"{prefix}/pred_next", pred_next + offset, self.bundle.obj_faces,
                            COLOR_PRED, opacity=opacity)
        if gt_next is not None:
            self.batch.mesh(f"{prefix}/gt_next", gt_next + offset, self.bundle.obj_faces,
                            COLOR_GT, opacity=opacity)

    def _sparse_flows(self, prefix: str, offset: np.ndarray, inputs, pred_flow):
        i, gap = int(self.frame_gui.value), self.gap
        stride = int(self.flow_stride.value)
        center = self._center(i)
        op = self.bundle.obj_points_world[i, self.bundle.object_indices].astype(np.float32) - center
        if self.legacy:
            hp = self.bundle.hand_face_centers["right"][i].astype(np.float32) - center
        else:
            hp, _, _, _, _ = self.bundle.sample_hands(i, gap)
            hp = hp - center
        pred = pred_flow.squeeze(0).cpu().numpy()
        gt = inputs["object_flow"].squeeze(0).cpu().numpy()
        hand = inputs["hand_flow"].squeeze(0).cpu().numpy()
        self.batch.arrows(f"{prefix}/flow_pred", op + offset, pred, COLOR_PRED, stride)
        self.batch.arrows(f"{prefix}/flow_gt", op + offset, gt, COLOR_GT, stride)
        self.batch.arrows(f"{prefix}/flow_hand", hp + offset, hand, (90, 90, 200), stride)

    @staticmethod
    def _label_height(*meshes) -> float:
        tops = [float(m[:, 1].max()) for m in meshes if m is not None]
        return max(tops) + 0.05 if tops else 0.2

    def _status(self, i: int, valid: bool, pred_epe=None, inverse=False, init_epe=None):
        b = self.bundle
        gap = self.gap
        raw_i, raw_j = int(b.raw_frame_id[i]), int(b.raw_frame_id[min(i + gap, b.frames - 1)])
        valid_count = len(self._valid_frame_list())
        parity = f"  \n⚠ geometry parity {b.hand_parity_mm:.2f} mm" if b.hand_parity_mm > 0.01 else ""
        unavailable = ("Model prediction: **unavailable** — R active & L inactive required"
                       if self.legacy else
                       "Model prediction: **unavailable** — (R∨L) not active over window")
        lines = [
            f"**{b.name}** | object **{b.object_name}** | subject **{b.subject}** (ckpt epoch {self.epoch})",
            f"Cached frame **{i} / {b.frames - 1}** | raw **{raw_i} → {raw_j} / {b.raw_frames_total - 1}** "
            f"| Δt = **{gap}** ({gap / 30 * 1000:.0f} ms) | valid transitions **{valid_count}**",
            "Model prediction: **available**" if valid else unavailable,
        ]
        if valid and pred_epe is not None:
            lines.append(f"Pred EPE: **{pred_epe:.2f} mm**")
        if valid and inverse and self.result is not None:
            lines.append("Inverse EPE — initial **{:.2f}** / optimized **{:.2f}** / "
                         "GT action **{:.2f}** mm".format(
                             self.result["initial"]["epe_mm"],
                             self.result["optimized"]["epe_mm"],
                             self.result["gt"]["epe_mm"]))
        if init_epe is not None:
            lines.append(f"Initial EPE: **{init_epe:.2f} mm**")
        self.info.content = "  \n".join(lines) + parity

    # ---------- Inverse ----------
    def _optimize(self):
        with self.lock:
            i = int(self.frame_gui.value)
            if not self._valid(i):
                self.inverse_info.content = "当前帧不在 valid transition，无法优化"
                return
            inputs = self._model_inputs(i)
            init_flow = (inputs["hand_flow"] if self.init_gui.value == "GT"
                         else torch.zeros_like(inputs["hand_flow"]))
            self.inverse_info.content = "optimizing..."
            if self.legacy:
                with torch.no_grad():
                    static = self._encode(inputs)
                result = legacy_optimize_hand_flow(
                    self.model, static, inputs["hand_points"], inputs["object_flow"],
                    gt_flow=inputs["hand_flow"], init_flow=init_flow,
                    parameterization="rigid",
                    target_kind=self.target_gui.value.lower(),
                    steps=int(self.steps_gui.value), lr=0.01,
                )
            else:
                with torch.no_grad():
                    static = self._encode(inputs)
                result = optimize_hand_flow(
                    self.model, static, inputs["hand_points"], inputs["object_flow"],
                    gt_flow=inputs["hand_flow"], init_flow=init_flow,
                    dt=inputs["gap"], parameterization="rigid",
                    target_kind=self.target_gui.value.lower(),
                    steps=int(self.steps_gui.value), lr=0.01,
                )
            self.result = result
            self.result_frame = i
            self.render()

    # ---------- 主循环 ----------
    def run(self):
        print(f"trajectory viewer -> http://localhost:{self.args.port}", flush=True)
        while True:
            if self.playing:
                frame = int(self.frame_gui.value)
                if frame >= self.frame_gui.max:
                    self.frame_gui.value = 0
                else:
                    self.frame_gui.value = frame + 1
            time.sleep(1.0 / float(self.fps_gui.value))


def main() -> None:
    viewer = TrajectoryViewer(parse_args())
    viewer.run()


if __name__ == "__main__":
    main()
