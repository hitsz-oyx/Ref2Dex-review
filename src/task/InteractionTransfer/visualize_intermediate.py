"""V0.10 Forward + Inverse 一体化 Viser viewer。

用法::

    python -m src.task.InteractionTransfer.visualize_intermediate \\
        --root <GRAB_CACHE_ROOT> \\
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \\
        --checkpoint outputs/InteractionTransfer/v08_full_20ep/best.pt \\
        --device cuda:0 --port 8080

两种 View Mode：
- Intermediate：ΔH → M_ij → C_obj → ΔO 单场景中间表示分析；
- Inverse Optimization：Initial / Optimized / GT 三路并排，冻结 model
  优化 hand action（Rigid SE(3) / Free Point Flow，Effect / C_obj target）。
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import torch

import viser

from src.task.InteractionTransfer.dataset import GRABOneStepDataset
from src.task.InteractionTransfer.inverse_optimize import optimize_hand_flow
from src.task.InteractionTransfer.metrics import epe
from src.task.InteractionTransfer.model import InteractionTransfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="GRAB stage4 cache 目录")
    parser.add_argument("--split", type=Path, required=True, help="test.txt")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--sequence", default=None, help="初始序列，默认 test.txt 第一个")
    return parser.parse_args()


def viridis(values: np.ndarray) -> np.ndarray:
    import matplotlib.cm as cm
    values = np.clip(values, 0.0, 1.0)
    return (cm.viridis(values)[:, :3] * 255).astype(np.uint8)


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-12)


class Viewer:
    # Inverse 三路并排的 x 偏移
    X_INIT, X_OPT, X_GT = 0.0, 0.6, 1.2

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.model = InteractionTransfer(dense_checkpoint=args.dense_checkpoint).to(self.device).eval()
        ckpt = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        current = self.model.state_dict()
        current.update(ckpt["model"])
        self.model.load_state_dict(current)
        self.epoch = int(ckpt.get("epoch", -1))

        self.sequences = [line.strip() for line in args.split.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.lock = threading.RLock()
        self._loading = False
        self._optimizing = False
        self.dataset = None
        self.static = None
        self.gt_flow = None
        self.cross_flow = None
        self.out = None
        self.result = None            # optimize_hand_flow 结果

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self._build_gui()
        self._load_sequence(args.sequence or self.sequences[0])

    # ---------- GUI ----------
    def _build_gui(self):
        gui = self.server.gui
        gui.add_markdown("# InteractionTransfer Viewer")
        self.info = gui.add_markdown("")

        gui.add_markdown("## View Mode")
        self.view_mode = gui.add_button_group("Mode", ("Intermediate", "Inverse Optimization"))
        self.view_mode.on_click(lambda _: self.refresh(recompute_static=False))

        gui.add_markdown("## Sample")
        self.seq_gui = gui.add_dropdown("Sequence", self.sequences)
        self.seq_gui.on_update(lambda _: self._load_sequence(self.seq_gui.value))
        self.tr_gui = gui.add_slider("Transition", min=0, max=1, step=1, initial_value=0)
        self.tr_gui.on_update(lambda _: self.refresh(recompute_static=True))

        gui.add_markdown("## Rendering")
        self.point_shape = gui.add_dropdown("Point Shape",
                                            ("circle", "rounded", "square", "diamond", "sparkle"))
        self.point_shape.on_update(lambda _: self.refresh(recompute_static=False))
        self.object_point_size = gui.add_slider("Object Point Size", min=0.0005, max=0.01,
                                                step=0.0005, initial_value=0.004)
        self.object_point_size.on_update(lambda _: self.refresh(recompute_static=False))
        self.hand_point_size = gui.add_slider("Hand Point Size", min=0.0005, max=0.01,
                                              step=0.0005, initial_value=0.003)
        self.hand_point_size.on_update(lambda _: self.refresh(recompute_static=False))

        gui.add_markdown("## Intermediate Mode")
        self.action_gui = gui.add_button_group("Action", ("GT", "Zero", "Reverse", "Cross"))
        self.action_gui.on_click(lambda _: self.refresh(recompute_static=False))
        self.show_object = gui.add_checkbox("Object", True)
        self.show_object.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_hand = gui.add_checkbox("Hand", True)
        self.show_hand.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_hand_flow = gui.add_checkbox("Hand Flow", True)
        self.show_hand_flow.on_update(lambda _: self.refresh(recompute_static=False))
        self.stride_gui = gui.add_slider("Hand flow stride", min=1, max=16, step=1, initial_value=8)
        self.stride_gui.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_edges = gui.add_checkbox("Edges", True)
        self.show_edges.on_update(lambda _: self.refresh(recompute_static=False))
        self.edge_mode = gui.add_button_group("Display", ("||M||", "Attention"))
        self.edge_mode.on_click(lambda _: self.refresh(recompute_static=False))
        self.topk_gui = gui.add_dropdown("Edges", ("Top 100", "Top 300", "Top 500"))
        self.topk_gui.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_field = gui.add_checkbox("C_obj norm heatmap", True)
        self.show_field.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_pred = gui.add_checkbox("Pred Flow", True)
        self.show_pred.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_gt = gui.add_checkbox("GT Flow", False)
        self.show_gt.on_update(lambda _: self.refresh(recompute_static=False))

        gui.add_markdown("## Inverse Optimization")
        self.opt_status = gui.add_markdown("")
        self.init_gui = gui.add_button_group("Initialization", ("Zero", "Cross", "Random", "GT"))
        self.init_gui.on_click(lambda _: self._on_inverse_param_change())
        self.param_gui = gui.add_button_group("Parameterization", ("Rigid SE(3)", "Free Point Flow"))
        self.param_gui.on_click(lambda _: self._on_inverse_param_change())
        self.target_gui = gui.add_button_group("Target", ("Object Effect", "C_obj"))
        self.target_gui.on_click(lambda _: self._on_inverse_param_change())
        self.steps_gui = gui.add_number("Steps", initial_value=300)
        self.lr_gui = gui.add_number("Learning Rate", initial_value=0.01)
        self.run_button = gui.add_button("Optimize")
        self.run_button.on_click(lambda _: self._start_optimize())
        self.reset_button = gui.add_button("Reset")
        self.reset_button.on_click(lambda _: self._on_inverse_param_change())
        self.show_cdiff = gui.add_checkbox("C_obj difference heatmap (opt vs GT)", True)
        self.show_cdiff.on_update(lambda _: self.refresh(recompute_static=False))

    # ---------- 数据 / forward ----------
    def _load_sequence(self, sequence: str):
        if self._loading:
            return
        with self.lock:
            self._loading = True
            try:
                self.dataset = GRABOneStepDataset(self.args.root, [sequence], max_transitions=0)
                if self.seq_gui.value != sequence:
                    self.seq_gui.value = sequence
                self.tr_gui.max = max(len(self.dataset) - 1, 1)
                self.tr_gui.value = 0
                self.static = None
                self.result = None
                self.refresh(recompute_static=True)
            finally:
                self._loading = False

    def _current_sample(self):
        idx = int(self.tr_gui.value)
        return self.dataset[idx], idx

    def _to_batch(self, x):
        t = x if torch.is_tensor(x) else torch.from_numpy(x)
        return t.unsqueeze(0).float().to(self.device)

    @torch.no_grad()
    def _forward(self, recompute_static: bool):
        sample, idx = self._current_sample()
        if recompute_static or self.static is None:
            op, on = self._to_batch(sample["object_points"]), self._to_batch(sample["object_normals"])
            hp, hn = self._to_batch(sample["hand_points"]), self._to_batch(sample["hand_normals"])
            self.static = self.model.encode_static(op, on, hp, hn)
            self.hand_points_t = hp
            self.gt_flow = self._to_batch(sample["hand_flow"])
            cross = self.dataset[(idx + 1) % len(self.dataset)]
            self.cross_flow = self._to_batch(cross["hand_flow"])
            self.object_np = sample["object_points"].cpu().numpy() if torch.is_tensor(sample["object_points"]) else sample["object_points"]
            self.hand_np = sample["hand_points"].cpu().numpy() if torch.is_tensor(sample["hand_points"]) else sample["hand_points"]
            self.gt_object_flow_t = self._to_batch(sample["object_flow"])

        action = self.action_gui.value
        a = {"GT": self.gt_flow, "Zero": torch.zeros_like(self.gt_flow),
             "Reverse": -self.gt_flow, "Cross": self.cross_flow}[action]
        self.out = self.model.forward_core(hand_flow=a, **self.static)
        self.sample_epe = float(epe(self.out["object_flow"], self.gt_object_flow_t)) * 1000
        self.action = action

    def _init_flow_for_inverse(self) -> torch.Tensor:
        kind = self.init_gui.value
        if kind == "Zero":
            return torch.zeros_like(self.gt_flow)
        if kind == "Cross":
            return self.cross_flow.clone()
        if kind == "Random":
            return (0.01 * torch.randn_like(self.gt_flow)).clamp(-0.03, 0.03)
        return self.gt_flow.clone()

    def _on_inverse_param_change(self):
        self.result = None
        self.refresh(recompute_static=False)

    def _start_optimize(self):
        if self._optimizing or self.static is None:
            return

        def worker():
            with self.lock:
                self._optimizing = True
                self.run_button.disabled = True
                try:
                    param = "rigid" if self.param_gui.value == "Rigid SE(3)" else "free"
                    target_kind = "effect" if self.target_gui.value == "Object Effect" else "c_obj"
                    self.result = optimize_hand_flow(
                        self.model, self.static, self.hand_points_t, self.gt_object_flow_t,
                        gt_flow=self.gt_flow, init_flow=self._init_flow_for_inverse(),
                        parameterization=param, target_kind=target_kind,
                        steps=int(self.steps_gui.value), lr=float(self.lr_gui.value),
                        history_every=25)
                    self._render_inverse()
                except Exception as exc:  # 浏览器可见错误，不静默
                    self.opt_status.content = f"**Optimize failed**\n\n```\n{exc}\n```"
                finally:
                    self._optimizing = False
                    self.run_button.disabled = False

        threading.Thread(target=worker, daemon=True).start()

    # ---------- 场景 ----------
    @staticmethod
    def _remove(scene, name: str):
        try:
            scene.remove_by_name(name)
        except KeyError:
            pass

    def _clear_scene(self):
        for name in ("/object", "/hand", "/hand_flow", "/edges", "/flow_pred", "/flow_gt",
                     "/init/object", "/init/hand", "/init/hand_flow", "/init/flow", "/init/label",
                     "/opt/object", "/opt/hand", "/opt/hand_flow", "/opt/flow", "/opt/label",
                     "/gt/object", "/gt/hand", "/gt/hand_flow", "/gt/flow", "/gt/label"):
            self._remove(self.server.scene, name)

    def refresh(self, recompute_static: bool):
        with self.lock:
            if self.dataset is None:
                return
            self._clear_scene()
            if self.view_mode.value == "Inverse Optimization":
                if self.result is None:
                    self._render_inverse_placeholder()
                else:
                    self._render_inverse()
            else:
                self._forward(recompute_static)
                out = {k: v.squeeze(0).detach().cpu().numpy() for k, v in self.out.items()}
                self._render_intermediate(out)

    # ----- Intermediate 渲染（V0.9 原有） -----
    def _render_intermediate(self, out):
        scene = self.server.scene
        op, hp = self.object_np, self.hand_np
        pred_flow = out["object_flow"]
        gt_flow = self.gt_object_flow_t.squeeze(0).cpu().numpy()

        self.info.content = (
            f"**Checkpoint** {Path(self.args.checkpoint).name} \\| epoch {self.epoch}  \n"
            f"**Mode** Intermediate \\| **Action** {self.action}  \n"
            f"**Sample EPE** {self.sample_epe:.2f} mm"
        )

        if self.show_field.value:
            colors = viridis(norm01(np.linalg.norm(out["object_field"], axis=1)))
        else:
            colors = np.tile(np.array([160, 160, 160], np.uint8), (len(op), 1))
        if self.show_object.value:
            scene.add_point_cloud("/object", points=op, colors=colors,
                                  point_size=float(self.object_point_size.value),
                                  point_shape=self.point_shape.value, point_shading="gradient")
        if self.show_hand.value:
            hand_colors = np.tile(np.array([200, 200, 210], np.uint8), (len(hp), 1))
            scene.add_point_cloud("/hand", points=hp, colors=hand_colors,
                                  point_size=float(self.hand_point_size.value),
                                  point_shape=self.point_shape.value, point_shading="gradient")

        if self.show_hand_flow.value:
            stride = int(self.stride_gui.value)
            flow = {"GT": self.gt_flow, "Zero": torch.zeros_like(self.gt_flow),
                    "Reverse": -self.gt_flow, "Cross": self.cross_flow}[self.action_gui.value]
            flow = flow.squeeze(0).cpu().numpy()
            idx = np.arange(0, len(hp), stride)
            segs = np.stack([hp[idx], hp[idx] + flow[idx]], axis=1)
            mag = norm01(np.linalg.norm(flow[idx], axis=1))
            colors = viridis(mag)[:, None, :].repeat(2, axis=1)
            scene.add_line_segments("/hand_flow", points=segs, colors=colors, line_width=2.0)

        if self.show_edges.value:
            valid = out["edge_valid"].astype(bool)
            edge_idx = out["edge_index"]
            metric = (np.linalg.norm(out["edge_message"], axis=-1) if self.edge_mode.value == "||M||"
                      else out["attention"])
            n_obj, k_nn = edge_idx.shape
            metric = np.where(valid, metric, -np.inf).reshape(-1)
            k = min(int(self.topk_gui.value.split()[-1]), int(valid.sum()))
            top = np.argsort(metric)[::-1][:k]
            obj_i, hand_j = top // k_nn, edge_idx.reshape(-1)[top]
            segs = np.stack([op[obj_i], hp[hand_j]], axis=1)
            strength = norm01(metric[top])
            colors = (strength[:, None] * np.array([[255, 40, 40]])).astype(np.uint8)
            colors = np.repeat(colors[:, None, :], 2, axis=1)
            scene.add_line_segments("/edges", points=segs, colors=colors, line_width=2.0)

        if self.show_pred.value:
            segs = np.stack([op, op + pred_flow], axis=1)
            colors = np.tile(np.array([255, 170, 0], np.uint8), (len(op), 2, 1))
            scene.add_line_segments("/flow_pred", points=segs, colors=colors, line_width=2.0)
        if self.show_gt.value:
            segs = np.stack([op, op + gt_flow], axis=1)
            colors = np.tile(np.array([0, 200, 120], np.uint8), (len(op), 2, 1))
            scene.add_line_segments("/flow_gt", points=segs, colors=colors, line_width=2.0)

    # ----- Inverse 渲染（V0.10 三路并排） -----
    def _render_inverse_placeholder(self):
        """尚未 Optimize：只显示 Initial（当前 init action 的 forward）与 GT 两路。"""
        init_flow = self._init_flow_for_inverse()
        with torch.no_grad():
            init_out = self.model.forward_core(hand_flow=init_flow, **self.static)
            gt_out = self.model.forward_core(hand_flow=self.gt_flow, **self.static)
        init = {"flow": init_flow.squeeze(0).cpu().numpy(),
                "object_flow": init_out["object_flow"].squeeze(0).cpu().numpy(),
                "object_field": init_out["object_field"].squeeze(0).cpu().numpy(),
                "epe_mm": float(epe(init_out["object_flow"], self.gt_object_flow_t)) * 1000}
        gt = {"flow": self.gt_flow.squeeze(0).cpu().numpy(),
              "object_flow": gt_out["object_flow"].squeeze(0).cpu().numpy(),
              "object_field": gt_out["object_field"].squeeze(0).cpu().numpy(),
              "epe_mm": float(epe(gt_out["object_flow"], self.gt_object_flow_t)) * 1000}
        self._render_inverse(init=init, optimized=None, gt=gt, history=None)

    def _render_inverse(self, init=None, optimized=None, gt=None, history=None):
        scene = self.server.scene
        if init is None:  # 已有优化结果
            r = self.result
            init = {k: v.squeeze(0).numpy() if torch.is_tensor(v) else v for k, v in r["initial"].items()}
            optimized = {k: v.squeeze(0).numpy() if torch.is_tensor(v) else v for k, v in r["optimized"].items()}
            gt = {k: v.squeeze(0).numpy() if torch.is_tensor(v) else v for k, v in r["gt"].items()}
            history = r["history"]

        op, hp = self.object_np, self.hand_np
        stride = 8
        hidx = np.arange(0, len(hp), stride)

        def render_column(prefix, x, data, flow_color, obj_color, label):
            scene.add_point_cloud(f"{prefix}/object", points=op + np.array([x, 0, 0]),
                                  colors=np.tile(np.array([170, 170, 170], np.uint8), (len(op), 1)),
                                  point_size=float(self.object_point_size.value),
                                  point_shape=self.point_shape.value, point_shading="gradient")
            scene.add_point_cloud(f"{prefix}/hand", points=hp + np.array([x, 0, 0]),
                                  colors=np.tile(np.array([205, 205, 215], np.uint8), (len(hp), 1)),
                                  point_size=float(self.hand_point_size.value),
                                  point_shape=self.point_shape.value, point_shading="gradient")
            flow = data["flow"]
            segs = np.stack([hp[hidx] + x, hp[hidx] + flow[hidx] + x], axis=1)
            colors = np.tile(np.array(flow_color, np.uint8), (len(hidx), 2, 1))
            scene.add_line_segments(f"{prefix}/hand_flow", points=segs, colors=colors, line_width=2.0)
            of = data["object_flow"]
            segs = np.stack([op + x, op + of + x], axis=1)
            colors = np.tile(np.array(obj_color, np.uint8), (len(op), 2, 1))
            scene.add_line_segments(f"{prefix}/flow", points=segs, colors=colors, line_width=2.0)
            scene.add_label(f"{prefix}/label", text=label, position=(x, 0.32, 0))

        # opt 路点云按 C_obj difference 着色（opt vs GT）
        diff = None
        if optimized is not None:
            diff = np.linalg.norm(optimized["object_field"] - gt["object_field"], axis=-1)

        render_column("/init", self.X_INIT, init,
                      (120, 150, 255), (150, 150, 150), f"Initial  EPE {init['epe_mm']:.2f} mm")
        if optimized is not None:
            render_column("/opt", self.X_OPT, optimized,
                          (255, 140, 0), (255, 140, 0), f"Optimized  EPE {optimized['epe_mm']:.2f} mm")
        render_column("/gt", self.X_GT, gt,
                      (0, 200, 120), (0, 200, 120), f"GT action  EPE {gt['epe_mm']:.2f} mm")

        if diff is not None:
            if self.show_cdiff.value:
                scene.add_point_cloud("/opt/object", points=op + np.array([self.X_OPT, 0, 0]),
                                      colors=viridis(norm01(diff)),
                                      point_size=float(self.object_point_size.value),
                                      point_shape=self.point_shape.value, point_shading="gradient")
                cdiff_text = f"mean \\|C_opt - C_gt\\| = {diff.mean():.4f}"
            else:
                cdiff_text = f"mean \\|C_opt - C_gt\\| = {diff.mean():.4f}（heatmap 关闭）"
        else:
            cdiff_text = ""

        lines = [
            f"**Mode** Inverse Optimization \\| init `{self.init_gui.value}` \\| "
            f"`{self.param_gui.value}` \\| target `{self.target_gui.value}`  \n"
            f"Initial EPE: **{init['epe_mm']:.2f} mm**"
        ]
        if optimized is not None:
            lines[0] += f"  →  Optimized EPE: **{optimized['epe_mm']:.2f} mm**"
            lines.append(f"GT-action EPE: {gt['epe_mm']:.2f} mm \\| "
                         f"Improvement: {init['epe_mm'] - optimized['epe_mm']:.2f} mm")
        if cdiff_text:
            lines.append(cdiff_text)
        if history:
            curve = " → ".join(f"{h['step']}:{h['epe_mm']:.1f}" for h in history)
            lines.append(f"history (step:EPE mm): {curve}")
        self.opt_status.content = "\n".join(lines)
        self.info.content = (
            f"**Checkpoint** {Path(self.args.checkpoint).name} \\| epoch {self.epoch}  \n"
            f"**Mode** Inverse Optimization（点 Optimize 运行）"
        )


def main() -> None:
    args = parse_args()
    viewer = Viewer(args)
    print(f"viser server -> http://localhost:{args.port}", flush=True)
    try:
        import time
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
