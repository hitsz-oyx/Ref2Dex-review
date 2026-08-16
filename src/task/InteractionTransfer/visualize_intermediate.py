"""V0.9 Viser 中间过程可视化：ΔH → M_ij → C_obj → ΔO。

用法::

    python -m src.task.InteractionTransfer.visualize_intermediate \\
        --root <GRAB_CACHE_ROOT> \\
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \\
        --checkpoint outputs/InteractionTransfer/v08_full_20ep/best.pt \\
        --device cuda:0 --port 8080

V0 最小功能：
1. object / hand point cloud；2. hand flow arrows（stride）；3. M norm Top-K edges；
4. attention 边着色模式；5. C_obj norm heatmap；6. GT / Pred flow 箭头；
7. GT/Zero/Reverse/Cross action 实时切换（只改 hand_flow 重 forward）；
8. 当前 sample EPE。
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path

import numpy as np
import torch

import viser

from src.task.InteractionTransfer.dataset import GRABOneStepDataset
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
    """0~1 标量映射到 viridis RGB (N,3) uint8。"""
    import matplotlib.cm as cm
    values = np.clip(values, 0.0, 1.0)
    return (cm.viridis(values)[:, :3] * 255).astype(np.uint8)


def norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-12)


class Viewer:
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
        self.dataset = None
        self.static = None          # encode_static 输出（action 无关，可复用）
        self.gt_flow = None
        self.cross_flow = None
        self.out = None             # 当前 action 的 forward_core 输出

        self.server = viser.ViserServer(host="0.0.0.0", port=args.port)
        self._build_gui()
        self._load_sequence(args.sequence or self.sequences[0])

    # ---------- GUI ----------
    def _build_gui(self):
        # viser 1.0.30 控件只能挂在 server.gui 上，用 markdown 标题分组
        gui = self.server.gui
        gui.add_markdown("# InteractionTransfer Viewer")
        self.info = gui.add_markdown("")

        gui.add_markdown("## Sample")
        self.seq_gui = gui.add_dropdown("Sequence", self.sequences)
        self.seq_gui.on_update(lambda _: self._load_sequence(self.seq_gui.value))
        self.tr_gui = gui.add_slider("Transition", min=0, max=1, step=1, initial_value=0)
        self.tr_gui.on_update(lambda _: self.refresh(recompute_static=True))

        gui.add_markdown("## Action")
        self.action_gui = gui.add_button_group("Action", ("GT", "Zero", "Reverse", "Cross"))
        self.action_gui.on_click(lambda _: self.refresh(recompute_static=False))

        gui.add_markdown("## Scene")
        self.show_object = gui.add_checkbox("Object", True)
        self.show_object.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_hand = gui.add_checkbox("Hand", True)
        self.show_hand.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_hand_flow = gui.add_checkbox("Hand Flow", True)
        self.show_hand_flow.on_update(lambda _: self.refresh(recompute_static=False))
        self.stride_gui = gui.add_slider("Hand flow stride", min=1, max=16, step=1, initial_value=8)
        self.stride_gui.on_update(lambda _: self.refresh(recompute_static=False))

        gui.add_markdown("## Edge Interaction")
        self.show_edges = gui.add_checkbox("Edges", True)
        self.show_edges.on_update(lambda _: self.refresh(recompute_static=False))
        self.edge_mode = gui.add_button_group("Display", ("||M||", "Attention"))
        self.edge_mode.on_click(lambda _: self.refresh(recompute_static=False))
        self.topk_gui = gui.add_dropdown("Edges", ("Top 100", "Top 300", "Top 500"))
        self.topk_gui.on_update(lambda _: self.refresh(recompute_static=False))

        gui.add_markdown("## Object Field")
        self.show_field = gui.add_checkbox("C_obj norm heatmap", True)
        self.show_field.on_update(lambda _: self.refresh(recompute_static=False))

        gui.add_markdown("## Effect")
        self.show_pred = gui.add_checkbox("Pred Flow", True)
        self.show_pred.on_update(lambda _: self.refresh(recompute_static=False))
        self.show_gt = gui.add_checkbox("GT Flow", False)
        self.show_gt.on_update(lambda _: self.refresh(recompute_static=False))

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
                self.refresh(recompute_static=True)
            finally:
                self._loading = False

    def _current_sample(self):
        idx = int(self.tr_gui.value)
        return self.dataset[idx], idx

    @torch.no_grad()
    def _forward(self, recompute_static: bool):
        sample, idx = self._current_sample()

        def to_batch(x):
            t = x if torch.is_tensor(x) else torch.from_numpy(x)
            return t.unsqueeze(0).float().to(self.device)

        op, on = to_batch(sample["object_points"]), to_batch(sample["object_normals"])
        hp, hn = to_batch(sample["hand_points"]), to_batch(sample["hand_normals"])
        flow = to_batch(sample["hand_flow"])
        if recompute_static or self.static is None:
            self.static = self.model.encode_static(op, on, hp, hn)
            self.gt_flow = flow
            cross = self.dataset[(idx + 1) % len(self.dataset)]
            self.cross_flow = to_batch(cross["hand_flow"])
            self.object_np = sample["object_points"].cpu().numpy() if torch.is_tensor(sample["object_points"]) else sample["object_points"]
            self.hand_np = sample["hand_points"].cpu().numpy() if torch.is_tensor(sample["hand_points"]) else sample["hand_points"]
            self.gt_object_flow_np = sample["object_flow"].cpu().numpy() if torch.is_tensor(sample["object_flow"]) else sample["object_flow"]
            self.gt_obj_flow_t = to_batch(sample["object_flow"])

        action = self.action_gui.value
        if action == "GT":
            a = self.gt_flow
        elif action == "Zero":
            a = torch.zeros_like(self.gt_flow)
        elif action == "Reverse":
            a = -self.gt_flow
        else:
            a = self.cross_flow
        self.out = self.model.forward_core(hand_flow=a, **self.static)
        self.sample_epe = float(epe(self.out["object_flow"], self.gt_obj_flow_t)) * 1000
        self.action = action

    # ---------- 场景 ----------
    @staticmethod
    def _remove(scene, name: str):
        try:
            scene.remove_by_name(name)
        except KeyError:
            pass

    def refresh(self, recompute_static: bool):
        with self.lock:
            if self.dataset is None:
                return
            self._forward(recompute_static)
            out = {k: v.squeeze(0).detach().cpu().numpy() for k, v in self.out.items()}
            self._render(out)

    def _render(self, out):
        scene = self.server.scene
        op, hp = self.object_np, self.hand_np
        pred_flow = out["object_flow"]
        gt_flow = self.gt_object_flow_np

        self.info.content = (
            f"**Checkpoint** {Path(self.args.checkpoint).name} \\| epoch {self.epoch}  \n"
            f"**Current Action** {self.action}  \n"
            f"**Sample EPE** {self.sample_epe:.2f} mm"
        )

        # Layer 3: object field heatmap
        if self.show_field.value:
            colors = viridis(norm01(np.linalg.norm(out["object_field"], axis=1)))
        else:
            colors = np.tile(np.array([160, 160, 160], np.uint8), (len(op), 1))
        if self.show_object.value:
            scene.add_point_cloud("/object", points=op, colors=colors, point_size=0.004)
        else:
            self._remove(scene, "/object")

        # Layer 1: hand + hand flow arrows
        if self.show_hand.value:
            hand_colors = np.tile(np.array([200, 200, 210], np.uint8), (len(hp), 1))
            scene.add_point_cloud("/hand", points=hp, colors=hand_colors, point_size=0.003)
        else:
            self._remove(scene, "/hand")

        if self.show_hand_flow.value:
            stride = int(self.stride_gui.value)
            hand_flow = self._current_action_flow_np()
            idx = np.arange(0, len(hp), stride)
            segs = np.stack([hp[idx], hp[idx] + hand_flow[idx]], axis=1)
            mag = norm01(np.linalg.norm(hand_flow[idx], axis=1))
            colors = viridis(mag)[:, None, :].repeat(2, axis=1)
            scene.add_line_segments("/hand_flow", points=segs, colors=colors, line_width=2.0)
        else:
            self._remove(scene, "/hand_flow")

        # Layer 2: edges（Top-K，仅 valid）。edge_message [N,K,32]，
        # edge_index [N,K] 存 hand 索引，attention [N,K]。
        if self.show_edges.value:
            valid = out["edge_valid"].astype(bool)
            edge_idx = out["edge_index"]
            metric = (np.linalg.norm(out["edge_message"], axis=-1) if self.edge_mode.value == "||M||"
                      else out["attention"])
            n_obj, k_nn = edge_idx.shape
            metric = np.where(valid, metric, -np.inf).reshape(-1)
            k = int(self.topk_gui.value.split()[-1])
            k = min(k, int(valid.sum()))
            top = np.argsort(metric)[::-1][:k]
            obj_i = top // k_nn
            hand_j = edge_idx.reshape(-1)[top]
            segs = np.stack([op[obj_i], hp[hand_j]], axis=1)
            strength = norm01(metric[top])
            # 强度越大越红：黑红 -> 亮红
            colors = (strength[:, None] * np.array([[255, 40, 40]])).astype(np.uint8)
            colors = np.repeat(colors[:, None, :], 2, axis=1)
            scene.add_line_segments("/edges", points=segs, colors=colors, line_width=2.0)
        else:
            self._remove(scene, "/edges")

        # Layer 4: object flow（GT / Pred）
        if self.show_pred.value:
            segs = np.stack([op, op + pred_flow], axis=1)
            colors = np.tile(np.array([255, 170, 0], np.uint8), (len(op), 2, 1))
            scene.add_line_segments("/flow_pred", points=segs, colors=colors, line_width=2.0)
        else:
            self._remove(scene, "/flow_pred")
        if self.show_gt.value:
            segs = np.stack([op, op + gt_flow], axis=1)
            colors = np.tile(np.array([0, 200, 120], np.uint8), (len(op), 2, 1))
            scene.add_line_segments("/flow_gt", points=segs, colors=colors, line_width=2.0)
        else:
            self._remove(scene, "/flow_gt")

    def _current_action_flow_np(self):
        action = self.action_gui.value
        f = self.gt_flow if action == "GT" else (
            torch.zeros_like(self.gt_flow) if action == "Zero" else
            (-self.gt_flow if action == "Reverse" else self.cross_flow))
        return f.squeeze(0).cpu().numpy()


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
