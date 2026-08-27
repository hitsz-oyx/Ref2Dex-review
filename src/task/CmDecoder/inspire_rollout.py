"""Action-conditioned closed-loop rollout for an Inspire F1 CmDecoder.

At every step the frozen Cm encoder receives the *ground-truth* Inspire
hand-flow for that pair, while the point-flow decoder receives the previously
predicted Inspire state.  This isolates decoder/state-feedback drift from the
separate problem of predicting the next Cm action.  The command writes a
trajectory NPZ, a diagnostic PNG, and can optionally serve the trajectory in a
small Viser viewer.
"""
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

from src.base.checkpoint import load_checkpoint
from src.base.base_config import task_config_from_dict
from src.base.utils import import_from_path
from src.task.CmDecoder.dataset import (
    _load_hrdex_io,
    _robot_hand_mesh,
    _rotate_to_frame,
    _to_frame,
)
from src.task.CmDecoder.q_optimizer import (
    DifferentiableInspireHand,
    optimize_q_from_hand_points,
    rotvec_to_matrix,
)
from src.task.CmDecoder.viewer import CmDecoderViewerData, _resolve_path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT / "outputs/cmdecoder/cm_decoder_20260823_000423/checkpoints/best.pt"
DEFAULT_MANIFEST = ROOT / "data/processed_data/cm_decoder/hrdexdb_inspire_f1_3hz/v2/selection_576_object_disjoint_seed42.json"
DEFAULT_30HZ_MANIFEST = ROOT / "data/processed_data/cm_decoder/hrdexdb_inspire_f1/v4/selection_576_seed42.json"
DEFAULT_30HZ_CACHE_ROOT = ROOT / "data/processed_data/cm_decoder/hrdexdb_inspire_f1"


def _local_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    relative = pred[:3, :3].T @ target[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _contact_ratio(hand: np.ndarray, obj: np.ndarray, threshold_m: float) -> float:
    distances, _ = cKDTree(obj).query(hand, k=1)
    return float(np.mean(distances < float(threshold_m)))


def _select_rollout_start(data: CmDecoderViewerData, threshold_rad: float = np.deg2rad(0.5)) -> int:
    """Pick an active start with the longest contiguous valid-pair chain."""
    q_delta = np.asarray(data.task_arrays["q_delta_abs_max"], dtype=np.float32)
    row_by_source = {int(source): row for row, source in enumerate(data.current_indices)}
    best_row, best_len, best_motion = 0, -1, -1.0
    for row in range(len(data)):
        if float(q_delta[row]) < float(threshold_rad):
            continue
        length = 0
        current = row
        while current is not None and current < len(data):
            length += 1
            current = row_by_source.get(int(data.target_indices[current]))
        motion = float(q_delta[row])
        if (length, motion) > (best_len, best_motion):
            best_row, best_len, best_motion = row, length, motion
    return int(best_row)


class InspireRollout:
    """Run one action-conditioned rollout on the cached Inspire sequence."""

    def __init__(self, checkpoint: Path, episode: str, device: str) -> None:
        self.data = CmDecoderViewerData(
            checkpoint,
            episode.split("/")[1],
            episode.split("/")[2],
            device=device,
            manifest_path=DEFAULT_30HZ_MANIFEST,
            cache_root=DEFAULT_30HZ_CACHE_ROOT,
        )
        self.device = self.data.device
        self.model = self.data.model.eval()
        self.hand_model = DifferentiableInspireHand(
            self.data.cfg.meta.robot_urdf,
            num_hand_points=int(self.data.cfg.meta.num_hand_points),
            sample_seed=int(self.data.cfg.meta.sample_seed),
            device=self.device,
        )
        self.mesh_cache: dict = {}
        self.io = self.data.io
        self.urdf = self.data.urdf

    def _mesh_world(self, source_index: int, q: np.ndarray, wrist_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_full = np.asarray(self.data.q_full[source_index], dtype=np.float32).copy()
        q_full[6:12] = np.asarray(q, dtype=np.float32)
        vertices, faces = _robot_hand_mesh(self.io, self.urdf, q_full, self.mesh_cache)
        vertices_base_world = vertices @ self.data.c2r[:3, :3].T + self.data.c2r[:3, 3]
        source_wrist = np.asarray(self.data.wrist_pose_world[source_index], dtype=np.float32)
        source_local = (vertices_base_world - source_wrist[:3, 3]) @ source_wrist[:3, :3]
        return _local_to_world(source_local, wrist_pose).astype(np.float32), faces

    def run(self, start_pair: int, frames: int, contact_threshold_m: float = 0.02) -> dict[str, np.ndarray]:
        start_pair = int(start_pair)
        if start_pair < 0 or start_pair >= len(self.data):
            raise ValueError(f"start_pair must be in [0, {len(self.data) - 1}]")
        row_by_source = {int(source): row for row, source in enumerate(self.data.current_indices)}
        stride = int(self.data.horizon_stride)
        rows: list[int] = []
        row = start_pair
        for _ in range(int(frames)):
            if row >= len(self.data):
                break
            source = int(self.data.current_indices[row])
            target = int(self.data.target_indices[row])
            rows.append(row)
            next_row = row_by_source.get(target)
            if next_row is None:
                break
            row = next_row
        if not rows:
            raise ValueError("No valid rollout pairs")

        first = rows[0]
        predicted_q = np.asarray(self.data.task_arrays["q_t"][first], dtype=np.float32).copy()
        predicted_wrist = np.asarray(self.data.wrist_pose_world[int(self.data.current_indices[first])], dtype=np.float32).copy()
        predicted_points = np.asarray(self.data.task_arrays["hand_points"][first], dtype=np.float32).copy()
        predicted_normals = np.asarray(self.data.task_arrays["hand_normals"][first], dtype=np.float32).copy()
        pred_world_all, gt_world_all, obj_world_all = [], [], []
        pred_mesh_all, gt_mesh_all = [], []
        pred_q_all, gt_q_all = [], []
        pred_wrist_all, gt_wrist_all = [], []
        point_epe, q_mae, wrist_epe, wrist_rot, pred_contact, gt_contact = [], [], [], [], [], []
        source_ids, target_ids, cm_flow_norm = [], [], []
        faces = None

        for row in rows:
                source_index = int(self.data.current_indices[row])
                target_index = int(self.data.target_indices[row])
                gt_batch = self.data._batch(row)
                # Cm is intentionally computed from the recorded action and
                # recorded source state; only the decoder state is rolled out.
                with torch.no_grad():
                    _, _, cm_tokens = self.model._frozen_features(gt_batch)
                object_world = np.asarray(self.data.obj_points_world[source_index], dtype=np.float32)
                object_normals_world = np.asarray(self.data.obj_normals_world[source_index], dtype=np.float32)
                decode_batch = dict(gt_batch)
                decode_batch["cm_tokens"] = cm_tokens
                decode_batch["hand_points"] = torch.from_numpy(predicted_points)[None].to(self.device)
                decode_batch["hand_normals"] = torch.from_numpy(predicted_normals)[None].to(self.device)
                decode_batch["obj_points"] = torch.from_numpy(_to_frame(object_world, predicted_wrist))[None].to(self.device)
                decode_batch["obj_normals"] = torch.from_numpy(_rotate_to_frame(object_normals_world, predicted_wrist))[None].to(self.device)
                decode_batch["q_t"] = torch.from_numpy(predicted_q)[None].to(self.device)
                with torch.no_grad():
                    prediction = self.model(decode_batch)
                    predicted_flow = prediction["pred_hand_flow"]
                target_points = decode_batch["hand_points"] + predicted_flow
                fit = optimize_q_from_hand_points(
                    self.hand_model,
                    q_t=decode_batch["q_t"],
                    current_hand_points=decode_batch["hand_points"],
                    target_hand_points=target_points,
                    steps=int(getattr(self.data.cfg.meta, "q_fit_steps", 100)),
                    lr=float(getattr(self.data.cfg.meta, "q_fit_lr", 0.05)),
                    prior_weight=float(getattr(self.data.cfg.meta, "q_fit_prior_weight", 1e-4)),
                    wrist_prior_weight=float(getattr(self.data.cfg.meta, "wrist_fit_prior_weight", 1e-6)),
                )
                predicted_q = fit["q"][0].detach().cpu().numpy().astype(np.float32)
                relative = np.eye(4, dtype=np.float32)
                relative[:3, :3] = rotvec_to_matrix(fit["wrist_delta_rotvec"])[0].detach().cpu().numpy()
                relative[:3, 3] = fit["wrist_delta_translation"][0].detach().cpu().numpy()
                predicted_wrist = predicted_wrist @ relative
                with torch.no_grad():
                    predicted_points = self.hand_model.points(fit["q"])[0].detach().cpu().numpy().astype(np.float32)
                    predicted_normals = self.hand_model.normals(fit["q"])[0].detach().cpu().numpy().astype(np.float32)

                gt_hand_world = np.asarray(self.data.hand_points_world[target_index], dtype=np.float32)
                pred_hand_world = _local_to_world(predicted_points, predicted_wrist)
                gt_wrist = np.asarray(self.data.wrist_pose_world[target_index], dtype=np.float32)
                object_target_world = np.asarray(self.data.obj_points_world[target_index], dtype=np.float32)
                pred_mesh, mesh_faces = self._mesh_world(source_index, predicted_q, predicted_wrist)
                gt_mesh, _ = self._mesh_world(source_index, np.asarray(self.data.task_arrays["q_next"][row], dtype=np.float32), gt_wrist)
                faces = mesh_faces
                pred_world_all.append(pred_hand_world)
                gt_world_all.append(gt_hand_world)
                obj_world_all.append(object_target_world)
                pred_mesh_all.append(pred_mesh)
                gt_mesh_all.append(gt_mesh)
                pred_q_all.append(predicted_q.copy())
                gt_q_all.append(np.asarray(self.data.task_arrays["q_next"][row], dtype=np.float32))
                pred_wrist_all.append(predicted_wrist.copy())
                gt_wrist_all.append(gt_wrist.copy())
                point_epe.append(float(np.linalg.norm(pred_hand_world - gt_hand_world, axis=-1).mean() * 1000.0))
                q_mae.append(float(np.abs(pred_q_all[-1] - gt_q_all[-1]).mean() * 180.0 / np.pi))
                wrist_epe.append(float(np.linalg.norm(predicted_wrist[:3, 3] - gt_wrist[:3, 3]) * 1000.0))
                wrist_rot.append(_rotation_error_deg(predicted_wrist, gt_wrist))
                pred_contact.append(_contact_ratio(pred_hand_world, object_target_world, contact_threshold_m))
                gt_contact.append(_contact_ratio(gt_hand_world, object_target_world, contact_threshold_m))
                source_ids.append(int(self.data.source_frame_ids[source_index]))
                target_ids.append(int(self.data.source_frame_ids[target_index]))
                cm_flow = gt_batch["hand_flow"][0].detach().cpu().numpy()
                cm_flow_norm.append(float(np.linalg.norm(cm_flow, axis=-1).mean() * 1000.0))

        return {
            "pred_hand_world": np.stack(pred_world_all),
            "gt_hand_world": np.stack(gt_world_all),
            "object_world": np.stack(obj_world_all),
            "pred_mesh_world": np.stack(pred_mesh_all),
            "gt_mesh_world": np.stack(gt_mesh_all),
            "faces": np.asarray(faces, dtype=np.int64),
            "pred_q": np.stack(pred_q_all),
            "gt_q": np.stack(gt_q_all),
            "pred_wrist_world": np.stack(pred_wrist_all),
            "gt_wrist_world": np.stack(gt_wrist_all),
            "point_epe_mm": np.asarray(point_epe, dtype=np.float32),
            "q_mae_deg": np.asarray(q_mae, dtype=np.float32),
            "wrist_epe_mm": np.asarray(wrist_epe, dtype=np.float32),
            "wrist_rotation_error_deg": np.asarray(wrist_rot, dtype=np.float32),
            "pred_contact_ratio": np.asarray(pred_contact, dtype=np.float32),
            "gt_contact_ratio": np.asarray(gt_contact, dtype=np.float32),
            "source_frame_id": np.asarray(source_ids, dtype=np.int64),
            "target_frame_id": np.asarray(target_ids, dtype=np.int64),
            "cm_gt_flow_norm_mm": np.asarray(cm_flow_norm, dtype=np.float32),
            "horizon_stride": np.int32(stride),
        }


def _save_figure(result: dict[str, np.ndarray], output: Path, title: str) -> None:
    pred = result["pred_hand_world"]
    gt = result["gt_hand_world"]
    obj = result["object_world"]
    pred_wrist = result["pred_wrist_world"]
    gt_wrist = result["gt_wrist_world"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    steps = np.arange(len(pred))
    axes[0, 0].plot(steps, result["point_epe_mm"], label="point EPE")
    axes[0, 0].set_title("Closed-loop point error")
    axes[0, 0].set_xlabel("rollout step"); axes[0, 0].set_ylabel("mm")
    axes[0, 0].legend()
    axes[0, 1].plot(steps, result["wrist_epe_mm"], label="wrist translation")
    axes[0, 1].plot(steps, result["wrist_rotation_error_deg"], label="wrist rotation")
    axes[0, 1].set_title("Wrist drift"); axes[0, 1].set_xlabel("rollout step"); axes[0, 1].legend()
    axes[0, 2].plot(steps, result["pred_contact_ratio"], label="pred")
    axes[0, 2].plot(steps, result["gt_contact_ratio"], label="GT")
    axes[0, 2].set_title("Contact ratio (<20 mm)"); axes[0, 2].set_xlabel("rollout step"); axes[0, 2].set_ylim(0, 1); axes[0, 2].legend()
    axes[1, 0].plot(steps, np.linalg.norm(pred_wrist[:, :3, 3] - pred_wrist[0, :3, 3], axis=-1) * 1000.0, label="pred")
    axes[1, 0].plot(steps, np.linalg.norm(gt_wrist[:, :3, 3] - gt_wrist[0, :3, 3], axis=-1) * 1000.0, label="GT")
    axes[1, 0].set_title("Wrist displacement from start"); axes[1, 0].set_xlabel("rollout step"); axes[1, 0].set_ylabel("mm"); axes[1, 0].legend()
    fig.delaxes(axes[1, 1])
    axes[1, 1] = fig.add_subplot(2, 3, 5, projection="3d")
    axes[1, 1].scatter(*obj[0].T, s=2, c="0.7", label="object")
    axes[1, 1].scatter(*gt[-1][::8].T, s=4, c="tab:green", label="GT final")
    axes[1, 1].scatter(*pred[-1][::8].T, s=4, c="tab:red", label="pred final")
    axes[1, 1].set_title("Final hand in world frame"); axes[1, 1].legend()
    axes[1, 2].plot(steps, result["q_mae_deg"])
    axes[1, 2].set_title("q MAE (diagnostic)"); axes[1, 2].set_xlabel("rollout step"); axes[1, 2].set_ylabel("deg")
    fig.suptitle(title)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _serve(result: dict[str, np.ndarray], port: int, title: str) -> None:
    import viser

    server = viser.ViserServer(host="0.0.0.0", port=int(port))
    frame = server.gui.add_slider("Rollout step", min=0, max=len(result["pred_hand_world"]) - 1, step=1, initial_value=0)
    play = server.gui.add_button("Play")
    stop = server.gui.add_button("Stop")
    status = server.gui.add_markdown("")
    state = {"playing": False}
    handles = []
    lock = threading.Lock()

    def render() -> None:
        nonlocal handles
        step = int(frame.value)
        with server.atomic():
            for handle in handles:
                handle.remove()
            handles = [
                server.scene.add_point_cloud("/world/object", result["object_world"][step], colors=(175, 175, 175), point_size=0.004),
                server.scene.add_point_cloud("/world/gt", result["gt_hand_world"][step], colors=(55, 210, 120), point_size=0.005),
                server.scene.add_point_cloud("/world/pred", result["pred_hand_world"][step], colors=(255, 100, 50), point_size=0.005),
                server.scene.add_mesh_simple("/world/gt_mesh", result["gt_mesh_world"][step], result["faces"], color=(55, 210, 120), opacity=0.22),
                server.scene.add_mesh_simple("/world/pred_mesh", result["pred_mesh_world"][step], result["faces"], color=(255, 100, 50), opacity=0.42),
            ]
        status.content = (
            f"**{title}**  \nstep: **{step}/{len(result['point_epe_mm']) - 1}**  "
            f"frames: **{int(result['source_frame_id'][step])} → {int(result['target_frame_id'][step])}**  \n"
            f"point EPE: **{result['point_epe_mm'][step]:.2f} mm**, "
            f"wrist: **{result['wrist_epe_mm'][step]:.2f} mm / {result['wrist_rotation_error_deg'][step]:.2f}°**, "
            f"contact: **{result['pred_contact_ratio'][step]:.1%}** (GT {result['gt_contact_ratio'][step]:.1%})"
        )

    def render_locked() -> None:
        with lock:
            render()

    frame.on_update(lambda _: render_locked())
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    render_locked()
    print(f"Inspire rollout viewer: http://localhost:{port}", flush=True)
    while True:
        if state["playing"]:
            frame.value = (int(frame.value) + 1) % len(result["point_epe_mm"])
        time.sleep(1.0 / 8.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episode", default="inspire_f1/apple/2")
    parser.add_argument("--start-pair", type=int, default=None)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--contact-threshold-mm", type=float, default=20.0)
    parser.add_argument("--output", type=Path, default=Path("output/research/inspire_rollout_cmdecoder.npz"))
    parser.add_argument("--figure", type=Path, default=Path("output/research/inspire_rollout_cmdecoder.png"))
    parser.add_argument("--load-trajectory", type=Path, default=None, help="Serve an existing rollout NPZ without recomputing it")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8096)
    args = parser.parse_args()

    if args.load_trajectory is not None:
        with np.load(args.load_trajectory, allow_pickle=False) as payload:
            result = {key: payload[key] for key in payload.files}
        episode = str(result.get("episode", np.asarray(args.episode)))
        start_pair = int(result.get("start_pair", np.int32(0)))
        title = f"Inspire F1 action-conditioned rollout: {episode}, start_pair={start_pair}"
    else:
        checkpoint = _resolve_path(args.decoder_checkpoint)
        runner = InspireRollout(checkpoint, args.episode, args.device)
        if args.start_pair is None:
            start_pair = _select_rollout_start(runner.data)
        else:
            start_pair = int(args.start_pair)
        result = runner.run(start_pair, int(args.frames), float(args.contact_threshold_mm) / 1000.0)
        result["episode"] = np.asarray(args.episode)
        result["start_pair"] = np.int32(start_pair)
        result["checkpoint"] = np.asarray(str(checkpoint))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **result)
        title = f"Inspire F1 action-conditioned rollout: {args.episode}, start_pair={start_pair}"
        _save_figure(result, args.figure, title)
        print(f"episode={args.episode} start_pair={start_pair} frames={len(result['point_epe_mm'])}")
        print(f"point_epe_mean/final_mm={result['point_epe_mm'].mean():.4f}/{result['point_epe_mm'][-1]:.4f}")
        print(f"wrist_epe_mean/final_mm={result['wrist_epe_mm'].mean():.4f}/{result['wrist_epe_mm'][-1]:.4f}")
        print(f"contact_pred_initial/final={result['pred_contact_ratio'][0]:.4f}/{result['pred_contact_ratio'][-1]:.4f}")
        print(f"saved trajectory: {args.output}")
        print(f"saved figure: {args.figure}")
    if args.serve:
        _serve(result, args.port, title)


if __name__ == "__main__":
    main()
