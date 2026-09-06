"""Teacher-forced CmDecoderv2 validation viewer backed by Viser.

The viewer uses the RL-Inspire validation split.  For every source frame the
model receives the ground-truth current Inspire state and the ground-truth
source Cm window, then predicts only the next state (h=1).  The predicted
state is *not* fed into the next frame; this is deliberately not a rollout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import default_collate

from src.base import load_config

from .dataset import CmDecoderV2Dataset
from .kinematics import InspireKinematics, extract_finger_q, rotvec_to_matrix
from .model import CmDecoderV2
from .visualize import InspireSurface, _resolve


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(pred, dtype=np.float64)[:3, :3].T @ np.asarray(target, dtype=np.float64)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:3, :2].T.reshape(-1), dtype=np.float32)


def _world_points_for_state(
    surface: InspireSurface,
    kinematics: InspireKinematics,
    finger_q: np.ndarray,
    wrist_pose: np.ndarray,
) -> np.ndarray:
    transforms = kinematics.link_transforms_from_state(finger_q, wrist_pose)
    return surface.world_points(transforms).astype(np.float32)


def _contact_ratio(hand_points: np.ndarray, object_points: np.ndarray, threshold_m: float = 0.02) -> float:
    distances, _ = cKDTree(np.asarray(object_points, dtype=np.float32)).query(
        np.asarray(hand_points, dtype=np.float32), k=1
    )
    return float(np.mean(distances < float(threshold_m)))


def _nearest_epe_mm(predicted: np.ndarray, target: np.ndarray) -> float:
    distances, _ = cKDTree(np.asarray(target, dtype=np.float32)).query(
        np.asarray(predicted, dtype=np.float32), k=1
    )
    return float(np.mean(distances) * 1000.0)


def _select_sequence(entries: list[dict[str, Any]], sequence: str, sequence_index: int) -> dict[str, Any]:
    if any(item.get("variant") != "inspire_rl" for item in entries):
        raise ValueError("Teacher-forced viewer requires RL-Inspire entries only")
    if sequence:
        matches = [item for item in entries if str(item["id"]) == sequence]
        if not matches:
            raise ValueError(f"Unknown RL-Inspire validation sequence {sequence!r}")
        return matches[0]
    if sequence_index < 0 or sequence_index >= len(entries):
        raise IndexError(f"sequence-index must lie in [0, {len(entries) - 1}]")
    return entries[sequence_index]


def _tensor_batch(samples: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    collated = default_collate(samples)
    return {
        key: value.to(device)
        for key, value in collated.items()
        if torch.is_tensor(value)
    }


def evaluate_sequence(
    *,
    cfg: Any,
    checkpoint_path: Path,
    sequence_entry: dict[str, Any],
    device: torch.device,
    batch_size: int,
    max_frames: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    view_root = _resolve(str(cfg.data.view_root))
    urdf_path = _resolve(str(cfg.data.urdf_path))
    index = json.loads((view_root / "index.json").read_text(encoding="utf-8"))
    dataset = CmDecoderV2Dataset(
        [sequence_entry],
        urdf_path=urdf_path,
        window_size=int(cfg.meta.window_size),
        num_obj_points=int(cfg.meta.num_obj_points),
        num_hand_points=int(cfg.meta.num_hand_points),
        seed=42,
        perturb=False,
        active_only=False,
        translation_noise_std_m=float(cfg.data.translation_noise_std_m),
        translation_noise_clip_m=float(cfg.data.translation_noise_clip_m),
        rotation_noise_std_deg=float(cfg.data.rotation_noise_std_deg),
        rotation_noise_clip_deg=float(cfg.data.rotation_noise_clip_deg),
        finger_q_noise_std_rad=float(cfg.data.finger_q_noise_std_rad),
        finger_q_noise_clip_rad=float(cfg.data.finger_q_noise_clip_rad),
    )
    sequence = dataset.sequences[0]
    frame_count = min(len(dataset), int(max_frames))
    if frame_count <= 0:
        raise ValueError(f"Sequence {sequence_entry['id']} has no complete teacher-forcing windows")

    cfg.model.meta = cfg.meta
    model = CmDecoderV2(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    kinematics = InspireKinematics(urdf_path)
    surface = InspireSurface(urdf_path)

    predicted_q: list[np.ndarray] = []
    current_q: list[np.ndarray] = []
    target_q: list[np.ndarray] = []
    current_wrist: list[np.ndarray] = []
    predicted_wrist: list[np.ndarray] = []
    target_wrist: list[np.ndarray] = []
    current_hand: list[np.ndarray] = []
    target_hand: list[np.ndarray] = []
    predicted_hand: list[np.ndarray] = []
    object_points: list[np.ndarray] = []
    source_frame_id: list[int] = []
    target_frame_id: list[int] = []
    cm_valid: list[np.ndarray] = []
    q_mae_deg: list[float] = []
    wrist_translation_error_mm: list[float] = []
    wrist_rotation_error_deg: list[float] = []
    hand_epe_mm: list[float] = []
    predicted_contact: list[float] = []
    target_contact: list[float] = []

    with torch.inference_mode():
        for batch_start in range(0, frame_count, max(1, int(batch_size))):
            indices = list(range(batch_start, min(frame_count, batch_start + max(1, int(batch_size)))))
            samples = [dataset[index_value] for index_value in indices]
            batch = _tensor_batch(samples, device)
            output = model(batch)
            pred_delta_q = output["pred_q_delta"][:, 0].detach().cpu().numpy()
            pred_translation = output["pred_wrist_translation"][:, 0].detach().cpu().numpy()
            pred_rotvec = output["pred_wrist_rotvec"][:, 0].detach().cpu().numpy()
            valid = output["cm_sample_valid"].detach().cpu().numpy().astype(bool)

            for local, frame in enumerate(indices):
                current_finger = np.asarray(samples[local]["current_finger_q"], dtype=np.float64)
                current_pose = np.asarray(sequence.wrist_pose[frame], dtype=np.float64)
                target_finger = extract_finger_q(np.asarray(sequence.q_native[frame + 1], dtype=np.float64))
                target_pose = np.asarray(sequence.wrist_pose[frame + 1], dtype=np.float64)
                predicted_finger = kinematics.clamp_finger_q(current_finger + pred_delta_q[local])
                delta = np.eye(4, dtype=np.float64)
                delta[:3, :3] = rotvec_to_matrix(pred_rotvec[local])
                delta[:3, 3] = pred_translation[local]
                predicted_pose = current_pose @ delta

                current_points = np.asarray(sequence.hand_points[frame], dtype=np.float32)
                target_points = np.asarray(sequence.hand_points[frame + 1], dtype=np.float32)
                obj_points = np.asarray(sequence.object_points[frame + 1], dtype=np.float32)
                predicted_points = _world_points_for_state(surface, kinematics, predicted_finger, predicted_pose)

                current_q.append(current_finger.astype(np.float32))
                predicted_q.append(predicted_finger.astype(np.float32))
                target_q.append(target_finger.astype(np.float32))
                current_wrist.append(current_pose.astype(np.float32))
                predicted_wrist.append(predicted_pose.astype(np.float32))
                target_wrist.append(target_pose.astype(np.float32))
                current_hand.append(current_points)
                target_hand.append(target_points)
                predicted_hand.append(predicted_points)
                object_points.append(obj_points)
                source_frame_id.append(int(sequence.source_frame[frame]))
                target_frame_id.append(int(sequence.source_frame[frame + 1]))
                cm_valid.append(valid[local])
                q_mae_deg.append(float(np.abs(predicted_finger - target_finger).mean() * 180.0 / np.pi))
                wrist_translation_error_mm.append(float(np.linalg.norm(predicted_pose[:3, 3] - target_pose[:3, 3]) * 1000.0))
                wrist_rotation_error_deg.append(_rotation_error_deg(predicted_pose, target_pose))
                hand_epe_mm.append(_nearest_epe_mm(predicted_points, target_points))
                predicted_contact.append(_contact_ratio(predicted_points, obj_points))
                target_contact.append(_contact_ratio(target_points, obj_points))

    result = {
        "current_q": np.asarray(current_q, dtype=np.float32),
        "predicted_q": np.asarray(predicted_q, dtype=np.float32),
        "target_q": np.asarray(target_q, dtype=np.float32),
        "current_wrist_world": np.asarray(current_wrist, dtype=np.float32),
        "predicted_wrist_world": np.asarray(predicted_wrist, dtype=np.float32),
        "target_wrist_world": np.asarray(target_wrist, dtype=np.float32),
        "current_hand_world": np.asarray(current_hand, dtype=np.float32),
        "target_hand_world": np.asarray(target_hand, dtype=np.float32),
        "predicted_hand_world": np.asarray(predicted_hand, dtype=np.float32),
        "object_world": np.asarray(object_points, dtype=np.float32),
        "source_frame_id": np.asarray(source_frame_id, dtype=np.int64),
        "target_frame_id": np.asarray(target_frame_id, dtype=np.int64),
        "cm_sample_valid": np.asarray(cm_valid, dtype=bool),
        "q_mae_deg": np.asarray(q_mae_deg, dtype=np.float32),
        "wrist_translation_error_mm": np.asarray(wrist_translation_error_mm, dtype=np.float32),
        "wrist_rotation_error_deg": np.asarray(wrist_rotation_error_deg, dtype=np.float32),
        "hand_epe_mm": np.asarray(hand_epe_mm, dtype=np.float32),
        "predicted_contact_ratio": np.asarray(predicted_contact, dtype=np.float32),
        "target_contact_ratio": np.asarray(target_contact, dtype=np.float32),
    }
    metadata = {
        "schema_name": "ref2dex_cmdecoderv2_teacherforce_v1",
        "split": "val",
        "sequence_id": str(sequence_entry["id"]),
        "variant": str(sequence_entry["variant"]),
        "frame_count": int(frame_count),
        "window_size": int(cfg.meta.window_size),
        "effective_fps": float(cfg.meta.effective_fps),
        "teacher_forcing": True,
        "feedback": "none; each frame uses GT current Inspire state and GT Cm window",
        "view_index": str(view_root / "index.json"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "urdf": str(urdf_path),
    }
    runtime = {
        "model": model,
        "dataset": dataset,
        "sequence": sequence,
        "kinematics": kinematics,
        "surface": surface,
        "device": device,
    }
    return result, metadata, runtime


def _rollout_step(
    runtime: dict[str, Any],
    result: dict[str, np.ndarray],
    index: int,
    current_q: np.ndarray,
    current_wrist: np.ndarray,
) -> dict[str, Any]:
    """Predict one step from an explicitly supplied state.

    The source Cm window remains teacher-forced from the recorded validation
    sequence; only the target-hand state is fed back between rollout steps.
    """
    dataset = runtime["dataset"]
    sequence = runtime["sequence"]
    model = runtime["model"]
    kinematics = runtime["kinematics"]
    surface = runtime["surface"]
    device = runtime["device"]
    sample = dataset[int(index)]
    batch = _tensor_batch([sample], device)
    object_pose = np.asarray(sequence.object_pose[index], dtype=np.float64)
    current_object = np.linalg.inv(object_pose) @ np.asarray(current_wrist, dtype=np.float64)
    batch.update({
        "current_finger_q": torch.from_numpy(np.asarray(current_q, dtype=np.float32)).unsqueeze(0).to(device),
        "current_wrist_translation_object": torch.from_numpy(current_object[:3, 3].astype(np.float32)).unsqueeze(0).to(device),
        "current_wrist_rotation_6d_object": torch.from_numpy(_rotation_6d(current_object[:3, :3])).unsqueeze(0).to(device),
        "current_link_features": torch.from_numpy(
            kinematics.query_features(current_q, current_wrist, object_pose)
        ).unsqueeze(0).to(device),
    })
    with torch.inference_mode():
        output = model(batch)
    delta_q = output["pred_q_delta"][0, 0].detach().cpu().numpy()
    translation = output["pred_wrist_translation"][0, 0].detach().cpu().numpy()
    rotvec = output["pred_wrist_rotvec"][0, 0].detach().cpu().numpy()
    next_q = kinematics.clamp_finger_q(np.asarray(current_q, dtype=np.float64) + delta_q)
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotvec_to_matrix(rotvec)
    delta[:3, 3] = translation
    next_wrist = np.asarray(current_wrist, dtype=np.float64) @ delta
    current_hand = _world_points_for_state(surface, kinematics, current_q, current_wrist)
    predicted_hand = _world_points_for_state(surface, kinematics, next_q, next_wrist)
    target_q = extract_finger_q(np.asarray(sequence.q_native[index + 1], dtype=np.float64))
    target_wrist = np.asarray(sequence.wrist_pose[index + 1], dtype=np.float64)
    target_hand = np.asarray(sequence.hand_points[index + 1], dtype=np.float32)
    object_points = np.asarray(sequence.object_points[index + 1], dtype=np.float32)
    return {
        "current_q": np.asarray(current_q, dtype=np.float32),
        "next_q": next_q.astype(np.float32),
        "current_wrist": np.asarray(current_wrist, dtype=np.float32),
        "next_wrist": next_wrist.astype(np.float32),
        "current_hand": current_hand,
        "predicted_hand": predicted_hand,
        "target_q": target_q.astype(np.float32),
        "target_wrist": target_wrist.astype(np.float32),
        "target_hand": target_hand,
        "object_points": object_points,
        "cm_valid": output["cm_sample_valid"][0].detach().cpu().numpy().astype(bool),
        "q_mae_deg": float(np.abs(next_q - target_q).mean() * 180.0 / np.pi),
        "wrist_translation_error_mm": float(np.linalg.norm(next_wrist[:3, 3] - target_wrist[:3, 3]) * 1000.0),
        "wrist_rotation_error_deg": _rotation_error_deg(next_wrist, target_wrist),
        "hand_epe_mm": _nearest_epe_mm(predicted_hand, target_hand),
        "predicted_contact_ratio": _contact_ratio(predicted_hand, object_points),
        "target_contact_ratio": _contact_ratio(target_hand, object_points),
    }


def serve(
    result: dict[str, np.ndarray],
    metadata: dict[str, Any],
    runtime: dict[str, Any],
    *,
    host: str,
    port: int,
    fps: float,
) -> None:
    import viser

    server = viser.ViserServer(host=host, port=int(port))
    mode = server.gui.add_dropdown(
        "Execution mode",
        options=("teacherforced", "rollout"),
        initial_value="teacherforced",
    )
    display = server.gui.add_dropdown(
        "Hand display",
        options=("GT", "pred", "both"),
        initial_value="both",
    )
    frame = server.gui.add_slider(
        "Validation frame",
        min=0,
        max=len(result["predicted_hand_world"]) - 1,
        step=1,
        initial_value=0,
    )
    point_size = server.gui.add_slider(
        "Point size (m)",
        min=0.001,
        max=0.020,
        step=0.001,
        initial_value=0.005,
    )
    play = server.gui.add_button("Play")
    stop = server.gui.add_button("Stop")
    status = server.gui.add_markdown("")
    server.gui.add_markdown(
        "### Point-cloud legend\n"
        "🔵 **blue** — GT current hand\n\n"
        "🟢 **green** — GT next hand\n\n"
        "🟠 **orange** — predicted current hand\n\n"
        "🔴 **red** — predicted next hand\n\n"
        "⚪ **gray** — object surface samples\n\n"
        "`teacherforced`: current state is GT at every frame.  "
        "`rollout`: switching at frame *t* seeds from GT(t), then feeds predictions forward."
    )
    state: dict[str, Any] = {
        "playing": False,
        "mode": "teacherforced",
        "rollout_last_frame": None,
        "rollout_current_q": None,
        "rollout_current_wrist": None,
        "rollout_cache": {},
    }
    handles: list[Any] = []
    lock = threading.Lock()

    def reset_rollout(index: int) -> None:
        state["rollout_cache"] = {}
        state["rollout_last_frame"] = int(index) - 1
        state["rollout_current_q"] = result["current_q"][index].astype(np.float64).copy()
        state["rollout_current_wrist"] = result["current_wrist_world"][index].astype(np.float64).copy()

    def rollout_frame(index: int) -> dict[str, Any]:
        cached = state["rollout_cache"].get(int(index))
        if cached is not None:
            return cached
        last_frame = state["rollout_last_frame"]
        if last_frame is None or int(index) != int(last_frame) + 1:
            reset_rollout(int(index))
        record = _rollout_step(
            runtime,
            result,
            int(index),
            state["rollout_current_q"],
            state["rollout_current_wrist"],
        )
        state["rollout_cache"][int(index)] = record
        state["rollout_current_q"] = record["next_q"].astype(np.float64).copy()
        state["rollout_current_wrist"] = record["next_wrist"].astype(np.float64).copy()
        state["rollout_last_frame"] = int(index)
        return record

    def frame_record(index: int) -> dict[str, Any]:
        if state["mode"] == "rollout":
            record = rollout_frame(index)
            return {
                "gt_current": result["current_hand_world"][index],
                "gt_next": result["target_hand_world"][index],
                "pred_current": record["current_hand"],
                "pred_next": record["predicted_hand"],
                "object": record["object_points"],
                "valid": record["cm_valid"],
                "q_mae_deg": record["q_mae_deg"],
                "wrist_translation_error_mm": record["wrist_translation_error_mm"],
                "wrist_rotation_error_deg": record["wrist_rotation_error_deg"],
                "hand_epe_mm": record["hand_epe_mm"],
                "predicted_contact_ratio": record["predicted_contact_ratio"],
                "target_contact_ratio": record["target_contact_ratio"],
            }
        return {
            "gt_current": result["current_hand_world"][index],
            "gt_next": result["target_hand_world"][index],
            "pred_current": result["current_hand_world"][index],
            "pred_next": result["predicted_hand_world"][index],
            "object": result["object_world"][index],
            "valid": result["cm_sample_valid"][index],
            "q_mae_deg": result["q_mae_deg"][index],
            "wrist_translation_error_mm": result["wrist_translation_error_mm"][index],
            "wrist_rotation_error_deg": result["wrist_rotation_error_deg"][index],
            "hand_epe_mm": result["hand_epe_mm"][index],
            "predicted_contact_ratio": result["predicted_contact_ratio"][index],
            "target_contact_ratio": result["target_contact_ratio"][index],
        }

    def render() -> None:
        nonlocal handles
        index = int(frame.value)
        record = frame_record(index)
        selected_display = str(display.value)
        size = float(point_size.value)
        with server.atomic():
            for handle in handles:
                handle.remove()
            clouds: list[Any] = [
                server.scene.add_point_cloud(
                    "/world/object", record["object"][::4], colors=(170, 170, 170), point_size=max(size * 0.75, 0.001)
                )
            ]
            if selected_display in {"GT", "both"}:
                clouds.extend([
                    server.scene.add_point_cloud(
                        "/world/gt_current", record["gt_current"], colors=(60, 130, 240), point_size=size
                    ),
                    server.scene.add_point_cloud(
                        "/world/gt_next", record["gt_next"], colors=(50, 210, 110), point_size=size
                    ),
                ])
            if selected_display in {"pred", "both"}:
                clouds.extend([
                    server.scene.add_point_cloud(
                        "/world/pred_current", record["pred_current"], colors=(255, 165, 0), point_size=size
                    ),
                    server.scene.add_point_cloud(
                        "/world/pred_next", record["pred_next"], colors=(255, 75, 45), point_size=size
                    ),
                ])
            handles = clouds
        valid = record["valid"]
        status.content = (
            f"**{metadata['sequence_id']}**  \n"
            f"mode: **{state['mode']}**, display: **{selected_display}**, point size: **{size:.3f} m**  \n"
            f"frame: **{index}/{len(result['predicted_hand_world']) - 1}**, "
            f"source: **{int(result['source_frame_id'][index])} → {int(result['target_frame_id'][index])}**  \n"
            f"Cm valid: **{int(valid.sum())}/{len(valid)}** (current={bool(valid[0])})  \n"
            f"q MAE: **{float(record['q_mae_deg']):.3f}°**, "
            f"wrist: **{float(record['wrist_translation_error_mm']):.2f} mm / "
            f"{float(record['wrist_rotation_error_deg']):.2f}°**, "
            f"hand NN-EPE: **{float(record['hand_epe_mm']):.2f} mm**  \n"
            f"contact: **{float(record['predicted_contact_ratio']):.1%}** "
            f"(GT {float(record['target_contact_ratio']):.1%})"
        )

    def render_locked() -> None:
        with lock:
            render()

    def on_mode_change(_: Any) -> None:
        with lock:
            state["mode"] = str(mode.value)
            if state["mode"] == "rollout":
                reset_rollout(int(frame.value))
            render()

    mode.on_update(on_mode_change)
    frame.on_update(lambda _: render_locked())
    display.on_update(lambda _: render_locked())
    point_size.on_update(lambda _: render_locked())
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    render_locked()
    print(f"Viser teacher-forcing viewer: http://localhost:{port}", flush=True)
    while True:
        if state["playing"]:
            frame.value = (int(frame.value) + 1) % len(result["predicted_hand_world"])
        time.sleep(1.0 / max(float(fps), 1e-3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml")
    parser.add_argument(
        "--checkpoint",
        default="outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt",
    )
    parser.add_argument("--split", choices=("val",), default="val")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--output-root", default="outputs/cmdecoderv2")
    parser.add_argument("--activity-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    index_path = _resolve(str(cfg.data.view_root)) / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = list(index["sequences"][args.split])
    sequence_entry = _select_sequence(entries, args.sequence, int(args.sequence_index))
    checkpoint_path = _resolve(args.checkpoint)
    device = torch.device(args.device)
    result, metadata, runtime = evaluate_sequence(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        sequence_entry=sequence_entry,
        device=device,
        batch_size=int(args.batch_size),
        max_frames=int(args.max_frames),
    )
    run_id = f"cmdecoderv2-teacherforce-val-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output = _resolve(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    trajectory = output / "teacherforce.npz"
    np.savez_compressed(trajectory, **result)
    metadata.update({
        "task": "CmDecoderv2",
        "run_id": run_id,
        "run_status": "RUNNING",
        "activity_id": args.activity_id,
        "modification_version": str(cfg.modification_version),
        "operation_category": ["experiment", "operation"],
        "created_at": _now(),
        "base_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "outputs": {"trajectory": str(trajectory)},
        "metrics": {
            "q_mae_deg_mean": float(result["q_mae_deg"].mean()),
            "wrist_translation_error_mm_mean": float(result["wrist_translation_error_mm"].mean()),
            "wrist_rotation_error_deg_mean": float(result["wrist_rotation_error_deg"].mean()),
            "hand_epe_mm_mean": float(result["hand_epe_mm"].mean()),
            "cm_current_valid_ratio": float(result["cm_sample_valid"][:, 0].mean()),
        },
    })
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "run_status": "RUNNING",
        "output": str(output),
        "sequence": sequence_entry["id"],
        "frames": int(len(result["predicted_hand_world"])),
        "metrics": metadata["metrics"],
        "viser": f"http://localhost:{args.port}",
    }, ensure_ascii=False), flush=True)
    serve(result, metadata, runtime, host=args.host, port=int(args.port), fps=float(args.fps))


if __name__ == "__main__":
    main()
