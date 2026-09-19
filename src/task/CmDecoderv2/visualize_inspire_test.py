"""Viser diagnostic for pure Dexplore Inspire test trajectories.

This viewer is intentionally separate from the formal MANO-only test view.
It uses the 125 held-out parent ids from the decoder test index, but replaces
the MANO source stream with the corresponding raw Dexplore RL-Inspire stream:
hand geometry, object pose, native q and Inspire wrist all come from the same
RL trajectory.  Teacher forcing uses the GT Inspire state at every source
frame.  Rollout uses the GT Inspire state only at the selected handoff frame,
then feeds the predicted state back recursively.

The branch is qualitative/diagnostic only.  It does not modify the formal
decoder split or use these test trajectories for training or checkpoint
selection.
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
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import (
    NATIVE_Q_START,
    NUM_DOFS,
    InspireUrdfModel,
    _fk_surface,
    _load_tensor,
    _pose_from_native,
    _safe_name,
)

from .dataset import _normals_world_to_frame, _points_world_to_frame
from .kinematics import InspireKinematics, extract_finger_q, relative_pose, rotvec_to_matrix
from .model import CmDecoderV2
from .visualize import InspireSurface
from .visualize_mano import RolloutEngine


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:3, :2].T.reshape(-1), dtype=np.float32)


def _rotation_error_deg(pred: np.ndarray, target: np.ndarray) -> float:
    relative = np.asarray(pred, dtype=np.float64)[:3, :3].T @ np.asarray(target, dtype=np.float64)[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _nearest_epe_mm(predicted: np.ndarray, target: np.ndarray) -> float:
    distances, _ = cKDTree(np.asarray(target, dtype=np.float32)).query(np.asarray(predicted, dtype=np.float32), k=1)
    return float(np.mean(distances) * 1000.0)


class InspireTestSequence:
    """Lazy-in-memory conversion of one held-out parent to RL-Inspire GT."""

    def __init__(
        self,
        entry: dict[str, Any],
        *,
        urdf_path: Path,
        rl_root: Path,
        parent_root: Path,
        num_hand_points: int = 1538,
        surface_seed: int = 2024,
    ) -> None:
        self.entry = dict(entry)
        self.id = str(entry["id"])
        self.frame_count = int(entry["frame_count"])
        safe = _safe_name(self.id)
        geometry_root = _resolve(str(entry["geometry_root"]))
        geometry_manifest = json.loads((geometry_root / "manifest.json").read_text(encoding="utf-8"))
        self.parent_cache = _resolve(str(geometry_manifest["input_parent_cache"]))
        self.rl_q_path = _resolve(str(rl_root / safe / "interaction_hand_inspire.pt"))
        if not self.rl_q_path.is_file():
            raise FileNotFoundError(f"Missing raw Dexplore RL-Inspire tensor: {self.rl_q_path}")
        self.parent_cache = _resolve(str(parent_root / self.id)) if not self.parent_cache.is_dir() else self.parent_cache
        q_payload = _load_tensor(self.rl_q_path)
        self.q_native = q_payload[:, NATIVE_Q_START:NATIVE_Q_START + NUM_DOFS].astype(np.float32)
        self.object_pose = np.stack([_pose_from_native(row[198:201], row[201:205]) for row in q_payload]).astype(np.float32)
        shared = self.parent_cache / "shared"
        parent_object = np.load(shared / "obj_points_world.npy", mmap_mode="r")
        parent_normals = np.load(shared / "obj_normals_world.npy", mmap_mode="r")
        parent_pose = np.load(shared / "obj_pose_world.npy", mmap_mode="r")
        self.source_frame = np.load(shared / "raw_frame_id.npy", mmap_mode="r")
        if len(q_payload) != self.frame_count or len(parent_object) != self.frame_count:
            raise ValueError(f"Frame mismatch for {self.id}: index={self.frame_count}, q={len(q_payload)}, object={len(parent_object)}")
        self.object_points = np.empty_like(np.asarray(parent_object), dtype=np.float32)
        self.object_normals = np.empty_like(np.asarray(parent_normals), dtype=np.float32)
        for frame in range(self.frame_count):
            local_points = _points_world_to_frame(parent_object[frame], parent_pose[frame])
            local_normals = _normals_world_to_frame(parent_normals[frame], parent_pose[frame])
            new_pose = self.object_pose[frame]
            self.object_points[frame] = local_points @ new_pose[:3, :3].T + new_pose[:3, 3]
            self.object_normals[frame] = local_normals @ new_pose[:3, :3].T
        self.object_normals /= np.clip(np.linalg.norm(self.object_normals, axis=-1, keepdims=True), 1e-8, None)
        urdf_model = InspireUrdfModel(urdf_path)
        sampled_points, sampled_normals, sampled_visual_ids = urdf_model.surface_samples(int(num_hand_points), int(surface_seed))
        self.hand_points, self.hand_normals = _fk_surface(
            urdf_model, self.q_native, sampled_points, sampled_normals, sampled_visual_ids
        )
        self.kinematics = InspireKinematics(urdf_path)
        self.wrist_pose = np.stack([
            self.kinematics.wrist_pose_from_native(np.asarray(row, dtype=np.float64))
            for row in self.q_native
        ]).astype(np.float32)
        self.finger_q = np.stack([
            extract_finger_q(np.asarray(row, dtype=np.float64)) for row in self.q_native
        ]).astype(np.float32)
        if self.hand_points.shape != (self.frame_count, int(num_hand_points), 3):
            raise ValueError(f"Unexpected Inspire hand shape: {self.hand_points.shape}")
        if not np.isfinite(self.object_points).all() or not np.isfinite(self.hand_points).all():
            raise ValueError(f"Non-finite Inspire test geometry for {self.id}")

    def source_window(self, start: int, window_size: int, num_obj_points: int) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        start = int(start)
        window_size = int(window_size)
        if start < 0 or start + window_size >= self.frame_count:
            raise IndexError(f"Cm window start {start} exceeds sequence {self.frame_count} with K={window_size}")
        object_points: list[np.ndarray] = []
        object_normals: list[np.ndarray] = []
        hand_points: list[np.ndarray] = []
        hand_normals: list[np.ndarray] = []
        hand_flow: list[np.ndarray] = []
        for offset in range(window_size):
            frame = start + offset
            pose = np.asarray(self.object_pose[frame], dtype=np.float32)
            rng = np.random.default_rng(2024 + start * 131 + offset)
            selected = rng.choice(self.object_points.shape[1], size=int(num_obj_points), replace=False)
            current_hand = _points_world_to_frame(self.hand_points[frame], pose)
            future_hand = _points_world_to_frame(self.hand_points[frame + 1], pose)
            object_points.append(_points_world_to_frame(self.object_points[frame, selected], pose))
            object_normals.append(_normals_world_to_frame(self.object_normals[frame, selected], pose))
            hand_points.append(current_hand)
            hand_normals.append(_normals_world_to_frame(self.hand_normals[frame], pose))
            hand_flow.append(future_hand - current_hand)
        batch = {
            "obj_points": torch.from_numpy(np.stack(object_points).astype(np.float32)).unsqueeze(0),
            "obj_normals": torch.from_numpy(np.stack(object_normals).astype(np.float32)).unsqueeze(0),
            "obj_valid_mask": torch.ones((1, window_size, int(num_obj_points)), dtype=torch.bool),
            "hand_points": torch.from_numpy(np.stack(hand_points).astype(np.float32)).unsqueeze(0),
            "hand_normals": torch.from_numpy(np.stack(hand_normals).astype(np.float32)).unsqueeze(0),
            "hand_flow": torch.from_numpy(np.stack(hand_flow).astype(np.float32)).unsqueeze(0),
            "hand_valid_mask": torch.ones((1, window_size, self.hand_points.shape[1]), dtype=torch.bool),
        }
        return batch, np.asarray(self.object_pose[start], dtype=np.float64)


class InspireRolloutEngine(RolloutEngine):
    """Reuse the MANO viewer's lazy rollout, but initialize from Inspire GT."""

    def start(self, frame: int) -> None:
        frame = int(frame)
        if frame < 0 or frame >= self.sequence.frame_count:
            raise IndexError(frame)
        finger_q = self.kinematics.clamp_finger_q(np.asarray(self.sequence.finger_q[frame], dtype=np.float64))
        wrist = np.asarray(self.sequence.wrist_pose[frame], dtype=np.float64).copy()
        self.handoff_frame = frame
        self.states = {
            frame: {
                "finger_q": finger_q,
                "wrist": wrist,
                "points": self._state_points(finger_q, wrist),
                "cm_valid": None,
            }
        }


def _load_sequence(cfg: Any, sequence: str, sequence_index: int, *, rl_root: Path, parent_root: Path) -> InspireTestSequence:
    view_root = _resolve(str(cfg.data.view_root))
    index = json.loads((view_root / "index.json").read_text(encoding="utf-8"))
    entries = list(index["sequences"]["test"])
    if any(item.get("variant") != "mano" for item in entries):
        raise ValueError("Inspire diagnostic expects the held-out parent ids from the MANO-only test index")
    if sequence:
        matches = [item for item in entries if str(item["id"]) == sequence]
        if not matches:
            raise ValueError(f"Unknown held-out test sequence {sequence!r}")
        entry = matches[0]
    else:
        if sequence_index < 0 or sequence_index >= len(entries):
            raise IndexError(f"sequence-index must lie in [0, {len(entries) - 1}]")
        entry = entries[sequence_index]
    return InspireTestSequence(
        entry,
        urdf_path=_resolve(str(cfg.data.urdf_path)),
        rl_root=rl_root,
        parent_root=parent_root,
        num_hand_points=int(cfg.meta.num_hand_points),
    )


def _load_model(cfg: Any, checkpoint_path: Path, device: torch.device) -> tuple[CmDecoderV2, InspireKinematics, InspireSurface]:
    cfg.model.meta = cfg.meta
    model = CmDecoderV2(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    urdf = _resolve(str(cfg.data.urdf_path))
    return model, InspireKinematics(urdf), InspireSurface(urdf)


class DiagnosticRunner:
    def __init__(self, sequence: InspireTestSequence, model: CmDecoderV2, kinematics: InspireKinematics, surface: InspireSurface, device: torch.device, cfg: Any) -> None:
        self.sequence = sequence
        self.model = model
        self.kinematics = kinematics
        self.surface = surface
        self.device = device
        self.window_size = int(cfg.meta.window_size)
        self.num_obj_points = int(cfg.meta.num_obj_points)
        self.teacher_cache: dict[int, dict[str, Any]] = {}
        self.rollout = InspireRolloutEngine(
            sequence=sequence,
            model=model,
            kinematics=kinematics,
            surface=surface,
            device=device,
            window_size=self.window_size,
            num_obj_points=self.num_obj_points,
        )

    def _predict(self, start: int, current_q: np.ndarray, current_wrist: np.ndarray) -> dict[str, Any]:
        source_batch, object_pose = self.sequence.source_window(start, self.window_size, self.num_obj_points)
        current_object = np.linalg.inv(object_pose) @ np.asarray(current_wrist, dtype=np.float64)
        source_batch.update({
            "current_finger_q": torch.from_numpy(np.asarray(current_q, dtype=np.float32)).unsqueeze(0),
            "current_wrist_translation_object": torch.from_numpy(current_object[:3, 3].astype(np.float32)).unsqueeze(0),
            "current_wrist_rotation_6d_object": torch.from_numpy(_rotation_6d(current_object[:3, :3])).unsqueeze(0),
            "current_link_features": torch.from_numpy(
                self.kinematics.query_features(current_q, current_wrist, object_pose)
            ).unsqueeze(0),
        })
        source_batch = {key: value.to(self.device) for key, value in source_batch.items()}
        with torch.inference_mode():
            output = self.model(source_batch)
        delta_q = output["pred_q_delta"][0, 0].detach().cpu().numpy()
        translation = output["pred_wrist_translation"][0, 0].detach().cpu().numpy()
        rotvec = output["pred_wrist_rotvec"][0, 0].detach().cpu().numpy()
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = rotvec_to_matrix(rotvec)
        delta[:3, 3] = translation
        next_q = self.kinematics.clamp_finger_q(np.asarray(current_q, dtype=np.float64) + delta_q)
        next_wrist = np.asarray(current_wrist, dtype=np.float64) @ delta
        next_frame = start + 1
        pred_points = self.surface.world_points(self.kinematics.link_transforms_from_state(next_q, next_wrist))
        target_q = np.asarray(self.sequence.finger_q[next_frame], dtype=np.float64)
        target_wrist = np.asarray(self.sequence.wrist_pose[next_frame], dtype=np.float64)
        target_points = np.asarray(self.sequence.hand_points[next_frame], dtype=np.float32)
        return {
            "current_q": np.asarray(current_q, dtype=np.float32),
            "current_wrist": np.asarray(current_wrist, dtype=np.float32),
            "next_q": next_q.astype(np.float32),
            "next_wrist": next_wrist.astype(np.float32),
            "pred_points": pred_points.astype(np.float32),
            "target_q": target_q.astype(np.float32),
            "target_wrist": target_wrist.astype(np.float32),
            "target_points": target_points,
            "cm_valid": output["cm_sample_valid"][0].detach().cpu().numpy().astype(bool),
            "q_mae_deg": float(np.abs(next_q - target_q).mean() * 180.0 / np.pi),
            "wrist_translation_error_mm": float(np.linalg.norm(next_wrist[:3, 3] - target_wrist[:3, 3]) * 1000.0),
            "wrist_rotation_error_deg": _rotation_error_deg(next_wrist, target_wrist),
            "hand_epe_mm": _nearest_epe_mm(pred_points, target_points),
        }

    def teacher(self, start: int) -> dict[str, Any] | None:
        start = int(start)
        # A K-frame Cm window needs K+1 geometry frames because every source
        # hand-flow is defined as frame tau -> tau+1.  Tail frames remain
        # valid GT playback, but do not have a complete teacher-forced input.
        if start < 0 or start >= self.sequence.frame_count - self.window_size:
            return None
        if start not in self.teacher_cache:
            self.teacher_cache[start] = self._predict(
                start,
                np.asarray(self.sequence.finger_q[start], dtype=np.float64),
                np.asarray(self.sequence.wrist_pose[start], dtype=np.float64),
            )
        return self.teacher_cache[start]


def _write_trajectory(path: Path, sequence: InspireTestSequence, runner: DiagnosticRunner) -> None:
    teacher_q = np.full((sequence.frame_count, 6), np.nan, dtype=np.float32)
    teacher_wrist = np.full((sequence.frame_count, 4, 4), np.nan, dtype=np.float32)
    teacher_points = np.full((sequence.frame_count, runner.surface.points.shape[0], 3), np.nan, dtype=np.float32)
    teacher_valid = np.zeros((sequence.frame_count, runner.window_size), dtype=bool)
    for frame, record in runner.teacher_cache.items():
        teacher_q[frame + 1] = record["next_q"]
        teacher_wrist[frame + 1] = record["next_wrist"]
        teacher_points[frame + 1] = record["pred_points"]
        teacher_valid[frame] = record["cm_valid"]
    rollout = runner.rollout.export()
    np.savez_compressed(
        path,
        inspire_gt_hand_points_world=np.asarray(sequence.hand_points, dtype=np.float32),
        dexplore_object_points_world=np.asarray(sequence.object_points, dtype=np.float32),
        inspire_gt_finger_q=np.asarray(sequence.finger_q, dtype=np.float32),
        inspire_gt_wrist_pose_world=np.asarray(sequence.wrist_pose, dtype=np.float32),
        dexplore_object_pose_world=np.asarray(sequence.object_pose, dtype=np.float32),
        source_frame_id=np.asarray(sequence.source_frame, dtype=np.int64),
        teacher_predicted_finger_q=teacher_q,
        teacher_predicted_wrist_pose_world=teacher_wrist,
        teacher_predicted_points_world=teacher_points,
        teacher_cm_sample_valid=teacher_valid,
        **{f"rollout_{key}": value for key, value in rollout.items()},
    )


def serve(*, runner: DiagnosticRunner, output: Path, manifest: dict[str, Any], host: str, port: int, fps: float) -> None:
    import viser

    sequence = runner.sequence
    server = viser.ViserServer(host=host, port=int(port))
    mode = server.gui.add_dropdown("Execution mode", options=("teacherforced", "rollout"), initial_value="rollout")
    display = server.gui.add_dropdown("Hand display", options=("GT", "pred", "both"), initial_value="both")
    frame = server.gui.add_slider("Frame", min=0, max=sequence.frame_count - 1, step=1, initial_value=0)
    frame_step = server.gui.add_slider("Jump size (frames)", min=1, max=30, step=1, initial_value=1)
    point_size = server.gui.add_slider("Point size (m)", min=0.001, max=0.020, step=0.001, initial_value=0.005)
    restart = server.gui.add_button("Restart rollout at current frame")
    previous = server.gui.add_button("Previous jump")
    following = server.gui.add_button("Next jump")
    play = server.gui.add_button("Play")
    stop = server.gui.add_button("Stop")
    status = server.gui.add_markdown("")
    server.gui.add_markdown(
        "### Pure Inspire test\n"
        "`teacherforced`: every frame uses GT Inspire state.  "
        "`rollout`: handoff uses GT Inspire state once, then feeds predictions.  "
        "`Previous jump`/`Next jump` stop playback and move by `Jump size (frames)`.  "
        "GT/pred/both and point size are display-only controls."
    )
    state: dict[str, Any] = {"mode": "rollout", "playing": False}
    handles: list[Any] = []
    # Viser property assignments invoke update callbacks synchronously.  The
    # restart button can therefore re-enter the mode callback while rendering;
    # use an RLock to avoid a self-deadlock in that callback path.
    lock = threading.RLock()
    trajectory_path = output / "trajectory.npz"
    last_saved = -1

    def save_trajectory(force: bool = False) -> None:
        nonlocal last_saved
        newest = max(list(runner.teacher_cache.keys()) + list(runner.rollout.states.keys()) + [-1])
        if not force and newest == last_saved:
            return
        if not force and newest not in {runner.rollout.max_rollout_frame} and newest % 16 != 0:
            return
        _write_trajectory(trajectory_path, sequence, runner)
        last_saved = newest

    def on_start_rollout(index: int) -> None:
        runner.rollout.start(int(index))
        save_trajectory(force=True)

    def prediction(index: int) -> dict[str, Any] | None:
        if state["mode"] == "teacherforced":
            result = runner.teacher(index)
            if result is not None:
                save_trajectory()
            return result
        if runner.rollout.handoff_frame is None or index < int(runner.rollout.handoff_frame):
            return None
        result = runner.rollout.ensure(index)
        if result is not None:
            save_trajectory()
        return result

    def render() -> None:
        nonlocal handles
        index = int(frame.value)
        result = prediction(index)
        size = float(point_size.value)
        selected_display = str(display.value)
        if state["mode"] == "teacherforced":
            object_index = min(index + 1, sequence.frame_count - 1)
            gt_current = np.asarray(sequence.hand_points[index])
            gt_next = np.asarray(sequence.hand_points[object_index])
            pred_current = None
            pred_next = None if result is None else result["pred_points"]
        else:
            object_index = index
            gt_current = np.asarray(sequence.hand_points[index])
            gt_next = None
            pred_current = None if result is None else np.asarray(result["points"])
            pred_next = None
        with server.atomic():
            for handle in handles:
                handle.remove()
            clouds: list[Any] = [server.scene.add_point_cloud(
                "/world/dexplore_object", np.asarray(sequence.object_points[object_index]), colors=(170, 170, 170), point_size=max(size * 0.75, 0.001)
            )]
            if selected_display in {"GT", "both"}:
                clouds.append(server.scene.add_point_cloud("/world/inspire_gt_current", gt_current, colors=(60, 130, 240), point_size=size))
                if gt_next is not None:
                    clouds.append(server.scene.add_point_cloud("/world/inspire_gt_next", gt_next, colors=(50, 210, 110), point_size=size))
            if selected_display in {"pred", "both"}:
                if pred_current is not None:
                    clouds.append(server.scene.add_point_cloud("/world/inspire_pred_current", pred_current, colors=(235, 65, 55), point_size=size))
                if pred_next is not None:
                    clouds.append(server.scene.add_point_cloud("/world/inspire_pred_next", pred_next, colors=(255, 145, 35), point_size=size))
            handles = clouds
        if result is None:
            text = "no prediction at this frame"
        elif state["mode"] == "teacherforced":
            valid = np.asarray(result["cm_valid"], dtype=bool)
            text = f"teacher h=1; Cm valid {int(valid.sum())}/{len(valid)}; hand EPE {result['hand_epe_mm']:.2f} mm"
        elif result["cm_valid"] is None:
            text = "rollout initialized from Inspire GT; no decoder step yet"
        else:
            valid = np.asarray(result["cm_valid"], dtype=bool)
            text = f"rollout; Cm valid {int(valid.sum())}/{len(valid)}"
        handoff = "—" if runner.rollout.handoff_frame is None else str(runner.rollout.handoff_frame)
        status.content = (
            f"**{sequence.id}**  \nmode: **{state['mode']}**, display: **{selected_display}**, "
            f"point size: **{size:.3f} m**, jump: **{int(frame_step.value)} frame(s)**  \nframe: **{index}/{sequence.frame_count - 1}**, "
            f"handoff: **{handoff}**  \n{text}"
        )

    def render_locked() -> None:
        with lock:
            render()

    def on_mode_change(_: Any) -> None:
        with lock:
            state["mode"] = str(mode.value)
            if state["mode"] == "rollout":
                on_start_rollout(int(frame.value))
            render()

    def on_restart(_: Any) -> None:
        with lock:
            if str(mode.value) != "rollout":
                mode.value = "rollout"
            else:
                state["mode"] = "rollout"
                on_start_rollout(int(frame.value))
                render()

    def on_jump(direction: int) -> None:
        """Stop playback and move by the selected number of frames."""
        with lock:
            state["playing"] = False
            current = int(frame.value)
            step = int(frame_step.value)
            target = max(0, min(sequence.frame_count - 1, current + int(direction) * step))
            frame.value = target

    mode.on_update(on_mode_change)
    frame.on_update(lambda _: render_locked())
    display.on_update(lambda _: render_locked())
    frame_step.on_update(lambda _: render_locked())
    point_size.on_update(lambda _: render_locked())
    restart.on_click(on_restart)
    previous.on_click(lambda _: on_jump(-1))
    following.on_click(lambda _: on_jump(1))
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    on_start_rollout(int(frame.value))
    render_locked()
    print(f"Viser pure-Inspire test viewer: http://localhost:{port}", flush=True)
    try:
        while True:
            if state["playing"]:
                frame.value = (int(frame.value) + 1) % sequence.frame_count
            time.sleep(1.0 / max(float(fps), 1e-3))
    except KeyboardInterrupt:
        state["playing"] = False
        save_trajectory(force=True)
        manifest["run_status"] = "STOPPED"
        manifest["stopped_at"] = _now()
        manifest["stop_reason"] = "KeyboardInterrupt"
        (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml")
    parser.add_argument("--checkpoint", default="outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--rl-root", default="data/processed_data/inspire_rl")
    parser.add_argument("--parent-root", default="data/processed_data/cm_object_v2_surface512_object_pose_20260830")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--output-root", default="outputs/cmdecoderv2")
    parser.add_argument("--activity-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    rl_root = _resolve(args.rl_root)
    parent_root = _resolve(args.parent_root)
    sequence = _load_sequence(cfg, args.sequence, int(args.sequence_index), rl_root=rl_root, parent_root=parent_root)
    checkpoint_path = _resolve(args.checkpoint)
    model, kinematics, surface = _load_model(cfg, checkpoint_path, device)
    runner = DiagnosticRunner(sequence, model, kinematics, surface, device, cfg)
    run_id = f"cmdecoderv2-inspire-test-vis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output = _resolve(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    trajectory_path = output / "trajectory.npz"
    _write_trajectory(trajectory_path, sequence, runner)
    manifest = {
        "schema_name": "ref2dex_cmdecoderv2_inspire_test_diagnostic_v1",
        "task": "CmDecoderv2",
        "activity_id": args.activity_id,
        "run_id": run_id,
        "run_status": "RUNNING",
        "conclusion": "INCONCLUSIVE",
        "conclusion_scope": "pure Inspire teacher/rollout qualitative diagnostic; not formal split or checkpoint selection",
        "work_version": str(cfg.work_version),
        "operation_category": ["data", "experiment", "operation"],
        "created_at": _now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "formal_decoder_view_index": str(_resolve(str(cfg.data.view_root)) / "index.json"),
        "sequence_id": sequence.id,
        "test_parent_contract": "same 125 held-out parent ids as formal MANO-only test",
        "source": {
            "variant": "dexplore_rl_inspire",
            "raw_q_tensor": str(sequence.rl_q_path),
            "raw_q_tensor_sha256": _sha256(sequence.rl_q_path),
            "parent_grab_cache": str(sequence.parent_cache),
            "rl_root": str(rl_root),
            "object_pose": "Dexplore native q tensor fields 198:205",
            "hand_geometry": "Dexplore Inspire URDF FK from native q 373:391",
        },
        "execution": {
            "teacherforced": "GT Inspire state at every frame",
            "rollout": "GT Inspire state only at selected handoff, then recursive predicted state",
            "window_size": int(cfg.meta.window_size),
            "paired_mano_comparison": False,
            "formal_test_gt": False,
        },
        "outputs": {"trajectory": str(trajectory_path), "run_manifest": str(output / "run_manifest.json")},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_status": "RUNNING", "output": str(output), "sequence": sequence.id, "frames": sequence.frame_count, "viser": f"http://localhost:{args.port}"}, ensure_ascii=False), flush=True)
    serve(runner=runner, output=output, manifest=manifest, host=args.host, port=int(args.port), fps=float(args.fps))


if __name__ == "__main__":
    main()
