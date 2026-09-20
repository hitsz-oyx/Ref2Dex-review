"""Viser viewer for the final MANO/GRAB-source qualitative test.

The decoder is trained on Dexplore RL-Inspire data, but the qualitative test
does not use the paired Dexplore Inspire trajectory as a label.  This viewer
therefore renders the original GRAB parent cache directly:

* ``GT playback`` shows the original MANO hand and GRAB object trajectory;
* switching to ``rollout`` at the current frame initializes Inspire with
  zero finger q and the MANO root wrist pose, then recursively predicts the
  next Inspire state from MANO-source Cm windows;
* ``GT``/``pred``/``both`` selects MANO reference points, predicted Inspire
  points, or both.  No mesh is rendered; the hand and object are point clouds.

This is qualitative-only.  It deliberately does not compare the predicted
Inspire hand against Dexplore RL-Inspire ground truth.
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

from src.base import load_config

from .dataset import _normals_world_to_frame, _points_world_to_frame
from .kinematics import InspireKinematics, rotvec_to_matrix
from .model import CmDecoderV2
from .visualize import InspireSurface


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


class GrabManoSequence:
    """Original parent GRAB geometry used as the final MANO source."""

    def __init__(self, entry: dict[str, Any]) -> None:
        self.entry = dict(entry)
        self.id = str(entry["id"])
        self.frame_count = int(entry["frame_count"])
        geometry_root = _resolve(str(entry["geometry_root"]))
        manifest_path = geometry_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing MANO geometry manifest: {manifest_path}")
        geometry_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parent_value = geometry_manifest.get("input_parent_cache")
        if not parent_value:
            raise ValueError(f"MANO geometry has no input_parent_cache: {manifest_path}")
        self.parent_cache = _resolve(str(parent_value))
        shared = self.parent_cache / "shared"
        right = self.parent_cache / "right"
        self.object_points = np.load(shared / "obj_points_world.npy", mmap_mode="r")
        self.object_normals = np.load(shared / "obj_normals_world.npy", mmap_mode="r")
        self.object_pose = np.load(shared / "obj_pose_world.npy", mmap_mode="r")
        self.hand_points = np.load(right / "hand_points_world.npy", mmap_mode="r")
        self.hand_normals = np.load(right / "hand_normals_world.npy", mmap_mode="r")
        self.wrist_pose = np.load(right / "hand_root_pose_world.npy", mmap_mode="r")
        self.source_frame = np.load(shared / "raw_frame_id.npy", mmap_mode="r")
        expected = self.frame_count
        arrays = {
            "object_points": self.object_points,
            "object_normals": self.object_normals,
            "object_pose": self.object_pose,
            "hand_points": self.hand_points,
            "hand_normals": self.hand_normals,
            "wrist_pose": self.wrist_pose,
            "source_frame": self.source_frame,
        }
        if any(len(value) != expected for value in arrays.values()):
            shapes = {key: tuple(value.shape) for key, value in arrays.items()}
            raise ValueError(f"Parent cache frame mismatch for {self.id}: expected {expected}, got {shapes}")
        if self.hand_points.shape[1:] != (1538, 3):
            raise ValueError(f"Expected MANO hand points [T,1538,3], got {self.hand_points.shape}")
        if not np.isfinite(np.asarray(self.object_pose[:1])).all() or not np.isfinite(np.asarray(self.wrist_pose[:1])).all():
            raise ValueError(f"Non-finite parent pose in {self.parent_cache}")

    def source_window(
        self,
        start: int,
        window_size: int,
        num_obj_points: int,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray]:
        """Build the same object-frame Cm inputs as training, from GRAB data."""
        start = int(start)
        window_size = int(window_size)
        if start < 0 or start + window_size >= self.frame_count:
            raise IndexError(
                f"Cm window start {start} requires frames [{start}, {start + window_size}], "
                f"but sequence has {self.frame_count} frames"
            )
        if num_obj_points > self.object_points.shape[1]:
            raise ValueError(f"num_obj_points={num_obj_points} exceeds parent pool {self.object_points.shape[1]}")
        object_points: list[np.ndarray] = []
        object_normals: list[np.ndarray] = []
        hand_points: list[np.ndarray] = []
        hand_normals: list[np.ndarray] = []
        hand_flow: list[np.ndarray] = []
        for offset in range(window_size):
            frame = start + offset
            pose = np.asarray(self.object_pose[frame], dtype=np.float32)
            # Match the deterministic 1024-point sampling used by the decoder view.
            rng = np.random.default_rng(2024 + start * 131 + offset)
            selected = rng.choice(self.object_points.shape[1], size=num_obj_points, replace=False)
            obj = _points_world_to_frame(self.object_points[frame, selected], pose)
            obj_normal = _normals_world_to_frame(self.object_normals[frame, selected], pose)
            hand = _points_world_to_frame(self.hand_points[frame], pose)
            future_hand = _points_world_to_frame(self.hand_points[frame + 1], pose)
            object_points.append(obj)
            object_normals.append(obj_normal)
            hand_points.append(hand)
            hand_normals.append(_normals_world_to_frame(self.hand_normals[frame], pose))
            hand_flow.append(future_hand - hand)
        batch = {
            "obj_points": torch.from_numpy(np.stack(object_points).astype(np.float32)).unsqueeze(0),
            "obj_normals": torch.from_numpy(np.stack(object_normals).astype(np.float32)).unsqueeze(0),
            "obj_valid_mask": torch.ones((1, window_size, num_obj_points), dtype=torch.bool),
            "hand_points": torch.from_numpy(np.stack(hand_points).astype(np.float32)).unsqueeze(0),
            "hand_normals": torch.from_numpy(np.stack(hand_normals).astype(np.float32)).unsqueeze(0),
            "hand_flow": torch.from_numpy(np.stack(hand_flow).astype(np.float32)).unsqueeze(0),
            "hand_valid_mask": torch.ones((1, window_size, self.hand_points.shape[1]), dtype=torch.bool),
        }
        return batch, np.asarray(self.object_pose[start], dtype=np.float64)


class RolloutEngine:
    """Lazy receding-horizon rollout for an arbitrary handoff frame."""

    def __init__(
        self,
        *,
        sequence: GrabManoSequence,
        model: CmDecoderV2,
        kinematics: InspireKinematics,
        surface: InspireSurface,
        device: torch.device,
        window_size: int,
        num_obj_points: int,
    ) -> None:
        self.sequence = sequence
        self.model = model
        self.kinematics = kinematics
        self.surface = surface
        self.device = device
        self.window_size = int(window_size)
        self.num_obj_points = int(num_obj_points)
        self.max_rollout_frame = self.sequence.frame_count - self.window_size
        self.handoff_frame: int | None = None
        self.states: dict[int, dict[str, Any]] = {}

    def _state_points(self, q: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        transforms = self.kinematics.link_transforms_from_state(q, wrist)
        return self.surface.world_points(transforms).astype(np.float32)

    def start(self, frame: int) -> None:
        frame = int(frame)
        if frame < 0 or frame >= self.sequence.frame_count:
            raise IndexError(frame)
        finger_q = self.kinematics.clamp_finger_q(np.zeros(6, dtype=np.float64))
        # This is the explicitly requested MANO-wrist initialization.  No
        # paired Inspire test pose is used and no axis calibration is inferred.
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

    def ensure(self, frame: int) -> dict[str, Any] | None:
        frame = int(frame)
        if self.handoff_frame is None or frame < self.handoff_frame:
            return None
        if frame in self.states:
            return self.states[frame]
        if frame > self.max_rollout_frame:
            return None
        current_frame = max(self.states)
        while current_frame < frame:
            current = self.states[current_frame]
            source_batch, object_pose = self.sequence.source_window(
                current_frame, self.window_size, self.num_obj_points
            )
            current_q = np.asarray(current["finger_q"], dtype=np.float64)
            current_wrist = np.asarray(current["wrist"], dtype=np.float64)
            current_object = np.linalg.inv(object_pose) @ current_wrist
            source_batch.update({
                "current_finger_q": torch.from_numpy(current_q.astype(np.float32)).unsqueeze(0),
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
            next_q = self.kinematics.clamp_finger_q(current_q + delta_q)
            next_wrist = current_wrist @ delta
            next_frame = current_frame + 1
            self.states[next_frame] = {
                "finger_q": next_q,
                "wrist": next_wrist,
                "points": self._state_points(next_q, next_wrist),
                "cm_valid": output["cm_sample_valid"][0].detach().cpu().numpy().astype(bool),
            }
            current_frame = next_frame
        return self.states[frame]

    def export(self) -> dict[str, np.ndarray]:
        frame_count = self.sequence.frame_count
        q = np.full((frame_count, 6), np.nan, dtype=np.float32)
        wrist = np.full((frame_count, 4, 4), np.nan, dtype=np.float32)
        points = np.full((frame_count, self.surface.points.shape[0], 3), np.nan, dtype=np.float32)
        valid = np.zeros((frame_count, self.window_size), dtype=bool)
        for frame, state in self.states.items():
            q[frame] = np.asarray(state["finger_q"], dtype=np.float32)
            wrist[frame] = np.asarray(state["wrist"], dtype=np.float32)
            points[frame] = np.asarray(state["points"], dtype=np.float32)
            if state["cm_valid"] is not None:
                valid[frame] = np.asarray(state["cm_valid"], dtype=bool)
        return {
            "predicted_finger_q": q,
            "predicted_wrist_pose_world": wrist,
            "predicted_inspire_points_world": points,
            "cm_sample_valid": valid,
            "handoff_frame": np.asarray(-1 if self.handoff_frame is None else self.handoff_frame, dtype=np.int32),
        }


def _load_sequence(cfg: Any, sequence: str, sequence_index: int) -> tuple[dict[str, Any], Path]:
    view_root = _resolve(str(cfg.data.view_root))
    index_path = view_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = list(index["sequences"]["test"])
    if any(item.get("variant") != "mano" or not item.get("qualitative_only") for item in entries):
        raise ValueError("MANO viewer requires qualitative-only MANO test entries")
    if sequence:
        matches = [item for item in entries if str(item["id"]) == sequence]
        if not matches:
            raise ValueError(f"Unknown MANO test sequence {sequence!r}")
        return matches[0], index_path
    if sequence_index < 0 or sequence_index >= len(entries):
        raise IndexError(f"sequence-index must lie in [0, {len(entries) - 1}]")
    return entries[sequence_index], index_path


def _load_model(cfg: Any, checkpoint_path: Path, device: torch.device) -> tuple[CmDecoderV2, InspireKinematics, InspireSurface]:
    cfg.model.meta = cfg.meta
    model = CmDecoderV2(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    urdf = _resolve(str(cfg.data.urdf_path))
    return model, InspireKinematics(urdf), InspireSurface(urdf)


def _write_trajectory(path: Path, sequence: GrabManoSequence, engine: RolloutEngine) -> None:
    exported = engine.export()
    np.savez_compressed(
        path,
        mano_hand_points_world=np.asarray(sequence.hand_points, dtype=np.float32),
        grab_object_points_world=np.asarray(sequence.object_points, dtype=np.float32),
        mano_wrist_pose_world=np.asarray(sequence.wrist_pose, dtype=np.float32),
        grab_object_pose_world=np.asarray(sequence.object_pose, dtype=np.float32),
        source_frame_id=np.asarray(sequence.source_frame, dtype=np.int64),
        **exported,
    )


def serve(
    *,
    sequence: GrabManoSequence,
    engine: RolloutEngine,
    output: Path,
    manifest: dict[str, Any],
    host: str,
    port: int,
    fps: float,
) -> None:
    import viser

    server = viser.ViserServer(host=host, port=int(port))
    mode = server.gui.add_dropdown("Execution mode", options=("gt_playback", "rollout"), initial_value="gt_playback")
    display = server.gui.add_dropdown("Hand display", options=("GT", "pred", "both"), initial_value="both")
    frame = server.gui.add_slider("Frame", min=0, max=sequence.frame_count - 1, step=1, initial_value=0)
    point_size = server.gui.add_slider("Point size (m)", min=0.001, max=0.020, step=0.001, initial_value=0.005)
    restart = server.gui.add_button("Restart rollout at current frame")
    play = server.gui.add_button("Play")
    stop = server.gui.add_button("Stop")
    status = server.gui.add_markdown("")
    server.gui.add_markdown(
        "### 语义\n"
        "- `gt_playback`: 原始 GRAB MANO 手 + 原始 GRAB 物体轨迹。\n"
        "- `rollout`: 在当前帧以 **q=0 + MANO wrist** 初始化 Inspire；之后只用 MANO-source Cm window 递归预测。\n"
        "- `GT/pred/both`: GT 是 MANO source，pred 是 Inspire prediction；不读取 paired Inspire test GT。"
    )
    state: dict[str, Any] = {"mode": "gt_playback", "playing": False}
    handles: list[Any] = []
    lock = threading.Lock()
    trajectory_path = output / "trajectory.npz"
    last_saved_state_frame = -1

    def save_trajectory(*, force: bool = False) -> None:
        nonlocal last_saved_state_frame
        newest = max(engine.states, default=-1)
        if not force and newest == last_saved_state_frame:
            return
        # Compression is intentionally amortized.  Viser callbacks should not
        # block on a full 271-frame trajectory write for every slider tick.
        if not force and newest not in {engine.max_rollout_frame} and newest % 16 != 0:
            return
        _write_trajectory(trajectory_path, sequence, engine)
        last_saved_state_frame = newest

    def start_rollout(index: int) -> None:
        try:
            engine.start(int(index))
        except ValueError as error:
            status.content = f"**无法在 frame {index} 开始 rollout**：`{error}`"
            return
        save_trajectory(force=True)

    def current_prediction(index: int) -> dict[str, Any] | None:
        if state["mode"] != "rollout" or engine.handoff_frame is None:
            return None
        if index < int(engine.handoff_frame):
            return None
        value = engine.ensure(index)
        if value is not None:
            save_trajectory()
        return value

    def render() -> None:
        nonlocal handles
        index = int(frame.value)
        prediction = current_prediction(index)
        selected_display = str(display.value)
        size = float(point_size.value)
        with server.atomic():
            for handle in handles:
                handle.remove()
            clouds: list[Any] = [
                server.scene.add_point_cloud(
                    "/world/grab_object", np.asarray(sequence.object_points[index]), colors=(170, 170, 170), point_size=max(size * 0.75, 0.001)
                )
            ]
            if selected_display in {"GT", "both"}:
                clouds.append(server.scene.add_point_cloud(
                    "/world/mano_gt", np.asarray(sequence.hand_points[index]), colors=(60, 130, 240), point_size=size
                ))
            if selected_display in {"pred", "both"} and prediction is not None:
                clouds.append(server.scene.add_point_cloud(
                    "/world/inspire_pred", np.asarray(prediction["points"]), colors=(235, 65, 55), point_size=size
                ))
            handles = clouds
        if prediction is None and state["mode"] == "rollout" and engine.handoff_frame is not None and index > engine.max_rollout_frame:
            prediction_text = f"pred unavailable after frame {engine.max_rollout_frame} (Cm window needs K+1 source frames)"
        elif prediction is None and state["mode"] == "rollout":
            prediction_text = "pred unavailable before handoff"
        elif prediction is None:
            prediction_text = "pred unavailable in GT playback; switch mode at the desired handoff frame"
        else:
            if prediction["cm_valid"] is None:
                prediction_text = "pred Inspire initialized at handoff; no decoder step has been executed yet"
            else:
                valid = np.asarray(prediction["cm_valid"], dtype=bool)
                prediction_text = f"pred Inspire available; Cm valid {int(valid.sum())}/{len(valid)}"
        handoff_text = "—" if engine.handoff_frame is None else str(engine.handoff_frame)
        status.content = (
            f"**{sequence.id}**  \n"
            f"mode: **{state['mode']}**, display: **{selected_display}**, point size: **{size:.3f} m**  \n"
            f"frame: **{index}/{sequence.frame_count - 1}**, GRAB source frame: **{int(sequence.source_frame[index])}**, "
            f"handoff: **{handoff_text}**  \n"
            f"{prediction_text}"
        )

    def render_locked() -> None:
        with lock:
            render()

    def on_mode_change(_: Any) -> None:
        with lock:
            state["mode"] = str(mode.value)
            if state["mode"] == "rollout":
                start_rollout(int(frame.value))
            render()

    def on_frame_change(_: Any) -> None:
        render_locked()

    def on_restart(_: Any) -> None:
        with lock:
            state["mode"] = "rollout"
            mode.value = "rollout"
            start_rollout(int(frame.value))
            render()

    mode.on_update(on_mode_change)
    frame.on_update(on_frame_change)
    display.on_update(lambda _: render_locked())
    point_size.on_update(lambda _: render_locked())
    restart.on_click(on_restart)
    play.on_click(lambda _: state.update(playing=True))
    stop.on_click(lambda _: state.update(playing=False))
    render_locked()
    print(f"Viser MANO/GRAB qualitative viewer: http://localhost:{port}", flush=True)
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
    parser.add_argument(
        "--checkpoint",
        default="outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_1_1_20260906_115701/checkpoints/best.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Limit GT frames for a smoke viewer; 0 means all")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8098)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--output-root", default="outputs/cmdecoderv2")
    parser.add_argument("--activity-id", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    sequence_entry, index_path = _load_sequence(cfg, args.sequence, int(args.sequence_index))
    sequence = GrabManoSequence(sequence_entry)
    if args.max_frames:
        limit = int(args.max_frames)
        if limit < int(cfg.meta.window_size) + 1:
            raise ValueError("max-frames must be at least window_size+1")
        # Keep this smoke-only option explicit: all source arrays are sliced,
        # while the original index/parent cache remain unchanged.
        sequence.frame_count = min(sequence.frame_count, limit)
        for name in ("object_points", "object_normals", "object_pose", "hand_points", "hand_normals", "wrist_pose", "source_frame"):
            setattr(sequence, name, getattr(sequence, name)[: sequence.frame_count])
    checkpoint_path = _resolve(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model, kinematics, surface = _load_model(cfg, checkpoint_path, device)
    engine = RolloutEngine(
        sequence=sequence,
        model=model,
        kinematics=kinematics,
        surface=surface,
        device=device,
        window_size=int(cfg.meta.window_size),
        num_obj_points=int(cfg.meta.num_obj_points),
    )
    run_id = f"cmdecoderv2-mano-grab-vis-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output = _resolve(args.output_root) / run_id
    output.mkdir(parents=True, exist_ok=False)
    trajectory_path = output / "trajectory.npz"
    _write_trajectory(trajectory_path, sequence, engine)
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "CmDecoderv2",
        "activity_id": args.activity_id,
        "run_id": run_id,
        "run_status": "RUNNING",
        "conclusion": "INCONCLUSIVE",
        "conclusion_scope": "qualitative Viser only; no paired Inspire test GT or quantitative test metric",
        "work_version": str(cfg.work_version),
        "operation_category": ["code", "experiment", "operation"],
        "created_at": _now(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "decoder_view_index": str(index_path),
        "sequence_id": sequence.id,
        "source": {
            "hand": "original_grab_mano_parent_cache",
            "object": "original_grab_object_parent_cache",
            "parent_cache": str(sequence.parent_cache),
            "parent_geometry_manifest": str(_resolve(str(sequence.entry["geometry_root"])) / "manifest.json"),
            "paired_inspire_test_gt": False,
        },
        "execution": "gt_playback_then_h1_rollout",
        "rollout_contract": {
            "handoff": "arbitrary selected frame",
            "initial_finger_q": [0.0] * 6,
            "initial_wrist": "direct MANO hand_root_pose_world at handoff frame",
            "source_cm": "GRAB MANO hand/object window, object_pose_t coordinates",
            "mesh_rendered": False,
        },
        "outputs": {"trajectory": str(trajectory_path), "run_manifest": str(output / "run_manifest.json")},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "run_id": run_id,
        "run_status": "RUNNING",
        "output": str(output),
        "sequence": sequence.id,
        "frames": sequence.frame_count,
        "rollout_max_frame": engine.max_rollout_frame,
        "viser": f"http://localhost:{args.port}",
    }, ensure_ascii=False), flush=True)
    serve(
        sequence=sequence,
        engine=engine,
        output=output,
        manifest=manifest,
        host=args.host,
        port=int(args.port),
        fps=float(args.fps),
    )


if __name__ == "__main__":
    main()
