"""Qualitative MANO-source receding-horizon rollout for CmDecoderv2.

No paired Inspire test target is read or reported. The script exports the
predicted trajectory and optional frames/video for human inspection only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh

from src.base import load_config

from .dataset import _normals_world_to_frame, _points_world_to_frame
from .kinematics import InspireKinematics, _origin, expand_finger_q, rotvec_to_matrix
from .model import CmDecoderV2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _load_mano_wrist_pose(sequence: dict[str, Any]) -> tuple[np.ndarray, Path]:
    """Load the MANO right-hand root pose for a MANO-only test sequence.

    The decoder view intentionally omits Inspire GT from MANO test entries, but
    the parent GRAB geometry cache still stores ``hand_root_pose_world``.  This
    pose is used only to initialize the predicted Inspire wrist; it is not read
    as a decoder target or evaluation label.
    """
    geometry_root = _resolve(str(sequence["geometry_root"]))
    manifest_path = geometry_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates: list[Path] = []
    parent_cache = manifest.get("input_parent_cache")
    if parent_cache:
        candidates.append(_resolve(str(parent_cache)) / "right" / "hand_root_pose_world.npy")
    candidates.append(geometry_root / "hand_root_pose_world.npy")
    pose_path = next((path for path in candidates if path.is_file()), None)
    if pose_path is None:
        raise FileNotFoundError(
            f"MANO wrist root pose is unavailable for {sequence['id']}; "
            f"checked {[str(path) for path in candidates]}"
        )
    pose = np.load(pose_path, mmap_mode="r")
    expected_shape = (int(sequence["frame_count"]), 4, 4)
    if pose.shape != expected_shape:
        raise ValueError(f"Expected MANO wrist pose shape {expected_shape}, got {pose.shape} from {pose_path}")
    if not np.isfinite(np.asarray(pose[:1])).all():
        raise ValueError(f"MANO wrist pose contains non-finite values: {pose_path}")
    return pose, pose_path


class InspireSurface:
    """Fixed URDF visual-vertex sample used only for qualitative rendering."""

    def __init__(self, urdf: Path, sample_count: int = 1538, seed: int = 2024) -> None:
        root = ET.parse(urdf).getroot()
        point_parts: list[np.ndarray] = []
        link_parts: list[np.ndarray] = []
        for link_index, link in enumerate(root.findall("link")):
            link_name = str(link.get("name"))
            for visual in link.findall("visual"):
                mesh = visual.find("./geometry/mesh")
                if mesh is None or not mesh.get("filename"):
                    continue
                path = urdf.parent / str(mesh.get("filename"))
                loaded = trimesh.load(path, force="mesh", process=False)
                vertices = np.asarray(loaded.vertices, dtype=np.float64)
                scale = np.asarray([float(value) for value in str(mesh.get("scale", "1 1 1")).split()])
                local = _origin(visual.find("origin"))
                vertices = (vertices * scale[None]) @ local[:3, :3].T + local[:3, 3]
                point_parts.append(vertices)
                link_parts.append(np.full(len(vertices), link_name, dtype=object))
        points = np.concatenate(point_parts)
        links = np.concatenate(link_parts)
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(points), size=sample_count, replace=len(points) < sample_count)
        self.points = points[selected]
        self.links = links[selected]

    def world_points(self, transforms: dict[str, np.ndarray]) -> np.ndarray:
        output = np.empty_like(self.points, dtype=np.float32)
        for link in np.unique(self.links):
            mask = self.links == link
            transform = transforms[str(link)]
            output[mask] = self.points[mask] @ transform[:3, :3].T + transform[:3, 3]
        return output


def _rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:3, :2].T.reshape(-1), dtype=np.float32)


def _make_source_window(sequence: dict[str, Any], start: int, k: int, num_obj_points: int) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    geometry = Path(sequence["geometry_root"])
    obj = np.load(geometry / "obj_points_pool_world.npy", mmap_mode="r")
    normals = np.load(geometry / "obj_normals_pool_world.npy", mmap_mode="r")
    pose = np.load(geometry / "obj_pose_world.npy", mmap_mode="r")
    hand = np.load(geometry / "hand_points_world.npy", mmap_mode="r")
    hand_normals = np.load(geometry / "hand_normals_world.npy", mmap_mode="r")
    object_points, object_normals, hand_points, hand_normal_values, hand_flow = [], [], [], [], []
    for offset in range(k):
        frame = start + offset
        current_pose = np.asarray(pose[frame], dtype=np.float32)
        rng = np.random.default_rng(2024 + start * 131 + offset)
        selected = rng.choice(obj.shape[1], size=num_obj_points, replace=False)
        object_points.append(_points_world_to_frame(obj[frame, selected], current_pose))
        object_normals.append(_normals_world_to_frame(normals[frame, selected], current_pose))
        current_hand = _points_world_to_frame(hand[frame], current_pose)
        future_hand = _points_world_to_frame(hand[frame + 1], current_pose)
        hand_points.append(current_hand)
        hand_normal_values.append(_normals_world_to_frame(hand_normals[frame], current_pose))
        hand_flow.append(future_hand - current_hand)
    batch = {
        "obj_points": torch.from_numpy(np.stack(object_points).astype(np.float32)).unsqueeze(0),
        "obj_normals": torch.from_numpy(np.stack(object_normals).astype(np.float32)).unsqueeze(0),
        "obj_valid_mask": torch.ones((1, k, num_obj_points), dtype=torch.bool),
        "hand_points": torch.from_numpy(np.stack(hand_points).astype(np.float32)).unsqueeze(0),
        "hand_normals": torch.from_numpy(np.stack(hand_normal_values).astype(np.float32)).unsqueeze(0),
        "hand_flow": torch.from_numpy(np.stack(hand_flow).astype(np.float32)).unsqueeze(0),
        "hand_valid_mask": torch.ones((1, k, hand.shape[1]), dtype=torch.bool),
    }
    # Decoder state is expressed against object pose t. The post-execution
    # visualization compares predicted t+1 against source/object frame t+1.
    return (
        batch,
        np.asarray(pose[start], dtype=np.float64),
        np.asarray(pose[start + 1], dtype=np.float64),
        np.asarray(obj[start + 1], dtype=np.float32),
        np.asarray(hand[start + 1], dtype=np.float32),
    )


def _render_frame(path: Path, object_points: np.ndarray, mano_points: np.ndarray, inspire_points: np.ndarray, title: str) -> None:
    figure = plt.figure(figsize=(7, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(object_points[::8, 0], object_points[::8, 1], object_points[::8, 2], s=2, c="#777777", alpha=0.45, label="object")
    axis.scatter(mano_points[::4, 0], mano_points[::4, 1], mano_points[::4, 2], s=2, c="#2b6cb0", alpha=0.45, label="MANO source")
    axis.scatter(inspire_points[::3, 0], inspire_points[::3, 1], inspire_points[::3, 2], s=3, c="#e53e3e", alpha=0.75, label="Inspire predicted")
    combined = np.concatenate([object_points, mano_points, inspire_points])
    center = 0.5 * (combined.min(axis=0) + combined.max(axis=0))
    radius = max(float(np.ptp(combined, axis=0).max()) * 0.55, 0.05)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_title(title)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def rollout(args: argparse.Namespace) -> Path:
    cfg = load_config(args.config)
    device = torch.device(args.device)
    view_root = _resolve(str(cfg.data.view_root))
    view = json.loads((view_root / "index.json").read_text(encoding="utf-8"))
    candidates = view["sequences"]["test"]
    if any(item.get("variant") != "mano" or not item.get("qualitative_only") for item in candidates):
        raise ValueError("Visualization input must be MANO qualitative-only and contain no Inspire GT")
    if args.sequence:
        matches = [item for item in candidates if item["id"] == args.sequence]
        if not matches:
            raise ValueError(f"Unknown MANO test sequence {args.sequence!r}")
        sequence = matches[0]
    else:
        sequence = candidates[args.sequence_index]
    if any(key in sequence for key in ("q_native", "wrist_pose_world", "q_finger_real")):
        raise ValueError("MANO test entry unexpectedly exposes Inspire GT")
    checkpoint_path = _resolve(args.checkpoint)
    cfg.model.meta = cfg.meta
    model = CmDecoderV2(cfg.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    kinematics = InspireKinematics(cfg.data.urdf_path)
    surface = InspireSurface(_resolve(str(cfg.data.urdf_path)))
    k = int(cfg.meta.window_size)
    maximum = int(sequence["frame_count"]) - k
    steps = min(int(args.max_steps), maximum)
    if steps <= 0:
        raise ValueError("Sequence is too short for a Cm window")
    if int(args.render_every) <= 0:
        raise ValueError("render_every must be positive")
    geometry_pose = np.load(Path(sequence["geometry_root"]) / "obj_pose_world.npy", mmap_mode="r")
    finger = kinematics.clamp_finger_q(np.asarray(args.initial_finger_q, dtype=np.float64))
    mano_wrist_path: Path | None = None
    mano_wrist_pose: np.ndarray | None = None
    if args.initial_state == "mano_wrist":
        mano_wrist_pose, mano_wrist_path = _load_mano_wrist_pose(sequence)
        wrist = np.asarray(mano_wrist_pose[0], dtype=np.float64).copy()
    else:
        initial_local = np.eye(4, dtype=np.float64)
        initial_local[:3, 3] = np.asarray(args.initial_wrist_translation_object, dtype=np.float64)
        initial_local[:3, :3] = rotvec_to_matrix(np.radians(np.asarray(args.initial_wrist_rotation_deg, dtype=np.float64)))
        wrist = np.asarray(geometry_pose[0], dtype=np.float64) @ initial_local
    predicted_finger = [finger.copy()]
    predicted_wrist = [wrist.copy()]
    predicted_native = [expand_finger_q(finger).astype(np.float32)]
    predicted_surface, source_hand, object_surface, source_frames, cm_valid = [], [], [], [], []
    output_root = _resolve(args.output_root)
    run_id = f"cmdecoderv2-mano-viz-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output = output_root / run_id
    frames_dir = output / "frames"
    frames_dir.mkdir(parents=True, exist_ok=False)
    for start in range(steps):
        source_batch, object_pose, render_pose, object_world, mano_world = _make_source_window(sequence, start, k, int(cfg.meta.num_obj_points))
        current_object = np.linalg.inv(object_pose) @ wrist
        source_batch.update({
            "current_finger_q": torch.from_numpy(finger.astype(np.float32)).unsqueeze(0),
            "current_wrist_translation_object": torch.from_numpy(current_object[:3, 3].astype(np.float32)).unsqueeze(0),
            "current_wrist_rotation_6d_object": torch.from_numpy(_rotation_6d(current_object[:3, :3])).unsqueeze(0),
            "current_link_features": torch.from_numpy(kinematics.query_features(finger, wrist, object_pose)).unsqueeze(0),
        })
        source_batch = {key: value.to(device) for key, value in source_batch.items()}
        with torch.no_grad():
            prediction = model(source_batch)
        q_delta = prediction["pred_q_delta"][0, 0].cpu().numpy()
        translation = prediction["pred_wrist_translation"][0, 0].cpu().numpy()
        rotvec = prediction["pred_wrist_rotvec"][0, 0].cpu().numpy()
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = rotvec_to_matrix(rotvec)
        delta[:3, 3] = translation
        finger = kinematics.clamp_finger_q(finger + q_delta)
        wrist = wrist @ delta
        transforms = kinematics.link_transforms_from_state(finger, wrist)
        predicted_world = surface.world_points(transforms)
        predicted_object = _points_world_to_frame(predicted_world, render_pose)
        mano_object = _points_world_to_frame(mano_world, render_pose)
        object_object = _points_world_to_frame(object_world, render_pose)
        predicted_finger.append(finger.copy())
        predicted_wrist.append(wrist.copy())
        predicted_native.append(expand_finger_q(finger).astype(np.float32))
        predicted_surface.append(predicted_object)
        source_hand.append(mano_object)
        object_surface.append(object_object)
        source_frames.append(start + 1)
        cm_valid.append(prediction["cm_sample_valid"][0].cpu().numpy())
        if start % int(args.render_every) == 0:
            _render_frame(frames_dir / f"frame_{start:05d}.png", object_object, mano_object, predicted_object, f"{sequence['id']} / frame {start}")
    trajectory_path = output / "trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        predicted_finger_q=np.asarray(predicted_finger, dtype=np.float32),
        predicted_native_finger_q=np.asarray(predicted_native, dtype=np.float32),
        predicted_wrist_pose_world=np.asarray(predicted_wrist, dtype=np.float32),
        predicted_hand_points_object=np.asarray(predicted_surface, dtype=np.float32),
        source_mano_points_object=np.asarray(source_hand, dtype=np.float32),
        object_points_object=np.asarray(object_surface, dtype=np.float32),
        source_cache_frame=np.asarray(source_frames, dtype=np.int32),
        cm_sample_valid=np.asarray(cm_valid, dtype=bool),
    )
    video_path: Path | None = None
    if args.write_mp4:
        try:
            video_path = output / "rollout.mp4"
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError("ffmpeg is required for --write-mp4")
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-framerate",
                    str(float(cfg.meta.effective_fps) / int(args.render_every)),
                    "-i",
                    str(frames_dir / "frame_%05d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(video_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if video_path.stat().st_size < 1024:
                raise RuntimeError(f"ffmpeg produced an unexpectedly small MP4: {video_path.stat().st_size} bytes")
        except Exception as error:
            video_path = None
            (output / "mp4_error.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema_name": "ref2dex_run_manifest_v1",
        "task": "CmDecoderv2",
        "activity_id": args.activity_id,
        "run_id": run_id,
        "run_status": "COMPLETED",
        "conclusion": "INCONCLUSIVE",
        "conclusion_scope": "qualitative export only; awaiting human inspection",
        "work_version": str(cfg.work_version),
        "operation_category": ["experiment", "operation"],
        "created_at": _now(),
        "base_commit": commit,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "view_index": str(view_root / "index.json"),
        "sequence_id": sequence["id"],
        "steps": steps,
        "execution": "receding_horizon_h1_only",
        "paired_inspire_test_gt": False,
        "initial_state": {
            "mode": args.initial_state,
            "finger_q": list(args.initial_finger_q),
            "wrist_translation_object_m": list(args.initial_wrist_translation_object),
            "wrist_rotation_object_deg_rotvec": list(args.initial_wrist_rotation_deg),
            "mano_wrist_pose_world_frame0": None if mano_wrist_pose is None else np.asarray(mano_wrist_pose[0]).tolist(),
            "mano_wrist_source": None if mano_wrist_path is None else str(mano_wrist_path),
        },
        "outputs": {"trajectory": str(trajectory_path), "frames": str(frames_dir), "video": None if video_path is None else str(video_path)},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_status": "COMPLETED", "output": str(output), "sequence": sequence["id"], "steps": steps}, ensure_ascii=False))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="src/task/CmDecoderv2/configs/active/dexplore_rl_v1_1_1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequence", default="")
    parser.add_argument("--sequence-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--initial-state", choices=("mano_wrist", "manual"), default="mano_wrist")
    parser.add_argument("--initial-finger-q", type=float, nargs=6, default=(0, 0, 0, 0, 0, 0))
    parser.add_argument("--initial-wrist-translation-object", type=float, nargs=3, default=(0, 0, 0.15))
    parser.add_argument("--initial-wrist-rotation-deg", type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--write-mp4", action="store_true")
    parser.add_argument("--output-root", default="outputs/cmdecoderv2")
    parser.add_argument("--activity-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    rollout(parse_args())
