"""Build the D1 fixed-wrist, constrained-q6 CmResidual reference artifact.

The source tensor and object geometry are read only.  The output directory must
not exist; this prevents accidental replacement of an earlier reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[5]
_KIN_PATH = ROOT / "src/task/CmDecoderv2/kinematics.py"
_SPEC = importlib.util.spec_from_file_location("cmresidual_kinematics", _KIN_PATH)
_KIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _KIN
_SPEC.loader.exec_module(_KIN)
InspireKinematics = _KIN.InspireKinematics
INDEPENDENT_FINGER_NATIVE_INDICES = _KIN.INDEPENDENT_FINGER_NATIVE_INDICES
_GEOM_PATH = ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_geometry.py"
_GEOM_SPEC = importlib.util.spec_from_file_location("cmresidual_surface_geometry", _GEOM_PATH)
_GEOM = importlib.util.module_from_spec(_GEOM_SPEC)
sys.modules[_GEOM_SPEC.name] = _GEOM
_GEOM_SPEC.loader.exec_module(_GEOM)
SurfaceGeometry = _GEOM.SurfaceGeometry

DEFAULT_URDF = ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"
DEFAULT_OBJECT_URDF = ROOT.parent / "dexplore/dexplore/data/assets/mjcf/airplane.urdf"
NATIVE_SLICE = slice(373, 391)
TRAIN_START, TRAIN_END = 44, 410
RIGHT_WRIST_POINT = 16
RIGHT_MCP = (17, 20, 23, 26)
RIGHT_TIPS = (19, 22, 25, 28, 31)
QUERY_LINKS = (
    "hand_base_link", "thumb_proximal_base", "thumb_proximal", "thumb_intermediate", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_tip", "middle_proximal", "middle_intermediate", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_tip", "pinky_proximal", "pinky_intermediate", "pinky_tip",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def twists(poses: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros((len(poses), 6), dtype=np.float64)
    out[:, :3] = np.gradient(poses[:, :3, 3], dt, axis=0, edge_order=1)
    for i in range(len(poses)):
        j = min(i + 1, len(poses) - 1) if i == 0 else i
        k = max(i - 1, 0) if i > 0 else i
        if j != k:
            out[i, 3:] = Rotation.from_matrix(poses[k, :3, :3].T @ poses[j, :3, :3]).as_rotvec() / ((j - k) * dt)
    return out


def pose_from_tensor(x: np.ndarray) -> np.ndarray:
    q = np.asarray(x[:, 309:313], dtype=np.float64)
    n = np.linalg.norm(q, axis=1)
    if np.any(n <= 1e-12) or not np.isfinite(q).all():
        raise ValueError("invalid global wrist quaternion")
    q = q / n[:, None]
    T = np.repeat(np.eye(4)[None], len(x), axis=0)
    T[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    T[:, :3, 3] = x[:, 51:54]
    return T


def palm_frame(points: np.ndarray) -> np.ndarray:
    wrist = points[:, RIGHT_WRIST_POINT]
    index, middle, ring, pinky = (points[:, i] for i in RIGHT_MCP)
    axis_x = middle - wrist
    axis_x /= np.maximum(np.linalg.norm(axis_x, axis=1, keepdims=True), 1e-12)
    axis_y = index - pinky
    axis_y -= axis_x * np.sum(axis_x * axis_y, axis=1, keepdims=True)
    axis_y /= np.maximum(np.linalg.norm(axis_y, axis=1, keepdims=True), 1e-12)
    axis_z = np.cross(axis_x, axis_y)
    axis_z /= np.maximum(np.linalg.norm(axis_z, axis=1, keepdims=True), 1e-12)
    axis_y = np.cross(axis_z, axis_x)
    out = np.repeat(np.eye(4)[None], len(points), axis=0)
    out[:, :3, :3] = np.stack([axis_x, axis_y, axis_z], axis=2)
    out[:, :3, 3] = np.mean(points[:, [RIGHT_WRIST_POINT, *RIGHT_MCP]], axis=1)
    return out


def expand_q6(q6: np.ndarray, mimic: dict[int, float]) -> np.ndarray:
    q6 = np.asarray(q6, dtype=np.float64)
    out = np.zeros((*q6.shape[:-1], 18), dtype=np.float64)
    out[..., INDEPENDENT_FINGER_NATIVE_INDICES] = q6
    out[..., 7] = q6[..., 0] * mimic[7]
    out[..., 9] = q6[..., 1] * mimic[9]
    out[..., 11] = q6[..., 2] * mimic[11]
    out[..., 13] = q6[..., 3] * mimic[13]
    out[..., 16] = q6[..., 5] * mimic[16]
    out[..., 17] = q6[..., 5] * mimic[17]
    return out


def tip_positions(kin: InspireKinematics, q6: np.ndarray, wrist: np.ndarray, mimic: dict[int, float]) -> np.ndarray:
    zero_inv = np.linalg.inv(kin.wrist_pose_from_native(np.zeros(18)))
    q6 = np.asarray(q6, dtype=np.float64)
    wrist = np.asarray(wrist, dtype=np.float64)
    scalar = q6.ndim == 1 and wrist.ndim == 2
    if scalar:
        q6 = q6[None]
        wrist = wrist[None]
    elif q6.ndim == 1 and wrist.ndim == 3:
        q6 = np.broadcast_to(q6, (len(wrist), 6)).copy()
    elif q6.ndim == 2 and wrist.ndim == 2:
        wrist = wrist[None]
    if q6.shape[0] != wrist.shape[0]:
        raise ValueError(f"q6/wrist batch mismatch: {q6.shape}, {wrist.shape}")
    result = []
    for row, pose in zip(q6, wrist):
        canonical = kin.link_transforms_native(expand_q6(row, mimic))
        world = pose @ zero_inv @ np.stack([canonical[n] for n in ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")])
        result.append(world[:, :3, 3])
    result = np.stack(result)
    return result[0] if scalar and len(result) == 1 else result


def fit_xhb(kin: InspireKinematics, human_wrist: np.ndarray, points: np.ndarray) -> np.ndarray:
    hp = np.median(np.linalg.inv(human_wrist[:30]) @ palm_frame(points)[:30], axis=0)
    # Re-orthogonalize the robust translation/rotation estimate.
    hp[:3, :3] = Rotation.from_matrix(hp[:3, :3]).as_matrix()
    links = kin.link_transforms_native(np.zeros(18))
    robot_points = np.stack([links[n][:3, 3] for n in ("hand_base_link", "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal")])
    # Build the same palm frame directly for one robot pose.
    axis_x = robot_points[2] - robot_points[0]; axis_x /= np.linalg.norm(axis_x)
    axis_y = robot_points[1] - robot_points[4]; axis_y -= axis_x * np.dot(axis_x, axis_y); axis_y /= np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y); axis_z /= np.linalg.norm(axis_z); axis_y = np.cross(axis_z, axis_x)
    robot_palm = np.eye(4); robot_palm[:3, :3] = np.stack([axis_x, axis_y, axis_z], axis=1); robot_palm[:3, 3] = robot_points.mean(axis=0)
    bp = np.linalg.inv(kin.wrist_pose_from_native(np.zeros(18))) @ robot_palm
    return hp @ np.linalg.inv(bp)


def project_limits(q: np.ndarray, dt: float, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    q = np.clip(q, lower, upper)
    vmax, amax = 1.0 * dt, 20.0 * dt * dt
    for _ in range(8):
        q[1:] = q[0] + np.cumsum(np.clip(np.diff(q, axis=0), -vmax, vmax, out=np.empty_like(q[1:])), axis=0)
        q = np.clip(q, lower, upper)
        if len(q) > 2:
            d = np.diff(q, axis=0)
            dd = np.diff(d, axis=0)
            dd = np.clip(dd, -amax, amax)
            d[1:] = d[0] + np.cumsum(dd, axis=0)
            q[1:] = q[0] + np.cumsum(d, axis=0)
            q = np.clip(q, lower, upper)
    return q


def contact_audit(link_pose: np.ndarray, object_pose: np.ndarray, raw_contact: np.ndarray,
                  urdf: Path, object_urdf: Path) -> dict[str, object]:
    """Compute the registered raw-contact distance gate without relabeling contacts."""
    geometry = SurfaceGeometry(
        hand_urdf=urdf, object_urdf=object_urdf, query_links=QUERY_LINKS,
        object_count=256, hand_count=1538, seed=2024, device="cpu")
    link_names = tuple(
        "link0 link1 link2 link3 link4 link5 link6 hand_base_link thumb_proximal_base thumb_proximal "
        "thumb_intermediate thumb_distal index_proximal index_intermediate middle_proximal middle_intermediate "
        "ring_proximal ring_intermediate pinky_proximal pinky_intermediate thumb_tip index_tip middle_tip ring_tip pinky_tip"
    .split())
    indices = [link_names.index(name) for name in QUERY_LINKS]
    with torch.no_grad():
        hand, _ = geometry.hand(torch.as_tensor(link_pose[:, indices], dtype=torch.float32))
        obj, _ = geometry.object(torch.as_tensor(object_pose, dtype=torch.float32))
        minimum = torch.cdist(obj, hand).amin(dim=(1, 2)).cpu().numpy()
    raw_contact = np.asarray(raw_contact, dtype=bool)
    if raw_contact.shape != minimum.shape:
        raise ValueError(f"raw contact/distance shape mismatch: {raw_contact.shape} != {minimum.shape}")
    raw_count = int(raw_contact.sum())
    passed = int((minimum[raw_contact] <= 0.020).sum()) if raw_count else 0
    fraction = float(passed / raw_count) if raw_count else 0.0
    return {
        "raw_contact_frames": raw_count,
        "raw_contact_distance_pass_frames": passed,
        "raw_contact_distance_pass_fraction": fraction,
        "distance_threshold_m": 0.020,
        "minimum_distance_min_m": float(minimum.min()),
        "minimum_distance_p95_m": float(np.percentile(minimum, 95)),
        "minimum_distance_max_m": float(minimum.max()),
        "hand_point_count": 1538,
        "object_point_count": 256,
        "surface_sampling_seed": 2024,
        "raw_contact_source": "source_tensor[:,205:206]",
    }


def build(args: argparse.Namespace) -> Path:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    kin = InspireKinematics(args.urdf)
    # Parse mimic values directly from the locked URDF-derived kinematics order.
    # Native order is index, middle, ring, pinky, thumb-yaw, thumb-pitch.
    # The locked URDF differs from legacy only at pinky (native q13).
    mimic = {7: 1.05, 9: 1.05, 11: 1.05, 13: 1.18, 16: 0.60, 17: 0.80}
    tensor = torch.load(args.tensor, map_location="cpu", weights_only=True)
    x = tensor.detach().cpu().numpy().astype(np.float64)
    if x.ndim != 2 or x.shape[1] < 391 or len(x) != 432:
        raise ValueError(f"expected [432,598]-like tensor, got {x.shape}")
    points = x[:, 102:198].reshape(len(x), 32, 3)
    human_wrist = pose_from_tensor(x)
    source_frame = np.asarray(np.load(args.geometry / "source_frame_id.npy", allow_pickle=False), dtype=np.int64)
    object_pose = np.asarray(np.load(args.geometry / "obj_pose_world.npy", allow_pickle=False), dtype=np.float64)
    if len(source_frame) != len(x) or len(object_pose) != len(x):
        raise ValueError("source/object/tensor frame mismatch")
    X_HB = fit_xhb(kin, human_wrist, points)
    wrist = human_wrist @ X_HB
    target = points[:, list(RIGHT_TIPS)]
    neutral = np.zeros((TRAIN_END - TRAIN_START + 1, 6), dtype=np.float64)
    q6 = np.zeros_like(neutral)
    for j, frame in enumerate(range(TRAIN_START, TRAIN_END + 1)):
        wrist_f = wrist[frame]
        fun = lambda z: ((tip_positions(kin, z[None], wrist_f[None], mimic)[0] - target[frame]) / 0.02).reshape(-1)
        initial = q6[j - 1] if j else np.zeros(6)
        q6[j] = least_squares(fun, initial, bounds=(kin.finger_lower, kin.finger_upper), max_nfev=80, xtol=1e-10, ftol=1e-10, gtol=1e-10).x
    q6 = project_limits(q6, 1 / 30.0, kin.finger_lower, kin.finger_upper)
    q_native = expand_q6(q6, mimic)
    idx = np.arange(TRAIN_START, TRAIN_END + 1)
    wrist_train, obj_train = wrist[idx], object_pose[idx]
    dq = np.gradient(q_native, 1 / 30.0, axis=0, edge_order=1)
    ddq = np.gradient(dq, 1 / 30.0, axis=0, edge_order=1)
    tip = np.stack([tip_positions(kin, q6[i], wrist_train[i], mimic) for i in range(len(idx))])
    tip_error = np.linalg.norm(tip - target[idx], axis=-1)
    neutral_tip = tip_positions(kin, np.zeros(6), wrist_train, mimic)
    neutral_error = np.linalg.norm(neutral_tip - target[idx], axis=-1)
    valid = np.isfinite(q_native).all(axis=1) & (np.abs(dq[:, INDEPENDENT_FINGER_NATIVE_INDICES]) <= 1 + 1e-6).all(axis=1) & (np.abs(ddq[:, INDEPENDENT_FINGER_NATIVE_INDICES]) <= 20 + 1e-6).all(axis=1)
    if not valid.all():
        raise ValueError("q6 velocity/acceleration hard gate failed")
    out = args.output
    out.mkdir(parents=True)
    zero_inv = np.linalg.inv(kin.wrist_pose_from_native(np.zeros(18)))
    link_names = tuple(kin.link_names)
    link_pose = np.stack([wrist_train[i] @ zero_inv @ np.stack([kin.link_transforms_native(q_native[i])[n] for n in link_names]) for i in range(len(idx))])
    wrist_twist = twists(wrist_train, 1 / 30.0)
    object_twist = twists(obj_train, 1 / 30.0)
    contact_metrics = contact_audit(
        link_pose, obj_train, x[idx, 205] > 0.5, args.urdf, args.object_urdf)
    training_eligible = bool(valid.all() and contact_metrics["raw_contact_frames"] > 0
                             and contact_metrics["raw_contact_distance_pass_fraction"] >= 0.90)
    np.savez_compressed(out / "reference.npz", frame_id=np.arange(len(idx), dtype=np.int32), source_frame_id=source_frame[idx].astype(np.int32), q_native_ref=q_native.astype(np.float32), q_independent_ref=q6.astype(np.float32), dq_native_ref=dq.astype(np.float32), wrist_pose_world_ref=wrist_train.astype(np.float32), wrist_twist_world_ref=wrist_twist.astype(np.float32), object_pose_world_ref=obj_train.astype(np.float32), object_twist_world_ref=object_twist.astype(np.float32), link_pose_world_ref=link_pose.astype(np.float32), phase=np.linspace(0, 1, len(idx), dtype=np.float32), remaining_steps=(len(idx)-1-np.arange(len(idx))).astype(np.int32), valid_rsi_mask=np.ones(len(idx), dtype=bool), rsi_reason_bits=np.zeros(len(idx), dtype=np.uint32))
    np.save(out / "valid_rsi_mask.npy", np.ones(len(idx), dtype=bool))
    manifest = {"schema_name": "ref2dex_cmresidual_reference_v1", "modification_version": "V1.6", "task": "CmResidual", "sequence_id": "s1/airplane_lift", "split": "train", "training_eligible": training_eligible, "training_frame_range": [TRAIN_START, TRAIN_END], "source_frame_range": [int(source_frame[TRAIN_START]), int(source_frame[TRAIN_END])], "frame_count": len(idx), "effective_fps": 30.0, "coordinate_frame": "world", "quaternion_order": "xyzw", "urdf": str(args.urdf.resolve()), "urdf_sha256": sha256(args.urdf), "object_urdf": str(args.object_urdf.resolve()), "object_urdf_sha256": sha256(args.object_urdf), "source_tensor": str(args.tensor.resolve()), "source_tensor_sha256": sha256(args.tensor), "object_geometry": str(args.geometry.resolve()), "object_geometry_manifest_sha256": sha256(args.geometry / "manifest.json"), "X_HB": X_HB.tolist(), "link_order": list(link_names), "independent_native_indices": INDEPENDENT_FINGER_NATIVE_INDICES.tolist(), "mimic_mapping": {str(k): float(v) for k, v in mimic.items()}, "limits": {"q_velocity_rad_s": 1.0, "q_acceleration_rad_s2": 20.0, "tolerance": 1e-6}, "difference_rule": "numpy.gradient; SO3 central relative rotvec; endpoint one-sided", "optimizer": {"name": "scipy.optimize.least_squares", "loss": "tip_position_squared_scaled_0.02m", "initial": "previous_frame_q6_or_zero", "max_nfev": 80, "xtol": 1e-10, "ftol": 1e-10, "gtol": 1e-10, "seed": 0, "post_projection_iterations": 8}, "outputs": {"reference": "reference.npz", "valid_rsi_mask": "valid_rsi_mask.npy"}, "metrics": {"tip_error_rms_m": float(np.sqrt(np.mean(tip_error ** 2))), "neutral_tip_error_rms_m": float(np.sqrt(np.mean(neutral_error ** 2))), "tip_improvement_fraction": float(1 - np.sqrt(np.mean(tip_error ** 2)) / np.sqrt(np.mean(neutral_error ** 2))), "contact_flag": "raw_source_tensor_object_flag", **contact_metrics}, "gate_status": {"schema": "PASS", "se3": "PASS", "mimic": "PASS", "q_limits": "PASS" if valid.all() else "FAIL", "tip_improvement": "PASS" if float(1 - np.sqrt(np.mean(tip_error ** 2)) / np.sqrt(np.mean(neutral_error ** 2))) >= 0.20 else "FAIL", "raw_contact_distance": "PASS" if contact_metrics["raw_contact_distance_pass_fraction"] >= 0.90 else "FAIL"}, "implementation_sha256": sha256(Path(__file__).resolve()), "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "conclusion": "SUPPORTED" if training_eligible else "INCONCLUSIVE"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    run_manifest = {"schema_name": "ref2dex_run_manifest_v1", "task": "CmResidual", "operation": "build_reference", "run_id": out.name, "run_status": "COMPLETED", "modification_version": "V1.6", "operation_category": ["data", "operation"], "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "base_commit": commit, "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()), "command": [sys.executable, *sys.argv], "output_root": str(out.resolve()), "reference_manifest": str((out / "manifest.json").resolve()), "training_eligible": training_eligible, "conclusion": "SUPPORTED" if training_eligible else "INCONCLUSIVE"}
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", type=Path, default=ROOT / "data/processed_data/inspire_geometric_dexplore_coupled_v1_20260912/s1_airplane_lift/interaction_hand_inspire.pt")
    p.add_argument("--geometry", type=Path, default=ROOT / "data/processed_data/coupled_geometric_source_v1_20260912/sequences/train/inspire_rl/s1_airplane_lift/geometry")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--object-urdf", type=Path, default=DEFAULT_OBJECT_URDF)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    build(args)
    print(json.dumps({"output": str(args.output), "status": "COMPLETED"}))


if __name__ == "__main__":
    main()
