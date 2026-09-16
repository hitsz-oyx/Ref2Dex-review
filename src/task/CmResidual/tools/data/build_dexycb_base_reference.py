"""Build an isolated DexYCB-to-Inspire reference for frozen DExplore evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.optimize import Bounds, LinearConstraint, least_squares, minimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.CmDecoderv2.kinematics import (
    INDEPENDENT_FINGER_NATIVE_INDICES,
    InspireKinematics,
    expand_finger_q,
)


SEQUENCE = "subject-10/20201022_110806"
OBJECT_NAME = "002_master_chef_can"
FPS = 30.0
DT = 1.0 / FPS
TIP_LINKS = ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip")
KEY_LINKS = (
    "hand_base_link", "index_proximal", "index_intermediate", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_tip",
    "pinky_proximal", "pinky_intermediate", "pinky_tip",
    "ring_proximal", "ring_intermediate", "ring_tip",
    "thumb_proximal_base", "thumb_intermediate", "thumb_tip",
)
DEFAULT_SHARED = ROOT / "data/processed_data/stage4/data/dexycb" / SEQUENCE / "shared.npz"
DEFAULT_RIGHT = ROOT / "data/processed_data/stage4/data/dexycb" / SEQUENCE / "right.npz"
DEFAULT_RAW = ROOT / "data/raw_data/DexYCB/raw/subject-10/20201022-subject-10/20201022_110806"
DEFAULT_MODEL = ROOT / "data/raw_data/DexYCB/raw/models" / OBJECT_NAME
DEFAULT_URDF = ROOT / "src/task/CmDecoderv2/assets/inspire_hand_new/inspire_hand_right.urdf"
DEFAULT_OUTPUT = ROOT / "data/processed_data/cm_residual/dexycb_base_v1" / SEQUENCE


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def transform_poses(left: np.ndarray, poses: np.ndarray) -> np.ndarray:
    return np.matmul(left, poses)


def transform_points(left: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("ij,tpj->tpi", left[:3, :3], points) + left[:3, 3]


def select_canonical_tips(canonical: np.ndarray, finger_id: np.ndarray,
                          region_id: np.ndarray | None = None) -> np.ndarray:
    """Select one stable farthest-from-wrist point for thumb through pinky."""
    selected = []
    for value in range(1, 6):
        mask = finger_id == value
        if region_id is not None:
            mask &= region_id == 1
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise ValueError(f"No canonical hand points for finger_id={value}")
        selected.append(int(candidates[np.linalg.norm(canonical[candidates], axis=1).argmax()]))
    return np.asarray(selected, dtype=np.int64)


def rigid_fit(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return target_from_source rigid transform without scale."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = target_center - rotation @ source_center
    return result


def wrist_native_from_pose(kinematics: InspireKinematics, hand_base_pose: np.ndarray) -> np.ndarray:
    fixed = kinematics.wrist_pose_from_native(np.zeros(18, dtype=np.float64))
    link6_pose = hand_base_pose @ np.linalg.inv(fixed)
    target_rotation = link6_pose[:3, :3]
    initial = Rotation.from_matrix(target_rotation).as_rotvec()

    def residual(angles: np.ndarray) -> np.ndarray:
        candidate = Rotation.from_euler("XYZ", angles).as_matrix()
        return Rotation.from_matrix(candidate.T @ target_rotation).as_rotvec()

    angles = least_squares(residual, initial, max_nfev=80, xtol=1e-12, ftol=1e-12, gtol=1e-12).x
    native = np.concatenate((link6_pose[:3, 3], angles))
    probe = np.zeros(18, dtype=np.float64)
    probe[:6] = native
    error = np.max(np.abs(kinematics.wrist_pose_from_native(probe) - hand_base_pose))
    if error > 1e-6:
        raise ValueError(f"Wrist pose decomposition error {error:.3e}")
    return native


def project_finger_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    target = np.clip(np.asarray(q, dtype=np.float64), lower, upper)
    q = target.copy()
    velocity_step = 1.0 * DT
    acceleration_step = 20.0 * DT * DT
    for _ in range(8):
        delta = np.clip(np.diff(q, axis=0), -velocity_step, velocity_step)
        q[1:] = q[0] + np.cumsum(delta, axis=0)
        q = np.clip(q, lower, upper)
        delta = np.diff(q, axis=0)
        if len(delta) > 1:
            second = np.clip(np.diff(delta, axis=0), -acceleration_step, acceleration_step)
            delta[1:] = delta[0] + np.cumsum(second, axis=0)
            q[1:] = q[0] + np.cumsum(delta, axis=0)
            q = np.clip(q, lower, upper)
    frames = len(q)
    first = np.zeros((frames - 1, frames), dtype=np.float64)
    rows = np.arange(frames - 1)
    first[rows, rows] = -1.0
    first[rows, rows + 1] = 1.0
    second = first[1:] - first[:-1]
    constraints = (
        LinearConstraint(first, -velocity_step, velocity_step),
        LinearConstraint(second, -acceleration_step, acceleration_step),
    )
    for joint in range(q.shape[1]):
        goal = target[:, joint].copy()
        result = minimize(
            lambda value: 0.5 * np.square(value - goal).sum(), q[:, joint],
            jac=lambda value: value - goal,
            bounds=Bounds(np.full(frames, lower[joint]), np.full(frames, upper[joint])),
            constraints=constraints, method="SLSQP",
            options={"maxiter": 400, "ftol": 1e-12, "disp": False},
        )
        if not result.success:
            raise ValueError(f"Constrained q projection failed for joint {joint}: {result.message}")
        q[:, joint] = result.x
    return q


def pose_twists(poses: np.ndarray) -> np.ndarray:
    result = np.zeros((len(poses), 6), dtype=np.float64)
    result[:, :3] = np.gradient(poses[:, :3, 3], DT, axis=0, edge_order=1)
    for index in range(len(poses)):
        previous, following = max(index - 1, 0), min(index + 1, len(poses) - 1)
        if following != previous:
            relative = poses[previous, :3, :3].T @ poses[following, :3, :3]
            result[index, 3:] = Rotation.from_matrix(relative).as_rotvec() / ((following - previous) * DT)
    return result


def create_assets(output: Path, model_root: Path) -> tuple[Path, Path]:
    asset_root = output / "assets"
    object_dir = asset_root / "objects/master_chef_can"
    object_dir.mkdir(parents=True)
    source_obj = model_root / "textured_simple.obj"
    object_obj = object_dir / "master_chef_can.obj"
    shutil.copy2(source_obj, object_obj)
    for name in ("textured_simple.obj.mtl", "texture_map.png"):
        source = model_root / name
        if source.is_file():
            shutil.copy2(source, object_dir / name)
    object_urdf = asset_root / "master_chef_can.urdf"
    object_urdf.write_text(
        """<?xml version="1.0"?>
<robot name="master_chef_can">
  <link name="object">
    <inertial><origin xyz="0 0 0"/><mass value="0.414"/><inertia ixx="0.0009" ixy="0" ixz="0" iyy="0.0009" iyz="0" izz="0.0006"/></inertial>
    <visual><geometry><mesh filename="objects/master_chef_can/master_chef_can.obj"/></geometry></visual>
    <collision><geometry><mesh filename="objects/master_chef_can/master_chef_can.obj"/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    table_urdf = asset_root / "table.urdf"
    table_urdf.write_text(
        """<?xml version="1.0"?>
<robot name="table">
  <link name="table">
    <inertial><origin xyz="0 0 0"/><mass value="1000"/><inertia ixx="100" ixy="0" ixz="0" iyy="100" iyz="0" izz="100"/></inertial>
    <visual><geometry><box size="1.0 1.0 0.05"/></geometry></visual>
    <collision><geometry><box size="1.0 1.0 0.05"/></geometry></collision>
  </link>
</robot>
""",
        encoding="utf-8",
    )
    return asset_root, object_obj


def build(args: argparse.Namespace) -> Path:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    for path in (args.shared, args.right, args.raw / "pose.npz", args.raw / "meta.yml", args.urdf):
        if not path.is_file():
            raise FileNotFoundError(path)

    with np.load(args.shared, allow_pickle=False) as values:
        shared = {key: values[key].copy() for key in values.files}
    with np.load(args.right, allow_pickle=False) as values:
        right = {key: values[key].copy() for key in values.files}
    with np.load(args.raw / "pose.npz", allow_pickle=False) as values:
        raw_pose = values["pose_y"].astype(np.float64)
    metadata = yaml.safe_load((args.raw / "meta.yml").read_text(encoding="utf-8"))

    frame_count = len(shared["raw_frame_id"])
    if str(shared["seq_id"].item()) != SEQUENCE or str(shared["object_name"].item()) != OBJECT_NAME:
        raise ValueError("Input sequence/object identity differs from the approved V1.11 contract")
    if str(right["side"].item()) != "right" or frame_count != 72 or len(right["hand_root_pose_world"]) != frame_count:
        raise ValueError("Expected the approved 72-frame right-hand stream")
    grasp_index = int(metadata["ycb_grasp_ind"])
    object_index = int(metadata["ycb_ids"][grasp_index])
    if object_index != 1:
        raise ValueError(f"Expected YCB id 1, got {object_index}")

    object_raw = np.repeat(np.eye(4)[None], frame_count, axis=0)
    object_raw[:, :3, :3] = Rotation.from_quat(raw_pose[:, grasp_index, :4]).as_matrix()
    object_raw[:, :3, 3] = raw_pose[:, grasp_index, 4:7]
    import trimesh
    mesh = trimesh.load(str(args.model / "textured_simple.obj"), force="mesh")
    desired_object0 = np.eye(4, dtype=np.float64)
    desired_object0[2, 3] = -float(mesh.bounds[0, 2]) + 0.002
    sim_from_dexycb = desired_object0 @ np.linalg.inv(object_raw[0])
    object_pose = transform_poses(sim_from_dexycb, object_raw)
    hand_root = transform_poses(sim_from_dexycb, right["hand_root_pose_world"].astype(np.float64))
    hand_points = transform_points(sim_from_dexycb, right["hand_points_world"].astype(np.float64))
    object_points = transform_points(sim_from_dexycb, shared["obj_points_world"].astype(np.float64))

    tip_indices = select_canonical_tips(
        right["hand_cano_points"], right["hand_finger_id"], right["hand_region_id"])
    target_tips = hand_points[:, tip_indices]
    kinematics = InspireKinematics(args.urdf)
    neutral_links = kinematics.link_transforms_from_state(np.zeros(6), np.eye(4))
    neutral_tips = np.stack([neutral_links[name][:3, 3] for name in TIP_LINKS])
    human_tips_local = np.median(
        np.einsum("tij,tpj->tpi", np.linalg.inv(hand_root)[:, :3, :3],
                  target_tips - hand_root[:, None, :3, 3]), axis=0)
    human_from_robot = rigid_fit(neutral_tips, human_tips_local)
    wrist_pose = transform_poses(hand_root, np.repeat(human_from_robot[None], frame_count, axis=0))

    q6 = np.zeros((frame_count, 6), dtype=np.float64)
    for index in range(frame_count):
        initial = q6[index - 1] if index else np.zeros(6)

        def residual(value: np.ndarray) -> np.ndarray:
            links = kinematics.link_transforms_from_state(value, wrist_pose[index])
            tips = np.stack([links[name][:3, 3] for name in TIP_LINKS])
            return ((tips - target_tips[index]) / 0.02).reshape(-1)

        q6[index] = least_squares(
            residual, initial, bounds=(kinematics.finger_lower, kinematics.finger_upper),
            max_nfev=100, xtol=1e-10, ftol=1e-10, gtol=1e-10).x
    raw_q6 = q6.copy()
    q6 = project_finger_limits(q6, kinematics.finger_lower, kinematics.finger_upper)
    wrist_native = np.stack([wrist_native_from_pose(kinematics, pose) for pose in wrist_pose])
    native = expand_finger_q(q6, wrist_native)
    dq = np.gradient(native, DT, axis=0, edge_order=1)
    ddq = np.gradient(dq, DT, axis=0, edge_order=1)

    link_names = tuple(kinematics.link_names)
    link_pose = np.stack([
        np.stack([kinematics.link_transforms_native(row)[name] for name in link_names])
        for row in native
    ])
    name_to_index = {name: index for index, name in enumerate(link_names)}
    key_pose = np.take(link_pose, [name_to_index[name] for name in KEY_LINKS], axis=1)
    actual_tips = np.take(link_pose, [name_to_index[name] for name in TIP_LINKS], axis=1)[:, :, :3, 3]
    tip_error = np.linalg.norm(actual_tips - target_tips, axis=-1)
    neutral_error = []
    raw_tip_error = []
    for index in range(frame_count):
        links = kinematics.link_transforms_from_state(np.zeros(6), wrist_pose[index])
        neutral_error.append(np.linalg.norm(
            np.stack([links[name][:3, 3] for name in TIP_LINKS]) - target_tips[index], axis=-1))
        raw_links = kinematics.link_transforms_from_state(raw_q6[index], wrist_pose[index])
        raw_tip_error.append(np.linalg.norm(
            np.stack([raw_links[name][:3, 3] for name in TIP_LINKS]) - target_tips[index], axis=-1))
    neutral_error = np.asarray(neutral_error)
    raw_tip_error = np.asarray(raw_tip_error)
    improvement = 1.0 - float(np.sqrt(np.mean(tip_error ** 2)) / np.sqrt(np.mean(neutral_error ** 2)))
    raw_improvement = 1.0 - float(np.sqrt(np.mean(raw_tip_error ** 2)) / np.sqrt(np.mean(neutral_error ** 2)))

    distances = np.linalg.norm(
        key_pose[:, :, None, :3, 3] - object_points[:, None, :, :], axis=-1).min(axis=-1)
    contact_parts = distances <= 0.020
    source = np.zeros((frame_count, 598), dtype=np.float32)
    tracking_pose = link_pose[:, name_to_index["link6"]]
    source[:, 51:54] = tracking_pose[:, :3, 3]
    source[:, 54:102] = Rotation.from_matrix(key_pose[:, :, :3, :3].reshape(-1, 3, 3)).as_rotvec().reshape(frame_count, 48)
    source[:, 102:198] = np.pad(key_pose[:, :, :3, 3], ((0, 0), (16, 0), (0, 0))).reshape(frame_count, 96)
    source[:, 198:201] = object_pose[:, :3, 3]
    source[:, 201:205] = Rotation.from_matrix(object_pose[:, :3, :3]).as_quat()
    source[:, 205] = contact_parts.any(axis=1)
    source[:, 222:238] = contact_parts.astype(np.float32)
    source[:, 238:245] = np.asarray([0.0, 0.0, -0.025, 0.0, 0.0, 0.0, 1.0])
    source[:, 309:373] = Rotation.from_matrix(key_pose[:, :, :3, :3].reshape(-1, 3, 3)).as_quat().reshape(frame_count, 64)
    source[:, 373:391] = native
    padded = np.concatenate((source[:1], source, source[-1:]), axis=0)

    finger_dq = dq[:, INDEPENDENT_FINGER_NATIVE_INDICES]
    finger_ddq = ddq[:, INDEPENDENT_FINGER_NATIVE_INDICES]
    gates = {
        "finite": bool(np.isfinite(source).all() and np.isfinite(link_pose).all()),
        "joint_limits": bool(((q6 >= kinematics.finger_lower - 1e-8) &
                              (q6 <= kinematics.finger_upper + 1e-8)).all()),
        "finger_velocity": bool((np.abs(finger_dq) <= 1.0 + 1e-5).all()),
        "finger_acceleration": bool((np.abs(finger_ddq) <= 20.0 + 1e-4).all()),
        "tip_improvement": improvement >= 0.20,
        "object_initial_pose": bool(np.max(np.abs(object_pose[0] - desired_object0)) <= 1e-6),
    }
    if not all(gates.values()):
        raise ValueError(
            f"DexYCB reference gate failed: {gates}; improvement={improvement:.6f}; "
            f"raw_improvement={raw_improvement:.6f}; "
            f"raw_vmax={np.abs(np.gradient(raw_q6, DT, axis=0)).max():.6f}")

    output.mkdir(parents=True)
    asset_root, object_mesh = create_assets(output, args.model)
    torch.save(torch.from_numpy(padded), output / "interaction_hand_inspire.pt")
    np.savez_compressed(
        output / "reference.npz",
        frame_id=np.arange(frame_count, dtype=np.int32),
        source_frame_id=shared["raw_frame_id"].astype(np.int32),
        q_native_ref=native.astype(np.float32),
        q_independent_ref=q6.astype(np.float32),
        dq_native_ref=dq.astype(np.float32),
        wrist_pose_world_ref=wrist_pose.astype(np.float32),
        wrist_twist_world_ref=pose_twists(wrist_pose).astype(np.float32),
        object_pose_world_ref=object_pose.astype(np.float32),
        object_twist_world_ref=pose_twists(object_pose).astype(np.float32),
        link_pose_world_ref=link_pose.astype(np.float32),
        phase=np.linspace(0, 1, frame_count, dtype=np.float32),
        remaining_steps=(frame_count - 1 - np.arange(frame_count)).astype(np.int32),
        valid_rsi_mask=np.ones(frame_count, dtype=bool),
        rsi_reason_bits=np.zeros(frame_count, dtype=np.uint32),
    )
    np.save(output / "valid_rsi_mask.npy", np.ones(frame_count, dtype=bool))
    manifest = {
        "schema_name": "ref2dex_cmresidual_dexycb_base_reference_v1",
        "modification_version": "V1.11.1",
        "task": "CmResidual",
        "sequence_id": SEQUENCE,
        "side": "right",
        "object_name": OBJECT_NAME,
        "split": "evaluation",
        "training_eligible": False,
        "evaluation_eligible": True,
        "coordinate_frame": "world",
        "quaternion_order": "xyzw",
        "frame_count": frame_count,
        "training_frame_range": [1, frame_count],
        "effective_fps": FPS,
        "source_tensor_semantics": "synthetic_dexplore_compatibility_tensor_not_original_dexplore_data",
        "zero_filled_source_ranges": [[0, 50], [391, 597]],
        "canonical_tip_indices": tip_indices.tolist(),
        "canonical_tip_point_ids": right["hand_point_id"][tip_indices].astype(int).tolist(),
        "finger_id_order": [1, 2, 3, 4, 5],
        "tip_candidate_region_id": 1,
        "tip_link_order": list(TIP_LINKS),
        "X_sim_from_dexycb": sim_from_dexycb.tolist(),
        "X_human_from_robot_base": human_from_robot.tolist(),
        "inputs": {
            "shared": {"path": str(args.shared.resolve()), "sha256": sha256(args.shared)},
            "right": {"path": str(args.right.resolve()), "sha256": sha256(args.right)},
            "raw_pose": {"path": str((args.raw / 'pose.npz').resolve()), "sha256": sha256(args.raw / "pose.npz")},
            "mesh": {"path": str((args.model / 'textured_simple.obj').resolve()), "sha256": sha256(args.model / "textured_simple.obj")},
            "urdf": {"path": str(args.urdf.resolve()), "sha256": sha256(args.urdf)},
        },
        "outputs": {
            "reference": "reference.npz",
            "source_tensor": "interaction_hand_inspire.pt",
            "asset_root": "assets",
            "object_mesh": str(object_mesh.relative_to(output)),
        },
        "metrics": {
            "tip_error_rms_m": float(np.sqrt(np.mean(tip_error ** 2))),
            "neutral_tip_error_rms_m": float(np.sqrt(np.mean(neutral_error ** 2))),
            "tip_improvement_fraction": improvement,
            "raw_optimizer_tip_improvement_fraction": raw_improvement,
            "projection_q_rms_rad": float(np.sqrt(np.mean((raw_q6 - q6) ** 2))),
            "geometric_contact_frame_fraction": float(contact_parts.any(axis=1).mean()),
            "geometric_contact_part_fraction": float(contact_parts.mean()),
            "max_finger_velocity_rad_s": float(np.abs(finger_dq).max()),
            "max_finger_acceleration_rad_s2": float(np.abs(finger_ddq).max()),
        },
        "gate_status": {name: "PASS" if passed else "FAIL" for name, passed in gates.items()},
        "created_at": now(),
        "conclusion": "SUPPORTED",
    }
    write_json(output / "manifest.json", manifest)
    base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    write_json(output / "run_manifest.json", {
        "manifest_schema": "ref2dex.run.v1",
        "run_id": output.name,
        "run_status": "COMPLETED",
        "task": "CmResidual",
        "mode": "build_dexycb_base_reference",
        "modification_version": "V1.11.1",
        "operation_category": ["data", "operation"],
        "created_at": now(),
        "base_commit": base_commit,
        "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "command": [sys.executable, *sys.argv],
        "output_dir": str(output),
        "reference_manifest": str((output / "manifest.json").resolve()),
        "last_step": frame_count,
        "best_metric": {"name": "tip_error_rms_m", "value": manifest["metrics"]["tip_error_rms_m"]},
        "checkpoint": None,
        "exit_reason": "Completed approved offline DexYCB reference build",
        "conclusion": "SUPPORTED",
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared", type=Path, default=DEFAULT_SHARED)
    parser.add_argument("--right", type=Path, default=DEFAULT_RIGHT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    output = build(parse_args())
    print(json.dumps({"output": str(output), "run_status": "COMPLETED"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
