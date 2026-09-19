"""OakInk2 official MANO reconstruction and bilateral Inspire geometric pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


REPO = Path(__file__).resolve().parents[5]
VERSION = "V1.4.1"
TIPS = np.array([4, 8, 12, 16, 20])


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def file_info(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def require_finite(name, value):
    if not np.isfinite(value).all():
        raise ValueError(f"{name}: non-finite values")


def select_frame_ids(raw_mano, limit):
    frames = np.asarray(sorted(raw_mano), dtype=np.int64)
    if not len(frames):
        raise ValueError("Empty MANO sequence")
    if limit > 0 and len(frames) > limit:
        frames = frames[np.linspace(0, len(frames) - 1, limit, dtype=np.int64)]
    return frames


def gather_params(raw_mano, frame_ids, prefix):
    values = {}
    for field, shape in (("pose_coeffs", (16, 4)), ("betas", (10,)), ("tsl", (3,))):
        key = prefix + "__" + field
        arrays = []
        for frame in frame_ids:
            array = np.asarray(raw_mano[int(frame)][key], dtype=np.float32)
            if array.shape != (1,) + shape:
                raise ValueError(f"frame={frame} {key}: {array.shape} != {(1,) + shape}")
            require_finite(key, array)
            arrays.append(array)
        values[field] = torch.from_numpy(np.concatenate(arrays))
    norms = np.linalg.norm(values["pose_coeffs"].numpy(), axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"{prefix}: quaternion norm outside tolerance")
    return values


def rebuild_mano(layer, params, reference_layer):
    vertices, joints = [], []
    with torch.no_grad():
        for start in range(0, len(params["tsl"]), 64):
            batch = {k: v[start:start + 64] for k, v in params.items()}
            output = layer(pose_coeffs=batch["pose_coeffs"], betas=batch["betas"])
            vertices.append((output.verts + batch["tsl"][:, None]).numpy())
            joints.append((output.joints + batch["tsl"][:, None]).numpy())
        vertices, joints = np.concatenate(vertices), np.concatenate(joints)
        # Independent implementation check: equivalent SMPL-X MANO must agree
        # after wrist centering, without adding the MANO pose mean a second time.
        q = params["pose_coeffs"][:3].numpy()
        aa = Rotation.from_quat(q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_rotvec()
        aa = torch.from_numpy(aa.astype(np.float32).reshape(-1, 48))
        reference = reference_layer(global_orient=aa[:, :3], hand_pose=aa[:, 3:],
                                    betas=params["betas"][:3])
        ref_vertices = (reference.vertices - reference.joints[:, :1] + params["tsl"][:3, None]).numpy()
    require_finite("MANO vertices", vertices)
    require_finite("MANO joints", joints)
    error = float(np.linalg.norm(vertices[:3] - ref_vertices, axis=-1).max())
    wrist_error = float(np.abs(joints[:, 0] - params["tsl"].numpy()).max())
    if error > 1e-5 or wrist_error > 1e-6:
        raise ValueError(f"MANO reconstruction mismatch: vertex={error} wrist={wrist_error} m")
    return vertices, joints, {"smplx_vertex_max_error_m": error, "wrist_max_error_m": wrist_error}


def retarget(joints, side, dex_root):
    from dex_retargeting.retargeting_config import RetargetingConfig
    config_path = dex_root / "dex_retargeting/configs/offline" / f"inspire_hand_{side}.yml"
    urdf_dir = dex_root / "assets/robots/hands"
    RetargetingConfig.set_default_urdf_dir(urdf_dir)
    retargeter = RetargetingConfig.load_from_file(config_path).build()
    robot = retargeter.optimizer.robot
    qpos, transforms = [], []
    # Keep native world, official objective, temporal state and initialization.
    # No hand-only world rotations or optimization-error-based frame filtering.
    for targets in joints[:, TIPS]:
        value = retargeter.retarget(targets)
        robot.compute_forward_kinematics(value)
        qpos.append(value)
        transforms.append([robot.get_link_pose(i) for i in range(len(robot.link_names))])
    qpos = np.asarray(qpos, dtype=np.float32)
    transforms = np.asarray(transforms, dtype=np.float32)
    predicted = transforms[:, :, :3, 3][:, retargeter.optimizer.target_link_indices]
    errors = np.linalg.norm(predicted - joints[:, TIPS], axis=-1)
    require_finite("Inspire qpos", qpos)
    require_finite("Inspire FK", transforms)
    report = {
        "qpos_shape": list(qpos.shape), "finite": True,
        "tip_mean_error_mm": float(errors.mean() * 1000),
        "tip_p95_error_mm": float(np.percentile(errors, 95) * 1000),
        "tip_max_error_mm": float(errors.max() * 1000),
        "frame_mean_over_20mm": int((errors.mean(axis=-1) > .02).sum()),
        "config": file_info(config_path),
        "urdf": file_info(urdf_dir / f"inspire_hand/inspire_hand_{side}.urdf"),
    }
    arrays = {"inspire_qpos": qpos, "inspire_link_poses_world": transforms,
              "inspire_joint_names": np.asarray(retargeter.joint_names),
              "inspire_link_names": np.asarray(robot.link_names),
              "inspire_tips_world": predicted, "inspire_tip_error_m": errors}
    return arrays, report


def object_fields(anno, frames, object_root, seed):
    ids, poses, points, records = anno["obj_list"], [], [], []
    if not ids:
        raise ValueError("Sequence has no objects")
    rng = np.random.default_rng(seed)
    for obj_id in ids:
        # No missing-frame interpolation; all sampled hands and all objects
        # must refer to the same original mocap frame ID.
        value = np.stack([anno["obj_transf"][obj_id][int(fid)] for fid in frames])
        require_finite(obj_id + " poses", value)
        if value.shape != (len(frames), 4, 4):
            raise ValueError(f"Invalid object pose shape: {obj_id}")
        rotation = value[:, :3, :3]
        if (not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-6)
                or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
            raise ValueError(f"Invalid object SE(3): {obj_id}")
        mesh_path = object_root / "object_repair/align_ds" / obj_id / "model.obj"
        if not mesh_path.is_file():
            mesh_path = object_root / "object_raw/align_ds" / obj_id / "model.obj"
        mesh = trimesh.load(mesh_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        require_finite(obj_id + " mesh", vertices)
        indices = rng.choice(len(vertices), 256, replace=len(vertices) < 256)
        points.append(vertices[indices])
        poses.append(value)
        records.append({"object_id": obj_id, "mesh": file_info(mesh_path),
                        "mesh_vertex_ids": indices.tolist(), "mesh_extent_m": mesh.extents.tolist()})
    poses = np.stack(poses, axis=1)
    canonical = np.stack(points)
    world = np.einsum("toij,opj->topi", poses[:, :, :3, :3], canonical) + poses[:, :, None, :3, 3]
    return {"object_ids": np.asarray(ids), "object_poses_world": poses,
            "object_sample_vertices_canonical": canonical,
            "object_sample_vertices_world": world.astype(np.float32)}, records


def inventory(annotation_root, object_root):
    annotations = sorted(annotation_root.glob("*.pkl"))
    metadata = sorted((object_root / "program_extension/frame_id").glob("*.pkl"))
    counts = Counter()
    for path in metadata:
        with path.open("rb") as f:
            value = pickle.load(f)
        counts["metadata_mocap_frames"] += len(value["mocap_frame_id_list"])
        counts["metadata_rgb_frames"] += len(value["frame_id_list"])
    return {"annotation_files": len(annotations), "frame_metadata_files": len(metadata),
            **dict(counts),
            "missing_annotation_names": sorted({p.name for p in metadata} - {p.name for p in annotations}),
            "note": "总帧数来自配套帧索引；未全量重建或验证 627 份 pose 标注。"}


def execute(args, output):
    # Required by the legacy chumpy objects stored in licensed MANO pickles.
    for key, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                       "object": object, "unicode": str, "str": str}.items():
        if key not in np.__dict__:
            setattr(np, key, value)
    from manotorch.manolayer import ManoLayer
    from smplx import MANO

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    report = {"inventory": inventory(args.annotation_root, args.object_root), "sequences": [], "failures": []}
    paths = sorted(args.annotation_root.glob("*.pkl"))
    if args.sequence:
        paths = [args.annotation_root / name for name in args.sequence]
    elif paths:
        paths = [paths[i] for i in sorted({0, len(paths) // 2, len(paths) - 1})]
    if not paths:
        raise ValueError("No annotation files")
    # manotorch expects a models/ child; staging symlink avoids altering NAS.
    with tempfile.TemporaryDirectory(prefix="oakink2_mano_") as staging:
        (Path(staging) / "models").symlink_to(args.mano_root.resolve(), target_is_directory=True)
        layers = {side: ManoLayer(mano_assets_root=staging, rot_mode="quat", side=side,
                                 center_idx=0, use_pca=False, flat_hand_mean=True).eval()
                  for side in ("left", "right")}
        references = {side: MANO(str(args.mano_root), is_rhand=(side == "right"),
                                use_pca=False, flat_hand_mean=True).eval() for side in layers}
        for path in paths:
            print(f"Loading {path.name}", flush=True)
            try:
                with path.open("rb") as f:
                    anno = pickle.load(f)
                frames = select_frame_ids(anno["raw_mano"], args.frames)
                if not set(frames).issubset(anno["mocap_frame_id_list"]):
                    raise ValueError("Selected frames missing from mocap frame ID list")
                arrays, objects = object_fields(anno, frames, args.object_root, args.seed)
                arrays["source_frame_id"] = frames
                record = {"input": file_info(path), "source_frames": len(anno["raw_mano"]),
                          "exported_frames": len(frames), "objects": objects, "hands": {}}
                for side, prefix in (("left", "lh"), ("right", "rh")):
                    params = gather_params(anno["raw_mano"], frames, prefix)
                    vertices, joints, reconstruction = rebuild_mano(layers[side], params, references[side])
                    converted, stats = retarget(joints, side, args.dex_root)
                    arrays.update({side + "_" + key: value for key, value in converted.items()})
                    arrays[side + "_mano_vertices_world"] = vertices
                    arrays[side + "_mano_joints_world"] = joints
                    arrays[side + "_mano_faces"] = layers[side].th_faces.numpy()
                    arrays.update({side + "_mano_" + key: value.numpy() for key, value in params.items()})
                    record["hands"][side] = {**stats, "reconstruction": reconstruction}
                destination = output / (path.stem + ".npz")
                np.savez_compressed(destination, **arrays)
                with np.load(destination, allow_pickle=False) as saved:
                    if not np.array_equal(saved["source_frame_id"], frames):
                        raise AssertionError("NPZ frame ID roundtrip mismatch")
                    for key, value in arrays.items():
                        if not np.array_equal(saved[key], value):
                            raise AssertionError(f"NPZ roundtrip mismatch: {key}")
                record["output"] = file_info(destination)
                report["sequences"].append(record)
                print(json.dumps({"sequence": path.stem, "frames": len(frames),
                                  "tip_mean_mm": {s: h["tip_mean_error_mm"] for s, h in record["hands"].items()}},
                                 ensure_ascii=False), flush=True)
            except Exception as exc:
                traceback.print_exc()
                report["failures"].append({"input": str(path), "error": repr(exc)})
    report["engineering_conclusion"] = "SUPPORTED" if not report["failures"] else "INCONCLUSIVE"
    report["research_conclusion"] = "INCONCLUSIVE"
    report["limitations"] = ["指尖位置拟合未验证穿透、接触保持、手掌朝向或训练收益。",
                              "逐对象位姿均保存；尚未定义多物体 OI-Cm 样本或正式 split。",
                              "本次使用 dex-retargeting 官方 URDF；未将 native qpos 重排为 Dexplore 格式。"]
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--dex-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", help="Annotation filename; repeat for multiple sequences")
    parser.add_argument("--frames", type=int, default=64, help="Uniformly select this many frames; 0 keeps all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.frames < 0 or args.threads < 1:
        parser.error("frames >= 0 and threads >= 1 required")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(output / "config.json", config)
    metadata = {"schema": "oakink2_inspire_geometric_pilot_v1", "units": "m",
                "coordinates": "OakInk2 native world; hands and all objects unchanged",
                "mano_pose": "quaternion wxyz; flat_hand_mean=True; center_idx=0; wrist+tsl",
                "landmark_order": "manotorch wrist, then thumb/index/middle/ring/pinky, 4 joints each",
                "tip_joint_indices": TIPS.tolist(),
                "tip_vertex_ids": {"right": [745, 317, 444, 556, 673], "left": [745, 317, 445, 556, 673]},
                "frame_selection": "all or uniform original mocap IDs, no interpolation; see config and NPZ",
                "inspire_qpos": "native dex-retargeting order; see per-side joint_names arrays",
                "object_points": "256 fixed canonical mesh vertices per object; visualization only"}
    write_json(output / "metadata.json", metadata)
    packages = {}
    for name in ("torch", "numpy", "smplx", "manotorch", "chumpy", "dex-retargeting", "nlopt", "pin",
                 "sapien", "pyrender", "anytree"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    repositories = {}
    for name in ("manotorch", "dex-retargeting"):
        dist = importlib.metadata.distribution(name)
        source_root = Path(dist.locate_file(""))
        result = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        repositories[name] = {"path": str(source_root),
                              "commit": result.stdout.strip() if result.returncode == 0 else None}
    write_json(output / "environment.json", {"python": sys.executable, "versions": packages,
                                             "repositories": repositories})
    (output / "pip_freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    (output / "run_source.py").write_bytes(Path(__file__).read_bytes())
    manifest = {"task": "ObjectInteractionCm", "work_version": VERSION,
                "run_id": output.name, "started_at": now(),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
                "command": shlex.join([sys.executable] + sys.argv), "config": "config.json",
                "metadata_snapshot": "metadata.json", "environment": "environment.json", "seed": args.seed,
                "checkpoint": None, "output": str(output), "source": file_info(__file__), "log": "run.log",
                "data": {"annotation_root": str(args.annotation_root.resolve()),
                         "object_root": str(args.object_root.resolve()), "input_records": "report.json"},
                "mano_assets": [file_info(args.mano_root / f"MANO_{s}.pkl") for s in ("LEFT", "RIGHT")]}
    write_json(output / "run_manifest.json", manifest)
    try:
        report = execute(args, output)
        manifest["report"] = "report.json"
        manifest["completed_at"] = now()
        write_json(output / "run_manifest.json", manifest)
        if report["failures"]:
            raise SystemExit(1)
    except Exception:
        (output / "error.txt").write_text(traceback.format_exc())
        manifest["error"] = "error.txt"
        manifest["completed_at"] = now()
        write_json(output / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
