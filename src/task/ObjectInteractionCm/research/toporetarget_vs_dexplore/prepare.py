"""Reconstruct one shared GRAB input and replay Dexplore's position optimizer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import time
import shlex
import subprocess
import traceback

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from inspire_adapter import NATIVE_ORDER, VERSION, info, prepare_assets, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grab-root", type=Path, required=True)
    p.add_argument("--sequence", default="s1/airplane_lift")
    p.add_argument("--start", type=int, default=240)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--sides", nargs="+", choices=("left", "right"), default=["left", "right"])
    p.add_argument("--mano-root", type=Path, required=True)
    p.add_argument("--dexplore-assets", type=Path, required=True)
    p.add_argument("--left-assets", type=Path, help="Explicit alternative left asset with standard tip links")
    p.add_argument("--dex-root", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def execute(args):
    write_json(args.output / "prepare_config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    torch.set_num_threads(1)
    torch.manual_seed(42)
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                        "object": object, "unicode": str, "str": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    from smplx import MANO
    from dex_retargeting.retargeting_config import RetargetingConfig
    source = args.grab_root / "grab" / (args.sequence + ".npz")
    raw = {k: v.item() for k, v in np.load(source, allow_pickle=True).items()}
    frames = np.arange(args.start, args.start + args.frames)
    if not len(frames) or frames[-1] >= raw["n_frames"]:
        raise ValueError("Requested frames outside source")
    prepare_assets(args.dexplore_assets, args.assets, sides=args.sides, left_assets=args.left_assets)
    report = {"modification_version": VERSION, "source": info(source), "sequence": args.sequence,
              "fps": float(raw["framerate"]), "source_frame_ids": frames.tolist(), "sides": {}}
    arrays = {"source_frame_id": frames, "timestamps": frames / raw["framerate"]}
    for side, key in (("left", "lhand"), ("right", "rhand")):
        if side not in args.sides:
            continue
        data = raw[key]
        vtemp_path = args.grab_root / data["vtemp"]
        vtemp = np.asarray(trimesh.load(vtemp_path, process=False).vertices, dtype=np.float32)
        model = MANO(str(args.mano_root), is_rhand=side == "right", v_template=vtemp,
                     use_pca=False, flat_hand_mean=True, batch_size=len(frames))
        params = {k: torch.from_numpy(v[frames]).float() for k, v in data["params"].items()}
        with torch.no_grad():
            result = model(global_orient=params["global_orient"], hand_pose=params["fullpose"], transl=params["transl"])
        vertices, joints = result.vertices.numpy(), result.joints.numpy()
        # Official GRAB uses flat_hand_mean=True; saved fullpose is decoded PCA.
        pca = MANO(str(args.mano_root), is_rhand=side == "right", v_template=vtemp,
                   use_pca=True, num_pca_comps=raw["n_comps"], flat_hand_mean=True, batch_size=len(frames))
        with torch.no_grad():
            reference = pca(global_orient=params["global_orient"], hand_pose=params["hand_pose"], transl=params["transl"])
        reconstruction_error = float(np.max(np.abs(vertices-reference.vertices.numpy())))
        if reconstruction_error > 1e-5:
            raise ValueError(f"{side}: GRAB fullpose/PCA disagreement {reconstruction_error}")
        landmarks = np.concatenate([joints[:, :1]] + [np.concatenate((joints[:, ids], vertices[:, [tip]]), axis=1)
                                   for ids, tip in [([13,14,15],744),([1,2,3],320),([4,5,6],443),
                                                    ([10,11,12],554),([7,8,9],671)]], axis=1)
        full_urdf = args.assets.resolve() / f"inspire_{side}_full.urdf"
        config = args.dex_root / "dex_retargeting/configs/offline" / f"inspire_hand_{side}.yml"
        retargeter = RetargetingConfig.load_from_file(config, override={
            "urdf_path": str(full_urdf), "target_joint_names": NATIVE_ORDER,
            "add_dummy_free_joint": False, "ignore_mimic_joint": True}).build()
        robot = retargeter.optimizer.robot
        qpos, fk, timings = [], [], []
        for points in landmarks[:, [4,8,12,16,20]]:
            t = time.perf_counter()
            value = retargeter.retarget(points)
            timings.append(time.perf_counter()-t)
            robot.compute_forward_kinematics(value)
            qpos.append(value[[retargeter.joint_names.index(name) for name in NATIVE_ORDER]])
            fk.append([robot.get_link_pose(i) for i in range(len(robot.link_names))])
        arrays.update({f"{side}_mano_vertices": vertices, f"{side}_mano_faces": model.faces,
                       f"{side}_source_points": landmarks, f"{side}_dex_native18": np.asarray(qpos),
                       f"{side}_dex_link_names": np.asarray(robot.link_names),
                       f"{side}_dex_link_poses": np.asarray(fk), f"{side}_dex_solve_time_s": np.asarray(timings)})
        report["sides"][side] = {"pca_fullpose_max_error_m": reconstruction_error,
                                  "v_template": info(vtemp_path), "mano_model": info(args.mano_root / f"MANO_{side.upper()}.pkl"),
                                  "dex_config": info(config), "baseline": "Dexplore-style position solver on shared native MANO targets",
                                  "target_asset_family": "official_dex_left" if side == "left" and args.left_assets else "dexplore_inspire_hand_new"}
    object_path = args.grab_root / raw["object"]["object_mesh"]
    mesh = trimesh.load(object_path, process=False)
    rotation = Rotation.from_rotvec(raw["object"]["params"]["global_orient"][frames]).as_matrix()
    pose = np.broadcast_to(np.eye(4), (len(frames),4,4)).copy()
    pose[:, :3, :3] = rotation.transpose(0,2,1)  # official GRAB ObjectModel uses row v @ R
    pose[:, :3, 3] = raw["object"]["params"]["transl"][frames]
    arrays.update(object_vertices=np.asarray(mesh.vertices), object_faces=np.asarray(mesh.faces),
                  object_poses=pose, object_id=np.asarray(raw["obj_name"]), native_joint_names=np.asarray(NATIVE_ORDER))
    # Independent official ObjectModel parity, including the row/column convention.
    repo = Path(__file__).resolve().parents[5]
    spec = importlib.util.spec_from_file_location("grab_objectmodel", repo / "dataset/GRAB/tools/objectmodel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    om = module.ObjectModel(np.asarray(mesh.vertices), batch_size=len(frames), dtype=torch.float64)
    with torch.no_grad():
        official = om(global_orient=torch.as_tensor(raw["object"]["params"]["global_orient"][frames], dtype=torch.float64),
                      transl=torch.as_tensor(pose[:, :3, 3])).vertices.numpy()
    transformed = np.einsum("tij,vj->tvi", pose[:, :3, :3], mesh.vertices) + pose[:, None, :3, 3]
    object_error = float(np.max(np.abs(official-transformed)))
    if object_error > 1e-7:
        raise ValueError(f"Object reconstruction mismatch {object_error}")
    np.savez_compressed(args.output / "source_and_dex.npz", **arrays)
    report.update(object_mesh=info(object_path), object_official_max_error_m=object_error,
                  coordinate_frame="GRAB native world", units="m/rad", finished_at=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "source_report.json", report)
    print(f"Prepared {len(frames)} shared GRAB frames, sides={args.sides}, at {args.output}", flush=True)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[5]
    manifest = {"Task": "ObjectInteractionCm", "modification_version": VERSION,
                "run_id": args.output.name, "run_status": "RUNNING", "seed": 42, "checkpoint": None,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "base_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
                "command": shlex.join([sys.executable]+sys.argv), "config": "prepare_config.json",
                "metadata_snapshot": "source_report.json", "output": str(args.output.resolve()),
                "code": [info(p) for p in sorted(Path(__file__).parent.glob("*.py"))]}
    write_json(args.output / "run_manifest.json", manifest)
    try:
        execute(args)
        manifest.update(run_status="COMPLETED", source_output=info(args.output / "source_and_dex.npz"))
    except Exception:
        manifest.update(run_status="FAILED", error=traceback.format_exc())
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(args.output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
