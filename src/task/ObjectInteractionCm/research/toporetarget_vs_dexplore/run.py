"""Inspire foundation checks and bounded comparison using unmodified Topo APIs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from inspire_adapter import (NATIVE_ORDER, VERSION, info, load_hand, native18_to_state,
                             state_to_native18, write_json)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def check_model(model, assets, data, side):
    from toporetarget.robots.urdf.parser import parse_urdf
    from toporetarget.robots.urdf.kinematics import forward_kinematics_numpy
    report = model.validate(seed=42, dtype="float64").as_dict()
    if report["status"] != "pass":
        raise ValueError(f"Target validation failed: {report}")
    full = parse_urdf(Path(assets) / f"inspire_{side}_full.urdf")
    native = data[f"{side}_dex_native18"]
    q = native[:, [NATIVE_ORDER.index(name) for name in model.dof_names]]
    base = native18_to_state(native)
    poses = model.forward_kinematics_scene(torch.as_tensor(q), torch.as_tensor(base))
    dex_names = data[f"{side}_dex_link_names"].tolist()
    error = max(float(np.max(np.abs(value.detach().numpy()-data[f"{side}_dex_link_poses"][:, dex_names.index(name)])))
                for name, value in poses.items())
    if error > 1e-6:
        raise ValueError(f"Pinocchio/Topo FK mismatch: {error}")
    rng = np.random.default_rng(42)
    full_names = [joint.name for joint in full.actuated_joints]
    random_error = 0.
    for _ in range(10):
        finger = rng.uniform(model.joint_lower, model.joint_upper)
        test_native = np.concatenate((rng.uniform(-.2,.2,3), rng.uniform(-1.,1.,3),
                                      finger[[list(model.dof_names).index(n) for n in NATIVE_ORDER[6:]]]))
        test_base = native18_to_state(test_native)
        back = state_to_native18(test_base, finger, model.dof_names)
        np.testing.assert_allclose(back, test_native, atol=1e-10)
        expected = forward_kinematics_numpy(full, test_native[[NATIVE_ORDER.index(n) for n in full_names]])
        got = model.forward_kinematics_scene(torch.as_tensor(finger), torch.as_tensor(test_base))
        random_error = max(random_error, max(float(np.max(np.abs(got[n].numpy()-expected[n]))) for n in got))
    if random_error > 1e-10:
        raise ValueError(f"Wrist factoring mismatch: {random_error}")
    middle = (model.joint_lower+model.joint_upper)*.5
    jacobian = torch.autograd.functional.jacobian(model.keypoints_base,
                                                 torch.tensor(middle, dtype=torch.float64)).numpy()
    fd = np.stack([(model.keypoints_base(middle+np.eye(model.num_dofs)[i]*1e-6).numpy()
                    -model.keypoints_base(middle-np.eye(model.num_dofs)[i]*1e-6).numpy())/2e-6
                   for i in range(model.num_dofs)], axis=-1)
    jacobian_error = float(np.max(np.abs(jacobian-fd)))
    if jacobian_error > 1e-7:
        raise ValueError(f"Anchor Jacobian mismatch: {jacobian_error}")
    report.update(pinocchio_fk_matrix_max_error=error, random_wrist_factor_max_error=random_error,
                  anchor_jacobian_max_error=jacobian_error, original_mimic_enforced=False)
    return report, q, base


def metrics(points, source):
    distance = np.linalg.norm(points-source, axis=-1)
    return {"tip_mean_mm": float(distance[:, [4,8,12,16,20]].mean()*1000),
            "tip_p95_mm": float(np.percentile(distance[:, [4,8,12,16,20]],95)*1000),
            "all21_mean_mm": float(distance.mean()*1000)}


def execute(args, manifest):
    sys.path.insert(0, str(args.topo_root / "src"))
    from toporetarget.data.schema import (HOISequence, HandTrack, KeypointTrack, MeshDefinition,
                                         PoseTrack, RigidObjectTrack, SequenceMetadata)
    from toporetarget.retarget.pipeline import build_warm_start_trajectory
    from toporetarget.retarget.frames import load_frame_profile
    from toporetarget.retarget.bones import load_bone_profile
    from toporetarget.retarget.solver import load_solver_profile
    from toporetarget.geometry.surface_sampling import load_surface_profile, sample_mesh_surface
    from toporetarget.geometry.robot_surface import load_robot_surface_profile, sample_robot_collision_surface
    from toporetarget.retarget.interaction_graph import build_source_interaction_graph
    from toporetarget.retarget.final_refinement import (
        CollisionQueryProfile, RefinementCoordinateProfile, RefinementSolverProfile,
        build_final_trajectory, prepare_refinement_resources, dynamic_collision_points_numpy)

    data = np.load(args.input, allow_pickle=False)
    source_report = json.loads((args.input.parent / "source_report.json").read_text())
    sides = list(source_report["sides"])
    count = len(data["source_frame_id"])
    source_mesh = MeshDefinition(data["object_vertices"], data["object_faces"])
    obj = RigidObjectTrack(str(data["object_id"]), source_mesh, PoseTrack(data["object_poses"]))
    hands = []
    for side in sides:
        wrist = np.broadcast_to(np.eye(4), (count,4,4)).copy()
        wrist[:, :3,3] = data[f"{side}_source_points"][:,0]
        hands.append(HandTrack(side, side, PoseTrack(wrist), keypoint_tracks={
            "mediapipe21": KeypointTrack(data[f"{side}_source_points"], "mediapipe21")}))
    sequence = HOISequence(SequenceMetadata(dataset_name="grab", sequence_id=source_report["sequence"],
                                           native_fps=source_report["fps"], timestamps=data["timestamps"]),
                           hands=hands, rigid_objects=[obj])
    sequence.validate()
    frame = load_frame_profile("canonical_keypoint_wrist_v1")
    bone = load_bone_profile("mediapipe21_full_finger_chain_v1")
    solver = load_solver_profile("paper_repro_scipy_trf")
    surface_profile = load_robot_surface_profile("engineering_collision_32_per_geometry")
    object_samples = sample_mesh_surface(source_mesh.vertices_local, source_mesh.faces,
                                         load_surface_profile("paper_strict_area_uniform"), mesh_id=obj.object_id)
    object_samples.save(args.output / "object_samples.npz")
    result = {"work_version": VERSION, "source_frame_ids": data["source_frame_id"].tolist(),
              "dataset": "GRAB", "sides": {}, "conclusion": "INCONCLUSIVE"}
    for side in sides:
        model = load_hand(args.assets, side)
        print(f"{side}: validate target and independent FK", flush=True)
        validation, dex_q, dex_base = check_model(model, args.assets, data, side)
        write_json(args.output / f"{side}_target_validation.json", validation)
        source = data[f"{side}_source_points"]
        dex_points = model.keypoints_scene(torch.as_tensor(dex_q), torch.as_tensor(dex_base)).numpy()
        print(f"{side}: Topo warm start for {count} frames", flush=True)
        start = time.perf_counter()
        warm, warm_report = build_warm_start_trajectory(sequence, side, model, frame, bone, solver)
        warm_wall = time.perf_counter()-start
        np.savez_compressed(args.output / f"{side}_warm.npz", **warm.arrays)
        write_json(args.output / f"{side}_warm_metadata.json", warm.metadata)
        warm_points = model.keypoints_scene(torch.as_tensor(warm.arrays["qpos"]),
                                           torch.as_tensor(warm.arrays["base_pose_scene"])).numpy()
        graph = build_source_interaction_graph(sequence, side, obj.object_id, object_samples)
        np.savez_compressed(args.output / f"{side}_graph.npz", **graph.arrays())
        surface = sample_robot_collision_surface(model, model.neutral_q, surface_profile)
        surface.save(args.output / f"{side}_collision_surface.npz")
        summary = {"foundation": "pass", "collision_samples": surface.count,
                   "dex_position": metrics(dex_points,source), "topo_warm": metrics(warm_points,source),
                   "warm_wall_s": warm_wall, "dex_solver_s": data[f"{side}_dex_solve_time_s"].tolist()}
        result["sides"][side] = summary
        comparison = {"source_points": source, "dex_points": dex_points, "warm_points": warm_points,
                      "dex_native18": data[f"{side}_dex_native18"],
                      "warm_native18": state_to_native18(warm.arrays["base_pose_scene"], warm.arrays["qpos"], model.dof_names),
                      "source_frame_id": data["source_frame_id"]}
        write_json(args.output / "report.json", result)
        np.savez_compressed(args.output / f"{side}_comparison.npz", **comparison)
        if args.foundation_only:
            continue
        print(f"{side}: prepare exact object SDF and constrained refinement", flush=True)
        final_solver = RefinementSolverProfile.load(args.refinement_profile)
        resources = prepare_refinement_resources(sequence, graph, final_solver,
                                                geometry_artifact_root=args.output / "object_sdf")
        write_json(args.output / f"{side}_sdf.json", {"solver": resources.sdf_report,
                                                     "geometry": resources.geometry_policy})
        def callback(index, frame_result, context):
            row = {"side": side, "local_frame": index, "source_frame_id": int(data['source_frame_id'][index]),
                   "accepted": bool(frame_result.accepted), "failure": frame_result.failure,
                   "elapsed_s": time.perf_counter()-start}
            with (args.output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(row)+"\n")
            print(row, flush=True)
        start = time.perf_counter()
        final, final_report = build_final_trajectory(sequence, warm, graph, model, surface, frame, bone,
            RefinementCoordinateProfile.load("local_seed_delta_v1"),
            CollisionQueryProfile.load("adaptive_active_set_v1"), final_solver,
            resources=resources, continue_on_failure=True, frame_callback=callback)
        np.savez_compressed(args.output / f"{side}_final.npz", **final.arrays)
        write_json(args.output / f"{side}_final_metadata.json", final.metadata)
        final_points = model.keypoints_scene(torch.as_tensor(final.arrays["qpos"]),
                                             torch.as_tensor(final.arrays["base_pose_scene"])).numpy()
        summary.update(topo_final=metrics(final_points,source), final_wall_s=time.perf_counter()-start,
                       accepted_frames=int(np.count_nonzero(final.arrays["accepted"])))
        comparison.update(final_points=final_points, final_native18=state_to_native18(
            final.arrays["base_pose_scene"], final.arrays["qpos"], model.dof_names))
        for method, q, base in [("dex_position",dex_q,dex_base),("topo_warm",warm.arrays["qpos"],warm.arrays["base_pose_scene"]),
                                ("topo_final",final.arrays["qpos"],final.arrays["base_pose_scene"])]:
            signed = np.stack([resources.reference_sdf.query_scene(
                dynamic_collision_points_numpy(model, surface, q[i], base[i]),data["object_poses"][i]).signed_distance
                               for i in range(count)])
            comparison[f"{method}_signed_distance_m"] = signed
            summary[method].update(penetration_over_1mm_fraction=float(np.mean(signed < -.001)),
                                   max_penetration_mm=float(max(0., -signed.min())*1000))
        np.savez_compressed(args.output / f"{side}_comparison.npz", **comparison)
        write_json(args.output / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--topo-root", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--foundation-only", action="store_true")
    p.add_argument("--refinement-profile", default="scipy_slsqp_active_set_contact_rich_v3_fixed")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    repo = Path(__file__).resolve().parents[5]
    manifest = {"Task": "ObjectInteractionCm", "work_version": VERSION, "run_id": args.output.name,
                "run_status": "RUNNING", "started_at": now(), "seed": 42, "checkpoint": None,
                "base_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
                "command": shlex.join([sys.executable]+sys.argv), "config": "config.json",
                "input": info(args.input), "metadata_snapshot": "metadata.json", "output": str(args.output.resolve())}
    write_json(args.output / "config.json", {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    write_json(args.output / "metadata.json", {"coordinate_frame": "GRAB native world", "units": "m/rad",
                                              "qpos_export_order": NATIVE_ORDER, "mimic_policy": "independent_12_match_dexplore"})
    write_json(args.output / "environment.json", {"python":sys.version,"executable":sys.executable,
              "packages":{d:importlib.metadata.version(d) for d in ["numpy","torch","scipy","trimesh","zarr"]}})
    manifest["external_commits"] = {str(root):subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip()
                                     for root in [args.topo_root,Path('/home/wbcd/workspace/oyx_ws/dexplore')]}
    manifest["code"] = [info(p) for p in sorted(Path(__file__).parent.glob("*.py"))]
    write_json(args.output / "run_manifest.json", manifest)
    try:
        execute(args, manifest)
        manifest["run_status"] = "COMPLETED"
    except Exception:
        manifest["run_status"] = "FAILED"
        manifest["error"] = traceback.format_exc()
        raise
    finally:
        manifest["finished_at"] = now()
        write_json(args.output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
