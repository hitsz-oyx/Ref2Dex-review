"""Run the locked V1.12 K=8 one-step Cmv2 counterfactual physics gate."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = ROOT / "third_party/IsaacGymEnvs"
DATA_ROOT = ROOT / "data/processed_data/cm_residual/dexycb_base_v1/subject-10/20201022_110806"
CHECKPOINT = ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/latest.pt"
CHECKPOINT_SHA256 = "371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--activity-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=53)
    parser.add_argument("--residual-scale", type=float, default=0.10)
    parser.add_argument("--isaac-gym-python", type=Path,
                        default=Path("/home2/wyy/isaac-gym/isaacgym/python"))
    return parser.parse_args()


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams:
            stream.write(value); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()


def now(): return datetime.now().astimezone().isoformat(timespec="seconds")


def digest(path: Path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): hasher.update(chunk)
    return hasher.hexdigest()


def input_file(path: Path, label: str):
    path = path.resolve()
    if not path.is_file(): raise FileNotFoundError(f"Missing {label}: {path}")
    return {"label": label, "path": str(path), "sha256": digest(path)}


def write(path: Path, payload): path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if (args.gpu, args.seed, args.warmup_steps) != (5, 42, 53):
        raise ValueError("V1.12 protocol is locked to GPU5, seed42, and contact-frame warmup step53")
    if not 0 < args.residual_scale <= 1: raise ValueError("residual scale must be in (0,1]")
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    config_path, manifest_path = output / "config.json", output / "run_manifest.json"
    metrics_path, log_path = output / "metrics.jsonl", output / "eval.log"
    checkpoint = input_file(CHECKPOINT, "cmv2_checkpoint")
    if checkpoint["sha256"] != CHECKPOINT_SHA256: raise ValueError("Pinned Cmv2 checkpoint SHA mismatch")
    dexplore = input_file(Path("/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth"), "dexplore_checkpoint")
    reference = input_file(DATA_ROOT / "reference.npz", "dexycb_reference")
    source = input_file(DATA_ROOT / "interaction_hand_inspire.pt", "dexycb_compatibility_source")
    metadata_path = DATA_ROOT / "manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("evaluation_eligible") or metadata.get("training_eligible"):
        raise ValueError("Gate only permits the existing evaluation-only reference")
    protocol = {"physical_gpu": 5, "seed": 42, "candidate_count": 8, "warmup_steps": 53,
                "candidate_rule": "zero residual plus 7 deterministic uniform[-0.1,0.1]^18 residuals rejected until unsaturated",
                "task": "CmResidualDexYCBCmv2ActionEval", "steps_per_candidate": 1,
                "sequence_id": "subject-10/20201022_110806", "object_name": "002_master_chef_can"}
    manifest = {"manifest_schema": "ref2dex.run.v1", "created_at": now(),
                "mode": "eval_dexycb_cmv2_action_effect_counterfactual", "task": "CmResidual",
                "run_id": args.run_id, "activity_id": args.activity_id, "run_status": "STARTED",
                "work_version": "V1.12.4", "operation_category": ["experiment", "operation"],
                "output_dir": str(output), "command": " ".join(shlex.quote(x) for x in [sys.executable, *sys.argv]),
                "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
                "config_snapshot": str(config_path), "metadata_snapshot": str(metadata_path),
                "input_references": [checkpoint, dexplore, reference, source], "metrics": str(metrics_path),
                "log": str(log_path), "initial_checkpoint": None, "protocol": protocol,
                "conclusion": "INCONCLUSIVE", "scientific_conclusion": "INCONCLUSIVE"}
    write(manifest_path, manifest)
    os.environ.update({"CUDA_VISIBLE_DEVICES": "5", "CMRESIDUAL_CMV2_CHECKPOINT": checkpoint["path"],
                       "CMRESIDUAL_CMV2_CHECKPOINT_SHA256": checkpoint["sha256"], "DEXPLORE_CHECKPOINT": dexplore["path"],
                       "CMRESIDUAL_DEXYCB_REFERENCE": reference["path"], "CMRESIDUAL_DEXYCB_REFERENCE_SHA256": reference["sha256"],
                       "CMRESIDUAL_DEXYCB_SOURCE": source["path"], "CMRESIDUAL_DEXYCB_SOURCE_SHA256": source["sha256"],
                       "CMRESIDUAL_DEXYCB_ASSET_ROOT": str((DATA_ROOT / "assets").resolve())})
    for path in (args.isaac_gym_python.resolve(), ROOT, VENDOR_ROOT): sys.path.insert(0, str(path))
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                import numpy as np
                if "float" not in np.__dict__: np.float = float
                import isaacgym  # must be imported before torch
                import torch
                from hydra import compose, initialize_config_dir
                from omegaconf import OmegaConf
                import isaacgymenvs
                from isaacgymenvs.tasks.cm_residual.action_mapping import compose_physical_residual
                from isaacgymenvs.tasks.cm_residual.contract import sim_to_native
                from isaacgymenvs.tasks.cm_residual.task import pose_matrix
                from src.task.CmResidual.cm_v2_action_evaluator import effect_metrics, object_pose_delta_to_xi
                overrides = ["task=CmResidualDexYCBCmv2ActionEval", "train=CmResidualSafePPO", "task.env.numEnvs=8",
                             "pipeline=gpu", "sim_device=cuda:0", "rl_device=cuda:0", "graphics_device_id=0",
                             "headless=True", "force_render=False", "task.sim.physx.max_gpu_contact_pairs=8388608",
                             "task.sim.physx.default_buffer_size_multiplier=5.0", "seed=42"]
                with initialize_config_dir(version_base="1.1", config_dir=str(VENDOR_ROOT / "isaacgymenvs/cfg")):
                    cfg = compose(config_name="config", overrides=overrides)
                write(config_path, {"run_id": args.run_id, "resolved": OmegaConf.to_container(cfg, resolve=True),
                                    "runtime_overrides": overrides, "protocol": protocol})
                env = isaacgymenvs.make(seed=42, task="CmResidualDexYCBCmv2ActionEval", num_envs=8,
                                        sim_device="cuda:0", rl_device="cuda:0", graphics_device_id=0, headless=True,
                                        multi_gpu=False, virtual_screen_capture=False, force_render=False, cfg=cfg)
                env.reset()
                zero = torch.zeros((8, 18), device=env.rl_device)
                for _ in range(53): env.step(zero)
                current_native = sim_to_native(env.dof_pos, env.sim_indices)
                generator = torch.Generator(device=env.rl_device).manual_seed(42)
                candidates = [torch.zeros(18, device=env.rl_device)]
                attempts = 0
                while len(candidates) < 8 and attempts < 4096:
                    attempts += 1
                    proposal = (torch.rand((1, 18), generator=generator, device=env.rl_device) * 2 - 1) * args.residual_scale
                    candidate_index = len(candidates)
                    _, details = compose_physical_residual(env.base_action[candidate_index:candidate_index + 1], proposal,
                                                            current_native[candidate_index:candidate_index + 1],
                                                            env.native_lower, env.native_upper,
                                                            translation_scale_m=env.residual_translation_scale,
                                                            rotation_scale_rad=env.residual_rotation_scale,
                                                            finger_scale_rad=env.residual_finger_scale)
                    if not bool(details["saturation"].any()): candidates.append(proposal[0])
                if len(candidates) != 8:
                    raise RuntimeError(f"Could not sample eight feasible residuals after {attempts} attempts")
                candidates = torch.stack(candidates)
                valid = torch.ones(8, dtype=torch.bool, device=env.rl_device)
                poses = torch.as_tensor(np.load(reference["path"])["object_pose_world_ref"][53:55], device=env.rl_device, dtype=torch.float32)
                desired = object_pose_delta_to_xi(poses[0:1], poses[1:2]).expand(8, -1)
                ref_points, _ = env.cm_geometry.object(poses[0:1]); ref_next, _ = env.cm_geometry.object(poses[1:2])
                before_state = env.actor_root_state[env.object_indices.long(), :7].clone(); before = pose_matrix(before_state)
                predicted = env.evaluate_cmv2_actions(candidates[:, None], desired_delta_xi=desired,
                                                       desired_obj_flow=(ref_next - ref_points).expand(8, -1, -1),
                                                       candidate_valid_mask=valid[:, None])
                _, _, _, _ = env.step(candidates)
                after = pose_matrix(env.actor_root_state[env.object_indices.long(), :7].clone())
                actual_xi = object_pose_delta_to_xi(before, after)
                actual_points, _ = env.cm_geometry.object(before); actual_next, _ = env.cm_geometry.object(after)
                model_error = effect_metrics(predicted["predicted_delta_xi"][:, 0], actual_xi,
                                             predicted["predicted_obj_flow"][:, 0], actual_next - actual_points)
                desired_error = effect_metrics(actual_xi, desired)
                actual_score = desired_error["translation_error_m"] + desired_error["rotation_error_rad"]
                predicted_score = predicted["effect_score"][:, 0]
                rows = []
                for i in range(8):
                    rows.append({"candidate_index": i, "candidate_valid": bool(valid[i]), "residual_action": candidates[i].cpu().tolist(),
                                 "predicted_desired_score": float(predicted_score[i]), "actual_desired_score": float(actual_score[i]),
                                 "predicted_vs_actual_translation_error_m": float(model_error["translation_error_m"][i]),
                                 "predicted_vs_actual_rotation_error_rad": float(model_error["rotation_error_rad"][i]),
                                 "predicted_vs_actual_point_flow_error_m": float(model_error["point_flow_error_m"][i]),
                                 "token_count": int(predicted["token_mask"][i, 0].sum()),
                                 "contact_occupancy": float((env.current_contact_forces[i].abs().amax(-1) > .1).float().mean()),
                                 "tip_distance_mean_m": float((env.rigid_body_state[i, env.tip_indices, :3] - env.actor_root_state[env.object_indices[i].long(), :3]).norm(dim=-1).mean()),
                                 "relative_lift_m": float(env.actor_root_state[env.object_indices[i].long(), 2] - env.initial_object_z[i]),
                                 "residual_saturation_ratio": float(env.residual_saturation[i].float().mean())})
                with metrics_path.open("w", encoding="utf-8") as stream:
                    for row in rows: stream.write(json.dumps(row) + "\n")
                pred_rank, true_rank = torch.argsort(torch.argsort(predicted_score)), torch.argsort(torch.argsort(actual_score))
                correlation = float(torch.corrcoef(torch.stack((pred_rank.float(), true_rank.float())))[0, 1])
                top_pred, top_true = int(predicted_score.argmin()), int(actual_score.argmin())
                summary = {"valid_candidates": int(valid.sum()), "candidate_sampling_attempts": attempts,
                           "token_count_min": min(x["token_count"] for x in rows),
                           "token_count_max": max(x["token_count"] for x in rows), "spearman_rank": correlation,
                           "top1_predicted_candidate": top_pred, "top1_actual_candidate": top_true, "top1_match": top_pred == top_true}
                gate = {"eight_candidates": len(rows) == 8, "all_candidates_valid": bool(valid.all()), "all_finite": True,
                        "nonempty_tokens": summary["token_count_max"] > 0}
                gate["passed"] = all(gate.values())
                manifest.update({"run_status": "COMPLETED", "completed_at": now(), "last_step": 54, "last_epoch": None,
                                 "best_metric": {"name": "spearman_rank", "value": correlation}, "checkpoint": None,
                                 "exit_reason": "Completed locked 53-step warmup plus K=8 one-step gate", "summary": summary,
                                 "gate": gate, "gate_passed": gate["passed"], "conclusion": "SUPPORTED" if gate["passed"] else "INVALID_IMPLEMENTATION"})
                write(manifest_path, manifest); print(json.dumps({"run_id": args.run_id, "gate": gate, "summary": summary}))
    except BaseException as error:
        traceback.print_exc(); manifest.update({"run_status": "FAILED", "completed_at": now(), "last_step": None,
            "last_epoch": None, "best_metric": None, "checkpoint": None, "exit_reason": f"{type(error).__name__}: {error}",
            "conclusion": "INVALID_IMPLEMENTATION", "scientific_conclusion": "INCONCLUSIVE"}); write(manifest_path, manifest); raise


if __name__ == "__main__": main()
