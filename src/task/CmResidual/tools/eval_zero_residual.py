"""Run the frozen DExplore policy with an exactly zero physical residual."""
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


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
VENDOR_ROOT = REPOSITORY_ROOT / "third_party/IsaacGymEnvs"
DEFAULT_DEXPLORE = Path("/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth")
DEFAULT_OI_CM = REPOSITORY_ROOT / (
    "outputs/objectinteractioncm/"
    "object_interaction_cm_dexplore_rl_v1_3_20260910_020856/checkpoints/best.pt"
)
DEFAULT_REFERENCE = REPOSITORY_ROOT / (
    "data/processed_data/cm_residual/reference_tracking_v1/"
    "s1_airplane_lift/reference.npz"
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True,
        text=True, stdout=subprocess.PIPE).stdout.strip()


def _input(path: Path, kind: str) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing {kind}: {resolved}")
    stat = resolved.stat()
    return {
        "kind": kind,
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(resolved),
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _scalar(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--activity-id", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=367)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dexplore-checkpoint", type=Path, default=DEFAULT_DEXPLORE)
    parser.add_argument("--oi-cm-checkpoint", type=Path, default=DEFAULT_OI_CM)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--isaac-gym-python", type=Path,
        default=Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python")))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.num_envs <= 0:
        raise ValueError("steps and num-envs must be positive")
    run_id = args.run_id or f"cmresidual_zero_v13_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output = (args.output or REPOSITORY_ROOT / "outputs/CmResidual" / run_id).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing evaluation: {output}")
    output.mkdir(parents=True)
    config_path = output / "config.json"
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    log_path = output / "eval.log"

    dexplore = _input(args.dexplore_checkpoint, "dexplore_checkpoint")
    oi_cm = _input(args.oi_cm_checkpoint, "oi_cm_checkpoint")
    reference = _input(args.reference, "reference")
    reference_manifest_path = Path(reference["path"]).with_name("manifest.json")
    if not reference_manifest_path.is_file():
        raise FileNotFoundError(f"Missing reference manifest: {reference_manifest_path}")
    reference_manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    base_commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    started_at = _now()
    activity_id = args.activity_id or (
        f"ACT-{datetime.now().strftime('%Y%m%d-%H%M%S')}-CMRESIDUAL-ZERO")
    manifest = {
        "manifest_schema": "ref2dex.run.v1",
        "created_at": started_at,
        "mode": "eval_zero_residual",
        "task": "CmResidual",
        "run_id": run_id,
        "run_name": run_id,
        "activity_id": activity_id,
        "run_status": "STARTED",
        "modification_version": "V1.3.2",
        "operation_category": ["experiment", "operation"],
        "output_dir": str(output),
        "command": command,
        "seed": args.seed,
        "base_commit": base_commit,
        "worktree_dirty": dirty,
        "git": {
            "repository": str(REPOSITORY_ROOT),
            "commit": base_commit,
            "branch": _git("branch", "--show-current"),
            "dirty": dirty,
        },
        "config_snapshot": str(config_path),
        "metadata_snapshot": str(reference_manifest_path.resolve()),
        "input_references": [dexplore, oi_cm, reference],
        "reference_training_eligible": bool(reference_manifest.get("training_eligible", False)),
        "initial_checkpoint": None,
        "metrics": str(metrics_path),
        "log": str(log_path),
        "conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)

    os.environ["DEXPLORE_CHECKPOINT"] = dexplore["path"]
    os.environ["OI_CM_CHECKPOINT"] = oi_cm["path"]
    os.environ["CMRESIDUAL_REFERENCE"] = reference["path"]
    for path in (args.isaac_gym_python.resolve(), REPOSITORY_ROOT, VENDOR_ROOT):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    env = None
    step_rows = []
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
            tee_out, tee_err = Tee(sys.stdout, log_stream), Tee(sys.stderr, log_stream)
            with redirect_stdout(tee_out), redirect_stderr(tee_err):
                print(f"run_id={run_id} run_status=STARTED started_at={started_at}")
                import isaacgym  # noqa: F401 - must precede torch
                import torch
                from hydra import compose, initialize_config_dir
                from omegaconf import OmegaConf
                import isaacgymenvs

                with initialize_config_dir(
                    version_base="1.1", config_dir=str(VENDOR_ROOT / "isaacgymenvs/cfg")):
                    cfg = compose(config_name="config", overrides=[
                        "task=CmResidual",
                        f"task.env.numEnvs={args.num_envs}",
                        "pipeline=cpu",
                        "sim_device=cpu",
                        "rl_device=cpu",
                        "graphics_device_id=-1",
                        "headless=True",
                        "force_render=False",
                        "num_subscenes=1",
                        f"seed={args.seed}",
                    ])
                resolved_task = OmegaConf.to_container(cfg.task, resolve=True)
                _write_json(config_path, {
                    "run_id": run_id,
                    "task": resolved_task,
                    "steps": args.steps,
                    "num_envs": args.num_envs,
                    "seed": args.seed,
                    "residual_action": "all zeros",
                })
                env = isaacgymenvs.make(
                    seed=args.seed, task="CmResidual", num_envs=args.num_envs,
                    sim_device="cpu", rl_device="cpu", graphics_device_id=-1,
                    headless=True, multi_gpu=False, virtual_screen_capture=False,
                    force_render=False, cfg=cfg)
                observation = env.reset()["obs"]
                actions = torch.zeros((args.num_envs, 18), dtype=torch.float32, device=env.rl_device)
                if tuple(observation.shape) != (args.num_envs, 2005):
                    raise RuntimeError(f"Unexpected observation shape: {tuple(observation.shape)}")
                with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics_stream:
                    for step in range(1, args.steps + 1):
                        observation_dict, reward, done, info = env.step(actions)
                        observation = observation_dict["obs"]
                        row = {
                            "step": step,
                            "reward_mean": _scalar(reward.mean()),
                            "lift_mean_m": _scalar(info["lift_mean"]),
                            "tip_distance_mean_m": _scalar(info["tip_distance_mean"]),
                            "contact_occupancy": _scalar(info["contact_occupancy"]),
                            "success_rate": _scalar(info["success_rate"]),
                            "done_count": int(done.sum().detach().cpu().item()),
                            "base_action_abs_max": _scalar(env.base_action.abs().amax()),
                            "residual_action_abs_max": _scalar(actions.abs().amax()),
                            "residual_target_delta_max": _scalar(info["residual_target_delta_max"]),
                            "residual_translation_m_rms": _scalar(info["residual_translation_m_rms"]),
                            "residual_rotation_rad_rms": _scalar(info["residual_rotation_rad_rms"]),
                            "residual_finger_rad_rms": _scalar(info["residual_finger_rad_rms"]),
                            "target_saturation_ratio": _scalar(info["residual_saturation_ratio"]),
                            "observation_finite": bool(torch.isfinite(observation).all().item()),
                            "target_finite": bool(torch.isfinite(env.native_targets).all().item()),
                        }
                        metrics_stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        step_rows.append(row)
                        if not row["observation_finite"] or not row["target_finite"]:
                            raise FloatingPointError(f"Non-finite rollout state at step {step}")
                        if row["residual_action_abs_max"] != 0.0 or row["residual_target_delta_max"] != 0.0:
                            raise RuntimeError(f"Zero-residual parity failed at step {step}: {row}")
                print(f"run_id={run_id} completed_steps={len(step_rows)}")
    except BaseException as error:
        traceback.print_exc()
        manifest.update({
            "run_status": "FAILED",
            "completed_at": _now(),
            "last_step": len(step_rows),
            "best_metric": None,
            "checkpoint": None,
            "exit_reason": f"{type(error).__name__}: {error}",
            "conclusion": "INVALID_IMPLEMENTATION",
        })
        _write_json(manifest_path, manifest)
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)

    lift_values = [row["lift_mean_m"] for row in step_rows]
    tip_values = [row["tip_distance_mean_m"] for row in step_rows]
    summary = {
        "steps": len(step_rows),
        "mean_lift_m": sum(lift_values) / len(lift_values),
        "max_lift_m": max(lift_values),
        "mean_tip_distance_m": sum(tip_values) / len(tip_values),
        "final_success_rate": step_rows[-1]["success_rate"],
        "reset_count": sum(row["done_count"] for row in step_rows),
        "mean_contact_occupancy": sum(row["contact_occupancy"] for row in step_rows) / len(step_rows),
        "max_target_saturation_ratio": max(row["target_saturation_ratio"] for row in step_rows),
        "max_base_action_abs": max(row["base_action_abs_max"] for row in step_rows),
        "max_residual_target_delta": max(row["residual_target_delta_max"] for row in step_rows),
    }
    manifest.update({
        "run_status": "COMPLETED",
        "completed_at": _now(),
        "last_step": len(step_rows),
        "last_epoch": None,
        "best_metric": {"name": "max_lift_m", "value": summary["max_lift_m"]},
        "checkpoint": None,
        "exit_reason": "Reached configured zero-residual evaluation budget",
        "conclusion": "SUPPORTED",
        "scientific_conclusion": "INCONCLUSIVE",
        "summary": summary,
    })
    _write_json(manifest_path, manifest)
    print(json.dumps({"run_id": run_id, "run_status": "COMPLETED", **summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
