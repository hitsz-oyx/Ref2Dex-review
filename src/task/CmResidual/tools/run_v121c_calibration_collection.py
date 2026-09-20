"""Collect the deterministic 64-state V1.21c duplicate-calibration pool."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

import numpy as np

from src.task.CmResidual.v121c_ranking import candidate_seed

ROOT = Path(__file__).resolve().parents[4]
DEXPLORE_ROOT = ROOT / "third_party" / "DExplore"
ISAAC_GYM_PYTHON = Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python"))
DEXPLORE_PYTHON = Path(os.environ.get(
    "REF2DEX_DEXPLORE_PYTHON",
    "/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python",
))
BOOTSTRAP = Path(__file__).resolve().with_name("v121c_collect_episode_bootstrap.py")
OUTPUT_ROOT = ROOT / "outputs" / "CmResidual"
MOTION_ROOT = ROOT / "data/processed_data/dexplore_reconstructed_v120_coordfix_v4/converted_attempt1"
CHECKPOINT = Path("/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth")
DEFAULT_PHYSICAL_GPU = 0
GPU_CAPACITY_GATE_MIB = 1024
CALIBRATION_QUOTAS = {0: 24, 1: 24, 2: 16}
MAX_EPISODES = 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _gpu_used_mib(gpu: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.splitlines()}
    if gpu not in values:
        raise RuntimeError(f"nvidia-smi did not report GPU{gpu}")
    return values[gpu]


def _command(seed: int, episode_output: Path) -> list[str]:
    return [
        str(DEXPLORE_PYTHON),
        str(BOOTSTRAP),
        "--test",
        "--checkpoint", str(CHECKPOINT),
        "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(MOTION_ROOT),
        "--num_envs", "1",
        "--seed", str(seed),
        "--headless",
        "--sim_device", "cuda:0",
        "--rl_device", "cuda:0",
        "--graphics_device_id", "0",
        "--output_path", str(episode_output / "unused_train"),
        "--export_rl",
        "--export_output_dir", str(episode_output / "unused_export"),
    ]


def _input_identity() -> dict[str, dict[str, object]]:
    manifest = MOTION_ROOT.parent / "manifest.json"
    tensor = MOTION_ROOT / "s1_airplane_lift/interaction_hand_inspire.pt"
    values = {
        "checkpoint": CHECKPOINT,
        "input_manifest": manifest,
        "motion_tensor": tensor,
        "bootstrap": BOOTSTRAP,
        "vendor_run": DEXPLORE_ROOT / "dexplore/run.py",
        "dexplore_python": DEXPLORE_PYTHON,
        "inspire_urdf": DEXPLORE_ROOT / "dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf",
        "airplane_mesh": DEXPLORE_ROOT / "dexplore/data/assets/mjcf/objects/airplane/airplane.obj",
        "table_mesh": DEXPLORE_ROOT / "dexplore/data/assets/mjcf/objects/table/table.obj",
    }
    result = {}
    for name, path in values.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        result[name] = {"path": str(resolved), "sha256": _sha256(resolved)}
    if result["checkpoint"]["sha256"] != "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553":
        raise ValueError("official DExplore checkpoint SHA256 mismatch")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("classification") != "reconstructed_baseline":
        raise ValueError("collection requires reconstructed_baseline input")
    return result


def _load_rows(
    path: Path, episode_id: int, seed: int, collection_batch_id: str
) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        if payload["candidate_seed"].dtype != np.uint64:
            raise ValueError("candidate_seed artifact must be uint64")
        count = int(payload["state_id"].shape[0])
        rows = []
        for index in range(count):
            state_id = str(payload["state_id"][index])
            stored_candidate_seed = int(payload["candidate_seed"][index])
            expected_candidate_seed = candidate_seed(collection_batch_id, state_id)
            if stored_candidate_seed != expected_candidate_seed:
                raise ValueError(
                    f"candidate_seed mismatch for {state_id}: "
                    f"{stored_candidate_seed} != {expected_candidate_seed}"
                )
            rows.append({
                "state_id": state_id,
                "episode_id": episode_id,
                "seed": seed,
                "frame_id": int(payload["frame_id"][index]),
                "reference_index": int(payload["reference_index"][index]),
                "progress": int(payload["progress"][index]),
                "phase_id": int(payload["phase_id"][index]),
                "active_reason_mask": int(payload["active_reason_mask"][index]),
                "candidate_seed": stored_candidate_seed,
                "executed_action_prefix_sha256": str(payload["executed_action_prefix_sha256"][index]),
                "raw_obs": payload["raw_obs"][index].astype(np.float32, copy=True),
                "policy_mu": payload["policy_mu"][index].astype(np.float32, copy=True),
                "policy_sigma": payload["policy_sigma"][index].astype(np.float32, copy=True),
            })
        return rows


def _select(rows: list[dict[str, object]]) -> list[dict[str, object]] | None:
    selected = []
    for phase, quota in CALIBRATION_QUOTAS.items():
        candidates = sorted(
            (row for row in rows if int(row["phase_id"]) == phase),
            key=lambda row: str(row["state_id"]),
        )
        if len(candidates) < quota:
            return None
        selected.extend(candidates[:quota])
    return sorted(selected, key=lambda row: (int(row["phase_id"]), str(row["state_id"])))


def _save_selection(path: Path, rows: list[dict[str, object]], batch_id: str) -> None:
    np.savez_compressed(
        path,
        collection_batch_id=np.asarray([batch_id] * len(rows)),
        state_id=np.asarray([row["state_id"] for row in rows]),
        episode_id=np.asarray([row["episode_id"] for row in rows], dtype=np.int64),
        seed=np.asarray([row["seed"] for row in rows], dtype=np.int64),
        frame_id=np.asarray([row["frame_id"] for row in rows], dtype=np.int64),
        reference_index=np.asarray([row["reference_index"] for row in rows], dtype=np.int64),
        progress=np.asarray([row["progress"] for row in rows], dtype=np.int64),
        phase_id=np.asarray([row["phase_id"] for row in rows], dtype=np.uint8),
        active_reason_mask=np.asarray([row["active_reason_mask"] for row in rows], dtype=np.uint8),
        candidate_seed=np.asarray([row["candidate_seed"] for row in rows], dtype=np.uint64),
        executed_action_prefix_sha256=np.asarray([row["executed_action_prefix_sha256"] for row in rows]),
        raw_obs=np.stack([row["raw_obs"] for row in rows]),
        policy_mu=np.stack([row["policy_mu"] for row in rows]),
        policy_sigma=np.stack([row["policy_sigma"] for row in rows]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, default=DEFAULT_PHYSICAL_GPU)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = (OUTPUT_ROOT / args.run_id).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    inputs = _input_identity()
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    used_mib = _gpu_used_mib(args.gpu)
    if used_mib > GPU_CAPACITY_GATE_MIB:
        raise RuntimeError(f"GPU{args.gpu} capacity gate failed: {used_mib} MiB used")
    batch_id = f"{args.run_id}:calibration"
    if args.dry_run:
        print(json.dumps({
            "preflight": "passed",
            "git_commit": _git("rev-parse", "HEAD"),
            "physical_gpu": args.gpu,
            "gpu_used_mib": used_mib,
            "collection_batch_id": batch_id,
            "episode0_command": _command(5909, output / "episodes/0000"),
            "inputs": inputs,
            "quotas": CALIBRATION_QUOTAS,
        }, sort_keys=True))
        return 0

    output.mkdir(parents=True)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir()
    log_path = output / "logs/eval.log"
    log_path.parent.mkdir()
    metrics_path = output / "metrics.jsonl"
    manifest_path = output / "run_manifest.json"
    config = {
        "collection_batch_id": batch_id,
        "first_episode_seed": 5909,
        "max_episodes": MAX_EPISODES,
        "phase_quotas": CALIBRATION_QUOTAS,
        "collector_envs": 1,
        "control_hz": 30,
        "physics_substeps_per_action": 2,
        "policy_actions": "official Gaussian sample; epsilon-greedy disabled by checkpoint config",
        "physical_gpu": args.gpu,
        "logical_gpu": 0,
        "resource_override": (
            "user-approved GPU3 override on 2026-09-20" if args.gpu == 3
            else "plan-default GPU0" if args.gpu == 0 else "non-default GPU requires recorded approval"
        ),
        "dexplore_surface_seed": 2024,
        "active_contract": {
            "contact_force_inf_threshold_n": 0.1,
            "actual_tip_distance_threshold_m": 0.04,
            "reference_tip_distance_threshold_m": 0.04,
            "reference_translation_threshold_m": 0.002,
            "reference_rotation_threshold_rad": 0.01,
            "reference_horizon": 6,
        },
    }
    _write_json(output / "config.json", config)
    manifest = {
        "manifest_schema": "ref2dex.cmresidual.v121c.calibration_collection.v1",
        "created_at": _now(),
        "task": "CmResidual",
        "work_version": "V1.21",
        "run_id": args.run_id,
        "run_status": "STARTED",
        "git_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "output_dir": str(output),
        "config": str(output / "config.json"),
        "metrics": str(metrics_path),
        "log": str(log_path),
        "inputs": inputs,
        "guidance_sha256": {
            name: _sha256(ROOT / "src/task/CmResidual/docs/指导" / name)
            for name in ("V1.21.md", "V1.21a.md", "V1.21c.md")
        },
        "plan_sha256": _sha256(ROOT / "src/task/CmResidual/docs/plan/V1.21c.md"),
        "gpu_preflight_mib": used_mib,
        "protocol": config,
        "command_template": _command(5909, episodes_dir / "{episode_id}"),
        "conclusion": "N/A",
        "scientific_conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONPATH"] = os.pathsep.join((
        str((DEXPLORE_ROOT / "dexplore").resolve()),
        str(ISAAC_GYM_PYTHON.resolve()),
        str(ROOT),
        environment.get("PYTHONPATH", ""),
    ))
    rows: list[dict[str, object]] = []
    selected = None
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log, metrics_path.open(
            "w", encoding="utf-8", buffering=1
        ) as metrics:
            for episode_id in range(MAX_EPISODES):
                seed = 5909 + episode_id
                episode_output = episodes_dir / f"{episode_id:04d}"
                command = _command(seed, episode_output)
                sim_identity = json.dumps(
                    {"seed": seed, "num_envs": 1, "control_hz": 30, "motion": str(MOTION_ROOT)},
                    sort_keys=True,
                ).encode("utf-8")
                episode_environment = environment.copy()
                episode_environment.update({
                    "REF2DEX_V121C_EPISODE_OUTPUT": str(episode_output),
                    "REF2DEX_V121C_COLLECTION_BATCH_ID": batch_id,
                    "REF2DEX_V121C_EPISODE_ID": str(episode_id),
                    "REF2DEX_V121C_EPISODE_SEED": str(seed),
                    "REF2DEX_V121C_SIM_CONFIG_SHA256": hashlib.sha256(sim_identity).hexdigest(),
                })
                log.write(f"episode_id={episode_id} seed={seed} command={shlex.join(command)}\n")
                log.flush()
                subprocess.run(
                    command,
                    cwd=DEXPLORE_ROOT,
                    env=episode_environment,
                    check=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                summary = json.loads((episode_output / "summary.json").read_text(encoding="utf-8"))
                rows.extend(_load_rows(
                    episode_output / "active_states.npz", episode_id, seed, batch_id
                ))
                phase_counts = {
                    str(phase): sum(int(row["phase_id"]) == phase for row in rows)
                    for phase in CALIBRATION_QUOTAS
                }
                metric = {**summary, "cumulative_phase_counts": phase_counts}
                metrics.write(json.dumps(metric, sort_keys=True) + "\n")
                metrics.flush()
                selected = _select(rows)
                if selected is not None:
                    break
        if selected is None:
            missing = {
                str(phase): quota - sum(int(row["phase_id"]) == phase for row in rows)
                for phase, quota in CALIBRATION_QUOTAS.items()
                if sum(int(row["phase_id"]) == phase for row in rows) < quota
            }
            raise RuntimeError(f"collection insufficient after {MAX_EPISODES} episodes: {missing}")
        _save_selection(output / "selected_states.npz", selected, batch_id)
        final_counts = {
            str(phase): sum(int(row["phase_id"]) == phase for row in selected)
            for phase in CALIBRATION_QUOTAS
        }
        manifest.update({
            "run_status": "COMPLETED",
            "completed_at": _now(),
            "episodes_used": 1 + max(int(row["episode_id"]) for row in selected),
            "active_states_seen": len(rows),
            "selected_state_count": len(selected),
            "selected_phase_counts": final_counts,
            "selected_states": str(output / "selected_states.npz"),
            "conclusion": "N/A",
        })
        _write_json(manifest_path, manifest, overwrite=True)
        print(json.dumps({
            "run_id": args.run_id,
            "run_status": "COMPLETED",
            "episodes_used": manifest["episodes_used"],
            "active_states_seen": len(rows),
            "selected_phase_counts": final_counts,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        manifest.update({
            "run_status": "FAILED",
            "completed_at": _now(),
            "failure_reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "scientific_conclusion": "INCONCLUSIVE",
        })
        _write_json(manifest_path, manifest, overwrite=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
