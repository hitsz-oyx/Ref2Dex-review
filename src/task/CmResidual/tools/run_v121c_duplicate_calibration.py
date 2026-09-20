"""Run the frozen 64-state V1.21c duplicate-noise calibration."""
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

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.CmResidual.v121c_ranking import calibrate_duplicate_anchor, candidate_seed
from src.task.CmResidual.v121c_artifacts import validate_episode_replay

DEXPLORE_ROOT = ROOT / "third_party/DExplore"
ISAAC_GYM_PYTHON = Path(os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python"))
DEXPLORE_PYTHON = Path(os.environ.get(
    "REF2DEX_DEXPLORE_PYTHON",
    "/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python",
))
BOOTSTRAP = Path(__file__).resolve().with_name("v121c_calibration_replay_bootstrap.py")
MOTION_ROOT = ROOT / "data/processed_data/dexplore_reconstructed_v120_coordfix_v4/converted_attempt1"
OUTPUT_ROOT = ROOT / "outputs/CmResidual"
PRIOR_FAILED_RUN = OUTPUT_ROOT / "cmresidual_v121c_duplicate_calibration_gpu3_20260920_2000"
GPU_CAPACITY_GATE_MIB = 1024
CALIBRATION_PHASE_COUNTS = {0: 24, 1: 24, 2: 16}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_identity() -> dict[str, str]:
    if not DEXPLORE_PYTHON.exists() or not os.access(DEXPLORE_PYTHON, os.X_OK):
        raise FileNotFoundError(f"DExplore Python is missing or not executable: {DEXPLORE_PYTHON}")
    resolved = DEXPLORE_PYTHON.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "executable": str(DEXPLORE_PYTHON),
        "resolved_target": str(resolved),
        "resolved_target_sha256": _sha256(resolved),
    }


def _document_identity() -> dict[str, dict[str, str]]:
    paths = {
        "guidance_v121": ROOT / "src/task/CmResidual/docs/指导/V1.21.md",
        "guidance_v121a": ROOT / "src/task/CmResidual/docs/指导/V1.21a.md",
        "guidance_v121c": ROOT / "src/task/CmResidual/docs/指导/V1.21c.md",
        "guidance_v121d": ROOT / "src/task/CmResidual/docs/指导/V1.21d.md",
        "plan_v121c": ROOT / "src/task/CmResidual/docs/plan/V1.21c.md",
        "plan_v121d": ROOT / "src/task/CmResidual/docs/plan/V1.21d.md",
    }
    result = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    return result


def _write(path: Path, payload: object, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _gpu_used_mib(gpu: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    values = {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.splitlines()}
    if gpu not in values:
        raise RuntimeError(f"nvidia-smi did not report GPU{gpu}")
    return values[gpu]


def _validate_collection(root: Path) -> tuple[dict, np.ndarray]:
    root = root.resolve()
    manifest_path = root / "run_manifest.json"
    config_path = root / "config.json"
    selected_path = root / "selected_states.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if manifest.get("run_status") != "COMPLETED" or manifest.get("selected_state_count") != 64:
        raise ValueError("calibration collection must be COMPLETED with 64 selected states")
    if Path(manifest.get("config", "")).expanduser().resolve() != config_path:
        raise ValueError("collection manifest config path mismatch")
    if Path(manifest.get("selected_states", "")).expanduser().resolve() != selected_path:
        raise ValueError("collection manifest selected_states path mismatch")
    batch_id = config.get("collection_batch_id")
    if not batch_id or manifest.get("protocol", {}).get("collection_batch_id") != batch_id:
        raise ValueError("collection_batch_id must agree between config and manifest protocol")
    with np.load(selected_path, allow_pickle=False) as selected:
        required = {"state_id", "collection_batch_id", "episode_id", "seed", "frame_id",
                    "reference_index", "progress", "phase_id", "candidate_seed",
                    "executed_action_prefix_sha256", "raw_obs", "policy_mu", "policy_sigma"}
        if required - set(selected.files):
            raise ValueError(f"selected states missing fields: {sorted(required - set(selected.files))}")
        if selected["candidate_seed"].dtype != np.uint64:
            raise ValueError("selected candidate_seed must be uint64")
        if selected["state_id"].shape != (64,):
            raise ValueError("selected states must contain 64 rows")
        for name, shape in (("raw_obs", (64, 1442)), ("policy_mu", (64, 18)),
                            ("policy_sigma", (64, 18))):
            value = np.asarray(selected[name])
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"selected {name} must be finite float32 with shape {shape}")
        if (np.asarray(selected["policy_sigma"]) < 0).any():
            raise ValueError("selected policy_sigma must be non-negative")
        for name in ("episode_id", "seed", "frame_id", "reference_index", "progress", "phase_id"):
            value = np.asarray(selected[name])
            if value.shape != (64,) or value.dtype.kind not in "iu":
                raise ValueError(f"selected {name} must be an integer vector with 64 rows")
        state_ids = [str(value) for value in selected["state_id"].tolist()]
        if any(not value for value in state_ids) or len(set(state_ids)) != 64:
            raise ValueError("selected state_id must contain 64 unique non-empty values")
        batches = [str(value) for value in selected["collection_batch_id"].tolist()]
        if any(not value for value in batches) or set(batches) != {str(batch_id)}:
            raise ValueError("selected states must share the manifest collection_batch_id")
        prefix_hashes = [str(value) for value in selected["executed_action_prefix_sha256"].tolist()]
        if any(len(value) != 64 for value in prefix_hashes):
            raise ValueError("selected action-prefix identity must be SHA256")
        phases = np.asarray(selected["phase_id"], dtype=np.int64)
        if not np.isin(phases, list(CALIBRATION_PHASE_COUNTS)).all():
            raise ValueError("selected phase_id contains an unknown phase")
        counts = {phase: int(np.count_nonzero(phases == phase)) for phase in CALIBRATION_PHASE_COUNTS}
        if counts != CALIBRATION_PHASE_COUNTS:
            raise ValueError(f"selected phase composition must be 24/24/16, got {counts}")
        manifest_counts = {
            int(key): int(value) for key, value in manifest.get("selected_phase_counts", {}).items()
        }
        if manifest_counts != counts:
            raise ValueError("manifest selected_phase_counts does not match selected_states.npz")
        for batch, state, stored in zip(
            selected["collection_batch_id"], selected["state_id"], selected["candidate_seed"]
        ):
            if candidate_seed(str(batch), str(state)) != int(stored):
                raise ValueError(f"candidate_seed mismatch for {state}")
        episodes = np.unique(selected["episode_id"]).astype(np.int64)
        selected_rows = {name: np.asarray(selected[name]).copy() for name in required}
    for episode_id in episodes.tolist():
        episode_path = root / f"episodes/{episode_id:04d}/episode.npz"
        if not episode_path.is_file():
            raise ValueError(f"missing episode artifact: {episode_path}")
        with np.load(episode_path, allow_pickle=False) as source:
            episode = {name: source[name] for name in source.files}
        horizon = validate_episode_replay(episode)
        artifact_episode = int(np.asarray(episode["episode_id"]).reshape(-1)[0])
        artifact_seed = int(np.asarray(episode["seed"]).reshape(-1)[0])
        if artifact_episode != int(episode_id):
            raise ValueError(f"episode identity mismatch for {episode_path}")
        row_indices = np.flatnonzero(selected_rows["episode_id"] == episode_id)
        for index in row_indices.tolist():
            frame = int(selected_rows["frame_id"][index])
            if frame < 0 or frame >= horizon:
                raise ValueError(f"selected frame_id is outside episode {episode_id}: {frame}")
            if int(selected_rows["seed"][index]) != artifact_seed:
                raise ValueError(f"selected seed mismatch for episode {episode_id}")
            if int(selected_rows["progress"][index]) != int(episode["progress_history"][frame]):
                raise ValueError(f"selected progress mismatch for episode {episode_id} frame {frame}")
            if int(selected_rows["reference_index"][index]) != int(episode["reference_index"][frame]):
                raise ValueError(f"selected reference index mismatch for episode {episode_id} frame {frame}")
            actions = np.asarray(episode["executed_action_history"][:frame], dtype="<f4")
            prefix_sha = hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest()
            if str(selected_rows["executed_action_prefix_sha256"][index]) != prefix_sha:
                raise ValueError(f"executed action-prefix SHA mismatch for episode {episode_id} frame {frame}")
    return manifest, episodes


def _command(seed: int) -> list[str]:
    return [
        str(DEXPLORE_PYTHON), str(BOOTSTRAP),
        "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(MOTION_ROOT),
        "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
        "--graphics_device_id", "0", "--num_envs", "9",
        "--episode_length", "432", "--seed", str(seed),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    collection = args.collection.expanduser().resolve()
    collection_manifest, episodes = _validate_collection(collection)
    runtime_identity = _runtime_identity()
    document_identity = _document_identity()
    output = (OUTPUT_ROOT / args.run_id).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    used_mib = _gpu_used_mib(args.gpu)
    if used_mib > GPU_CAPACITY_GATE_MIB:
        raise RuntimeError(f"GPU{args.gpu} capacity gate failed: {used_mib} MiB used")
    episode_seeds = {int(ep): 5909 + int(ep) for ep in episodes.tolist()}
    if args.dry_run:
        print(json.dumps({
            "preflight": "passed", "git_commit": _git("rev-parse", "HEAD"),
            "physical_gpu": args.gpu, "gpu_used_mib": used_mib,
            "collection_run_id": collection_manifest["run_id"],
            "episodes": episode_seeds,
            "dexplore_runtime": runtime_identity,
            "documents": document_identity,
        }, sort_keys=True))
        return 0

    output.mkdir(parents=True)
    episode_root = output / "episodes"
    episode_root.mkdir()
    log_path = output / "logs/eval.log"
    log_path.parent.mkdir()
    manifest_path = output / "run_manifest.json"
    metrics_path = output / "metrics.jsonl"
    config = {
        "collection": str(collection),
        "collection_run_id": collection_manifest["run_id"],
        "episode_seeds": episode_seeds,
        "physical_gpu": args.gpu,
        "logical_gpu": 0,
        "num_envs": 9,
        "control_hz": 30,
        "physics_substeps_per_action": 2,
        "env_assignment": "PCG64(candidate_seed) permutation of [0,0,1,2,3,4,5,6,7]",
        "position_ceiling_m": 5e-4,
        "rotation_ceiling_rad": 5e-3,
        "epsilon_formula": "max(5*p99(abs(duplicate_score_delta)),1e-6)",
        "score_producer": "v121c_duplicate_calibration_next_cost.v1",
        "object_goal_offset": 6,
        "ig_reference_offset": 1,
        "score_baseline": "shared_canonical_collected_pre_action_cancelled",
        "dexplore_runtime": runtime_identity,
        "documents": document_identity,
        "prior_failed_run": str(PRIOR_FAILED_RUN),
    }
    _write(output / "config.json", config)
    manifest = {
        "manifest_schema": "ref2dex.cmresidual.v121c.duplicate_calibration.v1",
        "created_at": _now(), "task": "CmResidual", "work_version": "V1.21",
        "run_id": args.run_id, "run_status": "STARTED",
        "git_commit": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current"),
        "output_dir": str(output), "config": str(output / "config.json"),
        "log": str(log_path), "metrics": str(metrics_path), "protocol": config,
        "collection_manifest": str(collection / "run_manifest.json"),
        "collection_manifest_sha256": _sha256(collection / "run_manifest.json"),
        "collection_git_commit": collection_manifest["git_commit"],
        "dexplore_runtime": runtime_identity,
        "documents": document_identity,
        "prior_failed_run": str(PRIOR_FAILED_RUN),
        "gpu_preflight_mib": used_mib, "conclusion": "N/A",
        "scientific_conclusion": "INCONCLUSIVE",
    }
    _write(manifest_path, manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["PYTHONPATH"] = os.pathsep.join((
        str((DEXPLORE_ROOT / "dexplore").resolve()), str(ISAAC_GYM_PYTHON.resolve()),
        str(ROOT), environment.get("PYTHONPATH", ""),
    ))
    rows = []
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log, metrics_path.open(
            "w", encoding="utf-8", buffering=1
        ) as metrics:
            for episode_id, seed in episode_seeds.items():
                episode_output = episode_root / f"{episode_id:04d}"
                episode_environment = environment.copy()
                episode_environment.update({
                    "REF2DEX_V121C_COLLECTION_ROOT": str(collection),
                    "REF2DEX_V121C_CALIBRATION_EPISODE_OUTPUT": str(episode_output),
                    "REF2DEX_V121C_EPISODE_ID": str(episode_id),
                })
                command = _command(seed)
                log.write(f"episode_id={episode_id} command={shlex.join(command)}\n")
                log.flush()
                subprocess.run(command, cwd=DEXPLORE_ROOT, env=episode_environment, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
                summary = json.loads((episode_output / "summary.json").read_text(encoding="utf-8"))
                metrics.write(json.dumps(summary, sort_keys=True) + "\n")
                metrics.flush()
                rows.extend(json.loads(line) for line in
                            (episode_output / "records.jsonl").read_text(encoding="utf-8").splitlines())
        if len(rows) != 64 or len({row["state_id"] for row in rows}) != 64:
            raise RuntimeError(f"calibration replay produced {len(rows)} non-unique rows")
        calibration = calibrate_duplicate_anchor(
            [row["duplicate_position_m"] for row in rows],
            [row["duplicate_rotation_rad"] for row in rows],
            [row["duplicate_score_delta"] for row in rows],
        )
        (output / "calibration_records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        summary = {
            **calibration,
            "state_count": len(rows),
            "numeric_parity_pass_count": sum(row["numeric_parity_valid"] for row in rows),
            "task_indices_equal_count": sum(row["task_indices_equal"] for row in rows),
            "duplicate_valid_count": sum(row["duplicate_valid"] for row in rows),
        }
        _write(output / "calibration_summary.json", summary)
        manifest.update({
            "run_status": "COMPLETED", "completed_at": _now(),
            "calibration_records": str(output / "calibration_records.jsonl"),
            "calibration_summary": str(output / "calibration_summary.json"),
            "result_metrics": summary, "epsilon_physx": calibration["epsilon_physx"],
        })
        _write(manifest_path, manifest, overwrite=True)
        print(json.dumps({"run_id": args.run_id, "run_status": "COMPLETED", **summary}, sort_keys=True))
        return 0
    except BaseException as error:
        manifest.update({
            "run_status": "FAILED", "completed_at": _now(),
            "failure_reason": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "conclusion": "INVALID_IMPLEMENTATION",
        })
        _write(manifest_path, manifest, overwrite=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
