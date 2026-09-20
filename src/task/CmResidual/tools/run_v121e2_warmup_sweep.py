"""Run the V1.21e.2 canonical-snapshot warm-up parity sweep."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from src.task.CmResidual.tools.run_v121c_calibration_collection import (
    DEXPLORE_PYTHON, DEXPLORE_ROOT, ISAAC_GYM_PYTHON, MOTION_ROOT, OUTPUT_ROOT,
    _gpu_used_mib, _input_identity, _sha256, _write_json,
)
from src.task.CmResidual.v121e_snapshot import PHASE_QUOTAS
from src.task.CmResidual.v121e2_warmup import (
    METHODS, WINDOWS, canonical_snapshot_sha256, classify_sweep, evaluate_state,
)

PLAN = ROOT / "src/task/CmResidual/docs/plan/V1.21e.2.md"
GENERATOR = Path(__file__).with_name("v121e2_snapshot_generation_bootstrap.py")
ARM_BOOTSTRAP = Path(__file__).with_name("v121e2_warmup_parity_bootstrap.py")
E1_RUN = OUTPUT_ROOT / "cmresidual_v121e1_independent_parity_gpu3_20260920_2225"
EPISODE_RUN = OUTPUT_ROOT / "cmresidual_v121e_snapshot_parity_gpu3_20260920_2130"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sim_command(bootstrap: Path, seed: int) -> list[str]:
    return [
        str(DEXPLORE_PYTHON), str(bootstrap.resolve()), "--test", "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(MOTION_ROOT), "--num_envs", "1", "--seed", str(seed),
        "--headless", "--sim_device", "cuda:0", "--rl_device", "cuda:0",
        "--graphics_device_id", "0",
    ]


def _arm_specs():
    return tuple((method, repeat) for method in ("full", "0", "1", "2", "4", "8")
                 for repeat in (1, 2))


def _load_selected(path: Path) -> list[dict[str, object]]:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "state_id", "frame_id", "phase_id", "episode_id", "seed", "policy_mu",
            "snapshot_actor_root_state", "canonical_current_ig", "progress",
        }
        if missing := required - set(payload.files):
            raise ValueError(f"V1.21e.1 selected artifact missing {sorted(missing)}")
        rows = [{name: payload[name][index].copy() if hasattr(payload[name][index], "copy")
                 else payload[name][index] for name in payload.files}
                for index in range(len(payload["state_id"]))]
    if len(rows) != 6 or len({str(row["state_id"]) for row in rows}) != 6:
        raise ValueError("V1.21e.2 requires the six unique V1.21e.1 states")
    phase_counts = {phase: sum(int(row["phase_id"]) == phase for row in rows) for phase in PHASE_QUOTAS}
    if phase_counts != PHASE_QUOTAS:
        raise ValueError(f"V1.21e.1 phase composition mismatch: {phase_counts}")
    if any(int(row["frame_id"]) < max(WINDOWS) for row in rows):
        raise ValueError("selected state is earlier than the maximum warm-up window")
    return rows


def _source_inputs() -> tuple[list[dict[str, object]], dict[str, object], dict[int, Path]]:
    e1_manifest_path = E1_RUN / "run_manifest.json"
    selected_path = E1_RUN / "selected_states.npz"
    manifest = json.loads(e1_manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("run_status"), manifest.get("conclusion"), manifest.get("run_id")) != (
            "COMPLETED", "REFUTED", E1_RUN.name):
        raise ValueError("V1.21e.1 source run identity/status/conclusion mismatch")
    rows = _load_selected(selected_path)
    episodes = {}
    episode_identity = {}
    for episode_id in sorted({int(row["episode_id"]) for row in rows}):
        path = EPISODE_RUN / f"episodes/{episode_id:04d}/episode.npz"
        with np.load(path, allow_pickle=False) as payload:
            required = {"seed", "initial_dof_state", "initial_actor_root_state",
                        "initial_task_indices", "executed_action_history"}
            if missing := required - set(payload.files):
                raise ValueError(f"episode {episode_id} missing {sorted(missing)}")
            actions = np.asarray(payload["executed_action_history"], dtype="<f4")
            seed = int(np.asarray(payload["seed"]).reshape(-1)[0])
        if any(int(row["frame_id"]) > len(actions) for row in rows if int(row["episode_id"]) == episode_id):
            raise ValueError(f"episode {episode_id} is shorter than a selected target")
        episodes[episode_id] = path
        import hashlib
        episode_identity[str(episode_id)] = {
            "path": str(path.resolve()), "sha256": _sha256(path), "seed": seed,
            "action_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
            "action_count": len(actions),
        }
    identity = {
        "v121e1_run": E1_RUN.name,
        "v121e1_manifest": {"path": str(e1_manifest_path.resolve()), "sha256": _sha256(e1_manifest_path)},
        "selected_states": {"path": str(selected_path.resolve()), "sha256": _sha256(selected_path)},
        "episodes": episode_identity,
    }
    return rows, identity, episodes


def _save_rows(path: Path, rows: list[dict[str, object]]) -> None:
    payload = {}
    for name in rows[0]:
        values = [row[name] for row in rows]
        payload[name] = np.stack(values) if np.asarray(values[0]).ndim else np.asarray(values)
    np.savez_compressed(path, **payload)


def _merge_snapshots(paths: list[Path], selected: list[dict[str, object]], output: Path) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as payload:
            rows.extend({name: payload[name][index].copy() if hasattr(payload[name][index], "copy")
                         else payload[name][index] for name in payload.files}
                        for index in range(len(payload["state_id"])))
    expected = {(str(state["state_id"]), window) for state in selected for window in WINDOWS}
    keys = [(str(row["state_id"]), int(row["window"])) for row in rows]
    if len(rows) != 30 or len(set(keys)) != 30 or set(keys) != expected:
        raise ValueError("canonical generation did not produce exactly 30 unique snapshots")
    frame_by_state = {str(state["state_id"]): int(state["frame_id"]) for state in selected}
    for row in rows:
        state_id, window = str(row["state_id"]), int(row["window"])
        if int(row["frame_id"]) != frame_by_state[state_id] - window:
            raise ValueError("canonical snapshot frame identity mismatch")
        computed = canonical_snapshot_sha256(
            state_id, window, row["snapshot_dof_state"], row["snapshot_actor_root_state"],
            row["snapshot_task_indices"], int(row["snapshot_reset_buf"]),
            int(row["snapshot_terminate_buf"]), row["snapshot_contact_reset"], row["snapshot_ig"],
        )
        if str(row["snapshot_sha256"]) != computed:
            raise ValueError("canonical snapshot content hash mismatch")
    rows.sort(key=lambda row: (str(row["state_id"]), int(row["window"])))
    _save_rows(output, rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = (OUTPUT_ROOT / args.run_id).resolve()
    if output.exists():
        raise FileExistsError(output)
    selected, source_identity, episodes = _source_inputs()
    inputs = _input_identity()
    inputs.update({
        "snapshot_generator": {"path": str(GENERATOR.resolve()), "sha256": _sha256(GENERATOR)},
        "warmup_arm_bootstrap": {"path": str(ARM_BOOTSTRAP.resolve()), "sha256": _sha256(ARM_BOOTSTRAP)},
        "source_v121e1": source_identity,
    })
    used_mib = _gpu_used_mib(args.gpu)
    if used_mib > 1024:
        raise RuntimeError(f"GPU{args.gpu} capacity gate failed: {used_mib} MiB used")
    if _git("status", "--porcelain"):
        raise RuntimeError("formal V1.21e.2 run requires a clean worktree")
    if args.dry_run:
        print(json.dumps({
            "preflight": "passed", "git_commit": _git("rev-parse", "HEAD"), "gpu": args.gpu,
            "gpu_used_mib": used_mib, "state_count": len(selected), "generation_passes": len(episodes),
            "parity_arms": len(selected) * len(_arm_specs()), "methods": METHODS, "inputs": inputs,
        }, sort_keys=True))
        return 0

    output.mkdir(parents=True)
    arms_dir = output / "arms"; arms_dir.mkdir()
    generation_dir = output / "canonical_generation"; generation_dir.mkdir()
    logs_dir = output / "logs"; logs_dir.mkdir()
    log_path = logs_dir / "eval.log"
    metrics_path = output / "metrics.jsonl"
    selected_path = output / "selected_states.npz"
    snapshots_path = output / "canonical_window_snapshots.npz"
    manifest_path = output / "run_manifest.json"
    _save_rows(selected_path, selected)
    config = {
        "methods": list(METHODS), "warmup_windows": list(WINDOWS), "states": 6,
        "generation_passes": len(episodes), "generation_passes_count_as_arms": False,
        "parity_arms": 72, "independent_processes_per_state_method": 2,
        "num_envs_per_process": 1, "control_hz": 30, "physics_substeps_per_action": 2,
        "action": "clip(actor_mean,-1,1)", "physical_gpu": args.gpu,
        "plan": str(PLAN.resolve()), "source_v121e1_run": str(E1_RUN.resolve()),
    }
    _write_json(output / "config.json", config)
    manifest = {
        "manifest_schema": "ref2dex.cmresidual.v121e2.warmup_sweep.v1",
        "created_at": _now(), "task": "CmResidual", "work_version": "V1.21",
        "run_id": args.run_id, "run_status": "STARTED", "git_commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"), "output_dir": str(output),
        "config": str(output / "config.json"), "metrics": str(metrics_path), "log": str(log_path),
        "inputs": inputs, "plan_sha256": _sha256(PLAN), "gpu_preflight_mib": used_mib,
        "scientific_conclusion": "INCONCLUSIVE",
    }
    _write_json(manifest_path, manifest)
    base_env = os.environ.copy()
    base_env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    base_env["PYTHONPATH"] = os.pathsep.join((str((DEXPLORE_ROOT / "dexplore").resolve()),
                                                str(ISAAC_GYM_PYTHON.resolve()), str(ROOT),
                                                base_env.get("PYTHONPATH", "")))
    try:
        generation_records = []
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            generated_paths = []
            for episode_id, episode_path in episodes.items():
                generation_output = generation_dir / f"episode_{episode_id:04d}.npz"
                command = _sim_command(GENERATOR, source_identity["episodes"][str(episode_id)]["seed"])
                run_env = base_env.copy()
                run_env.update({
                    "REF2DEX_V121E2_SELECTED": str(selected_path),
                    "REF2DEX_V121E2_EPISODE": str(episode_path),
                    "REF2DEX_V121E2_EPISODE_ID": str(episode_id),
                    "REF2DEX_V121E2_SNAPSHOT_OUTPUT": str(generation_output),
                })
                log.write(f"generation episode={episode_id} command={shlex.join(command)}\n")
                subprocess.run(command, cwd=DEXPLORE_ROOT, env=run_env, check=True,
                               stdout=log, stderr=subprocess.STDOUT)
                generated_paths.append(generation_output)
                generation_records.append({"episode_id": episode_id, "command": command,
                                           "output": str(generation_output)})
            snapshot_rows = _merge_snapshots(generated_paths, selected, snapshots_path)
            frozen_snapshot_file_sha = _sha256(snapshots_path)
            snapshots_path.chmod(0o444)
            manifest["canonical_snapshot_generation"] = {
                "git_commit": manifest["git_commit"], "passes": generation_records,
                "snapshot_file": str(snapshots_path), "snapshot_file_sha256": frozen_snapshot_file_sha,
                "snapshots": [{"state_id": str(row["state_id"]), "window": int(row["window"]),
                               "frame_id": int(row["frame_id"]),
                               "snapshot_sha256": str(row["snapshot_sha256"])} for row in snapshot_rows],
            }
            _write_json(manifest_path, manifest, overwrite=True)

            with metrics_path.open("w", encoding="utf-8", buffering=1) as metrics:
                results = []
                for index, state in enumerate(selected):
                    method_records = {method: [] for method in METHODS}
                    episode_id = int(state["episode_id"])
                    for method, repeat in _arm_specs():
                        arm_output = arms_dir / f"{index:02d}_{method}_{repeat}.json"
                        command = _sim_command(ARM_BOOTSTRAP, int(state["seed"]))
                        arm_env = base_env.copy()
                        arm_env.update({
                            "REF2DEX_V121E2_SELECTED": str(selected_path),
                            "REF2DEX_V121E2_SNAPSHOTS": str(snapshots_path),
                            "REF2DEX_V121E2_EPISODE": str(episodes[episode_id]),
                            "REF2DEX_V121E2_STATE_INDEX": str(index),
                            "REF2DEX_V121E2_METHOD": method,
                            "REF2DEX_V121E2_REPEAT": str(repeat),
                            "REF2DEX_V121E2_OUTPUT": str(arm_output),
                        })
                        log.write(f"state={index} method={method} repeat={repeat} command={shlex.join(command)}\n")
                        subprocess.run(command, cwd=DEXPLORE_ROOT, env=arm_env, check=True,
                                       stdout=log, stderr=subprocess.STDOUT)
                        if _sha256(snapshots_path) != frozen_snapshot_file_sha:
                            raise RuntimeError("canonical snapshot file changed after generation")
                        method_records[method].append(json.loads(arm_output.read_text(encoding="utf-8")))
                    result = evaluate_state(method_records)
                    result["phase_id"] = int(state["phase_id"])
                    results.append(result)
                    metrics.write(json.dumps(result, sort_keys=True) + "\n")
                    metrics.flush()
        classification = classify_sweep(results)
        summary = {
            "state_count": 6,
            "phase_counts": {str(phase): sum(int(result["phase_id"]) == phase for result in results)
                             for phase in PHASE_QUOTAS},
            **classification, "results": results,
        }
        _write_json(output / "summary.json", summary)
        scientific = classification["conclusion"] if classification["conclusion"] in {"SUPPORTED", "REFUTED"} else "INCONCLUSIVE"
        manifest.update({
            "run_status": "COMPLETED", "completed_at": _now(), "selected_states": str(selected_path),
            "summary": str(output / "summary.json"), "conclusion": classification["conclusion"],
            "conclusion_scope": classification["conclusion_scope"], "scientific_conclusion": scientific,
        })
        _write_json(manifest_path, manifest, overwrite=True)
        print(json.dumps({"run_id": args.run_id, "run_status": "COMPLETED", **classification}, sort_keys=True))
        return 0
    except BaseException as error:
        manifest.update({
            "run_status": "FAILED", "completed_at": _now(),
            "failure_reason": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
            "conclusion": "INVALID_IMPLEMENTATION", "scientific_conclusion": "INCONCLUSIVE",
        })
        _write_json(manifest_path, manifest, overwrite=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
