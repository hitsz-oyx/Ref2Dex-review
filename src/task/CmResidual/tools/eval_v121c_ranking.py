"""Evaluate V1.21c ranking metrics from a validated replay ``.npz`` artifact.

This runner is deliberately offline: it never creates an Isaac Gym environment and
never changes a checkpoint.  A future collector can write the same state/physics
schema and invoke this tool after the independent duplicate-anchor and input gates
have passed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.CmResidual.v121c_artifacts import (  # noqa: E402
    build_manifest,
    sha256_file,
    validate_physics_records,
    validate_state_records,
    write_json,
)
from src.task.CmResidual.v121c_ranking import (  # noqa: E402
    episode_block_bootstrap,
    pairwise_metrics,
    ranking_eligible,
    top1_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="validated state+physics .npz produced by a V1.21c collector")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="new outputs/CmResidual/<run_id> directory")
    parser.add_argument("--epsilon-physx", type=float, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=2021)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _state_and_physics(payload: np.lib.npyio.NpzFile) -> tuple[dict, dict]:
    state_names = {
        "raw_obs", "native_q", "native_dq", "object_pose", "object_twist", "contact_force",
        "policy_mu", "policy_sigma", "candidate_actions", "candidate_valid", "cm_delta_xi",
        "cm_valid", "cm_token_mask", "cm_token_mass", "cm_score", "state_id",
        "collection_batch_id", "episode_id", "seed", "frame_id", "reference_index",
        "checkpoint_sha256", "motion_source_sha", "obs_rms_snapshot_sha256", "active_reason_mask",
        "phase_id", "candidate_seed",
    }
    physics_names = {
        "physics_clone_valid", "physics_candidate_valid", "physics_score",
        "physics_next_object_pose", "physics_next_IG", "duplicate_delta_object_pose",
        "duplicate_delta_IG", "duplicate_delta_score",
    }
    state = {name: payload[name] for name in state_names if name in payload.files}
    physics = {name: payload[name] for name in physics_names if name in payload.files}
    state_count = validate_state_records(state)
    validate_physics_records(physics, state_count)
    return state, physics


def main() -> int:
    args = _parse_args()
    input_path = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if args.bootstrap_draws <= 0 or not np.isfinite(args.epsilon_physx) or args.epsilon_physx < 1e-6:
        raise ValueError("bootstrap-draws must be positive and epsilon-physx must be >= 1e-6")
    with np.load(input_path, allow_pickle=False) as payload:
        state, physics = _state_and_physics(payload)

    cm_score = torch.from_numpy(np.asarray(state["cm_score"], dtype=np.float32))
    physics_score = torch.from_numpy(np.asarray(physics["physics_score"], dtype=np.float32))
    cm_valid = torch.from_numpy(np.asarray(state["cm_valid"], dtype=np.bool_))
    physics_valid = torch.from_numpy(np.asarray(physics["physics_candidate_valid"], dtype=np.bool_))
    physics_clone_valid = torch.from_numpy(np.asarray(physics["physics_clone_valid"], dtype=np.bool_))
    pair = pairwise_metrics(cm_score, physics_score, cm_valid, physics_valid, args.epsilon_physx)
    top1 = top1_metrics(cm_score, physics_score, cm_valid, physics_valid, args.epsilon_physx)
    eligible = ranking_eligible(cm_valid) & physics_clone_valid & pair.valid_state
    episode_ids = np.asarray(state["episode_id"]).reshape(-1).astype(str)
    pair_values = pair.state_accuracy.detach().cpu().numpy()
    pair_bootstrap = episode_block_bootstrap(pair_values[eligible.cpu().numpy()], episode_ids[eligible.cpu().numpy()],
                                             draws=args.bootstrap_draws, seed=args.bootstrap_seed)
    top1_valid_tensor = top1.valid_state & physics_clone_valid
    success_values = top1.success.to(torch.float32).detach().cpu().numpy()
    success_valid = top1_valid_tensor.detach().cpu().numpy()
    success_bootstrap = episode_block_bootstrap(success_values[success_valid], episode_ids[success_valid],
                                                draws=args.bootstrap_draws, seed=args.bootstrap_seed + 1)
    improvement_values = top1.improvement.detach().cpu().numpy()
    improvement_values = np.where(success_valid, improvement_values, np.nan)
    improvement_valid = np.isfinite(improvement_values)
    improvement_bootstrap = episode_block_bootstrap(improvement_values[improvement_valid], episode_ids[improvement_valid],
                                                    draws=args.bootstrap_draws, seed=args.bootstrap_seed + 2)

    enough = (int(eligible.sum()) >= 384 and pair_bootstrap.episode_count >= 8 and
              success_bootstrap.episode_count >= 8 and improvement_bootstrap.episode_count >= 8)
    gate = {
        "n_rank_valid": int(eligible.sum()),
        "pairwise_accuracy": pair_bootstrap.estimate,
        "pairwise_ci_lower": pair_bootstrap.lower,
        "pairwise_pass": pair_bootstrap.estimate >= 0.60 and pair_bootstrap.lower > 0.50,
        "top1_success_rate": success_bootstrap.estimate,
        "top1_success_ci_lower": success_bootstrap.lower,
        "top1_pass": success_bootstrap.estimate >= 0.55 and success_bootstrap.lower > 0.50,
        "top1_mean_improvement": improvement_bootstrap.estimate,
        "top1_improvement_ci_lower": improvement_bootstrap.lower,
        "top1_improvement_pass": improvement_bootstrap.estimate > 0 and improvement_bootstrap.lower > 0,
        "enough_samples_and_episodes": enough,
    }
    if not enough:
        conclusion = "INCONCLUSIVE"
    elif all(gate[name] for name in ("pairwise_pass", "top1_pass", "top1_improvement_pass")):
        conclusion = "SUPPORTED"
    else:
        conclusion = "REFUTED"

    output.mkdir(parents=True)
    (output / "replay").mkdir()
    (output / "logs").mkdir()
    np.savez_compressed(output / "replay" / "state_records.npz", **state)
    np.savez_compressed(output / "replay" / "physics_records.npz", **physics)
    config = {
        "schema": "cmv2_native_action_ranking_v121c",
        "run_id": args.run_id,
        "input": str(input_path),
        "epsilon_physx": args.epsilon_physx,
        "bootstrap_draws": args.bootstrap_draws,
        "bootstrap_seed": args.bootstrap_seed,
    }
    write_json(output / "config.json", config)
    rows = []
    selected = top1.selected_index.detach().cpu().tolist()
    for index, state_id in enumerate(np.asarray(state["state_id"]).reshape(-1).astype(str)):
        rows.append({
            "state_id": state_id,
            "episode_id": str(episode_ids[index]),
            "phase_id": int(np.asarray(state["phase_id"]).reshape(-1)[index]),
            "pairwise_state_accuracy": None if not np.isfinite(pair_values[index]) else float(pair_values[index]),
            "pair_count": int(pair.pair_count[index]),
            "rank_eligible": bool(eligible[index]),
            "cm_top1_index": int(selected[index]),
            "top1_valid": bool(top1_valid_tensor[index]),
            "top1_success": bool(top1.success[index] and top1_valid_tensor[index]),
            "top1_improvement": None if not np.isfinite(improvement_values[index]) else float(improvement_values[index]),
        })
    _write_jsonl(output / "metrics.jsonl", rows)
    (output / "logs" / "eval.log").write_text(
        json.dumps({"run_id": args.run_id, "conclusion": conclusion, "gate": gate},
                   ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    guidance_dir = ROOT / "src/task/CmResidual/docs/指导"
    guidance = {name: sha256_file(guidance_dir / name) for name in ("V1.21.md", "V1.21a.md", "V1.21c.md")
                if (guidance_dir / name).is_file()}
    plan = ROOT / "src/task/CmResidual/docs/plan/V1.21c.md"
    manifest = build_manifest(
        run_id=args.run_id,
        output_dir=output,
        git_commit=_git_commit(),
        guidance_sha256=guidance,
        plan_sha256=sha256_file(plan),
        inputs={"replay_input": {"path": str(input_path), "sha256": sha256_file(input_path)}},
        protocol={"mode": "offline_metrics", "epsilon_physx": args.epsilon_physx,
                  "candidate_count": 8, "bootstrap_draws": args.bootstrap_draws,
                  "bootstrap_seed": args.bootstrap_seed},
        command=" ".join(shlex.quote(value) for value in [sys.executable, *sys.argv]),
    )
    manifest.update({"run_status": "COMPLETED", "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "gate": gate, "conclusion": conclusion, "scientific_conclusion": conclusion})
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"run_id": args.run_id, "conclusion": conclusion, "gate": gate}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
