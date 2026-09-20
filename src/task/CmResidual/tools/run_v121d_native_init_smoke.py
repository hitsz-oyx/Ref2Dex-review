"""Launch the V1.21d native-init/fresh-simulator Gate 0 smoke."""
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


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task.CmResidual.tools.run_v121c_prefix_smoke import (  # noqa: E402
    GPU_CAPACITY_GATE_MIB,
    ISAAC_GYM_PYTHON,
    MOTION_ROOT,
    resolve_inputs,
)
from src.task.CmResidual.v121c_artifacts import sha256_file, write_json  # noqa: E402
from src.task.CmResidual.v121d_artifacts import build_v121d_manifest  # noqa: E402
from src.task.CmResidual.v121d_fresh_sim import (  # noqa: E402
    BACKEND_VERSION,
    BRANCH_BACKEND,
    POSITION_CEILING_M,
    ROTATION_CEILING_RAD,
)


DEXPLORE_ROOT = ROOT / "third_party" / "DExplore"
BOOTSTRAP = Path(__file__).resolve().with_name("v121d_native_init_smoke_bootstrap.py")
OUTPUT_ROOT = ROOT / "outputs" / "CmResidual"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _gpu_used_mib(gpu: int) -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = {}
    for row in output.splitlines():
        index, used = (item.strip() for item in row.split(","))
        values[int(index)] = int(used)
    if gpu not in values:
        raise RuntimeError(f"nvidia-smi did not report GPU{gpu}")
    return values[gpu]


def smoke_command() -> list[str]:
    return [
        sys.executable,
        str(BOOTSTRAP),
        "--task", "Dexplore_Inspire",
        "--cfg_env", "dexplore/data/cfg/inspire.yaml",
        "--cfg_train", "dexplore/data/cfg/train/rlg/inspire.yaml",
        "--motion_file", str(MOTION_ROOT.resolve()),
        "--headless",
        "--sim_device", "cuda:0",
        "--rl_device", "cuda:0",
        "--graphics_device_id", "0",
        "--num_envs", "9",
        "--episode_length", "8",
        "--seed", "5909",
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required for a real PhysX run; dry-run does not require it",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.dry_run and not args.execute:
        raise RuntimeError("real V1.21d smoke requires explicit --execute")
    output = (OUTPUT_ROOT / args.run_id).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    inputs = resolve_inputs()
    gpu_preflight_mib = _gpu_used_mib(args.physical_gpu)
    if gpu_preflight_mib > GPU_CAPACITY_GATE_MIB:
        raise RuntimeError(
            f"GPU{args.physical_gpu} capacity gate failed: {gpu_preflight_mib} MiB used"
        )
    command = smoke_command()
    protocol = {
        "mode": "v121d_native_init_fresh_sim_gate0",
        "branch_backend": BRANCH_BACKEND,
        "backend_version": BACKEND_VERSION,
        "physical_gpu": args.physical_gpu,
        "logical_gpu": 0,
        "source_envs": 1,
        "validation_envs": 9,
        "source_states": 1,
        "prefix_steps": 1,
        "candidate_count": 8,
        "duplicate_candidate": 0,
        "state_init_mode": "Start",
        "hybrid_init_prob": 1.0,
        "public_state_parity_tolerance": 1e-5,
        "position_ceiling_m": POSITION_CEILING_M,
        "rotation_ceiling_rad": ROTATION_CEILING_RAD,
        "seed": 5909,
        "sequence": "s1_airplane_lift",
        "research_scope": "engineering Gate 0 only; no PPO and no ranking conclusion",
    }
    preview = {
        "preflight": "passed",
        "command": command,
        "inputs": inputs,
        "protocol": protocol,
        "gpu_preflight_mib": gpu_preflight_mib,
    }
    if args.dry_run:
        print(json.dumps(preview, ensure_ascii=False, sort_keys=True))
        return 0

    output.mkdir(parents=True)
    result_output = output / "replay_smoke"
    log_path = output / "logs" / "eval.log"
    log_path.parent.mkdir()
    guidance_dir = ROOT / "src/task/CmResidual/docs/指导"
    guidance = {
        name: sha256_file(guidance_dir / name)
        for name in ("V1.21.md", "V1.21a.md", "V1.21c.md")
    }
    plan = ROOT / "src/task/CmResidual/docs/plan/V1.21d.md"
    config = {
        "command": command,
        "runtime": {
            "python": sys.executable,
            "physical_gpu": args.physical_gpu,
            "cuda_visible_devices": str(args.physical_gpu),
            "gpu_preflight_mib": gpu_preflight_mib,
        },
        "protocol": protocol,
        "resolved_inputs": inputs,
    }
    write_json(output / "config.json", config)
    manifest = build_v121d_manifest(
        run_id=args.run_id,
        output_dir=output,
        git_commit=_git_commit(),
        guidance_sha256=guidance,
        plan_sha256=sha256_file(plan),
        inputs=inputs,
        protocol=protocol,
        command=" ".join(shlex.quote(value) for value in command),
    )
    manifest.update(
        {
            "config": str(output / "config.json"),
            "log": str(log_path),
            "result": str(result_output / "smoke_result.json"),
            "run_status": "STARTED",
            "scientific_conclusion": "INCONCLUSIVE",
        }
    )
    manifest_path = output / "run_manifest.json"
    write_json(manifest_path, manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str((DEXPLORE_ROOT / "dexplore").resolve()),
            str(ISAAC_GYM_PYTHON.resolve()),
            str(ROOT),
            environment.get("PYTHONPATH", ""),
        )
    )
    environment["REF2DEX_V121D_SMOKE_OUTPUT"] = str(result_output)
    environment["REF2DEX_V121D_MOTION_SHA256"] = inputs["motion_tensor"]["sha256"]
    try:
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            subprocess.run(
                command,
                cwd=DEXPLORE_ROOT,
                env=environment,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result = json.loads((result_output / "smoke_result.json").read_text(encoding="utf-8"))
        required = (
            "native_initial_parity",
            "pre_candidate_parity",
            "sim_destroy_ok",
            "duplicate_valid",
        )
        if not all(result.get(name) for name in required):
            raise RuntimeError("bootstrap returned a failed V1.21d Gate 0")
        manifest.update(
            {
                "run_status": "COMPLETED",
                "completed_at": _now(),
                "result_metrics": result,
                "conclusion": "N/A",
                "scientific_conclusion": "INCONCLUSIVE",
            }
        )
        write_json(manifest_path, manifest, overwrite=True)
        print(json.dumps({"run_id": args.run_id, "run_status": "COMPLETED", **result}))
        return 0
    except BaseException as error:
        manifest.update(
            {
                "run_status": "FAILED",
                "completed_at": _now(),
                "failure_reason": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "conclusion": "INVALID_IMPLEMENTATION",
                "scientific_conclusion": "INCONCLUSIVE",
            }
        )
        write_json(manifest_path, manifest, overwrite=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
