"""Launch the approved V1.21c 1-state/9-env PhysX duplicate smoke on GPU3."""
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

from src.task.CmResidual.v121c_artifacts import build_manifest, sha256_file, write_json


DEXPLORE_ROOT = ROOT / "third_party" / "DExplore"
ISAAC_GYM_PYTHON = Path(
    os.environ.get("ISAAC_GYM_PYTHON", "/home2/wyy/isaac-gym/isaacgym/python")
)
BOOTSTRAP = Path(__file__).resolve().with_name("v121c_prefix_smoke_bootstrap.py")
OUTPUT_ROOT = ROOT / "outputs" / "CmResidual"
MOTION_ROOT = (
    ROOT
    / "data" / "processed_data" / "dexplore_reconstructed_v120_coordfix_v4"
    / "converted_attempt1"
)
INPUT_MANIFEST = MOTION_ROOT.parent / "manifest.json"
DEXPLORE_CHECKPOINT = Path("/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth")
CMV2_CHECKPOINT = (
    ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/latest.pt"
)
PHYSICAL_GPU = 3
GPU_CAPACITY_GATE_MIB = 1024
EXPECTED_SHA256 = {
    "dexplore_checkpoint": "8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553",
    "cmv2_checkpoint": "371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591",
}


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


def _checked_input(path: Path, expected: str | None = None) -> dict[str, str]:
    resolved = path.resolve()
    digest = sha256_file(resolved)
    if expected is not None and digest != expected:
        raise ValueError(f"SHA256 mismatch for {resolved}: {digest}")
    return {"path": str(resolved), "sha256": digest}


def resolve_inputs() -> dict[str, dict[str, str]]:
    manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("classification") != "reconstructed_baseline":
        raise ValueError("V1.21c requires the reconstructed_baseline input manifest")
    tensor = MOTION_ROOT / "s1_airplane_lift" / "interaction_hand_inspire.pt"
    if not (ISAAC_GYM_PYTHON / "isaacgym/__init__.py").is_file():
        raise FileNotFoundError(f"missing Isaac Gym Python binding: {ISAAC_GYM_PYTHON}")
    assets = DEXPLORE_ROOT / "dexplore/data/assets"
    inputs = {
        "dexplore_checkpoint": _checked_input(
            DEXPLORE_CHECKPOINT, EXPECTED_SHA256["dexplore_checkpoint"]
        ),
        "cmv2_checkpoint": _checked_input(
            CMV2_CHECKPOINT, EXPECTED_SHA256["cmv2_checkpoint"]
        ),
        "input_manifest": _checked_input(INPUT_MANIFEST),
        "motion_tensor": _checked_input(tensor),
        "inspire_urdf": _checked_input(
            assets / "inspire_hand_new/inspire_hand_right.urdf"
        ),
        "airplane_mesh": _checked_input(
            assets / "mjcf/objects/airplane/airplane.obj"
        ),
        "table_mesh": _checked_input(assets / "mjcf/objects/table/table.obj"),
        "dexplore_run": _checked_input(DEXPLORE_ROOT / "dexplore/run.py"),
        "isaac_gym_python": {
            "path": str(ISAAC_GYM_PYTHON.resolve()),
            "sha256": sha256_file(ISAAC_GYM_PYTHON / "isaacgym/__init__.py"),
        },
    }
    return inputs


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


def smoke_gate_passed(result: dict[str, object]) -> bool:
    """Return the revised hard gate; numeric parity remains diagnostic only."""
    return bool(result.get("task_indices_equal")) and bool(result.get("duplicate_valid"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = (OUTPUT_ROOT / args.run_id).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    inputs = resolve_inputs()
    gpu_preflight_mib = _gpu_used_mib(PHYSICAL_GPU)
    if gpu_preflight_mib > GPU_CAPACITY_GATE_MIB:
        raise RuntimeError(
            f"GPU{PHYSICAL_GPU} capacity gate failed: {gpu_preflight_mib} MiB used"
        )
    command = smoke_command()
    protocol = {
        "mode": "prefix_duplicate_smoke",
        "physical_gpu": PHYSICAL_GPU,
        "logical_gpu": 0,
        "num_envs": 9,
        "source_states": 1,
        "prefix_steps": 1,
        "candidate_count": 8,
        "duplicate_candidate": 0,
        "env_assignment": "PCG64(candidate_seed) permutation of [0,0,1,2,3,4,5,6,7]",
        "numeric_parity": "diagnostic_only",
        "control_hz": 30,
        "physics_substeps_per_action": 2,
        "position_ceiling_m": 5e-4,
        "rotation_ceiling_rad": 5e-3,
        "seed": 5909,
        "sequence": "s1_airplane_lift",
        "research_scope": "engineering smoke only; no PPO and no ranking conclusion",
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "preflight": "passed",
                    "command": command,
                    "inputs": inputs,
                    "protocol": protocol,
                    "gpu_preflight_mib": gpu_preflight_mib,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
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
    plan = ROOT / "src/task/CmResidual/docs/plan/V1.21c.md"
    config = {
        "command": command,
        "runtime": {
            "python": sys.executable,
            "physical_gpu": PHYSICAL_GPU,
            "cuda_visible_devices": str(PHYSICAL_GPU),
            "gpu_preflight_mib": gpu_preflight_mib,
        },
        "protocol": protocol,
        "resolved_inputs": inputs,
    }
    write_json(output / "config.json", config)
    manifest = build_manifest(
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
    environment["CUDA_VISIBLE_DEVICES"] = str(PHYSICAL_GPU)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str((DEXPLORE_ROOT / "dexplore").resolve()),
            str(ISAAC_GYM_PYTHON.resolve()),
            str(ROOT),
            environment.get("PYTHONPATH", ""),
        )
    )
    environment["REF2DEX_V121C_SMOKE_OUTPUT"] = str(result_output)
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
        if not smoke_gate_passed(result):
            raise RuntimeError("bootstrap returned a failed smoke gate")
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
