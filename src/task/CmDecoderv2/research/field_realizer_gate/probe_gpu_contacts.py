"""Probe pairwise contacts after GPU simulation starts, without changing physics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log_path = args.output / "train.log"
    log = log_path.open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    manifest = {
        "manifest_schema": "ref2dex.run.v1", "task": "CmDecoderv2",
        "work_version": "V1.1.16", "operation_category": ["diagnostic", "operation"],
        "run_id": args.output.name, "run_status": "RUNNING", "seed": 42,
        "created_at": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_manifest": str(args.source_manifest.resolve()),
        "source_sha256": hashlib.sha256(args.source_manifest.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_snapshot": "config.json", "metadata_snapshot": "metadata.json",
        "scope": "instrument-only GPU contact capability; no physical Gate conclusion",
    }
    manifest_path = args.output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    env = None
    try:
        import isaacgym  # Must precede torch.
        from isaacgym import gymapi
        import torch
        from ...tools.rl.run_residual import make_config, create_env, code_hashes
        torch.set_num_threads(4)
        torch.manual_seed(42)
        config = make_config(SimpleNamespace(output=args.output, num_envs=64, iterations=1))
        config["task"]["basePolicy"]["sourceManifest"] = str(args.source_manifest.resolve())
        assert config["task"]["sim"]["use_gpu_pipeline"] is True
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        metadata = {"num_envs": 64, "pipeline": "gpu", "source": str(args.source_manifest),
                    "code_sha256": code_hashes()}
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        env = create_env(config)
        records = []
        for step in range(4):
            env.step(torch.zeros((env.num_envs, 12), device=env.device))
            contact = env.gym.get_env_rigid_contacts(env.envs[0])
            records.append({"step": step + 1, "record_count": len(contact),
                            "dtype": str(contact.dtype)})
        env.gym.destroy_sim(env.sim)
        env = None
        # The bindings may emit a native error and return an empty array instead
        # of raising. An empty array alone never establishes API support.
        import ctypes
        ctypes.CDLL(None).fflush(None)
        text = log_path.read_text()
        unsupported = "cannot be used with the GPU pipeline" in text
        result = {"samples": records, "gpu_pipeline": True,
                  "pairwise_capability": "UNAVAILABLE" if unsupported else "UNVERIFIED",
                  "conclusion": "INCONCLUSIVE", "formal_rollout_allowed": False}
        (args.output / "capability.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        manifest["run_status"] = "COMPLETED"
    except BaseException as error:
        manifest.update(run_status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
