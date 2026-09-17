#!/usr/bin/env python3
"""Resumable single-GPU cache job queue with durable per-job state.

The queue intentionally owns no cache schema.  Each job supplies an explicit
producer command and success manifest, so a failed or incomplete producer is
never silently treated as a reusable cache.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _completed(manifest_path: Path) -> bool:
    if not manifest_path.is_file():
        return False
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8")).get("run_status") == "COMPLETED"
    except json.JSONDecodeError:
        return False


def run(queue_path: Path, state_root: Path, poll_seconds: int) -> dict[str, Any]:
    queue_path, state_root = queue_path.resolve(), state_root.resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    jobs = queue.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("queue must contain a non-empty jobs list")
    state_root.mkdir(parents=True, exist_ok=True)
    state_path = state_root / "queue_state.json"
    state: dict[str, Any] = {"schema_name": "ref2dex_cmv2_cache_queue_state_v1", "created_at": _now(),
                             "queue": str(queue_path), "jobs": {}}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("id"), str):
            raise ValueError("each cache queue job needs a string id")
        command = job.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
            raise ValueError(f"{job['id']}: command must be a non-empty string list")
        manifest = Path(str(job.get("success_manifest", ""))).resolve()
        if not str(job.get("success_manifest", "")):
            raise ValueError(f"{job['id']}: success_manifest is required")
        if _completed(manifest):
            state["jobs"][job["id"]] = {"run_status": "COMPLETED", "finished_at": _now(),
                                          "success_manifest": str(manifest), "reused": True}
            _write_json(state_path, state)
            continue
        log_path = state_root / f"{job['id']}.log"
        state["jobs"][job["id"]] = {"run_status": "RUNNING", "started_at": _now(),
                                      "command": command, "success_manifest": str(manifest), "log": str(log_path)}
        _write_json(state_path, state)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        if result.returncode or not _completed(manifest):
            state["jobs"][job["id"]].update({"run_status": "FAILED", "finished_at": _now(),
                                                "exit_code": result.returncode})
            _write_json(state_path, state)
            raise RuntimeError(f"cache queue job failed: {job['id']}")
        state["jobs"][job["id"]].update({"run_status": "COMPLETED", "finished_at": _now(),
                                            "exit_code": 0})
        _write_json(state_path, state)
        time.sleep(max(0, int(poll_seconds)))
    state["run_status"] = "COMPLETED"; state["finished_at"] = _now()
    _write_json(state_path, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True, help="JSON: [{id, command, success_manifest}]")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run(args.queue, args.state_root, args.poll_seconds), ensure_ascii=False))


if __name__ == "__main__":
    main()
