#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REF2DEX_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY_JSON = REF2DEX_ROOT / "outputs" / "mano_fit" / "batch_summary.json"


def _load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    required = {"seq_id", "side"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest {path} missing columns: {sorted(missing)}")
    return rows


def _resolve_output_pkl(output_root: Path, seq_id: str, side: str) -> Path:
    subject_id, seq_name = seq_id.split("/", 1)
    return output_root / subject_id / f"{seq_name}_{side}.pkl"


def _run_one(
    row: dict[str, str],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    seq_id = str(row["seq_id"])
    side = str(row["side"])
    out_path = _resolve_output_pkl(output_root, seq_id, side)
    if out_path.exists() and (not args.overwrite):
        return {
            "seq_id": seq_id,
            "side": side,
            "status": "skipped_existing",
            "elapsed_sec": 0.0,
            "output_pkl": str(out_path),
            "returncode": 0,
        }

    cmd = [
        sys.executable,
        str((REF2DEX_ROOT / "process" / "opti" / "mano_smplx_fit.py").resolve()),
        "--processed-root",
        str(Path(args.processed_root).resolve()),
        "--seq-id",
        seq_id,
        "--side",
        side,
        "--device",
        args.device,
        "--output-dir",
        str(output_root),
        "--repulsion-mode",
        args.repulsion_mode,
        "--penetration-tol-mm",
        str(args.penetration_tol_mm),
        "--lambda-repulsion-loss",
        str(args.lambda_repulsion_loss),
        "--lambda-contact-loss",
        str(args.lambda_contact_loss),
        "--frame-batch-size",
        str(args.frame_batch_size),
        "--n-iter",
        str(args.n_iter),
        "--lr",
        str(args.lr),
    ]
    if args.progress:
        cmd.append("--progress")
    if args.keep_all_frames:
        cmd.append("--keep-all-frames")

    start = time.time()
    proc = subprocess.run(cmd, cwd=str(REF2DEX_ROOT))
    elapsed = float(time.time() - start)
    return {
        "seq_id": seq_id,
        "side": side,
        "status": "ok" if proc.returncode == 0 else "failed",
        "elapsed_sec": elapsed,
        "output_pkl": str(out_path),
        "returncode": int(proc.returncode),
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 MANO fitting over a CSV manifest of single-hand jobs.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--processed-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=str(DEFAULT_SUMMARY_JSON))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repulsion-mode", type=str, default="sdf_grid")
    parser.add_argument("--penetration-tol-mm", type=float, default=2.0)
    parser.add_argument("--lambda-repulsion-loss", type=float, default=0.03)
    parser.add_argument("--lambda-contact-loss", type=float, default=10.0)
    parser.add_argument("--frame-batch-size", type=int, default=8)
    parser.add_argument("--n-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=False)
    parser.add_argument("--keep-all-frames", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    processed_root = Path(args.processed_root).resolve()
    output_root = Path(args.output_root).resolve()
    summary_json = Path(args.summary_json).resolve()

    rows = _load_manifest(manifest_path)
    start_index = max(int(args.start_index), 0)
    selected = rows[start_index:]
    if int(args.max_jobs) > 0:
        selected = selected[: int(args.max_jobs)]

    results: list[dict[str, Any]] = []
    total = len(selected)
    t0 = time.time()
    for idx, row in enumerate(selected, start=1):
        seq_id = str(row["seq_id"])
        side = str(row["side"])
        print(f"[mano-fit-batch] job {idx}/{total} seq={seq_id} side={side}", flush=True)
        result = _run_one(row, args, output_root)
        results.append(result)
        print(
            f"[mano-fit-batch] status={result['status']} seq={seq_id} side={side} "
            f"elapsed={result['elapsed_sec']:.3f}s",
            flush=True,
        )
        if result["status"] == "failed":
            payload = {
                "manifest": str(manifest_path),
                "processed_root": str(processed_root),
                "output_root": str(output_root),
                "results": results,
                "total_elapsed_sec": float(time.time() - t0),
            }
            _write_summary(summary_json, payload)
            raise SystemExit(1)

    payload = {
        "manifest": str(manifest_path),
        "processed_root": str(processed_root),
        "output_root": str(output_root),
        "num_jobs": total,
        "num_ok": int(sum(1 for row in results if row["status"] == "ok")),
        "num_skipped_existing": int(sum(1 for row in results if row["status"] == "skipped_existing")),
        "num_failed": int(sum(1 for row in results if row["status"] == "failed")),
        "total_elapsed_sec": float(time.time() - t0),
        "results": results,
    }
    _write_summary(summary_json, payload)
    print(f"[mano-fit-batch] wrote summary {summary_json}")


if __name__ == "__main__":
    main()
