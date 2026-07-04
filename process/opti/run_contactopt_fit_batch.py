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
DEFAULT_SUMMARY_DIR = REF2DEX_ROOT / "tmp" / "logs"


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
        str((REF2DEX_ROOT / "process" / "opti" / "contactopt_fit.py").resolve()),
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
        "--contact-source",
        args.contact_source,
        "--mano-param-mode",
        args.mano_param_mode,
        "--frame-batch-size",
        str(args.frame_batch_size),
        "--pca-fit-batch-size",
        str(args.pca_fit_batch_size),
        "--pca-fit-iter",
        str(args.pca_fit_iter),
        "--pca-fit-lr",
        str(args.pca_fit_lr),
    ]
    if args.export_init_only:
        cmd.append("--export-init-only")
    if args.respect_hand_valid:
        cmd.append("--respect-hand-valid")
    else:
        cmd.append("--no-respect-hand-valid")

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
    parser = argparse.ArgumentParser(description="Run ContactOpt fitting/export over a CSV manifest of single-hand jobs.")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--processed-root", type=str, required=True)
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--contact-source", type=str, default="geom", choices=["deepcontact", "geom"])
    parser.add_argument("--mano-param-mode", type=str, default="auto", choices=["auto", "native", "contactopt_pca"])
    parser.add_argument("--frame-batch-size", type=int, default=16)
    parser.add_argument("--pca-fit-batch-size", type=int, default=32)
    parser.add_argument("--pca-fit-iter", type=int, default=200)
    parser.add_argument("--pca-fit-lr", type=float, default=0.03)
    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument("--respect-hand-valid", dest="respect_hand_valid", action="store_true")
    valid_group.add_argument("--no-respect-hand-valid", dest="respect_hand_valid", action="store_false")
    parser.set_defaults(respect_hand_valid=True)
    parser.add_argument("--export-init-only", action="store_true", default=False)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    processed_root = Path(args.processed_root).resolve()
    output_root = Path(args.output_root).resolve()
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else (DEFAULT_SUMMARY_DIR / f"{output_root.name}_batch_summary.json").resolve()
    )

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
        print(f"[contactopt-fit-batch] job {idx}/{total} seq={seq_id} side={side}", flush=True)
        result = _run_one(row, args, output_root)
        results.append(result)
        print(
            f"[contactopt-fit-batch] status={result['status']} seq={seq_id} side={side} "
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
        "export_init_only": bool(args.export_init_only),
        "total_elapsed_sec": float(time.time() - t0),
        "results": results,
    }
    _write_summary(summary_json, payload)
    print(f"[contactopt-fit-batch] wrote summary {summary_json}")


if __name__ == "__main__":
    main()
