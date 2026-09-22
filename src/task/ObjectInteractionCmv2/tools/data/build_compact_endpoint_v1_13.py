"""Build a V1.13 pilot or full compact endpoint cache from the V1.12 reference path."""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import traceback
from pathlib import Path

from src.task.ObjectInteractionCmv2.compact_endpoint import (
    CACHE_SCHEMA,
    CompactEndpointShardWriter,
    sha256_file,
)
from src.task.ObjectInteractionCmv2.mixed_training import GROUPS, REPO_ROOT
from src.task.ObjectInteractionCmv2.part_se3_training import (
    build_part_se3_dataset,
    load_part_se3_training_config,
)


class StopRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"received signal {signum}")
        self.signum = int(signum)


def input_identity(config: dict) -> dict[str, str]:
    paths = {
        str(Path(config["articulation_metadata"]).resolve()),
        str((Path(config["split_root"]) / "index.json").resolve()),
        str((Path(config["split_root"]) / "cache_manifest.json").resolve()),
    }
    for source in config["sources"].values():
        paths.update((str(Path(source["index"]).resolve()), str(Path(source["manifest"]).resolve())))
    adapter = Path(config["oakink2_parts"]["adapter_root"])
    paths.update(str((adapter / name).resolve()) for name in ("cache_manifest.json", "index.json"))
    return {path: sha256_file(path) for path in sorted(paths)}


def validate_full_gate(pilot_manifest: str | Path, source_sha256: dict[str, str]) -> None:
    manifest = json.loads(Path(pilot_manifest).read_text())
    if (manifest.get("schema_name") != CACHE_SCHEMA
            or manifest.get("source_sha256") != source_sha256
            or int(manifest.get("validation", {}).get("bad_count", 1)) != 0):
        raise ValueError("pilot manifest does not validate the current compact-cache inputs")
    ratio = float(manifest.get("statistics", {}).get("serialized_to_source_hand_ratio", float("inf")))
    if ratio > 1.25:
        raise ValueError(f"pilot compact/source ratio {ratio:.4f} exceeds the 1.25 full-build gate")


def build(config_path: str | Path, output: str | Path, run_id: str, *, mode: str,
          pilot_manifest: str | Path | None = None, shard_target_gib: float = 1.0,
          max_records_per_view: int | None = None) -> Path:
    if mode not in ("pilot", "full") or Path(run_id).name != run_id or not run_id:
        raise ValueError("invalid V1.13 compact-cache mode or run ID")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True):
        raise ValueError("compact-cache build requires a clean committed worktree")
    config = load_part_se3_training_config(config_path)
    source_sha256 = input_identity(config)
    if mode == "full":
        if pilot_manifest is None:
            raise ValueError("full compact-cache build requires an accepted pilot manifest")
        validate_full_gate(pilot_manifest, source_sha256)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    writer = CompactEndpointShardWriter(
        output, run_id=run_id, git_commit=commit, source_sha256=source_sha256,
        shard_target_bytes=int(float(shard_target_gib) * (1 << 30)))
    sequence_limit = 1 if mode == "pilot" else None
    def stop(signum, frame):
        raise StopRequested(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for split in ("train", "val"):
            strides = (None,) if split == "train" else tuple(config["data"]["validation_strides"])
            for group in GROUPS:
                for fixed_stride in strides:
                    dataset = build_part_se3_dataset(
                        config, group, split, fixed_stride=fixed_stride,
                        max_sequences=sequence_limit)
                    limit = len(dataset)
                    if max_records_per_view is not None:
                        limit = min(limit, int(max_records_per_view))
                    for index in range(limit):
                        writer.add(dataset[index], group=group, split=split)
        return writer.finalize(validation_bad_count=0)
    except StopRequested as error:
        writer.mark_incomplete("STOPPED", stop_signal=error.signum, stop_reason=str(error))
        raise SystemExit(128 + error.signum)
    except BaseException as error:
        writer.mark_incomplete("FAILED", error=repr(error), traceback=traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot-manifest", type=Path)
    parser.add_argument("--shard-target-gib", type=float, default=1.0)
    parser.add_argument("--max-records-per-view", type=int)
    args = parser.parse_args()
    result = build(
        args.config, args.output, args.run_id, mode=args.mode,
        pilot_manifest=args.pilot_manifest, shard_target_gib=args.shard_target_gib,
        max_records_per_view=args.max_records_per_view)
    print(json.dumps({"output": str(result), "manifest": str(result / "manifest.json")}))


if __name__ == "__main__":
    main()
