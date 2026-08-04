from __future__ import annotations

import argparse
import os.path as op
import traceback
from glob import glob
from pathlib import Path

from process.common.stage2 import pack_stage2_hand, save_stage2_payload, write_stage2_meta
from process.ARCTIC.raw import (
    DATA_ROOT,
    MANO_MODEL_DIR,
    RAW_SEQS_DIR,
    ArcticRawAdapter,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed_data" / "stage2" / "arctic_initonly_4096"


def _resolve_sequences(args: argparse.Namespace) -> list[str]:
    if args.raw_file:
        return [str(Path(args.raw_file).resolve())]
    sequences = sorted(glob(op.join(RAW_SEQS_DIR, "*", "*.mano.npy")))
    if not args.seq and not args.subject:
        return sequences
    if args.seq:
        suffix = ".mano.npy"
        target = args.seq[: -len(suffix)] if args.seq.endswith(suffix) else args.seq
        return [
            path
            for path in sequences
            if (
                Path(path).relative_to(RAW_SEQS_DIR).as_posix()[: -len(suffix)]
                == target
            )
        ]
    prefix = args.subject.rstrip("/") + "/"
    return [path for path in sequences if Path(path).relative_to(RAW_SEQS_DIR).as_posix().startswith(prefix)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw ARCTIC -> Ref2Dex common Stage 2 init-only payloads")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--seq", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument("--frame-keep-threshold", type=float, default=0.05)
    parser.add_argument("--preprocess-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--mano-batch-size", type=int, default=128)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="mm")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum(value is not None for value in (args.raw_file, args.seq, args.subject)) > 1:
        raise SystemExit("Use only one of --raw-file, --seq, or --subject")
    if args.num_obj_points != 4096:
        raise SystemExit("The Stage 2 schema requires --num-obj-points=4096")

    sequences = _resolve_sequences(args)
    if not sequences:
        raise FileNotFoundError("No ARCTIC sequences matched the requested input")

    output_root = Path(args.output_root).resolve()
    preprocessor = ArcticRawAdapter(
        num_obj_points=args.num_obj_points,
        device=args.device,
        preprocess_stride=args.preprocess_stride,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        obj_unit=args.obj_unit,
        nn_batch_size=args.nn_batch_size,
        mano_batch_size=args.mano_batch_size,
    )
    sides = ("left", "right") if args.side == "both" else (args.side,)
    config = {
        "object_surface_samples": args.num_obj_points,
        "object_surface_seed": 42,
        "preprocess_stride": args.preprocess_stride,
        "mano_path": MANO_MODEL_DIR,
        "flat_hand_mean": False,
        "requires_vtemplate": False,
        "processing_mode": "init_only",
    }
    stats = {"source_sequences": len(sequences), "written": 0, "empty": 0, "skipped": 0, "failed": 0}

    for raw_path in sequences:
        try:
            source = preprocessor.process_sequence(raw_path)
            source_rel = Path(raw_path).resolve().relative_to(Path(DATA_ROOT).resolve()).as_posix()
            for side in sides:
                expected = output_root / str(source["subject_id"]) / f"{source['seq_name']}_{side}.pkl"
                if expected.exists() and not args.overwrite:
                    stats["skipped"] += 1
                    continue
                payload = pack_stage2_hand(
                    source,
                    side=side,
                    source_raw_file=source_rel,
                    processing_mode="init_only",
                    frame_keep_threshold=args.frame_keep_threshold,
                    config=config,
                )
                if payload is None:
                    stats["empty"] += 1
                    continue
                path = save_stage2_payload(payload, output_root)
                stats["written"] += 1
                print(f"[ARCTIC] wrote {path} ({len(payload['raw_frame_id'])} frames)")
        except Exception as exc:
            stats["failed"] += 1
            print(f"[ARCTIC] failed {raw_path}: {exc}")
            traceback.print_exc()

    write_stage2_meta(
        output_root,
        dataset_name="arctic",
        processing_mode="init_only",
        source_root=DATA_ROOT,
        config={**config, "frame_keep_threshold": args.frame_keep_threshold},
        stats=stats,
    )
    print(f"[ARCTIC] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
