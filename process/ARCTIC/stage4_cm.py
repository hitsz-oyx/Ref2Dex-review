from __future__ import annotations

import argparse
import traceback
from glob import glob
from pathlib import Path

import numpy as np

from process.ARCTIC.stage2_optimize import RAW_SEQS_DIR
from process.ARCTIC.raw import ArcticRawAdapter, DATA_ROOT, MANO_MODEL_DIR
from process.common.stage4_cm import (
    ROOT,
    build_hand_sequence,
    build_shared_sequence,
    resolve_device,
    write_manifest,
    write_meta,
)


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed_data" / "stage4" / "data" / "arctic"
SOURCE_FPS = 30.0


def _resolve_sequences(args: argparse.Namespace) -> list[str]:
    if args.raw_file:
        return [str(Path(args.raw_file).resolve())]
    sequences = sorted(glob(str(Path(RAW_SEQS_DIR) / "*" / "*.mano.npy")))
    if not args.seq and not args.subject:
        return sequences
    if args.seq:
        suffix = ".mano.npy"
        target = args.seq[: -len(suffix)] if args.seq.endswith(suffix) else args.seq
        return [
            path
            for path in sequences
            if Path(path).relative_to(RAW_SEQS_DIR).as_posix()[: -len(suffix)] == target
        ]
    prefix = args.subject.rstrip("/") + "/"
    return [
        path for path in sequences if Path(path).relative_to(RAW_SEQS_DIR).as_posix().startswith(prefix)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw ARCTIC -> Cm sequence cache")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--seq", default=None)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument("--ds-rate", type=int, default=1)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--preprocess-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--mano-batch-size", type=int, default=128)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="mm")
    parser.add_argument("--save-compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sum(value is not None for value in (args.raw_file, args.seq, args.subject)) > 1:
        raise SystemExit("Use only one of --raw-file, --seq, or --subject")
    if args.num_obj_points != 4096:
        raise SystemExit("Current Cm Stage 4 contract requires --num-obj-points=4096")
    if args.ds_rate <= 0:
        raise SystemExit("--ds-rate must be positive")
    if args.preprocess_stride <= 0:
        raise SystemExit("--preprocess-stride must be positive")

    sequences = _resolve_sequences(args)
    if not sequences:
        raise FileNotFoundError("No ARCTIC sequences matched the requested input")

    device = resolve_device(args.device)
    output_root = Path(args.output_root).resolve()
    source_root = Path(DATA_ROOT).resolve()
    adapter = ArcticRawAdapter(
        num_obj_points=args.num_obj_points,
        device=str(device),
        preprocess_stride=args.ds_rate * args.preprocess_stride,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        obj_unit=args.obj_unit,
        nn_batch_size=args.nn_batch_size,
        mano_batch_size=args.mano_batch_size,
    )
    sides = ("left", "right") if args.side == "both" else (args.side,)
    stats = {
        "source_sequences": len(sequences),
        "shared_written": 0,
        "hand_written": 0,
        "skipped": 0,
        "empty_side": 0,
        "failed": 0,
        "frames": 0,
    }

    for raw_path_text in sequences:
        raw_path = Path(raw_path_text).resolve()
        try:
            source = adapter.process_sequence(str(raw_path))
            sequence_dir = output_root / str(source["subject_id"]) / str(source["seq_name"])
            shared_path = sequence_dir / "shared.npz"
            pending_sides = [side for side in sides if args.overwrite or not (sequence_dir / f"{side}.npz").exists()]
            if not pending_sides:
                stats["skipped"] += len(sides)
                continue
            if args.overwrite or not shared_path.exists():
                shared = build_shared_sequence(
                    source,
                    source_path=raw_path,
                    source_root=source_root,
                    ds_rate=args.ds_rate * args.preprocess_stride,
                    source_fps=SOURCE_FPS,
                )
                sequence_dir.mkdir(parents=True, exist_ok=True)
                if args.save_compressed:
                    np.savez_compressed(shared_path, **shared)
                else:
                    np.savez(shared_path, **shared)
                stats["shared_written"] += 1
            for side in sides:
                output_path = sequence_dir / f"{side}.npz"
                if side not in pending_sides:
                    stats["skipped"] += 1
                    continue
                try:
                    hand = build_hand_sequence(
                        source,
                        side=side,
                        candidate_threshold=args.candidate_threshold,
                        frame_batch_size=args.frame_batch_size,
                        device=device,
                    )
                except ValueError as exc:
                    if "no reconstructed" in str(exc):
                        stats["empty_side"] += 1
                        continue
                    raise
                if args.save_compressed:
                    np.savez_compressed(output_path, **hand)
                else:
                    np.savez(output_path, **hand)
                stats["hand_written"] += 1
                stats["frames"] += int(hand["obj_candidate_mask_5cm"].shape[0])
                candidate_counts = hand["obj_candidate_mask_5cm"].sum(axis=1)
                print(
                    f"[stage4-arctic] wrote {output_path} frames={len(hand['obj_candidate_mask_5cm'])} "
                    f"candidate[min/median/max]={int(candidate_counts.min())}/"
                    f"{int(np.median(candidate_counts))}/{int(candidate_counts.max())}"
                )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage4-arctic] failed {raw_path}: {exc}")
            traceback.print_exc()

    write_meta(
        output_root,
        source_description="raw ARCTIC; no Stage 2 or Stage 3 dependency",
        source_root=source_root,
        ds_rate=args.ds_rate * args.preprocess_stride,
        source_fps=SOURCE_FPS,
        candidate_threshold=args.candidate_threshold,
        stats=stats,
        extra={
            "dataset_name": "arctic",
            "mano_path": MANO_MODEL_DIR,
            "obj_unit": args.obj_unit,
            "requested_ds_rate": int(args.ds_rate),
            "preprocess_stride": int(args.preprocess_stride),
        },
    )
    write_manifest(output_root)
    print(f"[stage4-arctic] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
