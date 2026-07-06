from __future__ import annotations

import argparse
import os.path as op
import traceback
from glob import glob
from pathlib import Path

from process.common.stage2 import pack_stage2_hand, save_stage2_payload, write_stage2_meta
from process.GRAB.raw import (
    DEFAULT_GRAB_ROOT,
    DEFAULT_MANO_MODEL_DIR,
    GRABRawAdapter,
    load_manifest_seq_paths,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "processed_data" / "generated" / "stage2" / "grab_initonly_4096"


def _resolve_sequences(args: argparse.Namespace) -> list[str]:
    if args.raw_file:
        return [str(Path(args.raw_file).resolve())]
    if args.manifest:
        return load_manifest_seq_paths(args.manifest, args.grab_root)

    sequences = sorted(glob(op.join(args.grab_root, "grab", "*", "*.npz")))
    if not args.seq:
        return sequences
    target = args.seq[:-4] if args.seq.endswith(".npz") else args.seq
    selected = []
    for path in sequences:
        rel = Path(path).relative_to(Path(args.grab_root) / "grab").with_suffix("").as_posix()
        if rel == target or target in rel:
            selected.append(path)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw GRAB -> Ref2Dex common Stage 2 init-only payloads")
    parser.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--mano-path", default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--seq", default=None, help="Exact or substring sequence filter, e.g. s1/bowl_pass_1")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument("--frame-keep-threshold", type=float, default=0.05)
    parser.add_argument("--ds-rate", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="m")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest and (args.raw_file or args.seq):
        raise SystemExit("--manifest cannot be combined with --raw-file/--seq")
    if args.num_obj_points != 4096:
        raise SystemExit("The Stage 2 schema requires --num-obj-points=4096")

    sequences = _resolve_sequences(args)
    if not sequences:
        raise FileNotFoundError("No GRAB sequences matched the requested input")

    output_root = Path(args.output_root).resolve()
    preprocessor = GRABRawAdapter(
        num_obj_points=args.num_obj_points,
        device=args.device,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        grab_root=args.grab_root,
        mano_path=args.mano_path,
        ds_rate=args.ds_rate,
        obj_unit=args.obj_unit,
        nn_batch_size=args.nn_batch_size,
    )
    sides = ("left", "right") if args.side == "both" else (args.side,)
    config = {
        "object_surface_samples": args.num_obj_points,
        "object_surface_seed": 42,
        "ds_rate": args.ds_rate,
        "mano_path": str(Path(args.mano_path).resolve()),
        "flat_hand_mean": True,
        "requires_vtemplate": True,
        "processing_mode": "init_only",
    }
    stats = {"source_sequences": len(sequences), "written": 0, "empty": 0, "skipped": 0, "failed": 0}

    for raw_path in sequences:
        try:
            source = preprocessor.process_sequence(raw_path)
            source_rel = Path(raw_path).resolve().relative_to(Path(args.grab_root).resolve()).as_posix()
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
                print(f"[GRAB] wrote {path} ({len(payload['raw_frame_id'])} frames)")
        except Exception as exc:
            stats["failed"] += 1
            print(f"[GRAB] failed {raw_path}: {exc}")
            traceback.print_exc()

    write_stage2_meta(
        output_root,
        dataset_name="grab",
        processing_mode="init_only",
        source_root=args.grab_root,
        config={**config, "frame_keep_threshold": args.frame_keep_threshold},
        stats=stats,
    )
    print(f"[GRAB] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
