from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np

from process.common.stage3_corr import (
    _load_stage2,
    _resolve_device,
    _resolve_files,
    _write_meta,
    build_stage3_sequence,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE2_ROOT = ROOT / "data" / "processed_data" / "stage2" / "arctic_initonly_4096"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed_data" / "stage3" / "arctic_initonly_4096_hand_root_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARCTIC Stage 2 payloads -> Stage 3 v2 point pools")
    parser.add_argument("--stage2-root", default=str(DEFAULT_STAGE2_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--seq-id", default=None, help="Exact seq_id filter such as s05/box_grab_01")
    parser.add_argument("--side", choices=["left", "right"], default=None)
    parser.add_argument("--num-obj-pool", type=int, default=4096)
    parser.add_argument("--num-obj-train", type=int, default=512)
    parser.add_argument("--num-hand-points", type=int, default=1538)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mirror-left-to-right", action="store_true")
    parser.add_argument("--save-compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--coordinate-frame",
        choices=["object", "hand_root"],
        default="hand_root",
        help="Default keeps DenseToken inputs in hand-root coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_obj_pool != 4096 or args.num_obj_train != 512 or args.num_hand_points != 1538:
        raise SystemExit("Current schema requires num_obj_pool=4096, num_obj_train=512, num_hand_points=1538")

    stage2_root = Path(args.stage2_root).resolve()
    output_root = Path(args.output_root).resolve()
    files = _resolve_files(stage2_root, args.seq_id, args.side)
    if not files:
        raise FileNotFoundError(f"No ARCTIC Stage 2 pkl files found under {stage2_root}")

    device = _resolve_device(args.device)
    stats = {"source_files": len(files), "written": 0, "skipped": 0, "failed": 0, "frames": 0}

    for source_path in files:
        relative = source_path.relative_to(stage2_root).with_suffix(".npz")
        output_path = output_root / relative
        if output_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            continue
        try:
            payload = _load_stage2(source_path)
            stage3 = build_stage3_sequence(
                payload,
                num_obj_pool=args.num_obj_pool,
                num_hand_points=args.num_hand_points,
                candidate_threshold=args.candidate_threshold,
                frame_batch_size=args.frame_batch_size,
                device=device,
                mirror_left_to_right=args.mirror_left_to_right,
                coordinate_frame=args.coordinate_frame,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if args.save_compressed:
                np.savez_compressed(output_path, **stage3)
            else:
                np.savez(output_path, **stage3)
            num_frames = int(stage3["raw_frame_id"].shape[0])
            stats["written"] += 1
            stats["frames"] += num_frames
            candidate_counts = stage3["obj_candidate_mask_5cm"].sum(axis=1)
            print(
                f"[stage3-arctic] wrote {output_path} frames={num_frames} "
                f"candidate[min/median/max]={int(candidate_counts.min())}/"
                f"{int(np.median(candidate_counts))}/{int(candidate_counts.max())}"
            )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage3-arctic] failed {source_path}: {exc}")
            traceback.print_exc()

    _write_meta(output_root, args=args, stats=stats)
    print(f"[stage3-arctic] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
