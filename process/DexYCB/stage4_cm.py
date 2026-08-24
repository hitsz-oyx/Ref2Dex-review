# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np

from process.DexYCB.raw import (
    DEXYCB_ROOT,
    MANO_MODEL_DIR,
    DexYCBRawAdapter,
)
from process.common.stage4_cm import (
    ROOT,
    build_hand_sequence,
    build_shared_sequence,
    resolve_device,
    write_manifest,
    write_meta,
)


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "processed_data" / "stage4" / "data" / "dexycb"
# DexYCB 官方 release 给出的 RealSense 采集帧率（color+depth 同步流）。
SOURCE_FPS = 30.0


def _enumerate_captures(
    dexycb_root: Path,
    subjects: Iterable[str] | None = None,
) -> list[Path]:
    """列出 DexYCB 所有 capture 目录：<root>/raw/<...>/<capture_name>。

    ``capture_name`` 目录必须含 ``meta.yml``，否则视为无效。
    通过递归扫描 ``raw/`` 下所有 ``meta.yml`` 反推 capture 目录，
    因此对 ``raw/<subject>/<date_subject>/<capture>`` 这类多层 layout 同样兼容。
    """
    raw_root = dexycb_root / "raw"
    if not raw_root.is_dir():
        raise FileNotFoundError(f"DexYCB raw root not found: {raw_root}")
    subject_filter = set(subjects) if subjects else None
    captures: list[Path] = []
    for meta_path in raw_root.rglob("meta.yml"):
        capture_dir = meta_path.parent
        # 若指定了 --subject 过滤，确保 capture 路径上含对应 subject 名
        if subject_filter is not None:
            if not any(part in subject_filter for part in capture_dir.relative_to(raw_root).parts):
                continue
        captures.append(capture_dir.resolve())
    captures.sort()
    return captures


def _capture_side(capture: Path) -> str:
    """Read the single MANO side declared by a DexYCB capture."""
    import yaml

    meta = yaml.safe_load((capture / "meta.yml").read_text(encoding="utf-8"))
    sides = [str(value) for value in meta.get("mano_sides", [])]
    if len(sides) != 1:
        raise ValueError(f"{capture}: expected exactly one mano_side, got {sides}")
    return sides[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DexYCB -> Cm sequence cache")
    parser.add_argument("--dexycb-root", default=str(DEXYCB_ROOT))
    parser.add_argument("--mano-path", default=str(MANO_MODEL_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--subject", action="append", default=None,
                        help="Restrict to a specific subject dir name (repeatable)")
    parser.add_argument("--seq", default=None,
                        help="Substring filter on capture directory name")
    parser.add_argument("--side", choices=["left", "right", "both"], default="right",
                        help="Hand side to export; the current adapter implements right only")
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument("--ds-rate", type=int, default=1)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--preprocess-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--mano-batch-size", type=int, default=128)
    parser.add_argument("--save-compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_obj_points != 4096:
        raise SystemExit("Current Cm Stage 4 contract requires --num-obj-points=4096")
    if args.ds_rate <= 0:
        raise SystemExit("--ds-rate must be positive")
    if args.preprocess_stride <= 0:
        raise SystemExit("--preprocess-stride must be positive")
    if args.side != "right":
        raise SystemExit("The current DexYCB adapter implements --side=right only")

    dexycb_root = Path(args.dexycb_root).resolve()
    mano_path = Path(args.mano_path).resolve()
    output_root = Path(args.output_root).resolve()
    device = resolve_device(args.device)

    captures = _enumerate_captures(dexycb_root, subjects=args.subject)
    captures = [capture for capture in captures if _capture_side(capture) == args.side]
    if args.seq:
        captures = [c for c in captures if args.seq in c.name or args.seq in c.parent.name]
    if args.limit is not None:
        captures = captures[: int(args.limit)]
    if not captures:
        raise FileNotFoundError("No DexYCB captures matched the requested input")

    adapter = DexYCBRawAdapter(
        num_obj_points=args.num_obj_points,
        device=str(device),
        preprocess_stride=args.ds_rate * args.preprocess_stride,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        nn_batch_size=args.nn_batch_size,
        mano_batch_size=args.mano_batch_size,
        dexycb_root=str(dexycb_root),
    )
    # DexYCB 每条 sequence 只有一侧手；当前 adapter 只实现右手。
    sides = ("right",)
    stats = {
        "source_sequences": len(captures),
        "shared_written": 0,
        "hand_written": 0,
        "skipped": 0,
        "empty_side": 0,
        "failed": 0,
        "frames": 0,
    }

    for capture_path in captures:
        try:
            source = adapter.process_sequence(str(capture_path))
            sequence_dir = output_root / str(source["subject_id"]) / str(source["seq_name"])
            shared_path = sequence_dir / "shared.npz"
            pending_sides = [side for side in sides if args.overwrite or not (sequence_dir / f"{side}.npz").exists()]
            if not pending_sides:
                stats["skipped"] += len(sides)
                continue
            if args.overwrite or not shared_path.exists():
                shared = build_shared_sequence(
                    source,
                    source_path=capture_path,
                    source_root=dexycb_root,
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
                    f"[stage4-dexycb] wrote {output_path} frames={len(hand['obj_candidate_mask_5cm'])} "
                    f"candidate[min/median/max]={int(candidate_counts.min())}/"
                    f"{int(np.median(candidate_counts))}/{int(candidate_counts.max())}"
                )
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage4-dexycb] failed {capture_path}: {exc}")
            traceback.print_exc()

    write_meta(
        output_root,
        source_description="raw DexYCB; right-hand captures; world=reference/master-camera frame",
        source_root=dexycb_root,
        ds_rate=args.ds_rate * args.preprocess_stride,
        source_fps=SOURCE_FPS,
        candidate_threshold=args.candidate_threshold,
        stats=stats,
        extra={
            "dataset_name": "dexycb",
            "mano_path": str(mano_path.resolve()),
            "requested_ds_rate": int(args.ds_rate),
            "preprocess_stride": int(args.preprocess_stride),
            "note": "pose.npz object and MANO poses use the identity-extrinsic reference/master-camera frame; leading all-zero MANO frames are trimmed",
        },
    )
    write_manifest(output_root)
    print(f"[stage4-dexycb] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
