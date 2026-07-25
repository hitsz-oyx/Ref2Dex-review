"""Build Stage 5 Cp samples from the temporal Stage 4 GRAB representation.

Stage 5 deliberately keeps the complete object pool.  Runtime datasets choose
their own deterministic 512-point subset, while the saved object flow and
contact distances remain the clean task-effect supervision.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABRawAdapter, load_manifest_seq_paths
from process.stage4.prepare_cm import _resolve_device, _resolve_sequences, build_stage4_sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "processed_data" / "generated" / "stage5"
SCHEMA_NAME = "ref2dex_cp_stage5"
SCHEMA_VERSION = "1.0.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw GRAB -> Stage 5 Cp temporal pairs")
    parser.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--mano-path", default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument("--seq", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--side", choices=["left", "right", "both"], default="both")
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument("--ds-rate", type=int, default=1)
    parser.add_argument("--pair-stride", type=int, default=3)
    parser.add_argument("--pair-hop", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--frame-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="m")
    parser.add_argument("--mirror-left-to-right", action="store_true")
    parser.add_argument("--save-compressed", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest and (args.raw_file or args.seq):
        raise SystemExit("--manifest cannot be combined with --raw-file/--seq")
    if args.num_obj_points != 4096:
        raise SystemExit("Stage 5 contract requires --num-obj-points=4096")
    sequences = load_manifest_seq_paths(args.manifest, args.grab_root) if args.manifest else _resolve_sequences(args)
    if not sequences:
        raise FileNotFoundError("No GRAB sequences matched the requested input")
    device = _resolve_device(args.device)
    output_root, grab_root = Path(args.output_root).resolve(), Path(args.grab_root).resolve()
    adapter = GRABRawAdapter(num_obj_points=4096, device=str(device), max_frames=args.max_frames or None,
                             grab_root=str(grab_root), mano_path=args.mano_path, ds_rate=args.ds_rate,
                             obj_unit=args.obj_unit, nn_batch_size=args.nn_batch_size)
    stats = {"source_sequences": len(sequences), "written": 0, "pairs": 0, "empty_side": 0, "failed": 0}
    for raw_text in sequences:
        raw_path = Path(raw_text).resolve()
        try:
            source = adapter.process_sequence(str(raw_path))
            for side in (("left", "right") if args.side == "both" else (args.side,)):
                output = output_root / str(source["subject_id"]) / f"{source['seq_name']}_{side}.npz"
                if output.exists() and not args.overwrite:
                    continue
                try:
                    payload = build_stage4_sequence(source, source_path=raw_path, grab_root=grab_root, side=side,
                        pair_stride=args.pair_stride, pair_hop=args.pair_hop, candidate_threshold=args.candidate_threshold,
                        frame_batch_size=args.frame_batch_size, device=device, ds_rate=args.ds_rate,
                        mirror_left_to_right=args.mirror_left_to_right)
                except ValueError as exc:
                    if "no reconstructed" in str(exc):
                        stats["empty_side"] += 1
                        continue
                    raise
                payload["schema_name"] = np.asarray(SCHEMA_NAME)
                payload["schema_version"] = np.asarray(SCHEMA_VERSION)
                # Object effect contact is derived from this clean full-pool distance.
                payload["obj_contact_gt"] = np.clip(1.0 - payload["obj_to_hand_min_dist"] / 0.02, 0.0, 1.0).astype(np.float32)
                output.parent.mkdir(parents=True, exist_ok=True)
                (np.savez_compressed if args.save_compressed else np.savez)(output, **payload)
                stats["written"] += 1; stats["pairs"] += len(payload["pair_index"])
                print(f"[stage5] wrote {output} pairs={len(payload['pair_index'])}")
        except Exception as exc:
            stats["failed"] += 1
            print(f"[stage5] failed {raw_path}: {exc}")
            traceback.print_exc()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "meta.json").write_text(json.dumps({"schema_name": SCHEMA_NAME, "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "raw GRAB via Stage 4 geometry adapter",
        "coordinate_frame": "hand_root_t", "num_obj_pool": 4096, "num_hand_points": 1538,
        "task_fields": "obj_contact_gt,obj_flow_gt,hand_flow", "stats": stats}, indent=2), encoding="utf-8")
    print(f"[stage5] summary: {stats}")
    if stats["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
