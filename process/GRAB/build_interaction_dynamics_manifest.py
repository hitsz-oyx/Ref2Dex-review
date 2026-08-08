"""Build the V1 dominant-hand manifest at the fixed endpoint horizon 8."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from process.GRAB.filter_cm_dominant_hand import _resolve_sequences, build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/processed_data/interaction_dynamics_v1/data/grab"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed_data/interaction_dynamics_v1"))
    parser.add_argument("--split-file", type=Path); parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--dominance-ratio", type=float, default=2.0)
    parser.add_argument("--min-error-gap-m-per-step", type=float, default=0.0005)
    parser.add_argument("--max-main-error-m-per-step", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    sequences = _resolve_sequences(args.data_root.resolve(), args.split_file, args.max_sequences)
    summary = build_manifest(data_root=args.data_root, output_dir=args.output_dir,
        sequences=sequences, strides=(8,), dominance_ratio=args.dominance_ratio,
        min_error_gap_m_per_step=args.min_error_gap_m_per_step,
        max_main_error_m_per_step=args.max_main_error_m_per_step, num_workers=args.num_workers)
    summary["task"] = "InteractionDynamics"
    summary["chunk_len"] = 8
    summary["temporal_stride"] = 1
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
