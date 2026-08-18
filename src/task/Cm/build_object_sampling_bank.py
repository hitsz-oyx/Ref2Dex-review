"""Build deterministic uint32 sampling banks for object-only V2 caches."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_bank(sequence_dir: Path, *, bank_size: int = 4, num_points: int = 512, seed: int = 42) -> None:
    for side in ("left", "right"):
        offsets_path = sequence_dir / side / "candidate_offsets.npy"
        indices_path = sequence_dir / side / "candidate_indices.npy"
        if not offsets_path.exists():
            continue
        offsets = np.load(offsets_path, mmap_mode="r")
        indices = np.load(indices_path, mmap_mode="r")
        bank = np.empty((len(offsets) - 1, bank_size, num_points), dtype=np.uint32)
        for frame in range(len(bank)):
            candidates = np.asarray(indices[offsets[frame]:offsets[frame + 1]], dtype=np.uint32)
            if candidates.size == 0:
                bank[frame] = 0
                continue
            for b in range(bank_size):
                rng = np.random.default_rng(seed + frame * 1009 + b * 9176)
                bank[frame, b] = rng.choice(candidates, size=num_points, replace=candidates.size < num_points)
        np.save(sequence_dir / side / "sampling_indices.npy", bank)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--bank-size", type=int, default=4)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for shared in sorted(Path(args.root).glob("**/shared/meta.json")):
        build_bank(shared.parent.parent, bank_size=args.bank_size, num_points=args.num_points, seed=args.seed)


if __name__ == "__main__":
    main()
