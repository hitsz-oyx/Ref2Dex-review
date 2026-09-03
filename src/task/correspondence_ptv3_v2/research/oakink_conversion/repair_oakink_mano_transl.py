"""Atomically repair OakInk Stage-3 ``mano_transl`` for smplx.MANO.

The first true-hand-root export stored OakInk ``hand_tsl`` directly.  OakInk
defines it as the world-space wrist position, whereas smplx.MANO expects a
translation applied to a mesh whose shaped wrist joint has a non-zero template
offset.  This utility subtracts that per-frame offset while preserving every
other array in each compressed sample.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np


def _wrist_model(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    template = np.asarray(payload["v_template"], dtype=np.float32)
    shapedirs = np.asarray(payload["shapedirs"], dtype=np.float32)
    regressor = np.asarray(payload["J_regressor"][0].toarray(), dtype=np.float32).reshape(778)
    base_wrist = regressor @ template
    shape_wrist_basis = np.einsum("v,vck->ck", regressor, shapedirs)
    return base_wrist.astype(np.float32), shape_wrist_basis.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--mano-model",
        default="/mnt/ugreen_nas/storage/Ref2Dex_storage/shared_assets/body_models/mano/MANO_RIGHT.pkl",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    files = sorted(data_root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files under {data_root}")
    base_wrist, shape_wrist_basis = _wrist_model(Path(args.mano_model).resolve())

    for index, path in enumerate(files, start=1):
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
        betas = np.asarray(payload["mano_betas"], dtype=np.float32)
        root_pose = np.asarray(payload["hand_root_pose"], dtype=np.float32)
        if betas.ndim != 2 or betas.shape[1] != 10:
            raise ValueError(f"{path}: expected mano_betas (T, 10), got {betas.shape}")
        if root_pose.shape != (betas.shape[0], 4, 4):
            raise ValueError(f"{path}: incompatible hand_root_pose shape {root_pose.shape}")
        wrist_offsets = base_wrist[None, :] + betas @ shape_wrist_basis.T
        payload["mano_transl"] = (root_pose[:, :3, 3] - wrist_offsets).astype(np.float32)

        temp_path = path.with_name(path.stem + ".repairing.npz")
        np.savez_compressed(temp_path, **payload)
        os.replace(temp_path, path)
        if index % 50 == 0 or index == len(files):
            print(f"[{index}/{len(files)}] {path.name}", flush=True)

    stats_path = data_root / "oakink_pilot_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["schema_version"] = "2.0.0"
        stats["mano_transl_semantics"] = "smplx translation; shaped wrist joint equals OakInk hand_tsl"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
