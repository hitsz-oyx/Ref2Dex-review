from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.task.PointWorldWAM.config import load_config
from src.task.PointWorldWAM.dataset import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 GRAB PointWorld 11 帧窗口")
    parser.add_argument("--config", default="src/task/PointWorldWAM/configs/grab_forward_overfit.yaml")
    parser.add_argument("--output", default="output/research/pointworld_wam/data_sanity")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = build_dataset(cfg.data)
    sample = dataset[0]
    obj = sample["object_points"].numpy()
    hand = sample["hand_points"].numpy()
    obj_normals = sample["object_normals"].numpy()
    hand_normals = sample["hand_normals"].numpy()
    obj_endpoint = np.linalg.norm(obj[-1] - obj[0], axis=-1)
    report = {
        "windows": len(dataset),
        "sequence": sample["sequence"],
        "raw_frame_id": sample["raw_frame_id"].tolist(),
        "object_shape": list(obj.shape),
        "hand_shape": list(hand.shape),
        "object_dtype": str(obj.dtype),
        "hand_dtype": str(hand.dtype),
        "object_id_unique": int(np.unique(sample["obj_point_id"].numpy()).size),
        "hand_id_unique": int(np.unique(sample["hand_point_id"].numpy()).size),
        "object_t0_centroid_norm_m": float(np.linalg.norm(obj[0].mean(0))),
        "object_endpoint_mean_m": float(obj_endpoint.mean()),
        "object_endpoint_max_m": float(obj_endpoint.max()),
        "hand_extent_m": float(np.linalg.norm(np.ptp(hand[0], axis=0))),
        "object_normal_norm_mean": float(np.linalg.norm(obj_normals, axis=-1).mean()),
        "hand_normal_norm_mean": float(np.linalg.norm(hand_normals, axis=-1).mean()),
        "finite": bool(np.isfinite(obj).all() and np.isfinite(hand).all()),
    }
    if report["object_id_unique"] != cfg.data.object_points:
        raise AssertionError("object point ID 不唯一")
    if report["hand_id_unique"] != cfg.data.hand_points:
        raise AssertionError("hand point ID 不唯一")
    if not report["finite"]:
        raise FloatingPointError("窗口包含 NaN/Inf")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    np.savez(output / "window.npz", object_points=obj, hand_points=hand)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
