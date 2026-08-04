#!/usr/bin/env python3
"""把 Stage 3 NPZ 的指定帧范围导出为 PLY 点云（双颜色：物体 + 手）。

用法：
    # 导出 epoch 3 实际采样的物体点
    python -m tools.stage3_npz_to_ply \
        --input data/processed_data/stage3/<variant>/s1/bowl_pass_1_right.npz \
        --object-view sampled --epoch 3 --frame-start 0 --frame-end 9

    # 默认 frame-start=0, frame-end=9
    python -m tools.stage3_npz_to_ply --input <npz>

输出：
    <output-dir>/<npz-stem>/frame_<T>_obj.ply
    <output-dir>/<npz-stem>/frame_<T>_hand.ply
    <output-dir>/<npz-stem>/frame_<T>_combined.ply    # 一帧手+物体合在一起
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from src.task.correspondence_ptv3.sampling import sample_object_indices, stable_frame_seed


# 两种固定颜色
COLOR_OBJ = (0.20, 0.55, 1.00)   # 蓝
COLOR_HAND = (0.20, 0.85, 0.30)  # 绿


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="Stage 3 训练 NPZ 路径",
    )
    p.add_argument(
        "--frame-start", type=int, default=0,
        help="起始帧（含），默认 0",
    )
    p.add_argument(
        "--frame-end", type=int, default=9,
        help="结束帧（含），默认 9",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="/home/oyx/test_ws/Ref2Dex/tmp/ply_visualize_stage3",
        help="输出根目录（默认 tmp/ply_visualize_stage3）",
    )
    p.add_argument(
        "--object-view",
        choices=["pool", "candidates", "sampled"],
        default="pool",
        help="导出全量4096、5cm候选，或指定epoch采样的512点",
    )
    p.add_argument("--epoch", type=int, default=0, help="object-view=sampled 时使用")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--num-obj-points", type=int, default=512)
    return p.parse_args()


def _scalar(payload: Any, key: str, default: str = "") -> str:
    if key not in payload:
        return default
    value = np.asarray(payload[key])
    return str(value.item()) if value.size == 1 else default


def _validate_npz(payload: Any) -> dict[str, Any]:
    """Validate the current Stage 3 point-pool schema."""
    required = {"obj_points", "hand_points", "obj_candidate_mask_5cm"}
    missing = required.difference(payload.files)
    if missing:
        raise KeyError(f"NPZ 缺少 Stage 3 字段：{sorted(missing)}")
    obj = np.asarray(payload["obj_points"])
    hand = np.asarray(payload["hand_points"])
    candidate = np.asarray(payload["obj_candidate_mask_5cm"])
    if obj.ndim != 3 or obj.shape[-1] != 3:
        raise ValueError(f"obj_points 应为 [T,N,3]，实际 {obj.shape}")
    if hand.ndim != 3 or hand.shape[0] != obj.shape[0] or hand.shape[-1] != 3:
        raise ValueError(f"hand_points 与 obj_points 不匹配：{hand.shape} / {obj.shape}")
    if candidate.shape != obj.shape[:2]:
        raise ValueError(f"obj_candidate_mask_5cm 形状错误：{candidate.shape}")
    return {
        "schema": "pool",
        "frames": obj.shape[0],
        "num_obj": obj.shape[1],
        "num_hand": hand.shape[1],
    }


def _new_schema_frame(
    payload: Any,
    frame: int,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, int]:
    obj = np.asarray(payload["obj_points"][frame], dtype=np.float32)
    hand = np.asarray(payload["hand_points"][frame], dtype=np.float32)
    padding = 0
    if args.object_view == "candidates":
        obj = obj[np.asarray(payload["obj_candidate_mask_5cm"][frame], dtype=bool)]
    elif args.object_view == "sampled":
        raw_frame_id = int(np.asarray(payload["raw_frame_id"])[frame])
        seed = stable_frame_seed(
            base_seed=args.base_seed,
            seq_id=_scalar(payload, "seq_id"),
            side=_scalar(payload, "side"),
            raw_frame_id=raw_frame_id,
            epoch=max(0, args.epoch),
        )
        selected, valid = sample_object_indices(
            np.asarray(payload["obj_candidate_mask_5cm"][frame], dtype=bool),
            num_samples=args.num_obj_points,
            seed=seed,
        )
        obj = obj[np.maximum(selected[valid], 0)]
        padding = int((~valid).sum())
    return obj, hand, padding


def _save_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """保存 binary little-endian PLY（XYZ 为 float32，RGB 为 uint8）。"""
    assert points.shape == colors.shape, f"{points.shape} vs {colors.shape}"
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    rgb_u8 = np.clip(np.asarray(colors) * 255.0, 0, 255).astype(np.uint8)
    pts_f32 = np.asarray(points, dtype=np.float32)
    if not np.isfinite(pts_f32).all():
        raise ValueError(f"PLY contains non-finite point coordinates: {path}")

    # PLY 是逐顶点交错布局。不能把 RGB 转成 float32 后直接 hstack：
    # header 声明 uchar 时，读取器会按 15 bytes/vertex 解析
    # (3 * float32 + 3 * uint8)，而 6 个 float32 实际是 24 bytes。
    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
        align=False,
    )
    vertices = np.empty(n, dtype=vertex_dtype)
    vertices["x"] = pts_f32[:, 0]
    vertices["y"] = pts_f32[:, 1]
    vertices["z"] = pts_f32[:, 2]
    vertices["red"] = rgb_u8[:, 0]
    vertices["green"] = rgb_u8[:, 1]
    vertices["blue"] = rgb_u8[:, 2]

    path.write_bytes(header + vertices.tobytes())


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到 NPZ: {input_path}")
    payload = np.load(str(input_path), allow_pickle=False)
    info = _validate_npz(payload)
    T = int(info["frames"])
    print(f"[stage3-ply] input  = {input_path}")
    print(
        f"[stage3-ply] schema={info['schema']} frames={T} "
        f"obj_pts={info['num_obj']} hand_pts={info['num_hand']}"
    )

    if args.frame_start < 0 or args.frame_end >= T:
        raise IndexError(
            f"--frame-start/--frame-end 越界：T={T}，合法范围 [0, {T - 1}]"
        )
    if args.frame_end < args.frame_start:
        raise ValueError(
            f"--frame-end ({args.frame_end}) 必须 >= --frame-start ({args.frame_start})"
        )

    out_root = Path(args.output_dir).resolve()
    out_dir = out_root / input_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    obj_color = np.asarray(COLOR_OBJ, dtype=np.float32)
    hand_color = np.asarray(COLOR_HAND, dtype=np.float32)

    saved = 0
    for t in range(args.frame_start, args.frame_end + 1):
        obj_pts, hand_pts, padding = _new_schema_frame(payload, t, args)
        obj_rgb = np.tile(obj_color, (obj_pts.shape[0], 1))
        hand_rgb = np.tile(hand_color, (hand_pts.shape[0], 1))
        combined = np.vstack([obj_pts, hand_pts])
        combined_rgb = np.vstack([obj_rgb, hand_rgb])

        out_obj = out_dir / f"frame_{t:04d}_obj.ply"
        out_hand = out_dir / f"frame_{t:04d}_hand.ply"
        out_comb = out_dir / f"frame_{t:04d}_combined.ply"
        _save_ply(out_obj, obj_pts, obj_rgb)
        _save_ply(out_hand, hand_pts, hand_rgb)
        _save_ply(out_comb, combined, combined_rgb)

        print(
            f"[stage3-ply] frame {t:4d} obj={obj_pts.shape[0]:4d} "
            f"hand={hand_pts.shape[0]:4d} padding={padding:3d} -> {out_comb}"
        )
        saved += 3

    payload.close()
    print(f"[stage3-ply] done. saved {saved} files under {out_dir}")


if __name__ == "__main__":
    main()
