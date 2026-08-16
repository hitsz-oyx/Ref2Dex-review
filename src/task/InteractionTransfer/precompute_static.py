"""V0.8 离线预计算 frozen DenseToken static cache。

对每个 right-active/left-inactive 的 one-step transition 起始帧，用
``InteractionTransfer.encode_static`` 一次性算出冻结静态特征并落盘；
后续 20 epoch 训练直接读 cache，不再重复跑 PTv3。

cache 布局（每个 sequence 一组 .npy，fp32/int64/bool 原始格式，可 mmap）::

    <output>/
    ├── meta.json
    └── s1__airplane_fly_1.object_points.npy      [T,512,3]
        s1__airplane_fly_1.object_normals.npy     [T,512,3]
        s1__airplane_fly_1.edge_idx.npy           [T,512,16] int64
        s1__airplane_fly_1.edge_valid.npy         [T,512,16] bool
        s1__airplane_fly_1.dense_edge.npy         [T,512,16,32]
        s1__airplane_fly_1.geom_contact.npy       [T,512,16,14]
        s1__airplane_fly_1.hand_flow.npy          [T,1538,3]
        s1__airplane_fly_1.object_flow.npy        [T,512,3]

结尾自动抽样做 sanity check：raw forward 与 cached forward_core 的
object_flow 最大绝对误差应 < 1e-5。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.InteractionTransfer.cached_dataset import ARRAYS, cache_file_name
from src.task.InteractionTransfer.dataset import fixed_point_indices
from src.task.InteractionTransfer.model import InteractionTransfer

STATIC_KEYS = ("object_points", "object_normals", "edge_idx", "edge_valid", "dense_edge", "geom_contact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="GRAB stage4 cache 根目录")
    parser.add_argument("--output", required=True, help="static cache 输出目录")
    parser.add_argument("--sequences-file", type=Path, default=None,
                        help="可选：只处理该文件列出的 sequence（每行 s*/seq）；默认全部")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-batch", type=int, default=16)
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--skip-existing", action="store_true", help="跳过已完成的 sequence（增量续算）")
    parser.add_argument("--sanity-samples", type=int, default=4, help="结尾 sanity check 抽样 sequence 数")
    return parser.parse_args()


def load_split_sequences(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def load_sequence(seq_dir: Path):
    with np.load(seq_dir / "shared.npz") as s, np.load(seq_dir / "right.npz") as r, \
            np.load(seq_dir / "left.npz") as l:
        raw_frame_id = np.asarray(s["raw_frame_id"], dtype=np.int64)
        if float(np.asarray(s["source_fps"]).item()) != 120.0 or not np.all(np.diff(raw_frame_id) == 4):
            raise ValueError(f"{seq_dir}: raw_frame_id/source_fps 与 dataset 校验不一致")
        right_active = np.asarray(r["obj_candidate_mask_5cm"], dtype=bool).any(1)
        left_active = np.asarray(l["obj_candidate_mask_5cm"], dtype=bool).any(1)
        starts = np.nonzero(right_active[:-1] & right_active[1:] & ~left_active[:-1] & ~left_active[1:])[0]
        arrays = (s["obj_points_world"], s["obj_normals_world"],
                  r["hand_points_world"], r["hand_normals_world"])
    return starts, arrays


def chunk_inputs(op, on, hp, hn, idx, obj_idx, hand_idx) -> dict[str, np.ndarray]:
    """与 GRABOneStepDataset.__getitem__ 完全一致的输入构建（batch 版）。"""
    nxt = idx + 1
    o = op[np.ix_(idx, obj_idx)].astype("f4")
    center = o.mean(1, keepdims=True)
    return {
        "object_points": o - center,
        "object_normals": on[np.ix_(idx, obj_idx)].astype("f4"),
        "hand_points": hp[np.ix_(idx, hand_idx)].astype("f4") - center,
        "hand_normals": hn[np.ix_(idx, hand_idx)].astype("f4"),
        "hand_flow": (hp[np.ix_(nxt, hand_idx)] - hp[np.ix_(idx, hand_idx)]).astype("f4"),
        "object_flow": (op[np.ix_(nxt, obj_idx)] - op[np.ix_(idx, obj_idx)]).astype("f4"),
    }


@torch.no_grad()
def compute_sequence(model: InteractionTransfer, seq_dir: Path, device: torch.device,
                     obj_idx: np.ndarray, frame_batch: int) -> dict[str, np.ndarray] | None:
    starts, (op, on, hp, hn) = load_sequence(seq_dir)
    if len(starts) == 0:
        return None
    hand_idx = fixed_point_indices(hp.shape[1], model.static_encoder.num_hand_points)
    chunks = []
    for begin in range(0, len(starts), frame_batch):
        idx = starts[begin:begin + frame_batch]
        raw = chunk_inputs(op, on, hp, hn, idx, obj_idx, hand_idx)
        batch = {k: torch.from_numpy(raw[k]).to(device) for k in ("object_points", "object_normals",
                                                                  "hand_points", "hand_normals")}
        static = model.encode_static(**batch)
        chunk = {key: static[key].cpu().numpy() for key in STATIC_KEYS}
        chunk["hand_flow"] = raw["hand_flow"]
        chunk["object_flow"] = raw["object_flow"]
        chunks.append(chunk)
    return {key: np.concatenate([c[key] for c in chunks], 0) for key in ARRAYS}


@torch.no_grad()
def sanity_check(model: InteractionTransfer, root: Path, output: Path, meta: dict,
                 device: torch.device, obj_idx: np.ndarray, samples: int) -> float:
    names = sorted(n for n, c in meta["sequences"].items() if c > 0)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(names), size=min(samples, len(names)), replace=False)
    worst = 0.0
    for pick in picks:
        name = names[int(pick)]
        starts, (op, on, hp, hn) = load_sequence(root / name)
        hand_idx = fixed_point_indices(hp.shape[1], model.static_encoder.num_hand_points)
        idx = starts[:8]
        raw = chunk_inputs(op, on, hp, hn, idx, obj_idx, hand_idx)
        batch = {k: torch.from_numpy(raw[k]).to(device) for k in raw}
        raw_out = model(batch["object_points"], batch["object_normals"], batch["hand_points"],
                        batch["hand_normals"], batch["hand_flow"])
        cached = {key: torch.from_numpy(np.load(output / cache_file_name(name, key),
                                                mmap_mode="r")[:len(idx)]).to(device) for key in STATIC_KEYS}
        cached_out = model.forward_core(**cached, hand_flow=batch["hand_flow"])
        diff = float((raw_out["object_flow"] - cached_out["object_flow"]).abs().max())
        print(f"[sanity] {name}: max|ΔO_raw - ΔO_cached| = {diff:.3e}", flush=True)
        worst = max(worst, diff)
    return worst


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    sequences = load_split_sequences(args.sequences_file) if args.sequences_file else None
    seq_dirs = sorted(p.parent for p in root.glob("*/*/shared.npz") if (p.parent / "right.npz").is_file())
    if sequences is not None:
        wanted = set(sequences)
        seq_dirs = [d for d in seq_dirs if d.relative_to(root).as_posix() in wanted]
        missing = wanted - {d.relative_to(root).as_posix() for d in seq_dirs}
        if missing:
            raise FileNotFoundError(f"{len(missing)} 个 sequence 不在 raw cache 中，例如 {sorted(missing)[:3]}")

    meta_path = output / "meta.json"
    meta = {"config": None, "sequences": {}}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    model = InteractionTransfer(dense_checkpoint=args.dense_checkpoint).to(device).eval()
    obj_idx = fixed_point_indices(4096, model.static_encoder.num_obj_points)

    done = 0
    for seq_dir in seq_dirs:
        name = seq_dir.relative_to(root).as_posix()
        if args.skip_existing and name in meta["sequences"]:
            done += 1
            continue
        arrays = compute_sequence(model, seq_dir, device, obj_idx, args.frame_batch)
        if arrays is None:
            print(f"[skip] {name}: 无 right-active transition", flush=True)
            meta["sequences"][name] = 0
        else:
            for key, value in arrays.items():
                np.save(output / cache_file_name(name, key), value)
            meta["sequences"][name] = int(arrays["object_flow"].shape[0])
        if meta["config"] is None:
            meta["config"] = {"object_points": int(model.static_encoder.num_obj_points),
                              "hand_points": int(model.static_encoder.num_hand_points),
                              "k": model.k, "radius": model.radius, "gap": 1,
                              "dense_checkpoint": args.dense_checkpoint}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        done += 1
        total_t = sum(meta["sequences"].values())
        print(f"[{done}/{len(seq_dirs)}] {name}: {meta['sequences'][name]} transitions (total {total_t})", flush=True)

    total = sum(v for v in meta["sequences"].values() if v)
    print(f"cache 完成：{len(meta['sequences'])} sequences, {total} transitions -> {output}", flush=True)

    if args.sanity_samples > 0 and total > 0:
        worst = sanity_check(model, root, output, meta, device, obj_idx, args.sanity_samples)
        if worst >= 1e-5:
            raise RuntimeError(f"sanity check 失败：max abs diff {worst:.3e} >= 1e-5")
        print(f"sanity check 通过：worst {worst:.3e} < 1e-5", flush=True)


if __name__ == "__main__":
    main()
