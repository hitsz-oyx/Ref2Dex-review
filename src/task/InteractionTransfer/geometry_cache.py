"""V1.0 Stage4 Sequence Geometry Cache：MANO 顶点 + 静态拓扑 + 在线表面采样。

Stage4 原始 cache 只存 1538 个 face-center（``hand_points_world``）。由于
face-center 是 778 个顶点的线性映射 ``C = A·V``（A 为常量 face-vertex 平均
矩阵，rank(A)=778，cond≈10.6），可用伪逆从现有 cache **精确**重建每帧顶点
（实测残差 ~2e-7 m），无需重跑 raw GRAB MANO forward。

geometry cache 布局（object 侧继续由 stage4 提供，不重复存储）::

    <output>/
    ├── meta.json                     # 记录 stage4_root / 点数 / 序列统计
    ├── faces_right.npy               # [1538,3] int64，静态 MANO 拓扑
    ├── faces_left.npy
    └── s1/airplane_fly_1/
        ├── right.npz
        │   ├── hand_vertices_world   [T,778,3] float32
        │   └── active_mask           [T] bool   (obj_candidate_mask_5cm.any(1))
        └── left.npz                  # 同上

采样约定（V1.0 指导 §1）：同一 transition 的 t 与 t+g 必须使用同一组
face id + 同一组 barycentric 权重，``Δh = p_{t+g} - p_t`` 才是同一物理
表面点的运动；两端绝不允许独立采样。

构建::

    python -m src.task.InteractionTransfer.geometry_cache \
        --stage4-root data/processed_data/stage4/data/grab \
        --output data/processed_data/stage4/interactiontransfer_geometry_cache \
        [--sequences-file ...] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

NUM_MANO_VERTICES = 778
NUM_MANO_FACES = 1538
DEFAULT_MANO_PKL_DIR = Path(__file__).resolve().parents[3] / "dataset" / "arctic" / "data" / "body_models" / "mano"


# --------------------------------------------------------------------------- #
# MANO 拓扑
# --------------------------------------------------------------------------- #
def _load_mano_faces_from_pkl(pkl_path: Path) -> np.ndarray:
    """直接从 MANO pkl 读取静态拓扑（不依赖 smplx/chumpy 运行时）。"""
    # 新版 numpy 移除了这些别名，而旧 chumpy pickle 需要。
    for name, value in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("str", str), ("unicode", str),
                        ("nan", float("nan")), ("inf", float("inf"))):
        if not hasattr(np, name):
            setattr(np, name, value)
    with open(pkl_path, "rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    faces = np.asarray(data["f"], dtype=np.int64)
    if faces.shape != (NUM_MANO_FACES, 3):
        raise ValueError(f"{pkl_path}: faces 形状 {faces.shape}，预期 ({NUM_MANO_FACES}, 3)")
    return faces


def load_mano_faces(mano_pkl_dir: str | Path = DEFAULT_MANO_PKL_DIR) -> dict[str, np.ndarray]:
    return {side: _load_mano_faces_from_pkl(Path(mano_pkl_dir) / f"MANO_{side.upper()}.pkl")
            for side in ("left", "right")}


@lru_cache(maxsize=4)
def _face_center_pinv(faces_bytes: bytes) -> np.ndarray:
    """A [F,778] 的伪逆 [778,F]；faces 以 bytes 作 cache key。"""
    faces = np.frombuffer(faces_bytes, dtype=np.int64).reshape(-1, 3)
    A = np.zeros((faces.shape[0], NUM_MANO_VERTICES), dtype=np.float64)
    rows = np.repeat(np.arange(faces.shape[0]), 3)
    A[rows, faces.ravel()] = 1.0 / 3.0
    return np.linalg.pinv(A)  # rank 778，解唯一


def reconstruct_vertices(face_centers: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """face-center [T,1538,3] -> 顶点 [T,778,3]（精确线性重建）。"""
    P = _face_center_pinv(faces.tobytes())
    vertices = np.einsum("vc,tcx->tvx", P, face_centers.astype(np.float64))
    return vertices.astype(np.float32)


# --------------------------------------------------------------------------- #
# 在线随机表面采样（uniform faces + Dirichlet barycentric）
# --------------------------------------------------------------------------- #
def sample_surface_refs(num_faces: int, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """随机采样 count 个 (face_idx, barycentric) 引用；t 与 t+g 复用同一组。"""
    face_idx = rng.integers(0, num_faces, size=count)
    bary = rng.dirichlet(np.ones(3), size=count).astype(np.float32)
    return face_idx.astype(np.int64), bary


def gather_surface_points(vertices: np.ndarray, faces: np.ndarray,
                          face_idx: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """p = b0·v0 + b1·v1 + b2·v2；vertices [778,3] -> points [N,3]。"""
    tri = vertices[faces[face_idx]]              # [N,3,3]
    return np.einsum("nc,ncx->nx", bary, tri).astype(np.float32)


# --------------------------------------------------------------------------- #
# cache 构建
# --------------------------------------------------------------------------- #
def build_sequence(stage4_root: Path, output_root: Path, sequence: str,
                   faces: dict[str, np.ndarray], residual_check: bool = False) -> int:
    seq_dir = stage4_root / sequence
    out_dir = output_root / sequence
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for side in ("left", "right"):
        dst = out_dir / f"{side}.npz"
        with np.load(seq_dir / f"{side}.npz", allow_pickle=False) as hand:
            centers = np.asarray(hand["hand_points_world"], dtype=np.float32)
            active = np.asarray(hand["obj_candidate_mask_5cm"], dtype=bool).any(1)
        vertices = reconstruct_vertices(centers, faces[side])
        if residual_check:
            residual = float(np.abs(vertices[:, faces[side], :].mean(2) - centers).max())
            if residual > 1e-5:
                raise RuntimeError(f"{sequence}/{side}: 顶点重建残差 {residual:.3e} > 1e-5")
        np.savez(dst, hand_vertices_world=vertices, active_mask=active)
        written += 1
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequences-file", type=Path, default=None,
                        help="可选：每行一个 s*/seq；默认处理 stage4 下全部序列")
    parser.add_argument("--mano-pkl-dir", type=Path, default=DEFAULT_MANO_PKL_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage4_root, output = Path(args.stage4_root), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    faces = load_mano_faces(args.mano_pkl_dir)
    np.save(output / "faces_left.npy", faces["left"])
    np.save(output / "faces_right.npy", faces["right"])

    if args.sequences_file:
        sequences = [line.strip() for line in args.sequences_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        sequences = sorted(p.parent.relative_to(stage4_root).as_posix()
                           for p in stage4_root.glob("*/*/shared.npz"))
    if not sequences:
        raise FileNotFoundError(f"{stage4_root} 下没有 shared.npz")

    meta_path = output / "meta.json"
    meta = {"stage4_root": str(stage4_root.resolve()), "num_vertices": NUM_MANO_VERTICES,
            "num_faces": NUM_MANO_FACES, "sequences": {}}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    for done, sequence in enumerate(sequences, 1):
        if args.skip_existing and sequence in meta["sequences"]:
            continue
        n = build_sequence(stage4_root, output, sequence, faces, residual_check=(done == 1))
        with np.load(output / sequence / "right.npz", allow_pickle=False) as r:
            frames = int(r["hand_vertices_world"].shape[0])
        meta["sequences"][sequence] = frames
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[{done}/{len(sequences)}] {sequence}: {frames} frames", flush=True)

    total = sum(meta["sequences"].values())
    print(f"geometry cache 完成：{len(meta['sequences'])} sequences, {total} frames -> {output}")


if __name__ == "__main__":
    main()
