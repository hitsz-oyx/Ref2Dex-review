#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAB → GeneOH-Diffusion denoising → Ref2Dex Stage 2 .pkl 流水线
====================================================================

本脚本仿造 ``process/GRAB/optimize.py`` 的结构（CLI / 输出路径 / 字段 / 统计），
**只把 obj 轨迹的生成方式**从 Ref2Dex 的 ``processing_mode='init_only'``
替换为 GeneOH-Diffusion 的扩散去噪（motion_diff + spatial_diff），
手部沿用 Ref2Dex 的 ``GRABRawAdapter`` 准备 baseline（同样的 MANO
forward / vtemplate 注入逻辑），保证下游消费者读到的 Stage 2 .pkl
字段与 ``process/GRAB/optimize.py`` 完全一致。

----------------------------------------------------------------------
GeneOH-Diffusion 去噪的范围与前提
----------------------------------------------------------------------
1. **范围**：只 denoise 物体 SE(3) 轨迹（GeneOH 最稳定的部分），
   手部姿态不做改动（仍由 ``GRABRawAdapter`` 给出 init baseline）。
2. **去噪步骤**：依次跑
     - ``sample.predict_grab_all_seq``（motion_diff，时间维度）
     - ``sample.predict_grab_spatial_all_seq``（spatial_diff，空间维度）
3. **数据前提**：
     - GRAB 原始 .npz 由 Ref2Dex 的 ``GRABRawAdapter`` 读取（与
       ``process/GRAB/optimize.py`` 完全一致）。
     - GeneOH 自己的预处理数据 ``data/grab/GRAB_processed/{split}/``
       以及 ``data/grab/object_meshes/`` 由用户从 GeneOH README 的
       OneDrive / Google Drive 链接下载并放到 ``--geneoh-root`` 下。
     - GeneOH 预训练权重（``ckpts/model.pt`` + ``ckpts/model_spatial.pt``）
       由用户放到 ``--geneoh-root/ckpts/`` 下。
4. **运行约束**：GeneOH 自带 conda 环境 (``DeepMetaHandles5``)，
   脚本通过 ``--geneoh-env`` 参数指定 conda env name（默认
   ``DeepMetaHandles5``），subprocess 内用 ``conda run -n <env>`` 激活。

----------------------------------------------------------------------
GeneOH seq_id ↔ GRAB raw .npz 的映射
----------------------------------------------------------------------
GeneOH 的预处理数据以 ``<seq_id>.npy`` 形式命名（test split 形如
``1.npy, 2.npy, ..., 14.npy``），其索引是 GeneOH 仓库内 test split
列表中的顺序。本脚本采用如下映射策略：

    - 优先按 ``--geneoh-seq-map`` 指定的 JSON 文件查找（格式
      ``{"<geneoh_seq_id>": "<ref2dex_seq_id>", ...}``，例如
      ``{"14": "s1/bowl_pass_1"}``）。如不提供则用下面启发式。
    - 启发式：取 raw .npz 文件名中第一个下划线前后的 ``<action>``，
      把它作为 ``geneoh_seq_id`` 候选（用户可写自定义规则）。

    简单场景下，**先用 ``--list-mapping`` 模式打印当前启发式下
    找到的映射**，根据打印结果决定是否要写 ``--geneoh-seq-map``
    JSON 显式指定。

----------------------------------------------------------------------
输出
----------------------------------------------------------------------
与 ``process/GRAB/optimize.py`` 完全一致：

    <output_root>/<subject_id>/<seq_name>_<side>.pkl
    <output_root>/meta.json

``.pkl`` schema 由 ``process.common.stage2.pack_stage2_hand`` 给出；
``processing_mode`` 字段写为 ``geneoh_denoised``（区别于
``init_only``），并把 GeneOH 的去噪参数写入 ``config`` 字段。
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as op
import subprocess
import time
import traceback
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    ROOT / "data" / "processed_data" / "stage2" / "grab_geneoh_4096"
)
# GeneOH-Diffusion 仓库根（与 Ref2Dex 平级，默认在 ~/test_ws/GeneOH-Diffusion）
DEFAULT_GENEOH_ROOT = ROOT.parent / "GeneOH-Diffusion"
DEFAULT_GENEOH_ENV = "DeepMetaHandles5"
DEFAULT_GENEOH_SEQ_ROOT = "data/grab/GRAB_processed"
DEFAULT_GENEOH_SUBJ_ROOT = "data/grab/GRAB_processed_wsubj"
DEFAULT_GENEOH_OBJ_MESH_ROOT = "data/grab/object_meshes"
DEFAULT_GENEOH_MODEL_PATH = "ckpts/model.pt"
DEFAULT_GENEOH_SPATIAL_MODEL_PATH = "ckpts/model_spatial.pt"


# ---------------------------------------------------------------------------
# 工具：延迟导入 Ref2Dex 的公共 Stage 2 schema 和 GRAB raw adapter
# ---------------------------------------------------------------------------
def _import_ref2dex() -> Tuple[Any, Any]:
    """延迟导入 Ref2Dex 内部模块，避免提前加载 torch/trimesh 等大依赖。"""
    from process.common.stage2 import (  # type: ignore
        pack_stage2_hand,
        save_stage2_payload,
        write_stage2_meta,
    )
    from process.GRAB.raw import (  # type: ignore
        DEFAULT_GRAB_ROOT,
        DEFAULT_MANO_MODEL_DIR,
        GRABRawAdapter,
        load_manifest_seq_paths,
    )

    return (
        (pack_stage2_hand, save_stage2_payload, write_stage2_meta),
        (DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABRawAdapter,
         load_manifest_seq_paths),
    )


# ===========================================================================
# (1) GeneOH seq_id → GRAB raw .npz 映射
# ===========================================================================
def load_seq_map(path: Optional[str]) -> Dict[str, str]:
    """读取 ``--geneoh-seq-map`` 指定的 JSON 映射文件。"""
    if not path:
        return {}
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"--geneoh-seq-map not found: {p}")
    with p.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"--geneoh-seq-map must be a JSON dict, got {type(data)}")
    return {str(k): str(v) for k, v in data.items()}


def guess_geneoh_seq_id(raw_path: Path) -> str:
    """启发式：从 raw .npz 文件名反推 GeneOH seq_id。

    GeneOH 的 test split 形如 ``1.npy, 2.npy, ..., 14.npy``，其 seq_id
    来自 GRAB 内部 test list 的顺序。本函数只提供一个 fallback：直接用
    action 名（不含 subject 前缀）。用户应当用 ``--geneoh-seq-map`` 显式
    指定，否则可能匹配不到。
    """
    return raw_path.stem  # e.g. "bowl_pass_1"


# ===========================================================================
# (2) 解析序列列表（与 optimize.py 的选择规则一致）
# ===========================================================================
def _resolve_sequences(
    args: argparse.Namespace,
    load_manifest_seq_paths: Any,
) -> List[str]:
    if args.raw_file:
        return [str(Path(args.raw_file).resolve())]
    if args.manifest:
        return load_manifest_seq_paths(args.manifest, args.grab_root)
    sequences = sorted(glob(op.join(args.grab_root, "grab", "*", "*.npz")))
    if not args.seq:
        return sequences
    target = args.seq[:-4] if args.seq.endswith(".npz") else args.seq
    selected: List[str] = []
    for path in sequences:
        rel = (
            Path(path)
            .relative_to(Path(args.grab_root) / "grab")
            .with_suffix("")
            .as_posix()
        )
        if rel == target or target in rel:
            selected.append(path)
    return selected


# ===========================================================================
# (3) GeneOH denoising 调起（subprocess）
# ===========================================================================
def _run_geneoh_predict(
    *,
    geneoh_root: Path,
    geneoh_env: str,
    split: str,
    single_seq_path: str,
    save_dir: Path,
    model_path: str,
    spatial_model_path: str,
    grab_path: str,
    grab_processed_dir: str,
    cuda_ids: str,
    pert_type: str,
    stage: str = "spatial",
    extra_overrides: Optional[Dict[str, str]] = None,
) -> None:
    """调起 GeneOH 的预测脚本一次（motion 或 spatial 阶段）。

    通过 ``conda run -n <env>`` 在指定 conda env 内跑 Python，因此本进程
    不必激活 GeneOH 的 DeepMetaHandles5 环境。

    Parameters
    ----------
    stage : ``"motion"`` 调 ``sample.predict_grab_all_seq``（时间维度去噪），
            ``"spatial"`` 调 ``sample.predict_grab_spatial_all_seq``（空间维度精修）
    extra_overrides : 额外覆盖 GeneOH 默认超参
    """
    if stage == "motion":
        sample_module = "sample.predict_grab_all_seq"
    elif stage == "spatial":
        sample_module = "sample.predict_grab_spatial_all_seq"
    else:
        raise ValueError(f"stage must be 'motion' or 'spatial', got {stage}")

    # GeneOH 脚本的命令行超长（130+ 个 flag），全部从 scripts/val/*.sh 抄过来
    # 再按需覆盖。默认对应 ``scripts/val/predict_grab_rndseed_spatial.sh``。
    base_cmd = [
        "conda", "run", "-n", geneoh_env, "--no-capture-output",
        "python", "-m", sample_module,
        "--dataset", "motion_ours",
        "--save_dir", str(save_dir),
        "--single_seq_path", str(single_seq_path),
        "--window_size", "60",
        "--unconstrained",
        "--inst_normalization",
        "--model_path", model_path,
        "--model_spatial_path", spatial_model_path,
        "--rep_type", "obj_base_rel_dist_we_wj_latents",
        "--batch_size", "1",
        "--denoising_stra", "rep",
        "--seed", "77",
        "--diff_basejtsrel",
        "--use_sep_models",
        "--jts_sclae_stra", "std",
        "--without_dec_pos_emb",
        "--pred_diff_noise",
        "--not_load_opt",
        "--deep_fuse_timeemb",
        "--use_ours_transformer_enc",
        "--set_attn_to_none",
        "--wo_rel_normalization",
        "--use_dec_rel_v2",
        "--pred_basejtsrel_avgjts",
        "--use_t", "400",
        "--not_cond_base",
        "--not_pred_avg_jts",
        "--latent_dim", "512",
        "--diff_spatial",
        "--noise_schedule", "linear",
        "--pred_joints_offset",
        "--not_diff_avgjts",
        "--joint_std_v3",
        "--use_var_sched",
        "--e_normalization_stra", "cent",
        "--real_basejtsrel_norm_stra", "none",
        "--use_objbase_v5",
        "--use_objbase_out_v5",
        "--out_objbase_v5_bundle_out",
        "--add_noise_onjts",
        "--v5_out_not_cond_base",
        "--v5_in_not_base",
        "--v5_in_without_glb",
        "--nn_base_pts", "700",
        "--pert_type", pert_type,
        "--test_tag", "geneoh_ref2dex",
        "--theta_dim", "24",
        "--start_idx", "0",
        "--scale_obj", "1",
        "--seq_root", str(Path(grab_processed_dir) / split),
        "--grab_path", str(grab_path),
        "--grab_processed_dir", str(grab_processed_dir),
    ]
    if extra_overrides:
        for k, v in extra_overrides.items():
            # 若 key 以 -- 开头，保留；否则加上 --
            key = k if k.startswith("--") else f"--{k}"
            if v == "__REMOVE__":
                # 标记：从 base_cmd 里删掉该 flag
                target = key
                while target in base_cmd:
                    idx = base_cmd.index(target)
                    # flag 后可能跟一个 value，删 1 或 2 个元素
                    if idx + 1 < len(base_cmd) and not base_cmd[idx + 1].startswith("--"):
                        del base_cmd[idx : idx + 2]
                    else:
                        del base_cmd[idx]
                continue
            base_cmd += [key, v]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_ids)
    env["PYTHONPATH"] = (
        str(geneoh_root) + os.pathsep + env.get("PYTHONPATH", "")
    )

    print(f"[geneoh] cwd={geneoh_root}  cmd='python -m sample.predict_grab_spatial_all_seq ...'")
    subprocess.run(base_cmd, cwd=str(geneoh_root), env=env, check=True)


def run_geneoh_denoise_two_stage(
    *,
    raw_path: str,
    geneoh_seq_id: str,
    split: str,
    geneoh_root: Path,
    geneoh_env: str,
    geneoh_processed_dir: Path,
    geneoh_model_path: str,
    geneoh_spatial_model_path: str,
    grab_path: str,
    cuda_ids: str,
    pert_type: str,
    denoise_cache_dir: Path,
) -> Path:
    """两步去噪：先 motion_diff，再用 spatial_diff 精修。

    Returns
    -------
    spatial 结果的 save_dir 路径（其中包含 denoised 的 .npy / .pkl）
    """
    denoise_cache_dir.mkdir(parents=True, exist_ok=True)
    save_dir_motion = denoise_cache_dir / f"{geneoh_seq_id}_motion"
    save_dir_spatial = denoise_cache_dir / f"{geneoh_seq_id}_spatial"
    if save_dir_spatial.exists() and any(save_dir_spatial.glob("*.npy")):
        print(f"[geneoh] cache hit: {save_dir_spatial}")
        return save_dir_spatial

    single_seq_path = str(
        geneoh_processed_dir / split / f"{geneoh_seq_id}.npy"
    )
    if not Path(single_seq_path).exists():
        raise FileNotFoundError(
            f"GeneOH preprocessed .npy not found: {single_seq_path}\n"
            f"请先按 GeneOH README 把 GRAB_processed 数据解压到 "
            f"{geneoh_processed_dir} 下"
        )

    # ----- Step A: motion_diff（先跑时间维度去噪）-----
    _run_geneoh_predict(
        geneoh_root=geneoh_root,
        geneoh_env=geneoh_env,
        split=split,
        single_seq_path=single_seq_path,
        save_dir=save_dir_motion,
        model_path=geneoh_model_path,
        spatial_model_path=geneoh_spatial_model_path,
        grab_path=grab_path,
        grab_processed_dir=str(geneoh_processed_dir),
        cuda_ids=cuda_ids,
        pert_type=pert_type,
        stage="motion",
        # motion 阶段不加 --diff_spatial（base_cmd 里默认带，从 cmd 里移除）
        extra_overrides={"--diff_spatial": "__REMOVE__"},
    )

    # ----- Step B: spatial_diff（基于 motion 结果做空间维度精修）-----
    _run_geneoh_predict(
        geneoh_root=geneoh_root,
        geneoh_env=geneoh_env,
        split=split,
        single_seq_path=single_seq_path,
        save_dir=save_dir_spatial,
        model_path=geneoh_model_path,
        spatial_model_path=geneoh_spatial_model_path,
        grab_path=grab_path,
        grab_processed_dir=str(geneoh_processed_dir),
        cuda_ids=cuda_ids,
        pert_type=pert_type,
        stage="spatial",
        # spatial 阶段需找到 motion 阶段的输出目录
        extra_overrides={"prev_test_tag": "geneoh_ref2dex"},
    )

    return save_dir_spatial


# ===========================================================================
# (4) 解析 GeneOH 输出 → 替换 source dict 的 obj 轨迹
# ===========================================================================
def _load_geneoh_obj_se3(
    save_dir: Path,
    geneoh_seq_id: str,
    n_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """从 GeneOH spatial 结果目录里读出 denoised obj (rotvec, transl)。

    Returns
    -------
    obj_rot_aa : (T, 3) float32   # axis-angle
    obj_transl : (T, 3) float32
    """
    candidate = save_dir / f"{geneoh_seq_id}_obj_orient_transl.npy"
    if not candidate.exists():
        # GeneOH 实际命名不固定，尝试 glob
        matches = list(save_dir.glob(f"*{geneoh_seq_id}*orient*.npy")) + list(
            save_dir.glob(f"*{geneoh_seq_id}*transl*.npy")
        )
        if not matches:
            raise FileNotFoundError(
                f"GeneOH spatial output not found under {save_dir} for "
                f"{geneoh_seq_id}. 请检查 {save_dir} 下的输出文件命名。"
            )
        candidate = matches[0]
    arr = np.load(candidate, allow_pickle=True)
    # 期望 shape: (T, 6) = [rotvec(3), transl(3)]；某些版本会分开存
    if arr.ndim == 2 and arr.shape[-1] == 6:
        return arr[:, :3].astype(np.float32), arr[:, 3:].astype(np.float32)
    raise ValueError(
        f"Unexpected GeneOH output shape {arr.shape} from {candidate}；"
        f"需要 (T, 6)。请检查 GeneOH 输出命名。"
    )


def apply_denoised_obj(
    source: Dict[str, Any],
    obj_rot_aa: np.ndarray,
    obj_transl: np.ndarray,
    obj_points_cano: np.ndarray,
    obj_normals_cano: np.ndarray,
) -> None:
    """把 denoised obj SE(3) 写回 source dict，**就地修改**。

    同步更新：
        - obj_root_pose  (T, 4, 4)
        - obj_points_world / obj_points  (T, N, 3)
        - obj_normals_world / obj_normals  (T, N, 3)
        - obj_flow / obj_flow_valid
        - obj_to_*_nn_id / obj_to_*_dist （需要从世界坐标重算）
    """
    T = obj_rot_aa.shape[0]
    if T != source["obj_root_pose"].shape[0]:
        # GeneOH 默认 window=60，长序列是分段处理后 concat；
        # 若长度不匹配，做线性裁剪到 T_target。
        T_target = source["obj_root_pose"].shape[0]
        if T >= T_target:
            obj_rot_aa = obj_rot_aa[:T_target]
            obj_transl = obj_transl[:T_target]
        else:
            pad_rot = np.tile(obj_rot_aa[-1:], (T_target - T, 1))
            pad_trans = np.tile(obj_transl[-1:], (T_target - T, 1))
            obj_rot_aa = np.concatenate([obj_rot_aa, pad_rot], axis=0)
            obj_transl = np.concatenate([obj_transl, pad_trans], axis=0)
        T = T_target

    # rotmat + 4x4
    # 复用 grab_preprocess 的 axis_angle_to_rotmat 实现
    from process.GRAB.raw import axis_angle_to_rotmat  # type: ignore

    rotmat = axis_angle_to_rotmat(
        torch.from_numpy(obj_rot_aa).float()
    ).cpu().numpy()
    obj_root_pose = np.zeros((T, 4, 4), dtype=np.float32)
    obj_root_pose[:, :3, :3] = rotmat
    obj_root_pose[:, :3, 3] = obj_transl
    obj_root_pose[:, 3, 3] = 1.0

    # 用更新后的 SE(3) 重新变换 canonical obj 表面点
    from process.GRAB.raw import (  # type: ignore
        transform_object_points_batch,
    )

    obj_points_world, obj_normals_world = transform_object_points_batch(
        obj_points_cano,
        obj_normals_cano,
        obj_rot_aa,
        obj_transl,
        device="cpu",  # 单序列计算量小，CPU 足矣
    )
    obj_points_world = obj_points_world.astype(np.float32)
    obj_normals_world = obj_normals_world.astype(np.float32)

    # flow
    obj_flow = np.zeros_like(obj_points_world)
    if T > 1:
        obj_flow[:-1] = obj_points_world[1:] - obj_points_world[:-1]
    obj_flow_valid = np.ones((T, obj_points_world.shape[1]), dtype=bool)
    obj_flow_valid[-1] = False

    # 写回
    source["obj_root_pose"] = obj_root_pose
    source["obj_points_world"] = obj_points_world
    source["obj_normals_world"] = obj_normals_world
    source["obj_points"] = obj_points_world
    source["obj_normals"] = obj_normals_world
    source["obj_flow"] = obj_flow
    source["obj_flow_valid"] = obj_flow_valid

    # 重新计算 obj_to_hand NN（基于更新后的 obj_points）
    try:
        from scipy.spatial import cKDTree

        for side in ("right", "left"):
            key_world = f"{side}_hand_points_world"
            if key_world not in source:
                continue
            hand_pts = source[key_world]
            new_nn_id = np.zeros((T, hand_pts.shape[1]), dtype=np.int32)
            new_dist = np.zeros((T, hand_pts.shape[1]), dtype=np.float32)
            for t in range(T):
                tree = cKDTree(obj_points_world[t])
                d, idx = tree.query(hand_pts[t], k=1)
                new_nn_id[t] = idx.astype(np.int32)
                new_dist[t] = d.astype(np.float32)
            source[f"obj_to_{side}_hand_nn_id"] = new_nn_id
            source[f"obj_to_{side}_hand_dist"] = new_dist
    except Exception as exc:
        print(f"[warn] obj-to-hand NN recompute failed: {exc}")


# ===========================================================================
# (5) 主流程
# ===========================================================================
def main() -> None:
    (
        (pack_stage2_hand, save_stage2_payload, write_stage2_meta),
        (DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR, GRABRawAdapter,
         load_manifest_seq_paths),
    ) = _import_ref2dex()

    parser = argparse.ArgumentParser(
        description=(
            "GRAB raw .npz → GeneOH-Diffusion denoising (motion+spatial) → "
            "Ref2Dex Stage 2 .pkl"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- 与 process/GRAB/optimize.py 一致的输入/输出参数 ----
    parser.add_argument("--grab-root", default=DEFAULT_GRAB_ROOT)
    parser.add_argument("--mano-path", default=DEFAULT_MANO_MODEL_DIR)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--raw-file", default=None)
    parser.add_argument(
        "--seq", default=None,
        help="Exact or substring sequence filter, e.g. s1/bowl_pass_1",
    )
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--side", choices=["left", "right", "both"], default="both",
    )
    parser.add_argument("--num-obj-points", type=int, default=4096)
    parser.add_argument("--frame-keep-threshold", type=float, default=0.05)
    parser.add_argument("--ds-rate", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nn-batch-size", type=int, default=8)
    parser.add_argument("--obj-unit", choices=["m", "mm", "auto"], default="m")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split", choices=["test", "val", "train"], default="test",
        help="对应 GeneOH 的 GRAB_processed 子目录",
    )

    # ---- GeneOH 专属参数 ----
    parser.add_argument(
        "--geneoh-root", default=str(DEFAULT_GENEOH_ROOT),
        help="GeneOH-Diffusion 仓库根目录（含 sample/ ckpts/ data/）",
    )
    parser.add_argument(
        "--geneoh-env", default=DEFAULT_GENEOH_ENV,
        help="GeneOH 的 conda 环境名（默认 DeepMetaHandles5）",
    )
    parser.add_argument(
        "--geneoh-processed-dir", default=None,
        help=(
            "GeneOH 预处理数据根（包含 test/ val/ train/ 子目录的父目录）。"
            " 默认 <geneoh-root>/data/grab/GRAB_processed"
        ),
    )
    parser.add_argument(
        "--geneoh-grab-path", default=None,
        help="GRAB 原始数据根（GeneOH 内部也需要，包含 tools/object_meshes/）",
    )
    parser.add_argument(
        "--geneoh-model-path", default=DEFAULT_GENEOH_MODEL_PATH,
        help="相对 geneoh-root 的 motion_diff ckpt 路径",
    )
    parser.add_argument(
        "--geneoh-spatial-model-path", default=DEFAULT_GENEOH_SPATIAL_MODEL_PATH,
        help="相对 geneoh-root 的 spatial_diff ckpt 路径",
    )
    parser.add_argument(
        "--geneoh-seq-map", default=None,
        help=(
            "JSON 文件: {<geneoh_seq_id>: <ref2dex_seq_id>, ...}，"
            "用于 GeneOH 预处理文件名 ↔ GRAB raw .npz 之间的映射"
        ),
    )
    parser.add_argument(
        "--pert-type", choices=["beta", "gaussian", "none"], default="beta",
        help="GeneOH 注入噪声类型；beta/gaussian 会加扰动再 denoise",
    )
    parser.add_argument("--cuda-ids", default="0")
    parser.add_argument(
        "--denoise-cache-dir", default=None,
        help="GeneOH 去噪中间结果保存目录（默认 <output_root>/_geneoh_cache）",
    )
    parser.add_argument(
        "--skip-geneoh", action="store_true",
        help="只跑 Ref2Dex GRABRawAdapter（init_only 模式），不做 GeneOH denoise",
    )
    parser.add_argument(
        "--list-mapping", action="store_true",
        help="只打印启发式 seq_id 映射，不实际处理",
    )

    args = parser.parse_args()

    # ---- 参数校验 ----
    if args.num_obj_points != 4096:
        raise SystemExit("The Stage 2 schema requires --num-obj-points=4096")
    geneoh_root = Path(args.geneoh_root).resolve()
    if not geneoh_root.exists():
        raise FileNotFoundError(f"--geneoh-root not found: {geneoh_root}")
    geneoh_processed_dir = (
        Path(args.geneoh_processed_dir).resolve()
        if args.geneoh_processed_dir
        else geneoh_root / DEFAULT_GENEOH_SEQ_ROOT
    )
    geneoh_grab_path = (
        Path(args.geneoh_grab_path).resolve()
        if args.geneoh_grab_path
        else Path(args.grab_root).resolve()
    )
    denoise_cache_dir = (
        Path(args.denoise_cache_dir).resolve()
        if args.denoise_cache_dir
        else Path(args.output_root).resolve() / "_geneoh_cache"
    )

    # 读取显式 seq map
    seq_map = load_seq_map(args.geneoh_seq_map)
    rev_map = {v: k for k, v in seq_map.items()}  # ref2dex → geneoh

    # ---- 解析序列 ----
    sequences = _resolve_sequences(args, load_manifest_seq_paths)
    if not sequences:
        raise FileNotFoundError("No GRAB sequences matched the requested input")

    # ---- 打印映射模式 ----
    if args.list_mapping:
        print("[mapping] ref2dex_seq_id → geneoh_seq_id")
        for raw in sequences:
            rp = Path(raw).resolve()
            rel = rp.relative_to(Path(args.grab_root) / "grab").with_suffix("").as_posix()
            if rel in rev_map:
                gid = rev_map[rel]
            else:
                gid = guess_geneoh_seq_id(rp)
            print(f"  {rel:50s}  →  {gid}")
        return

    # ---- 准备 GRABRawAdapter（手部 baseline）----
    output_root = Path(args.output_root).resolve()
    preprocessor = GRABRawAdapter(
        num_obj_points=args.num_obj_points,
        device=args.device,
        max_frames=args.max_frames if args.max_frames > 0 else None,
        grab_root=args.grab_root,
        mano_path=args.mano_path,
        ds_rate=args.ds_rate,
        obj_unit=args.obj_unit,
        nn_batch_size=args.nn_batch_size,
    )
    sides = ("left", "right") if args.side == "both" else (args.side,)

    config: Dict[str, Any] = {
        "object_surface_samples": args.num_obj_points,
        "object_surface_seed": 42,
        "ds_rate": args.ds_rate,
        "mano_path": str(Path(args.mano_path).resolve()),
        "flat_hand_mean": True,
        "requires_vtemplate": True,
        "processing_mode": "geneoh_denoised",
        "denoise_engine": "GeneOH-Diffusion",
        "denoise_steps": ["motion_diff", "spatial_diff"],
        "denoise_pert_type": args.pert_type,
        "denoise_ckpt_motion": args.geneoh_model_path,
        "denoise_ckpt_spatial": args.geneoh_spatial_model_path,
        "denoise_split": args.split,
    }
    stats = {
        "source_sequences": len(sequences),
        "written": 0,
        "empty": 0,
        "skipped": 0,
        "failed": 0,
        "denoise_failed": 0,
    }

    # ---- 逐序列处理 ----
    for raw_path in sequences:
        rp = Path(raw_path).resolve()
        try:
            source = preprocessor.process_sequence(str(rp))
        except Exception as exc:
            stats["failed"] += 1
            print(f"[grab-geneoh-opti] preprocessor failed {rp}: {exc}")
            traceback.print_exc()
            continue

        source_rel = rp.relative_to(Path(args.grab_root).resolve()).as_posix()
        # 找对应的 GeneOH seq_id
        ref2dex_seq_id = (
            rp.parent.name + "/" + rp.stem  # "s1/bowl_pass_1"
        )
        geneoh_seq_id = rev_map.get(ref2dex_seq_id, guess_geneoh_seq_id(rp))

        # ---- (A) GeneOH 去噪（可选）----
        if not args.skip_geneoh:
            try:
                t0 = time.time()
                save_dir = run_geneoh_denoise_two_stage(
                    raw_path=str(rp),
                    geneoh_seq_id=geneoh_seq_id,
                    split=args.split,
                    geneoh_root=geneoh_root,
                    geneoh_env=args.geneoh_env,
                    geneoh_processed_dir=geneoh_processed_dir,
                    geneoh_model_path=args.geneoh_model_path,
                    geneoh_spatial_model_path=args.geneoh_spatial_model_path,
                    grab_path=str(geneoh_grab_path),
                    cuda_ids=args.cuda_ids,
                    pert_type=args.pert_type,
                    denoise_cache_dir=denoise_cache_dir,
                )
                print(
                    f"[geneoh] {geneoh_seq_id} done in {time.time()-t0:.1f}s, "
                    f"output: {save_dir}"
                )
                obj_rot_aa, obj_transl = _load_geneoh_obj_se3(
                    save_dir, geneoh_seq_id, source["obj_root_pose"].shape[0]
                )
                apply_denoised_obj(
                    source,
                    obj_rot_aa=obj_rot_aa,
                    obj_transl=obj_transl,
                    obj_points_cano=source["obj_points_cano"],
                    obj_normals_cano=source["obj_normals_cano"],
                )
            except Exception as exc:
                stats["denoise_failed"] += 1
                print(
                    f"[grab-geneoh-opti] GeneOH denoise failed for "
                    f"{ref2dex_seq_id} (geneoh_id={geneoh_seq_id}): {exc}\n"
                    f"  保留 init_only obj 轨迹继续写 .pkl"
                )
                traceback.print_exc()

        # ---- (B) 写公共 Stage 2 .pkl ----
        for side in sides:
            expected = (
                output_root
                / str(source["subject_id"])
                / f"{source['seq_name']}_{side}.pkl"
            )
            if expected.exists() and not args.overwrite:
                stats["skipped"] += 1
                continue
            payload = pack_stage2_hand(
                source,
                side=side,
                source_raw_file=source_rel,
                processing_mode="geneoh_denoised",
                frame_keep_threshold=args.frame_keep_threshold,
                config={**config, "geneoh_seq_id": geneoh_seq_id},
            )
            if payload is None:
                stats["empty"] += 1
                continue
            path = save_stage2_payload(payload, output_root)
            stats["written"] += 1
            print(
                f"[grab-geneoh-opti] wrote {path} "
                f"({len(payload['raw_frame_id'])} frames, "
                f"geneoh_id={geneoh_seq_id})"
            )

    write_stage2_meta(
        output_root,
        dataset_name="grab",
        processing_mode="geneoh_denoised",
        source_root=args.grab_root,
        config={**config, "frame_keep_threshold": args.frame_keep_threshold},
        stats=stats,
    )
    print(f"[grab-geneoh-opti] summary: {stats}")
    if stats["failed"] or stats["denoise_failed"]:
        # denoise 失败不回退为非零 exit，方便用户看到哪几条没去噪
        # 但默认也不 raise（部分失败不影响其它序列）
        print(
            f"[grab-geneoh-opti] WARNING: {stats['failed']} preprocess failures, "
            f"{stats['denoise_failed']} GeneOH denoise failures"
        )


if __name__ == "__main__":
    main()
