"""viewer 包 + V1.0 数据流无头冒烟测试（不启动 viser server）。

验证:
1. geometry cache 顶点重建与 stage4 face-center 的 parity（伪逆正确性）;
2. object mesh 经 raw GRAB params 重建后与 cache obj_points_world 的 parity;
3. GRABRandomTransitionDataset：deterministic 播种可复现、手工重放采样一致、
   gap ∈ 该 start 的 valid gaps、Δt 通道正确；
4. provider.sample_hands / model_inputs：同一 (i, gap) 内部一致（flow 是同一
   组 face+bary 表面点在 t 与 t+g 的位移）、双手 concat 顺序 [left, right];
5. rigid 12D twist 对 (twist_pair_to_flow -> fit_rigid_twist_pair) 往返复原;
6. forward：zero-preserving（zero flow 且 dt=0 时 edge message 严格为 0）、
   输出 shape/finite；inverse optimize rigid 12D 在单帧上 loss 下降。
   提供 --checkpoint 时额外加载并报告 GT/zero forward EPE。

用法:
    PYTHONPATH=. python -m src.task.InteractionTransfer.viewer.smoke_test \
        --stage4-root data/processed_data/stage4/data/grab \
        --geometry-root data/processed_data/stage4/interactiontransfer_geometry_cache \
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \
        [--checkpoint outputs/InteractionTransfer/v10_full_20ep/best.pt]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.task.InteractionTransfer.dataset import (
    GRABRandomTransitionDataset, fixed_point_indices,
)
from src.task.InteractionTransfer.geometry_cache import (
    gather_surface_points, sample_surface_refs,
)
from src.task.InteractionTransfer.inverse_optimize import (
    fit_rigid_twist_pair, optimize_hand_flow, twist_pair_to_flow,
)
from src.task.InteractionTransfer.metrics import epe
from src.task.InteractionTransfer.model import InteractionTransfer
from src.task.InteractionTransfer.viewer.mesh_provider import flow_to_mesh_twist, twist_transform
from src.task.InteractionTransfer.viewer.provider import TrajectoryProvider, load_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4-root", required=True)
    parser.add_argument("--geometry-root", required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help="split txt；实际只用其中已在 geometry cache 里的前几条")
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="可选 V1.0 checkpoint；缺省用随机初始化 trainable 模块")
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--grab-root", default=None)
    parser.add_argument("--mano-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def built_sequences(geometry_root: Path, candidates: list[str], count: int) -> list[str]:
    meta = json.loads((geometry_root / "meta.json").read_text(encoding="utf-8"))
    built = [s for s in candidates if s in meta["sequences"]]
    if len(built) < count:
        raise FileNotFoundError(
            f"geometry cache 只有 {len(built)} 条候选序列已构建，至少需要 {count}；"
            f"等待 geometry_cache.py 构建完成")
    return built[:count]


def main():
    args = parse_args()
    device = torch.device(args.device)
    split_seqs = [line.strip() for line in
                  args.split.read_text(encoding="utf-8").splitlines() if line.strip()]
    seqs = built_sequences(Path(args.geometry_root), split_seqs, args.num_sequences)
    print(f"smoke sequences: {seqs}")

    provider = TrajectoryProvider(args.stage4_root, args.split, args.geometry_root,
                                  args.grab_root, args.mano_path, str(device))
    name = seqs[0]
    print(f"[1] load sequence: {name}")
    b = provider.load(name)
    print(f"    frames={b.frames} object={b.object_name} subject={b.subject} "
          f"raw_total={b.raw_frames_total}")
    print(f"    geometry/cache face-center parity: {b.hand_parity_mm:.4f} mm "
          f"({'OK' if b.hand_parity_mm < 1.0 else 'FAIL'})")
    assert b.hand_parity_mm < 1.0, "顶点重建与 stage4 face-center 不一致"

    print("[2] object raw params 与 cache 点云运动一致性")
    i = b.first_valid_frame()
    center = b.obj_points_world[i, b.object_indices].mean(0)
    R, t = provider.mesh.object_poses(b.raw_rel, b.raw_frame_id[[i, i + 1]])
    rel = R[1] @ R[0].T
    p_i = torch.from_numpy(b.obj_points_world[i]).to(device)
    p_j = torch.from_numpy(b.obj_points_world[i + 1]).to(device)
    pred_j = (rel @ p_i.T).T + t[1] - rel @ t[0]
    err = float((pred_j - p_j).norm(dim=-1).max()) * 1000
    print(f"    frame i->j 点云最大误差 = {err:.4f} mm")
    assert err < 0.5, "raw object params 与 cache 点云不一致，mesh 摆放将错位"

    print("[3] GRABRandomTransitionDataset（deterministic + 手工重放）")
    ds = GRABRandomTransitionDataset(args.stage4_root, args.geometry_root, seqs,
                                     deterministic=True)
    idx = 0
    sample_a, sample_b = ds[idx], ds[idx]
    for key in sample_a:
        assert torch.equal(sample_a[key], sample_b[key]), f"{key} 不满足 deterministic 播种"
    seq, i_ds, gaps_i = ds.items[idx]
    rng = np.random.default_rng([ds.seed, idx])
    gap = int(gaps_i[int(rng.integers(len(gaps_i)))])
    assert int(sample_a["gap"].item()) == gap and gap in gaps_i
    obj_points, obj_normals = ds._object(seq)
    oi = fixed_point_indices(obj_points.shape[1], ds.object_points)
    o = obj_points[i_ds, oi].astype("f4")
    c = o.mean(0, keepdims=True)
    pts, nrm, flow = [], [], []
    for side in ("left", "right"):
        vertices, _ = ds._geometry(seq, side)
        face_normals = ds._hand_static(seq, side)
        face_idx, bary = sample_surface_refs(ds.faces[side].shape[0], ds.hand_points_per_side, rng)
        p_i2 = gather_surface_points(vertices[i_ds], ds.faces[side], face_idx, bary)
        p_j2 = gather_surface_points(vertices[i_ds + gap], ds.faces[side], face_idx, bary)
        pts.append(p_i2 - c)
        nrm.append(face_normals[i_ds][face_idx].astype("f4"))
        flow.append(p_j2 - p_i2)
    manual = {"object_points": o - c,
              "hand_points": np.concatenate(pts),
              "hand_normals": np.concatenate(nrm),
              "hand_flow": np.concatenate(flow).astype("f4")}
    for key, value in manual.items():
        diff = float((sample_a[key].numpy() - value).__abs__().max())
        print(f"    {key}: max|diff|={diff:.2e}")
        assert diff < 1e-6, f"{key} 与手工重放不一致"
    print(f"    item 0: seq={seq} i={i_ds} gaps={gaps_i} -> gap={gap} "
          f"({len(ds)} starts total)")

    print("[4] provider sample_hands / model_inputs 内部一致性")
    gap_v = b.valid_gaps(i)
    assert gap_v, "无 valid gap"
    gap = gap_v[0]
    pts_v, nrm_v, flow_v, _, _ = b.sample_hands(i, gap)
    pts_v2, _, flow_v2, _, _ = b.sample_hands(i, gap)
    assert np.array_equal(pts_v, pts_v2) and np.array_equal(flow_v, flow_v2), "采样不可复现"
    rng = np.random.default_rng([7, i, gap, 0])
    manual_flow = []
    for side in ("left", "right"):
        face_idx, bary = sample_surface_refs(b.hand_faces[side].shape[0],
                                             b.hand_points_per_side, rng)
        p_i2 = gather_surface_points(b.hand_verts[side][i], b.hand_faces[side], face_idx, bary)
        p_j2 = gather_surface_points(b.hand_verts[side][i + gap], b.hand_faces[side], face_idx, bary)
        manual_flow.append(p_j2 - p_i2)
    diff = float(np.abs(np.concatenate(manual_flow) - flow_v).max())
    print(f"    hand flow 与手工重放 max|diff|={diff:.2e} m")
    assert diff < 1e-6
    inputs = b.model_inputs(i, device, gap)
    assert int(inputs["gap"].item()) == gap
    assert float((inputs["hand_points"][:, :769] - inputs["hand_points"][:, 769:]).abs().max()) >= 0
    assert inputs["hand_flow"].shape == (1, 2 * b.hand_points_per_side, 3)
    print(f"    model_inputs ok（gap={gap}, hand {2 * b.hand_points_per_side} 点 concat [L,R]）")

    print("[5] rigid 12D twist 往返复原")
    n_side = b.hand_points_per_side
    hp = inputs["hand_points"]
    left, right = hp[:, :n_side], hp[:, n_side:]
    twist = torch.randn(1, 12, device=device) * 0.05
    flow_pair = twist_pair_to_flow(twist, left, right)
    recovered = fit_rigid_twist_pair(flow_pair, left, right)
    err = float((recovered - twist).abs().max())
    print(f"    max|twist - recovered| = {err:.2e}")
    assert err < 1e-3, "rigid 12D 拟合不复原"

    print("[6] forward / inverse（random init trainable）")
    model = InteractionTransfer(dense_checkpoint=args.dense_checkpoint).to(device).eval()
    epoch = -1
    if args.checkpoint is not None:
        model, epoch = load_model(args.checkpoint, args.dense_checkpoint, device)
        print(f"    loaded checkpoint {args.checkpoint} (epoch {epoch})")
    with torch.no_grad():
        static = model.encode_static(inputs["object_points"], inputs["object_normals"],
                                     left, inputs["hand_normals"][:, :n_side],
                                     right, inputs["hand_normals"][:, n_side:])
        zero_out = model.forward_core(hand_flow=torch.zeros_like(inputs["hand_flow"]),
                                      dt=torch.zeros(1, device=device), **static)
    assert zero_out["edge_message"].abs().max().item() == 0.0, \
        "zero flow + dt=0 时 message 应严格为 0（bias-free action path）"
    with torch.no_grad():
        out = model.forward_core(hand_flow=inputs["hand_flow"], dt=inputs["gap"], **static)
    assert out["object_flow"].shape == inputs["object_flow"].shape
    assert torch.isfinite(out["object_flow"]).all()
    fwd_epe = float(epe(out["object_flow"], inputs["object_flow"])) * 1000
    print(f"    forward GT-action EPE = {fwd_epe:.2f} mm "
          f"({'ckpt' if epoch >= 0 else 'random init'})")

    result = optimize_hand_flow(model, static, inputs["hand_points"], inputs["object_flow"],
                                gt_flow=inputs["hand_flow"],
                                init_flow=torch.zeros_like(inputs["hand_flow"]),
                                dt=inputs["gap"], parameterization="rigid",
                                target_kind="effect", steps=300, lr=0.01)
    h0, h1 = result["history"][0], result["history"][-1]
    print(f"    rigid 12D inverse: loss {h0['loss']:.2e} -> {h1['loss']:.2e}, "
          f"EPE {h0['epe_mm']:.2f} -> {h1['epe_mm']:.2f} mm "
          f"(GT action {result['gt']['epe_mm']:.2f} mm)")
    assert h1["loss"] < h0["loss"], "inverse 优化未降低 loss"

    print("[7] GT object flow -> twist -> mesh 对齐下一帧")
    gt_flow = inputs["object_flow"]
    twist_obj, _ = flow_to_mesh_twist(gt_flow, inputs["object_points"])
    mesh_t = torch.from_numpy(b.obj_world[i] - center).to(device).unsqueeze(0)
    pred_next = twist_transform(twist_obj, mesh_t,
                                inputs["object_points"].mean(1, keepdim=True)).squeeze(0)
    gt_next = b.obj_world[i + gap] - center
    align = float(np.linalg.norm(pred_next.cpu().numpy() - gt_next, axis=1).mean()) * 1000
    print(f"    mean vertex error (gap={gap}) = {align:.3f} mm")
    assert align < 1.0, "GT flow 的刚体提升与下一帧 mesh 不对齐"

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
