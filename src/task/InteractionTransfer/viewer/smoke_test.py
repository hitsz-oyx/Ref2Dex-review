"""viewer 包无头冒烟测试（不启动 viser server）。

验证:
1. MANO face centers 与 cache hand_points_world 的 parity（mesh 重建正确性）;
2. object mesh 经 raw GRAB params 重建后与 cache obj_points_world 的 parity;
3. model_inputs 与 GRABOneStepDataset sample 完全一致;
4. GT object flow 拟合的刚体 twist 应用到 mesh 后与下一帧 mesh 对齐;
5. inverse optimize 在单帧上可跑通且 EPE 与 V0.10 结果同量级。

用法:
    PYTHONPATH=. python -m src.task.InteractionTransfer.viewer.smoke_test \
        --root data/processed_data/stage4/data/grab \
        --split src/task/InteractionTransfer/splits/grab_seed42/test.txt \
        --checkpoint outputs/InteractionTransfer/v08_full_20ep/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.task.InteractionTransfer.dataset import GRABOneStepDataset
from src.task.InteractionTransfer.inverse_optimize import optimize_hand_flow
from src.task.InteractionTransfer.metrics import epe
from src.task.InteractionTransfer.viewer.mesh_provider import flow_to_mesh_twist, twist_transform
from src.task.InteractionTransfer.viewer.provider import TrajectoryProvider, load_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(
        "outputs/InteractionTransfer/v08_full_20ep/best.pt"))
    parser.add_argument("--dense-checkpoint", default="src/task/Cm/densetoken_ckpt/best.pt")
    parser.add_argument("--grab-root", default=None)
    parser.add_argument("--mano-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    provider = TrajectoryProvider(args.root, args.split, args.grab_root, args.mano_path,
                                  str(device))
    name = provider.sequences[0]
    print(f"[1] load sequence: {name}")
    b = provider.load(name)
    print(f"    frames={b.frames} object={b.object_name} subject={b.subject} "
          f"raw_total={b.raw_frames_total} valid={len(b.valid_frames())}")
    print(f"    MANO/cache hand parity: {b.hand_parity_mm:.4f} mm "
          f"({'OK' if b.hand_parity_mm < 1.0 else 'FAIL'})")
    assert b.hand_parity_mm < 1.0, "MANO 重建与 cache 不一致"

    print("[2] object raw params 与 cache 点云运动一致性")
    # cache 生成即 obj_points_world = R_f @ canonical_samples + t_f，
    # 因此 frame-j 点应可由 frame-i 点经 (R_j R_i^{-1}, t_j - R_j R_i^{-1} t_i) 得到。
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

    print("[3] model_inputs vs GRABOneStepDataset sample")
    dataset = GRABOneStepDataset(args.root, [name])
    ref = None
    for idx in range(len(dataset)):
        if dataset.items[idx].current == i:
            ref = dataset[idx]
            break
    assert ref is not None
    inputs = b.model_inputs(i, device)
    for key in ("object_points", "object_normals", "hand_points", "hand_normals",
                "hand_flow", "object_flow"):
        diff = float((inputs[key].cpu() - ref[key][None]).abs().max())
        print(f"    {key}: max|diff|={diff:.2e}")
        assert diff < 1e-5, f"{key} 与 dataset 不一致"

    print("[4] GT object flow -> twist -> mesh 对齐下一帧")
    gt_flow = inputs["object_flow"]
    twist, _ = flow_to_mesh_twist(gt_flow, inputs["object_points"])
    mesh_t = torch.from_numpy(b.obj_world[i] - center).to(device).unsqueeze(0)
    pred_next = twist_transform(twist, mesh_t,
                                inputs["object_points"].mean(1, keepdim=True)).squeeze(0)
    gt_next = b.obj_world[i + 1] - center
    align = float(np.linalg.norm(pred_next.cpu().numpy() - gt_next, axis=1).mean()) * 1000
    print(f"    mean vertex error = {align:.3f} mm")
    assert align < 1.0, "GT flow 的刚体提升与下一帧 mesh 不对齐"

    print("[5] inverse optimize（rigid, zero init, 300 steps）")
    model, epoch = load_model(Path(args.checkpoint), args.dense_checkpoint, device)
    with torch.no_grad():
        static = model.encode_static(inputs["object_points"], inputs["object_normals"],
                                     inputs["hand_points"], inputs["hand_normals"])
    result = optimize_hand_flow(model, static, inputs["hand_points"], inputs["object_flow"],
                                gt_flow=inputs["hand_flow"],
                                init_flow=torch.zeros_like(inputs["hand_flow"]),
                                parameterization="rigid", steps=300, lr=0.01)
    print(f"    epoch={epoch} zero={result['initial']['epe_mm']:.2f} mm "
          f"optimized={result['optimized']['epe_mm']:.2f} mm "
          f"gt={result['gt']['epe_mm']:.2f} mm")
    assert result["optimized"]["epe_mm"] < result["gt"]["epe_mm"]

    pred_epe = None
    with torch.no_grad():
        out = model.forward_core(hand_flow=inputs["hand_flow"], **static)
    pred_epe = float(epe(out["object_flow"], inputs["object_flow"])) * 1000
    print(f"    forward GT-action EPE = {pred_epe:.2f} mm")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
