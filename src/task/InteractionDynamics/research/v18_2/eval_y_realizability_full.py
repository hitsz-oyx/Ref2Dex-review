"""V18.2 全量、可分片的 MANO Y 可实现性诊断。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from process.GRAB.raw import DEFAULT_GRAB_ROOT, DEFAULT_MANO_MODEL_DIR
from src.task.InteractionDynamics.dataset_grasp_v18 import (
    CachedGraspDataset, _rigid_world_to_reference, _transform, load_events)
from src.task.InteractionDynamics.grasp_interaction_diffusion import (
    GraspInteractionDiffusion, sample_grasp_v)
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.inverse_mano import build_mano, face_centers
from src.task.InteractionDynamics.residual_interaction_regression import persistence_future
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future


MODES = ("r", "rd", "rdu")


def target_loss(prediction: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    difference = (prediction - target).reshape(*prediction.shape[:-1], 8, 7)
    if mode == "r": difference = difference[..., 3:6]
    elif mode == "rd": difference = difference[..., 3:7]
    elif mode != "rdu": raise ValueError(mode)
    return difference.square().mean((-1, -2, -3))


def component_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = (prediction - target).reshape(128, 8, 7)
    metrics = {name + "_rmse_cm": float(difference[..., section].square().mean().sqrt())
               for name, section in (("u", slice(0, 3)), ("r", slice(3, 6)), ("d", slice(6, 7)))}
    target_d = target.reshape(128, 8, 7)[..., 6]
    prediction_d = prediction.reshape(128, 8, 7)[..., 6]
    target_contact, prediction_contact = target_d < 2., prediction_d < 2.
    true_positive = (target_contact & prediction_contact).sum().float()
    precision = true_positive / prediction_contact.sum().clamp_min(1)
    recall = true_positive / target_contact.sum().clamp_min(1)
    metrics.update({"target_contact_count": int(target_contact.sum()),
                    "reconstructed_contact_count": int(prediction_contact.sum()),
                    "contact_precision": float(precision), "contact_recall": float(recall),
                    "contact_f1": float(2 * precision * recall / (precision + recall).clamp_min(1e-8))})
    return metrics


def rd_metrics(target: torch.Tensor) -> tuple[float, float]:
    value = target.reshape(128, 8, 7)
    violation = (value[..., 3:6].norm(dim=-1) - value[..., 6]).clamp_min(0)
    return float((violation > 1e-6).float().mean()), float(violation.mean())


def stage(item: dict[str, torch.Tensor]) -> str:
    if int((item["state"][:, 3] < 2).sum()) < 4: return "formation"
    return "transition" if int(item["frame"]) < int(item["grasp_frame"]) else "maintenance"


def optimize(mano, faces, betas, current, future, transforms, anchors, target,
             mode, init_mode, starts, steps, lr, noise, seed):
    pose0, orient0, transl0 = current
    gt_pose, gt_orient, gt_transl = future
    if init_mode == "gt_future":
        pose_init, orient_init, transl_init = gt_pose[None], gt_orient[None], gt_transl[None]
    else:
        count = starts if init_mode == "multistart" else 1
        pose_init = pose0.expand(count, 8, -1).clone()
        orient_init = orient0.expand(count, 8, -1).clone()
        transl_init = transl0.expand(count, 8, -1).clone()
        if init_mode == "multistart" and count > 1:
            generator = torch.Generator(device=pose0.device).manual_seed(seed)
            pose_init[1:] += torch.randn(pose_init[1:].shape, device=pose0.device,
                                         generator=generator) * noise[0]
            orient_init[1:] += torch.randn(orient_init[1:].shape, device=pose0.device,
                                           generator=generator) * noise[1]
            transl_init[1:] += torch.randn(transl_init[1:].shape, device=pose0.device,
                                           generator=generator) * noise[2]
    pose, orient, transl = map(torch.nn.Parameter, (pose_init, orient_init, transl_init))
    optimizer = torch.optim.Adam([pose, orient, transl], lr=lr)
    count = len(pose)

    def predict() -> torch.Tensor:
        all_pose = torch.cat([pose0.expand(count, -1, -1), pose], 1).flatten(0, 1)
        all_orient = torch.cat([orient0.expand(count, -1, -1), orient], 1).flatten(0, 1)
        all_transl = torch.cat([transl0.expand(count, -1, -1), transl], 1).flatten(0, 1)
        output = mano(global_orient=all_orient, hand_pose=all_pose,
                      betas=betas.expand(count, -1, -1).flatten(0, 1), transl=all_transl)
        surface = face_centers(output.vertices, faces).reshape(count, 9, -1, 3)
        rotation, translation = transforms
        surface_object = _transform(surface, rotation[None], translation[None])
        packed = []
        for start in range(count):
            y = {key: 100 * value for key, value in
                 build_interaction_y(surface_object[start], anchors, .015).items()}
            packed.append(pack_state_future(y, 8)[1])
        return torch.stack(packed)

    with torch.no_grad():
        prediction = predict(); initial = target_loss(prediction, target[None], mode)
    for _ in range(steps):
        optimizer.zero_grad(); loss = target_loss(predict(), target[None], mode).mean()
        loss.backward(); optimizer.step()
    with torch.no_grad():
        prediction = predict(); losses = target_loss(prediction, target[None], mode)
        best = int(losses.argmin())
    return float(initial.min().sqrt()), float(losses[best].sqrt()), prediction[best]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--steps", type=int, default=300); parser.add_argument("--lr", type=float, default=.003)
    parser.add_argument("--multistart", type=int, default=4)
    parser.add_argument("--pose-noise-rad", type=float, default=.15)
    parser.add_argument("--orient-noise-rad", type=float, default=.10)
    parser.add_argument("--trans-noise-m", type=float, default=.02)
    parser.add_argument("--shard-id", type=int, default=0); parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--reverse", action="store_true", help="从分片尾部开始，便于增量补算")
    parser.add_argument("--gt-only", action="store_true",
                        help="validation 建阈值时只运行 GT+current_repeat")
    parser.add_argument("--grab-root", type=Path, default=Path(DEFAULT_GRAB_ROOT) / "data")
    parser.add_argument("--mano-path", type=Path, default=Path(DEFAULT_MANO_MODEL_DIR))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args(); device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu"); cfg = checkpoint["config"]
    model_cfg = cfg["model"]
    model = GraspInteractionDiffusion(model_cfg["horizon"], model_cfg["dim"],
                                      model_cfg["heads"], model_cfg["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval(); model.requires_grad_(False)
    stats = {key: value.to(device) for key, value in checkpoint["stats"].items()}
    dataset = CachedGraspDataset(args.cache, args.split); events = load_events(args.events)
    indices = list(range(args.shard_id, len(dataset), args.num_shards))
    if args.reverse: indices.reverse()
    if args.max_samples is not None: indices = indices[:args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output_file:
        for sample_index in indices:
            item = dataset[sample_index]; event_index = int(item["event_index"]); event = events[event_index]
            state, gt = item["state"].to(device), item["future"].to(device)
            torch.manual_seed(100000 + sample_index)
            with torch.no_grad():
                normalized = (state[None] - stats["state_mean"]) / stats["state_std"]
                residual = sample_grasp_v(model, normalized,
                                          item["anchors_cm"][None].to(device),
                                          item["object_patches"][None].to(device),
                                          sampling_steps=cfg["diffusion"]["sampling_steps"])[0]
                generated = (persistence_future(state[None], 8)
                             + residual * stats["residual_std"] + stats["residual_mean"])[0]
            frame = int(item["frame"]); anchors = item["anchors_cm"].to(device) / 100
            with np.load(event.path.parent / "shared.npz", allow_pickle=False) as shared:
                source = str(shared["source_raw_file"].item())
                raw_ids = np.asarray(shared["raw_frame_id"][frame:frame + 9], np.int64)
                object_world = torch.from_numpy(np.asarray(shared["obj_points_world"], np.float32)).to(device)
            with np.load(event.path, allow_pickle=False) as hand: side = str(hand["side"].item())
            mano, _, params = build_mano(args.grab_root / source, args.grab_root, side, args.mano_path, device)
            def parameter(name, width):
                return torch.from_numpy(np.asarray(params[name][raw_ids], np.float32).reshape(9, width)).to(device)
            orient, pose, transl = parameter("global_orient", 3), parameter("hand_pose", 24), parameter("transl", 3)
            beta = np.asarray(params["betas"], np.float32)
            beta = beta[raw_ids] if beta.ndim > 1 else np.broadcast_to(beta, (9, beta.shape[-1])).copy()
            betas = torch.from_numpy(beta).to(device); rotation, translation = _rigid_world_to_reference(object_world)
            common = (mano, torch.as_tensor(mano.faces, dtype=torch.long, device=device), betas,
                      (pose[:1], orient[:1], transl[:1]), (pose[1:], orient[1:], transl[1:]),
                      (rotation[frame:frame + 9], translation[frame:frame + 9]), anchors)
            comparisons = [("gt", "current_repeat", gt)]
            if not args.gt_only:
                comparisons += [("generated", init, generated)
                                for init in ("current_repeat", "gt_future", "multistart")]
            for target_name, init_mode, target in comparisons:
                violation_rate, violation_mean = rd_metrics(target)
                for mode in MODES:
                    initial, final, prediction = optimize(
                        *common, target, mode, init_mode, args.multistart, args.steps, args.lr,
                        (args.pose_noise_rad, args.orient_noise_rad, args.trans_noise_m),
                        1000000 + sample_index * 10 + MODES.index(mode))
                    row = {"sample_index": sample_index, "event_index": event_index, "frame": frame,
                           "stage": stage(item), "target": target_name, "init": init_mode,
                           "loss_mode": mode, "initial_rmse_cm": initial, "final_rmse_cm": final,
                           "rd_violation_rate": violation_rate,
                           "rd_violation_mean_cm": violation_mean, **component_metrics(prediction, target)}
                    output_file.write(json.dumps(row, ensure_ascii=False) + "\n"); output_file.flush()
            print(f"[{args.shard_id}] {sample_index}/{len(dataset)}", flush=True)


if __name__ == "__main__": main()
