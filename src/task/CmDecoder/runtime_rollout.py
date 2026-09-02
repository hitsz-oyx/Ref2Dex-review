"""Runtime object-pose rollout for the current CmPointFlowModel checkpoint.

This evaluator deliberately uses the same object_pose_t pair construction as
the current decoder training cache.  It feeds ground-truth Inspire hand-flow
to the frozen Cm encoder, while feeding the previously predicted hand state
to the point-flow decoder and q/wrist fitting loop.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.base.base_config import task_config_from_dict
from src.base.checkpoint import load_checkpoint
from src.base.utils import import_from_path
from src.task.CmDecoder.dataset import (
    _load_hrdex_io,
    _robot_hand_mesh,
    _rotate_to_frame,
    _to_frame,
)
from src.task.CmDecoder.inspire_rollout import (
    _contact_ratio,
    _rotation_error_deg,
    _save_figure,
)
from src.task.CmDecoder.q_optimizer import DifferentiableInspireHand, optimize_q_from_hand_points, rotvec_to_matrix


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT / "outputs/cmdecoder/cm_decoder_20260901_151052/checkpoints/best.pt"
DEFAULT_CACHE_ROOT = ROOT / "data/processed_data/cm_decoder/hrdexdb_inspire_f1_object_pose_t_20260901"
DEFAULT_MANIFEST = DEFAULT_CACHE_ROOT / "v4/selection_all_object_disjoint_seed42.json"


def _local_to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _episode_geometry(cache_root: Path, manifest: dict, episode: str) -> Path:
    if episode not in manifest["cache_dirs"]:
        raise KeyError(f"Episode {episode!r} is absent from {manifest}")
    geometry = cache_root / manifest["cache_dirs"][episode] / "geometry"
    if not geometry.is_dir():
        raise FileNotFoundError(geometry)
    return geometry


def _select_start(q_full: np.ndarray, *, stride: int, frames: int) -> int:
    q = np.asarray(q_full[:, 6:12], dtype=np.float32)
    delta = np.max(np.abs(q[stride:] - q[:-stride]), axis=1)
    need = max(1, int(frames)) * int(stride)
    candidates = np.flatnonzero(np.arange(len(q) - stride) + need < len(q))
    if len(candidates) == 0:
        raise ValueError("Episode is too short for requested rollout")
    active = candidates[delta[candidates] >= np.deg2rad(0.5)]
    return int((active if len(active) else candidates)[np.argmax(delta[(active if len(active) else candidates)])])


class RuntimeRollout:
    def __init__(self, checkpoint: Path, cache_root: Path, manifest_path: Path, episode: str, device: str, stride: int):
        payload = load_checkpoint(checkpoint, map_location="cpu")
        self.cfg = task_config_from_dict(payload["config"])
        if str(getattr(self.cfg.meta, "coordinate_frame", "")) != "object_pose_t":
            raise ValueError("Current runtime evaluator requires checkpoint coordinate_frame=object_pose_t")
        model_cfg = self.cfg.model
        model_cfg.meta = self.cfg.meta
        model_cls = import_from_path(model_cfg.class_path)
        self.model = model_cls(model_cfg)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model = self.model.to(device).eval()
        self.device = torch.device(device)
        self.checkpoint = checkpoint.resolve()
        self.cache_root = cache_root.resolve()
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.episode = episode
        self.geometry = _episode_geometry(self.cache_root, self.manifest, episode)
        self.a = {
            name: np.load(self.geometry / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in (
                "hand_points_world", "hand_normals_world", "obj_points_world", "obj_normals_world",
                "obj_pose_world", "wrist_pose_world", "q_full", "frame_time", "source_frame_id",
            )
        }
        self.stride = int(stride)
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        dataset_root = Path(self.cfg.meta.dataset_root).expanduser().resolve()
        self.io = _load_hrdex_io(dataset_root.parent)
        self.urdf = self.io.parse_urdf(Path(self.cfg.meta.robot_urdf).expanduser().resolve())
        self.hand_model = DifferentiableInspireHand(
            self.cfg.meta.robot_urdf,
            num_hand_points=int(self.cfg.meta.num_hand_points),
            sample_seed=int(self.cfg.meta.sample_seed),
            device=self.device,
        )
        self.mesh_cache: dict = {}

    def _batch(self, source: int, target: int, predicted_world: np.ndarray, predicted_normals_world: np.ndarray) -> dict[str, torch.Tensor]:
        pose = np.asarray(self.a["obj_pose_world"][source], dtype=np.float32)
        hand = _to_frame(predicted_world, pose)
        hand_normals = _rotate_to_frame(predicted_normals_world, pose)
        gt_hand = np.asarray(self.a["hand_points_world"][source], dtype=np.float32)
        gt_hand_target = np.asarray(self.a["hand_points_world"][target], dtype=np.float32)
        obj = _to_frame(np.asarray(self.a["obj_points_world"][source], dtype=np.float32), pose)
        obj_normals = _rotate_to_frame(np.asarray(self.a["obj_normals_world"][source], dtype=np.float32), pose)
        return {
            "hand_points": torch.from_numpy(hand)[None].to(self.device),
            "hand_normals": torch.from_numpy(hand_normals)[None].to(self.device),
            "hand_flow": torch.from_numpy(_to_frame(gt_hand_target, pose) - _to_frame(gt_hand, pose))[None].to(self.device),
            "obj_points": torch.from_numpy(obj)[None].to(self.device),
            "obj_normals": torch.from_numpy(obj_normals)[None].to(self.device),
            "obj_valid_mask": torch.ones((1, len(obj)), dtype=torch.bool, device=self.device),
            "q_t": torch.from_numpy(np.array(self.a["q_full"][source, 6:12], dtype=np.float32, copy=True))[None].to(self.device),
            "q_next": torch.from_numpy(np.array(self.a["q_full"][target, 6:12], dtype=np.float32, copy=True))[None].to(self.device),
            "delta_time_s": torch.tensor([float(self.a["frame_time"][target] - self.a["frame_time"][source])], device=self.device),
        }

    def _mesh_world(self, source: int, q: np.ndarray, wrist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_full = np.asarray(self.a["q_full"][source], dtype=np.float32).copy()
        q_full[6:12] = q
        vertices, faces = _robot_hand_mesh(self.io, self.urdf, q_full, self.mesh_cache)
        source_wrist = np.asarray(self.a["wrist_pose_world"][source], dtype=np.float32)
        source_local = (vertices - source_wrist[:3, 3]) @ source_wrist[:3, :3]
        return _local_to_world(source_local, wrist).astype(np.float32), faces

    def run(self, start: int, frames: int, contact_threshold_m: float) -> dict[str, np.ndarray]:
        start = int(start)
        rows = [(start + i * self.stride, start + (i + 1) * self.stride) for i in range(int(frames))]
        if rows[-1][1] >= len(self.a["source_frame_id"]):
            rows = [row for row in rows if row[1] < len(self.a["source_frame_id"])]
        if not rows:
            raise ValueError("No valid rollout pairs")
        source = rows[0][0]
        predicted_q = np.asarray(self.a["q_full"][source, 6:12], dtype=np.float32).copy()
        predicted_wrist = np.asarray(self.a["wrist_pose_world"][source], dtype=np.float32).copy()
        predicted_points = _to_frame(np.asarray(self.a["hand_points_world"][source]), predicted_wrist)
        predicted_normals = _rotate_to_frame(np.asarray(self.a["hand_normals_world"][source]), predicted_wrist)
        result = {key: [] for key in ("pred_hand_world", "gt_hand_world", "object_world", "pred_mesh_world", "gt_mesh_world", "pred_q", "gt_q", "pred_wrist_world", "gt_wrist_world", "point_epe_mm", "q_mae_deg", "wrist_epe_mm", "wrist_rotation_error_deg", "pred_contact_ratio", "gt_contact_ratio", "source_frame_id", "target_frame_id", "cm_flow_norm_mm")}
        faces = None
        for source, target in rows:
            predicted_world = _local_to_world(predicted_points, predicted_wrist)
            predicted_normals_world = predicted_normals @ predicted_wrist[:3, :3].T
            batch = self._batch(source, target, predicted_world, predicted_normals_world)
            with torch.no_grad():
                _, _, cm_tokens = self.model._frozen_features(batch)
                decode = dict(batch)
                decode["cm_tokens"] = cm_tokens
                prediction = self.model(decode)
                target_obj = decode["hand_points"] + prediction["pred_hand_flow"]
            pose = np.asarray(self.a["obj_pose_world"][source], dtype=np.float32)
            target_world = _local_to_world(target_obj[0].cpu().numpy(), pose)
            target_local = _to_frame(target_world, predicted_wrist)
            fit = optimize_q_from_hand_points(
                self.hand_model,
                q_t=torch.from_numpy(predicted_q)[None].to(self.device),
                current_hand_points=torch.from_numpy(predicted_points)[None].to(self.device),
                target_hand_points=torch.from_numpy(target_local)[None].to(self.device),
                steps=int(getattr(self.cfg.meta, "q_fit_steps", 100)),
                lr=float(getattr(self.cfg.meta, "q_fit_lr", 0.05)),
                prior_weight=float(getattr(self.cfg.meta, "q_fit_prior_weight", 1e-4)),
                wrist_prior_weight=float(getattr(self.cfg.meta, "wrist_fit_prior_weight", 1e-6)),
            )
            predicted_q = fit["q"][0].detach().cpu().numpy().astype(np.float32)
            relative = np.eye(4, dtype=np.float32)
            relative[:3, :3] = rotvec_to_matrix(fit["wrist_delta_rotvec"])[0].detach().cpu().numpy()
            relative[:3, 3] = fit["wrist_delta_translation"][0].detach().cpu().numpy()
            predicted_wrist = predicted_wrist @ relative
            with torch.no_grad():
                predicted_points = self.hand_model.points(fit["q"])[0].cpu().numpy().astype(np.float32)
                predicted_normals = self.hand_model.normals(fit["q"])[0].cpu().numpy().astype(np.float32)
            gt_hand = np.asarray(self.a["hand_points_world"][target], dtype=np.float32)
            pred_hand = _local_to_world(predicted_points, predicted_wrist)
            gt_wrist = np.asarray(self.a["wrist_pose_world"][target], dtype=np.float32)
            obj = np.asarray(self.a["obj_points_world"][target], dtype=np.float32)
            pred_mesh, faces = self._mesh_world(source, predicted_q, predicted_wrist)
            gt_mesh, _ = self._mesh_world(source, np.asarray(self.a["q_full"][target, 6:12], dtype=np.float32), gt_wrist)
            result["pred_hand_world"].append(pred_hand); result["gt_hand_world"].append(gt_hand); result["object_world"].append(obj)
            result["pred_mesh_world"].append(pred_mesh); result["gt_mesh_world"].append(gt_mesh)
            result["pred_q"].append(predicted_q.copy()); result["gt_q"].append(np.asarray(self.a["q_full"][target, 6:12], dtype=np.float32))
            result["pred_wrist_world"].append(predicted_wrist.copy()); result["gt_wrist_world"].append(gt_wrist.copy())
            result["point_epe_mm"].append(float(np.linalg.norm(pred_hand - gt_hand, axis=-1).mean() * 1000.0))
            result["q_mae_deg"].append(float(np.abs(result["pred_q"][-1] - result["gt_q"][-1]).mean() * 180.0 / np.pi))
            result["wrist_epe_mm"].append(float(np.linalg.norm(predicted_wrist[:3, 3] - gt_wrist[:3, 3]) * 1000.0))
            result["wrist_rotation_error_deg"].append(_rotation_error_deg(predicted_wrist, gt_wrist))
            result["pred_contact_ratio"].append(_contact_ratio(pred_hand, obj, contact_threshold_m)); result["gt_contact_ratio"].append(_contact_ratio(gt_hand, obj, contact_threshold_m))
            result["source_frame_id"].append(int(self.a["source_frame_id"][source])); result["target_frame_id"].append(int(self.a["source_frame_id"][target]))
            result["cm_flow_norm_mm"].append(float(torch.linalg.vector_norm(batch["hand_flow"], dim=-1).mean().cpu() * 1000.0))
        result["faces"] = faces
        return {key: np.stack(value) if isinstance(value, list) and value and isinstance(value[0], np.ndarray) else np.asarray(value) for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoder-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episode", default="inspire_f1/bamboo_basket/5")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--start-pair", type=int, default=None)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--device", default="cuda:7" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("output/research/inspire_runtime_rollout_cmdecoder.npz"))
    parser.add_argument("--figure", type=Path, default=Path("output/research/inspire_runtime_rollout_cmdecoder.png"))
    args = parser.parse_args()
    runner = RuntimeRollout(args.decoder_checkpoint.resolve(), args.cache_root.resolve(), args.manifest.resolve(), args.episode, args.device, args.stride)
    start = _select_start(runner.a["q_full"], stride=args.stride, frames=args.frames) if args.start_pair is None else int(args.start_pair)
    result = runner.run(start, args.frames, 0.02)
    result["episode"] = np.asarray(args.episode); result["start_pair"] = np.int32(start); result["stride"] = np.int32(args.stride); result["checkpoint"] = np.asarray(str(args.decoder_checkpoint.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True); args.figure.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    _save_figure(result, args.figure, f"Inspire F1 runtime rollout: {args.episode}, stride={args.stride}, start={start}")
    print(f"episode={args.episode} start_pair={start} stride={args.stride} frames={len(result['point_epe_mm'])}")
    print(f"point_epe_mean/final_mm={result['point_epe_mm'].mean():.4f}/{result['point_epe_mm'][-1]:.4f}")
    print(f"wrist_epe_mean/final_mm={result['wrist_epe_mm'].mean():.4f}/{result['wrist_epe_mm'][-1]:.4f}")
    print(f"q_mae_mean/final_deg={result['q_mae_deg'].mean():.4f}/{result['q_mae_deg'][-1]:.4f}")
    print(f"saved trajectory: {args.output}"); print(f"saved figure: {args.figure}")


if __name__ == "__main__":
    main()
