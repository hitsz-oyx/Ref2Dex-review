"""Frozen same-source Cm swap diagnostic, followed by 16-step hand rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.task.CmDecoderv2.dataset import CmDecoderV2Dataset, _collate_cm_decoder
from src.task.CmDecoderv2.model import CmDecoderV2
from src.task.CmDecoderv2.pointflow import make_relative_transform, world_to_object
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel
from .diagnostics import (CONDITIONS, MATCH_LIMITS, continuous_starts, pair_differences,
                          paired_summary, rotation_difference_deg, select_donors, spread_starts)


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "outputs/cmdecoderv2/cm_decoder_v2_dexplore_rl_v1_3_full10135_20260910_123812/checkpoints/best.pt"
CHECKPOINT_SHA = "e60f0e954c062d15e7c2fa217e1bebcb0a4b0be8795d1b5729771ee1f8f5fef7"
VERSION = "V1.1.8"
MODEL_KEYS = ("obj_points", "obj_normals", "obj_valid_mask", "hand_points", "hand_normals", "hand_flow",
              "hand_valid_mask", "knn_edge_indices", "knn_edge_valid_mask", "current_finger_q",
              "current_wrist_translation_object", "current_wrist_rotation_6d_object", "current_link_features",
              "current_wrist_pose_world", "object_pose_world")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def namespace(value):
    return SimpleNamespace(**{k: namespace(v) if isinstance(v, dict) else v for k, v in value.items()})


def state_digest(model):
    digest = hashlib.sha256()
    # Include nonpersistent FK buffers, which are not stored in the checkpoint.
    for name, value in sorted(list(model.named_parameters()) + list(model.named_buffers())):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def error_values(points, target, q, target_q, wrist, target_wrist):
    error = torch.linalg.vector_norm(points - target, dim=-1).mean(dim=-1) * 1000
    qerror = (q - target_q).abs().mean(dim=-1)
    terror = torch.linalg.vector_norm(wrist[:, :3, 3] - target_wrist[:, :3, 3], dim=-1) * 1000
    rotations = rotation_difference_deg(wrist[:, :3, :3].double().cpu().numpy(), target_wrist[:, :3, :3].double().cpu().numpy())
    return dict(hand_epe_mm=error.cpu().numpy(), q_mae_rad=qerror.cpu().numpy(),
                wrist_translation_mm=terror.cpu().numpy(), wrist_rotation_deg=rotations)


def native_surface_points(helper, surface, native_q):
    """Replay the cache's full native joint state, including measured mimic joints.

    This is a GT-only correspondence audit, never a decoder input or prediction.
    """
    links = helper.link_transforms(helper.qpos_to_urdf_order(np.asarray(native_q, dtype=np.float64)))
    visual_ids = surface.surface_visual_ids.cpu().numpy()
    local = surface.surface_points_local.cpu().numpy()
    result = np.empty_like(local)
    for visual_id in np.unique(visual_ids):
        mask = visual_ids == visual_id
        visual = helper.visuals[int(visual_id)]
        transform = (links[visual.link] @ visual.local_transform).astype(np.float32)
        result[mask] = local[mask] @ transform[:3, :3].T + transform[:3, 3]
    return result


def run(args):
    started = time.monotonic()
    output = HERE / "output" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "run.log"

    def log(message):
        line = datetime.now().astimezone().isoformat(timespec="seconds") + " " + message
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def budget():
        if time.monotonic() - started > args.budget_seconds:
            raise TimeoutError("30-minute diagnostic runtime budget exceeded")
        if args.device.startswith("cuda") and torch.cuda.max_memory_allocated() > 8 * 1024**3:
            raise MemoryError("8 GiB GPU allocation budget exceeded")

    manifest = dict(schema_name="ref2dex.cm_condition_dependence.v1", task="CmDecoderv2",
                    work_version=VERSION, operation_category=["diagnostic", "experiment", "operation"],
                    run_id=args.run_id, activity_id=args.activity_id, run_status="STARTED",
                    base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                    created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                    config_snapshot="config.json", metadata_snapshot="metadata.json",
                    checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=CHECKPOINT_SHA,
                    seed=42, output_directory=str(output.relative_to(ROOT)), command_args=vars(args))
    write_json(output / "run_manifest.json", manifest)
    try:
        assert sha256(CHECKPOINT) == CHECKPOINT_SHA
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        cfg = namespace(config)
        assert cfg.meta.window_size == 4 and cfg.meta.num_hand_points == 10135
        assert cfg.meta.hand_stream_mode == "unique_knn_edges" and cfg.meta.knn_k == 32
        assert cfg.meta.interaction_radius_m == 0.02
        cfg.model.meta = cfg.meta
        model = CmDecoderV2(cfg.model).to(args.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        # The decoder checkpoint includes its frozen OICM: check both copies agree.
        oicm_state = torch.load(model.oicm_checkpoint, map_location="cpu", weights_only=False)["model"]
        for key, value in oicm_state.items():
            assert torch.equal(model.oicm.state_dict()[key].cpu(), value.cpu()), key
        model.requires_grad_(False).eval()
        native_helper = InspireUrdfModel(ROOT / cfg.data.urdf_path)
        initial_state = state_digest(model)
        view_root = ROOT / cfg.data.view_root
        index = json.loads((view_root / "index.json").read_text())
        entries = index["sequences"]["val"]
        assert len(entries) == 30 and all(e["variant"] == "inspire_rl" for e in entries)
        dataset = CmDecoderV2Dataset(entries, urdf_path=ROOT / cfg.data.urdf_path,
                                    window_size=4, num_obj_points=1024, num_hand_points=10135,
                                    hand_stream_mode="unique_knn_edges", knn_k=32,
                                    hand_supervision_radius_m=0.02, seed=42, perturb=False, active_only=True)
        dataset.set_epoch(0)
        selection = list(range(len(dataset)))
        if args.smoke:
            # Two whole sequences give the smoke real donor and contiguous-rollout coverage.
            selected_sequences = sorted({s for s, _ in dataset.rows})[:2]
            selection = [i for i, (s, _) in enumerate(dataset.rows) if s in selected_sequences]
        paths = [CHECKPOINT, Path(model.oicm_checkpoint), view_root / "index.json", view_root / "manifest.json",
                 ROOT / cfg.data.index_path, ROOT / cfg.data.urdf_path, ROOT / cfg.model.oicm_config,
                 ROOT / "data/processed_data/object_interaction_cm_dexplore_rl_v1_3/scales_train_v1_3_unique_knn.json"]
        paths += list(HERE.glob("*.py"))
        paths += [ROOT / "src/task/CmDecoderv2" / p for p in ("model.py", "dataset.py", "pointflow.py", "kinematics.py")]
        paths += [ROOT / "src/task/ObjectInteractionCm" / p for p in ("model.py", "decoder.py", "slot_attention.py")]
        paths += [ROOT / "src/task/ObjectInteractionCm/tools/data/build_dexplore_rl_cache.py"]
        protected = {str(p.resolve()): sha256(p) for p in paths}
        geometry_stats = {}
        for s in sorted({dataset.rows[i][0] for i in selection}):
            sequence = dataset.sequences[s]
            for p in list(sequence.geometry_root.glob("*.npy")) + [Path(entries[s]["q_native"]), Path(entries[s]["wrist_pose_world"])]:
                st = p.stat()
                geometry_stats[str(p.resolve())] = [st.st_size, st.st_mtime_ns]
        metadata = dict(evaluation_partition="val", source="inspire_rl", total_val_sequences=30,
                        total_active_windows=len(dataset), selected_dataset_rows=selection, sequence_entries=entries,
                        invariant=dict(coordinate_frame="object_pose_t", fps=30, window_size=4, stride=1,
                                       surface_points=10135, object_points=1024, seed=42, epoch=0, perturb=False),
                        protected_sha256=protected, geometry_stats=geometry_stats,
                        checkpoint_epoch=int(checkpoint["epoch"]), checkpoint_step=int(checkpoint["step"]),
                        checkpoint_best_metric=float(checkpoint["best_metric"]),
                        torch_version=torch.__version__, numpy_version=np.__version__,
                        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                        model_state_digest_before=initial_state)
        write_json(output / "metadata.json", metadata)
        write_json(output / "config.json", dict(checkpoint_config=config, diagnostic=dict(conditions=CONDITIONS,
                   match_limits=MATCH_LIMITS, rollout_steps=16, rollout_starts_per_sequence=3, bootstrap_repeats=2000,
                   smoke_only=args.smoke, budget_seconds=args.budget_seconds)))
        log(f"START selected={len(selection)}/{len(dataset)} active windows; checkpoint epoch={checkpoint['epoch']}")
        loader = DataLoader(Subset(dataset, selection), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=_collate_cm_decoder, pin_memory=True)
        stored = {}
        baseline_rows = []
        max_gt_fk = 0.0
        max_native_replay = 0.0

        def append(key, value):
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            value = np.asarray(value)
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite bank field {key}")
            stored.setdefault(key, []).append(value)

        with torch.inference_mode():
            cursor = 0
            for cpu in loader:
                budget()
                batch = {key: cpu[key].to(args.device) for key in MODEL_KEYS}
                pred = model(batch)
                size = len(cpu["start_frame"])
                selected = selection[cursor:cursor + size]
                q = batch["current_finger_q"]
                wrist = batch["current_wrist_pose_world"]
                target_q = q + cpu["target_q_delta"][:, 0].to(args.device)
                relative = torch.eye(4, device=args.device)[None].repeat(size, 1, 1)
                relative[:, :3, :3] = cpu["target_wrist_rotation"][:, 0].to(args.device)
                relative[:, :3, 3] = cpu["target_wrist_translation"][:, 0].to(args.device)
                target_wrist = wrist @ relative
                target = cpu["target_hand_points_object"][:, 0].to(args.device)
                predicted_q = q + pred["pred_q_delta"][:, 0]
                predicted_wrist = wrist @ make_relative_transform(pred["pred_wrist_rotvec"][:, 0], pred["pred_wrist_translation"][:, 0])
                correct = error_values(pred["pred_hand_points_object"][:, 0], target, predicted_q, target_q, predicted_wrist, target_wrist)
                identity = error_values(pred["current_hand_points_object"], target, q, target_q, wrist, target_wrist)
                # Reduced six-DOF GT replay is not necessarily identical to native simulator joints.
                fk_gt = world_to_object(model.point_surface(target_q, target_wrist), batch["object_pose_world"])
                gt_difference = float((fk_gt - target).abs().max())
                max_gt_fk = max(max_gt_fk, gt_difference)
                reduced_error = torch.linalg.vector_norm(fk_gt-target,dim=-1).mean(dim=-1)*1000
                append("gt_reduced_fk_epe_mm", reduced_error)
                native_future = torch.as_tensor(np.stack([np.asarray(dataset.sequences[dataset.rows[i][0]].q_native[dataset.rows[i][1]+1]) for i in selected]),device=args.device)
                reduced_native = model.point_surface._expand_native(target_q)
                mimic_ids = [7,9,11,13,16,17]
                append("gt_mimic_q_rms_rad", (native_future[:,mimic_ids]-reduced_native[:,mimic_ids]).square().mean(-1).sqrt())
                clipped_q = torch.maximum(torch.minimum(target_q, model.point_surface.finger_upper),model.point_surface.finger_lower)
                append("gt_independent_limit_violation_rad", (clipped_q-target_q).abs().max(-1).values)
                current_cache = torch.as_tensor(np.stack([np.asarray(dataset.sequences[dataset.rows[i][0]].knn_hand_points[dataset.rows[i][1]]) for i in selected]),device=args.device)
                current_cache = world_to_object(current_cache,batch['object_pose_world'])
                append("cache_hold_epe_mm",torch.linalg.vector_norm(current_cache-target,dim=-1).mean(-1)*1000)
                # Full native joints must reproduce the same ordered cache surface (no tolerance relaxation).
                seq_number, frame = dataset.rows[selected[0]]
                sequence = dataset.sequences[seq_number]
                native_replay = native_surface_points(native_helper, model.point_surface, sequence.q_native[frame+1])
                native_difference = float(np.abs(native_replay-sequence.knn_hand_points[frame+1]).max())
                max_native_replay = max(max_native_replay,native_difference)
                if native_difference > 1e-4:
                    raise ValueError(f"Native GT FK/cache correspondence differs by {native_difference} m")
                for key, value in (("cm_tokens", pred["cm_tokens"]), ("anchor_pos", pred["cm_anchor_pos"]),
                                   ("anchor_normal", pred["cm_anchor_normal"]), ("cm_valid", pred["cm_sample_valid"]),
                                   ("current_q", q), ("current_wrist", wrist), ("object_pose", batch["object_pose_world"]),
                                   ("target_q", target_q), ("target_wrist", target_wrist),
                                   ("link_features", batch["current_link_features"]),
                                   ("current_state", torch.cat([q, batch["current_wrist_translation_object"], batch["current_wrist_rotation_6d_object"]], dim=-1)),
                                   ("correct_q_delta", pred["pred_q_delta"]), ("correct_translation", pred["pred_wrist_translation"]),
                                   ("correct_rotvec", pred["pred_wrist_rotvec"]),
                                   ("sequence_number", [dataset.rows[i][0] for i in selected]),
                                   ("start_frame", cpu["start_frame"]), ("dataset_row", selected)):
                    append(key, value)
                valid = batch["hand_valid_mask"][:, 0].float()
                flow = batch["hand_flow"][:, 0]
                denominator = valid.sum(dim=1).clamp_min(1)
                append("flow_mean_mm", (flow * valid[..., None]).sum(dim=1) / denominator[:, None] * 1000)
                append("flow_rms_mm", ((flow.square().sum(-1) * valid).sum(1) / denominator).sqrt() * 1000)
                append("active_fraction", batch["knn_edge_valid_mask"][:, 0].any(dim=-1).float().mean(dim=-1))
                for j in range(size):
                    common = dict(sample_id=cursor+j, sequence_id=cpu["sequence_id"][j],
                                  start_frame=int(cpu["start_frame"][j]), current_valid=bool(pred["cm_sample_valid"][j, 0]))
                    for condition, values in (("correct", correct), ("identity", identity)):
                        baseline_rows.append(dict(common, condition=condition,
                                                  **{key: float(value[j]) for key, value in values.items()}))
                cursor += size
                if cursor % (args.batch_size * 25) == 0 or cursor == len(selection):
                    log(f"EXTRACT {cursor}/{len(selection)}")
            bank = {key: np.concatenate(value) for key, value in stored.items()}
            np.savez_compressed(output / "cm_bank.npz", **bank)
            donors = select_donors(bank)
            eligible, lookup = continuous_starts(bank)
            rollout_donors = select_donors(bank, eligible=eligible)
            starts = spread_starts(bank, eligible)
            np.savez_compressed(output / "donors.npz", **donors, rollout_swap=rollout_donors["swap"],
                                rollout_matched_swap=rollout_donors["matched_swap"], rollout_eligible=eligible,
                                rollout_starts=np.asarray(starts))
            log(f"MATCH teacher swap={int((donors['swap']>=0).sum())} matched={int((donors['matched_swap']>=0).sum())}; rollout starts={len(starts)}")
            rows, predictions, replay_error = teacher_controls(model, dataset, bank, baseline_rows, donors, args, budget, log)
            with (output / "metrics.jsonl").open("w") as stream:
                for row in rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            np.savez_compressed(output / "teacher_predictions.npz", **predictions)
            rollout_rows, trajectories = rollouts(model, dataset, bank, starts, rollout_donors, lookup, args, budget, log)
            with (output / "rollout_metrics.jsonl").open("w") as stream:
                for row in rollout_rows:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
            np.savez_compressed(output / "rollout_predictions.npz", **trajectories)

        valid_rows = [r for r in rows if r["current_valid"]]
        moving = {r["sample_id"] for r in valid_rows if r["condition"] == "identity" and r["hand_epe_mm"] >= 2}
        summary = dict(smoke_only=args.smoke, conclusion="INCONCLUSIVE",
                       conclusion_scope="frozen within-Inspire conditioning diagnostic; not cross-hand transfer or physical closed-loop",
                       coverage=dict(active_windows=len(bank["start_frame"]), h1_valid=int(bank["cm_valid"][:, 0].sum()),
                                     all_four_valid=int(bank["cm_valid"].all(axis=1).sum()),
                                     teacher_swap=int((donors["swap"] >= 0).sum()), teacher_matched=int((donors["matched_swap"] >= 0).sum()),
                                     rollout_starts=len(starts), rollout_sequences=len({int(bank['sequence_number'][i]) for i in starts})),
                       teacher=paired_summary(valid_rows),
                       teacher_moving_ge2mm=paired_summary([r for r in valid_rows if r["sample_id"] in moving]),
                       rollout={str(step): paired_summary([r for r in rollout_rows if r["step"] == step]) for step in (1, 4, 8, 16)},
                       gt_reduced_fk_audit=dict(mean_epe_mm=float(bank['gt_reduced_fk_epe_mm'].mean()),
                                               median_epe_mm=float(np.median(bank['gt_reduced_fk_epe_mm'])),
                                               p95_epe_mm=float(np.quantile(bank['gt_reduced_fk_epe_mm'],.95)),
                                               max_epe_mm=float(bank['gt_reduced_fk_epe_mm'].max()),
                                               mimic_q_rms_rad_mean=float(bank['gt_mimic_q_rms_rad'].mean()),
                                               independent_limit_violation_fraction=float((bank['gt_independent_limit_violation_rad']>1e-5).mean()),
                                               cache_hold_epe_mm_mean=float(bank['cache_hold_epe_mm'].mean()),
                                               interpretation="GT six-joint fixed-mimic reconstruction discrepancy; not an optimized error lower bound"),
                       engineering_checks=dict(max_reduced_gt_fk_coordinate_difference_m=max_gt_fk,
                                               max_native_gt_fk_coordinate_difference_m=max_native_replay, max_core_replay_difference=replay_error,
                                               no_train=True, no_target_to_model=True))
        final_state = state_digest(model)
        assert final_state == initial_state
        for path, digest in protected.items():
            assert sha256(path) == digest, path
        for path, expected in geometry_stats.items():
            stat = Path(path).stat()
            assert [stat.st_size, stat.st_mtime_ns] == expected, path
        summary["engineering_checks"].update(model_unchanged=True, inputs_unchanged=True, strict_checkpoints=True)
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2 if args.device.startswith("cuda") else 0
        write_json(output / "dependence_summary.json", summary)
        plot(summary, output)
        size_bytes = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        if size_bytes > 1024**3:
            raise RuntimeError("1 GiB output budget exceeded")
        budget()
        manifest.update(run_status="COMPLETED", conclusion="INCONCLUSIVE", last_step=len(bank["start_frame"]),
                        elapsed_seconds=summary["elapsed_seconds"], output_bytes=size_bytes,
                        outputs=[p.name for p in output.iterdir() if p.is_file()])
        write_json(output / "run_manifest.json", manifest)
        log(f"COMPLETED {summary['elapsed_seconds']:.1f}s; peak_gpu={summary['gpu_peak_allocated_mib']:.1f} MiB; output={size_bytes/1024**2:.1f} MiB")
        return output
    except Exception as error:
        manifest.update(run_status="FAILED", error=f"{type(error).__name__}: {error}", elapsed_seconds=time.monotonic()-started)
        write_json(output / "run_manifest.json", manifest)
        log(f"FAILED {type(error).__name__}: {error}")
        raise


def decode_core(model, bank, recipients, condition, donors, device, *, q=None, wrist=None, kinematics=None):
    """Decoder sees only current target state and source Cm; no target future fields."""
    ids = np.asarray(recipients, dtype=np.int64)
    def tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device=device)
    current_q = tensor(bank["current_q"][ids]) if q is None else q
    current_wrist = tensor(bank["current_wrist"][ids]) if wrist is None else wrist
    if condition == "identity":
        return current_q, current_wrist
    source = np.asarray(donors, dtype=np.int64) if condition in {"swap", "matched_swap"} else ids
    tokens, anchors, normals = [tensor(bank[key][source]) for key in ("cm_tokens", "anchor_pos", "anchor_normal")]
    if condition == "zero_all":
        tokens, anchors, normals = torch.zeros_like(tokens), torch.zeros_like(anchors), torch.zeros_like(normals)
    if q is None:
        state, links = tensor(bank["current_state"][ids]), tensor(bank["link_features"][ids])
    else:
        object_pose = tensor(bank["object_pose"][ids])
        pose = torch.linalg.inv(object_pose) @ current_wrist
        state = torch.cat([current_q, pose[:, :3, 3], pose[:, :3, :2].transpose(-1, -2).reshape(-1, 6)], dim=-1)
        links = tensor(np.stack([kinematics.query_features(finger, hand, obj)
                                 for finger, hand, obj in zip(current_q.cpu().numpy(), current_wrist.cpu().numpy(), bank["object_pose"][ids])]))
    result = model.core(tokens, anchors, normals, state, links)
    return (current_q + result["pred_q_delta"][:, 0],
            current_wrist @ make_relative_transform(result["pred_wrist_rotvec"][:, 0], result["pred_wrist_translation"][:, 0]))


def teacher_controls(model, dataset, bank, baseline_rows, donors, args, budget, log):
    rows = list(baseline_rows)
    predictions = {key: [] for key in ("sample_id", "condition_id", "q", "wrist")}
    max_replay = 0.0
    count = len(bank["start_frame"])
    for start in range(0, count, args.batch_size):
        budget()
        ids = np.arange(start, min(start + args.batch_size, count))
        target = torch.as_tensor(np.stack([np.asarray(dataset.sequences[int(bank["sequence_number"][i])].knn_hand_points[int(bank["start_frame"][i])+1]) for i in ids]), device=args.device)
        target_q = torch.as_tensor(bank["target_q"][ids], device=args.device)
        target_wrist = torch.as_tensor(bank["target_wrist"][ids], device=args.device)
        correct_q, correct_wrist = decode_core(model, bank, ids, "correct", ids, args.device)
        expected_q = torch.as_tensor(bank["current_q"][ids] + bank["correct_q_delta"][ids, 0], device=args.device)
        expected_wrist = torch.as_tensor(bank["current_wrist"][ids], device=args.device) @ make_relative_transform(
            torch.as_tensor(bank["correct_rotvec"][ids, 0], device=args.device), torch.as_tensor(bank["correct_translation"][ids, 0], device=args.device))
        replay = max(float((correct_q-expected_q).abs().max()), float((correct_wrist-expected_wrist).abs().max()))
        max_replay = max(max_replay, replay)
        torch.testing.assert_close(correct_q, expected_q, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(correct_wrist, expected_wrist, atol=1e-6, rtol=1e-5)
        correct_points = model.point_surface(correct_q, correct_wrist)
        for condition_id, condition in enumerate(CONDITIONS):
            keep = np.ones(len(ids), dtype=bool) if condition not in donors else donors[condition][ids] >= 0
            selected = ids[keep]
            if not len(selected):
                continue
            source = donors[condition][selected] if condition in donors else selected
            q, wrist = decode_core(model, bank, selected, condition, source, args.device)
            points = model.point_surface(q, wrist)
            values = error_values(points, target[keep], q, target_q[keep], wrist, target_wrist[keep])
            change = torch.linalg.vector_norm(points-correct_points[keep], dim=-1).mean(dim=-1).cpu().numpy()*1000
            for j, sample in enumerate(selected):
                # Original extraction uses object-frame EPE, controls use the same points in world.
                if condition in {"correct", "identity"}:
                    previous = rows[2*int(sample) + (condition == "identity")]
                    if abs(previous["hand_epe_mm"] - float(values["hand_epe_mm"][j])) > 1e-3:
                        raise ValueError("World/object metric replay mismatch")
                    previous["output_change_mm"] = float(change[j])
                else:
                    row = dict(sample_id=int(sample), sequence_id=dataset.sequences[int(bank["sequence_number"][sample])].id,
                               start_frame=int(bank["start_frame"][sample]), condition=condition,
                               current_valid=bool(bank["cm_valid"][sample, 0]), donor_id=int(source[j]),
                               output_change_mm=float(change[j]), **{k:float(v[j]) for k,v in values.items()})
                    if condition in donors:
                        row["donor_differences"] = {k:float(v[0]) for k,v in pair_differences(bank, int(sample), [int(source[j])]).items()}
                    rows.append(row)
            for key, value in (("sample_id", selected), ("condition_id", np.full(len(selected), condition_id)),
                               ("q", q.cpu().numpy()), ("wrist", wrist.cpu().numpy())):
                predictions[key].append(value)
        if start % (args.batch_size*50) == 0 or ids[-1] == count-1:
            log(f"TEACHER {int(ids[-1])+1}/{count}")
    return rows, {k:np.concatenate(v) for k,v in predictions.items()}, max_replay


def rollouts(model, dataset, bank, starts, donors, lookup, args, budget, log):
    rows = []
    saved = {key: [] for key in ("sample_id", "condition_id", "step", "q", "wrist")}
    for condition_id, condition in enumerate(CONDITIONS):
        selected = [i for i in starts if condition not in donors or donors[condition][i] >= 0]
        for chunk in range(0, len(selected), args.batch_size):
            initial = np.asarray(selected[chunk:chunk+args.batch_size])
            q = torch.as_tensor(bank["current_q"][initial], device=args.device)
            wrist = torch.as_tensor(bank["current_wrist"][initial], device=args.device)
            for step in range(1, 17):
                budget()
                ids = np.asarray([lookup[int(bank["sequence_number"][i]), int(bank["start_frame"][i])+step-1] for i in initial])
                donor_base = donors[condition][initial] if condition in donors else initial
                source = np.asarray([lookup[int(bank["sequence_number"][i]), int(bank["start_frame"][i])+step-1] for i in donor_base])
                q, wrist = decode_core(model, bank, ids, condition, source, args.device,
                                       q=q, wrist=wrist, kinematics=dataset.kinematics)
                q = torch.maximum(torch.minimum(q, model.point_surface.finger_upper), model.point_surface.finger_lower)
                points = model.point_surface(q, wrist)
                target = torch.as_tensor(np.stack([np.asarray(dataset.sequences[int(bank["sequence_number"][i])].knn_hand_points[int(bank["start_frame"][i])+1]) for i in ids]), device=args.device)
                values = error_values(points, target, q, torch.as_tensor(bank["target_q"][ids], device=args.device),
                                      wrist, torch.as_tensor(bank["target_wrist"][ids], device=args.device))
                for j, sample in enumerate(initial):
                    rows.append(dict(sample_id=int(sample), sequence_id=dataset.sequences[int(bank["sequence_number"][sample])].id,
                                     condition=condition, step=step, start_frame=int(bank["start_frame"][sample]),
                                     donor_start=int(donor_base[j]), **{k:float(v[j]) for k,v in values.items()}))
                for key,value in (("sample_id", initial), ("condition_id", np.full(len(initial),condition_id)),
                                  ("step", np.full(len(initial),step)), ("q",q.cpu().numpy()), ("wrist",wrist.cpu().numpy())):
                    saved[key].append(value)
        log(f"ROLLOUT {condition}: {len(selected)} starts x 16 steps")
    return rows, {k: np.concatenate(v) if v else np.array([]) for k,v in saved.items()}


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for i, condition in enumerate(CONDITIONS):
        values = summary["teacher"][condition]
        if values:
            penalty=values["frame_micro"]["penalty_vs_correct_mm"]
            ci=values["micro_ci95"]["penalty_vs_correct_mm"]
            axes[0].bar(i, penalty, label=condition)
            axes[0].errorbar(i,penalty,yerr=np.array([[max(0,penalty-ci[0])],[max(0,ci[1]-penalty)]]),color="black",capsize=3)
        steps = [h for h in (1,4,8,16) if summary["rollout"][str(h)][condition]]
        axes[1].plot(steps,[summary["rollout"][str(h)][condition]["frame_micro"]["penalty_vs_correct_mm"] for h in steps],marker="o",label=condition)
    axes[0].set_xticks(range(len(CONDITIONS)),CONDITIONS,rotation=25)
    axes[0].set_title("Teacher-forced h1; paired control differences")
    axes[1].set_title("Recursive hand state; reference object poses")
    for ax in axes:
        ax.set_ylabel("EPE minus correct on same recipients (mm)")
        ax.grid(axis="y",alpha=.2)
    axes[1].set_xlabel("Executed steps")
    axes[1].legend(fontsize=8)
    fig.savefig(output/"dependence.png",dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--activity-id",required=True)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--budget-seconds",type=int,default=1800)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".",".."}:
        parser.error("run-id must be a single new directory name")
    torch.set_num_threads(2)
    run(args)
