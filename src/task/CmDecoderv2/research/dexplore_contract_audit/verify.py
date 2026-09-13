"""Independently check saved raw states and NumPy/torch FK reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ...kinematics import InspireKinematics
from ...pointflow import DifferentiableInspireSurface, _v13_surface_samples
from src.task.ObjectInteractionCm.tools.data.build_dexplore_rl_cache import InspireUrdfModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    manifest = json.loads((root / "run_manifest.json").read_text())
    config = json.loads((root / "config.json").read_text())
    pairs = json.loads((root / "paired_manifest.json").read_text())["sequences"]
    samples = np.load(root / "state_samples.npz", allow_pickle=False)
    kin = InspireKinematics(config["urdf"])
    helper = InspireUrdfModel(Path(config["urdf"]))
    points, _, visual_ids = _v13_surface_samples(helper, config["surface_points"], 2024)
    surface = DifferentiableInspireSurface(config["urdf"], sample_count=config["surface_points"], surface_sampling="v1_3_cache")
    # Kinematics parses and traverses the URDF independently from the cache FK.
    def independent_points(q):
        links = kin.link_transforms_native(q)
        out = np.empty_like(points, dtype=np.float64)
        for visual_id, visual in enumerate(helper.visuals):
            mask = visual_ids == visual_id
            transform = links[visual.link] @ visual.local_transform
            out[mask] = points[mask] @ transform[:3, :3].T + transform[:3, 3]
        return out, links

    max_surface_diff, max_tip_diff, max_torch_diff = 0.0, 0.0, 0.0
    stored_tips = ("index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip")
    all_independent_surface, all_independent_tips = [], []
    torch_rows = set(np.linspace(0, len(samples["q"]) - 1, min(16, len(samples["q"])), dtype=int).tolist())
    for sample_index, (q, rec) in enumerate(zip(samples["q"], samples["reconstructed_q"])):
        finger = np.clip(q[[6, 8, 10, 12, 14, 15]], kin.finger_lower, kin.finger_upper)
        expected = np.array([*q[:6], finger[0], 1.05 * finger[0], finger[1], 1.05 * finger[1],
                             finger[2], 1.05 * finger[2], finger[3], 1.05 * finger[3],
                             finger[4], finger[5], .6 * finger[5], .8 * finger[5]])
        np.testing.assert_allclose(rec, expected, atol=1e-12, rtol=0)
        raw_points, raw_links = independent_points(q)
        rec_points, rec_links = independent_points(rec)
        error = np.linalg.norm(raw_points - rec_points, axis=-1).mean() * 1000
        tip_error = np.mean([np.linalg.norm(raw_links[name][:3, 3] - rec_links[name][:3, 3]) for name in stored_tips]) * 1000
        all_independent_surface.append(error)
        all_independent_tips.append(tip_error)
        max_surface_diff = max(max_surface_diff, abs(error - samples["surface_epe_mm"][sample_index]))
        max_tip_diff = max(max_tip_diff, abs(tip_error - samples["tip_epe_mm"][sample_index]))
        if sample_index in torch_rows:
            with torch.no_grad():
                predicted = surface(torch.tensor(finger[None], dtype=torch.float32),
                                    torch.tensor(rec_links["hand_base_link"][None], dtype=torch.float32)).numpy()[0]
            max_torch_diff = max(max_torch_diff, float(np.linalg.norm(predicted - rec_points, axis=-1).max() * 1000))
    if max_surface_diff > .002 or max_tip_diff > 1e-6 or max_torch_diff > .01:
        raise AssertionError(f"Independent FK mismatch: {max_surface_diff}, {max_tip_diff}, {max_torch_diff}")
    residual_all = {source: [] for source in ("geometric", "actual")}
    joint_rmse_all = {source: [] for source in residual_all}
    max_summary_difference = 0.0
    summary = json.loads((root / "summary.json").read_text())
    for sequence_index, pair in enumerate(pairs):
        for source_index, source in enumerate(("geometric", "actual")):
            array = torch.load(pair[source]["path"], map_location="cpu", weights_only=True).numpy()
            selected = (samples["sequence"] == sequence_index) & (samples["source"] == source_index)
            frames = samples["frames"][selected]
            np.testing.assert_array_equal(samples["q"][selected], array[frames, 373:391])
            q = array[:, 373:391].astype(np.float64)
            residual = q[:, [7, 9, 11, 13, 16, 17]] - q[:, [6, 8, 10, 12, 15, 15]] * [1.05, 1.05, 1.05, 1.05, .6, .8]
            residual_all[source].append(np.rad2deg(np.abs(residual)))
            # Independent least-squares solver, instead of the closed-form producer.
            basis = np.zeros((12, 6))
            basis[[0, 2, 4, 6, 8, 9], np.arange(6)] = 1
            basis[[1, 3, 5, 7, 10, 11], [0, 1, 2, 3, 5, 5]] = [1.05, 1.05, 1.05, 1.05, .6, .8]
            solved = np.linalg.lstsq(basis, q[:, 6:].T, rcond=None)[0].T
            solved = np.clip(solved, kin.finger_lower, kin.finger_upper)
            joint_rmse_all[source].append(np.rad2deg(np.sqrt(np.square(solved @ basis.T - q[:, 6:]).mean(axis=1))))
    def verify_stats(values, expected):
        values = np.asarray(values).reshape(-1)
        measured = {"count": len(values), "mean": values.mean(), "p50": np.median(values),
                    "p95": np.quantile(values, .95), "max": values.max()}
        nonlocal max_summary_difference
        for key, value in measured.items():
            difference = abs(float(value) - expected[key])
            max_summary_difference = max(max_summary_difference, difference)
            if difference > .002:
                raise AssertionError(f"Summary mismatch: {key} {value} != {expected[key]}")
    for source_index, source in enumerate(("geometric", "actual")):
        expected = summary["sources"][source]
        residual = np.concatenate(residual_all[source])
        verify_stats(residual, expected["micro"]["mimic_abs_deg"])
        verify_stats(residual.max(axis=1), expected["micro"]["max_mimic_abs_deg"])
        verify_stats(np.concatenate(joint_rmse_all[source]), expected["micro"]["optimal_joint_rmse_deg"])
        assert abs((residual.max(axis=1) > 1).mean() - expected["frame_fraction_over_1deg"]) < 1e-12
        for key, values in (("surface_epe_mm", all_independent_surface), ("tip_epe_mm", all_independent_tips)):
            source_mask = samples["source"] == source_index
            verify_stats(np.asarray(values)[source_mask], expected["micro"][key])
            macro = [np.asarray(values)[source_mask & (samples["sequence"] == i)].mean() for i in range(len(pairs))]
            verify_stats(macro, expected["macro"][key])
    for record in [*manifest["input_files"], *manifest["code_files"], *manifest["assets"],
                   *(pair[key] for pair in pairs for key in ("geometric", "actual", "canonical_human", "canonical_object"))]:
        path = Path(record["path"])
        if path.stat().st_size != record["size"] or path.stat().st_mtime_ns != record["mtime_ns"]:
            raise AssertionError(f"Input stat changed: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise AssertionError(f"Input hash changed: {path}")
    result = {"passed": True, "sequences": len(pairs), "sampled_states": len(samples["q"]),
              "max_surface_epe_difference_mm": max_surface_diff, "max_tip_epe_difference_mm": max_tip_diff,
              "max_torch_numpy_point_difference_mm": max_torch_diff,
              "max_checked_summary_difference": max_summary_difference,
              "input_stat_and_hash": "unchanged", "canonical_mesh_coordinates": "not_checked"}
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
