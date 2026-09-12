"""Independently recompute archived point errors and audit diagnostic artifacts."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .run import CONTACT_BINS, MOTION_BINS, ROOT, SOURCES, sha256


def verify(directory):
    out = Path(directory)
    summary = json.loads((out / "effect_summary.json").read_text())
    metadata = json.loads((out / "metadata.json").read_text())
    manifest = json.loads((out / "run_manifest.json").read_text())
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert all(Path(record["resolved_path"]).exists() for record in manifest["input_references"])
    assert all(sha256(path) == digest for path, digest in metadata["protected_digests"])
    assert all(sha256(ROOT / path) == digest for path, digest in metadata["code_digests"].items())
    assert [r["sample_index"] for r in rows] == list(range(len(rows)))
    seen = []
    max_error = 0.
    for item in summary["artifacts"]:
        with np.load(out / item["file"], allow_pickle=False) as arrays:
            indices = arrays["sample_index"].tolist()
            seen.extend(indices)
            local = [rows[i] for i in indices]
            gt = arrays["gt_obj_flow"].astype(np.float64)
            predictions = {"epe_mm": arrays["pred_obj_flow"], "zero_epe_mm": np.zeros_like(gt),
                           "shuffle_epe_mm": arrays["shuffled_obj_flow"],
                           "zero_token_epe_mm": arrays["zero_token_obj_flow"]}
            for key, prediction in predictions.items():
                epe = np.linalg.norm(prediction.astype(np.float64) - gt, axis=-1).mean(axis=-1) * 1000
                recorded = np.array([r[key] for r in local])
                max_error = max(max_error, float(np.max(np.abs(recorded - epe))))
                np.testing.assert_allclose(recorded, epe, atol=1e-4, rtol=2e-6)
            mse = ((arrays["pred_obj_flow"].astype(np.float64) - gt) ** 2).sum(axis=-1).mean(axis=-1) * 1e6
            np.testing.assert_allclose([r["mse_mm2"] for r in local], mse, atol=1e-3, rtol=2e-6)
            assert arrays["cm_tokens"].shape == (len(local), 16, 32)
            for i, row in enumerate(local):
                assert row["sample_valid"] == bool(arrays["sample_valid"][i])
                assert row["primary_valid"] == (row["sample_valid"] and row["full_active_count"] > 0)
                assert row["sequence_id"] == item["sequence_id"]
            assert all(np.isfinite(arrays[key]).all() for key in arrays.files)
    assert seen == list(range(len(rows)))
    selection = json.loads((out / "matched_selection.json").read_text())
    selected = selection["sample_indices"]
    assert len(selected) == len(set(selected))
    stratum_ids = []
    for stratum in selection["strata"]:
        assert len({len(ids) for ids in stratum["selected"].values()}) == 1
        for source, ids in stratum["selected"].items():
            stratum_ids.extend(ids)
            for i in ids:
                row = rows[i]
                assert row["primary_valid"] and row["source"] == source
                key = [row["object_name"]] + [int(np.searchsorted(bins, row[name], side="right") - 1)
                    for bins, name in ((MOTION_BINS, "hand_rms_mm"), (MOTION_BINS, "object_rms_mm"),
                                       (CONTACT_BINS, "active_fraction"))]
                assert key == stratum["stratum"]
    assert sorted(stratum_ids) == selected
    selected = set(selected)
    for name in ("primary", "matched"):
        for source in SOURCES:
            subset = [r for r in rows if r["source"] == source and
                      (r["primary_valid"] if name == "primary" else r["sample_index"] in selected)]
            stats = summary[name][source]
            if not subset:
                assert stats is None
                continue
            assert stats["samples"] == len(subset)
            assert stats["sequences"] == len({r["sequence_id"] for r in subset})
            np.testing.assert_allclose(stats["frame_micro"]["epe_mm"], np.mean([r["epe_mm"] for r in subset]))
    if not summary["smoke_only"]:
        assert len(summary["artifacts"]) == 58
        expected = {e["parent_seq_id"]: e["frame_count"] - 2 for e in metadata["entries"]}
        assert Counter(r["sequence_id"] for r in rows) == expected
    return {"verified_rows": len(rows), "verified_archives": len(summary["artifacts"]),
            "matched_rows": len(selected), "max_numpy_torch_epe_difference_mm": max_error,
            "manifest_references_exist": True, "digests_unchanged": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
