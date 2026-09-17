import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def _module():
    path = Path("src/task/CmResidual/cm_buffer.py")
    spec = importlib.util.spec_from_file_location("cm_residual_cm_buffer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cm_buffer_writes_reconstructable_state_without_point_clouds(tmp_path):
    module = _module()
    buffer = module.CmTransitionBuffer(tmp_path, rank=2, flush_every=2,
                                       metadata={"reference_sha256": "pinned", "schema_version": 1})
    record = {
        "q_pre": torch.zeros(2, 18), "dq_pre": torch.zeros(2, 18),
        "q_post": torch.ones(2, 18), "dq_post": torch.ones(2, 18),
        "object_pose_pre": torch.zeros(2, 7), "object_pose_post": torch.ones(2, 7),
        "residual_action": torch.zeros(2, 18), "reference_target": torch.zeros(2, 18),
        "pd_target": torch.zeros(2, 18), "applied_delta": torch.zeros(2, 18),
        "authority_limited": torch.zeros(2, 18), "reference_index": torch.tensor([4, 5]),
        "reference_start_index": torch.tensor([1, 1]), "window_progress": torch.tensor([3, 4]),
        "reference_delta_xi": torch.zeros(2, 6), "predicted_delta_xi": torch.zeros(2, 6),
        "predicted_effect_score": torch.zeros(2), "candidate_valid": torch.ones(2, dtype=torch.bool),
        "actual_delta_xi": torch.zeros(2, 6),
    }
    buffer.append(record)
    manifest = json.loads((tmp_path / "rank_002" / "manifest.json").read_text())
    assert manifest["schema"] == module.SCHEMA
    assert manifest["sample_count"] == 2
    assert manifest["storage"] == "compressed_npz_no_point_cloud"
    chunk = np.load(tmp_path / "rank_002" / "chunk_000000.npz")
    assert set(chunk.files) == set(record)
    assert not any("point" in key or "normal" in key or "flow" in key for key in chunk.files)
    np.testing.assert_array_equal(chunk["reference_index"], [4, 5])
