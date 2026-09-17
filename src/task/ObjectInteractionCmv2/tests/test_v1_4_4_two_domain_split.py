from __future__ import annotations

import json

import pytest

from src.task.ObjectInteractionCmv2.multi_domain import InspireSequenceView
from src.task.ObjectInteractionCmv2.tools.data.build_two_domain_mano_split_v1_4 import build
from src.task.ObjectInteractionCmv2.tests.test_v1_4_three_domain import _make_domain


def test_arctic_split_is_trajectory_stable_and_preserves_grab_splits(tmp_path):
    source_index = tmp_path / "source_index.json"
    source_manifest = tmp_path / "source_manifest.json"
    arctic = [
        {"id": f"arctic/s{i:02d}/tool_use", "dataset": "arctic", "path": f"/cache/arctic/{i}"}
        for i in range(30)
    ]
    source_index.write_text(json.dumps({"sequences": {
        "train": [{"id": "grab/train", "dataset": "grab", "path": "/cache/grab/train"}, *arctic],
        "val": [{"id": "grab/val", "dataset": "grab", "path": "/cache/grab/val"}],
        "test": [{"id": "grab/test", "dataset": "grab", "path": "/cache/grab/test"}],
    }}))
    source_manifest.write_text("{}")
    first = build(source_index, source_manifest, tmp_path / "first", 42)
    second = build(source_index, source_manifest, tmp_path / "second", 42)
    first_index = json.loads((tmp_path / "first/index.json").read_text())
    second_index = json.loads((tmp_path / "second/index.json").read_text())
    assert first["train"]["grab"] == 1
    assert first["val"]["grab"] == 1
    assert first["test"]["grab"] == 1
    assert first["train"]["arctic"] and first["val"]["arctic"]
    assert first_index["sequences"] == second_index["sequences"]


def test_derived_trajectory_split_requires_explicit_manifest_override(tmp_path):
    sequence, _, _ = _make_domain(tmp_path, "arctic", "mano")
    with pytest.raises(ValueError, match="split mismatch"):
        InspireSequenceView(sequence, "arctic", "val", "mano")
    view = InspireSequenceView(sequence, "arctic", "val", "mano", allow_manifest_split_override=True)
    assert view.split == "val"
