from __future__ import annotations

import json

from src.task.ObjectInteractionCmv2.tools.data.build_two_domain_mano_split_v1_4 import build


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
