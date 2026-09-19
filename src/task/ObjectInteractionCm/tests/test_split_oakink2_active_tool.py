import json
import pickle

import numpy as np

from src.task.ObjectInteractionCm.tools.data.split_oakink2_active_tool import (
    SEGMENT_OBJECT_OVERRIDES,
    _resolve_active_roots,
    _roots,
    build,
)


def _resolve(sequence, frame_def, item, tree, names):
    root_of, members = _roots(tree)
    return _resolve_active_roots(
        sequence=sequence,
        frame_def=frame_def,
        item=item,
        root_of=root_of,
        members=members,
        names=names,
    )


def test_bilateral_multi_active_objects_are_split_by_root():
    item = {
        "obj_list": ["fork", "spoon"],
        "interaction_mode": "bh_main",
        "obj_list_lh": ["fork"],
        "obj_list_rh": ["spoon"],
    }
    result = _resolve("sequence", "range", item, {}, {"fork": "fork", "spoon": "spoon"})

    assert result["selected_roots"] == ["fork", "spoon"]
    assert result["selection_reason"] == "bilateral_multi_active_split"
    assert result["side_roots"] == {"left": ["fork"], "right": ["spoon"]}


def test_unique_tool_still_wins_over_passive_target():
    item = {
        "obj_list": ["scissors_part", "paper_part"],
        "interaction_mode": "bh_main",
        "obj_list_lh": ["paper_part"],
        "obj_list_rh": ["scissors_part"],
    }
    tree = {"scissors": ["scissors_part"], "paper": ["paper_part"]}
    names = {"scissors_part": "scissors", "paper_part": "paper"}
    result = _resolve("sequence", "range", item, tree, names)

    assert result["selected_roots"] == ["scissors"]
    assert result["selection_reason"] == "unique_active_tool"


def test_missing_laptop_object_list_uses_exact_semantic_override():
    sequence, frame_def = next(
        key for key, value in SEGMENT_OBJECT_OVERRIDES.items()
        if value["reason"] == "open_laptop_lid_missing_object_list"
    )
    item = {
        "obj_list": [],
        "interaction_mode": "lh_main",
        "obj_list_lh": [],
        "obj_list_rh": None,
    }
    tree = {"O02@0053@00003": ["O02@0053@00001", "O02@0053@00002"]}
    result = _resolve(sequence, frame_def, item, tree, {})

    assert result["object_ids"] == ["O02@0053@00001"]
    assert result["selected_roots"] == ["O02@0053@00003"]
    assert result["selection_reason"] == "semantic_object_override"
    assert result["side_roots"]["left"] == ["O02@0053@00003"]


def test_laptop_override_excludes_incidental_gamecontroller():
    sequence, frame_def = next(
        key for key, value in SEGMENT_OBJECT_OVERRIDES.items()
        if value["reason"] == "close_laptop_lid_excludes_incidental_gamecontroller"
    )
    item = {
        "obj_list": ["O02@0053@00001", "C43001"],
        "interaction_mode": "lh_main",
        "obj_list_lh": ["O02@0053@00001", "C43001"],
        "obj_list_rh": None,
    }
    tree = {"O02@0053@00003": ["O02@0053@00001", "O02@0053@00002"]}
    result = _resolve(sequence, frame_def, item, tree, {})

    assert result["original_object_ids"] == ["O02@0053@00001", "C43001"]
    assert result["object_ids"] == ["O02@0053@00001"]
    assert result["selected_roots"] == ["O02@0053@00003"]


def test_build_emits_two_candidates_and_run_manifest(tmp_path):
    annotation_root = tmp_path / "annotations"
    object_root = tmp_path / "objects"
    stage3_root = tmp_path / "stage3"
    output = tmp_path / "output"
    sequence = "synthetic_sequence"
    frame_def = "((0, 1), (0, 1))"
    annotation_root.mkdir()
    (object_root / "object_affordance").mkdir(parents=True)
    (object_root / "object_raw").mkdir()
    (object_root / "program" / "program_info").mkdir(parents=True)
    stage3_root.mkdir()

    (object_root / "object_affordance" / "object_part_tree.json").write_text("{}\n")
    (object_root / "object_raw" / "obj_desc.json").write_text(json.dumps({
        "fork": {"obj_name": "fork"},
        "spoon": {"obj_name": "spoon"},
    }))
    (object_root / "program" / "program_info" / f"{sequence}.json").write_text(json.dumps({
        frame_def: {
            "primitive": "rearrange",
            "obj_list": ["fork", "spoon"],
            "interaction_mode": "bh_main",
            "obj_list_lh": ["fork"],
            "obj_list_rh": ["spoon"],
        }
    }))
    pose0 = np.eye(4, dtype=np.float32)
    pose1 = np.eye(4, dtype=np.float32)
    pose1[0, 3] = 0.01
    with (annotation_root / f"{sequence}.pkl").open("wb") as stream:
        pickle.dump({
            "raw_mano": {0: {}, 1: {}},
            "obj_transf": {
                "fork": {0: pose0, 1: pose1},
                "spoon": {0: pose0, 1: pose1},
            },
        }, stream)

    stats_outputs = []
    for object_id, side in ((obj, side) for obj in ("fork", "spoon") for side in ("left", "right")):
        path = stage3_root / f"{object_id}_{side}.npz"
        np.savez(path, raw_frame_id=np.array([0, 1]), hand_to_obj_min_dist=np.array([0.01, 0.01]))
        stats_outputs.append({"seq_id": f"{sequence}/{object_id}/{side}", "file": str(path)})
    (stage3_root / "oakink2_stage3_stats_0.json").write_text(json.dumps({"outputs": stats_outputs}))

    result = build(
        annotation_root, object_root, stage3_root, output,
        run_id="synthetic-run", work_version="V1.4.3",
    )
    index = json.loads((output / "index.json").read_text())
    run_manifest = json.loads((output / "run_manifest.json").read_text())

    assert result["segment_count"] == 2
    assert result["uncertain_count"] == 0
    assert {row["selected_root_id"] for row in index["segments"]} == {"fork", "spoon"}
    assert all(row["selection_reason"] == "bilateral_multi_active_split" for row in index["segments"])
    assert run_manifest["run_id"] == "synthetic-run"
    assert run_manifest["run_status"] == "COMPLETED"
    assert run_manifest["counts"]["resolved_bilateral_multi_active_segments"] == 1
