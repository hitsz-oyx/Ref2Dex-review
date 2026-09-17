from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.task.ObjectInteractionCmv2.multi_domain import InspireSequenceView
from src.task.ObjectInteractionCmv2.tools.data import build_oakink2_mano_highres_v1_4 as oak_mano
from src.task.ObjectInteractionCmv2.tools.data import build_stage4_inspire_highres_v1_4 as stage4_inspire


def _geometry(path: Path, *, schema: str, dataset: str, variant: str, points: int) -> Path:
    geometry = path / "geometry"
    geometry.mkdir(parents=True)
    frames = 2
    manifest = {
        "schema_name": schema, "sequence_id": f"{dataset}/demo", "dataset": dataset,
        "source_dataset": dataset, "source": variant, "hand_variant": variant, "split": "train",
        "coordinate_frame": "object_pose_t", "hand_side": "bilateral_merged_left_then_right",
        "frame_count": frames, "object_pool_points": 4096, "decoder_hand_points": 3076,
        "knn_hand_points": points, "knn_points_per_side": points // 2, "knn_k": 32,
        "effective_fps": 30.0,
    }
    (geometry / "manifest.json").write_text(json.dumps(manifest))
    np.save(geometry / "hand_points_world.npy", np.zeros((frames, 3076, 3), dtype=np.float32))
    np.save(geometry / "hand_normals_world.npy", np.zeros((frames, 3076, 3), dtype=np.float32))
    np.save(geometry / "knn_hand_points_world.npy", np.zeros((frames, points, 3), dtype=np.float32))
    np.save(geometry / "knn_hand_normals_world.npy", np.zeros((frames, points, 3), dtype=np.float32))
    np.save(geometry / "obj_points_pool_world.npy", np.zeros((frames, 4096, 3), dtype=np.float32))
    np.save(geometry / "obj_normals_pool_world.npy", np.zeros((frames, 4096, 3), dtype=np.float32))
    np.save(geometry / "obj_pose_world.npy", np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)))
    np.save(geometry / "source_frame_id.npy", np.asarray([0, 1], dtype=np.int32))
    np.save(geometry / "frame_time.npy", np.asarray([0.0, 1 / 30], dtype=np.float32))
    np.save(geometry / "obj_knn_indices.npy", np.zeros((frames, 4096, 32), dtype=np.uint16))
    np.save(geometry / "obj_candidate_mask_2cm.npy", np.zeros((frames, 4096), dtype=bool))
    return path


def test_decoder_is_derived_from_each_highres_side():
    points = np.arange(2 * 20270 * 3, dtype=np.float32).reshape(2, 20270, 3)
    normals = points + 1
    decoder, decoder_normals = stage4_inspire._decoder_from_highres(points, normals)
    assert decoder.shape == (2, 3076, 3)
    assert np.array_equal(decoder_normals, decoder + 1)
    assert np.array_equal(decoder[:, 0], points[:, 0])
    assert np.array_equal(decoder[:, 1538], points[:, 10135])


def test_new_producer_schemas_validate_and_loader_accepts_them(tmp_path):
    inspire = _geometry(tmp_path / "inspire", schema="ref2dex_object_interaction_cmv2_stage4_inspire_v1_4",
                        dataset="grab", variant="inspire_f1", points=20270)
    mano = _geometry(tmp_path / "mano", schema="ref2dex_object_interaction_cmv2_oakink2_mano_v1_4",
                     dataset="oakink2", variant="mano", points=4096)
    assert stage4_inspire._validate(inspire)["knn_index_max"] == 0
    assert oak_mano._validate(mano)["knn_index_max"] == 0
    assert InspireSequenceView(inspire, "grab", "train", "inspire_f1").hand_points == 20270
    assert InspireSequenceView(mano, "oakink2", "train", "mano").hand_points == 4096
