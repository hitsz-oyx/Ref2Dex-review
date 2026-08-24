from __future__ import annotations

import json

import numpy as np

from src.task.correspondence_ptv3_v2.hand_noise_profiles import (
    HandGeometryNoiseProfiles,
    infer_stage3_dataset_id,
    normalize_dataset_id,
)


class _Data(dict):
    @property
    def files(self):
        return list(self)


def test_dataset_id_normalization_and_legacy_signature_inference() -> None:
    assert normalize_dataset_id("ContactPose") == "contactpose"
    assert infer_stage3_dataset_id(_Data(dataset_name=np.asarray("ARCTIC"))) == "arctic"
    assert infer_stage3_dataset_id(
        _Data(
            mano_use_pca=np.asarray(True),
            mano_num_pca_comps=np.asarray(24),
            mano_flat_hand_mean=np.asarray(True),
            mano_pose=np.zeros((2, 24), dtype=np.float32),
        )
    ) == "grab"
    assert infer_stage3_dataset_id(
        _Data(
            seq_id=np.asarray("contactpose:full1_use/mug"),
            mano_pose=np.zeros((2, 15), dtype=np.float32),
        )
    ) == "contactpose"


def test_geometry_profiles_are_dispatched_by_dataset_and_descriptor(tmp_path) -> None:
    path = tmp_path / "grab.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ref2dex_mano_geometry_noise_calibration_v1",
                "results": [
                    {
                        "descriptor": "left_pca3_flat",
                        "num_components": 3,
                        "balanced_coefficient_std_per_dimension": [0.1, 0.2, 0.3],
                        "clip_sigma": 3.0,
                        "target_median_rms_mm": 9.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    profiles = HandGeometryNoiseProfiles.from_paths({"GRAB": path})

    profile = profiles.get(
        dataset_id="grab",
        side="left",
        use_pca=True,
        num_components=3,
        flat_hand_mean=True,
    )
    assert profile is not None
    assert profile.std_per_dimension == (0.1, 0.2, 0.3)
    assert profile.target_median_rms_mm == 9.0
    assert profiles.get(
        dataset_id="arctic",
        side="left",
        use_pca=True,
        num_components=3,
        flat_hand_mean=True,
    ) is None
