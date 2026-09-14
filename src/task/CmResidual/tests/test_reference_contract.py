from pathlib import Path

import numpy as np

from src.task.CmResidual.tools.data.build_reference import expand_q6


def test_urdf_authoritative_mimic_expansion():
    q = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    native = expand_q6(q, {7: 1.05, 9: 1.05, 11: 1.05, 13: 1.18, 16: 0.6, 17: 0.8})
    np.testing.assert_allclose(native[[6, 8, 10, 12, 14, 15]], q)
    np.testing.assert_allclose(native[[7, 9, 11, 13, 16, 17]], [0.105, 0.21, 0.315, 0.472, 0.36, 0.48])


def test_written_reference_has_explicit_frame_contract():
    root = Path("data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift")
    if not (root / "reference.npz").is_file():
        return
    with np.load(root / "reference.npz", allow_pickle=False) as z:
        assert z["frame_id"].shape == (367,)
        assert z["source_frame_id"][0] == 176
        assert z["source_frame_id"][-1] == 1640
        assert z["wrist_twist_world_ref"].shape == (367, 6)
        assert np.isfinite(z["q_native_ref"]).all()
