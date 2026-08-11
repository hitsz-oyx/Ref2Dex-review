import json

import numpy as np

from src.task.InteractionDynamics.viewer.server import mano_payload


def test_mano_payload_preserves_dynamic_frame_count_and_metadata(tmp_path):
    gt = np.zeros((9, 4, 3), dtype=np.float32)
    optimized = gt.copy()
    optimized[-1, :, 0] = .001
    path = tmp_path / "inverse.npz"
    np.savez(
        path, initial_points_world=gt, optimized_points_world=optimized,
        gt_points_world=gt, side=np.asarray("right"), target=np.asarray("interaction"),
        initialization=np.asarray("noise"), candidate_beta_offset=np.asarray(1.5),
        candidate_beta_seed=np.asarray(143),
        current_raw_frame=np.asarray(10), future_raw_frames=np.arange(11, 19),
        history_json=np.asarray(json.dumps([{"contact_f1": .9}])))

    payload = mano_payload(path)

    assert payload["num_frames"] == 9
    assert payload["candidate_beta_offset"] == 1.5
    assert payload["candidate_beta_seed"] == 143
    assert payload["target"] == "interaction"
    assert len(payload["step_epe_mm"]) == 9
    assert np.isclose(payload["step_epe_mm"][-1], 1.0)
