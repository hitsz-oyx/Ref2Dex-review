import numpy as np

from src.task.ObjectInteractionCm.tools.data.pilot_arctic_inspire_geometric import TIP_VERTEX_IDS, TIP_NAMES


def test_arctic_pilot_landmark_contract():
    assert TIP_VERTEX_IDS.shape == (5,)
    assert len(TIP_NAMES) == 5
    # The five targets are appended after MANO's 16 wrist/joint positions.
    dummy = np.zeros((3, 21, 3), dtype=np.float32)
    assert dummy[:, 16:21].shape == (3, 5, 3)
