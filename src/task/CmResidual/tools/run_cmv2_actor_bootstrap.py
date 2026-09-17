"""Python-script entry point for the V1.16 torchrun launcher."""
from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
TRAIN_SCRIPT = REPOSITORY_ROOT / "third_party/IsaacGymEnvs/isaacgymenvs/train.py"


def main() -> None:
    # IsaacGymEnvs' pinned dependency still accesses this NumPy compatibility alias.
    np.float = float
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
