"""Run external DExplore with Isaac Gym's implicit tensor device bound to the Horovod rank."""
from __future__ import annotations

import os
from pathlib import Path
import runpy

import numpy as np


# Isaac Gym's Python 3.8 bindings and DExplore still use these aliases.
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int


def main() -> None:
    local_rank = int(os.environ.get(
        "HOROVOD_LOCAL_RANK", os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    ))

    # Import Isaac Gym before Torch.  Its stock helper defaults device='cuda:0',
    # which crosses devices on nonzero Horovod ranks.  Explicit device callers
    # retain their requested device; only omitted devices become rank-local.
    from isaacgym import torch_utils

    torch = torch_utils.torch
    original_to_torch = torch_utils.to_torch

    def rank_local_to_torch(x, dtype=torch.float, device=None, requires_grad=False):
        if device is None:
            device = f"cuda:{local_rank}"
        return original_to_torch(x, dtype=dtype, device=device, requires_grad=requires_grad)

    torch_utils.to_torch = rank_local_to_torch
    dexplore_run = Path("/home2/wyy/oyx_ws/dexplore/dexplore/run.py")
    runpy.run_path(str(dexplore_run), run_name="__main__")


if __name__ == "__main__":
    main()
