"""Run InterAct's GRAB canonicalizer without initializing unused SMPL-H models.

The pinned InterAct canonicalizer imports ``human_body_prior`` and constructs
SMPL-H models at module import time even when only the GRAB branch is run.
GRAB uses SMPL-X exclusively.  This launcher stubs only those unused import-time
objects, restores ``smplx.create`` immediately after import, and then calls the
original ``process_dataset('grab', ...)`` implementation unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
from pathlib import Path


class _UnusedBodyModel:
    """Import-time placeholder; the GRAB branch must never call this object."""

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs


def _install_unused_body_prior_stub() -> None:
    package = types.ModuleType("human_body_prior")
    body_package = types.ModuleType("human_body_prior.body_model")
    body_module = types.ModuleType("human_body_prior.body_model.body_model")
    body_module.BodyModel = _UnusedBodyModel
    package.body_model = body_package
    body_package.body_model = body_module
    sys.modules[package.__name__] = package
    sys.modules[body_package.__name__] = body_package
    sys.modules[body_module.__name__] = body_module


def run(interact_root: Path, workspace: Path) -> None:
    interact_root = interact_root.resolve()
    workspace = workspace.resolve()
    sys.path.insert(0, str(interact_root))
    sys.path.insert(0, str(interact_root / "text2interaction"))
    _install_unused_body_prior_stub()

    import smplx

    original_create = smplx.create

    def create_grab_only(*args, **kwargs):
        model_type = kwargs.get("model_type")
        if model_type is None and len(args) >= 2:
            model_type = args[1]
        if str(model_type).lower() == "smplh":
            return _UnusedBodyModel()
        return original_create(*args, **kwargs)

    previous_cwd = Path.cwd()
    try:
        os.chdir(workspace)
        smplx.create = create_grab_only
        module = importlib.import_module(
            "process.canonicalize_human_multi_thread"
        )
        smplx.create = original_create
        module.process_dataset("grab", "data/grab")
    finally:
        smplx.create = original_create
        os.chdir(previous_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interact-root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    run(args.interact_root, args.workspace)


if __name__ == "__main__":
    main()
