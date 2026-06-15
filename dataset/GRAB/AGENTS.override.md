# Local Codex Override

This local AGENTS.override.md is intentionally ignored by Git and is only for the user's Mac Docker + Miniconda dry-run workflow.

## Mandatory validation rule

All validation for this repository must run through:

```bash
./.codex/dryrun.sh light
```

The dryrun must use Docker and the Miniconda environment inside the container.

Do not run full dryrun unless the user explicitly requests it.

## Forbidden on macOS host

Do not run project dependency or validation commands directly on macOS host, including:

* conda create
* conda install
* conda activate
* pip install
* pytest
* cmake
* colcon
* make
* ninja
* gradle
* training commands
* simulation commands
* robot/hardware commands

## Forbidden commands and workflows

Do not run:

* hardware commands
* camera commands
* CAN commands
* ADB commands
* CUDA/GPU commands
* simulation commands
* training commands
* dataset deletion/cleanup commands
* full dryrun unless explicitly requested

## Repository safety

Do not modify:

* submodule pointers
* large datasets
* checkpoints
* output directories
* `.codex/` infrastructure unless explicitly requested

Do not auto-commit.

If validation fails, classify the cause as one of:

* code error
* Docker environment issue
* Apple Silicon / arm64 architecture issue
* Miniconda/conda environment issue
* missing optional dependency
* missing hardware/simulation/CUDA
* unknown

If the failure is caused by hardware, simulation, CUDA, camera, CAN, ADB, or architecture limitations, report the limitation instead of forcing business-code changes.
