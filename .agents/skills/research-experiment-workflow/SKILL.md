---
name: research-experiment-workflow
description: Run and document reproducible research experiments, evaluations, data processing, and long-running jobs without mixing status, evidence, or generated artifacts.
metadata:
  short-description: Keep experiments reproducible and traceable
---

# Research experiment workflow

Use this skill for training, evaluation, benchmark, data processing, visualization, model
download, compilation, simulation, or any task expected to run for more than a short command.

## Before running

- Read the project's current status, architecture, memory, and relevant experiment/decision
  records. Follow the project's local paths and schema; do not replace them with machine-local
  assumptions.
- State the hypothesis, variables, data split, checkpoint initialization, metric, budget, and
  stopping condition. If any of these change scientific meaning, ask the user first.
- Keep formal experiment evidence separate from current status and implementation history.

## During and after running

- Use a new, explicit output directory for a new experiment. Never overwrite a baseline or
  silently select a latest checkpoint.
- Keep caches, checkpoints, raw data, and large generated results out of version control.
- Record the exact command, configuration snapshot, code revision, input/cache identifiers,
  and quantitative result. Mark conclusions as supported, refuted, inconclusive, or invalid
  implementation when appropriate.
- A smoke test proves wiring only; it is not a scientific result.

## Waiting

Start long jobs once and let the execution tool wait for minutes rather than repeatedly polling
from the model. Check earlier only when intermediate output changes the next decision. Use the
project's documented waiting mechanism and stop safely on errors, resource conflicts, or an
unapproved scope change.

For empty status polling with `functions.wait` or `write_stdin`, use a wait of at least 180
seconds (prefer 300 seconds when no intermediate output is needed). An outer wait must exceed
the longest nested wait by at least 30 seconds. Send non-empty interactive input immediately;
the long-wait rule is only for status polling.
