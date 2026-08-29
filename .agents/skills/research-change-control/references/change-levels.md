# Change level reference

Levels describe impact, not line count. A three-line coordinate-frame change can be L2; a
large generated diagnostic may still be L0 if it is not committed.

## L0 — routine

Configuration-only experiment variants, documentation, tests, formatting, and diagnostics
that preserve semantics. Use a new experiment config instead of mutating a baseline. No user
approval is required, but the change and validation are recorded.

## L1 — task-local implementation

Task-local code changes that preserve input/output schema, GT meaning, coordinate system,
split, metric, and checkpoint interpretation. Add focused tests and keep the existing entry
point compatible.

## L2 — scientific or public-contract change

Changing data fields, sampling, GT, coordinate frames, units, train/val/test boundaries,
metrics, cache schema, checkpoint compatibility, or a public component manifest. Explain the
old and proposed semantics and wait for user approval before editing.

## L3 — shared governance or infrastructure

Changing shared base runtime behavior, repository governance, dependencies, data migration,
destructive operations, external services, or long-running jobs. Present scope, risks,
rollback, and validation before implementation.
