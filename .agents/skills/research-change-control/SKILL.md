---
name: research-change-control
description: Govern changes in research or software repositories with transparent risk levels, minimal diffs, user approval gates, and document-to-diff consistency checks.
metadata:
  short-description: Keep research changes small and auditable
---

# Research change control

Use this skill whenever you modify repository code, configuration, data-processing logic,
experiments, or project documentation.

## Core behavior

- Confirm the requested outcome and preserve the user's existing changes.
- Prefer the smallest semantic diff. Do not refactor unrelated code, reformat whole files,
  or create a new component for a one-off helper.
- Declare a change level in the project's modification record. The level is an AI judgment
  for the user to review; do not pretend that a script can determine scientific impact.
- Stop before implementation when the change affects scientific meaning, public contracts,
  shared infrastructure, destructive state, or a long-running external job. Describe the
  alternatives and ask the user.
- Keep implementation, validation, and documentation in the same handoff. Do not claim an
  experiment result from a smoke test or an implementation error.

## Levels and approval

Read [change-levels.md](references/change-levels.md) when assigning a level. As a default:

- L0: configuration, documentation, tests, or diagnostics with unchanged semantics; may be
  completed automatically.
- L1: local implementation that preserves the public data/metric/checkpoint contract; may be
  completed automatically with focused validation.
- L2: changes to data fields, coordinate systems, GT, splits, caches, metrics, checkpoint
  interpretation, or public component contracts; ask before editing.
- L3: changes to shared framework behavior, repository governance, dependencies, migration,
  destructive actions, or long-running jobs; ask before editing and present an implementation
  plan.

If uncertain, choose the higher level. Record `approval: auto`, `pending`, or
`user-approved` according to the actual conversation. Do not infer approval from silence.

## Modification record

Use the nearest project-defined modification log. If the project has no format, use a concise
entry containing:

```markdown
## YYYY-MM-DD — <summary>

- change_level: L0 / L1 / L2 / L3
- approval: auto / pending / user-approved
- branch:
- post-commit:
- scope:

**Files**
- `path/to/file` — what changed

**Reason**
...

**Validation**
...
```

Run the project's consistency audit, or the bundled helper, before handoff:

```bash
python .agents/skills/research-change-control/scripts/audit_diff.py \
  --log <project-modification-log> --staged
```

The helper checks only objective consistency between the latest log entry and staged paths;
it does not grade the declared level or replace user review. Read
[audit-contract.md](references/audit-contract.md) for its limits.

## Validation and handoff

- Run the narrowest meaningful tests first, then broader tests when shared code or contracts
  changed.
- Record the exact validation command and result in the modification entry.
- Inspect `git status`, `git diff`, and staged diff before committing. Stage explicit paths;
  never include unrelated user changes.
- Keep generated data, caches, checkpoints, and outputs out of commits unless explicitly
  requested.
