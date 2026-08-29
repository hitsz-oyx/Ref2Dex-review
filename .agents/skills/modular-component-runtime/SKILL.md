---
name: modular-component-runtime
description: Design or modify task-agnostic Component, Artifact, Contract, and Pipeline systems with explicit interfaces and safe composition.
metadata:
  short-description: Build composable research components
---

# Modular component runtime

Use this skill when adding or changing reusable components, manifests, artifact contracts,
pipeline composition, adapters, or registry behavior.

## Abstraction boundary

- Keep `Component` as the executable extension point; data adapters, models, solvers,
  evaluators, and exporters may all implement it.
- Use `Artifact` for values or persisted references and `Contract` for type, shape, dtype,
  schema, unit, coordinate, and temporal semantics.
- Use capabilities and tags for discovery. Do not make domain words such as encoder, decoder,
  robot, or dataset into framework-level categories.
- Registry discovery should be side-effect free. Import or instantiate entrypoints only in an
  explicit validation or execution command.

## Compatibility

- A manifest is the public declaration of a component. Keep existing IDs and semantics stable;
  create a new version when inputs, outputs, coordinate frames, GT, cache schema, or
  checkpoint interpretation changes.
- Do not create a new Component for a one-off helper or a hyperparameter-only experiment.
  Prefer a new configuration variant when the public contract is unchanged.
- Wrap existing task runners/models with adapters instead of moving or rewriting them during a
  first migration. Training lifecycle and inference composition may remain separate.

## Safety

- Validate contracts before execution and fail fast on incompatible metadata or missing inputs.
- Require explicit configuration, checkpoint, resource, and output locations for execution.
- Keep real training, data loading, DDP, and external side effects out of discovery and dry-run
  commands.
- When a proposed component change crosses a scientific or shared-framework boundary, use the
  change-control skill and obtain user approval before implementation.
