# Hand-pose prior experiment archive

This directory records the 2026-07-15/16 hand-prior exploration. It is kept
outside the task package because the controlled result does **not** support
adding pooled whole-hand context to the production correspondence model.

## Final result

- The sequence-held-out H-only probe could not predict contact topology or
  dense anchors better than an object-and-side conditional mean.
- A matched global-hand residual had a small temporary gain at step 3,000, but
  lost it by step 5,000; the `g_hand=0` capacity control was better on every
  final reported validation metric.
- The production choice is B: original 11-D local relation features plus the
  weak object-conditioned dense hand-contact loss (`0.005`).

`实验记录.md` contains the metrics and interpretation. `implementation.md`
and `archived_components.py` preserve the rejected mechanisms for a future,
candidate-gated study without reintroducing them into the main model.

The original full H-only probe script remains recoverable at
`hand_prior:tests/hand_prior_test/analyze_hand_pose_prior.py`; it is not copied
here because it is a standalone offline analysis utility rather than a mainline
training dependency.
