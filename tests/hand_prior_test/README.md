# Hand-pose prior experiment archive

This directory archives the 2026-07-15/16 hand-prior exploration. It is
intentionally outside the task package: the current conclusion does **not**
support adding pooled whole-hand context to the production correspondence model.

## Result

- The H-only, sequence-held-out data probe could not predict contact topology
  or dense anchors better than an object-and-side conditional mean.
- An edge-conditioned residual had a small, temporary matched-`g_hand` gain
  at step 3,000, but the gain disappeared by step 5,000. The `g_hand=0`
  capacity control was better on all reported validation metrics at the end.
- Direct global pooling / broadcasting and direct additive edge residuals are
  rejected for the main path. The retained baseline is `hand_heatmap.yaml`
  with its 11-D local relation features.

The detailed protocol, metric tables, and output directories are in
`src/task/correspondence_ptv3_v2/实验记录.md`.

## Contents

- `analyze_hand_pose_prior.py`: hand-only data probe, runnable with `graspenv`.
- `archived_components.py`: the tested residual, relation-selector, and
  model-only warm-start code in reusable form.
- `configs/`: inputs for context, residual-control, warm-start, and relation
  ablations. They require reapplying `implementation.md` before retraining.
- `implementation.md`: exact architecture, warm-start, and checkpoint-policy
  changes removed from the main code after the negative result.

No experiment process is active. Final clean outputs:

- `outputs/train/correspondence_ptv3_v2_global_prior_full_zero_clean_5k_20260716`
- `outputs/train/correspondence_ptv3_v2_global_prior_full_matched_clean_5k_20260716`
