# Removed implementation notes

These changes were deliberately removed from `src/task/correspondence_ptv3_v2`
after the clean 5,000-step control did not support them. This document preserves
the implementation sufficiently to reproduce the archived configs.

## Model variants

All global-pose variants reused `HandGlobalEncoder`, which maps only
`(hand_points, hand_normals, hand_cano_points, finger_id, region_id)` to:

```text
a_j     : 64-D local anatomy / pose descriptor for hand surface point j
g_hand  : 32-D pooled hand-pose descriptor
```

The three tested fusions were:

```python
# C: early concatenation, with zero object context channels.
obj_feat = torch.cat([obj_feat, zeros(B, N_obj, 32)], dim=-1)
hand_feat = torch.cat([hand_feat, g_hand[:, None].expand(-1, N_hand, -1)], dim=-1)

# D: preserve B's 11-D PTv3 input; a scalar gate begins at exact identity.
z_hand = z_hand + torch.tanh(hand_context_gate) * hand_context_adapter(g_hand)[:, None]

# E: B's local edge logit plus a zero-initialized edge residual.
logit_ij = logit_local_ij + residual_head(concat(edge_shared_ij, a_j, g_hand))
```

For E, `residual_head` was `Linear(token_dim/2 + 64 + 32, token_dim/2)`,
`GELU`, `Linear(token_dim/2, 1)`; the last layer's weights and bias were zero.
The controls were implemented immediately before the last expression:

```python
if edge_global_context_mode == "zero":
    g_hand = torch.zeros_like(g_hand)
elif edge_global_context_mode == "batch_roll" and g_hand.shape[0] > 1:
    g_hand = g_hand.roll(1, dims=0)
```

For late-gated and edge-residual construction, the CPU RNG state was saved
before creating the extra modules and restored after it. This ensured that the
shared backbone/head initialization and subsequent stochastic stream matched B.

## Model-only warm start

The 5k pair started from a compatible B model rather than `resume`: shared
model tensors were loaded with `strict=False`, with only
`hand_global_encoder.*` and `edge_global_residual_head.*` allowed to be
missing. Optimizer, scheduler, RNG and global step remained fresh. For the
first 1,000 steps, only parameters under those two prefixes had
`requires_grad=True`; all parameters were released at step 1,000.

## Disk-safe checkpoint policy

Each full PTv3 + Adam checkpoint is about 1.1 GB. The clean rerun used scalar
validation every 1,000 steps but wrote checkpoints only at completed epoch
boundaries (`save_every_steps: null`, `save_every_epochs: 1`, `max_to_keep: 2`).
It additionally disabled the runner's automatic best-checkpoint write after a
step-level validation. This policy is archived rather than kept as a generic
base-runner option because it was only required by this short probe.

## Relation-feature ablations

The archival `relation_*.yaml` configurations used a runtime selector around
the established three opposite-cloud channels:

```python
full = torch.cat([nearest_distance, query_normal_dot_nn_direction,
                  query_normal_dot_opposite_centroid_direction], dim=-1)
mode_to_features = {
    "full": full,
    "none": full[..., :0],
    "nn_distance": nearest_distance,
    "nn_surface": torch.cat([
        nearest_distance,
        query_normal_dot_nn_direction,
        (-nn_direction * nearest_opposite_normal).sum(dim=-1, keepdim=True),
    ], dim=-1),
}
```

The main baseline intentionally keeps the original full 11-D formulation,
without this experimental selector.
