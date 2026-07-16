# Archived implementation notes

The rejected variants encoded hand-only inputs
`(hand_points, hand_normals, hand_cano_points, finger_id, region_id)` as:

```text
a_j: 64-D local anatomy / pose descriptor at hand point j
g_hand: 32-D pooled whole-hand descriptor
```

The tested fusions were:

```python
# Early global broadcast
hand_feat = torch.cat([hand_feat, g_hand[:, None].expand(-1, n_hand, -1)], dim=-1)

# Late identity-start gate
z_hand = z_hand + torch.tanh(gate) * adapter(g_hand)[:, None]

# Zero-initialized edge residual
logit_ij = logit_local_ij + residual_head(torch.cat([edge_ij, a_j, g_hand], dim=-1))
```

For the edge probe, the residual head was
`Linear(token_dim/2 + 64 + 32, token_dim/2) -> GELU -> Linear(token_dim/2, 1)`
with the last layer weight and bias initialized to zero. `batch_roll` replaced
`g_hand` with another batch element; `zero` set only `g_hand` to zero.

The full-data pair used a model-only warm start: shared B tensors were loaded,
but optimizer/scheduler/RNG/global step stayed fresh. The residual branch alone
trained for 1,000 steps before B was unfrozen. This is an experiment-only
procedure, not part of the production runner.

The relation ablation retained the original three channels as the production
choice: nearest distance, query-normal / nearest-neighbour direction, and
query-normal / opposite-centroid direction. Removing them slowed optimization;
nearest-distance-only degraded zero-edge calibration.
