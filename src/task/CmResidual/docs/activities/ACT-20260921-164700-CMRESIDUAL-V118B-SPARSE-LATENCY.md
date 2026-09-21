# V1.18b sparse local-interaction latency exploration

- timestamp: 2026-09-21T16:47:00+08:00
- activity_id: ACT-20260921-164700-CMRESIDUAL-V118B-SPARSE-LATENCY
- work_version: V1.18b
- git_commit: `90b61ca705200c569ea0ffc8711997d13f8a77b2`
- branch: `ai/cmresidual/v118b-sparse-interaction`
- mode: run
- change_level: L2 implementation；GPU6 benchmark 为 L3
- approval: 用户确认 [V1.18b FINAL plan](../plan/V1.18b.md) 与 GPU6 重复测量
- run_ids: `cmresidual_v118b_latency_interleaved_gpu6_20260921_1640`、`cmresidual_v118b_latency_interleaved_repeat_gpu6_20260921_1645`
- run_status: `COMPLETED` / `COMPLETED`
- conclusion: `REFUTED`（仅限本次三个 fast-path implementation 的可重复提速）

## Scope and evidence

新增默认关闭的 cumulative variants：squared-distance merge-topk、link-level swept AABB + exact narrow phase、
以及只对 radius-valid edges 执行 MLP。legacy dense 仍为生产默认；checkpoint/state_dict、K=8 teacher、2 cm
radius、V1.19 microbatch/chunk 和 PPO 合同不变。CPU 合同测试覆盖 tie、mask、zero flow、radius boundary、
dense/fast 有效边及模型输出，共 24 tests 通过；统一 verify 通过。

两个正式 run 均复用 V1.18a 的 state SHA `9f66e70f…b905e75` 与 candidate SHA
`0c3723c7…12628381`，GPU6、20 warm-up、100 measurements/variant，并在全部变体 warm-up 后使用循环移位的
interleaved schedule。所有变体的 `teacher_action/weight/activation/valid_fraction` parity 最大误差均为 0。

| repeat | variant | total mean / median / p90 (ms) | swept mean (ms) | edge mean (ms) | peak allocated (MiB) |
| --- | --- | --- | --- | --- | --- |
| 1 | baseline | 128.51 / 140.92 / 149.81 | 41.82 | 46.90 | 162.21 |
| 1 | merge | 161.71 / 179.98 / 188.01 | 75.29 | 46.23 | 158.64 |
| 1 | link-AABB | 111.37 / 121.49 / 128.59 | 27.14 | 45.28 | 256.64 |
| 1 | link+sparse | 131.60 / 143.58 / 150.94 | 27.43 | 65.07 | 256.65 |
| 2 | baseline | 69.05 / 65.11 / 71.19 | 22.15 | 24.88 | 162.21 |
| 2 | merge | 87.41 / 82.40 / 91.74 | 40.65 | 24.88 | 158.64 |
| 2 | link-AABB | 69.53 / 67.58 / 71.26 | 24.29 | 23.71 | 256.64 |
| 2 | link+sparse | 80.91 / 78.32 / 82.07 | 24.42 | 34.92 | 256.65 |

首次非交错 run `cmresidual_v118b_latency_gpu6_20260921_1630` 暴露固定顺序 confound，仅保留为运行证据，
不用于结论。交错 repeat 1 的 link-AABB mean 改善 13.33%，但 repeat 2 为回退 0.69%，且显存增加 94.43 MiB；
收益不可重复。merge 两次均慢约 26%，sparse-edge 两次均变慢。因此三个变体均不采用，未改变生产默认。

无训练 epoch、best metric 或 checkpoint。运行目录位于 `research/v118b_sparse_interaction/output/<run_id>/`。
回滚入口为 revert `90b61ca` 与 `b9fffb3`；所有运行产物保留且不纳入 Git。
