# V1.18a planner latency profile

- timestamp: 2026-09-21T11:55:10+08:00
- activity_id: ACT-20260921-115510-CMRESIDUAL-V118A-LATENCY
- work_version: V1.18a
- git_commit: `2cb3e5e0f99975973048a0e8f5f0ce1010d1221a`
- branch: `ai/cmresidual/v118a-latency-profile`
- mode: run
- change_level: L2 implementation；GPU6 run 为 L3
- approval: 用户确认 V1.18a、当前基线、GPU6、Phase-B 首个 active state
- run_id: `cmresidual_v118a_latency_gpu6_20260921_1205`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（仅限 sampled-state latency hotspot）

## Operation and evidence

用 Phase-A checkpoint、seed 42、128 env 在 planner call 1 捕获首个 active pre-action state（env 0，state SHA
`9f66e70f…b905e75`），对同一 ordered candidate set 的 `K={1,2,4,8}` 分别做 20 次 warm-up 和 100 次
CUDA Event 测量。GPU6 preflight 为 4 MiB；输入 checkpoint、Cmv2、reference 与 source SHA 全部通过。

[run directory](../../research/v118a_latency/output/cmresidual_v118a_latency_gpu6_20260921_1205/) 包含 manifest、
config、log 和 profile。K=8 profiler 开关前后 `teacher_action/weight/activation/valid_fraction` 最大误差均为 0。
`predicted_cost_improvement` 自身 production duplicate 漂移 `136.89746`，profiled cross-run 漂移相同，不作为
rollout 行为 parity 字段。此前 run `cmresidual_v118a_latency_gpu6_20260921_1200` 因把该遥测要求 bitwise parity
而 `FAILED/INVALID_IMPLEMENTATION`，产物原样保留。

| K | total mean / median / p90 (ms) | swept + edge mean (ms) | local 占比 | peak allocated (MiB) |
| --- | --- | --- | --- | --- |
| 1 | 133.61 / 63.16 / 361.47 | 92.25 | 69.04% | 128.33 |
| 2 | 102.65 / 121.56 / 128.26 | 70.85 | 69.02% | 132.03 |
| 4 | 62.04 / 62.42 / 64.42 | 42.83 | 69.03% | 139.38 |
| 8 | 62.06 / 62.61 / 64.08 | 42.86 | 69.07% | 154.09 |

K=8 中 `swept_topk=20.54 ms (33.10%)`、`edge/contact=22.32 ms (35.97%)`，其次为 FK
`8.07 ms (13.00%)`；token/attention/effect 只有 `2.02 ms (3.26%)`，hand surface 为 `0.66 ms (1.06%)`。
因此 sampled state 上的主要耗时明确位于 dense local interaction。K1/K2 方差较大且 sweep 非单调，不能据此
建立 K-scaling 模型；该部分保持 `INCONCLUSIVE`。

无训练 epoch、best metric 或新 checkpoint。回滚入口为 revert `2cb3e5e`、`2c6c448`；诊断输出不纳入 Git。
