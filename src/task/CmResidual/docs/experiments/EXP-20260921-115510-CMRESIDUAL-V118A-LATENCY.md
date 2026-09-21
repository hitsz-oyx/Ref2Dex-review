# V1.18a frozen-Cmv2 planner latency hotspot

- experiment_id: EXP-20260921-115510-CMRESIDUAL-V118A-LATENCY
- timestamp: 2026-09-21T11:55:10+08:00
- work_version: V1.18a
- git_commit: `2cb3e5e0f99975973048a0e8f5f0ce1010d1221a`
- run_id: `cmresidual_v118a_latency_gpu6_20260921_1205`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（sampled-state hotspot）；K-scaling `INCONCLUSIVE`

## Hypothesis and protocol

假设：V1.19 streaming planner 的主要同步延迟来自 `swept_topk + edge/contact local interaction`，不是 PPO
update、surface generation 或 token/attention/effect head。固定真实 Phase-B 首个 active state、同一候选序列、
GPU6 和 `K={1,2,4,8}`，每个 K 做 20 warm-up + 100 CUDA Event measurements；不修改网络或 teacher 数学。

## Evidence and conclusion

[profile](../../research/v118a_latency/output/cmresidual_v118a_latency_gpu6_20260921_1205/profile.json) 的 K=8
端到端 mean 为 `62.055 ms`。`swept_topk` 与 edge/contact 分别为 `20.538 ms` 和 `22.324 ms`，合计
`42.863 ms / 69.07%`；FK 为 `8.070 ms / 13.00%`，token/attention/effect 为 `2.021 ms / 3.26%`，
hand surface 为 `0.655 ms / 1.06%`。四个 rollout 行为输出的 profiler parity 为数值零。

因此在这个 pinned active state 上，local interaction hotspot 假设为 `SUPPORTED`，并解释了 Cm 参与 rollout 时
同步 play-time 的主要增量；下一步若优化，应优先针对 dense swept neighborhood 与 edge/contact path，而不是 PPO
或小型 attention/head。K1/K2 的高方差导致 total latency 对 K 非单调，因此不能从本 run 推断候选数缩放规律、
整轮 FPS 或其它 state 分布；这些外推仍为 `INCONCLUSIVE`。
