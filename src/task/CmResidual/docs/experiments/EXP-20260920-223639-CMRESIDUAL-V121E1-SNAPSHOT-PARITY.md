# V1.21e.1 public snapshot one-step parity（independent single-env）

- experiment_id: EXP-20260920-223639-CMRESIDUAL-V121E1-SNAPSHOT-PARITY
- timestamp: 2026-09-20T22:36:39+08:00
- work_version: V1.21
- git_commit: `0613a05ca81a6c1d0e381c66a062b9c3fced7eec`
- run_id: `cmresidual_v121e1_independent_parity_gpu3_20260920_2225`
- run_status: `COMPLETED`
- conclusion: `REFUTED`
- scientific_conclusion: `REFUTED`

## Hypothesis and protocol

假设：完全独立的 fresh single-env simulator 中，episode-initial + prefix replay（A1/A2）与 direct
public snapshot restore（B1/B2）在相同 actor-mean one-step action 后具有可重复且相等的 next state。
协议、保护项和阈值固定于 [V1.21e.1 plan](../plan/V1.21e.1.md)。样本为 frozen V1.21e collection
中的 6 states，覆盖 `2/2/2` 三 phase；四个 arm 分别启动 subprocess，不共享 simulator。

## Evidence

[run directory](../../../../../outputs/CmResidual/cmresidual_v121e1_independent_parity_gpu3_20260920_2225/)
包含 frozen input identity、24 个 producer arm records、逐 state metrics 和 summary。A 与 B 各自的
duplicate divergence 为数值零（rotation 最大舍入 `4.21e-8 rad`），全部通过既有
`5e-4 m / 5e-3 rad` object ceiling；setter exact-copy、返回状态、task/action/snapshot identity 均通过。

相反，A/B cross-method position 差异为 `0.040614–0.278021 m`，rotation 为
`0.379410–2.379353 rad`，6/6 states 均超过 object ceiling，且其它主要状态/score gate 同向失败。

## Conclusion and limitations

实现合同有效、独立重复可复现，但 A/B parity 为 0/6，因此假设为 `REFUTED`。证据支持的窄结论是：
对本次官方 checkpoint、单序列、6 个 frozen interaction states 和 actor-mean one-step action，当前保存的
public q/dq、actor roots 与 task bookkeeping 不足以使 direct restore 等价于真实 prefix replay。

该结果不识别缺失量究竟是 contact solver cache、rigid-body/contact state、PD history 或其它内部状态；
也不证明所有 snapshot 设计均不可行。样本量仅 6、动作仅 actor mean、序列仅 `s1_airplane_lift`，不得
外推到 Cm ranking utility、PPO、抓取或泛化。按计划不自动扩大到 16 states，也不切换后续 ranking 到
snapshot branching。
