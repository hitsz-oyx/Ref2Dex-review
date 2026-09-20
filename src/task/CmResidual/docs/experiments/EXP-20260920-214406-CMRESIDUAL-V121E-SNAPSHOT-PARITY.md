# V1.21e public snapshot 是否足够支持 one-step branching

- experiment_id: EXP-20260920-214406-CMRESIDUAL-V121E-SNAPSHOT-PARITY
- timestamp: 2026-09-20T21:44:06+08:00
- work_version: V1.21
- git_commit: `80568253b614205e226ae81385992748c91d0f09`
- run_id: `cmresidual_v121e_snapshot_parity_gpu3_20260920_2130`
- run_status: `COMPLETED`
- conclusion: `INVALID_IMPLEMENTATION`
- scientific_conclusion: `INCONCLUSIVE`

## Hypothesis and protocol

假设：collector 保存的公开 pre-action state 在 one-step actor-mean action 下，与从 episode initial state
进行 prefix replay 所产生的 next object/q/dq/IG/score 差异不超过 duplicate noise 和固定 object ceiling。
协议与阈值固定于 [V1.21e plan](../plan/V1.21e.md)，样本覆盖 6/6/4 三个 phase。

## Evidence and conclusion

[run directory](../../../../../outputs/CmResidual/cmresidual_v121e_snapshot_parity_gpu3_20260920_2130/)
包含完整 manifest、selection、32 arm records、逐 state metrics 和 summary。prefix duplicates 为 `0/16`
通过 object ceiling，restore duplicates 为 `2/16` 通过；因此对照臂自身不具备足够 repeatability，A/B
差异不可归因于未恢复的 PhysX hidden state。

实验为 `INVALID_IMPLEMENTATION`，关于“公开 snapshot 是否足够近似 Markov state”的科学结论仍为
`INCONCLUSIVE`。不得据此切换 64/512-state ranking 到 snapshot branching，也不得以事后放宽阈值解释。
