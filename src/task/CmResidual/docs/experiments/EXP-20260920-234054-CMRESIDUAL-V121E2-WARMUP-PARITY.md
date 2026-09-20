# V1.21e.2 public snapshot short warm-up parity

- experiment_id: EXP-20260920-234054-CMRESIDUAL-V121E2-WARMUP-PARITY
- timestamp: 2026-09-20T23:40:54+08:00
- work_version: V1.21
- git_commit: `281d440316b217884b1054e0505c0ccf18ce9991`
- run_id: `cmresidual_v121e2_warmup_gpu3_20260920_2317`
- run_status: `COMPLETED`
- conclusion: `REFUTED`
- scientific_conclusion: `REFUTED`（仅限 `warmup<=8 insufficient`）

## Hypothesis and protocol

假设：从 canonical full-prefix 路径保存的 public `state_(t-L)` 恢复后，replay 最多 8 个真实 control
actions，足以在 candidate 前和执行相同 actor-mean candidate 后复现 fresh full-prefix 路径。协议固定
`L={0,1,2,4,8,full}`、6 个既有 `2/2/2` states，每个 `(state,L)` 两个独立 fresh single-env arms；
阈值、30 Hz/two-substep、shared score baseline 与 `t+6/t+1` clock 沿用 V1.21e。

canonical snapshots 由单独 generation pass 从 frozen episode initial state 完整 replay 后一次性物化，
不计入 72 arms且在 arms 期间只读。full 两臂的 sign-aligned centroid 是 candidate 前后 parity target。
只有某个 L 及全部更长窗口均在 6/6 states 通过，才能宣布最小可用 L。

## Evidence

[run directory](../../../../../outputs/CmResidual/cmresidual_v121e2_warmup_gpu3_20260920_2317/) 包含完整
manifest、episode/action/snapshot hashes、30 个 canonical snapshots、72 arm records、6 条 state metrics
与 summary。所有方法 duplicate 均为数值零，仅 quaternion geodesic 最大舍入 `5.16e-8 rad`；setter、
task/action/source identity、finite、snapshot immutability 与 object hard ceiling 全部通过。

五个 L 的 state-level pass 均为 `0/6`。`L=0` 在 candidate 前与 full public state 完全相等，但 candidate
后 object position 的 median/max 已为 `0.04537/0.13327 m`，DOF q median/max 为
`0.25158/0.31739 rad`。`L=1/2/4/8` 的 pre-position median 分别为
`0.05693/0.08724/0.05932/0.15325 m`，post-position median 分别为
`0.06685/0.11512/0.06305/0.15041 m`，没有随 L 增加而收敛或形成稳定通过后缀。

## Conclusion and limitations

实现合同有效，但所有 `L<=8` 均失败，所以窄假设为 `REFUTED`：本次 official checkpoint、单序列、
6 个 frozen interaction states 和 actor-mean one-step 协议下，最多 8 个 warm-up control steps 不足以
恢复 full-prefix parity。

结论不等于 snapshot/warm-up 总体被证伪；`L=16/32`、其它 public state schema 或不同重建协议未测试。
它也不定位 contact manifold、solver warm-start 或其它 PhysX internal state。不得外推到 Cm ranking、PPO、
抓取或泛化；当前证据不支持启动 512-state ranking。
