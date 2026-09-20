# V1.21e snapshot-restore parity terminal state

- timestamp: 2026-09-20T21:44:06+08:00
- activity_id: ACT-20260920-214406-CMRESIDUAL-V121E-SNAPSHOT-PARITY-INVALID
- work_version: V1.21
- git_commit: `80568253b614205e226ae81385992748c91d0f09`
- branch: `ai/cmresidual/v121e-snapshot-restore`
- mode: run
- approval: user-approved [V1.21e FINAL plan](../plan/V1.21e.md) and GPU3 run
- run_id: `cmresidual_v121e_snapshot_parity_gpu3_20260920_2130`
- run_status: `COMPLETED`
- conclusion: `INVALID_IMPLEMENTATION`
- scientific_conclusion: `INCONCLUSIVE`

## Operation and evidence

GPU3（preflight `6 MiB`）使用官方 checkpoint、pinned DExplore runtime 和独立 fresh process 完成
16 states × prefix/restore × 2 duplicates。collection 为 exactly `6 moving / 6 contact / 4 precontact`；
每臂均执行 actor mean 的一个 30 Hz native action。运行目录为
`outputs/CmResidual/cmresidual_v121e_snapshot_parity_gpu3_20260920_2130/`，主要入口为
`run_manifest.json`、`config.json`、`selected_states.npz`、`metrics.jsonl`、`summary.json` 和
`logs/eval.log`；无 checkpoint。

16/16 state 均未通过完整 gate。prefix duplicate object hard ceiling 失败 `16/16`，restore duplicate
失败 `14/16`，cross-method object ceiling 失败 `16/16`：

- prefix duplicate position min/median/max：`0.003156 / 0.038611 / 6.313354 m`；rotation：
  `0.086144 / 0.634496 / 2.788521 rad`；
- restore duplicate position：`3.96e-7 / 0.043867 / 0.179051 m`；rotation：
  `2.16e-6 / 0.571079 / 1.657842 rad`；
- A/B centroid position：`0.001261 / 0.046306 / 3.498974 m`；rotation：
  `0.044814 / 0.604224 / 3.042105 rad`。

因此 duplicate repeatability 本身不满足 `5e-4 m / 5e-3 rad`，本协议不能把 A/B 差异解释为
snapshot hidden-state effect，也不能据此支持或反驳 direct restore 的 Markov 近似。

## Failed precursors and protected boundary

两个新 run_id 的 launcher failures 原样保留：`..._2121` 在 simulator 启动前因 Python 3.8 dict-union
不兼容失败；`..._2123` 完成 collection 后因 parity bootstrap 相对路径失败、未生成 A/B 数据。
两项已分别由提交 `9ce1ca9`、`8056825` 修复；临时 single-state wiring diagnostics 不作为科研证据。

未修改 threshold、reference/action/score/Cmv2/PPO，未启动 64-state calibration 或 512-state ranking。
GPU3 已释放至约 `6 MiB`。运行 manifest 的旧 producer 把 protocol `INVALID_IMPLEMENTATION` 同时写入
`scientific_conclusion`；本 Activity 按治理合同将科学结论纠正为 `INCONCLUSIVE`，后续 launcher 已修复
该元数据映射，不改写已完成 run artifact。

回滚入口是 revert V1.21e 三个实现提交；所有 outputs 保留且不纳入 Git。
