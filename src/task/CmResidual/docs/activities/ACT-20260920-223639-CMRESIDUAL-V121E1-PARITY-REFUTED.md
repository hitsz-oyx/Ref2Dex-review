# V1.21e.1 independent single-env snapshot parity terminal state

- timestamp: 2026-09-20T22:36:39+08:00
- activity_id: ACT-20260920-223639-CMRESIDUAL-V121E1-PARITY-REFUTED
- work_version: V1.21
- git_commit: `0613a05ca81a6c1d0e381c66a062b9c3fced7eec`
- branch: `ai/cmresidual/v121e-snapshot-restore`
- mode: run
- change_level: L3（正式 GPU PhysX 小规模运行）
- approval: 用户批准 [V1.21e.1 FINAL 微补充](../plan/V1.21e.1.md) 与 GPU3 运行
- run_id: `cmresidual_v121e1_independent_parity_gpu3_20260920_2225`
- run_status: `COMPLETED`
- conclusion: `REFUTED`
- scientific_conclusion: `REFUTED`

## Operation and provenance

命令：

```text
python3 src/task/CmResidual/tools/run_v121e_snapshot_parity.py \
  --run-id cmresidual_v121e1_independent_parity_gpu3_20260920_2225 --gpu 3
```

physical GPU3 preflight 为 `2 MiB`，logical device 为 `cuda:0`。运行只读复用 V1.21e source run
`cmresidual_v121e_snapshot_parity_gpu3_20260920_2130`：manifest SHA256
`10556a3b3d07e854c8c1619f361813b0d9e1edebf0a5a19a3bc2cae34dcda9ca`，selected states SHA256
`7021f85fa2f132683bd5adacb030debabd8ccc384ffd5094caa0404df1c2aa55`。源 artifacts 未改写。

按 canonical `state_id` 选择 exactly `2 moving / 2 contact / 2 precontact`。每个 state 的 A1/A2/B1/B2
均为独立 fresh subprocess 且 `num_envs=1`，共生成 24 个 arm records。所有 setter input q/root
max-abs error 为 `0`，root/dof setter 均返回成功；6 个 canonical snapshot identity 与 action/task
identity gate 全部通过。

## Evidence and terminal result

输出目录：
`outputs/CmResidual/cmresidual_v121e1_independent_parity_gpu3_20260920_2225/`。主要入口为
`run_manifest.json`、`config.json`、`selected_states.npz`、`arms/*.json`、`metrics.jsonl`、
`summary.json` 和 `logs/eval.log`；无 step/epoch、best metric 或 checkpoint。

A1/A2 与 B1/B2 的所有数值 duplicate difference 均为 `0`，仅 quaternion geodesic 的最大舍入量
分别为 `2.98e-8 / 4.21e-8 rad`；两个 duplicate object hard ceiling 均为 `6/6` 通过。因此独立
single-env repeatability 成立，未触发 implementation failure。

但 A/B cross-method object ceiling 为 `0/6`：

- position min/median/max：`0.040614 / 0.051763 / 0.278021 m`；
- rotation min/median/max：`0.379410 / 0.774195 / 2.379353 rad`；
- DOF q max-abs min/median/max：`0.209528 / 0.261183 / 0.342512 rad`；
- physics score absolute difference min/median/max：`97.9114 / 355.7134 / 1088.9097`。

6/6 state 的 object position、rotation、DOF q/dq、IG 和 physics score gate 均失败；object linear
velocity 为 5/6 失败。协议有效而 hypothesis 未通过，所以 run 与科学结论均为 `REFUTED`：在这个
6-state、one-step、actor-mean 合同下，当前 public snapshot direct restore 不能替代 episode prefix
replay。它不确定缺失的具体 hidden state，也不产生 Cm ranking、PPO、抓取或泛化结论。

GPU3 终态占用约 `2 MiB`。未启动 16-state 扩大实验、64-state calibration、512-state ranking 或
PPO。回滚入口为 revert implementation commit `0613a05`；运行产物保留且不纳入 Git。
