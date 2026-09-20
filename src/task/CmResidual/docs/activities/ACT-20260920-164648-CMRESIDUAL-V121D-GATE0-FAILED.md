# V1.21d native-init fresh-sim Gate 0 failed

- timestamp: 2026-09-20T16:46:48+08:00
- activity_id: ACT-20260920-164648-CMRESIDUAL-V121D-GATE0-FAILED
- work_version: V1.21
- mode: run
- change_level: L3（已确认的单次 GPU/PhysX 工程 smoke）
- approval: user-approved
- approval_basis: 在明确告知下一步需要单独确认 GPU/资源后，用户于 2026-09-20 要求“你现在继续吧”；授权范围仅为 V1.21d 单状态 Gate 0
- branch: ai/cmresidual/v121-cm-actor
- git_commit: 4607422dcc840a1e258133231aafb5641da2bd69
- run_id: cmresidual_v121d_native_init_gate0_gpu3_20260920_1645
- run_status: FAILED
- conclusion: INVALID_IMPLEMENTATION

## Scope and command

在预检时 GPU3 仅占用 `6 MiB`，满足 `1024 MiB` 容量门，因此固定 physical GPU3、logical
GPU0、seed 5909、`s1_airplane_lift`、1 个 source env 和 9 个 validation env。source simulator
从 DExplore `StateInit.Start` 采集初态与一个 native hold prefix step 后销毁；validation 使用新建
simulator 原生初始化并重放同一个 prefix。未加载或更新 PPO/Cmv2，未运行 collector、Gate A/B/C。

```text
python3 src/task/CmResidual/tools/run_v121d_native_init_smoke.py \
  --run-id cmresidual_v121d_native_init_gate0_gpu3_20260920_1645 \
  --physical-gpu 3 --execute
```

## Terminal evidence

输出目录：
`outputs/CmResidual/cmresidual_v121d_native_init_gate0_gpu3_20260920_1645/`

- manifest：`run_manifest.json`
- config：`config.json`
- log：`logs/eval.log`
- metrics：`replay_smoke/smoke_result.json`；本 smoke 不生成 `metrics.jsonl`
- episode artifact：`replay_smoke/artifacts/episodes/0.npz`
- pre-candidate snapshot：`replay_smoke/pre_candidate_state.npz`
- checkpoint、best metric、last epoch：不适用；最后完成 `1` 个 prefix control step，candidate step 为 `0`

source 和 validation simulator 均成功创建、销毁，各为一次；峰值 PyTorch GPU allocation
`44,175,872 bytes`，运行后 GPU3 回落至 `6 MiB`，无遗留进程。

原生初态 parity 通过：

```text
max public-state error       = 4.1723251e-7
max |delta q|                = 0
max |delta dq|               = 0
task/reference indices exact = true
```

执行一个相同 prefix step 后，candidate 前 parity 失败：

```text
max |delta q|                = 0.0111185312 rad
max |delta dq|               = 0.5535187721 rad/s
max root position error      = 1.5810132e-5 m
max root quaternion error    = 0.0004215240
max root twist error         = 0.0241000652
task/reference indices exact = true
required public-state parity <= 1e-5
```

q/dq 数值与 V1.21c 最终 tensor-restore smoke 的 `0.0111185312 / 0.5535187721` 基本逐值相同。
因此按门禁没有执行 candidate 或 duplicate step，没有生成 Cm ranking 证据。

## Protected

没有换 seed、重跑取最好结果、放宽 `1e-5` parity、删除 q/dq 检查或进入后续 gate。DExplore
checkpoint、frozen Cmv2、observation、action、reference、坐标、单位、GT、score、split 和统计门槛
不变；未修改代码、vendor、数据、cache、checkpoint 或旧 outputs。

## Scientific conclusion and blocker

run_status 为 `FAILED`，protocol conclusion 为 `INVALID_IMPLEMENTATION`，Cm ranking 科学结论仍为
`INCONCLUSIVE`。窄工程假设“tensor restore 是一 step 后 q/dq 分叉的主因，native init + fresh sim
可以恢复 9-env parity”被本次结果 `REFUTED`。当前证据把问题进一步定位到同一个 9-env simulator
内执行相同 PhysX step 后的跨 env 分叉，而不是初态恢复误差。

V1.21d 只允许 Gate A 失败后的 isolated single-env 诊断，没有授权在 Gate 0 已失败时自动运行该诊断；
因此本次停止。下一步需以新的后缀计划明确允许 12-state 之前的单状态、两次独立 1-env simulator
duplicate 诊断，并固定其资源与比较合同。

## Rollback

运行产物作为失败证据保留且不提交。文档回滚只撤回本 Activity、experiment card 和状态页更新；
不删除输出，不修改 V1.21d 最终计划或历史 V1.21c 失败证据。
