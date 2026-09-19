# V1.21b Cm-off DDP4 topology smoke

- timestamp: 2026-09-19 23:03:40 +0800
- activity_id: ACT-20260919-230340-CMRESIDUAL-V121B-CMOFF-DDP4-SMOKE
- work_version: V1.21
- mode: run
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 V1.21b 最终计划并指示继续。
- branch: ai/cmresidual/v121-cm-actor
- git_commit: ae43d1f7834d4497db59805e2e3f69cc4657dd23
- run_id: cmresidual_v121b_cmoff_ddp4_topology_smoke_v117_e1_20260919_231500
- run_status: COMPLETED

## Scope

执行 [V1.21b](../plan/V1.21b.md) 第 2 阶段的小规模 Cm-off DDP engineering smoke：
vendor DExplore `Dexplore_Inspire`、coordfix_v4 reconstructed baseline 的
`s1_airplane_lift`、598-D input、18-D native action、原 reward/termination/physics、
seed 42、GPU `0,1,3,6`、4 rank x 64 env、horizon 64、local minibatch 256、实际 1 epoch。
Cm-off bootstrap 显式固定 `cm_distill_coef=0` 与 `actual_epochs=1`。

## Runtime and evidence

- 使用 V1.20e 已验证的 `/home2/wyy/oyx_ws/.runtime_envs/dexplore_v117/bin/python`；
  该 runtime 的 `rl_games.Runner.model_builder` 与 vendor DExplore 兼容。`graspenv`
  的 Runner 缺少该属性，独立尝试
  `cmresidual_v121b_cmoff_ddp4_topology_smoke_20260919_221500` 在构建算法前失败，
  未产生 checkpoint。
- 严格预算 bootstrap 成功使 epoch 1 后所有 rank 同步退出；[manifest](../../../../../outputs/CmResidual/cmresidual_v121b_cmoff_ddp4_topology_smoke_v117_e1_20260919_231500/run_manifest.json)、[log](../../../../../outputs/CmResidual/cmresidual_v121b_cmoff_ddp4_topology_smoke_v117_e1_20260919_231500/train.log)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v121b_cmoff_ddp4_topology_smoke_v117_e1_20260919_231500/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth) 与 TensorBoard event 均存在。
- checkpoint 为 `epoch=1`、`frame=65536`，递归检查 87 个 tensor 全 finite；日志记录
  native observation `(1442,)`、action `(18,)`、epoch-1 mean reward `3.63` 和 total FPS
  `10603.7`。
- 先前 `cmresidual_v121b_cmoff_ddp4_topology_smoke_v117_20260919_230200` 因把 upstream
  `--max_iterations=0` 错当一实际 epoch 而未限预算；在 epoch 4 后已由预算门中断，目录、
  `STARTED` manifest 与日志保留，不能当作通过的 smoke 或科研证据。

## Protected

未修改 vendor DExplore、Cmv2、598-D input、18-D action、reward、termination、physics、
DDP global-gradient contract、数据、cache 或既有 checkpoint。Cm 模块未构造，Cmv2 未更新。

## Result and next step

工程结论 `SUPPORTED`：在 V1.20e 兼容 runtime 中，同一 Task-local Cm runner 身份可在
`cm_distill_coef=0` 下完成 4-rank DExplore PPO、同步停止并写出 finite checkpoint。
科学结论为 `INCONCLUSIVE`；本次只有 1 epoch、小规模 env，不能证明抓取、收敛、与
4x2048 等价或 Cm utility。

下一步是按 V1.21b 的容量门决定是否运行 4 x 2048 / local 4096 / global 16384 的有限
Cm-off run；通过前不开始 geometry bridge 或 Cm-on。

## Rollback

代码回滚只回退 `ae43d1f` 及其前的 Cm-off Task-local commits；所有三个独立 output 目录、
输入数据、vendor 和 checkpoint 保留。
