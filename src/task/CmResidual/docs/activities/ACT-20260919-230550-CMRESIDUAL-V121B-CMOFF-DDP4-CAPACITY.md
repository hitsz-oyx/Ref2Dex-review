# V1.21b Cm-off DDP4 2048-env capacity smoke

- timestamp: 2026-09-19 23:05:50 +0800
- activity_id: ACT-20260919-230550-CMRESIDUAL-V121B-CMOFF-DDP4-CAPACITY
- work_version: V1.21
- mode: run
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认 V1.21b 最终计划并指示继续。
- branch: ai/cmresidual/v121-cm-actor
- git_commit: d9f8e7db1d53d2a0812a768ecfd01c50af01664a
- run_id: cmresidual_v121b_cmoff_ddp4_2048_globalmb16384_e1_20260919_231900
- run_status: COMPLETED

## Scope and fixed contract

执行 [V1.21b](../plan/V1.21b.md) Cm-off 容量门：V1.20e 的 `dexplore_v117` runtime、vendor
DExplore、coordfix_v4 reconstructed-baseline `s1_airplane_lift`、598-D input、18-D native
action、原 reward/termination/physics、seed 42、GPU `0,1,3,6`、4 rank x 2048 env、horizon
64、local minibatch 4096（global 16384）与 6 mini-epochs。Cm-off bootstrap 显式固定
`cm_distill_coef=0` 和 `actual_epochs=1`，不构造 Cmv2。

## Evidence

实际一 epoch 后所有 rank 同步退出，未见 OOM、NCCL、PhysX、non-finite 或 traceback。
[manifest](../../../../../outputs/CmResidual/cmresidual_v121b_cmoff_ddp4_2048_globalmb16384_e1_20260919_231900/run_manifest.json)、[log](../../../../../outputs/CmResidual/cmresidual_v121b_cmoff_ddp4_2048_globalmb16384_e1_20260919_231900/train.log)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v121b_cmoff_ddp4_2048_globalmb16384_e1_20260919_231900/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB.pth) 和 event 均存在。日志报告 epoch-1 mean reward `4.65`、total FPS `174591.6`；checkpoint 可读，递归 87 个 tensor 均 finite，保存字段为 `epoch=1`、`frame=2097152`。

`frame` 是上游 checkpoint 内部计数，未将其重解释为物理 env-step；按固定 rollout 合同的物理采样量为
4 rank x 2048 env x 64 horizon。reward 仅作运行状态，不是抓取或收敛证据。

## Result and next step

工程结论 `SUPPORTED`：同一 future-Cm runner 身份在 Cm-off 下通过了 V1.20e 的四卡 2048-env
容量配置。科学结论 `INCONCLUSIVE`：单 epoch、单 seed、无 Cm 且无行为评估，不能支持抓取、
收敛、官方等价性或 Cm utility。

无 Cm 门已完成；下一步是 V1.21b 第 3 阶段的 geometry bridge 审计和实现。未经 Cmv2/DExplore
mesh、frame、normal、1538/1024 sampling、18-DoF order 与 `dt=1/30` 合同验证，不启动 Cm-on。

## Rollback

本 Activity 不改变代码；删除或回滚文档不影响独立 output。所有数据、vendor、cache 和 checkpoint 保留。
