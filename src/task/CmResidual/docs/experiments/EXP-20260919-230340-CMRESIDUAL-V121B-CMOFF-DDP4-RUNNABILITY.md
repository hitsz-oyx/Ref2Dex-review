# V1.21b DExplore Cm-off DDP runnability

- work_version: V1.21
- conclusion: INCONCLUSIVE
- related_activity: [ACT-20260919-230340-CMRESIDUAL-V121B-CMOFF-DDP4-SMOKE](../activities/ACT-20260919-230340-CMRESIDUAL-V121B-CMOFF-DDP4-SMOKE.md)

## Hypothesis and protocol

假设：V1.20e vendor DExplore reconstructed-baseline 训练合同可以通过同一未来 Cm runner
的 Cm-off 分支运行，且零系数不构造 Cmv2。

固定 V1.20e 的 coordfix_v4 `s1_airplane_lift`、598-D input、18-D action、reward/physics、
seed 42 和 DDP facade；运行 4 rank x 64 env、horizon 64、local minibatch 256 的实际一 epoch
工程 smoke。运行使用 V1.20e 验证过的 `dexplore_v117` runtime，Cm-off bootstrap 要求
`cm_distill_coef=0` 和 `actual_epochs=1`。

## Evidence and limits

运行完成 65,536 frame，日志同时报告 4-rank 原生 `(1442,)` observation 与 `(18,)` action；
checkpoint `epoch=1` 可读，87 个递归 tensor 全 finite，manifest、event 和 log 完整。

同一合同的 4 rank x 2048 env / local-4096 容量门也已完成一实际 epoch：日志 total FPS
`174591.6`，checkpoint 内 `epoch=1`、`frame=2097152`，87 个递归 tensor 全 finite。完整
运行证据见 [capacity Activity](../activities/ACT-20260919-230550-CMRESIDUAL-V121B-CMOFF-DDP4-CAPACITY.md)。

这支持精确的 Cm-off DDP 接线、runtime 兼容性与 4 x 2048 容量。它只有一个 epoch、没有 Cm
planner，也没有独立抓取/lift 指标或多 seed 对照；因此不支持对抓取、收敛、V1.20e 完全训练
等价性或 Cm utility 的结论。
