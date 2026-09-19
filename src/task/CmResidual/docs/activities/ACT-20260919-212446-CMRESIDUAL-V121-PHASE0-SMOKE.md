# V1.21 Phase 0 Cm-on engineering smoke

- timestamp: 2026-09-19 21:24:46 +0800
- activity_id: ACT-20260919-212446-CMRESIDUAL-V121-PHASE0-SMOKE
- work_version: V1.21
- mode: change → run
- change_level: L2
- approval: user-approved
- approval_basis: 用户确认采用 1-epoch Phase 0 与后续 10-epoch Phase A 的保守预算；本 Activity 只记录 Phase 0。
- branch: ai/cmresidual/v121-cm-actor
- git_commit: 6b4910f95d09d3e45f6aceadace7d929a11bb9c5
- run_id: cmresidual_v121_phase0_smoke_retry5_gpu6_20260919
- run_status: COMPLETED

## Contract and evidence

- GPU6、单 rank、128 env、64 horizon、seed 42、`cm_distill_coef=0.10`、K=8、`plannerEnvMicrobatch=16`、`interactionObjectChunk=32`；输入为 pinned `s1_airplane_lift` reference/source，Cmv2 checkpoint SHA 为 `371fb3396d8fc4ecea61de25178e58954090e26b2f2856aad925f25cb3b01591`。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v121_phase0_smoke_retry5_gpu6_20260919/)、[run manifest](../../../../../outputs/CmResidual/cmresidual_v121_phase0_smoke_retry5_gpu6_20260919/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v121_phase0_smoke_retry5_gpu6_20260919/config.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v121_phase0_smoke_retry5_gpu6_20260919/train.log)、[buffer manifest](../../../../../outputs/CmResidual/cmresidual_v121_phase0_smoke_retry5_gpu6_20260919/cm_buffer/rank_000/manifest.json)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v121_phase0_smoke_retry5_gpu6_20260919/train/CmResidualGrabReferenceV118_smoke/nn/CmResidualGrabReferenceV118PPO_19-21-20-09.pth)。
- 实际完成两次 epoch/update：vendored trainer 以 `epoch > max_epochs` 停止，旧 launcher 的 `max_epochs=1` 因而产生 epoch 1/2。随后 launcher 已修正为将请求 epoch 映射为 `trainer_max_epochs=epochs-1`；该已完成 smoke 不覆盖或删除。
- train log 记录 8 个 minibatch update、约 `59.5` total FPS、checkpoint 写入和正常退出；无 OOM、NCCL、PhysX error 或 traceback。checkpoint `epoch=2`、23 个 model entries 均 finite，`torch.load` 可读；`metrics.jsonl` 未由该 legacy runner 写入。

## Result, verification, and rollback

工程结论为 `SUPPORTED`：当前 GPU6 环境可以完成 Cm-on 的 128-env rollout、teacher、PPO update 和 finite checkpoint 写入。科学结论为 `INCONCLUSIVE`；它不表明抓取、Cm utility 或 Phase A/B 的学习效果。

此前四个独立失败 run 保留为诊断证据，分别暴露 critic normalizer、teacher dataset、value/policy output keys 和 KL helper signature 的 vendored API 兼容问题；它们不覆盖本 run。

定向 `py_compile`、13 个 Cmv2/adapter/planner tests、`git diff --check` 与 `python tools/verify.py --changed` 在相关修复提交上通过。回滚入口为 V1.21 的显式代码提交；任何回滚均不删除上述 outputs、checkpoint 或 buffer。
