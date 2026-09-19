# V1.21 Phase A reference-PPO warm-up

- timestamp: 2026-09-19 21:28:59 +0800
- activity_id: ACT-20260919-212859-CMRESIDUAL-V121-PHASEA
- work_version: V1.21
- mode: run
- change_level: L2
- approval: user-approved
- approval_basis: [V1.21 final plan](../plan/V1.21.md)；用户明确要求先完成 Phase A 并在其可运行后审阅。
- branch: ai/cmresidual/v121-cm-actor
- git_commit: 77176924f8a67607b8e363402a432391687c3837
- run_id: cmresidual_v121_phasea_refppo_gpu6_20260919
- run_status: COMPLETED

## Contract and evidence

- GPU6、单 rank、128 env、64-step window、seed 42、从零初始化，实际 10 epochs / 81,920 frames；`cm_distill_coef=0`，V1.18 agent 因而不调用 Cm planner/Cmv2 teacher。
- [运行目录](../../../../../outputs/CmResidual/cmresidual_v121_phasea_refppo_gpu6_20260919/)、[run manifest](../../../../../outputs/CmResidual/cmresidual_v121_phasea_refppo_gpu6_20260919/run_manifest.json)、[config](../../../../../outputs/CmResidual/cmresidual_v121_phasea_refppo_gpu6_20260919/config.json)、[train log](../../../../../outputs/CmResidual/cmresidual_v121_phasea_refppo_gpu6_20260919/train.log)、[checkpoint](../../../../../outputs/CmResidual/cmresidual_v121_phasea_refppo_gpu6_20260919/train/CmResidualGrabReferenceV118_a/nn/CmResidualGrabReferenceV118PPO_19-21-27-14.pth)、[buffer manifest](../../../../../outputs/CmResidual/cmresidual_v121_phasea_refppo_gpu6_20260919/cm_buffer/rank_000/manifest.json)。
- 运行时间为 21:27:00–21:28:59 +0800。每 epoch total FPS 为 735.8–866.2；checkpoint 记录 `epoch=10`、`frame=81920`、23 个 model entries 全部 finite，`torch.load` 可读。该 legacy runner 未写 `metrics.jsonl`。

## Result and rollback

工程结论为 `SUPPORTED`：当前环境可按 V1.21 固定合同完成不调用 Cm planner 的 reference-PPO Phase A。科学结论为 `INCONCLUSIVE`：短预算的训练 log、FPS 或 finite checkpoint 不代表抓取、lift、收敛或 Cm utility。

回滚代码仅限 V1.21 显式提交；此独立 output、checkpoint 与 buffer 保留，不删除或覆盖。
