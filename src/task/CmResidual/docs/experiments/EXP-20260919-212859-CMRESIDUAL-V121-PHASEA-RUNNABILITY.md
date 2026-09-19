# V1.21 Phase A runnability

- work_version: V1.21
- conclusion: INCONCLUSIVE
- related_activity: [ACT-20260919-212859-CMRESIDUAL-V121-PHASEA](../activities/ACT-20260919-212859-CMRESIDUAL-V121-PHASEA.md)

## Hypothesis and protocol

在固定 `s1_airplane_lift` reference、1442-D actor、18-D residual、GPU6、128 env、64-step window 和 seed 42 下，`cm_distill_coef=0` 的 reference-PPO 路径能从零初始化完成有限训练预算，不调用 Cmv2 planner。

运行 `cmresidual_v121_phasea_refppo_gpu6_20260919` 共 10 epochs / 81,920 frames；Cm planner bypass 是该 Phase 的设计对照，不是 Cm 效果实验。

## Evidence and limits

运行在 1 分 59 秒内正常完成，日志的 total FPS 为 735.8–866.2，epoch-10 checkpoint 可读且参数 finite。证据支持工程 runnability。

本运行没有独立抓取、稳定 lift、contact/occupancy 汇总或多 seed 对照，且不含 Cm distillation；因此不能对抓取、学习效果、收敛或 Cm utility 得出结论。后续是否用该明确 checkpoint 进入 Phase B 须由用户另行授权。
