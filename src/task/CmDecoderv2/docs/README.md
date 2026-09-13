# CmDecoderv2 文档入口

CmDecoderv2 是独立的 Temporal-D2 decoder Task：冻结 ObjectInteractionCm，用连续 `K=4` 个
object-side Cm 解码右手 Inspire 的 6 个独立 finger q residual 与 wrist 相对 `SE(3)`，推理时只执行
第一个 horizon 并滚动重算。

## 当前入口

- 研究指导：[指导/v1.1.md](指导/v1.1.md)
- 最终计划：[plan/v1.1.md](plan/v1.1.md)
- 活动时间线：[logs/activity_log.md](logs/activity_log.md)
- 架构事实：[logs/architecture_log.md](logs/architecture_log.md)
- 实验记录：[logs/experiment_log.md](logs/experiment_log.md)
- 可迁移事实：[logs/repo_memory.md](logs/repo_memory.md)
- Rollout self-effect 诊断：[research/inspire_rollout_effect](../research/inspire_rollout_effect/README.md)
- MANO rollout self-effect 对照：[research/mano_rollout_effect](../research/mano_rollout_effect/README.md)
- MANO/Inspire Cm 来源分类：[research/cm_hand_source_classifier](../research/cm_hand_source_classifier/README.md)

代码事实最终以 `model.py`、`dataset.py`、`kinematics.py`、`runner.py` 和实际运行产物为准。
