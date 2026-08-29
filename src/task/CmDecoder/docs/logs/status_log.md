# CmDecoder 当前状态

- scope: task:CmDecoder
- last_updated: 2026-08-27
- last_verified: 2026-08-27
- related: [架构](architecture_log.md), [实验](experiment_log.md), [接手记忆](repo_memory.md)

## 当前状态

- 当前阶段 / 指导: wrist-aware baseline 保持相对 wrist SE(3)+6维手指 q 输出；已在修正腕部语义的3 Hz v2 cache 上完成纯固定对应点 loss 三组对照。逐手点 Cm flow → 联合 wrist+q fitting 路线也已实现。
- 当前进行中: 2026-08-27 已在物理 GPU 1 启动新版 Inspire F1 `Cm` checkpoint 驱动的全量 object-disjoint 3 Hz point-flow decoder，输出 `outputs/cmdecoder/cm_decoder_20260827_102629`；当前约 step `80000` / epoch `6`，epoch 5 已完成 validation。使用在线 Cm（`use_cached_cm_tokens=false`），10 epoch 总预算约 `143110` steps。旧的 geometry-only/no-time GRAB decoder 仍按原记录处理。统一 viewer 已支持显式 rollout 模式与原单帧模式共用 UI；未传轨迹时 rollout 选择器置灰。
- 最近可靠结论: 新版 Inspire decoder validation hand-flow EPE 从 epoch 1--5 的 `6.753/6.329/6.222/5.698/5.201 mm` 持续下降；epoch 5 相对 zero-flow=`12.118 mm` 改善约 `57.1%`，当前最佳 checkpoint 为 step `71555` / epoch `5`。仍未达到旧 decoder 的最终 `3.3256 mm` val / `3.726 mm` test，且尚无平台期证据。原 mixed C=64 decoder 已完成 10 epoch，GRAB val best hand-flow EPE 为 `2.702 mm`（step 36000）；其 step 22000 ARCTIC 复评为 teacher-forced `2.398 mm`、rollout 平均/末帧 `32.134/79.872 mm`。新增 Inspire→Inspire action-conditioned rollout（旧 point-flow checkpoint、31 个 30 Hz pair）point EPE mean/final=`47.675/94.188 mm`、wrist EPE mean/final=`48.136/94.727 mm`，当前状态反馈闭环明显发散。
- 阻塞 / 风险: v1 仅保留历史复现，不可用于 wrist-aware 点监督；旧版 Cm 与新版 subject-template Cm 的效果不能混作同一实验结论。当前逐点训练使用旧版 Cm，以隔离 decoder 架构变量。GRAB→Inspire F1 单序列有效窗口重定向仍出现约 `324 mm` object-relative wrist 漂移，当前无约束逐帧 fitting 不可作为可用重定向方案。
- 下一步: 等待 EXP-018 完成后，以最终 best 评估 Inspire F1 held-out hand-flow EPE、zero-flow 改善率，并用同一 `inspire_rollout` 入口复跑闭环；当前 epoch 1--5 仍持续改善，需继续观察 epoch 6--10 是否接近旧 decoder 水平及是否形成平台；不把单步 decoder 指标直接解释为无配对跨手型通用性或闭环 rollout 稳定性。
- 证据与相关文档: [接手记忆](repo_memory.md), [实验记录](experiment_log.md)
