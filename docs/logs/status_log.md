# Ref2Dex 当前状态

- scope: root
- last_updated: 2026-08-28
- last_verified: 2026-08-28
- related: [仓库记忆](repo_memory.md) / [修改记录](modification_log.md)

## 当前状态

- 当前阶段 / 指导: 已以 `oyx` 为主完成 `feature/hand-pca-perturbation` 的内容整合、递归 repo/machine memory 重构和全量回归验证。
- 当前进行中: Cm 保留 GRAB C=32 geometry-only/no-time 与双卡 GRAB C=64 容量对照；CmDecoder Inspire F1 全量 object-disjoint 3 Hz point-flow decoder 仍在训练。通用 Component/Artifact/Contract/ExecutionContext/PipelineSpec 原型已加入 `src/base`，当前三条主线已有只读 manifest，示例组件已支持显式 entrypoint 解析和 pipeline dry-run，尚未迁移真实执行逻辑。
- 最近可靠结论: feature 分支的 HOCap subject_1 Stage 3 已生成 28 个文件、23,896 个 frame samples并通过 loader 核验；Cm mixed 两条 run 已分别保留 epoch 48/40 的最近完整 checkpoint，详细指标见 Cm 状态与实验日志。
- 阻塞 / 风险: ARCTIC 全量 Stage 2/3 的最终产物状态仍需独立核验；HOCap 仅是外部测试集。合并代码已通过全量测试，但尚未在真实全量外部数据上重新导出。
- 下一步: 等待 CmDecoder GRAB→ARCTIC MANO 泛化结果，同时等待 Cm C=32/C=64 完成可比的中长程 validation，再决定是否恢复 correspondence 训练。
- 证据与相关文档: 全量 pytest `294 passed, 3 skipped`；`researchctl check --resolve-entrypoints` 与 `run --dry-run` 已通过；[仓库修改记录](modification_log.md)、[通用组件架构](architecture_log.md)、[correspondence 状态](../../src/task/correspondence_ptv3_v2/docs/logs/status_log.md)、[Cm 状态](../../src/task/Cm/docs/logs/status_log.md)。
