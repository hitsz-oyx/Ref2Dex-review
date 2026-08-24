# CmDecoder 当前状态

- scope: task:CmDecoder
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [架构](architecture_log.md), [实验](experiment_log.md), [接手记忆](memory_log.md)

## 当前状态

- 当前阶段 / 指导: wrist-aware baseline 保持相对 wrist SE(3)+6维手指 q 输出；已在修正腕部语义的3 Hz v2 cache 上完成纯固定对应点 loss 三组对照。逐手点 Cm flow → 联合 wrist+q fitting 路线也已实现。
- 当前进行中: CmDecoder 的旧版 Cm C=256 point-flow baseline 与随机 horizon `stride=1..10` pilot 均已完成；四类 HRDexDB geometry cache 已完成 2088 个有效 episode 的导出和 object-disjoint manifest 生成。目前没有 CmDecoder 训练进程在运行。
- 最近可靠结论: pure-point `qt_cm` 的 full test 点 EPE 为 `3.801 mm`，相对 identity `12.697 mm` 改善约70%；wrist translation/rotation 为 `4.030 mm/1.908°`，均明显优于 identity `10.693 mm/2.288°`。`cm_only` 与其非常接近，`qt_only` 基本不能预测整体运动；但 `qt_cm` q MAE 为 `1.593°`，反而差于 identity `1.298°`，证明纯点几何监督不能唯一约束关节参数分解。旧版 Cm 的 point-flow baseline 已完成10 epoch，完整 full/high-motion val/test 的 hand-flow EPE 分别为 `3.325/3.726 mm` 和 `3.725/4.192 mm`；256 样本 fitting pilot 显示腕部误差明显下降，但 q 分解仍接近或略差于 identity。
- 阻塞 / 风险: v1 仅保留历史复现，不可用于 wrist-aware 点监督；旧版 Cm 与新版 subject-template Cm 的效果不能混作同一实验结论。当前逐点训练使用旧版 Cm，以隔离 decoder 架构变量。GRAB→Inspire F1 单序列有效窗口重定向仍出现约 `324 mm` object-relative wrist 漂移，当前无约束逐帧 fitting 不可作为可用重定向方案。
- 下一步: 如需正式报告，再对 point-flow baseline 做全量 point-flow→q/wrist fitting；当前 fitting 结果仅为每个 split 256 样本 pilot。HRDexDB Cm loader 可使用 `data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json`。当前 cache 只验证数据构建，不作为科研结果。
- 证据与相关文档: [接手记忆](memory_log.md), [实验记录](experiment_log.md)
