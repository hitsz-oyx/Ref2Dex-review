# ObjectInteractionCm 当前状态

- scope: task:ObjectInteractionCm
- last_updated: 2026-09-03
- last_verified: 2026-09-03
- 当前阶段: V1.1.3 混合训练后的表征可视化诊断
- 当前进行中: 无；已使用 best checkpoint 完成 test t-SNE
- 最近可靠结论: ObjectInteractionCm best 的 pooled-Cm dataset silhouette 为 `0.1835`；t-SNE 图显示 GRAB/Inspire-F1 仍有明显区域差异，stride 颜色在主体区域内大量交叠。
- 下一步: 如需判断 domain gap 是否由运动幅度造成，可在本产物基础上增加 hand-RMS 匹配子集或与旧 Cm 做同坐标协议对比。
- 证据: [experiment_log.md](experiment_log.md)、`output/research/objectinteractioncm_tsne_best_20260903/`
