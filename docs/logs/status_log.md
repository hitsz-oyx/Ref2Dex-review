# Ref2Dex 当前状态

- scope: root
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [仓库路径记忆](repo_notes_log.md) / [修改记录](modification_log.md)

## 当前状态

- 当前阶段 / 指导: `correspondence_ptv3_v2` 正在进行三域等比例训练，并准备加入 ARCTIC 数据域。
- 当前进行中: GPU 1、2 运行 GRAB / ContactPose / OakInk mixed 训练；GPU 3 在 tmux `arctic_full_mano_v21_20260823` 中导出全量 ARCTIC MANO Stage 2，成功后会自动继续 Stage 3。HOCap subject_1 annotation-only Stage 3 已完成并通过 loader 核验。
- 最近可靠结论: ARCTIC 原始数据、物体模板和 MANO assets 已通过启动检查，Stage 2 已开始向 NAS 写出首批文件；HOCap subject_1 已生成 28 个 Stage3 文件、23,896 个 frame samples，当前 loader 可 clean-load。
- 阻塞 / 风险: ARCTIC 全量数据导出尚未完成；完成前不得把目标 Stage 3 目录作为正式训练输入。HOCap 仅是外部测试集，接触距离为几何派生量。
- 下一步: 等待 ARCTIC 导出完成并核验；使用 HOCap Stage3 做外部 clean test 时，按 object/subject 统计结果，不接入当前四域训练。
- 证据与相关文档: `output/research/arctic_full_mano_v21_20260823/export.log`；任务级 [`status_log.md`](../../src/task/correspondence_ptv3_v2/docs/logs/status_log.md)。
