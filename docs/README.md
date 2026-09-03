# Ref2Dex documentation

这是文档目录的稳定入口。规范文档使用 ASCII canonical 文件名。

- [项目总览](项目总览.md)：全局目标、Task 索引和非机器事实。
- [当前版本指针](current_versions.yaml)：各作用域唯一的 `modification_version`。
- [AI 任务执行与交接清单](ai_task_checklist.md)：开始、运行和完成任务时的最小检查。
- [目录规范](目录规范.md)：数据、运行、研究和递归文档的放置规则。
- [修改版本与操作分类](modification_policy.md)：`modification_version` 和操作类别。
- [根级活动记录](logs/activity_log.md)：从切换点起的唯一工作时间线。
- [根级架构记录](logs/architecture_log.md)：共享 pipeline、schema 和 invariant。
- [测试目录与迁移清单](../tests/README.md)：现有根测试的 legacy 分类和未来测试归属。

Task 的入口位于 `src/task/<Task>/docs/README.md`（若该 Task 尚未迁移，则使用其现有 README 或
项目说明文件）。Task 不复制本文件或项目总览；局部事实、实验和运行从 Task 入口递归链接。
