# Ref2Dex documentation

这是文档目录的稳定入口。规范文档使用 ASCII canonical 文件名。

- [项目总览](项目总览.md)：全局目标、Task 索引和非机器事实。
- [当前版本指针](current_versions.yaml)：当前 `governance_version` 与各 Task 的 `research_version`。
- [根指导谱系](指导/V1.1.md)：基线指导及其按字母排序的补充指导共同定义当前根治理要求。
- [目录规范](目录规范.md)：数据、运行、研究和递归文档的放置规则。
- [修改版本与操作分类](modification_policy.md)：版本身份、操作类别、指导谱系与记录合同。
- [根级 Activities](activities/README.md)：当前跨 Task 与治理工作单元索引。
- [根级 Experiments](experiments/README.md)：当前跨 Task 科学证据索引。
- [历史根日志](logs/activity_log.md)：只读审计入口。
- [根级架构记录](logs/architecture_log.md)：共享 pipeline、schema 和 invariant。
- [测试目录与迁移清单](../tests/README.md)：现有根测试的 legacy 分类和未来测试归属。

Task 的入口位于 `src/task/<Task>/docs/README.md`（若该 Task 尚未迁移，则使用其现有 README 或
项目说明文件）。Task 不复制本文件或项目总览；局部事实、实验和运行从 Task 入口递归链接。
