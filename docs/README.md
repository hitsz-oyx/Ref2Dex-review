# Ref2Dex documentation

这是文档目录的稳定入口。

- [项目总览](项目总览.md)：全局目标、Task 索引和非机器事实。
- [当前工作版本指针](current_versions.yaml)：各作用域唯一的 `work_version`。
- [根级 Activities](activities/)：跨 Task 与治理工作单元的索引和详情。
- [根级 Experiments](experiments/)：跨 Task 科学证据的索引和卡片。
- [研究变更 Skill](../.agents/skills/research-change-control/SKILL.md)：change/governance、版本、计划、记录与审批合同。
- [目录与产物 Skill](../.agents/skills/directory-and-artifacts/SKILL.md)：代码、数据、cache、资产、运行产物、manifest 与测试归属。
- [实验工作流 Skill](../.agents/skills/research-experiment-workflow/SKILL.md)：实验变量、run manifest、科学证据与 conclusion 合同。
- [长任务 Skill](../.agents/skills/long-running-tasks/SKILL.md)：启动、等待、监控、停止与恢复的操作合同。
- [测试目录与迁移清单](../tests/README.md)：现有根测试的 legacy 分类和未来测试归属。

Task 的入口位于 `src/task/<Task>/docs/README.md`。Task 不复制本文件或项目总览；局部事实、计划、Activity、实验和运行从 Task 入口递归链接。
