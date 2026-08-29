# Cm 文档入口

## 规范目录

- `logs/status_log.md`：当前状态入口；
- `logs/architecture_log.md`：完整架构事实源；
- `logs/repo_memory.md`：可迁移记忆；
- `logs/experiment_log.md`：正式实验记录；
- `logs/decision_log.md`：非平凡自主决策；
- `logs/modification_log.md`：AI 实际修改记录；
- `指导/`：研究方案版本；
- `README.md`：本目录导航。

旧版 `Cm框架.md`、`总体框架.md`、`Pipeline.md` 和 `项目说明.md` 仅作为历史研究记录保留，不再作为运行入口或兼容壳；新增架构事实应写入 `logs/architecture_log.md`，不要继续扩展旧文档。

代码入口已经完成破坏性迁移：实现统一位于 `src/`、`dataset/` 和 `visualization/`，旧的顶层 Python 入口不再维护。历史运行如需复现，应固定到迁移前的提交或实验记录中的提交号。
