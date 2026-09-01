# Cm 文档入口

## 规范目录

- `logs/status_log.md`：当前状态入口；
- `logs/architecture_log.md`：迁移期架构导航与历史事实；版本化架构快照只在用户指令下写入 `architecture/Vn.m.md`；
- `logs/repo_memory.md`：可迁移记忆；
- `logs/experiment_log.md`：正式实验记录；
- `logs/decision_log.md`：非平凡自主决策；
- `logs/modification_log.md`：AI 实际修改记录；
- `assets/`：Cm 专属外部资产入口（大型文件被 `.gitignore` 排除）；
- `指导/Vn.m.md`：用户维护的研究方案版本；
- `plan/Vn.m.md`：同版本执行计划，必须定稿后才能执行；
- `architecture/Vn.m.md`：用户明确要求后冻结的版本架构快照；
- `README.md`：本目录导航。

研究脚本统一位于 `../research/<experiment>/`。每个实验目录包含 `README.md`、`experiment.yaml`
和被忽略的 `output/<run_id>/`；正式训练和评估仍使用仓库根 `outputs/cm/<run_id>/`。根级
`output/research/` 与 `results/` 仅保留历史内容，不作为新入口。

旧版 `Cm框架.md`、`总体框架.md`、`Pipeline.md` 和 `项目说明.md` 仅作为历史研究记录保留，不再作为运行入口或兼容壳；新的架构快照只有在用户明确指令下写入 `architecture/Vn.m.md`。`logs/architecture_log.md` 在过渡期只作导航和待核对事实，不自动覆盖历史版本。

代码入口已经完成破坏性迁移：实现统一位于 `src/`、`dataset/` 和 `visualization/`，旧的顶层 Python 入口不再维护。历史运行如需复现，应固定到迁移前的提交或实验记录中的提交号。
