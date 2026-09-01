# AI 交接清单

## AI 开始前

固定阅读：

1. `AGENTS.md`；
2. `docs/版本线与AI行为分类.md`；
3. 根级 `docs/logs/status_log.md`、`architecture_log.md`、`repo_memory.md`；
4. 当前 Task 的 `docs/logs/`、`指导/Vn.m.md`、同版本已定稿的 `docs/plan/Vn.m.md`；
5. `git status --short`、当前 diff，以及涉及运行目录的进程状态。

按任务追加：

- 设计或历史兼容：`decision_log.md`、`modification_log.md`；
- 训练、评估、benchmark：`experiment_log.md` 和相关 `experiment.yaml`；
- 数据/cache：对应 cache 根的 `manifest.json`；
- checkpoint、MANO、URDF：对应 `asset_manifest.json` 和 `machine_memory.md`。

## AI 完成后

用户优先检查下面四项：

1. `git diff --stat` 和 `git diff`：确认实际修改范围；
2. 当前作用域最新的 `modification_log.md`：确认原因、等级、审批和验证；
3. `status_log.md`：确认进行中任务、阻塞和下一步；
4. 本次运行目录中的 `run_manifest.json`：确认代码、配置、数据、资产、seed 和输出没有漂移。

每次实质操作都要确认完整版本号 `Vn.m.k` 和操作类别；参数并行实验只新增细分版本和独立
run，不修改全局架构快照。指导文档只能由用户更新或在用户明确指令下更新。

如果执行了实验，还要检查 `experiment_log.md` 的假设、证据和结论状态，以及是否产生了规范
目录之外的新文件。AI 不得覆盖旧 run、不得创建 `results/`，不得把输出写到源码、Dataset 或
工具目录。
