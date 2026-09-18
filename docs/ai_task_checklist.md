# AI 任务执行与交接清单

> 本文件是开始、运行和完成任务时的快速检查表，不是独立规范源。安全与审批以
> [`AGENTS.md`](../AGENTS.md) 为准；版本和操作分类以 [`modification_policy.md`](modification_policy.md)
> 为准；路径与产物职责以 [`目录规范.md`](目录规范.md) 为准。

## AI 开始前

固定阅读：

1. `AGENTS.md`；
2. `docs/modification_policy.md`（涉及版本、记录或治理时）；
3. `docs/目录规范.md`（涉及数据、cache、资产、research、运行目录或 manifest 时）；
4. `docs/current_versions.yaml`、根级 `docs/activities/README.md`、`architecture_log.md`、`repo_memory.md`；
5. 当前 Task 的 `docs/README.md`、`activities/README.md`、完整指导谱系和对应的已定稿 plan；
6. `git status --short`、当前 diff，以及涉及运行目录的进程状态。

按任务追加：

- 设计或历史兼容：`decision_log.md`、legacy `activity_log.md`、历史 `modification_log.md`；
- 训练、评估、benchmark：`experiments/README.md`、相关 experiment card 和 `experiment.yaml`；
- 数据/cache：对应 cache 根的 `manifest.json`；
- checkpoint、MANO、URDF：对应 `asset_manifest.json` 和 `machine_memory.md`。

## AI 完成后

用户优先检查下面四项：

1. `git diff --stat` 和 `git diff`：确认实际修改范围；
2. 当前作用域最新的独立 Activity：确认时间戳、版本、git_commit、原因、等级、审批和验证；
3. 本次运行目录中的 `run_manifest.json`：确认代码、配置、数据、资产、seed、research_version 和输出没有漂移；
4. 最终回复中的 Markdown 链接：确认 plan、activity、run、manifest、metrics/train.log、checkpoint
   和关键产物都可跳转；BaseRunner 新运行不要求 `summary.json`。

链接显示文本使用仓库根目录相对路径；仓库文档的目标按文档位置解析，聊天回复的 target 使用客户端
支持的本地绝对文件路径。交接前运行 `python3 tools/verify.py --changed`。
回复中的路径链接与前后中文或标点之间留半角空格；运行终态不得留下缺失目标。

每次实质操作都要确认 governance/research/work version（适用时）和操作类别；参数并行实验只新增 Activity 和独立
run，不修改全局架构快照。指导文档只能由用户更新或在用户明确指令下更新。

如果执行了实验，还要检查 experiment card 的假设、证据和结论状态，以及是否产生了规范
目录之外的新文件。AI 不得覆盖旧 run、不得创建 `results/`，不得把输出写到源码、Dataset 或
工具目录。只要产生或更新 plan、activity、运行目录、manifest 或实验产物，最终回复必须提供可点击路径。
