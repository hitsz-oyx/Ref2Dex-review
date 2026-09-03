---
name: research-experiment-workflow
description: 运行并记录可复现的科研实验、评估、数据处理和长时任务，分离状态、证据与生成产物。
metadata:
  short-description: 让实验可复现且可追溯
---

# 科研实验工作流

训练、评估、benchmark、数据处理、可视化、模型下载、编译、仿真，或预计超过短命令的任务，都使用本 Skill。

## 运行前

- 阅读项目当前状态、架构、记忆，以及相关 experiment/decision 记录。遵循项目路径和 schema，不用机器私有假设替代它们。
- 明确假设、变量、数据划分、checkpoint 初始值、指标、预算和停止条件。如果其中任何一项改变科研语义，先请用户确认。
- 将正式实验证据与当前状态、实现历史分开记录。

## 运行中与运行后

- 新实验使用明确的新输出目录；不要覆盖基线，也不要静默选择“最新 checkpoint”。
- 缓存、checkpoint、原始数据和大型生成结果不纳入版本控制。
- 记录确切命令、配置快照、代码版本、输入/缓存标识和定量结果。运行清单至少包含 Task、`modification_version`、配置、数据、seed、checkpoint 和输出入口。活动记录还应登记精确到秒的时间戳、`activity_id`、可选 `run_id`、`run_status` 和 `base_commit`。结论标记为 `SUPPORTED`、`REFUTED`、`INCONCLUSIVE` 或 `INVALID_IMPLEMENTATION`。
- smoke test 只能证明 wiring 正常，不能作为科研结果。

## 默认产物位置与 Run Manifest

- 训练或评估运行使用新的 `outputs/<Task>/<run_id>/`（遵循仓库 Runner）；只读诊断使用
  项目目录规范声明的 Task-local research output path。Skill 不硬编码具体仓库路径。除非用户明确指定，不覆盖已有运行。
- 每次正式训练、评估、benchmark 或数据处理运行都生成一个小型
  `run_manifest.json`，记录 Task、`modification_version`、Git 提交、配置快照、输入数据或
  cache manifest 及其文件基本信息、schema、坐标系、seed、初始 checkpoint 和输出入口；指导/plan
  通过 activity 和 Markdown 路径链接关联，不在 JSON 中重复建立版本关联。
- 每次运行完成后生成用户可读的 `summary.json`（若入口不产生指标则明确写 `N/A`），并在 activity 中链接运行目录、config、manifest、summary 和关键产物。
- 重要实验在 `experiment.yaml` 用 `importance: primary|reference|exploratory`、`pinned: true`
  和 `tags` 标记导航优先级；该标记不等同于科学结论，结论仍须在 experiment log 中由证据支持。
- 数据 cache 的 manifest、训练运行的 run manifest 和实验日志职责不同：前者描述输入数据，
  中者锁定一次运行，后者记录科研假设与证据；不要用其中一个冒充另外两个。
- manifest、config 和 metadata 可以提交到外部实验存档，但大型 cache、checkpoint 和生成结果
  仍按仓库规则忽略，不纳入 Git。

## 运行状态与长时任务等待规则

- 不要求单独的实时状态 JSON 或 heartbeat；运行启动、停止、完成、失败和人工检查结果写入最近作用域的 activity log。
- `run_status`（`STARTED`、`RUNNING`、`COMPLETED`、`FAILED`、`STOPPED`、`UNKNOWN`）与科研 `conclusion`（`SUPPORTED`、`REFUTED`、`INCONCLUSIVE`、`INVALID_IMPLEMENTATION`）分开记录。
- `base_commit` 必须是活动发生时可见的最近一次已提交 HEAD 哈希；未提交工作区用独立的 dirty 标记说明，不把“未提交”写成 commit 值。
- 运行产物、activity、manifest 和 summary 的路径必须在 Agent 回复中以可点击 Markdown 链接给出。

完整规则见 [long-running-tasks.md](references/long-running-tasks.md)。核心要求是：让执行工具等待，不要让模型每几秒重新推理一次来查询状态。

- 启动长任务后按分钟级等待；只有中间输出会改变下一步决策时才提前检查。
- 出现错误、资源冲突或未批准的范围变化时安全停止。
- 空的状态查询与实际交互输入必须区分：前者长等待，后者立即发送。
