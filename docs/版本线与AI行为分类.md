# 版本线与 AI 行为分类规范

本仓库从 `V1.2.1` 治理操作起，按版本线组织研究、代码、数据和日志。版本号是事实关联键，
不是目录装饰；每个配置、实验和运行都应能回到一个版本线。

## 版本层级

```text
Vn                 # 大版本，由用户明确确认创建
└── Vn.m           # 小版本/计划边界，必须存在 docs/plan/Vn.m.md
    ├── Vn.m.1     # 细分操作、并行实验或状态变更，不单独创建 plan
    ├── Vn.m.2
    └── Vn.m.k
```

- `Vn` 只能由用户创建或确认。
- `Vn.m` 必须有同版本、标记为 `final` 的 `docs/plan/Vn.m.md`；计划未定稿时只能阅读、分析和修改计划。
- `Vn.m.k` 不需要单独 plan，但每个实质操作必须在相关日志中标明完整版本号。
- 纯粹的长任务进度刷新可以更新 `status_log.md` 而不创建新 `k`；改变代码、配置、数据、实验语义或结论时必须增加 `k`。
- `指导/Vn.m.md` 只由用户更新，或在用户明确指令下由 AI 更新；AI 不得自行改写指导。

## 版本化文档

- `docs/plan/Vn.m.md`：执行边界、接口、不变量、验证和回滚；由 AI 起草并与用户协商定稿。
- `docs/architecture/Vn.m.md`：该版本的架构快照；只有用户明确要求更新架构时创建或修改。
- `docs/logs/status_log.md`：当前细分版本、进行中任务、阻塞和下一步。
- `docs/logs/modification_log.md`：AI 实际修改的文件、类别、版本、审批和验证。
- `docs/logs/experiment_log.md`：实验假设、运行、证据和结论。
- `docs/logs/decision_log.md`：非平凡设计选择及其版本。

旧版本架构快照不得覆盖；查看历史优先使用 `docs/architecture/` 和 Git 历史。

## 操作类别

每个 `Vn.m.k` 至少选择一个类别：

| 类别 | 范围 |
| --- | --- |
| `governance` | 仓库规则、目录、版本、日志和交接规范 |
| `architecture` | Pipeline、Contract、数据语义和不变量 |
| `code` | 模型、loader、runner 和库实现 |
| `data` | 数据生成、cache、split、统计和数据 manifest |
| `experiment` | 训练、评估、benchmark 和并行对照 |
| `diagnostic` | t-SNE、可视化和离线诊断 |
| `operation` | 启动、停止、恢复、等待和资源状态 |
| `documentation` | 不改变语义的说明文档整理 |

如果一个操作无法归类，AI 应在回复和修改记录中说明并补充类别，而不是隐藏在“其他”中。

## 并行实验

并行实验使用同一个 `Vn.m` plan，在其下分配不同的 `Vn.m.k`、配置和 `run_id`。每个运行必须独立保存
配置快照、输入数据引用和输出目录。参数变体不更新全局架构；只有用户明确要求、且共享
合同或科学语义发生变化时，才创建或更新 `architecture/Vn.m.md`。

## Task 数据路径

Task 内保存代码、配置、实验索引和该 Task 的资产入口；实际数据/cache 不复制到 Task，训练/评估配置
直接声明外部路径。机器特有资产放在被忽略的 `src/task/<Task>/assets/`，运行时的配置快照、输入引用
和输出入口进入 `run_manifest.json`。
