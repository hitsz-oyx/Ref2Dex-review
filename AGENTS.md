# Ref2Dex Agent 规范

本文件只规定 Ref2Dex 中必须始终成立的规则。工作方法由以下独立 Skill 说明：
`research-change-control`（change/governance）、`directory-and-artifacts`（路径与产物）、
`research-experiment-workflow`（实验方法与证据）和 `long-running-tasks`（长任务操作）。
项目当前事实由各级 README、指导、计划、Activity 和实验卡说明。

## 1. 权威入口

- 仓库导航：`docs/README.md`
- 当前工作版本：`docs/current_versions.yaml`
- Task 当前状态：`src/task/<Task>/docs/README.md`
- 研究目标：`src/task/<Task>/docs/指导/V<n>.md` 及同基线已确认的 `V<n><letter>.md`
- 仓库治理指导：`docs/指导/V<n>.md` 及同基线已确认的 `V<n><letter>.md`
- 已定稿方案：`src/task/<Task>/docs/plan/V<n>.md` 及适用的后缀补充
- 长期工作与科学证据：当前作用域的 `docs/activities/`、`docs/experiments/`

默认上下文只加载 AGENTS、current_versions、当前 Task README、当前指导谱系、相关代码和测试。
只有历史兼容、证据追溯或冲突需要时，才搜索 Activity archive、experiment、decision 或 Git 历史；
不得每次批量读取旧日志。

## 2. AI 工作模式

每个请求开始时声明一个主模式。模式切换时先说明新的范围和审批闸门。

| 模式 | 允许动作 | 入口要求 |
| --- | --- | --- |
| `inspect` | 阅读、扫描、解释和只读诊断 | 不改代码、配置、数据或研究结论 |
| `change` | 修改代码、配置、测试和普通文档 | 判断 L0–L2，必要时使用最终 plan |
| `run` | 训练、评估、benchmark、数据处理和长任务 | 代码、配置和研究变量已固定，使用 run manifest |
| `governance` | 修改 AGENTS、Skill、公共规范、目录或版本合同 | 按 L3，先说明范围、风险、回滚和验证并获用户确认 |

模式不改变研究目标，也不能覆盖用户已有修改、旧运行、数据、cache 或 checkpoint。

## 3. 影响等级和审批

- **L0**：说明文档、格式、测试、只读诊断和不改变运行变量的元数据。
- **L1**：Task 内实现，保持 observation、action、reward、GT、坐标、单位、split、指标和 checkpoint 解释不变。
- **L2**：改变研究语义或公共合同，包括 observation、action、reward、GT、坐标、单位、split、cache/schema、metric、checkpoint 解释和共享接口。
- **L3**：仓库治理、AGENTS、Skill、共享 `src/base`、依赖、数据迁移、破坏性操作或长时外部任务。

L0/L1 可以执行，完成后汇报。L2/L3 必须在编辑前说明拟改变和保持不变的内容、风险、验证和回滚，并得到用户确认。
修改 AGENTS、Skill 和公共治理合同固定为 L3。拿不准时按更高等级处理，不把沉默当作批准。

## 4. 身份、指导与计划

- `work_version` 是唯一版本术语，标识已确认的工作边界；根级用于治理或跨 Task 合同，Task 级用于局部工作。
- `git_commit` 是具体实现身份，`run_id` 是一次训练、评估、benchmark 或数据处理运行的身份；两者不是 version 字段。
- `work_version` 不因命令、测试、轮询、checkpoint 或训练过程状态自动递增。当前指针仅存于 `current_versions.yaml`；新配置、manifest、Activity 和实验卡只写 `work_version`。
- 既有 `modification_version`、`research_version`、`governance_version` 仅是历史输入或审计事实，不回写；新产物不得新增它们。

### 指导谱系与冲突

同一数值基线使用 `V<n>.md`，已确认细化使用单个小写字母后缀 `V<n><letter>.md`（`a`–`z`）。开始 change、run 或 governance 前，必须按无后缀基线、`a` 至最高已确认后缀的顺序读取全部同基线指导，不能只读最新文件。

不同文件的明确冲突由较后字母覆盖该冲突部分，其他约束共同生效；超过 `z` 不得自行定义排序。若同一指导文件内部存在冲突、无法同时满足或语义不充分，立即停止相关实施，向用户指出原文和影响范围，并先把候选解释、拟定措辞、保护项、风险、验证与回滚写入计划草案。仅在用户确认后，才可修改指导、定稿计划并继续。

### 指导与 plan 的独立生命周期

指导补充不自动要求新建 plan，也不与 plan 一一对应。已有 `a`、`b`、`c` 指导后才协商定稿的计划，仍使用 `plan/V<n>.md`。当已定稿并执行的基线 plan 受后续 `指导/V<n><letter>.md` 改变了可执行范围、保护项、风险、验证或回滚合同时，才建立 `plan/V<n><letter>.md`；后缀与触发变更的指导相同。仅澄清意图、未改变实施合同的指导补充不产生 plan。草案原地修订；已定稿 plan 不得静默覆盖，合同变化必须以追加 plan 并经用户确认。

开始 change、run 或 governance 时，先读取完整指导谱系，再读取基线最终 plan 与被后续指导实际触发的已确认计划补充。指导与计划无法同时满足时，停止实施并按上述冲突流程处理。

## 5. Git 和用户改动保护

`oyx` 是稳定集成分支。除非用户明确要求，不在 `oyx` 上直接开发；独立任务使用 `ai/<task>/<description>` 分支，完成实现、验证和交接后再合并。实验参数变化不创建分支。

开始分支工作前检查现有 worktree 及其 Git 状态。没有真实并行需求时，复用一个干净的 AI worktree，
不得为每个顺序分支新增 worktree；仅在并行工作、既有 worktree 有待保护修改，或用户要求长期保留隔离环境时
新增，并记录原因。已合并 worktree 只有在干净、无未跟踪内容且用户明确授权后才可移除；未知、用户拥有、含
修改或临时 worktree 不得触碰。移除 worktree 不会也不得隐含删除分支，删除分支须另获确认。

开始工作前检查 git status、git diff 和 git diff --cached。不得 reset、revert、overwrite 或带入用户已有修改；不确定归属时暂停并询问。只显式提交当前任务文件。outputs、checkpoint、cache、原始数据和大型日志不提交，根 output 只读且不新增内容。

## 6. 文档、记录和路径

- Task `docs/README.md` 是当前状态页，记录 work_version、研究问题、已接受架构、invariant、blocker、证据入口和重要 entrypoint；它不是历史流水账。
- 指导由用户维护研究目标、假设、成功标准和禁止事项；仅在用户确认同文件冲突的修订措辞后，AI 才可修改相应指导。
- 新 work_version 边界或 L2/L3 需要最终 `plan/V<n>.md`；L0/L1 修复不为每次任务创建 plan。
- `architecture/V<n>.md` 是按需冻结的架构快照，普通 bug fix 不更新。
- `activities/` 的独立 Activity 记录具有长期追溯价值的完成工作单元，`activities/README.md` 只做索引；重要实现、正式指导修订、重要诊断事实、重要操作、版本或治理切换应写 Activity。命令、单次测试、轮询、逐 step/checkpoint 和普通 smoke 不写 Activity。
- `experiments/` 的 card 只记录科学假设、证据与结论，一个 card 可关联多个 run。Activity 回答“做了什么”，experiment 回答“知道了什么”。
- `logs/activity_log.md`、`logs/experiment_log.md`、modification log、旧 experiment log、repo memory 和 machine memory 仅作历史审计或本机事实，不是默认上下文、当前入口或新记录目标，不批量回写。

Activity 至少包含 timestamp、activity_id 或 work_version、work_version、git_commit 或 base_commit、分支、scope、审批、验证和回滚入口。运行事件还包含 run_id、run_status、命令、输出目录、最后 step/epoch、best metric、关键 checkpoint 和 metrics.jsonl/train.log 入口（存在时）。run_status 与科学 conclusion 分开记录。

目录、数据、资产、运行产物、manifest 与测试归属的完整合同见 `directory-and-artifacts`；
涉及训练、评估、benchmark、数据处理等长任务时，run 模式同时使用 `long-running-tasks`。

当目录合同不能决定新内容的位置时，在创建、移动或写入前必须向用户确认位置。若发现已在
未确认位置先行创建、移动或写入内容，下一次回复必须优先汇报精确路径、内容/规模、原因、
合同偏离、影响、Git/忽略状态和保留/迁移/删除选项；该报告不构成追认。报告时还必须询问
用户是否将该例外提升为 `directory-and-artifacts` 的通用规范，未确认不得自行固化。

## 7. 标准工作流与门禁

1. inspect：读取最小上下文，确认 Task、work_version、保护边界和 Git 状态。
2. change：读取全部适用指导并审查冲突；新研究方案或 L2/L3 先形成并确认 plan；在任务分支实施最小差异并补定向测试。
3. run：读取全部适用指导并审查冲突，固定代码提交、work_version、假设、变量、指标、预算和停止条件；生成独立 run_id、配置快照和 run_manifest，结束后记录 Activity 与实验卡。
4. governance：治理修改按 L3 处理；审阅 diff、文档链接、兼容边界和回滚入口。

统一机器验证入口是：

    python tools/verify.py --changed

该入口在治理迁移第二阶段建立。第一阶段过渡期不能声称 VERIFY PASS；必须记录实际定向测试、语法检查和链接审计。

## 8. Ref2Dex 科学边界与交接

共享运行时在 `src/base/`，数据预处理在 `process/`，Task 实现在 `src/task/<Task>/`，独立研究在 `src/task/<Task>/research/<experiment>/`。实际数据和 cache 按 Skill 的目录合同放置，不复制到 Task 目录。代码、配置、数据处理和实验实现不得擅自改变坐标系、单位、GT、split、cache/schema、指标、checkpoint 解释或其他研究 invariant。

最终交接使用固定字段：Task、work_version、branch、git commit、research semantics、changed、protected、verification、scientific conclusion 和 next step。工程 smoke 只能证明接线或可运行性，不能冒充科研效果。
