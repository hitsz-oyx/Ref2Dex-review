# Ref2Dex Agent 规范

本文件只规定 Ref2Dex 中必须始终成立的规则。工作方法由
`.agents/skills/research-change-control/` 和
`.agents/skills/research-experiment-workflow/` 说明；项目当前事实由各级
`README.md`、指导、实验卡和活动记录说明。

## 1. 权威入口

- 仓库导航：`docs/README.md`
- 当前治理和研究版本：`docs/current_versions.yaml`
- Task 当前状态：`src/task/<Task>/docs/README.md`
- 研究目标：`src/task/<Task>/docs/指导/V<n>.md` 及同基线已确认的 `V<n><letter>.md`
- 仓库治理指导：`docs/指导/V<n>.md` 及同基线已确认的 `V<n><letter>.md`
- 重大版本方案：`src/task/<Task>/docs/plan/V<n>.md`
- 实验索引和实验卡：`src/task/<Task>/docs/experiments/`
- 目录职责：`docs/目录规范.md`
- 版本和操作规则：`docs/modification_policy.md`

默认上下文只加载 AGENTS、current_versions、当前 Task README、当前指导谱系、相关代码和测试。只有出现历史
兼容、证据追溯或冲突时，才搜索 activity archive、experiment、decision 或 Git 历史；不得每次批量读取全部历史日志。

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
- **L1**：Task 内实现，保持 observation、action、reward、GT、坐标、单位、split、指标和 checkpoint
  解释不变。
- **L2**：改变研究语义或公共合同，包括 observation、action、reward、GT、坐标、单位、split、cache/schema、
  metric、checkpoint 解释和共享接口。
- **L3**：仓库治理、AGENTS、Skill、共享 `src/base`、依赖、数据迁移、破坏性操作或长时外部任务。

L0/L1 可以执行，完成后汇报。L2/L3 必须在编辑前说明拟改变和保持不变的内容、风险、验证和回滚，并得到用户确认。
修改 AGENTS、Skill 和公共治理合同固定为 L3。拿不准时按更高等级处理，不把沉默当作批准。

## 4. 三种版本身份

- `research_version`：研究方法身份；只有论文 Methods 需要改变时才递增。
- `git_commit`：具体实现身份；每个运行和重要活动都记录可复现的提交哈希。
- `run_id`：一次训练、评估、benchmark 或数据处理运行的身份。

治理合同的当前身份是 `governance_version`，记录在 `docs/current_versions.yaml`。bug fix、性能或显存优化、
测试、日志、seed、GPU 和训练失败不递增 research version；改变 observation、action、reward、planner formulation、
GT、split 或核心训练目标时才创建新的研究版本。

`work_version`（如 `V1.18.4`）可作为一个已确认工作单元的辅助追溯编号，通常对应一个 branch、Activity 和 PR；它不
替代 research_version，不进入 current_versions，也不因命令、测试或轮询自动递增。

### 指导谱系与冲突

同一数值基线使用 `V<n>.md`，其已确认细化使用单个小写字母后缀 `V<n><letter>.md`（`a`–`z`）。开始 change、run
或 governance 前，必须按无后缀基线、`a` 至最高已确认后缀的顺序读取全部同基线指导，不能只读最新文件。

不同文件的明确冲突由较后字母覆盖该冲突部分，其他约束共同生效；超过 `z` 不得自行定义排序。若同一指导文件内部
存在冲突、无法同时满足或语义不充分，立即停止相关实施，向用户指出原文和影响范围，并先把候选解释、拟定措辞、
保护项、风险、验证与回滚写入计划草案。仅在用户确认后，才可修改该指导、定稿计划并继续。

## 5. Git 和用户改动保护

`oyx` 是稳定集成分支。除非用户明确要求，不在 `oyx` 上直接开发；独立任务使用
`ai/<task>/<description>` 分支，完成实现、验证和交接后再合并。实验参数变化不创建分支。

开始工作前检查 git status、git diff 和 git diff --cached。不得 reset、revert、overwrite 或带入用户已有修改；
不确定归属时暂停并询问。只显式提交当前任务文件。outputs、checkpoint、cache、原始数据和大型日志不提交，
根 output 只读且不新增内容。

## 6. 文档、计划和记录

- Task `docs/README.md` 是当前状态页，记录 research version、研究问题、已接受架构、invariant、blocker、证据
  入口和重要 entrypoint；它不是历史流水账。
- `指导/V<n>.md` 及其字母补充指导由用户维护研究目标、假设、成功标准和禁止事项；仅在用户确认同文件冲突的修订
  措辞后，AI 才可修改相应指导。
- 只有新研究版本或 L2/L3 方案需要最终 `plan/V<n>.md`；L0/L1 修复不为每次任务创建 plan。
- `architecture/V<n>.md` 是按需冻结的架构快照，普通 bug fix 不更新。
- 重要正式实验使用 experiment card，记录假设、配置、run、证据、结论和限制。
- `activities/` 的独立 Activity 记录具有长期追溯价值的完成工作单元；`activities/README.md` 只做索引。
  重要实现、正式指导修订、重要诊断事实、重要操作、研究版本或治理切换应写 Activity；单行修复、命令、逐 step
  进度、轮询和普通 smoke 不写。`experiments/` 的 card 只记录科学假设、证据与结论；一个 card 可关联多个 run。
- `logs/activity_log.md`、`experiment_log.md` 及其他旧日志仅作只读历史审计，不批量回填或拆分。
- 历史 modification log、旧 experiment log、repo memory 和 machine memory 保留为只读或本机事实，不批量重写。

Activity 至少包含 timestamp、activity_id 或 work_version、governance_version 或 research_version、git_commit、分支、scope、
审批、验证和回滚入口。运行事件还包含 run_id、run_status、命令、输出目录、最后 step/epoch、best metric、关键
checkpoint 和 metrics.jsonl/train.log 入口（存在时）。run_status 与科学 conclusion 分开记录。

## 7. 标准工作流

1. inspect：读取最小上下文，确认 Task、研究版本、保护边界和当前 Git 状态。
2. change：读取全部适用指导并审查冲突；新研究方案或 L2/L3 先形成并确认 plan；在任务分支实施最小差异并补定向测试。
3. run：读取全部适用指导并审查冲突，固定代码提交、研究版本、假设、变量、指标、预算和停止条件；生成独立 run_id、配置快照和 run_manifest，
   结束后记录 activity 与实验卡。
4. governance：治理修改按 L3 处理；审阅 diff、文档链接、兼容边界和回滚入口。

统一机器验证入口是：

    python tools/verify.py --changed

该入口在治理迁移第二阶段建立。第一阶段过渡期不能声称 VERIFY PASS；必须记录实际定向测试、语法检查和链接审计。

## 8. Ref2Dex 路径和科学边界

共享运行时在 `src/base/`，数据预处理在 `process/`，Task 实现在 `src/task/<Task>/`，独立研究在
`src/task/<Task>/research/<experiment>/`。实际数据和 cache 按 `docs/目录规范.md` 放置，不复制到 Task 目录。
代码、配置、数据处理和实验实现不得擅自改变坐标系、单位、GT、split、cache/schema、指标、checkpoint 解释或其他
研究 invariant。

最终交接使用固定字段：Task、研究版本、branch、git commit、research semantics、changed、protected、verification、
scientific conclusion 和 next step。工程 smoke 只能证明接线或可运行性，不能冒充科研效果。
