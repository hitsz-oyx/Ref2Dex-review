# Ref2Dex 工作版本与操作规则

本文件定义治理、研究版本、Git 提交和运行身份的职责。它不回写历史记录，也不把每一次 bug fix、训练启动或失败
包装成新的研究版本。

## 一、三个身份

    research_version  → 论文 Methods 中描述的研究方法
    git_commit        →  实际执行的代码和配置
    run_id            →  一次独立运行

仓库治理另有 `governance_version`，只记录治理合同的当前版本。当前指针见
`current_versions.yaml`。历史 modification_version、operation_version、guide_version 和 plan_version 字段只作
兼容审计，不再作为新记录的版本身份。

`work_version` 可作为同一 research version 下的已确认工作单元编号（例如 `V1.18.4`）；它是 Activity、branch 或
PR 的辅助追溯字段，不替代 research_version，也不进入 current_versions。

## 二、research_version 的递增条件

以下变化不创建新的 research version：bug fix、显存或性能优化、测试、launcher 和日志修复、训练启动或失败、
seed、GPU、预算调整以及 checkpoint 选择。

只有论文 Methods 需要改变时才递增，例如 observation、action semantics、reward、planner formulation、GT、
数据 split、核心训练目标或 checkpoint 解释。研究版本改变时，由用户提供或确认新的指导，Agent 起草同版本的最终 plan。

## 三、工作模式和影响等级

AI 只使用四种工作模式：

- inspect：只读阅读、扫描和诊断。
- change：修改代码、配置、测试或普通文档。
- run：训练、评估、benchmark、数据处理和长任务；代码和研究变量必须先固定。
- governance：修改 AGENTS、Skill、公共规范、目录或版本合同。

| 等级 | 范围 | 审批 |
| --- | --- | --- |
| L0 | 文档、格式、测试、只读诊断和不改变运行变量的元数据 | 可执行，完成后汇报 |
| L1 | Task 内实现，研究合同不变 | 可执行，补定向验证 |
| L2 | 研究语义或公共接口合同变化 | 编辑前用户确认 |
| L3 | 治理、共享 `src/base`、依赖、迁移、破坏性操作或长任务 | 编辑前用户确认 |

修改 AGENTS、Skill 或本文件统一按 L3 处理。审批必须记录 approval 和 approval_basis，不能把用户沉默当作批准。

## 四、plan、指导和当前状态

- `src/task/<Task>/docs/README.md` 是 Task 当前状态页，维护 research version、研究问题、invariant、blocker、证据
  入口和重要 entrypoint。
- `指导/V<n>.md` 只写研究意图、假设、边界、成功标准和禁止事项，由用户维护。同一数值基线的确认细化使用
  `指导/V<n><letter>.md`，后缀限单个小写 `a`–`z`；必须从无后缀基线读至最高确认后缀，不能只读最后一个文件。
  跨文件的明确冲突以较后字母为准，非冲突部分共同生效。单一指导文件内部冲突、无法同时满足或语义不充分时，Agent
  必须暂停，向用户列出冲突和影响，并先在 plan 草案记录候选解释、拟定措辞、保护项、风险、验证与回滚；用户确认后
  才可更新指导、定稿 plan 并继续。
- 只有新 research version 或 L2/L3 方案需要 `plan/V<n>.md`；L0/L1 修复不创建新 plan。
- `architecture/V<n>.md` 是按需冻结的架构 snapshot；普通修复不更新。
- 正式实验使用独立 experiment card，记录 hypothesis、configuration、runs、evidence、conclusion 和 limitations；
  实验参数变化不改全局架构，也不创建 Git branch。

## 五、记录和运行合同

当前作用域 `activities/` 下的独立 Activity 记录重要实现、正式指导修订、重要诊断事实、重要操作、版本切换和治理
事件；`activities/README.md` 只做索引。Activity 至少包含 timestamp、activity_id 或 work_version、governance_version
或 research_version、git_commit、branch、scope、approval 和 verification。命令、单次测试、轮询、逐 step/checkpoint
进度和普通 smoke 不建立 Activity。旧 `logs/activity_log.md` 只读保留。

运行条目另含 run_id、run_status、命令、输出入口、最后 step/epoch、best metric、关键 checkpoint 和实际生成的
metrics.jsonl/train.log。使用：

    run_status: STARTED | RUNNING | COMPLETED | FAILED | STOPPED | UNKNOWN
    conclusion: SUPPORTED | REFUTED | INCONCLUSIVE | INVALID_IMPLEMENTATION | N/A

正式实验的科学证据写入 `experiments/` 下的独立 card，一个 card 可以关联多个 run；Activity 不替代 experiment card。
正式训练、评估、benchmark 和数据处理必须产生小型 run_manifest.json，至少引用 Task、research_version、git commit、
配置、输入 manifest、schema、坐标/单位、seed、初始 checkpoint 和输出目录。不得把完整数据合同复制进 manifest，
不得提交 cache、checkpoint、原始数据或大型输出。

## 六、Git 和历史兼容

`oyx` 是稳定集成分支。独立任务使用 `ai/<task>/<description>` 分支，经验证和交接后合并。不得 reset、覆盖或带入
用户已有改动；只显式提交当前任务文件。

旧 modification log、旧 activity、experiment log、plan、checkpoint、cache 和输出不删除、不回写，仍可作为只读审计
证据。迁移从治理切换点开始；新规则不要求一次性整理所有 Task。

## 七、验证门禁

所有修改完成后统一执行：

    python tools/verify.py --changed

该工具在第二阶段建立。工具存在前，Agent 必须记录实际定向测试、编译/导入检查、Markdown 链接审计和 git diff --check，
不得把临时检查称为 VERIFY PASS。CI 在工具建立后复用同一入口。

## 八、规范反馈

若规则迫使 Agent 重复记录、无法审计或需要兼容 workaround，完成汇报单列“规范反馈”，说明场景、风险、建议规则、
兼容影响和是否需要用户确认。未经用户确认不得擅自修改 AGENTS、Skill 或公共合同。
