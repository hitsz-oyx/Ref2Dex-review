# Ref2Dex 活动记录

- scope: root
- last_updated: 2026-09-02
- current_pointer: [docs/current_versions.yaml](../current_versions.yaml)
- historical_audit: [modification_log.md](modification_log.md)

## 2026-09-02 15:30:36 +0800 — V1.2.15 提交治理与运行入口改动

- activity_id: ACT-20260902-153036
- timestamp: 2026-09-02 15:30:36 +0800
- modification_version: V1.2.15
- type: governance / code / documentation / operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求提交本轮我们共同完成的相关改动；仅提交已确认属于本轮治理和 Cm/CmDecoder 入口的路径
- skills_used: research-change-control, research-experiment-workflow
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: Ref2Dex 治理闭环、递归活动记录、配置/运行追溯、Cm/CmDecoder 研究入口和链接审计；不提交被忽略的 output、cache、数据和 checkpoint

**文件**

- [.agents/](../../.agents/) — 治理与实验 Skill、审计合同和审计脚本。
- [AGENTS.md](../../AGENTS.md) — 仓库常驻规则、任务模式、交接和路径导航合同。
- [docs/](../) — 文档入口、版本指针、目录/版本规范、plan、日志迁移和兼容入口删除。
- [src/base/](../../src/base/) — 配置版本字段、运行 manifest 和终态 summary 追溯支持。
- [src/task/Cm/](../../src/task/Cm/) — Cm 配置、研究入口和局部活动/事实记录迁移。
- [src/task/CmDecoder/](../../src/task/CmDecoder/) — CmDecoder 配置、rollout 入口、实验记录和活动导航。
- [src/task/correspondence_ptv3_v2/](../../src/task/correspondence_ptv3_v2/) — Task 活动/事实记录迁移。
- [tests/](../../tests/) — run manifest 回归和 activity 链接审计测试。

旧中文入口和历史 `status_log.md` 的删除属于上述 `docs/`、Task 日志迁移范围；没有把其他用户未确认的
实验产物、数据、cache、checkpoint 或 output 加入提交。

**原因**

将本轮已经确认的治理、运行追溯、Task 入口和链接导航改动形成可回滚的 Git 提交，保留工作区中不属于
本轮范围的内容不被带入。

**验证**

- 提交前检查 `git status`、完整 unstaged diff 和 staged diff；仅使用显式路径 stage。
- `python3 -m pytest -q tests/test_audit_diff.py`：`5 passed`；审计脚本 `py_compile` 通过。
- `audit_diff.py --staged --check-links`：根级本次提交范围的本地链接均可导航；`git diff --cached --check` 通过。
- 本次为治理/工程提交，不构成科研效果证据；`conclusion: N/A`。

## 2026-09-02 11:43:51 +0800 — V1.2.15 activity Markdown 链接诊断

- activity_id: ACT-20260902-114351
- timestamp: 2026-09-02 11:43:51 +0800
- modification_version: V1.2.15
- type: diagnostic
- change_level: L0
- approval: auto
- approval_basis: 用户请求诊断根 activity 的 Ctrl+Click 导航；本次不修改链接或科研内容
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 只读解析 [docs/logs/activity_log.md](activity_log.md) 的 Markdown 本地链接；不改变代码、配置、数据、产物或运行状态

**文件**

- [docs/logs/activity_log.md](activity_log.md) — 作为被检查的根级 activity 文档。
- [.agents/skills/research-change-control/scripts/audit_diff.py](../../.agents/skills/research-change-control/scripts/audit_diff.py) — 使用既有链接审计器核对最新条目。

**原因**

用户反馈根 activity 中的导航无法 Ctrl+Click，需要区分 Markdown target 错误与 IDE 链接服务问题。

**验证**

- 按文档所在目录解析相对 target；当前文件中的本地链接均指向仓库内已存在文件或目录，未发现失效 target。
- 例如 `../../AGENTS.md` 实际解析为仓库根 [AGENTS.md](../../AGENTS.md)；CmDecoder 的当天产物链接也通过 `--check-links`。
- 结论：Markdown 写法和 target 解析正常；若连 ASCII 链接都不能点击，优先检查 IDE 的 Markdown 插件、项目根打开方式和链接导航设置。
- 本次为环境/编辑器诊断，不构成科研结论；`conclusion: N/A`。

## 2026-09-02 11:32:42 +0800 — V1.2.15 产物导航与链接审计

- activity_id: ACT-20260902-113242
- timestamp: 2026-09-02 11:32:42 +0800
- modification_version: V1.2.15
- type: governance / documentation / tooling
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确批准补充链接规则，并要求 AI 回复中的路径与前后文本留空格
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 最新 activity 的本地链接审计、产物导航合同和 CmDecoder 当日产物链接修正；不涉及模型、配置、数据、GT、坐标、split、checkpoint 内容、训练进程或科研结论

**文件**

- [AGENTS.md](../../AGENTS.md) — 固化仓库相对显示路径、可点击导航、回复路径空格和交接链接校验要求。
- [.agents/skills/research-change-control/SKILL.md](../../.agents/skills/research-change-control/SKILL.md) — 将链接格式和 `--check-links` 纳入通用交接流程。
- [.agents/skills/research-change-control/references/audit-contract.md](../../.agents/skills/research-change-control/references/audit-contract.md) — 说明最新条目选择、链接检查范围和 `PENDING` 例外。
- [.agents/skills/research-change-control/scripts/audit_diff.py](../../.agents/skills/research-change-control/scripts/audit_diff.py) — 按 `timestamp` 选择最新活动并检查本地 Markdown 目标。
- [tests/test_audit_diff.py](../../tests/test_audit_diff.py) — 覆盖乱序活动、有效/失效链接、运行中待生成目标和外部链接。
- [docs/目录规范.md](../目录规范.md) — 定义各类运行必须提供的关键产物导航。
- [docs/ai_task_checklist.md](../ai_task_checklist.md) — 增加交接命令和回复空格检查项。
- [docs/plan/V1.2.15.md](../plan/V1.2.15.md) — 记录用户对链接审计与回复格式的补充批准。
- [src/task/CmDecoder/docs/logs/activity_log.md](../../src/task/CmDecoder/docs/logs/activity_log.md) — 修正当天 rollout 少一级 `..` 的目标，并补全仓库相对产物标签、目录、manifest、summary 和 checkpoint 导航。
- [docs/logs/activity_log.md](activity_log.md) — 记录本次治理事件和验证入口。

**原因**

现有合同虽然要求可点击路径，但缺少自动检查，且部分 CmDecoder 产物只写简称或使用错误的相对层级，导致 IDE 无法跳转。将显示路径与 Markdown 目标分工并加入本地审计，可让用户从 activity 递归进入运行目录和关键证据。

**验证**

- `python3 -m py_compile .agents/skills/research-change-control/scripts/audit_diff.py`：通过。
- `python3 -m pytest -q tests/test_audit_diff.py`：`5 passed`。
- 根 activity 定向 `--worktree --check-links`：覆盖 `8` 个变更路径，`10` 个本地链接均可导航。
- CmDecoder 最新 activity 定向 `--check-links`：`7` 个本地链接均可导航；8 份 rollout manifest/summary JSON 可解析。
- `git diff --check`：通过。
- 工程文档与审计工具修改不构成科研效果证据；`conclusion: N/A`。

**回滚**

仅需回退本条“文件”列出的治理文档、审计脚本、定向测试和 CmDecoder 活动链接；不需要处理任何运行产物或训练状态。

## 2026-09-01 22:53:59 +0800 — V1.2.15 删除中文兼容入口

- activity_id: ACT-20260901-225359
- timestamp: 2026-09-01 22:53:59 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求删除 `docs/版本线与AI行为分类.md` 和 `docs/AI交接清单.md`
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 删除根级中文兼容文档入口并清理现行导航；不涉及代码、数据、模型、GT、坐标、split、checkpoint 或运行产物

**文件**
- `docs/版本线与AI行为分类.md`、`docs/AI交接清单.md` — 删除中文兼容跳转页；canonical 规范分别保留在 `docs/modification_policy.md` 和 `docs/ai_task_checklist.md`。
- `docs/README.md`、`docs/logs/activity_log.md` — 移除现行兼容入口并记录本次删除事件；历史活动中的旧文件名改为非链接历史说明。
- `docs/` — 其他既有根级文档和历史审计记录未做语义迁移。

**原因**
用户确认不再保留两个中文兼容入口，仓库从此只使用 ASCII canonical 文件名。现行导航不再指向已删除路径，历史记录保留文件名以便理解过去的目录状态。

**验证**
- `git diff --check`：通过。
- `docs/README.md`、`AGENTS.md` 和根级活动记录的 Markdown 链接扫描：缺失 `0`。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log docs/logs/activity_log.md --worktree --scope-prefix docs --scope-prefix AGENTS.md`：通过。

<a id="act-20260901-221137"></a>

## 2026-09-01 22:11:37 +0800 — V1.2.15 文档路由与 canonical 命名

- activity_id: ACT-20260901-221137
- timestamp: 2026-09-01 22:11:37 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认补充 AGENTS 文档触发路由，并要求重新命名版本政策和 AI 交接文档
- skills_used: research-change-control
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 仓库文档加载规则、规范权威层级和根级说明文档 canonical 路径；不涉及代码、数据、模型、GT、坐标、split、checkpoint 或运行产物

**文件**
- [`AGENTS.md`](../../AGENTS.md) — 增加规范文档的条件式加载、权威层级和 canonical 路径导航；收窄重复的目录/manifest 说明。
- [`docs/modification_policy.md`](../modification_policy.md)、[`docs/ai_task_checklist.md`](../ai_task_checklist.md) — 使用清晰的 ASCII canonical 文件名并更新职责说明。
- `docs/版本线与AI行为分类.md`、`docs/AI交接清单.md` — 当时的中文兼容入口，已由本次删除操作移除。
- [`docs/README.md`](../README.md)、[`docs/项目总览.md`](../项目总览.md)、[`docs/plan/V1.2.15.md`](../plan/V1.2.15.md)、[`docs/logs/repo_memory.md`](repo_memory.md) — 更新导航、计划和当前事实引用。
- [`docs/`](../) — 本次变更涉及的其他根级治理文档引用保持历史记录可追溯。

**原因**
将“AI 什么时候必须读取哪份规范”写入常驻的 `AGENTS.md`，同时把版本政策、目录规范和交接清单分层，避免普通文档未加载造成规则遗漏，也避免 `AGENTS.md` 与政策正文重复漂移。新路径遵循 ASCII canonical 命名；旧路径保留跳转以兼容历史日志。

**验证**
- `git diff --check`：通过。
- 旧路径引用扫描：仅保留兼容入口和历史审计记录；当前导航与规范均指向 canonical 路径。
- Markdown 相对链接扫描：根级文档和兼容入口均可解析，缺失链接 `0`。
- `python3 .agents/skills/research-change-control/scripts/audit_diff.py --log docs/logs/activity_log.md --worktree --scope-prefix docs --scope-prefix AGENTS.md`：通过，覆盖 `8` 个变更路径。

<a id="act-20260901-211226"></a>

## 2026-09-01 21:12:26 +0800 — V1.2.15 治理闭环切换

- activity_id: ACT-20260901-211226
- timestamp: 2026-09-01 21:12:26 +0800
- modification_version: V1.2.15
- type: governance / documentation
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认“按照你想的来，直接修改吧”，并批准任务模式、current pointer、activity 时间线、链接合同和命名收口方案
- skills_used: research-change-control, research-experiment-workflow
- branch: feature/modular-component-runtime
- base_commit: d9cecd0889087fbe7bff6ba85751e269cee8aac7
- worktree_dirty: true
- scope: 根级治理合同、活动记录迁移、Cm 重点入口和审计工具；不涉及模型、数据、GT、坐标、split、checkpoint 或训练变量

**文件**
- [`AGENTS.md`](../../AGENTS.md) — 增加任务模式、activity 唯一时间线、current pointer、运行反馈和可点击路径合同。
- [`docs/current_versions.yaml`](../current_versions.yaml) — 新增各作用域的 `modification_version` 指针。
- [`docs/plan/V1.2.15.md`](../plan/V1.2.15.md) — 将治理方案定稿并记录已确认决定。
- [`docs/README.md`](../README.md) — 新增 ASCII 文档目录稳定入口，保留中文说明文件兼容路径。
- [`docs/ai_task_checklist.md`](../ai_task_checklist.md)、[`docs/目录规范.md`](../目录规范.md)、[`docs/modification_policy.md`](../modification_policy.md)、[`docs/项目总览.md`](../项目总览.md) — 收口入口、目录、版本和机器记忆边界；中文旧名仅为当时的历史文件名。
- [`docs/logs/repo_memory.md`](repo_memory.md) — 更新根级活动入口。
- [`.agents/skills/research-change-control/SKILL.md`](../../.agents/skills/research-change-control/SKILL.md)、[`.agents/skills/research-change-control/references/audit-contract.md`](../../.agents/skills/research-change-control/references/audit-contract.md)、[`.agents/skills/research-change-control/references/change-levels.md`](../../.agents/skills/research-change-control/references/change-levels.md)、[`.agents/skills/research-change-control/scripts/audit_diff.py`](../../.agents/skills/research-change-control/scripts/audit_diff.py)、[`.agents/skills/research-experiment-workflow/SKILL.md`](../../.agents/skills/research-experiment-workflow/SKILL.md) — 同步任务模式、活动字段、manifest/summary 职责和差异审计。
- [`docs/logs/activity_log.md`](activity_log.md)、[`src/task/Cm/docs/logs/activity_log.md`](../../src/task/Cm/docs/logs/activity_log.md)、[`src/task/CmDecoder/docs/logs/activity_log.md`](../../src/task/CmDecoder/docs/logs/activity_log.md)、[`src/task/correspondence_ptv3_v2/docs/logs/activity_log.md`](../../src/task/correspondence_ptv3_v2/docs/logs/activity_log.md) — 将 `status_log.md` 从切换点移交为 activity 时间线，保留旧历史内容。
- [`docs/logs/machine_memory.md`](machine_memory.md)、[`src/task/Cm/docs/README.md`](../../src/task/Cm/docs/README.md)、[`src/task/Cm/research/README.md`](../../src/task/Cm/research/README.md) — 明确机器事实和 Cm 入口。
- [`src/task/Cm/docs/logs/repo_memory.md`](../../src/task/Cm/docs/logs/repo_memory.md)、[`src/task/Cm/docs/logs/experiment_log.md`](../../src/task/Cm/docs/logs/experiment_log.md)、[`src/task/Cm/docs/logs/decision_log.md`](../../src/task/Cm/docs/logs/decision_log.md) — 更新 Cm 相关导航。
- [`src/task/CmDecoder/docs/logs/repo_memory.md`](../../src/task/CmDecoder/docs/logs/repo_memory.md)、[`src/task/CmDecoder/docs/logs/experiment_log.md`](../../src/task/CmDecoder/docs/logs/experiment_log.md) — 更新 CmDecoder 相关导航。
- [`src/task/correspondence_ptv3_v2/docs/logs/repo_memory.md`](../../src/task/correspondence_ptv3_v2/docs/logs/repo_memory.md) — 更新 correspondence 相关导航。
- [`docs/logs/architecture_log.md`](architecture_log.md)、[`src/task/Cm/docs/logs/architecture_log.md`](../../src/task/Cm/docs/logs/architecture_log.md)、[`src/task/CmDecoder/docs/logs/architecture_log.md`](../../src/task/CmDecoder/docs/logs/architecture_log.md)、[`src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md`](../../src/task/correspondence_ptv3_v2/docs/logs/architecture_log.md) — 将架构记录头部收窄为更新时间。
- [`src/base/base_config.py`](../../src/base/base_config.py)、[`src/base/run_manifest.py`](../../src/base/run_manifest.py)、[`src/base/base_runner.py`](../../src/base/base_runner.py)、[`src/base/__init__.py`](../../src/base/__init__.py) — 将运行版本字段收口到 `modification_version`，并为 train/eval 增加终态 `summary.json`。
- [`src/task/Cm/src/config.py`](../../src/task/Cm/src/config.py)、[`src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml`](../../src/task/Cm/configs/active/grab_inspire_f1_hand_flow_cm64_additive.yaml)、[`src/task/Cm/research/dense_cache_v1_1_1/experiment.yaml`](../../src/task/Cm/research/dense_cache_v1_1_1/experiment.yaml)、[`src/task/Cm/research/hand_flow_decoder/experiment.yaml`](../../src/task/Cm/research/hand_flow_decoder/experiment.yaml)、[`src/task/Cm/research/tsne_slots/experiment.yaml`](../../src/task/Cm/research/tsne_slots/experiment.yaml)、[`src/task/Cm/research/tsne_slots/run.py`](../../src/task/Cm/research/tsne_slots/run.py) — 当前 Cm 配置/实验定义只使用 `modification_version`。
- [`src/task/CmDecoder/config.py`](../../src/task/CmDecoder/config.py)、[`src/task/CmDecoder/current_cm_point_config.py`](../../src/task/CmDecoder/current_cm_point_config.py)、[`src/task/correspondence_ptv3_v2/config.py`](../../src/task/correspondence_ptv3_v2/config.py) — 当前 Task 配置只使用 `modification_version`。
- [`tests/test_run_manifest.py`](../../tests/test_run_manifest.py) — 覆盖旧 guidance/plan/operation 字段不进入新 manifest，以及终态 summary 合同。

**原因**
将任务按只读诊断、已有实验运行、实际变更和治理变更分流；以 activity 记录事件、以 `current_versions.yaml` 提供当前指针，并让每次交接都能跳转到计划、运行和产物。旧 `modification_log.md` 仅作为历史审计，不回写旧条目。

**验证**
- `git diff --check`：通过。
- `python3 -m py_compile .agents/skills/research-change-control/scripts/audit_diff.py src/base/base_config.py src/base/run_manifest.py src/base/base_runner.py`：通过。
- `python3` + PyYAML `docs/current_versions.yaml` 结构校验：通过，只有 `modification_version` 字段。
- `audit_diff.py --worktree`（根级及 Cm/CmDecoder/correspondence 三个作用域）：通过，分别覆盖 `39/12/5/3` 个变更路径。
- 变更文档 Markdown 相对链接扫描：`14 files; missing 0`。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q`：`311 passed, 3 skipped`。
- 系统 `python3` 的全量收集因环境缺少 `smplx`/`viser` 失败；已切换项目 `graspenv` 完成完整回归，属于环境差异而非仓库测试回归。
- 工程验证不代表任何科研结论：`conclusion: N/A`。

## V1.2.13 — 删除根 Component 目录（2026-09-01）

- category: `governance`、`operation`、`documentation`
- 状态: 根 `components/` 已彻底删除；仓库不再提供默认 manifest 根或通用示例，Task 继续直接使用各自 Python 实现。
- 兼容边界: `src/base/` 的通用协议代码和 `tools/researchctl.py` 暂时保留；工具只接受显式传入的外部 manifest 根，不再猜测仓库路径。
- 保护边界: 未修改 Task 训练配置、数据、cache、checkpoint、output 或运行进程。
- 验证: 根目录/活动引用扫描无残留；定向测试 `18 passed`；全量 pytest `310 passed, 3 skipped`；`git diff --check` 通过。

## V1.2.11 — Task-local 外部资产入口（2026-08-31）

- category: `governance`、`operation`
- 状态: 根 `assets/` 不再承载新资产；Cm 的外部资产入口下沉到 `src/task/Cm/assets/`，根目录只保留被忽略的兼容软链接。
- 保护边界: 未移动大型 checkpoint、数据、cache 或运行中的进程；DenseToken 真实文件仍在旧 Cm 目录。
- 验证: 新旧 DenseToken 路径解析到同一文件，Cm 配置导入和全量 pytest 通过。

## V1.2.12 — 撤出两个 Task 的 Component、data 和 registry 入口（2026-09-01）

- category: `governance`、`operation`、`documentation`
- 状态: 已删除 CmDecoder 与 correspondence_ptv3_v2 的 Task-local `components/`、`data/`、`registry/` 及根级兼容软链接；两个配置恢复为直接选择各自 Python Task 实现，并继续使用根级 `data/`、`dataset/`、`third_party/`。
- 保护边界: 未移动或删除真实数据、cache、checkpoint、output 或运行进程；历史日志和用户指导保留。
- 文档: 当前目录规范、交接清单和根架构入口不再把 Task Component/registry 作为现行结构；历史日志条目不改写。
- 验证: Task 目录入口扫描、配置导入、定向测试、全量 pytest 与 `git diff --check`。

## V1.2.10 — 取消 Cm 路径索引（2026-08-31）

- category: `governance`、`data`、`documentation`
- 状态: Cm 不再维护 Task 内路径 registry 或便捷数据软链接；配置直接声明外部数据、cache 和资产路径，实体数据未移动。
- 验证: Cm 配置/loader 回归和全量 pytest 通过。

## V1.2.9 — 撤销隔离试验线（2026-08-31）

- category: `governance`、`code`、`documentation`
- 状态: 已删除 Cm 隔离试验目录、Cm Task 的清单/适配器入口，并清理活动文档中的失效引用；数据、cache、checkpoint、output、outputs 和既有训练实现保留。
- 规则: AGENTS.md 与保留 Skill 已恢复为不含该试验线的通用维护规则；后续历史条目仍保留原始记录，避免重写审计历史。
- 验证: AGENTS/Skill 文本扫描无相关术语；Cm 配置和 loader 定向回归通过。

## V1.2.8 — 声明层与运行层闭合（2026-08-31）

- category: `architecture`、`code`、`governance`、`experiment`、`documentation`
- 状态: 已完成；`check-task-config` 强制核验 Task config↔`components.json`↔manifest，CmComponent 启动核验 Python Pipeline 与 `pipeline.yaml` 拓扑，YAML 保持声明/启动断言定位。
- 运行时: `TrainableComponent` 提供 build、参数组前缀和 `id@version` checkpoint 命名空间；首个 batch 做 materialized Tensor shape/dtype/finite/batch 对齐检查，后续步骤走轻量 schema 路径。
- 清单修正: Cm registry 补登记 `ref2dex.correspondence.ptv3_v2@2.1.0/dense_encoder` 并与旧配置的 `cm_task_runner` role 对齐。
- 验证: 全量 pytest `329 passed, 3 skipped, 22 warnings`；两种坐标配置 CPU max_steps=2 train/eval 通过；证据为 `outputs/cmcomponent/cm_component_smoke_20260831_172705/` 与 `outputs/cmcomponent/cm_component_smoke_object_pose_20260831_172708/`；两个最新 run manifest 均记录 `V1.2.8`、递归 component_tree 和 parameter_prefix。
- 结论: `SUPPORTED: declaration/runtime audit and trainable lifecycle smoke`；仅为工程 smoke，不构成真实数据或坐标策略科研结论。用户指导与架构快照未修改。

## V1.2.7 — Component 晋升与内部模块边界（2026-08-31）

- category: `governance`、`architecture`、`documentation`
- 状态: 已完成；明确 `helper`、Task-local module、可发现 Component 三层边界，补充小开关的适用条件和必须晋升 Component 的语义/生命周期触发条件。
- 规则: Component 不是文件拆分单位；内部可拆多个类/模块而不增加 manifest。只有出现独立替换、版本、资源、生命周期、跨 Task 复用、并行回滚或公共 Contract/数据语义边界时才建立 Component。
- 保护: 用户指导、架构快照、运行代码、数据/cache/checkpoint/output 和运行中的旧 Cm 进程未修改。
- 结论: `SUPPORTED: explicit component-promotion boundary and anti-over-splitting rule`；本轮无科研实验结论。

## V1.2.6 — 根节点唯一声明与递归配置解耦（2026-08-31）

- category: `governance`、`architecture`、`code`、`experiment`、`documentation`
- 状态: 已完成；CmComponent 配置只声明 `coordinate_transform` 与 `pipeline` 两个根 Component，encoder/head 仅在 `pipeline.children` 中选择；外层 Pipeline 通过父 role 解析子组件。
- 追溯: `component_tree` 成为嵌套选择的规范视图，`run_manifest.components` 由树自动生成 child-first 扁平兼容视图，配置不再重复登记子组件。
- 验证: 两种坐标配置均完成 CPU `max_steps=2` smoke train/eval；定向 pytest `19 passed`；全量测试 `325 passed, 3 skipped`；最新证据为 `outputs/cmcomponent/cm_component_smoke_20260831_143911/` 与 `outputs/cmcomponent/cm_component_smoke_object_pose_20260831_143914/`。
- 结论: `SUPPORTED: nested component selection, stable outer Pipeline wiring, and provenance`; smoke 不构成真实数据效果结论。

## V1.2.5 — 递归 Component 选择与运行追溯（2026-08-31）

- category: `governance`、`architecture`、`code`、`experiment`、`documentation`
- 状态: 已完成；`children` 选择、父 role 解析和递归 `component_tree` 已接入 CmComponent，未更新用户指导或架构快照。
- 验证: 两种坐标配置均完成 CPU `max_steps=2` smoke train/eval；CmComponent + run manifest 定向测试 `18 passed`；全量测试 `324 passed, 3 skipped`。
- 证据: `outputs/cmcomponent/cm_component_smoke_20260831_131246/` 与 `outputs/cmcomponent/cm_component_smoke_object_pose_20260831_131249/`。
- 结论: `SUPPORTED: recursive component selection/wiring and provenance`；不构成真实数据效果结论。

## V1.2.1 — 版本线与 Task Component 治理收口（2026-08-30）

- category: `governance`、`architecture`、`documentation`
- 状态: 已完成本轮目录、清单、运行追溯和交接规则改造；没有修改用户指导文档，也没有创建或覆盖架构快照。
- 版本关系: `V1.2` 为已定稿执行计划；`V1.2.1` 为本次细分操作。后续并行实验沿用同一 `V1.2` plan，并分配新的 `V1.2.k`。
- Component: Task 专属 manifest 已迁移至 `src/task/<Task>/components/`，旧 `components/ref2dex/*` 保留相对软链接；各 Task 新增 `components.json` 清单。
- 数据路径: 各 Task 新增 `registry/data_paths.json` 和被忽略的 `external_paths.json` 覆盖入口，Task 内 `data/` 仅保留相对软链接，不复制外部数据。
- 运行追溯: 新运行的 `run_manifest.json` 记录版本线、plan/guide、细分操作、操作类别、Component registry 和解析后的 Component 清单；历史 manifest 不回写。
- 架构规则: `architecture_log.md` 继续作为过渡导航和历史事实；只有用户明确发出架构更新指令时才建立/更新 `docs/architecture/Vn.m.md`。
- 下一步: 运行中的 Cm 训练完成后，按用户指令决定是否创建下一小版本或架构快照；普通状态刷新不改变 `k`。

## V1.2.2 — CmComponent 隔离 Task 建立（2026-08-30）

- category: `architecture`、`code`、`governance`
- 状态: 已建立独立的 `src/task/CmComponent/`，作为 Component Pipeline 的实验边界；旧 Cm、数据/cache、checkpoint 和运行进程保持不变。
- 当前计划: `src/task/CmComponent/docs/plan/V1.2.md`（final）；本次操作使用 `V1.2.2`，不创建新的大版本。
- 组件: Dense encoder 和 Cm head 已实现为真实 Component，Pipeline/Runner 负责显式组合；当前采用 adapter-first，下一步先做固定 batch parity。
- 风险: 新 Task 仍复用旧 Cm 的 dataset/loss/metric 生命周期，若要进一步拆分 head 内部，必须继续在新 Task 内进行并补充兼容测试。

## V1.2.3 — CmComponent 独立插拔式训练 smoke（2026-08-31）

- category: `architecture`、`code`、`experiment`
- 状态: 已完成独立 Component、Task-local synthetic 数据契约和 BaseRunner 训练/评估闭环；定向测试 `9 passed`。
- 组件: `PointFeatureEncoderComponent`、`SlotFlowHeadComponent`、`CmComponentPipeline` 均为独立 `Component`，清单版本统一为 `2.0.0`；Pipeline 拓扑为 `dense_encoder -> cm_head`。
- 训练证据: CPU `max_steps=2` smoke 已生成 `outputs/cmcomponent/cm_component_smoke_20260831_111831/`，包含 run manifest、配置、metadata、metrics 和 latest/best checkpoint；使用 latest checkpoint 的 test eval 已通过。
- 兼容边界: CmComponent Python/config 不再导入或继承旧 `src/task/Cm`；新 checkpoint 不承诺旧 Cm checkpoint 兼容。旧 Cm 代码、数据、cache、checkpoint、output 和训练进程未改动。
- 结论: `SUPPORTED: wiring/training lifecycle`；smoke 不代表真实数据上的科学效果。下一步需另建小版本 plan 接入真实数据或进行并行组件实验。

## V1.2.4 — 坐标策略 Component 化（2026-08-31）

- category: `governance`、`architecture`、`code`、`data`、`experiment`
- 状态: 已完成；`hand_root_t` 与 `object_pose_t` 已成为同一 `coordinate_transform` 角色下的两个独立 Component。
- 规则: 已将“出现第二种候选实现即提升为 Component”“每个运行 role 恰好选择一个实现”“可发现 manifest 统一使用 `component.yaml`”“新增/替换组件必须有清单与合同测试”“耦合处只依赖稳定接口”“YAML/JSON 只选择不执行”和“完成后报告规范限制”写入 `AGENTS.md`。
- 验证: 两种配置均完成 CPU smoke train/eval；CmComponent 定向测试 `14 passed`；全量测试 `323 passed, 3 skipped`；最新两个 run manifest 均记录坐标 Component、版本、pose source、source/output frame 和完整 Component 类别。
- 结论: `SUPPORTED: coordinate-component wiring/training lifecycle`；不构成真实数据效果比较。旧 Cm 代码、数据/cache/checkpoint/output 和进程未修改。

## 当前状态

- 当前阶段 / 指导: 已以 `oyx` 为主完成 `feature/hand-pca-perturbation` 的内容整合、递归 repo/machine memory 重构和全量回归验证。
- 当前进行中: Cm 混合 GRAB+Inspire-F1 手流重建当前运行于 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260830_185013`（torchrun world size=3）；旧 epoch 29 / step `124410` 仍作为历史恢复锚点。CmDecoder 三卡 EXP-022 运行于 GPU0/1/2，当前约 step `86500` / epoch 19，validation best=`1.4249 mm`（epoch 15），预计剩余约 `5.5 h`；该 best 已在 held-out test episode 完成 EXP-025 rollout，point EPE=`6.896/11.415 mm`（mean/final），短程反馈泛化尚可但仍有累积误差。Cm 已完成破坏性目录迁移：真实实现位于 `src/`、`dataset/`、`visualization/`，配置位于 `configs/{active,archive}`，旧顶层 Python/YAML 入口已移除。
- 最近可靠结论: feature 分支的 HOCap subject_1 Stage 3 已生成 28 个文件、23,896 个 frame samples并通过 loader 核验；Cm mixed 两条 run 已分别保留 epoch 48/40 的最近完整 checkpoint，详细指标见 Cm 状态与实验日志。
- 阻塞 / 风险: ARCTIC 全量 Stage 2/3 的最终产物状态仍需独立核验；HOCap 仅是外部测试集。合并代码已通过全量测试，但尚未在真实全量外部数据上重新导出。
- 下一步: 等待 CmDecoder GRAB→ARCTIC MANO 泛化结果，同时等待 Cm C=32/C=64 完成可比的中长程 validation，再决定是否恢复 correspondence 训练。
- 证据与相关文档: 全量 pytest `311 passed, 3 skipped`；[历史修改记录](modification_log.md)、[架构记录](architecture_log.md)、[correspondence 活动](../../src/task/correspondence_ptv3_v2/docs/logs/activity_log.md)、[Cm 活动](../../src/task/Cm/docs/logs/activity_log.md)。
- 2026-08-30 框架更新: `BaseRunner` 在 train/eval 启动时生成不会覆盖历史尝试的 `run_manifest.json`（续跑/评估使用时间戳文件）；`ref2dex.run.v1` 记录 Git 状态、输入 manifest/cache 引用、文件基本信息、数据合同、seed 和初始 checkpoint，不计算加密 hash。Cm loader 对坐标根做全量一致性检查。

## 2026-08-30 目录规范兼容迁移

- Cm 的 t-SNE 诊断已迁入 `src/task/Cm/research/tsne_slots/`；旧 `tsne_slots.py` 和模块调用方式保留为软链接/兼容入口。
- 新增统一的研究实验包格式、`experiment.yaml`、研究输出规则、Manifest 职责说明和 AI 交接清单。
- 新增 `assets/checkpoints/densetoken` 过渡软链接，实际 checkpoint 文件仍保留在旧目录，未影响当前运行。
- 本次没有移动或删除 `data/`、`output/`、`outputs/`、checkpoint 或历史 `results/`；根 `output/` 仅标记为历史兼容路径。

## 2026-09-03 22:15:00 +0800 — 合并前迁移本地状态快照

- activity_id: ACT-20260903-221500-ROOT-MERGE
- type: merge / documentation / operation
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求以远端 `oyx` 为主合并，并将本地 status 内容迁移到 activity 记录
- branch: oyx
- scope: root activity；对应的本地 `status_log.md` 已按远端治理结构移除

**本地状态摘要**

- 本地此前已完成 correspondence object-centered 七域训练准备：OakInk2 Stage3 与 object-disjoint train/val 划分完成，ARCTIC/HRDexDB 四手型 object-frame cache 完成。
- 七域训练已在 GPU 2/3 从头运行至约 step 143,700，最近 checkpoint 为 step 140,000；data-wait ratio 约 `0.00027`，无 traceback/OOM。
- 首次训练曾因 DataLoader worker 的 `dataset.HRDexDB` 导入冲突退出，随后已修复并重新启动。
- OakInk2 train split 约 672.97 万帧，GRAB 约 32.78 万帧；equal-domain sampler 下两域按 batch 等权曝光。

**验证**

- 合并预演确认远端与本地共有 9 个日志冲突路径；本地 status 内容已转入本 activity 与 correspondence task activity。
