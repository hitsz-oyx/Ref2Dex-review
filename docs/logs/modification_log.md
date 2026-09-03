# 全局 AI 修改记录

## 2026-09-01 — V1.2.14 重组 AGENTS 规范分层

- change_level: L3（仓库治理文档结构调整）
- approval: user-approved（用户明确要求将通用规则与仓库特有约定分离）
- skills_used: `research-change-control`
- branch: `feature/modular-component-runtime`
- version: `V1.2.14`（根级治理细分操作；不涉及 Task plan）
- category: `governance`、`documentation`
- post-commit: 本次提交；代码、配置、数据、cache、checkpoint、output 和运行进程未修改
- scope: `AGENTS.md` 章节组织及根治理日志

**文件**

- `AGENTS.md` — 第 5 节改为通用交接/版本规范（`5.1`、`5.2`）；原路径、运行和目录路由内容移至文档末尾第 8 节（`8.1`）。
- `docs/logs/status_log.md`、`docs/logs/repo_memory.md`、`docs/logs/decision_log.md`、`docs/logs/modification_log.md` — 记录 V1.2.14 的结构调整和保护边界。

**原因**

用户指出第 5 节混合了通用 AI 行为规则与 Ref2Dex 特有约定。分层后，通用交接和版本规则可以独立复用，仓库路径规则集中在最后，阅读边界更清晰。

**验证**

- `AGENTS.md` 章节扫描确认第 5 节仅含通用交接/版本规则，第 8 节承载仓库特有路径与运行约定。
- `git diff --check` 通过；未运行训练或修改生成产物。

## 2026-09-01 — V1.2.13 删除根 Component 目录

- change_level: L3（破坏性根目录治理）
- approval: user-approved（用户明确要求删除根里的 Component）
- skills_used: `research-change-control`
- branch: `feature/modular-component-runtime`
- version: `V1.2.13`（根级治理细分操作；不涉及 Task plan）
- category: `governance`、`operation`、`documentation`
- post-commit: 未提交；未 stage；数据、cache、checkpoint、output 和运行进程未修改
- scope: 根 `components/`、其默认发现入口、当前目录/架构说明和相关协议测试

**文件**

- `components/` — 删除 README、identity/scale 示例、pipeline 示例、此前已撤出的 Ref2Dex 兼容入口及未跟踪缓存，最终目录本身不存在。
- `tools/researchctl.py` — 移除默认 `components` root；通用校验必须显式传入 manifest 路径。
- `tests/test_component_registry.py`、`tests/test_framework_contracts.py` — 不再加载仓库示例；使用临时 manifest 校验基础协议，并断言根目录保持缺失。
- `AGENTS.md`、`docs/目录规范.md`、`docs/logs/` — 删除当前规则中的公共组件入口，并在根状态、架构、决策、记忆和修改日志记录 V1.2.13 边界。

**原因**

用户要求在已撤出各 Task 组件入口之后继续删除根组件目录。根示例已无当前消费者，删除可消除虚假的默认发现入口；保留 `src/base/` 兼容代码避免把本次目录治理扩大为未经确认的共享基础设施删除。

**验证**

- 根 `components/` 物理目录不存在，活动规则/工具/测试不再引用 `components/examples` 或 `components/ref2dex`。
- 定向 pytest：`18 passed`；`researchctl list` 在无显式 root 时返回空清单，两个已清理 Task 的 config check 均为 `registry=none components=0`。
- 全量 pytest：`310 passed, 3 skipped, 22 warnings`；`git diff --check` 通过。

## 2026-09-01 — V1.2.12 撤出 CmDecoder 与 correspondence 的 Task Component/data/registry

- change_level: L3（破坏性目录治理与数据入口迁移）
- approval: user-approved（用户确认彻底删除 Component、Task data/registry，并将数据归回根空间）
- skills_used: `research-change-control`
- branch: `feature/modular-component-runtime`
- version: `V1.2.12`（两个 Task 的 `docs/plan/V1.md` 均 final）
- category: `governance`、`operation`、`documentation`
- post-commit: 未提交；真实数据、cache、checkpoint、output 和运行进程未移动或删除
- scope: `src/task/CmDecoder/`、`src/task/correspondence_ptv3_v2/` 的 Task-local 入口、根组件兼容入口、当前目录文档

**文件**

- `src/task/CmDecoder/components/`、`data/`、`registry/` — 删除此前新增的 Task-local Component 清单、数据软链接和路径 registry。
- `src/task/correspondence_ptv3_v2/components/`、`data/`、`registry/` — 同上。
- `components/ref2dex/cmdecoder_pointflow`、`components/ref2dex/correspondence_ptv3_v2` — 删除根级兼容软链接。
- 两个 Task 的 `config.py` — 移除 `component_registry`/`components` 选择，版本更新为 `V1.2.12`。
- `docs/目录规范.md`、`docs/AI交接清单.md`、`docs/版本线与AI行为分类.md`、`components/README.md`、根架构日志 — 删除当前有效的 Task Component/registry 说明，保留历史日志事实。
- 两个 Task 的 V1 plan、状态/架构/记忆/修改日志及根状态/决策/记忆/架构日志 — 记录边界、保护范围和回滚方式。
- `tests/test_component_registry.py`、`tests/test_framework_contracts.py` — 将任务 manifest 测试改为确认已撤出，不再要求加载两个 Task 的入口。
- `.gitignore` — 删除 Task registry 覆盖文件的过时忽略规则。

**原因**

用户明确要求两个 Task 的 Component 彻底删除，Task 内 data/registry 也撤出并统一使用根级数据空间。实际数据本来就位于根级目录，因此只删除入口和声明，不做数据迁移。

**验证**

- 两个 Task 与根组件入口扫描无残留；配置导入成功。
- 全量 pytest、定向组件/框架测试和 `git diff --check` 通过。

## 2026-08-31 — V1.2.11 将 Cm 外部资产入口下沉到 Task

- change_level: L3（仓库路径治理与运行中资产兼容迁移）
- approval: user-approved（用户要求将根 `assets/` 下放到具体 Task 并加入 `.gitignore`）
- skills_used: `research-change-control`
- branch: `feature/modular-component-runtime`
- version: `V1.2.11`（沿用 `src/task/Cm/docs/plan/V1.2.md`）
- category: `governance`、`operation`
- post-commit: 未提交；大型 checkpoint、数据、cache 和运行进程未移动或停止
- scope: Cm Task 资产入口、根兼容软链接、全局目录规则

**文件**

- `src/task/Cm/assets/README.md`、`src/task/Cm/assets/checkpoints/densetoken` — 建立 Cm Task-local 资产入口和旧 checkpoint 软链接。
- `assets` — 改为指向 Cm 资产入口的被忽略兼容软链接。
- `src/task/Cm/src/config.py` — DenseToken 默认路径改用 Task-local canonical 入口。
- `.gitignore`、`AGENTS.md`、`docs/目录规范.md`、`docs/项目总览.md`、`docs/版本线与AI行为分类.md` — 将 Task-local 资产设为规范位置并忽略实际文件。
- `src/task/Cm/docs/README.md`、`src/task/Cm/docs/plan/V1.2.md` 及 Cm/root 状态、架构、记忆和修改日志 — 同步迁移事实、版本和回滚边界。

**原因**

根 `assets/` 当前只承载 Cm 的 DenseToken 入口。资产归属下沉到具体 Task 后，Task 配置、资产和研究边界一致；根兼容软链接保留旧路径，避免正在运行的 Cm 任务和历史命令受到影响。

**验证**

- `readlink -f src/task/Cm/assets/checkpoints/densetoken/best.pt` 与 `readlink -f assets/checkpoints/densetoken/best.pt` 指向同一 checkpoint。
- Cm 配置导入、相关 loader 回归和全量 pytest：`311 passed, 3 skipped, 22 warnings`。
- `git diff --check` 通过；未移动大型 checkpoint，当前 Cm 训练进程仍存在。

## 2026-08-31 — V1.2.10 取消 Cm 路径索引

- change_level: L2（Task 数据入口与目录治理）
- approval: user-approved（用户明确决定不再增加路径索引）
- branch: `feature/modular-component-runtime`
- post-commit: 未提交；外部数据、cache、资产和 checkpoint 未移动或删除
- scope: Cm Task 与全局路径规则说明

**文件**

- `src/task/Cm/registry/`、`src/task/Cm/data/` — 删除 Task 内路径 registry 和便捷软链接。
- `AGENTS.md`、Cm 计划/README/状态 — 改为由配置直接声明外部路径，不再要求路径索引。

**验证**

- Cm 配置/loader 回归和全量 pytest（`311 passed, 3 skipped, 22 warnings`）通过。
- `git diff --check` 通过；外部数据、cache、资产和 checkpoint 未触碰。

## 2026-08-31 — V1.2.9 撤销隔离试验线

- change_level: L3（破坏性目录回退、仓库治理）
- approval: user-approved（用户明确要求停止该试验线并直接删除）
- branch: `feature/modular-component-runtime`
- post-commit: 未提交；用户既有的训练、数据、cache、checkpoint 和输出未覆盖
- scope: 全局维护规范、Cm Task 和已删除的隔离目录

**文件**

- `src/task/CmComponent/`、`.agents/skills/modular-component-runtime/` — 删除隔离试验代码和专用维护 Skill。
- `AGENTS.md`、保留 Skill — 移除该试验线规则，保留通用版本、审批、日志和实验流程。
- `src/task/Cm/`、`docs/项目总览.md`、`components/README.md`、相关测试 — 移除 Cm 的清单/适配器入口和失效引用，保留数据/训练迁移。

**原因**

按用户指令停止隔离试验，避免旧入口和活动文档继续把它当作当前架构。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q`：`311 passed, 3 skipped, 22 warnings`。
- `git diff --check` 通过；AGENTS.md 与 `.agents/skills/` 文本扫描无相关术语；Cm 配置导入通过。

## 2026-08-31 — V1.2.8 声明层与运行层闭合

- change_level: L2（Contract、Pipeline 启动校验和训练生命周期）+ L3（Task registry/CLI/治理文档）
- approval: user-approved（用户明确要求按声明层/运行层审计、TrainableComponent 和首批 Tensor 合同继续改造）
- skills_used: `research-change-control`、`modular-component-runtime`、`research-experiment-workflow`
- branch: `feature/modular-component-runtime`
- version: `V1.2.8`（沿用 final plan `V1.2`）
- category: `architecture`、`code`、`governance`、`experiment`、`documentation`
- post-commit: 未提交；未停止或修改运行中的旧 Cm 训练进程
- scope: `src/base` Component/Contract/registry、`tools/researchctl.py`、CmComponent 训练运行时与 Task 清单

**文件与变更**

- `src/base/component.py`、`src/base/__init__.py` — 增加 `TrainableComponent` 的 build、参数组前缀、`id@version` checkpoint namespace 和兼容加载协议。
- `src/base/contract.py` — 对 materialized tensor-like Artifact 检查声明 shape/dtype；保留 path-only Artifact 的 metadata 合同路径。
- `src/base/registry.py`、`tools/researchctl.py` — 增加只读 `check_task_config` 和 `check-task-config` 命令，核验 config、Task `components.json`、manifest 与 entrypoint_kind。
- `src/task/CmComponent/src/component_factory.py`、`src/task/CmComponent/src/pipeline.py`、`src/task/CmComponent/src/runner.py` — 构造阶段强制 registry 闸门和 pipeline.yaml 拓扑一致性；Pipeline/Runner 使用 TrainableComponent.build；首个 batch 严格合同检查。
- `src/task/CmComponent/dataset/contracts.py` — 增加首批 dtype、finite、batch 对齐检查，后续支持 fast path。
- `src/task/CmComponent/components/{dense_encoder,cm_head,pipeline}/*` — 实现 build 协议并在 manifest 声明 TrainableComponent 要求。
- `src/task/Cm/components/components.json` — 补登记旧配置使用的 `ref2dex.correspondence.ptv3_v2@2.1.0/dense_encoder`，并对齐 runner role。
- `AGENTS.md`、`components/README.md`、组件/Task 文档和测试 — 固化四个事实源职责、YAML 启动断言边界、Trainable 生命周期和 Contract 首检规则；补充回归测试。

**验证**

- `PYTHONPATH=. .../graspenv/bin/python -m pytest -q`：`329 passed, 3 skipped, 22 warnings`。
- 两种坐标配置 CPU `max_steps=2` train/eval：hand_root test EPE `7.41653 mm`，object_pose test EPE `7.33270 mm`；这只是工程 smoke。
- `researchctl check-task-config`：CmComponent smoke 与旧 Cm config 均通过；Pipeline `check --resolve-entrypoints` 通过。
- 最新 run manifest：`outputs/cmcomponent/cm_component_smoke_20260831_172705/run_manifest.json`、`outputs/cmcomponent/cm_component_smoke_object_pose_20260831_172708/run_manifest.json`。
- 用户指导、架构快照、旧 Cm 模型/数据/cache、历史产物和训练进程未被改写；未 stage/commit。

## 2026-08-31 — V1.2.7 明确 Component 晋升与内部模块边界

- change_level: L3（仓库治理与 Component 组织规范）
- approval: user-approved（用户明确提出需区分小开关、内部模块和可插拔 Component，并要求保持解耦规范）
- skills_used: `research-change-control`、`modular-component-runtime`
- branch: `feature/modular-component-runtime`
- version: `V1.2.7`（沿用父计划 `V1.2`）
- category: `governance`、`architecture`、`documentation`
- post-commit: 未提交；未停止或修改运行中的旧 Cm 训练进程
- scope: `AGENTS.md` 的 Component 晋升、文件粒度、开关和 Pipeline 同步边界

**文件**

- `AGENTS.md` — 增加 helper / Task-local module / 可发现 Component 三层边界；规定小开关必须保持完整 Contract、资源和生命周期不变；区分并存候选与单一路线 Contract/schema 迁移；明确 Component 不是一文件一组件，父 Contract 稳定时内部拆分不要求外层 Pipeline 同步。
- `docs/项目总览.md`、`src/task/CmComponent/docs/README.md`、根/Task 状态、决策、修改和记忆日志 — 记录 V1.2.7 的治理决策、版本关系和当前有效规则。

**原因**

现有规则已说明何时晋升 Component，但没有明确告诉 Agent 何时只需内部模块或小开关，可能导致过度拆分文件和 manifest。补充判定层次，使代码保持解耦而不产生不必要的运行时组件。

**验证**

- `git diff --check` 通过。
- change-control 局部审计通过；本轮无运行代码或实验变更，未重复执行训练测试。
- 用户指导、架构快照、旧 Cm、数据/cache/checkpoint/output 和运行进程保持不变。

## 2026-08-31 — V1.2.6 根节点唯一声明与递归配置解耦

- change_level: L2（Component 选择与运行追溯合同）+ L3（Task 配置、规范和文档）
- approval: user-approved（用户明确回复“可以，你直接修改吧”）
- skills_used: `research-change-control`、`modular-component-runtime`、`research-experiment-workflow`
- branch: `feature/modular-component-runtime`
- version: `V1.2.6`（沿用父计划 `V1.2`）
- category: `governance`、`architecture`、`code`、`experiment`、`documentation`
- post-commit: 未提交；未停止旧 Cm 训练进程
- scope: `src/task/CmComponent/` 嵌套 Component 配置、`src/base/run_manifest.py`、回归测试和规范文档

**文件**

- `src/task/CmComponent/src/config.py`、`configs/active/smoke_object_pose.yaml`、`components/components.json` — 删除根级重复的 encoder/head 选择，只保留 `coordinate_transform` 与 `pipeline` 根节点，子选择集中在 `pipeline.children`；操作版本升为 `V1.2.6`。
- `src/task/CmComponent/src/component_factory.py`、`src/task/CmComponent/src/pipeline.py` — Pipeline 的 encoder/head 只从 `pipeline.children` 解析，支持保持父级合同时的内部递归拆分。
- `src/base/run_manifest.py` — 以递归 `component_tree` 为规范来源，生成 child-first、去重的扁平 `components` 兼容视图，并修正叶节点空 `children`。
- `AGENTS.md`、CmComponent README/registry README、项目总览 — 固化根节点唯一声明、扁平清单自动生成和外层 Pipeline 合同边界。
- `tests/test_cm_component_task.py`、`tests/test_run_manifest.py` — 回归根/子 Component 选择、扁平/递归 manifest 和叶节点结构。

**原因**

用户指出父 Component 继续拆分时不应让外层 Pipeline 与配置维护重复选择；采用单一嵌套选择源，保持外层只依赖父级稳定合同。

**验证**

- 两种坐标配置均完成 CPU `max_steps=2` smoke train/eval；最新证据：`outputs/cmcomponent/cm_component_smoke_20260831_143310/`、`outputs/cmcomponent/cm_component_smoke_object_pose_20260831_143320/`。
- 最新 run manifest：根节点为 `coordinate_transform`、`pipeline`；扁平视图按 `coordinate_transform`、`dense_encoder`、`cm_head`、`pipeline` 生成；配置快照无根级重复子选择。
- 定向 pytest：`19 passed`；全量 pytest：`325 passed, 3 skipped, 22 warnings`。
- `researchctl list`、Pipeline `check --resolve-entrypoints` 和 `run --dry-run` 均通过；`compileall`、YAML/JSON parse、`git diff --check` 和局部 modification audit 通过。
- `src/task/Cm/`、用户指导、架构快照、旧数据/cache/checkpoint/output 和运行中的旧 Cm 进程未修改。

## 2026-08-31 — V1.2.5 递归 Component 选择与组件树追溯

- change_level: L2（递归 Component 选择与运行追溯合同）+ L3（Task 组件治理与文档）
- approval: user-approved（用户明确回复“可以，你继续吧”）
- skills_used: `research-change-control`、`modular-component-runtime`、`research-experiment-workflow`
- branch: `feature/modular-component-runtime`
- version: `V1.2.5`（沿用父计划 `V1.2`）
- category: `governance`、`architecture`、`code`、`experiment`、`documentation`
- post-commit: 未提交；未停止旧 Cm 训练进程
- scope: `src/task/CmComponent/` 递归 Component 选择、`src/base/run_manifest.py`、AGENTS 与日志

**文件**

- `AGENTS.md` — 将 `children` 递归选择、扁平/递归双重追溯和 `component_tree` 要求纳入运行规范。
- `src/task/CmComponent/src/component_factory.py`、`src/task/CmComponent/src/pipeline.py` — 支持父 role 路径和 `children` 递归解析。
- `src/task/CmComponent/src/config.py`、`configs/active/smoke_object_pose.yaml`、`components/components.json` — 登记 `pipeline -> dense_encoder/cm_head` 子组件选择，操作版本升为 `V1.2.5`。
- `src/base/run_manifest.py`、`tests/test_run_manifest.py` — 保留扁平 `components`，新增去重递归 `component_tree`。
- `src/task/CmComponent/docs/README.md`、`registry/README.md`、根/Task 日志和项目总览 — 记录递归规则、版本和证据入口；未更新用户指导或架构快照。

**原因**

用户要求继续支持 Component 内部拆分。采用显式 `children` 和父 role 路径，避免隐式类扫描，且不改变旧的扁平清单消费者。

**验证**

- 两种 V1.2.5 CPU `max_steps=2` smoke train/eval 通过；run manifest 显示 `coordinate_transform` 与 `pipeline -> dense_encoder/cm_head`。
- `researchctl` 两个坐标 manifest check、Pipeline dry-run 通过。
- 定向 pytest：`18 passed`；全量 pytest：`324 passed, 3 skipped`。
- `git diff --check` 与局部 modification audit 通过；旧 Cm 代码、数据/cache/checkpoint/output 和进程未修改。

## 2026-08-31 — V1.2.4 坐标策略 Component 化与汇报规范

- change_level: L2（坐标/数据合同与 Component 接口）+ L3（仓库 Agent 规范）
- approval: user-approved（用户明确要求把两个坐标选项抽象为 Component，并补充 `AGENTS.md` 汇报标准）
- skills_used: `research-change-control`、`modular-component-runtime`、`research-experiment-workflow`
- branch: `feature/modular-component-runtime`
- version: `V1.2.4`（沿用父计划 `V1.2`）
- category: `governance`、`architecture`、`code`、`data`、`experiment`
- post-commit: 未提交；未停止旧 Cm 训练进程
- scope: `AGENTS.md`、`src/task/CmComponent/` 坐标 Component、smoke 配置与测试、根/Task 日志

**文件**

- `AGENTS.md` — 明确多个候选实现必须晋升为 Component、每个运行 role 恰好选择一个实现、可发现 manifest 统一使用 `component.yaml`、新增/替换组件必须有清单与合同测试，耦合处定义稳定接口，YAML/JSON 只选择而不执行，并新增完成汇报和规范反馈标准。
- `src/base/run_manifest.py`、`tests/test_run_manifest.py` — 过滤 Component ID/version 等标量，保留 registry/spec 路径，避免把 provenance 标识误记为输入文件。
- `src/task/CmComponent/src/component_factory.py`、`tests/test_cm_component_task.py` — 必需 role 缺失时显式失败；默认实现仅允许由调用方明确传入 `required=False` 的兼容场景。
- `src/task/CmComponent/src/eval.py`、`tests/test_cm_component_task.py` — eval 未显式指定输出目录时，追溯文件默认写回 checkpoint 所属 run 目录。
- `outputs/cmcomponent/_legacy_eval_root_20260831/` — 保留并归档本轮早期误写到仓库根目录的 12 个 eval 追溯副本；未删除，便于恢复核对。
- `src/task/CmComponent/components/coordinate_transform/` — 新增 `HandRootFrameComponent` 与 `ObjectPoseFrameComponent`，共享 world→canonical 合同。
- `src/task/CmComponent/src/`、`dataset/`、`configs/active/`、`components/components.json` — 接入坐标 role/version 选择、frame metadata、失败校验和双配置 smoke。
- `tests/test_cm_component_task.py` — 增加坐标 Component registry、精确变换、候选选择和双 loader 验证。
- 根/Task `docs/logs/`、Task README — 同步 V1.2.4 状态、决策、证据和规范反馈；未更新用户指导或架构快照。

**原因**

用户要求将出现多个候选实现的坐标处理抽象成真正的可插拔 Component，并让后续 Agent 遵循这一规则；同时要求完成后报告规范限制和改进建议。

**验证**

- registry 发现 5 个 CmComponent manifest，两个坐标 Component 的 entrypoint 和 Artifact contract 校验通过。
- `hand_root_t` 与 `object_pose_t` 坐标变换、flow、缺失 pose 失败和 metadata 选择测试通过。
- 两种 CPU `max_steps=2` smoke train 与 latest checkpoint test eval 通过；run manifest 记录实际坐标 Component、版本和 pose source。
- 定向 pytest：`14 passed`（CmComponent）；全量 pytest：`323 passed, 3 skipped`。
- `git diff --check` 通过；旧 Cm 代码、数据/cache/checkpoint/output 和进程未修改。


## 2026-08-31 — V1.2.3 CmComponent 独立插拔式训练闭环

- change_level: L2（Task-local Component 合同、模型和 checkpoint schema）+ L3（独立 Dataset、Runner、训练/评估入口和 smoke 运行）
- approval: user-approved（用户明确回复“确认”）
- skills_used: `research-change-control`、`modular-component-runtime`、`research-experiment-workflow`
- branch: `feature/modular-component-runtime`
- version: `V1.2.3`（plan: `src/task/CmComponent/docs/plan/V1.2.3.md`, final）
- category: `architecture`、`code`、`experiment`
- post-commit: 未提交；未停止当前旧 Cm 训练进程
- scope: `src/task/CmComponent/` 独立组件运行时、Task-local 数据契约、smoke 配置和定向测试；根级状态导航

**文件**

- `src/task/CmComponent/` — 移除旧 adapter 依赖，新增独立 encoder、slot-flow head、Pipeline、Dataset、Runner、smoke 配置和 `2.0.0` Component 清单。
- `tests/test_cm_component_task.py` — 覆盖无旧 Cm 引用、清单、拓扑、版本、Artifact 合同、梯度和 state-dict roundtrip。
- `docs/项目总览.md`、`docs/logs/status_log.md`、`docs/logs/repo_memory.md`、`docs/logs/modification_log.md` — 更新 V1.2.3 状态导航和可迁移事实；未更新架构快照或用户指导。

**原因**

用户确认将 CmComponent 做成不耦合旧 Cm、可通过统一 Runner 组合的可插拔训练框架，并先跑通最小 smoke。

**验证**

- `rg -n "src\\.task\\.Cm(?!Component)" src/task/CmComponent --pcre2 -g '*.py' -g '*.yaml' -g '*.json'`：无旧 Cm 运行时引用。
- 三个 `2.0.0` manifest 的 `researchctl check --resolve-entrypoints`、Pipeline `check`、`graph` 和 `run --dry-run` 通过，拓扑为 `dense_encoder -> cm_head`。
- `PYTHONPATH=. /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_cm_component_task.py`：`9 passed`；全量 pytest：`317 passed, 3 skipped`。
- CPU `max_steps=2` train 通过，生成 `outputs/cmcomponent/cm_component_smoke_20260831_111831/` 的 run manifest、配置、metadata、metrics 和 checkpoint；latest checkpoint 的 test eval 通过（`test/flow/epe_mm=9.17651`）。
- `git diff --check` 通过；未修改旧 Cm、旧数据/cache/checkpoint/output 或运行进程。


## 2026-08-30 — V1.2.2 创建隔离的 CmComponent Task

- change_level: L2（Component 合同与 checkpoint 兼容）+ L3（新增 Task-local Pipeline Runner）
- approval: user-approved（用户明确要求“你先做吧”）
- skills_used: `research-change-control`、`modular-component-runtime`
- branch: `feature/modular-component-runtime`
- version: `V1.2.2`（plan: `src/task/CmComponent/docs/plan/V1.2.md`, final）
- category: `architecture`、`code`、`governance`
- post-commit: 未提交；未停止当前训练进程
- scope: `src/task/CmComponent/` 新 Task、Component registry、Pipeline Runner 和项目索引

**文件与变更**

- `src/task/CmComponent/components/` — 新增真实 Dense encoder/Cm head Component、Pipeline manifest、显式 `pipeline.yaml` 拓扑和 Task `components.json`。
- `src/task/CmComponent/src/` — 新增 Pipeline、Runner、配置、组件选择解析、train/eval 入口；Runner 复用旧 Cm 的数据、loss、指标生命周期，但模型由清单驱动的 Component Pipeline 构造。
- `src/task/CmComponent/registry/`、`data/` — 新增相对数据入口、producer/consumer 追溯和只读软链接。
- `src/task/CmComponent/docs/` — 新 Task README、V1.2 final plan、指导只读软链接及状态/决策/记忆/修改日志。
- `docs/项目总览.md`、`docs/logs/status_log.md`、`docs/logs/repo_memory.md` — 增加新 Task 索引、当前细分版本和可迁移架构事实。

**原因**

用户要求用独立 Task 承载 Component 版本的 Cm，以便新架构可试验、可删除回滚，避免影响旧 Cm。

**验证**

- `researchctl check` 对 dense encoder、Cm head、pipeline 三个 manifest 的入口解析均通过；未实例化组件、未加载真实权重。
- `researchctl check` 与 `graph` 对新 Task 的显式 Pipeline 拓扑通过，顺序为 `dense_encoder -> cm_head`。
- `researchctl run ... --dry-run` 通过，仅打印拓扑，未启动训练或加载真实权重。
- 新增 `tests/test_cm_component_task.py`，定向 pytest：`8 passed`；新 Task Python 文件通过 `py_compile`，`git diff --check` 通过。
- 共享回归：全量 pytest `316 passed, 3 skipped`。
- 尚未完成：使用真实 DenseToken checkpoint 的固定 batch 数值 parity；在该 parity 通过前不启动新 Task 正式训练。
- 本轮未启动新 Task 正式训练，未移动或覆盖旧 Cm 的数据、cache、checkpoint、output 或训练进程。

## 2026-08-30 — V1.2.1 版本线、Task Component 与数据注册治理

- change_level: L3（仓库治理、Component 路径和运行追溯）
- approval: user-approved（用户授权按 Agent 方案执行）
- skills_used: `research-change-control`、`research-experiment-workflow`、`modular-component-runtime`
- branch: `feature/modular-component-runtime`
- version: `V1.2.1`（plan: `V1.2`, final）
- category: `governance`、`architecture`、`data`、`documentation`
- post-commit: 未提交；未停止当前训练进程
- scope: 全局组件发现、Task 组件目录、版本线、运行 manifest、Task 数据路径注册

**文件与变更**

- `src/task/Cm/components/`、`src/task/CmDecoder/components/`、`src/task/correspondence_ptv3_v2/components/` — Task 专属 manifest、adapter、`__init__.py` 和 `components.json`；记录组件 ID、版本、角色、manifest 和参数前缀。
- `components/ref2dex/cm`、`components/ref2dex/cmdecoder_pointflow`、`components/ref2dex/correspondence_ptv3_v2` — 由旧实体目录改为指向 Task canonical 目录的相对软链接，保留旧入口兼容。
- `src/task/*/registry/data_paths.json`、`external_paths.json.example`、Task 内 `data/` 软链接和 `.gitignore` — 以仓库相对路径区分数据/cache/共享资产，并记录 producer/consumer 配置关系；机器特有路径由忽略的覆盖文件提供。
- `src/base/base_config.py`、`src/base/run_manifest.py`、`src/base/registry.py`、`tools/researchctl.py` — 配置显式携带版本/类别/组件清单，运行追溯写入这些字段，registry 默认发现 Task 组件并去重兼容软链接。
- `AGENTS.md`、`docs/版本线与AI行为分类.md`、`docs/目录规范.md`、`docs/AI交接清单.md`、`docs/项目总览.md`、`components/README.md`、`src/task/Cm/docs/README.md`、`src/task/Cm/docs/plan/V1.2.md` — 固化版本层级、操作类别、组件/数据职责和交接入口。
- `docs/logs/`、`src/task/Cm/docs/logs/` — 增加本次版本、类别、决策、状态和可迁移事实记录；架构事实文件不因本次治理自动建立新快照。

**原因**

用户要求从现在开始以 `Vn.m` plan 为主线、以 `Vn.m.k` 管理细小改动和并行实验，将 Task Component 下放并能在每次运行中看到组件及参数关系，同时保留软链接迁移的可逆性。

**验证**

- 配置加载可解析 `version_line=V1.2`、`plan_version=V1.2`、`operation_version=V1.2.1` 和组件清单。
- `researchctl list/check --resolve-entrypoints` 可发现并校验 Task 组件；registry 对旧软链接与 canonical 路径去重。
- 相关 `py_compile`、定向 pytest（`7 passed`）、全量 pytest（`308 passed, 3 skipped`）、`git diff --check` 和路径/JSON/YAML 检查通过；历史 run manifest 未回写。

## 2026-08-30 — 统一研究产物与共享资产路径

- change_level: L3（仓库治理与共享资产路径迁移；未移动大型文件）
- approval: user-approved
- skills_used: `research-change-control`、`research-experiment-workflow`、`modular-component-runtime`
- branch: `feature/modular-component-runtime`
- post-commit: 未提交
- scope: 全局目录规范、Cm research 实验包、共享 DenseToken checkpoint 入口

**文件**

- `AGENTS.md`、`.gitignore`、`docs/目录规范.md`、`docs/AI交接清单.md`、`docs/项目总览.md`、`outputs/README.md`、`assets/README.md` — 固定源码、研究、数据、资产、运行产物和 manifest 的目录职责。
- `src/task/Cm/research/`、`src/task/Cm/docs/README.md` — 将 t-SNE、DenseToken parity、hand-flow calibration 组织为实验包，并为每个实验增加 `README.md`、`experiment.yaml` 和独立 `output/`；三个入口默认写入各自的 `output/<run_id>/` 并生成 `run_manifest.json`。
- `src/task/Cm/src/config.py`、`assets/checkpoints/densetoken` — 采用新共享资产入口；旧 checkpoint 目录暂不移动，保留软链接兼容。
- `docs/logs/architecture_log.md`、`docs/logs/decision_log.md`、`docs/logs/repo_memory.md`、`docs/logs/status_log.md`、`docs/logs/modification_log.md`、`src/task/Cm/docs/logs/architecture_log.md`、`src/task/Cm/docs/logs/status_log.md`、`src/task/Cm/docs/logs/modification_log.md` — 同步治理决策、路径事实和交接记录。

**改动原因**

用户确认先以软链接方式迁移，解决 AI 随意创建 `output/`、`results/` 和源码旁产物的问题，同时不打断正在运行的多卡任务。根 `output/` 与现有历史结果保留，但不再作为新入口。

**验证**

- `python -m src.task.Cm.research.tsne_slots --help` 与旧软链接入口帮助命令通过；
- 新旧 DenseToken 路径均能解析到同一 checkpoint；
- `git diff --check` 通过；
- 未停止训练，未删除或覆盖 data、cache、output、outputs、checkpoint 或历史 result。

## 2026-08-30 — 收口 Component 合同、运行追溯与 Cm 坐标校验

- change_level: L3（共享运行时与仓库治理）+ L2（公共坐标合同）
- approval: user-approved（用户明确要求直接修改）
- skills_used: `research-change-control`、`modular-component-runtime`
- branch: `feature/modular-component-runtime`
- post-commit: 未提交
- scope: 共享 Component/Artifact/Contract/Registry、BaseRunner run manifest、Cm loader 坐标一致性

**文件**

- `src/base/component.py`、`src/base/contract.py`、`src/base/registry.py` — 增加 Component 输入/输出校验入口、坐标字段归一化、可选严格 metadata 校验，并用 `entrypoint_kind` 防止把 task runner 误报为 Component。
- `src/base/run_manifest.py`、`src/base/base_runner.py` — 过滤 manifest 自引用；train/eval/resume 生成配置、metadata 和不覆盖历史的时间戳 manifest。
- `components/ref2dex/cm/component.yaml`、`components/ref2dex/cmdecoder_pointflow/component.yaml`、`components/ref2dex/correspondence_ptv3_v2/component.yaml` — 显式声明 task 入口；`components/ref2dex/cm/inference/component.yaml`、`components/ref2dex/cm/inference_v2/component.yaml` 与 `components/ref2dex/cm/adapter.py` — 保留 hand-root v1 并增加 object-pose v2 adapter 合同。
- `components/README.md`、`.agents/skills/modular-component-runtime/references/component-template.md`、`tools/researchctl.py` — 同步 manifest 类型和显式入口检查说明。
- `.agents/skills/research-change-control/scripts/audit_diff.py` — 增加局部 scope 前缀审计，避免 dirty worktree 中把其他任务改动混入一次审计。
- `docs/logs/architecture_log.md`、`docs/logs/status_log.md`、`tests/test_framework_contracts.py` — 更新共享事实并覆盖入口、坐标和自引用回归。

**原因**

当前框架已有原型但存在三个闭环缺口：任务 runner manifest 与真正 Component 没有机器可区分的入口类型，eval/resume 可能覆盖运行追溯，数据根只看第一条序列会允许坐标系混用。本轮以最小差异补齐这些边界；不改模型、loss、GT、split 内容、checkpoint 或正在运行的训练。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_framework_contracts.py tests/test_component_registry.py tests/test_run_manifest.py tests/test_cm_object_v2.py tests/test_cm_sequence_dataset.py tests/test_cm_scene.py tests/test_cm_hrdexdb.py tests/test_cm_surface_sampling.py tests/test_hand_root_frame.py`：79 passed。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q`：307 passed，3 skipped。
- `graspenv` 下修改模块 `py_compile`：通过；`git diff --check`：通过。
- `researchctl list/check --resolve-entrypoints`：task/component 入口均按声明校验通过。

- scope: root
- last_updated: 2026-08-30
- related: [架构记录](architecture_log.md)、[仓库记忆](repo_memory.md)、[决策记录](decision_log.md)

## 2026-08-30 — 增加通用目录路由、Component 晋升与 Run Manifest 规范

- change_level: L3（仓库治理规则）
- approval: user-approved
- skills_used: `skill-creator`、`research-change-control`、`research-experiment-workflow`、`modular-component-runtime`
- branch: working tree
- post-commit: 未提交
- scope: 全局治理 / 可复制工作流

**文件**

- `AGENTS.md` — 增加默认目录路由、自动归类和训练运行追溯要求。
- `.agents/skills/research-change-control/SKILL.md` — 修改记录增加 `skills_used` 审计字段。
- `.agents/skills/research-experiment-workflow/SKILL.md` — 增加默认产物位置和 Run Manifest 规则，并与 `outputs/<Task>/<run_id>/` Runner 路径对齐。
- `.agents/skills/modular-component-runtime/SKILL.md` — 增加无需逐次指定场景的 Component 晋升规则。
- `src/base/run_manifest.py` — 新增任务无关的配置、Git、输入引用和合同追溯清单生成器，不计算加密 hash。
- `src/base/base_runner.py` — 训练启动时自动写入 `run_manifest.json`，并将其路径纳入运行 metadata。
- `src/base/base_config.py` — 记录从文件加载配置时的私有来源路径，供 run manifest 使用。
- `src/base/__init__.py` — 导出 run manifest 构建与写入接口。
- `tests/test_run_manifest.py` — 覆盖配置快照引用、输入 manifest 文件基本信息、合同字段和原子写入。
- `tests/test_overfit_diagnosis.py` — 增加 BaseRunner 启动时实际生成 run manifest 的集成断言。
- `docs/logs/architecture_log.md`、`docs/logs/status_log.md` — 同步共享运行追溯合同和当前框架状态。

**改动原因**

让 Agent 能根据内容自动决定代码、配置、Component 和输出位置；并让训练入口实际生成 run manifest，
区分数据 manifest、训练 run manifest 和实验日志，减少每次依赖用户手工指定目录或是否创建 Component。

**验证**

- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q tests/test_run_manifest.py tests/test_base_runner_max_steps.py tests/test_base_runner_validation.py`：9 passed。
- `/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q`：302 passed，3 skipped。
- 移除 SHA-256 后重新运行完整测试：仍为 `302 passed，3 skipped`。
- 三个仓库 Skill 通过 `skill-creator/scripts/quick_validate.py`。
- `git diff --check`：通过。

## 2026-08-30 — 同步 CmDecoder held-out test rollout 结果

- branch: working tree
- post-commit: 未提交
- scope: 跨 task 状态导航；change_level: L0；approval: auto

**文件**

- `docs/logs/status_log.md` — 记录 EXP-025 held-out Inspire test rollout 及当前训练状态。
- `src/task/CmDecoder/docs/logs/{status,experiment,modification}_log.md` — 记录测试 episode、协议、指标和输出路径。

**改动原因**

响应用户要求使用 test 集验证当前 CmDecoder best checkpoint；未修改训练配置或训练进程。

**验证**

已确认 `inspire_f1/bamboo_basket/5` 为 test split，32 个有效 pair 成功生成 trajectory/PNG。

## 2026-08-30 — 同步 CmDecoder EXP-022 当前验证状态

- branch: working tree
- post-commit: 未提交
- scope: 跨 task 状态导航；change_level: L0；approval: auto

**文件**

- `docs/logs/status_log.md` — 更新三卡 CmDecoder 当前进度、最佳验证指标和 ETA。
- `src/task/CmDecoder/docs/logs/{status,experiment,modification}_log.md` — 记录详细验证曲线与证据。

**改动原因**

响应用户训练状态检查；只同步运行证据，没有修改训练配置或进程。

**验证**

已核对活跃 torchrun、三卡利用率、训练日志和 checkpoint payload；未发现 OOM、NaN 或 NCCL 错误。

## 2026-08-30 — 同步 CmDecoder best rollout 结果

- branch: working tree
- post-commit: 未提交
- scope: 跨 task 状态导航；change_level: L0；approval: auto

**文件**

- `docs/logs/status_log.md` — 更新 EXP-023 rollout 已完成及关键指标。
- `src/task/CmDecoder/docs/logs/{status,experiment,modification}_log.md` — 记录详细 rollout 证据。

**改动原因**

响应当前研究状态检查；只同步已生成的诊断结果，没有修改训练进程。

**验证**

已核对 trajectory、PNG、rollout 控制台输出及 EXP-022 三卡训练仍在运行；31 个有效 pair 成功生成。

## 2026-08-29 — 停止 Cm 三卡训练并保留恢复锚点

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 长时任务；change_level: L3；approval: user-approved

**文件**

- `docs/logs/status_log.md` — 更新 Cm 停止及 CmDecoder 三卡迁移待确认状态。
- `src/task/Cm/docs/logs/{status,experiment,modification}_log.md` — 记录 Cm 停止、恢复 checkpoint 与最终完整验证证据。
- `src/task/CmDecoder/docs/logs/{status,modification}_log.md` — 记录 decoder 当前单卡进度和待迁移状态。

**改动原因**

用户明确要求停止 GPU0/1/2 上的 Cm，将资源转给 CmDecoder。已确认 Cm torchrun 与 worker 全部退出，epoch 29 / step `124410` checkpoint 保留；decoder 尚未改为三卡。

## 2026-08-29 — CmDecoder 改为三卡 global batch 48 并固定总 step

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 长时任务；change_level: L3；approval: user-approved

**文件**

- `docs/logs/status_log.md` — 更新 CmDecoder 三卡 run 状态。
- `src/task/CmDecoder/docs/logs/{status,experiment,decision,modification}_log.md` — 记录 EXP-022、训练预算和验证。

**改动原因**

用户确认保持总 optimizer step 数 `143110`，将 global batch 改为 `48` 并扩宽 epoch；已在 GPU0/1/2 启动并通过 distributed smoke。

## 2026-08-29 — 记录 Cm 与 CmDecoder 单卡并行造成的吞吐变化

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 运行状态记录
- change_level: L3（长时实验状态）
- approval: user-approved

**文件**

- `docs/logs/status_log.md` — 记录 CmDecoder 并行后 Cm DDP 吞吐和 GPU1 慢卡现象。
- `src/task/Cm/docs/logs/{status,experiment}_log.md` — 同步 Cm 当前 step、吞吐变化和显存状态。
- `src/task/CmDecoder/docs/logs/status_log.md` — 记录共享 GPU 的性能风险。

**改动原因**

对比并行前后 Cm 日志：step time 约由 `300--313 ms` 增至 `650 ms`，global throughput 约由 `306--335` 降至 `146--148 samples/s`；当前无 OOM/NaN，判断 GPU1 竞争通过 DDP 同步拖慢整体 Cm。

## 2026-08-29 — 增加指导与执行计划协商流程

- branch: `feature/modular-component-runtime`
- post-commit: 未提交
- scope: 全局治理 / 可复制工作流
- change_level: L3（仓库治理规则）
- approval: user-approved

**文件**

- `AGENTS.md` — 规定 `指导/V<n>.md` 与同版本 `plan/V<n>.md` 的职责、定稿闸门和偏离处理。
- `.agents/skills/research-change-control/SKILL.md` — 增加任务无关的指导—plan 协商流程，明确草稿不写修改日志、定稿后记录最终实现。

**原因**

让用户提供研究方向、Agent 负责形成可执行计划，并在计划定稿前阻止直接修改代码；同时避免把协商过程的每次草稿变化伪装成代码修改历史。

**验证**

- 已核对规则与现有 `src/task/Cm/docs/指导/V2.md`、`plan/V2.md` 的版本配对方式；未修改或代填 V2 内容。

## 2026-08-29 — 完成 Cm 破坏性目录迁移并移除旧入口

- branch: `feature/modular-component-runtime`
- post-commit: 本提交（以 `git log` 为准）
- scope: 跨 task / Cm 运行时
- change_level: L3（破坏性结构迁移）
- approval: 用户已明确确认完全搬迁，并授权纳入当前未提交修改

**文件**

- `src/task/Cm/` — 直接承载唯一真实实现、配置和文档迁移。
- `src/task/CmDecoder/`、`src/task/InteractionDynamics/`、`components/ref2dex/cm/`、`process/GRAB/`、`tests/`、`tools/`、`docs/logs/` — 同步新入口、实验改动和文档事实。

**改动原因**

用户确认不再维护旧壳，并授权将当前未提交的 Cm/CmDecoder 实验改动作为本次迁移的一部分一起提交。历史运行需固定到迁移前 commit 复现。

**验证**

- 41 个 `active/archive` YAML 均可由 `load_config` 解析，新模块入口可导入；结构回归测试确认旧顶层入口不存在；全量 pytest `300 passed, 3 skipped`。

## 2026-08-28 — 增加显式 entrypoint 解析、示例执行组件与 pipeline dry-run

- branch: `feature/modular-component-runtime`
- post-commit: 本提交（以 `git log` 为准）
- scope: 全局 / 通用运行时

**文件**

- `src/base/component.py`、`src/base/registry.py`、`src/base/__init__.py` — 增加 `Component` 最小执行接口和显式 entrypoint 解析；registry 仍保持发现阶段不导入。
- `components/examples/{identity,scale}.py` — 增加不依赖具体 Task 的 Artifact 传递示例。
- `components/examples/{identity,scale}/component.yaml` — 将示例 manifest 入口指向可解析实现。
- `tools/researchctl.py` — 增加 `check --resolve-entrypoints` 和 `run --dry-run`。
- `tests/test_component_registry.py` — 覆盖入口解析、示例执行和 pipeline dry-run 所需合同。
- `docs/logs/{architecture,status,decision,modification}_log.md` — 同步运行时边界、证据和决策。

**改动原因**

把“可发现”与“可执行”明确分层：先用纯 manifest 建索引，再由显式命令验证入口；在尚未迁移真实
训练任务前，用 toy component 验证 Artifact、Context 和拓扑合同，降低热插拔改造的风险。

**验证**

- 定向组件测试：`11 passed`。
- 全量 pytest：`294 passed, 3 skipped`。
- `researchctl check --resolve-entrypoints` 与 `run --dry-run` smoke 通过。

## 2026-08-28 — 增加 Cm inference adapter pilot

- branch: `feature/modular-component-runtime`
- post-commit: 本提交（以 `git log` 为准）
- scope: 全局组件框架 / task adapter

**文件**

- `components/ref2dex/cm/adapter.py` — 将现有 `CmFlowModel` 包装为单步 `Component`，执行显式输入校验并输出 representation/object flow Artifact；提供显式 `from_checkpoint` 构造入口。
- `components/ref2dex/cm/inference/component.yaml` — 新增独立 Cm inference pilot manifest，补齐 normals、valid mask、delta time 输入合同；原 `components/ref2dex/cm/component.yaml` runner manifest 保持不变。
- `tests/test_component_registry.py` — 增加 Cm adapter 的 fake-model 映射和缺失输入测试。
- `components/README.md`、`docs/logs/{architecture,status,decision,modification}_log.md` — 记录 Cm pilot 边界和当前证据。

**改动原因**

用户确认第一个真实 pilot 使用 Cm，其他执行边界保持不变。因此只包装 inference，不移动或改写 `src/task/Cm` 的训练实现。

**验证**

- Cm adapter 定向测试：`13 passed`。
- 全量 pytest：`296 passed, 3 skipped`。

## 2026-08-29 — 将通用维护流程拆为可复制 Skill 并收缩 AGENTS

- branch: `feature/modular-component-runtime`
- change_level: L3
- approval: user-approved
- post-commit: 本提交（以 `git log` 为准）
- scope: 全局 / 仓库治理

**文件**

- `.agents/skills/research-change-control/SKILL.md` — 新增任务无关的修改分级、审批和修改记录一致性 Skill。
- `.agents/skills/research-change-control/references/change-levels.md` — 记录通用 L0–L3 影响等级。
- `.agents/skills/research-change-control/references/audit-contract.md` — 说明审计脚本的能力边界。
- `.agents/skills/research-change-control/scripts/audit_diff.py` — 提供 staged diff 与修改记录的一致性审计。
- `.agents/skills/research-experiment-workflow/SKILL.md` — 新增通用实验、产物和长任务工作流 Skill。
- `.agents/skills/modular-component-runtime/SKILL.md` — 新增任务无关的组件组合和接口兼容 Skill。
- `AGENTS.md` — 收缩为 Ref2Dex 本地路径、日志、科学边界、Skill 路由和 Git 安全规则。
- `docs/logs/{architecture,status,modification}_log.md` — 同步 Skill 分层和当前状态。

**改动原因**

用户希望维护流程可复制到其他任务，同时避免根 AGENTS 承载所有通用规范。Skill 内容不写死 Ref2Dex 路径；项目特殊事实继续由 AGENTS 提供。

**验证**

- 三个 Skill 均通过 `skill-creator` 的 `quick_validate.py`。
- `git diff --check` 通过。
- Cm manifest 显式 entrypoint 解析和 `researchctl describe` 通过。

## 2026-08-29 — 中文化通用 Skill 并恢复完整长时任务规则

- change_level: L0
- approval: user-approved
- branch: `feature/modular-component-runtime`
- post-commit: 本提交（以 `git log` 为准）
- scope: 全局 / 可复制 Skill 文档

**文件**

- `.agents/skills/research-change-control/SKILL.md` — 将修改分级、审批闸门和交接流程改为中文。
- `.agents/skills/research-change-control/references/change-levels.md` — 将 L0–L3 影响等级参考改为中文。
- `.agents/skills/research-change-control/references/audit-contract.md` — 将审计脚本能力边界说明改为中文。
- `.agents/skills/research-change-control/scripts/audit_diff.py` — 将脚本帮助信息、错误信息和 docstring 改为中文，行为不变。
- `.agents/skills/research-experiment-workflow/SKILL.md` — 将实验工作流改为中文，并链接长时任务参考。
- `.agents/skills/research-experiment-workflow/references/long-running-tasks.md` — 完整恢复旧版长时任务等待、polling、嵌套工具和交互输入规则。
- `.agents/skills/modular-component-runtime/SKILL.md` — 将通用组件运行时边界与安全规则改为中文。

**改动原因**

统一可复制维护入口的语言，同时保留原 AGENTS.md 中关于长时任务的详细操作约束；本次不改变组件、实验或脚本的运行语义。

**验证**

- `quick_validate.py`：三个 Skill 通过。
- `python3 -m py_compile .agents/skills/research-change-control/scripts/audit_diff.py`：通过。
- `git diff --check`：通过。

## 2026-08-29 — 建立 Cm 目录迁移层与 Component 模板

- change_level: L3
- approval: user-approved
- branch: `feature/modular-component-runtime`
- post-commit: 本提交（以 `git log` 为准）
- scope: 全局 Skill + task:Cm 结构治理

**文件**

- `.agents/skills/modular-component-runtime/SKILL.md` — 增加单实现不强制抽象、逻辑组件可共文件和兼容迁移规则。
- `.agents/skills/modular-component-runtime/references/component-template.md` — 增加统一 Component manifest、Python 接口和版本兼容模板。
- `.agents/skills/research-change-control/scripts/audit_diff.py` — 支持用目录前缀归组审计路径。
- `docs/logs/architecture_log.md`、`docs/logs/status_log.md` — 记录 Cm 结构迁移入口和当前状态。
- `src/task/Cm/src/` — 建立模型、runner、配置、训练、评估和提取的规范入口，暂时转发旧实现。
- `src/task/Cm/dataset/` — 建立数据集规范包，迁入 Stage4 实现并提供 ObjectV2、Scene、HRDexDB 和工厂入口。
- `src/task/Cm/visualization/` — 建立可视化与 viewer server 规范入口。
- `src/task/Cm/configs/` — 将受版本管理配置按生命周期归入 `active/` 与 `archive/`；旧顶层路径保留兼容软链接。
- `src/task/Cm/docs/README.md`、`src/task/Cm/configs/{README.md,active/README.md,archive/README.md}` — 增加目录职责与文档入口说明。
- `tests/test_cm_structure.py` — 验证新旧 import、配置路径和生命周期目录。

**改动原因**

按用户确认的结构整理 Cm，同时避免覆盖当前未提交的核心实验改动。采用兼容迁移层，使旧配置、旧 import 和现有 checkpoint 入口继续工作；本次不引入 Hydra，不改变训练或推理语义。

**验证**

- 新旧配置加载及 `src.task.Cm.src`、`src.task.Cm.dataset`、`src.task.Cm.visualization` 兼容 import 通过。
- Cm 定向测试：`37 passed`。
- 新迁移模块 `py_compile` 通过。
- `git diff --check` 及 staged modification log 审计将在提交前执行。

## 2026-08-28 — 扩展通用组件协议与 pipeline dry-run

- branch: `oyx` working tree
- post-commit: 未提交
- scope: 全局 / 通用运行时

**文件**

- `src/base/artifact.py` — 增加任务无关的 Artifact/ArtifactRef provenance 描述。
- `src/base/contract.py` — 增加 Artifact 合同校验。
- `src/base/context.py` — 增加不依赖 PyTorch 的 ExecutionContext。
- `src/base/component.py` — 增加 Component 抽象接口。
- `src/base/pipeline.py` — 增加 PipelineSpec、节点/边解析和拓扑/环检测。
- `src/base/registry.py`、`tools/researchctl.py` — 接入 PipelineSpec，增加 `describe` 和 `run --dry-run`。
- `tests/test_component_registry.py` — 补充 Artifact 合同、拓扑排序和环检测测试。
- `docs/logs/{architecture,status,modification}_log.md` — 同步通用运行时事实和当前状态。

**改动原因**

按用户确认的通用热插拔路线，将第一版 registry 升级为可被未来 runtime adapter 使用的核心协议；仍不导入或执行真实 Task，避免改变既有训练语义。

**验证**

- `researchctl list/describe/check/graph/run --dry-run` smoke 通过。
- 全量 pytest：`291 passed, 3 skipped`。

## 2026-08-28 — 增加通用 Component registry 原型

- branch: `oyx` working tree
- post-commit: 未提交
- scope: 全局 / 通用运行时

**文件**

- `src/base/component.py`、`src/base/registry.py` — 增加任务无关的 manifest、端口合同和组件发现/检查接口。
- `src/base/__init__.py` — 导出通用组件协议。
- `tools/researchctl.py` — 提供 `list`、`check`、`graph` 命令。
- `components/` — 增加两个领域无关 toy manifest 和示例 pipeline。
- `tests/test_component_registry.py` — 覆盖发现、重复 ID、manifest 错误和端口不兼容。
- `AGENTS.md` — 补充 Component/Artifact/Contract/Pipeline 规范，明确 decoder 等术语不是框架一级抽象。
- `docs/项目总览.md`、`docs/logs/{architecture,status,decision,modification}_log.md` — 同步通用运行时入口、当前状态、决策和改动记录。

**改动原因**

根据用户确认，将热插拔设计落实为不依赖 Ref2Dex 领域的最小原型；不迁移现有 Task，不改变既有训练和实验语义。

**验证**

- `python tools/researchctl.py list components`
- `python tools/researchctl.py check components/examples/example_pipeline.yaml components`
- 全量 pytest：`289 passed, 3 skipped`。

## 2026-08-28 — 以只读 manifest 接入三条 Ref2Dex 主线

- branch: `oyx` working tree
- post-commit: 未提交
- scope: 全局 / 通用运行时索引

**文件**

- `components/ref2dex/{correspondence_ptv3_v2,cm,cmdecoder_pointflow}/component.yaml` — 为现有主线声明通用 component ID、版本、capabilities、端口合同和 tags。
- `components/README.md` — 补充当前只读接入清单。
- `tests/test_component_registry.py` — 增加三条主线 manifest 的发现和 CmDecoder frame 合同检查。
- `docs/logs/{architecture,status,decision,modification}_log.md` — 同步只读接入边界和状态。

**改动原因**

验证通用 registry 可以描述真实 Ref2Dex 组件，同时保持原 Task 代码、配置、训练入口和 checkpoint 不变。

**验证**

- `researchctl list/describe` 可发现并展示三条 active 主线。
- 全量 pytest：`291 passed, 3 skipped`。

## 2026-08-24 — 合并 hand PCA 分支并拆分仓库/机器记忆

- branch: `oyx`
- post-commit: 未提交
- scope: 全局 / 跨 task

**文件**

- `origin/feature/hand-pca-perturbation` — 合入 hand PCA perturbation、MANO reconstruction、HOCap/ARCTIC/ContactPose 数据处理、跨域评估、配置和测试；以 `oyx` 为主线。
- `.gitignore`、`pytest.ini` — 采用 feature 的 output/test 扫描策略，并忽略所有递归 `machine_memory.md`。
- `AGENTS.md` — 将 memory 拆为受版本管理的 `repo_memory.md` 和本机忽略的 `machine_memory.md`，保留根级/Task 级递归作用域。
- `docs/logs/repo_memory.md` 与各 Task 同名文件 — 迁移跨机器成立的相对路径、约定、历史遗留和兼容性事实。
- 根级及 Cm、CmDecoder、correspondence 的 `machine_memory.md` — 保存当前服务器的解释器、绝对路径、NAS、GPU 和代理事实；文件不纳入 Git。
- 各级 `repo_notes_log.md` — 删除，内容分别迁入 repo/machine memory 或已存在的 status/architecture。
- `docs/logs/modification_log.md` — 保留 `oyx` 外壳并完整并入 feature 分支的根级修改历史。

**改动原因**

用户要求在合并分支时消除 repo notes，并明确区分可上传的仓库记忆和不可上传的机器记忆，避免把单机绝对路径固化为项目公共事实。

**验证**

- 全量 pytest：`282 passed, 3 skipped`。
- 相关 hand PCA/MANO/Stage3 定向测试：`50 passed, 3 skipped`。
- `compileall` 覆盖本次修改的 process、base、correspondence 和 calibration tool；无语法错误。
- `git diff --check`、冲突标记和 machine memory ignore 检查通过。

## 2026-08-23 — 将 HRDexDB 非视频数据迁入 dataset 并统一入口

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 全局数据路径

**文件 / 数据**

- `dataset/HRDexDB/v0_nonvideo` — 从外部 HRDexDB 目录迁入四手型非视频原始数据；不含视频。
- `dataset/HRDexDB/assets`、`dataset/HRDexDB/hrdexdb_contact_heatmaps` — 复制机器人 URDF/mesh 和自包含读取 helper，形成可独立消费的本地数据入口。
- `.gitignore` — 整体忽略 `/dataset/HRDexDB/`，防止原始数据、模型和大量小文件进入 Git。
- `src/task/CmDecoder/{config,build_cache,grab_retarget,migrate_point_bindings}.py` — 默认路径改为仓库内规范位置。
- `docs/logs/memory_log.md`、Cm/CmDecoder 对应日志 — 同步规范路径及旧 symlink 兼容边界。

**改动原因**

用户要求整理本地 HRDexDB 数据位置。迁移前短暂停止在途 cache worker，完成原子移动并建立旧路径 symlink 后恢复，避免破坏已经生成的 cache 进度。

## 2026-08-23 — 优化共享 HRDexDB candidate cache 构建

- branch: 当前工作分支
- post-commit: 未提交
- scope: 跨 task / 全局共享 cache

**文件**

- `src/task/CmDecoder/build_cache.py` — 用 `cKDTree` 进行精确 5 cm 半径候选查询，保持共享 geometry schema 不变。
- `src/task/Cm/docs/logs/status_log.md` — 记录全量 cache 后台运行 PID 和日志。

**改动原因**

candidate mask 构建由逐帧全量距离张量改为半径邻域查询；单 episode CPU 探针由约 192.6 s 降至约 37.4 s。全量构建已按 4 workers、单线程 BLAS 启动。


## 2026-08-29 — 完成 Cm 破坏性目录迁移并移除旧入口

- branch: `feature/modular-component-runtime`
- post-commit: 待提交
- scope: 跨 task / Cm 运行时
- change_level: L3（破坏性结构迁移）
- approval: 用户已明确确认完全搬迁，并授权纳入当前未提交修改

**文件**

- `src/task/Cm/` — 直接承载唯一真实实现、配置和文档迁移。
- `src/task/CmDecoder/`、`src/task/InteractionDynamics/`、`components/ref2dex/cm/`、`process/GRAB/`、`tests/`、`tools/`、`docs/logs/` — 同步新入口、实验改动和文档事实。
- `src/task/Cm/configs/{active,archive}/` — 迁入全部配置并删除顶层 YAML 软链接。
- `docs/logs/{status,architecture,modification}_log.md` — 同步全局入口和事实。

**改动原因**

用户确认不再维护旧壳，并授权将当前未提交的 Cm/CmDecoder 实验改动作为本次迁移的一部分一起提交。历史运行需固定到迁移前 commit 复现。

**验证**

- 41 个 `active/archive` YAML 均可由 `load_config` 解析，新模块入口可导入；结构回归测试确认旧顶层入口不存在。

## 2026-08-24 — 启动 CmDecoder MANO 跨域泛化实验

- branch: `oyx`
- post-commit: 未提交
- scope: 全局状态同步

**文件**

- `docs/logs/status_log.md` — 记录 CmDecoder GRAB-trained MANO decoder 正在运行。

**改动原因**

CmDecoder 新增一条会影响仓库当前研究状态的 GRAB→ARCTIC MANO 泛化实验，根级入口需与任务级状态保持一致。

## 2026-08-24 — 同步全仓库当前训练状态

- branch: `oyx`
- post-commit: 未提交
- scope: 全局状态文档

**文件**

- `docs/logs/status_log.md` — 将当前进行中任务改为 Cm GRAB C=32 与双卡 C=64，并记录 mixed 平台期停止状态。

**改动原因**

Cm 运行资源和当前研究下一步发生变化，需要让根级接手入口与任务级状态一致。

## 从 feature/hand-pca-perturbation 合并的历史记录

以下条目保留 feature 分支的根级修改历史；路径和文件名按当时状态记录。

### 2026-08-23 — 启动全量 ARCTIC MANO 数据导出

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: 跨 task / 全局

**文件 / 数据**

- `docs/logs/status_log.md` — 新建根级状态入口，记录 ARCTIC 导出与现有 mixed 训练的并行状态。
- `src/task/correspondence_ptv3_v2/docs/logs/status_log.md` — 记录全量 ARCTIC Stage 2/3 导出正在运行及完成后的核验要求。
- `output/research/arctic_full_mano_v21_20260823/run_export.sh` — 后台顺序执行 Stage 2、Stage 3 的被忽略启动脚本。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage2/arctic_full_mano_v21_20260823` — 全量 ARCTIC MANO Stage 2 输出目录。
- `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_full_mano_v21_20260823` — Stage 2 成功后自动生成的 v2.1 hand-root Stage 3 目录。

**改动原因**

用户确认先把全量 ARCTIC 左右手、stride 1、含 MANO 参数的数据导出到 NAS，并要求后台运行。任务使用物理 GPU 3 的 tmux `arctic_full_mano_v21_20260823`，不覆盖旧 ARCTIC 数据，也不干扰 GPU 1、2 上的 mixed 训练。

### 2026-08-21 — 将六类日志规则统一收回文档模板区

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree (未提交)`
- scope: 全局

**文件**

- `AGENTS.md` — 删除独立的实验、决策和修改记录章节，将其边界与模板统一收回六类文档职责说明，并顺延后续章节编号。
- `docs/logs/modification_log.md` — 记录本次规范统一。

**改动原因**

用户确认六类文档应采用统一结构，避免实验、决策和修改记录被单独拎出形成重复且过强的规范。

### 2026-08-21 — 统一六类日志模板并收敛实验规范

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree (未提交)`
- scope: 全局

**文件**

- `AGENTS.md` — 为六类日志加入统一元信息外壳和最小模板；将运行环境并入当时的 `memory_log.md`，将自主决策和修改记录模板集中到文档职责说明；裁剪 `experiment_log.md` 的必填字段并将详细观察、解释和下一步改为可选。
- `docs/logs/modification_log.md` — 记录本次规范收敛。

**改动原因**

按用户确认降低模板的强制性，减少重复规则，保留科研追溯所需的最小信息，同时给不同 Task 保留正文结构自主性。

### 2026-08-21 — 将递归日志规范拆分为状态、记忆与完整架构事实源

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree (未提交)`
- scope: 全局

**文件**

- `AGENTS.md` — 将 AI 维护文档规范从五类调整为六类，新增 `status_log.md` 与当时的 `memory_log.md`，并规定 `architecture_log.md` 为不依赖 `docs/架构.md` 的完整架构事实源。
- `docs/logs/modification_log.md` — 记录本次全局规范修改。

**改动原因**

根据用户确认，分离当前状态与接手记忆，明确环境、路径、历史遗留和架构事实的归属，避免状态、memory、架构和修改记录互相混写。

### 2026-08-20 — 将 GRAB 大体积数据入口切到 NAS

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- scope: 跨 task / 全局

**文件**

- `dataset/GRAB/data` — 软链到 NAS 上的 GRAB 数据根。
- `docs/logs/repo_notes_log.md` — 记录仓库级 NAS 数据根路径。

**改动原因**

本地根分区已接近满盘，而 GRAB 的大体积只读数据在 NAS 上已有完整副本。把入口改到 NAS 可以释放本地空间，并让后续任务统一引用同一份数据根。

### 2026-08-20 — 将 processed_data 迁到 NAS 并补结果配置摘要

- branch: `feature/hand-pca-perturbation`
- post-commit: `d5510e0`
- scope: 跨 task / 全局

**文件**

- `data/processed_data` — 迁移到 NAS 路径 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data` 的软链。
- `src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.md` — 补充三模型对比对应的训练配置摘要。
- `docs/logs/repo_notes_log.md` — 记录仓库级 processed_data NAS 路径。

**改动原因**

用户要求把本地 `processed_data` 也放到 NAS，并把当前对比结果里对应的训练配置写清楚，方便后续复现和检查配置差异。

### 2026-08-24 — 新增 HOCap 外部测试数据转换入口

- branch: `feature/hand-pca-perturbation`
- post-commit: `working tree`
- scope: 跨 task / 全局

**文件 / 数据**

- `process/HOCap/__init__.py` — 新增 HOCap 处理包入口。
- `process/HOCap/stage3_export.py` — 新增 annotation-only HOCap 到 Ref2Dex Stage 3 的转换器；输出按序列、物体实例和有效手拆分。
- NAS `processed_data/stage3/hocap_subject1_annotation_v1` — 启动 subject_1 外部测试数据转换，不加入训练。

**改动原因**

HOCap 是新增外部数据域，影响仓库级数据处理路径；转换器固定 HOCap MANO PCA45、标准 1538 hand face points和 hand-root 坐标契约，同时保留原始 HOCap 数据不变。

## 2026-08-24 — 完成 HRDexDB 全量 geometry cache manifest

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 全局共享 cache

**文件 / 产物**

- `data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json` — 2088 个有效 episode，object-disjoint train/val/test=`1642/232/214`。
- `src/task/CmDecoder/build_cache.py` — 缺失机器人 C2R 预筛选、schema cache resume 和 all/object-disjoint manifest 命名。

**改动原因**

首次全量导出被单个缺失 `C2R.npy` episode 中断；保留已有完整 cache，通过复用模式快速补写正式 manifest，没有重复生成 2088 个 episode。

## 2026-08-23 — 扩展 HRDexDB 共享 geometry layer 以恢复 Cm candidate 合同

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 共享 cache schema

**文件**

- `src/task/CmDecoder/build_cache.py` — 保留 decoder 的 512 点 task 字段，同时在 geometry layer 增加 4096 稳定物体池、法向和逐帧 5cm candidate mask。
- `docs/logs/architecture_log.md` — 记录共享 HRDexDB geometry 必须区分 4096 pool 与运行时 512 sample。

**改动原因**

Cm 的现有数据合同要求从当前手附近 5cm candidate 中采样物体点；仅保存全表面 512 点会把远离手的点错误纳入 Cm loss。新增字段只扩展共享 geometry，不保存 DenseToken 特征，也不启动全量导出。

## 2026-08-23 — 记录 CmDecoder 多手型 HRDexDB cache 入口

- branch: working tree
- post-commit: HEAD
- scope: 跨 task / 外部数据路径记忆

**文件**

- `docs/logs/memory_log.md` — 记录全量非视频 HRDexDB 副本及四类 CmDecoder cache builder 的稳定路径/命令约定。

**改动原因**

CmDecoder builder 现在可消费 MANO、Allegro-V5、Inspire-DFTP、Inspire-F1；该外部数据路径会影响后续接手者和跨任务实验。

## 2026-08-22 — 重建修复版 DexYCB Stage4 cache

- branch: `oyx`
- post-commit: 未提交
- scope: 跨 task / 共享数据

**文件 / 数据**

- `data/processed_data/stage4/data/dexycb` — 重建 subject-10/right 的 50 条序列、2853 帧共享 Stage4 cache。
- `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/` — 新建可追溯的独立评测 split metadata。
- `docs/logs/{architecture_log,memory_log,modification_log}.md` — 将 DexYCB 路径从“待重建”更新为修复版 active 状态。
- `src/task/Cm/` 的评测配置、实验与日志 — 详见 Cm task modification log。

**改动原因**

用户授权使用已修复的共享 DexYCB adapter 重建数据并执行正式 Cm 测试。

**验证**

50 sequences / 2853 frames / 80.30% active frames；all-finite，object rigid drift max `4.06e-5 mm`，50/50 sequence 可生成 stride≤10 样本。

## 2026-08-22 — 移除失效 DexYCB Stage4 cache

- branch: `oyx`
- post-commit: 未提交
- scope: 跨 task / 共享数据

**文件 / 数据**

- `data/processed_data/stage4/data/dexycb` — 将修复前 489M / 102 files 的失效 cache 移入系统回收站，原路径已不存在。
- 根级与 Cm 任务级日志 — 更新 cache 已移除、split 待重生成和训练 checkpoint 当前进度。

**改动原因**

用户明确要求删除已确认错误的 DexYCB cache。采用可恢复的系统回收站，不影响 raw dataset、旧评估日志或训练输出。

## 2026-08-22 — 修复共享 DexYCB Stage4 坐标适配

- branch: `oyx`
- post-commit: 未提交
- scope: 跨 task

**文件**

- `process/DexYCB/raw.py` — 改为 `xyzw` 四元数、正向 SE(3) 点/法向变换、identity-extrinsic reference camera 坐标和官方 MANO PCA/non-flat mean 解码，并裁掉连续的全零 MANO 前缀。
- `process/DexYCB/stage4_cm.py` — 只枚举当前支持的右手 capture，修正输出 metadata 坐标说明。
- `tests/test_dexycb_raw.py` — 增加四元数、SE(3)、reference camera 和 MANO 无效帧回归测试。
- 根级与 `src/task/Cm/docs/logs/` 文档 — 同步数据合同、旧 cache 风险、无效实验结论与修改记录。

**改动原因**

DexYCB subject-10 评估异常经 raw label、外参和刚体回代互证后确认为共享 adapter 实现错误，旧 cache 不能表示真实手物运动。

**验证**

- `pytest`: 10 passed（DexYCB 新测试 + Cm sequence 相关测试）。
- subject-10 50 条右手 capture 都唯一解析到 reference serial `840412060917`。
- MANO 展开后 21 关节与官方 `joint_3d` 直接对齐平均误差 `0.78 mm`；真实 mug sequence 刚体漂移 `3.6e-5 mm`，51 帧中 43 帧有 5 cm candidate。
- 未重建或覆盖任何数据 cache。

## 2026-08-22 — 补齐 HRDexDB Inspire F1 非视频运动资产

- branch: working tree
- post-commit: HEAD
- 范围: 跨 task / 外部数据

**文件**

- `/home2/wyy/oyx_ws/HRDexDB/v0` — 按白名单下载缺失的 arm、hand、timestamp、C2R、mesh_v2 OBJ 和 compact v1/v2 object pose；未新增视频。
- `docs/logs/{memory_log,repo_notes_log,modification_log}.md` — 记录外部数据状态、代理路径和下载方式。
- `src/task/CmDecoder/docs/logs/{memory_log,status_log,repo_notes_log,modification_log}.md` — 同步 CmDecoder 可用 episode 上限与后续阻塞。

**改动原因**

原下载命令显式排除了 `raw/arm` 且 LFS 同步失败，导致592个目录中只有164个满足 CmDecoder 几何合同。补齐后完整模态交集提升到576组。

## 2026-08-20 — 接入外部 HRDexDB Inspire F1 数据路径

- branch: working tree
- post-commit: HEAD
- 范围: 跨 task

**文件**

- `docs/logs/repo_notes_log.md` — 记录 HRDexDB 数据根、Inspire F1 URDF 和 LFS 下载环境约定。
- `src/task/CmDecoder/` — 新建 frozen-Cm 动作重建任务（详细改动见任务级 modification log）。

**改动原因**

用户要求利用外部 HRDexDB Inspire F1 数据和既有 GRAB-only Cm checkpoint 训练关节角 Decoder；外部数据路径与下载约定具有仓库级复用意义。

## 2026-08-20 — 修正共享 GRAB raw asset 解析

- branch: `oyx`
- post-commit: `HEAD`
- 范围: 跨 task

**文件**

- `process/GRAB/raw.py` — 统一 sequence root 与 subject `v_template` 的相对路径解析，并支持严格失败。
- `process/GRAB/stage4_cm.py` — 正式 Cm Stage4 默认禁止缺失 subject template，写入可追溯 metadata。
- `tests/test_cm_sequence_dataset.py` — 增加 raw layout 回归测试。

**改动原因**

共享 GRAB adapter 的旧拼接逻辑会在 `dataset/GRAB` root 下找错 `tools/subject_meshes`，静默回退平均 MANO；这会污染 Cm/V2 hand geometry。

**影响范围**

共享 GRAB 数据处理；Cm/V2 需重建受影响 cache。

## 2026-08-18 — 建立根级规范文档入口

- branch: 当前工作分支
- post-commit: `HEAD`
- 范围: 全局

**文件**

- `docs/logs/architecture_log.md` — 补充仓库架构、共享数据流和 Task/Cm 关系。
- `docs/logs/repo_notes_log.md` — 归档运行环境、数据路径和文档索引。
- `docs/logs/decision_log.md` — 建立全局决策入口。
- `docs/logs/modification_log.md` — 建立全局修改记录入口。

**改动原因**

按仓库 `AGENTS.md` 将原有单一项目总览整理为根级五类文档入口；不改变代码、数据或实验定义。

**影响范围**

全局文档结构。

## 2026-08-23 — 重整端到端重定向架构说明

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 全局

**文件**

- `docs/logs/architecture_log.md` — 按 correspondence_ptv3_v2 → Cm → CmDecoder 的实际依赖关系，重写端到端 Pipeline、数据/张量契约、训练边界和重定向路径说明。

**改动原因**

用户要求将 DenseToken 提取、Cm 通用表征、目标手专用 Decoder 与最终重定向整理成简明、可供他人理解的全局架构入口。

**影响范围**

仅更新架构文档，不改变代码、数据、模型 checkpoint 或实验定义。

## 2026-08-23 — 细化端到端模块与重定向执行架构

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 全局

**文件**

- `docs/logs/architecture_log.md` — 增加 Stage 3/4 数据构造、PTv3 DenseToken 特征、Cm Slot Attention/gate/object-flow、CmDecoder 两种解码器、FK/q 拟合和逐帧重定向顺序的模块级说明。

## 2026-08-23 — 下载 HRDexDB 全量非视频原始数据

- branch: working tree
- post-commit: 未提交
- scope: 跨 task / 外部数据

**文件**

- `/home2/wyy/oyx_ws/HRDexDB/v0_nonvideo` — 新建独立 worktree，下载 human/MANO、Allegro-V5、Inspire-DFTP、Inspire-F1、assets、object poses 和 metadata；明确排除所有 `vid/` 文件。
- `process/HRDexDB/lfs_batch_proxy.py` — 新增本地 Git-LFS batch 代理，修复镜像返回的不可解析 CDN hostname 后完成下载。

**改动原因**

用户要求下载 HRDexDB 其余数据。完整仓库视频约 1.15 TB，而 Cm geometry cache 不消费视频，因此按已确认的训练输入边界下载全量非视频模态；最终完整性检查确认 0 个 LFS pointer、0 个视频文件。

**改动原因**

用户要求在原有总览基础上进一步说明具体架构，使读者能够理解各阶段的输入、输出、冻结关系和目标手重定向过程。

**影响范围**

仅更新全局架构文档，不改变代码、数据、模型 checkpoint 或实验定义。
## 2026-08-23 — 优化共享 HRDexDB candidate cache 构建

- branch: 当前工作分支
- post-commit: 未提交
- scope: 跨 task / 全局共享 cache

**文件**

- `src/task/CmDecoder/build_cache.py` — 用 `cKDTree` 进行精确 5 cm 半径候选查询，保持共享 geometry schema 不变。
- `src/task/Cm/docs/logs/status_log.md` — 记录全量 cache 后台运行 PID 和日志。

**改动原因**

candidate mask 构建由逐帧全量距离张量改为半径邻域查询；单 episode CPU 探针由约 192.6 s 降至约 37.4 s。全量构建已按 4 workers、单线程 BLAS 启动。
