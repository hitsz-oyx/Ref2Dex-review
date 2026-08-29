# 全局 AI 修改记录

- scope: root
- last_updated: 2026-08-29
- related: [架构记录](architecture_log.md)、[仓库记忆](repo_memory.md)、[决策记录](decision_log.md)

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
