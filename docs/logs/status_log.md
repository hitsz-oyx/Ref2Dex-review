# Ref2Dex 当前状态

- scope: root
- last_updated: 2026-09-01
- last_verified: 2026-09-01
- current_version: V1.2.14
- current_plan: AGENTS 规范分层调整（用户直接批准；不涉及 Task plan）
- related: [仓库记忆](repo_memory.md) / [修改记录](modification_log.md)

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
- 证据与相关文档: 全量 pytest `311 passed, 3 skipped`；[仓库修改记录](modification_log.md)、[架构记录](architecture_log.md)、[correspondence 状态](../../src/task/correspondence_ptv3_v2/docs/logs/status_log.md)、[Cm 状态](../../src/task/Cm/docs/logs/status_log.md)。
- 2026-08-30 框架更新: `BaseRunner` 在 train/eval 启动时生成不会覆盖历史尝试的 `run_manifest.json`（续跑/评估使用时间戳文件）；`ref2dex.run.v1` 记录 Git 状态、输入 manifest/cache 引用、文件基本信息、数据合同、seed 和初始 checkpoint，不计算加密 hash。Cm loader 对坐标根做全量一致性检查。

## 2026-08-30 目录规范兼容迁移

- Cm 的 t-SNE 诊断已迁入 `src/task/Cm/research/tsne_slots/`；旧 `tsne_slots.py` 和模块调用方式保留为软链接/兼容入口。
- 新增统一的研究实验包格式、`experiment.yaml`、研究输出规则、Manifest 职责说明和 AI 交接清单。
- 新增 `assets/checkpoints/densetoken` 过渡软链接，实际 checkpoint 文件仍保留在旧目录，未影响当前运行。
- 本次没有移动或删除 `data/`、`output/`、`outputs/`、checkpoint 或历史 `results/`；根 `output/` 仅标记为历史兼容路径。
