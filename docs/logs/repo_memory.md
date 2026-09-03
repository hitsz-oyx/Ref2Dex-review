# Ref2Dex 仓库记忆

- scope: root
- last_updated: 2026-09-03
- last_verified: 2026-09-03
- related: [活动记录](activity_log.md)、[架构记录](architecture_log.md)、[历史修改](modification_log.md)

## V1.2.15 — 活动时间线与版本指针收口

- category: governance / documentation
- status: active
- last_verified: 2026-09-01
- fact: 新活动统一写入最近作用域的 `activity_log.md`；历史 `modification_log.md` 只读保留，`status_log.md` 不再作为规范入口。
- fact: `docs/current_versions.yaml` 是当前版本指针，只记录各作用域的 `modification_version`；运行 manifest 不再记录 guidance/plan 版本字段。
- fact: BaseRunner 负责写入配置、`metadata.json`、`run_manifest.json` 和运行日志；不再自动生成标准 `summary.json`。训练/评估终态、最后 step/epoch、best metric、checkpoint 及失败/停止原因由最近作用域的 `activity_log.md` 登记；历史或 Task 专属 summary 可保留。
- source / anchor: [`AGENTS.md`](../../AGENTS.md)、[`docs/current_versions.yaml`](../current_versions.yaml)、[`docs/目录规范.md`](../目录规范.md)。

## V1.2.14 — AGENTS 通用规则与仓库约定分层

- category: governance / documentation
- status: active
- last_verified: 2026-09-01
- fact: `AGENTS.md` 的第 5 节只承载通用交接与版本规范；Ref2Dex 特有的路径、运行和目录路由集中到文档末尾的第 8 节，避免通用规则与仓库布局混排。
- source / anchor: `AGENTS.md` §5、§8。

## V1.2.13 — 根 Component 目录已撤出

- category: governance / path
- status: active
- last_verified: 2026-09-01
- fact: 仓库根不再存在 `components/`；当前 Task 不依赖组件清单。`tools/researchctl.py` 不设置默认 manifest 根，只在调用方显式给出外部路径时执行通用校验。
- compatibility: `src/base/` 内现存协议代码暂作为共享基础设施兼容代码保留；本次没有扩大为公共运行时删除。
- source / anchor: 根目录扫描、`tools/researchctl.py`、`docs/目录规范.md`。

## V1.2.12 — Task 数据入口回到根空间

- category: governance / operation
- status: active
- last_verified: 2026-09-01
- fact: CmDecoder 与 correspondence_ptv3_v2 不再维护 Task-local `components/`、`data/` 或 `registry/`；真实数据/cache 入口统一由根级 `data/`、`dataset/` 和 `third_party/` 承载，配置直接引用这些路径。根组件示例随后在 V1.2.13 删除。
- source / anchor: 两个 Task 的 `config.py`、`docs/plan/V1.md`、根 `.gitignore`。

## V1.2.11 — Task-local 外部资产入口

- category: path / convention
- status: active
- last_verified: 2026-08-31
- fact: 外部资产按实际归属放在 `src/task/<Task>/assets/` 并由 Task 配置直接引用；根 `assets` 不再承载新资产，只允许被忽略的历史兼容软链接。Cm 的 DenseToken canonical 入口为 `src/task/Cm/assets/checkpoints/densetoken/`，真实文件仍在 `src/task/Cm/densetoken_ckpt/`。
- source / anchor: `AGENTS.md`、`.gitignore`、`src/task/Cm/src/config.py`、`src/task/Cm/docs/logs/architecture_log.md`。

## 2026-08-31 — V1.2.9 隔离试验线撤销

- category: governance / legacy
- status: active
- last_verified: 2026-08-31
- fact: `src/task/CmComponent/`、Cm Task 内对应的清单和适配器入口已删除；Cm 的模型、数据、cache、训练配置和历史运行产物保留。下方关于该隔离试验线的条目仅作为历史记录，不再是当前入口。
- source / anchor: `src/task/Cm/docs/logs/modification_log.md`、`AGENTS.md`。

## 历史 V1.2.8 — Component 声明层/运行层审计规则

- category: `architecture` / `governance`
- fact: Task `components.json`、Task config、单组件 manifest 和 PipelineSpec 分别负责库存、选择、端口合同和拓扑；`tools/researchctl.py check-task-config` 是只读一致性闸门。
- fact: Task-specific Python Pipeline 可以继续作为执行实现，但必须在启动时证明与 PipelineSpec 拓扑/端口一致；YAML 不自动成为执行引擎。
- fact: 训练 Component 的共享生命周期协议为 `TrainableComponent`；运行时首批 Tensor 合同验证和后续 fast path 的边界写入 `AGENTS.md`。
- source / anchor: `AGENTS.md` §5.1、`src/base/registry.py`、`src/base/component.py`。

本文只记录可随仓库迁移的长期事实。解释器绝对路径、GPU、代理、NAS 挂载点和仓库软链的本机目标写入同目录下被 Git 忽略的 `machine_memory.md`。

## 2026-08-24 — 仓库与机器记忆分离

- category: convention
- status: active
- last_verified: 2026-08-24
- fact: 根级和 Task 级记忆统一拆为受版本管理的 `repo_memory.md` 与本机私有的 `machine_memory.md`。凡可用仓库相对路径表达的入口、schema、命名约定、历史遗留和兼容性事实必须留在 repo memory；只有随机器变化的绝对路径、环境和挂载映射进入 machine memory。
- source / anchor: `AGENTS.md`、`.gitignore`。

## 2026-08-22 — DexYCB Stage4 修复版 cache

- category: path / legacy / pitfall
- status: active
- last_verified: 2026-08-22
- fact: `data/processed_data/stage4/data/dexycb` 已用修复 adapter 重建为 subject-10/right 的 50 条序列、2853 帧，active-frame ratio 80.30%；正式评测 split 为 `data/processed_data/stage4/splits/dexycb_subject10_fixed_v1/split.json`。修复前 cache、旧 `dexycb_v1` split 和旧评测输出不得复用。
- source / anchor: `process/DexYCB/raw.py`、`process/DexYCB/stage4_cm.py`、Cm EXP-010。

## 2026-08-23 — HRDexDB 仓库内规范入口

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: HRDexDB 非视频数据的仓库内规范根为 `dataset/HRDexDB/v0_nonvideo`；机器人资产和读取 helper 分别位于 `dataset/HRDexDB/assets`、`dataset/HRDexDB/hrdexdb_contact_heatmaps`。`dataset/HRDexDB/` 整体被 Git 忽略。旧的仓库外路径只允许作为本机兼容 symlink，不是配置或文档的规范入口。
- source / anchor: `.gitignore`、`src/task/CmDecoder/config.py`、`src/task/CmDecoder/build_cache.py`。

## 2026-08-23 — HRDexDB geometry cache 的 Cm 输入边界

- category: convention / pitfall
- status: active
- last_verified: 2026-08-23
- fact: `cmdecoder_layered_v4` geometry layer 需要同时保存 legacy decoder 的 512 点字段和 Cm 使用的 `obj_points_pool_world [T,4096,3]`、`obj_normals_pool_world`、`obj_candidate_mask_5cm [T,4096]`。缺少 candidate mask 的旧 512 点 smoke cache 不能直接用于 Cm 训练；DenseToken feature 禁止写入该层。
- source / anchor: `src/task/CmDecoder/build_cache.py`、`src/task/Cm/dataset/hrdexdb.py`、Cm V1.2 candidate contract。

## 2026-08-23 — CmDecoder 多手型 cache builder

- category: path / convention
- status: active
- last_verified: 2026-08-23
- fact: `src/task/CmDecoder/build_cache.py` 支持 `human,allegro_v5,inspire_dftp,inspire_f1`；数据入口为 `dataset/HRDexDB/v0_nonvideo`，机器人 URDF 根为 `dataset/HRDexDB/assets/robots`。全量构建使用 `--robot-types human,allegro_v5,inspire_dftp,inspire_f1 --episodes 0`，输出应写到独立 cache root。
- source / anchor: `src/task/CmDecoder/build_cache.py`。

## 2026-08-24 — HRDexDB 全量 geometry cache manifest

- category: path / convention
- status: active
- last_verified: 2026-08-24
- fact: `data/processed_data/cm_decoder/hrdexdb_all_v1/v4/selection_all_object_disjoint_seed42.json` 是四手型 HRDexDB 的正式 object-disjoint manifest，共 2088 个有效 episode，train/val/test=`1642/232/214`。缺少机器人 `C2R.npy` 的 16 个 episode 被 selector 排除；cache 不包含 DenseToken 输出。
- source / anchor: `src/task/CmDecoder/build_cache.py`、正式 manifest。

## 2026-08-20 — 共享 GRAB raw asset 约定

- category: convention / pitfall
- status: active
- last_verified: 2026-08-24
- fact: `process/GRAB/raw.py` 被多个 Task 共用。GRAB sequence root 与 `tools/subject_meshes` 可能位于不同层级，调用方不得假设所有相对 asset 都直接位于传入 root 下。正式 Cm/V2 Stage4 强制使用 subject-specific MANO template；缺失时必须显式失败，旧流程只有通过兼容开关才允许回退。
- source / anchor: `process/GRAB/raw.py`、`process/GRAB/stage4_cm.py`。

## 2026-08-30 — 研究目录与共享 checkpoint 迁移入口

- category: path / convention
- status: active
- last_verified: 2026-08-30
- fact: 新研究实验使用 `src/task/<Task>/research/<experiment>/`，产物使用实验内的
  `output/<run_id>/`；正式训练/评估使用 `outputs/<task>/<run_id>/`。根 `output/`、`results/` 和
  `result/` 均为历史兼容路径，不得新增内容。
- fact: `assets/checkpoints/densetoken` 是指向旧 `src/task/Cm/densetoken_ckpt` 的过渡软链接；
  现阶段旧路径仍被多个 Task 引用，真实文件迁移必须在相关运行结束后完成。
- source / anchor: `AGENTS.md`、`docs/目录规范.md`、`assets/README.md`。

## 2026-08-30 — V1.2.1 版本线与 Task Component 注册约定

- category: `governance` / `architecture` / `path`
- status: active
- last_verified: 2026-08-30
- fact: 版本层级统一为 `Vn`（用户确认的大版本）→ `Vn.m`（必须存在并定稿的 plan）→ `Vn.m.k`（细分操作/并行实验）；指导只由用户更新，架构快照只在用户明确指令下更新。`V1.2.1` 是治理收口操作，当前活动细分操作为 `V1.2.3`，对应父计划 `src/task/CmComponent/docs/plan/V1.2.md` 和已确认的操作计划 `V1.2.3.md`。
- fact: Task 专属 Component 的 canonical 入口是 `src/task/<Task>/components/`，根 `components/` 仅保留跨 Task 组件和示例；旧 `components/ref2dex/*` 允许保留相对软链接兼容。每个 Task 的 `components.json` 是运行选择清单，不替代单组件 `component.yaml`。
- fact: Task 的仓库内数据/cache/资产入口写在 `src/task/<Task>/registry/data_paths.json`，Task 内 `data/` 可用相对软链接便于查找；机器特有绝对路径放入被 Git 忽略的 `external_paths.json`。
- source / anchor: `AGENTS.md`、`docs/modification_policy.md`、`src/task/Cm/docs/plan/V1.2.md`。

## 2026-08-30 — V1.2.2 CmComponent 隔离 Task

- category: `architecture` / `component` / `path`
- status: active
- last_verified: 2026-08-30
- fact: `src/task/CmComponent/` 是旧 Cm 的可回滚 Component 试验边界。其 `DenseEncoderComponent`、`CmHeadComponent` 和 `CmComponentPipeline` 是真实 `Component` 入口；`CmComponentRunner` 只替换模型构造，复用旧 Cm 的数据、loss、指标和 checkpoint 生命周期。
- fact: `src/task/CmComponent/components/pipeline/pipeline.yaml` 是可读的组件拓扑声明；运行时由 `component_factory.py` 按 Task `components` 清单校验 manifest、版本和 `entrypoint_kind` 后实例化角色，替换组件不需要改旧 Cm。
- fact: 新 Task 使用 `V1.2` plan，当前细分操作为 `V1.2.2`；旧 `src/task/Cm/` 的代码、数据、cache、checkpoint、output 和运行进程不因该 Task 改写。真实 checkpoint parity 通过前不启动新 Task 正式训练。
- source / anchor: `src/task/CmComponent/docs/plan/V1.2.md`、`src/task/CmComponent/components/components.json`、`src/task/CmComponent/src/runner.py`。

## 2026-08-31 — V1.2.3 CmComponent 独立组件运行时

- category: `component` / `experiment` / `path`
- status: active
- fact: `src/task/CmComponent/` 的 encoder、slot-flow head、pipeline、dataset 和 runner 均为 Task-local 实现，不导入旧 `src/task/Cm`；组件清单和三个 manifest 使用 `2.0.0`，由 `component_factory.py` 按 role/id/version 选择。
- fact: CmComponent 的 smoke 数据 schema 为 `cm_component.synthetic@1.0`，训练/评估输出使用 `outputs/cmcomponent/<run_id>/`，run manifest 记录 `V1.2`、`V1.2.3`、seed、数据合同和组件版本。smoke 仅证明 wiring/training lifecycle。
- source / anchor: `src/task/CmComponent/docs/plan/V1.2.3.md`、`src/task/CmComponent/configs/active/smoke.yaml`、`outputs/cmcomponent/cm_component_smoke_20260831_111831/run_manifest.json`。

## 2026-08-31 — V1.2.4 可插拔职责与汇报规范

- category: `governance` / `component` / `data`
- status: active
- fact: 出现两个候选实现的职责必须提升为 Component；每个运行 role 必须恰好选择一个实现；可发现 Component 使用固定 `component.yaml` manifest；新增/替换组件必须同步清单和合同测试。Pipeline/Runner/Dataset 只依赖稳定的 `Component / Artifact / Contract` 接口，YAML/JSON 只选择 role/id/version，不承载执行逻辑。该规则已写入根 `AGENTS.md`。
- fact: `CmComponent` 的 `coordinate_transform` 角色有 `HandRootFrameComponent` 和 `ObjectPoseFrameComponent` 两个实现，统一将 world 数据转换为 canonical CmBatch；运行 manifest 记录实际选择和 pose source。
- fact: `run_manifest` 的输入追溯只收集 registry、manifest、split、cache、checkpoint 和路径字段；Component ID/version 等 provenance 标识不会被误解析成文件路径。
- fact: CmComponent 的运行时 role 默认是 required；`component_factory.py` 对缺失/重复/版本或 manifest 不匹配显式失败，默认类 fallback 只可由调用方明确关闭 required 检查。
- fact: CmComponent eval 的默认 provenance 目录是 checkpoint 所在 run 目录；根目录旧 eval 副本仅作为 `_legacy_eval_root_20260831/` 归档，不是新入口。
- source / anchor: `AGENTS.md` §5.1、`src/task/CmComponent/components/coordinate_transform/`、`src/task/CmComponent/dataset/factory.py`。

## 2026-08-31 — V1.2.5 递归 Component 选择

- category: `architecture` / `component` / `provenance`
- status: active
- fact: Component 选择可以在父条目的 `children` 中递归声明；`resolve_task_component(..., parent_role=...)` 沿父 role 路径解析，当前 CmComponent Pipeline 使用 `pipeline -> dense_encoder/cm_head`。
- fact: `run_manifest.json` 保留兼容的扁平 `components`，并新增去重后的 `component_tree`；子组件的 manifest、版本和参数前缀随树记录。
- source: `src/task/CmComponent/src/component_factory.py`、`src/base/run_manifest.py`、`src/task/CmComponent/src/config.py`。

## 2026-08-31 — V1.2.6 嵌套 Component 配置唯一来源

- category: `architecture` / `component` / `provenance`
- status: active
- fact: CmComponent 运行配置只声明根 Component；子 Component 只在父条目的 `children` 中选择。当前根为 `coordinate_transform` 与 `pipeline`，后者包含 `dense_encoder`、`cm_head`。
- fact: `src/base/run_manifest.py` 以递归 `component_tree` 为规范来源，自动生成 child-first、去重的扁平 `components` 兼容视图；配置不再重复维护两份选择清单。
- fact: 若父 Component 的外部 Contract、端口、shape、单位、坐标或 checkpoint 语义保持不变，内部继续拆分不要求外层 Pipeline 同步；合同不兼容时必须升级父 Component 版本并修改耦合处。
- source / anchor: `AGENTS.md` §5.1、`src/task/CmComponent/src/component_factory.py`、`src/base/run_manifest.py`、`src/task/CmComponent/configs/active/smoke.yaml`。

## 2026-08-31 — V1.2.7 Component 晋升边界

- category: `governance` / `architecture`
- status: active
- fact: Component 不是文件拆分单位。实现可分为 helper、Task-local module、可发现 Component 三层；只有两个需要独立维护/比较/回滚的实现，或独立替换/资源/生命周期、跨 Task 复用等边界才晋升。
- fact: 小开关仅适用于有限局部参数且完整 Contract、资源和生命周期不变；坐标、单位、GT、schema、split、checkpoint 解释或候选算法变化不能用开关隐藏，需按并存候选拆 Component，或为单一路线迁移升级所属 Contract/schema/版本并提供 adapter。
- source / anchor: `AGENTS.md` §5.1、`.agents/skills/modular-component-runtime/SKILL.md`。
