# 全局 AI 自主决策记录

## 2026-09-01 — V1.2.14 分离 AGENTS 通用规则与仓库约定

- branch: `feature/modular-component-runtime`
- version: `V1.2.14`（根级治理细分操作；不涉及 Task plan）
- category: `governance` / `documentation`
- approval: user-approved（用户要求重组 `AGENTS.md` 第 5 节）

**实际选择**

- 将第 5 节改为“通用交接与版本规范”，保留 `5.1` 完成汇报/规范反馈和 `5.2` 版本线/AI 行为。
- 将原第 5 节的 Ref2Dex 路径、运行目录和自动归类规则整体移到文档末尾的第 8 节，并设为 `8.1`。
- 不改变任何规则内容、Task 代码、实验配置或架构快照，仅调整规范的作用域和阅读顺序。

**理由 / 回滚**

通用规则可以迁移到其他仓库，路径路由只能在 Ref2Dex 语境下解释；分层后减少误把本仓库路径当作通用规则的风险。回滚只需恢复 `AGENTS.md` 的章节顺序和标题，其他运行状态不受影响。

## 2026-09-01 — V1.2.13 删除根 Component 目录

- branch: `feature/modular-component-runtime`
- version: `V1.2.13`（根级治理细分操作；不涉及 Task plan）
- category: `governance` / `operation` / `documentation`
- approval: user-approved（用户明确要求“根里面的component也删了”）

**实际选择**

- 将根 `components/` 连同 README、通用 identity/scale 示例、pipeline 示例和缓存彻底删除。
- `tools/researchctl.py` 不再内置根目录默认值；保留通用只读命令时，调用方必须显式传入外部 manifest 路径。
- 保留 `src/base/` 的通用协议兼容代码，本次不扩大到共享基础设施删除。

**理由 / 回滚**

根示例已经没有 Task 消费者，继续保留会让目录规范和工具误导后续 Agent 认为仓库仍维护公共组件入口。若需回滚，可从 Git 恢复根目录及示例，并恢复工具默认 root；Task 数据、配置和训练状态不受本次操作影响。

## 2026-09-01 — V1.2.12 撤出 CmDecoder 与 correspondence 的 Task 入口

- branch: `feature/modular-component-runtime`
- version: `V1.2.12`（两个 Task 的 V1 清理计划已 final）
- category: `governance` / `operation` / `documentation`
- approval: user-approved（用户确认彻底删除 Component，删除 Task data/registry，并让数据回到根空间）

**实际选择**

- 删除两个 Task 的 `components/`、`data/`、`registry/`，同时删除根 `components/ref2dex/` 对应兼容入口。
- 不移动真实数据；CmDecoder 和 correspondence 配置直接引用根级 `data/`、`dataset/`、`third_party/` 或现有 Task 校准文件。
- 根通用组件示例和基础协议代码暂不删除，因为本次范围只针对两个 Task 的 Task-local 入口。

**理由 / 回滚**

Task-local 清单和路径软链接已经不再作为当前工作流，保留它们会继续制造双重入口。回滚可恢复本次删除的目录、配置字段和根兼容入口；真实数据未被触碰。

## 2026-08-31 — V1.2.11 按 Task 归属外部资产

- branch: `feature/modular-component-runtime`
- version: `V1.2.11`（沿用 `src/task/Cm/docs/plan/V1.2.md`）
- category: `governance` / `operation`
- approval: user-approved（用户明确要求将根 `assets/` 下放到具体 Task，并加入 `.gitignore`）

**实际选择**

- 以实际消费者作为资产归属；当前根 `assets/` 内容只有 Cm DenseToken，因此 canonical 入口改为 `src/task/Cm/assets/`。
- 保留 `assets -> src/task/Cm/assets` 的被忽略兼容软链接，并在 Task-local 入口下保留指向旧 `densetoken_ckpt` 的软链接；不移动运行中的大型 checkpoint。
- `.gitignore` 忽略根兼容入口和所有 Task-local 资产实体，仅允许资产 README 用于说明。

**理由 / 回滚**

按 Task 归属能避免一个看似共享的根目录掩盖真实依赖边界，同时旧配置和当前 Cm 进程仍可通过兼容入口解析。回滚只需恢复 Cm 配置到根入口并移除新软链接，不触碰 checkpoint 文件。

## 2026-08-31 — V1.2.8 收口声明层与运行层

- scope: root / Component registry、Contract 与训练生命周期
- version: `V1.2.8`（plan: `V1.2`, final）
- category: `architecture`、`code`、`governance`
- approval: user-approved（用户要求按审查意见继续改造）

**决策**

将四个来源的职责固定为：Task `components.json` 管库存，Task config 管选择，单组件
`component.yaml` 管端口/版本/入口合同，`PipelineSpec` 管拓扑。新增只读
`check-task-config` 并在 CmComponent 构造时复核；Task-specific Python Pipeline 暂为执行源，
启动时必须与 YAML 节点、子选择和端口一致，暂不引入通用 YAML executor。

训练参数组件统一采用 `TrainableComponent` 的 build、参数前缀和 `id@version` checkpoint
namespace；真实 Tensor 只在数据边界/首批严格校验，后续采用等价的轻量路径。

**理由 / 影响**

这样替换候选实现时，声明层与运行层不会静默漂移，同时不要求当前高性能 Task Pipeline 立即
重写成通用解释器。checkpoint identity 和参数前缀可在后续并行 Component 版本中保持可审计。

**可逆性**

删除 registry 闸门、启动校验和 Trainable 协议即可回到 V1.2.7；不影响旧 Cm、历史 checkpoint、
数据/cache、output 或运行中的训练进程。

## 2026-08-30 — V1.2.1 采用版本线和 Task-local Component 边界

- scope: root / 仓库治理与组件发现
- version: `V1.2.1`（plan: `V1.2`, final）
- category: `governance`、`architecture`
- approval: user-approved（用户回复“就按照你想的”）

**决策**

将 `Vn` 作为用户确认的大版本、`Vn.m` 作为必须有 final plan 的小版本、`Vn.m.k` 作为细分操作和并行实验编号；Task 专属 Component 的 canonical 目录统一放在 `src/task/<Task>/components/`，根 `components/` 只保留跨 Task 组件和示例，旧路径改为软链接。每个 Task 同时维护 `components.json` 和 `registry/data_paths.json`，运行配置显式列出组件角色、版本和参数前缀。

**理由 / 影响**

该边界让研究代码、组件合同、数据入口和每次运行的追溯关系在 Task 内可见，又不破坏已有命令和路径；版本号可承载并行对照实验，架构快照只在用户明确要求时更新。

**可逆性**

可将 canonical 文件移回旧目录并删除新清单，恢复原目录结构；不触碰数据、cache、checkpoint、output 或历史 manifest。

## 2026-08-28 — 采用任务无关的 Component/Artifact/Contract/Registry 原型

- scope: root / 通用运行时
- anchor: working tree / 2026-08-28

**未指定点**

用户希望代码可热插拔并适配 Ref2Dex 以外的任务，但没有要求立即迁移现有 Task 或更换当前训练框架。

**实际选择**

以 `Component` 作为唯一通用扩展抽象，使用 manifest 声明 capabilities、输入输出端口和生命周期状态；
暂不把 `encoder`、`decoder`、`dataset` 固化为框架一级类别，也不移动现有 Task 目录。第一版只提供
manifest 解析、registry 发现、端口合同检查和 `researchctl list/check/graph`。

**选择理由与影响**

该抽象不依赖具体科研领域，可由现有 `BaseRunner` 逐步适配，同时让新组件通过少量元数据自动进入索引。
合同检查先覆盖 type/shape/dtype/unit，领域语义留在可扩展 constraints 中，避免热插拔牺牲坐标系和数据
schema 的科学不变量。

**可逆性 / 是否需要用户确认**

代码可逆；现有训练入口、配置和 checkpoint 未改变。后续将现有 Task 注册为 active/reference 前，需单独
确认迁移边界和兼容性。

## 2026-08-28 — 以只读 manifest 接入当前三条主线

- scope: root / 通用运行时
- anchor: working tree / `components/ref2dex/`

**未指定点**

用户确认继续推进热插拔原型，但没有要求移动现有 Task 文件或改变训练入口。

**实际选择**

为 `correspondence_ptv3_v2`、`Cm`、`CmDecoder` 增加只读 `component.yaml`，声明稳定输入输出合同和
active 状态；不导入 entrypoint，不自动执行训练。

**选择理由与影响**

先验证通用 registry 能正确描述现有主线，再决定 runtime adapter 和执行语义，避免插件化过程改变既有
实验。manifest 允许未来自动生成索引，同时保留原目录和 checkpoint 兼容性。

**可逆性 / 是否需要用户确认**

完全可逆；删除 manifest 不影响原有代码。后续从只读索引升级为可执行 adapter 前，需要重新确认每个
Task 的输入输出和 checkpoint 语义。

## 2026-08-28 — 将入口解析限制为显式校验并提供安全 dry-run

- scope: root / 通用运行时
- anchor: branch `feature/modular-component-runtime` / working tree

**未指定点**

用户要求继续推进热插拔，但没有授权自动启动现有训练或实例化真实 Task 的副作用运行时。

**实际选择**

registry 的发现、list 和 describe 保持纯 manifest 操作；只有 `check --resolve-entrypoints` 或
未来明确的执行器才导入 entrypoint。`researchctl run` 当前只接受 `--dry-run`，执行拓扑和合同
检查后返回，不加载 checkpoint、不创建 dataloader。

**选择理由与影响**

这样可以尽早发现入口拼写错误，同时避免“列出组件”触发 CUDA、数据路径或模型权重加载；示例
组件提供可测试的最小 execute 闭环，真实 Task 仍保留原入口。

**可逆性 / 是否需要用户确认**

可逆；未来增加 runtime adapter 时需要逐个确认配置、checkpoint、资源和副作用边界。

## 2026-08-28 — Cm pilot 采用外部 inference adapter

- scope: root / `components/ref2dex/cm`
- anchor: branch `feature/modular-component-runtime` / working tree

**未指定点**

用户只指定首个真实 pilot 使用 Cm，并要求其他执行边界暂不改变；没有要求移动 `src/task/Cm` 或把训练接入通用 pipeline。

**实际选择**

增加独立的 `ref2dex.cm.inference.v1` manifest 和 `CmInferenceComponent`，包装现有 `CmFlowModel`，以 Artifact 映射到单步 inference batch；checkpoint 只能通过显式 `from_checkpoint` 加载。原 `ref2dex.cm.v1` runner manifest 保持不变，pilot manifest 单独声明 normals、valid mask 和可选时间间隔，输出保留 `representation` 与米制 `object_flow`。

**选择理由与影响**

这样能先验证真实 Cm 张量接口和 Artifact 传递，同时不触碰 CmActionRunner 的训练、dataloader、DDP、wandb 或 checkpoint 保存语义。adapter 只做形状/类型 fail-fast，科研指标仍由原 Runner 负责。

**可逆性 / 是否需要用户确认**

可逆；删除 wrapper 或恢复 manifest entrypoint 即可回到只读索引。若要接入真实 checkpoint 运行，仍需单独确认资源路径和执行权限。

## 2026-08-22 — DexYCB 聚合姿态统一使用 reference-camera frame

- scope: 跨 task 共享数据处理
- anchor: branch `oyx` / 2026-08-22

**未指定点**

DexYCB sequence-level `pose.npz` 没有在仓库中携带独立 schema 文档，但实测其 object/MANO 聚合姿态都与外参为单位阵的 master camera 逐帧 label 一致。

**实际选择**

将唯一 identity-extrinsic capture serial 作为 reference camera，直接使用 `pose_y`/`pose_m` 的该坐标；非 reference `view_serial` 直接报错。MANO 按官方 45 维 PCA basis 和 non-flat mean 解码。全零 MANO 标注仅允许作为前缀裁掉，内部断裂直接报错。

**选择理由与影响**

这与本地官方 3x4 label 和 calibration 可数值互证，可防止 object/MANO 被放入不同坐标系。对时间内部缺帧不做压缩，避免伪造等间隔 stride。

**可逆性 / 是否需要用户确认**

代码可逆；旧 cache 保留但标记 deprecated。该选择是对已确认数据 bug 的修复，不改变用户指定的研究目标，无需额外确认。

## 2026-08-23 — HRDexDB 迁移期间保留旧路径 symlink

- scope: root / 跨 task 本地数据路径
- anchor: working tree / 2026-08-23

**未指定点**

用户要求将非视频 HRDexDB 迁入仓库 `dataset/`，但迁移时全量 cache builder 正在通过旧绝对路径读取 2103 个 episode。

**实际选择**

短暂停止 builder 主进程和四个 worker，原子移动 `v0_nonvideo` 后在旧位置建立指向新规范根的 symlink，再恢复全部进程；小体积 robots assets 和自包含 helper 复制到新入口。

**选择理由与影响**

避免停止并丢弃长时间 cache 进度，同时让未来配置只依赖仓库内规范路径。旧 symlink 仅是兼容层，不再是文档规范入口。

**可逆性 / 是否需要用户确认**

可通过切回旧目录恢复；用户已明确授权非视频数据迁移。

## 2026-08-30 — 统一研究产物与共享资产路径，先采用兼容软链接

- scope: 全局仓库治理 / Task:Cm 研究目录 / 共享 checkpoint 入口
- approval: user-approved
- branch: `feature/modular-component-runtime`

**实际选择**

- 独立研究脚本迁入 `src/task/<Task>/research/<experiment>/`，每个实验包含
  `README.md`、`experiment.yaml` 和被忽略的 `output/<run_id>/`；旧入口暂时保留软链接。
- 正式训练和评估继续使用 `outputs/<task>/<run_id>/`；根级 `output/` 只作为历史兼容目录，禁止新增内容。
- 预训练模型的规范入口为 `assets/checkpoints/<asset_id>/`。在当前训练未结束前，新增
  `assets/checkpoints/densetoken` 指向旧 `src/task/Cm/densetoken_ckpt`，不移动大型 checkpoint。
- 不再创建 `results/` 或 `result/`；结论进入作用域内 `experiment_log.md`。

**理由与回滚**

目录规则需要立即稳定，但当前工作树有未提交修改且存在多卡训练。先使用软链接可以让新代码采用规范入口，
同时不改变打开中的文件和旧配置；待运行结束后可将真实文件移动到 `assets/`，再删除兼容层。
回滚只需恢复旧默认路径并删除新增软链接，不触碰数据和训练输出。

## 2026-08-31 — V1.2.6 以嵌套树作为 Component 选择唯一来源

- scope: `src/task/CmComponent` / nested component composition
- version: `V1.2.6`（沿用父计划 `V1.2`）
- category: `architecture`、`component`、`provenance`
- approval: user-approved（用户明确回复“可以，你直接修改吧”）

**决策**

运行配置只声明根 Component；父 Component 的子选择统一放在 `children`，不再把同一
`dense_encoder`/`cm_head` 同时登记在根级和 `pipeline.children`。`component_tree` 是运行追溯的
规范嵌套视图，扁平 `components` 由系统按 child-first 顺序自动生成，服务旧消费者。

**理由 / 影响**

消除两份选择清单的同步耦合。以后在 `cm_head.children` 内拆分 tokenizer、slot attention 或
decoder 时，只要 `cm_head` 对外的 `Contract`、端口和参数语义不变，外层 Pipeline 的 Python
和根配置无需同步改写；若合同不兼容，则必须升级父 Component 版本并调整耦合处。

**可逆性**

恢复 V1.2.5 的根级重复选择、回退 `run_manifest.py` 的原始扁平读取即可回滚；旧 Cm、旧
checkpoint、历史 run manifest、用户指导和架构快照不受影响。

## 2026-08-31 — V1.2.7 采用三层 Component 晋升边界

- scope: root / Component 治理与内部模块组织
- version: `V1.2.7`（沿用父计划 `V1.2`）
- category: `governance`、`architecture`
- approval: user-approved（用户明确要求区分小开关、内部模块和可插拔 Component，并保持解耦规范）

**决策**

将实现分为 `helper`、Task-local module、可发现 Component 三层。文件数量不决定 Component 数量；
只有两个需要独立维护/比较/回滚的实现，或独立替换、资源、生命周期、跨 Task 复用、并行回滚等边界才晋升；单一路线的 Contract/schema 不兼容迁移应升级其所属版本并提供 adapter，不机械新增 Component。
小开关仅用于保持完整 Contract、资源和生命周期不变的有限局部参数变化，并集中使用约束性的
enum/结构化参数；算法候选、坐标/单位/GT/schema/split/checkpoint 语义变化不得用布尔开关掩盖。

**理由 / 影响**

在保持代码内部显式参数、依赖注入和局部 Contract 解耦的同时，避免为每个函数和层级创建 manifest，
也确保真正需要替换或追溯的边界有稳定 role。父 Contract 稳定时可在父内部递归拆分而不改外层 Pipeline；
Contract 不兼容时必须升级父版本并同步其 manifest、Pipeline spec/代码和测试。

**可逆性**

删除本条治理规则即可回到 V1.2.6 的原则性晋升标准；不影响现有运行代码、组件清单或历史产物。
