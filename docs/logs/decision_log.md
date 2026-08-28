# 全局 AI 自主决策记录

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
