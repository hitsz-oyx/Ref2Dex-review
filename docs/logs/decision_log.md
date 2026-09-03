# 全局 AI 自主决策记录

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
## 2026-08-25 — 七域训练统一采用 sample-wise MANO/robot contract

- scope: root / correspondence_ptv3_v2 与共享 Stage3 数据
- anchor: working tree / 2026-08-25

**实际选择**

将 GRAB、ContactPose、OakInk、HRDexDB human 和三个 HRDexDB robot domain 放入一个等比例 sampler；不同手类型不再靠全局 reconstruction 开关区分，而由每个样本的 MANO/robot contract 分流。ContactPose 采用 5 cm clean 交互帧过滤，机器人采用经 FK 标定的约 10 mm domain-specific q-space 扰动。

**影响**

共享 loader 可保持统一模型和 loss 接口，同时避免把 robot qpos 误解释为 MANO。ContactPose 新导出目录独立于旧 v2.0 数据，不改变历史实验可复现性。
