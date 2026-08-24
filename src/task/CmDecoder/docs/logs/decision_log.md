# CmDecoder AI 自主决策记录

## 2026-08-20 — 首版采用 token flatten + 当前 q 的轻量解码器

- branch: working tree
- post-commit: HEAD

**未指定点**

用户未指定 Cm token 的聚合方式和 Decoder 的具体层数。

**实际选择**

保留 `[16,256]` token 的 slot 顺序并 flatten，拼接当前 6 维 q，使用 LayerNorm + 512/256 两层 GELU MLP 输出下一帧 q。

**其他合理选择**

对 slot pooling、slot attention 或直接使用 masked token。

**选择理由**

flatten 不额外引入 slot pooling 假设，最直接检验冻结 Cm 是否携带可解码动作信息；MLP 便于先建立可运行 baseline。

**对结果的影响**

参数量和 slot 顺序依赖高于 permutation-invariant decoder；该选择只用于首版动作重建，不作为最终表示结构结论。

**可逆性**

完全可逆。

**建议用户确认**

否；首版结果后再比较 pooling 消融。

## 2026-08-21 — Poisson 拓扑只构建一次并由对应采样点驱动变形

- branch: working tree
- post-commit: HEAD

**未指定点**

用户确认使用 Poisson 自动建面，但未指定播放时应逐帧重新运行 Poisson，还是复用拓扑。

**实际选择**

首次显示 reconstructed mesh 时运行一次 Poisson；用 Poisson 顶点到 4 个最近采样点的逆距离权重建立绑定，后续帧以具有固定 correspondence 的手部采样点实时驱动该拓扑。

**其他合理选择**

每帧独立运行 Poisson，或离线预缓存每帧重建 mesh。

**选择理由**

逐帧 Poisson 会阻塞交互，而 1538 个采样点在相邻帧间具有固定对应关系；共享拓扑更适合检查采样点随 GT 动作的表面复现效果。

**对结果的影响**

拓扑由首次选择 reconstructed 模式时的帧决定；大幅关节运动下可能保留该参考帧的局部连接伪影。该 mesh 仅用于 viewer，不进入训练和指标。

**可逆性**

完全可逆。

**建议用户确认**

否；若观察到跨手指连接伪影，可改为按姿态区间重建多个拓扑。

## 2026-08-21 — 小规模 cache 采用物体分层抽样与两层 schema

- branch: working tree
- post-commit: HEAD

**未指定点**

用户确认先提取 50 个 episode，但未指定具体 episode；新增字段时也需要确定 cache 的复用粒度。

**实际选择**

固定 seed=42，按物体 round-robin 分层选择；cache 拆成 world geometry core 和 CmDecoder task 两层，使用 source SHA256 和 schema/config 校验。

**其他合理选择**

路径前 50、完全随机 50；或继续使用单体压缩 NPZ。

**选择理由**

分层选择减少样本集中于少数物体的风险；两层 cache 允许新增监督字段时复用昂贵的 FK 和 surface sampling。

**对结果的影响**

当前 50 episode 覆盖 25 个具有完整 pose/mesh/robot stream 的物体；该子集只用于小规模模型观察，不能替代全量结果。

**可逆性**

完全可逆。

**建议用户确认**

否；正式全量实验前应固定并审查最终 split manifest。

## 2026-08-21 — 残差预测保留旧 checkpoint 兼容分支

- branch: working tree
- post-commit: HEAD

**未指定点**

用户要求改为预测残差，但未指定此前 direct-q checkpoint 是否仍需可加载。

**实际选择**

新增 `prediction_target=delta_q` 作为新训练默认值；若旧 checkpoint/config 中缺少该字段，则按 `q_next` direct prediction 语义加载。

**其他合理选择**

彻底移除 direct-q 路径，使旧 checkpoint 无法通过当前代码按原语义评估。

**选择理由**

兼容分支不改变新训练目标，同时保留 EXP-004/005/006 的可复现性，且两种模式共用相同网络参数结构。

**对结果的影响**

新 checkpoint 的 Decoder 输出监督为 `q_next-q_t`；旧 checkpoint 结果不受默认目标切换影响。

**可逆性**

完全可逆。

**建议用户确认**

否。

## 2026-08-22 — 对比 mesh 固定当前 arm/wrist 位姿

- scope: task 内部可视化
- anchor: working tree / 2026-08-22

> 状态：已被下方“统一采用相对 wrist SE(3)”决策取代；仅保留为历史记录。

**未指定点**

用户要求同时显示当前手、GT 手和 Decoder 预测手，但 Decoder 只预测6维手指关节，未指定 GT/预测 mesh 应采用当前帧还是目标帧的 arm/wrist 位姿。

**实际选择**

三张 mesh 均采用当前帧 arm/wrist 位姿；GT 和预测 mesh 只分别替换为 `q_next` 与 `pred_q_next` 的6维手指关节角。

**选择理由与影响**

叠加差异只反映 CmDecoder 的手指预测，不会混入3 Hz间隔内机械臂整体运动。该约定在 viewer 状态栏明确显示，只影响可视化，不改变训练数据、loss 或评价指标。

**可逆性 / 是否需要用户确认**

完全可逆；不改变科研语义，无需额外确认。

## 2026-08-22 — 逐点模型采用 slot candidate flow 并将 q fitting 隔离于训练

- scope: task 内部模型与评估
- anchor: working tree / 2026-08-22

**未指定点**

用户确认以当前手点/法向/DenseToken/contact 和 Cm 为输入、逐点预测 flow，再从 `q_t` 初始化优化 q；未指定 point-slot decoder 的具体宽度、q 优化算法、正则和 checkpoint 选模指标。

**实际选择**

- 每个手点和每个 Cm slot 构造 edge，分别预测 candidate 3D flow 与 slot routing，沿 slot 加权得到逐点 flow；隐维度跟随 frozen Cm 的 `cm_dim`。
- 训练仅以 hand-flow Smooth-L1 更新逐点 decoder，以 validation point EPE 选模，不对 q optimizer 反传。
- q fitting 使用 Adam，默认100 steps、lr=0.05、q prior=`1e-4`；从 `q_t` 开始并按 URDF joint limit 截断，保留优化过程中 loss 最低的 q。
- cache 点先在 `q_t` 处反绑到各自 URDF link，避免重复 surface sampling 的浮点边界差异破坏逐点 correspondence。

**选择理由与影响**

该结构最直接对应 Cm object-flow decoder 的逐点 slot routing，同时保持当前整体 q baseline 不变。独立选模避免 q 优化超参数反向影响表示学习；保留 best iterate 防止近零动作样本被 Adam 数值步长劣化。

**可逆性 / 是否需要用户确认**

网络宽度、优化步数/lr/prior 均可配置；不改变用户已确认的输入、监督和训练边界，无需再次确认。

## 2026-08-22 — baseline 与逐点后处理统一采用相对 wrist SE(3)

- scope: task 内部模型、监督与可视化
- anchor: working tree / 2026-08-22

**未指定点**

用户已指定方案 A：baseline 直接预测腕部运动，且腕部仍预测相对运动；逐点路线需从完整手点运动恢复腕部与手指，但未指定旋转参数化及旧 checkpoint 兼容方式。

**实际选择**

- 相对变换统一定义为 `inverse(wrist_t) @ wrist_next`，平移以米表示、旋转以3维 axis-angle 表示。
- baseline 前6维保持手指 `delta_q` 语义，后6维回归相对腕平移与 rotvec；用显式配置开关保留历史6维 checkpoint 的 strict-load 兼容性。
- 逐点后处理从 `q_t`、单位 wrist 变换开始，联合优化6维 q、3维平移和3维 rotvec；使用小 wrist prior，并保留 loss 最低迭代。
- viewer 的 GT 和预测手均应用各自相对 wrist 变换；不通过未来 arm q/FK 获得目标腕。

**选择理由与影响**

6D axis-angle 是当前差值回归与可微优化都可直接使用的最小 SE(3) 参数化；显式兼容开关避免改变历史实验 checkpoint 的网络 shape。联合拟合消除了“完整 point flow 与固定腕”之间的人为不可约残差。

**可逆性 / 是否需要用户确认**

参数化和 prior 可配置；相对运动语义与不使用未来 arm FK 已由用户确认。

## 2026-08-22 — wrist-aware baseline 改用纯固定对应点 loss

- scope: task 内部训练目标与选模
- anchor: working tree / 2026-08-22

**未指定点**

用户确认先试纯点 loss，并要求保留原 q、平移、旋转 loss 代码但将权重设为0；未指定点坐标缩放和选模指标。

**实际选择**

- 使用现有1538个固定 correspondence 手点，不使用 Chamfer；目标为 `hand_points + hand_flow`。
- 预测 q 和相对 wrist SE(3) 经可微 FK 重建预测点，米制 Smooth-L1（beta=`0.01 m`）为唯一反传目标。
- 参数 loss 均继续计算和记录，wrist baseline 配置中权重置0；基础配置保留旧默认值以兼容历史 checkpoint。
- checkpoint 按 `val/hand_points/epe_mm` 选取。
- 不缩放点坐标；100倍缩放 smoke 中梯度范数约40且每步触发裁剪，米制原值梯度范数约0.31且不裁剪。

**选择理由与影响**

固定对应点直接测量整手几何误差，并让 q 与 wrist head 通过同一几何目标接收梯度。米制原值避免人为缩放配合全局梯度裁剪改变有效优化方向；mm EPE 提供直观展示。纯点监督仍可能存在 wrist/q 分解不唯一，必须继续独立报告参数指标。

**可逆性 / 是否需要用户确认**

四类 loss 权重与选模字段均可配置；纯点设置已由用户确认。
