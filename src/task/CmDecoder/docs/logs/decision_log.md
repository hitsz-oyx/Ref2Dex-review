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
