# 研究日志

## 实验：InteractionDynamics V1 数据链路

**假设**
现有 GRAB raw adapter 已包含构造固定 8 步 chunk 所需的全部几何和根位姿，不需要依赖 Stage4 pair 文件。

**观察到的失败 / 现象**
旧 Cm cache 没有写入 object root pose，并且只在当前 hand frame 中构造 endpoint motion。

**诊断**
V1 需要独立的 sequence schema，并在运行时分别构造 current-object frame 和 current-hand frame 表示。

**改动**
新增 V1 配置、raw cache builder、固定 horizon=8 的 dominant-hand wrapper 和 runtime chunk Dataset；cache 新增 `obj_root_pose_world`。

**结果**
静态编译和配置加载通过。合成轨迹验证了 8 步累计手/物位移、稳定 object index、tensor shape、坐标往返变换以及 normal 不受平移影响。`graspenv` 中成功重建并缓存真实 GRAB 序列。

**决策**
保留。

## 实验：V1 模型结构与梯度链路

**假设**
以 object patch 为索引的 cross-attention 能建立 action → interaction → effect 路径，并保留 World、Action 和 Effect 各模块的梯度。

**改动**
实现 deterministic FPS/KNN、联合 World Transformer、复用 hand patch index 的 Action Encoder、Action-World fusion、无 object residual 的 canonicalization、temporal effect query、厘米内部尺度 masked MSE、轨迹指标和 collapse diagnostics。

**结果**
Uni3D-S DeepSpeed checkpoint 成功加载 22,105,474 个参数，shape mismatch 为 0；新增的 150,144 个参数仅为 normal/entity embedding。真实 FrozenDenseToken + Uni3D-S 的全尺寸 batch 完成 forward/backward，输出为 `[1,128,384]`、`[1,64,384]`、`[1,64,384]`、`[1,8,512,3]`，峰值显存 845.7 MiB。World、Action、canonicalization 和 Effect 均有梯度，DenseToken 保持冻结。

**决策**
保留。

## 实验：真实数据单步训练闭环

**假设**
真实 cache、FrozenDenseToken、Uni3D-S、AMP loss 和双学习率 optimizer 能通过共享 Runner 完成训练。

**观察到的失败 / 现象**
首次 bfloat16 训练在 frozen spconv kernel 中失败，原因是 autocast 把 PTv3 输入改成了不受支持的 dtype。

**诊断**
FrozenDenseToken 的 checkpoint contract 是 FP32，不应继承新 Runner 的 bfloat16 autocast。

**改动**
仅在 FrozenDenseToken 周围禁用 autocast，其余可训练 V1 模块继续使用 bfloat16。

**结果**
真实 optimizer step 成功，耗时约 2 秒，loss 0.006959，Object ADE 1.444 mm。

**决策**
保留 FP32 frozen-adapter 边界。

## 实验：原始 V1 的 32-chunk overfit

**假设**
仅使用 effect supervision，V1 bottleneck 可以记住一个小型 contact-active sequence 子集。

**改动**
固定 `toruslarge_lift` 的 32 个 chunk，以 batch size 8 训练 200 steps，不增加 loss 或输入。

**结果**
Zero-flow ADE/FDE 为 42.53/76.98 mm；step-200 为 21.40/35.46 mm，改善 49.7%/53.9%，但没有充分 overfit。Interaction token variance 为 7.09e-6，pairwise cosine 约为 1。逐层诊断显示 world-object variance/cosine 为 0.399/0.713，action-context 为 0.0112/0.9952，interaction 为 4.27e-6/0.999996。Canonical attention entropy 为 4.15866，接近 64-way 均匀分布上限 4.15888。

**诊断**
主要 collapse 发生在 residual-free object canonicalization，而不是 pretrained world-object tokens。

**决策**
结论不充分：effect 优于 zero flow，但 representation-collapse 条件失败。

## 实验：Canonical query scale=10

**假设**
Canonicalization collapse 是 attention logits 长期处在近均匀 softmax 区间导致的。

**改动**
仅将 canonicalizer 的 normalized query 放大 10 倍，重复相同的 200-step overfit。

**结果**
Interaction variance 从 7.09e-6 增加到 2.55e-3，但 ADE 21.40 → 21.38 mm，FDE 35.46 → 34.58 mm，基本没有实际收益。

**决策**
撤回。Sharper attention 能改变 token diversity，但不能解释 overfit plateau。

## 实验：Effect Decoder 的 token 使用干预

**假设**
Effect Decoder 可能只使用 sample-level mean interaction，而忽略 token-level spatial variation。

**改动**
在 baseline step-200 checkpoint 上分别将 64 个 interaction tokens 替换成样本内均值、跨样本交换或全部清零，不重新训练。

**结果**
相对原预测的 RMSE 变化分别为：均值替换 0.0048 mm、跨样本交换 0.260 mm、清零 14.56 mm。

**诊断**
Decoder 强依赖 interaction feature，但几乎只使用其 sample-level mean，没有利用 object-patch-indexed 结构。

**决策**
原始 V1 虽然优于 zero flow，但没有达到 object-indexed representation 目标。

## 实验：Effect spatial routing bias

**假设**
用空间 bias 将 effect query 显式路由到附近 object patch，可以迫使 Decoder 使用局部 interaction。

**改动**
加入 sigma=5 cm、仅影响 attention routing 的 Gaussian bias；没有把 object geometry 加入 interaction value 或 flow head。

**结果**
ADE 21.40 → 21.39 mm，FDE 35.46 → 41.40 mm，interaction variance/cosine 仍为约 8.52e-6/1。

**诊断**
上游 values 已经几乎相同，空间路由无法恢复局部 interaction。

**决策**
撤回。

## 实验：Action-context local gain

**假设**
Canonicalization 前，全局 action component 在数值上淹没了 patch-local residual。

**改动**
将 action context 分解为 mean 和 local residual，使用：

```text
action_context = mean + 10 × local_residual
```

没有增加输入、辅助 loss 或 object residual。

**结果**
相同 32 个 chunk 训练 400 steps 后，epoch ADE/FDE 达到 6.71/12.43 mm，相比 zero flow 改善 84.2%/83.9%；最后一个 batch 为 5.39/7.74 mm。最终 interaction variance/cosine 为 0.204/0.793，不再存在严重 token collapse。

**决策**
保留 `action_local_gain=10` 作为 V1 默认值。单序列 controlled overfit 验收通过。

**下一步**
建立 sequence-disjoint 80/10/10 split，并在 full GRAB 训练前运行小规模多序列实验。

## 实验：10 序列小规模泛化验证

**假设**
通过单序列 overfit 的表示能够在少量多序列数据上学习，并在未见序列上优于 zero-flow，同时不发生严重 token collapse。

**观察到的失败 / 现象**
原配置的最佳指标名写成 `val/object_ade_mm`，而 Runner 实际产出 `val/object/ade_mm`；该问题只在存在验证集时触发。全局 `max_samples` 还会按路径排序截断，使小规模训练偏向靠前序列。

**诊断**
指标键是配置拼写错误；多序列采样需要按 sequence directory 单独限制，而不能对排序后的全体样本直接截断。

**改动**
缓存 10 条 GRAB 序列，生成 3412 个 dominant-hand chunk，以序列隔离方式划分 8/1/1；每条序列最多取 128 个 chunk。修正最佳指标名，并增加 `max_samples_per_sequence`。使用 batch size 8、AdamW、5 epochs，共训练 640 steps。

**结果**
训练 ADE：35.53 → 19.54 mm。验证 zero-flow ADE/FDE 为 42.94/66.76 mm，模型为 32.07/46.03 mm，分别改善 25.3%/31.1%。测试 zero-flow 为 30.98/54.62 mm，模型为 25.19/44.81 mm，分别改善 18.7%/18.0%。验证 interaction variance/cosine 为 0.1108/0.8874；测试为 0.0367/0.9627。

**决策**
保留。小规模 sequence-disjoint 泛化成立，模型在两个未见序列上均优于 zero-flow；测试序列 token 多样性弱于验证序列，需要在扩大数据前继续观察，而不是据此增加结构。

**下一步**
先实现 representation/attention 导出并检查不同序列的 interaction token 与注意力，再决定是否扩大到完整 GRAB。

## 实验：未见序列的表征与注意力导出

**假设**
逐样本导出 interaction tokens 和三类 attention，可以确认测试序列较高的 token cosine 是否同时伴随更均匀的 object-to-action attention。

**改动**
实现 `InteractionDynamics/extract.py`，按 `seq_id/side/raw_frame_id.npz` 独立保存 V1 规定的 patch center、action/interaction tokens、预测与真值 object effect，以及三类 attention。对小规模实验的验证集和测试集各导出前 8 个样本。

**结果**
全部字段和 shape 与 V1 schema 一致；单样本压缩文件约 5.6 MiB。验证样本 interaction variance/cosine/attention entropy 为 0.0645/0.9345/4.0201；测试样本为 0.0231/0.9765/4.0819。64-way 均匀 attention 的熵上限为 4.1589。

**诊断**
测试序列的 interaction tokens 确实更相似，并伴随更接近均匀分布的 object-to-action attention。当前只有单个测试 sequence 的 8 个相邻片段，证据不足以归因于模型结构，也可能来自动作或接触模式本身。

**决策**
保留导出工具；不增加 regularization 或新结构。

**下一步**
扩充 sequence-disjoint 验证/测试序列后重复同一诊断，区分序列特性与系统性 attention bottleneck。

## 实验：small10 训练强度与均匀时间采样

**假设**
原 small10 只训练 5 epochs 且每序列取最前 128 个片段，训练不足和时间覆盖偏差共同限制了泛化结论。

**改动**
先保持旧采样从 5 epochs 续训到总计 20 epochs；再将 `max_samples_per_sequence` 改为在全部合法片段上均匀取样，从头训练相同 20 epochs。

**结果**
旧采样第 20 轮 train ADE 为 10.87 mm，最佳验证 ADE 为 30.78 mm，最佳 checkpoint 测试 ADE 为 29.31 mm，说明继续训练形成明确 train/val gap。均匀采样把 `toruslarge_lift` 覆盖从前段扩展到 current frame 37–847；最佳验证 ADE 进一步降至 27.62 mm。均匀测试集 zero-flow ADE/FDE 为 45.99/79.83 mm，模型为 46.01/81.19 mm，表明旧测试前段采样明显低估了后段难度。

**决策**
保留均匀时间采样，并将 small10 默认训练改为 20 epochs。原 5-epoch、前 128 帧结果不能作为稳定泛化结论。

## 实验：post-gain interaction token 干预

**假设**
即使 `action_local_gain=10` 提高了 token 方差，Effect Decoder 仍可能只使用样本级全局 interaction。

**改动**
在均匀采样 small10 最佳 checkpoint 上比较 normal、mean-token replacement、token shuffle 和 cross-sample replacement。

**结果**
验证 ADE 分别为 27.62/27.34/27.62/30.03 mm；测试为 46.01/46.48/46.01/47.17 mm。mean-token 几乎不损失性能，shuffle 与 normal 数值相同，cross-sample 只产生有限退化。

**诊断**
Decoder 使用样本级 interaction 信息，但没有建立有效的 object-indexed token-level dependency。仅观察 variance/cosine 不足以证明局部表征被使用。

**决策**
保留 `intervention.py` 作为后续结构验收工具；当前 object-flow supervision 下的 global shortcut 得到直接确认。

## 实验：small20 的 12/4/4 序列泛化

**假设**
增加验证和测试序列可以区分单条 sequence 特性与系统性泛化问题。

**改动**
扩展到 20 条 GRAB 序列，使用 sequence-disjoint 12/4/4 划分和均匀时间采样；训练集 1536、验证集 362、测试集 512 个片段，训练 20 epochs。

**结果**
train ADE 收敛到 14.38 mm。zero-flow 验证/测试 ADE 为 49.05/45.82 mm；最佳 checkpoint 为 46.93/44.80 mm，仅改善 4.3%/2.2%。最佳 checkpoint 的验证/测试 interaction cosine 为 0.9965/0.9987，接近完全 collapse；继续训练后验证 ADE 恶化至 52.61 mm。

**决策**
结论明确：训练可以收敛，但现有结构对未见 sequence 的提升很弱，并存在系统性 global-token shortcut。下一步应修改 canonicalization 或 supervision，而不是继续单纯增加 epoch。

## 实验：Uni3D 联合单位球输入

**假设**
pretrained Uni3D 的单位球坐标分布与当前米制输入不一致，可能导致多序列泛化弱。

**改动**
仅对送入 Uni3D backbone 的 hand+object 联合副本使用共享中心和尺度归一化，其余 action、contact 和 effect 几何保持米制；在 small20 上训练 5 epochs。

**结果**
单位球版本最佳验证 ADE 为 49.47 mm；原始米制版本相同早期阶段最佳为 46.93 mm。interaction cosine 仍约 0.99。

**决策**
撤回。坐标分布 shift 不是当前主要瓶颈，下一实验转向 canonicalization / action-preserve supervision。

## 实验：V3 hand/object patch 直接监督

**假设**
参考 Fast-WAM 的逐 token 直接目标，在 `action_context_tokens` 上重建 hand patch motion、在 `interaction_tokens` 上重建 object patch effect，可以约束中间表征并让 Effect Decoder 使用 object-indexed 局部信息。

**观察到的失败 / 现象**
small20 仅使用 dense effect loss 时，最佳验证/测试 improvement 很弱，interaction cosine 为 0.9965/0.9987，mean 与 shuffle intervention 几乎不改变预测。

**改动**
增加两个线性 head，分别预测 `[8,64,3]` 的 hand/object patch 平均位移；Runner 使用 Uni3D 的稳定 KNN index pooling GT，三项厘米制 MSE 权重均为 1。完整 future object displacement 只作为 Runner target，不进入模型 forward。

**结果**
真实 batch 的初始 effect/action/patch-effect loss 为 19.02/20.60/19.65，量级相当且一步更新成功。small20 训练 5 epochs 后最佳 checkpoint 位于 step 384，验证 ADE 为 48.82 mm，弱于原 baseline 的 46.93 mm；验证 interaction cosine 仍接近 1。最佳 checkpoint 的 normal/mean/shuffle/cross-sample ADE 为 48.82/48.81/48.82/49.09 mm。

**诊断**
两个辅助 head 能优化，但它们是主 Effect Decoder 之外的旁路；模型仍可让 patch head拟合以全局刚体运动为主的平均 target，同时让 dense decoder 继续忽略局部 token。直接监督存在不等于主预测路径建立了局部依赖。

**决策**
保留 V3 三层监督实现作为明确对照，但该实验未通过局部表征验收，不扩大训练规模，也不调辅助 loss 权重。

**下一步**
最小化地把 object-patch prediction 与 dense effect prediction 结构性耦合，再用同一 small20 与 intervention 验收。

## 实验：V4 patch-routed Effect Decoder

**假设**
把 `interaction token → patch motion` 串入 dense effect 主链，并让每个 effect point 只读取最近 object patch 的 token，可以消除全局 attention 与 absolute xyz 的刚体捷径，使 shuffle interaction tokens 显著破坏预测。

**改动**
移除 Effect Decoder 的全局 cross-attention。effect point 根据当前 object-frame 坐标硬路由到最近 Uni3D object patch；预测为对应 patch motion 加局部 residual，residual 仅输入对应 `C_j`、point-to-patch 相对坐标、normal 和时间编码。干预工具同步重算被干预 token 的 patch motion，避免保留原始旁路结果。

**结果**
真实 batch forward/backward 和 5-epoch small20 训练均成功。最佳 checkpoint 位于 step 384。验证 ADE 为 49.45 mm，测试 ADE 为 42.94 mm；旧仅 effect baseline 为 46.93/44.80 mm，表现一升一降。验证 normal/mean/shuffle/cross-sample ADE 为 49.45/49.40/49.42/49.76 mm，shuffle 相对 normal 变化 -0.05%；测试为 42.94/42.93/42.95/43.74 mm，shuffle 相对变化 +0.03%。两者均远低于预设 +10% 验收线。

**诊断**
局部路由本身已经进入主链，但上游 interaction cosine 仍接近 1；当所有 `C_j` 和 patch-motion prediction 近似相同时，硬路由仍等价于读取全局刚体状态。结构耦合无法单独创造 object effect 中不存在或很弱的局部辨识信号。

**决策**
保留 V4 实现和负结果作为强结构对照，但判定未通过 object-indexed representation 验收；不继续调辅助 loss 权重或扩大训练。

**下一步**
重新审视刚体 object effect 是否支持 64-token object-indexed 表示；若继续该表示，需要引入能区分相同 SE(3) 下不同 human mechanism 的目标。

## 实验：canonicalization 定位与 patch target 能量分解

**假设**
如果 action-context 本身保留局部差异而 canonicalizer 将其抹平，空间局部 canonicalization 可能恢复 object-indexed interaction；同时 patch target 的空间能量占比决定这种表示是否真的受到数据监督。

**观察到的失败 / 现象**
V4 已将 patch prediction 接入 dense 主链，但 shuffle 仍几乎不影响 ADE。

**改动**
先导出 V4 最佳 checkpoint 的 16 个验证样本，比较 action-context、interaction 和 attention。随后以 5 cm Gaussian 权重按 object-hand 相对位置聚合 action tokens，训练相同 small20 5 epochs。最后增加 `target_diagnostics.py`，将 hand/object patch motion 分解为跨 patch 均值和 patch-local residual 能量。

**结果**
原 V4 action-context variance/cosine 为 0.592/0.814，interaction 为 0.000406/0.99959，canonical attention entropy 为 4.044，确认坍缩发生在 canonicalizer。空间 canonicalization 初始验证 interaction variance/cosine 改善到 0.0276/0.9720，但最佳 checkpoint 验证/测试 ADE 为 51.31/43.23 mm；shuffle 相对 normal 仅退化 0.07%/0.85%，仍未通过 10% 验收，结构改动已撤回。target 分解中，验证 hand/object local energy ratio 为 1.16%/1.01%，测试为 1.90%/3.30%；全局共同运动占 96.70–98.99%。

**诊断**
原 canonicalizer 的确会平均 action token，但这不是唯一根因。GRAB 刚体 patch motion target 的绝大多数能量本来就在全局共同分量，raw patch MSE 自然鼓励 64 个 token 给出相同预测。人为制造 token 差异不能让 decoder 使用数据中缺乏辨识力的局部信息。

**决策**
撤回 spatial canonicalization，保留 target 诊断工具。停止继续调 canonicalizer、loss 权重或 diversity 正则。

**下一步**
将表示拆分为全局 object motion/SE(3) 与 hand-patch mechanism；object-patch 表示只在有 local residual/contact 等信息性目标时使用。

## 实验：V5 hand-object 相对轨迹先验统计

**假设**
对 hand patch 与当前最近 object patch 的共同运动作差后，法向/切向相对运动应比 raw patch flow 含有更强的跨局部差异，因而更适合定义 interaction mechanism。

**观察到的失败 / 现象**
V4 的 hand/object raw patch target 只有约 1%–3% 局部能量，无法辨识相同刚体效果背后的不同手部作用方式。

**改动**
Dataset 增加仅供 Runner 监督与诊断使用的 current-object-frame future hand displacement，不加入模型 forward。新增只读 `relative_diagnostics.py`：复用 Uni3D patch index，为每个 hand patch 固定匹配当前最近 object patch，统计未来 8 步累计距离变化、法向/切向相对运动，以及 2 cm 接触状态转移。

**结果**
验证集法向/切向相对运动的跨边局部能量比例为 33.11%/22.16%，测试集为 41.58%/37.49%；相比 raw hand/object patch flow 的 1.16%/1.01%（验证）和 1.90%/3.30%（测试）明显更强。距离变化局部能量仅为 2.92%/3.20%。2 cm 阈值下验证/测试接触占比为 9.45%/17.86%，接触建立或解除只占全部 edge-step 的 0.98%/1.26%。切向累计相对运动中位数为 0.58/0.59 cm。

**诊断**
共同 global transport 被相对运动有效抵消，`v_n/v_t` 确实提供了 raw flow 缺失的局部辨识信号。距离变化仍主要受样本级共同分量控制；硬 contact transition 很稀疏，第一版若以分类为主会面临严重类别不均衡。

**决策**
保留相对轨迹诊断和数据 target，判定 V5 值得进入最小训练原型。训练时优先直接监督法向/切向相对运动，距离只作辅助，不先加入 contact classification。

**下一步**
构造 hand→object 局部 edge token 和 `v_n/v_t` 预测头，在 small20 上比较 target 拟合、token 局部差异与 effect consistency。

## 实验：V5 最近邻 edge token 直接监督

**假设**
将 hand patch 的 action-context、当前最近 object patch token 和相对几何融合成 edge token，并直接监督未来累计法向/切向相对运动，可以学习不坍缩且能跨序列预测 mechanism 的表示。

**改动**
新增 64 个 hand→nearest-object edge token 和 `[8,64,4]` 的 `[v_n,v_t]` 预测头。V5 配置保留 effect consistency，关闭 raw hand/object patch 辅助损失。先监督全部最近邻边，再按 V5 建议仅监督当前距离小于 5 cm 的近场边；不加入稀疏 contact classification，也不调 loss 权重。

**结果**
全部 edge 实验位于 `outputs/interactiondynamics/interaction_dynamics_20260810_001729`。5 epochs 后最佳验证/测试 relative RMSE 为 1.389/1.178 cm，差于 zero predictor 的 1.370/1.110 cm；effect ADE 为 49.96/45.31 mm。

5 cm edge 实验位于 `outputs/interactiondynamics/interaction_dynamics_20260810_002606`，最佳 checkpoint 为 step 768。验证/测试有效 edge 占 38.66%/56.35%；relative RMSE 为 0.999/0.874 cm，仍差于 masked zero 的 0.968/0.841 cm。effect ADE 为 48.77/44.10 mm。edge token variance/cosine 为验证 0.125/0.871、测试 0.155/0.840，未发生坍缩。

**诊断**
排除远距离 edge 可降低 target 绝对误差，却不能使相对轨迹预测优于零基线。当前失败不是 token collapse，也不是单纯由远处非交互手部区域污染；更可能来自固定 current-frame 最近邻无法描述未来 correspondence、累计轨迹 MSE 受重尾样本影响，或 small20 跨序列 mechanism 本身难以从当前输入泛化。

**决策**
保留 edge-token 和 5 cm mask 实现作为 V5 明确负对照，不接入 Effect Decoder，不继续调 loss 权重或扩大数据规模。

**下一步**
先分解近场 target 的逐步增量、重尾程度和 correspondence 稳定性；只有诊断支持时，再最小测试逐帧匹配或 robust relative loss。
