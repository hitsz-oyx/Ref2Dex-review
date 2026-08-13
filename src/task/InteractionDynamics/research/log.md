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

## 实验：V6 correspondence、mask 与逐步增量

**假设**
V5 打不过零预测可能来自固定 current-frame nearest 在未来失效、current-near mask 漏掉接触建立，以及累计 target 的重尾；逐步增量可能是最小可预测 target。

**改动**
Dataset 增加仅供 Runner/诊断使用的 future object normals。新增 `correspondence_diagnostics.py`，使用稳定 patch point index 重建未来 patch centers，逐帧计算 nearest object patch；比较 fixed/dynamic correspondence、current-near/chunk-active mask，以及 fixed cumulative、fixed increment、dynamic increment 的分布。随后只改变一个变量，在相同 small20、5 cm current mask、MSE 和网络上训练 `increment_fixed` 5 epochs。

**结果**
验证/测试分别有 51.57%/53.83% edge 至少换过一次最近 patch，fixed 逐帧匹配率为 68.29%/66.11%，平均切换 0.83/0.88 次。current-near mask 占 38.66%/56.35%，chunk-active 占 43.89%/60.28%，即漏掉 active edges 的 11.91%/6.51%。

逐步增量把 current-near RMS 从累计 target 的 1.331/1.179 cm 降至 0.352/0.300 cm，但重尾比 `q99/q50` 仍为 19.53/22.21。dynamic increment 的 RMS 为 0.354/0.299 cm，与 fixed increment 几乎相同。

训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_084655`，最佳 effect checkpoint 位于 step 192。验证/测试 relative RMSE 为 0.275/0.240 cm，仍差于 zero predictor 的 0.247/0.206 cm；effect ADE 为 48.62/43.51 mm。各 epoch 验证 relative RMSE 均未优于 zero。

dominant-hand summary 的 8,488 个候选窗口中保留 5,770（67.98%）；主要排除 1,836 个双方不活跃窗口（21.63%）和 841 个双手协作窗口（9.91%）。

**诊断**
固定 correspondence 确实经常失效，静态 mask 也会漏掉接触建立，但它们不是当前误差分布的唯一主因：动态/固定增量统计近似，而单独改成增量仍不优于 zero。直接加入“全量”还会引入大量双方不活跃窗口；真正可能补充 mechanism 多样性的是 bilateral-active 子集。

**决策**
保留诊断工具、future normals target 和 increment 配置作为负对照。不继续做仅替换 dynamic nearest 的 5-epoch 低信息量训练，也不调 relative loss 权重。

**下一步**
重新设计 interaction-change 与 bilateral-active 数据采样，并先建立条件均值等强基线；目标在新采样上可预测后，再训练 mechanism representation。

## 实验：V7 时空 object-centric interaction field 与全局 SE(3)

**假设**
V5/V6 的困难来自把 interaction 压缩为单帧最近邻 edge 并直接拟合高噪声局部 target。保留完整时间 action token，以连续 Gaussian 场将已知手部轨迹投影到物体表面，再显式预测全局刚体 SE(3)，应更符合 GRAB 刚体 effect 的结构并改善未见序列泛化。

**观察到的失败 / 现象**
V4 的 object patch flow 约 97%–99% 为全局共同运动；V5/V6 的局部相对目标均未优于零预测。继续训练 nearest edge 或调相对损失权重的信息量较低。

**改动**
Action Encoder 输出 `[B,8,64,D]` temporal tokens。对每个时间步和 object patch，使用 5 cm Gaussian 对 64 个 hand patches 连续加权，融合 action、object token、相对几何、法向和手部法向/切向增量，形成时空 interaction field。全局池化后预测 8 个逐步 SE(3) increment；旋转由 axis-angle 转换并逐步复合，dense point flow 由刚体变换解析生成。训练只使用厘米制 translation MSE 与 rotation-matrix MSE（权重 1/10），关闭旧 effect/patch/relative losses。增加 field 空间/时间方差、描述子方差、接近密度和 soft-weight entropy 指标。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_102612`，5 epochs / 1920 steps，最佳 checkpoint 位于 step 1152。验证 ADE 为 46.70 mm，zero-flow 为 49.05 mm，改善 2.35 mm（4.8%）。测试 ADE/FDE 为 42.08/73.09 mm，zero-flow ADE 为 45.82 mm，改善 3.75 mm（8.2%）；此前 V4/V6 测试 ADE 为 42.94/43.51 mm。测试 SE(3) translation error 为 0.953 cm，rotation error 为 2.572°。测试 field 空间/时间 feature variance 为 0.0411/0.0171，descriptor variance 为 0.00340，soft hand-weight entropy 为 3.621（64-way 上限 4.159），说明场在空间和时间上都不是常量。训练 batch ADE 可达到约 10 mm。实现与坐标链路聚焦测试共 15 项全部通过。

**诊断**
显式刚体归纳偏置避免了逐点 decoder 学习一个几乎纯全局的 flow，同时连续 splatting 保留了完整动作时序并消除了 hard nearest correspondence。验证和测试均优于各自 zero-flow，方向成立；但当前尚不能区分收益主要来自 SE(3) head，还是来自局部时空 field。

**决策**
保留 V7 路径，作为下一阶段主实验方向。旧 V1–V6 分支只保留作对照，不据此继续调 nearest-edge target。

**下一步**
做 action shuffle、移除显式 descriptor、global-action-only 三个最小消融，确认 V7 是否真实利用局部时空交互，再考虑扩大数据。

## 实验：V8 full-resolution local-edge sanity check

**假设**
V6 的 64-patch 表示可能在交互关系形成前丢失指尖和接触边界等稀疏高频信号。若保留全部 1538 个 hand surface points，并只连接每点最近的 4 个 object points，逐点 dense interaction representation 应明显优于 patch-edge 和 zero baseline。

**观察到的失败 / 现象**
V6 patch-edge 的验证/测试增量 relative RMSE 为 0.275/0.240 cm，差于 zero 0.247/0.206 cm；但不能排除 FPS/KNN patch pooling 已破坏接触几何。

**改动**
新增不加载 Uni3D/DenseToken 的 `DenseEdgeInteractionModel`。对完整 1538×4096 当前几何计算 KNN，每个 hand point 保留 4 条边；以逐步 hand position、相对位置/距离、hand/object normals、hand increment 及其法向/切向分解构造 edge，经 64 维 MLP 和距离 soft pooling 得到 `[B,8,1538,64]` dense interaction set。Runner 使用 future object 仅构造逐点固定对应的增量 relative target，并同时报告 zero 与 hand-motion-only 基线。尚未加入 bottleneck `C`。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_110406`，5 epochs / 1920 steps，耗时 2 分 36 秒。所有 epoch 验证结果均未超过 zero；最佳 checkpoint 为 step 384。最佳验证 relative RMSE 为 0.2470 cm，zero 为 0.2465 cm，hand-only 为 0.5971 cm；测试为 0.2031/0.2027/0.5507 cm。验证/测试 5 cm 内有效 hand-point 比例为 60.76%/77.85%。dense token feature variance 从验证 epoch 1 的 0.000894 增至 epoch 5 的 0.001322，但误差同步恶化。相关聚焦测试通过。

**诊断**
全量 hand points 和更高近场覆盖率没有让相对增量跨序列变得可预测，说明 64-patch 过早压缩不是 V6 失败的主要原因。hand-only 基线远差于 zero，也说明 target 不能靠直接复制已知 hand action 解决。继续增加 K、edge width 或 learned compressor 只会在尚未成立的 `E_dense` 上增加复杂度。

**决策**
保留轻量 V8 分支和负结果作为 dense-first 对照；暂不实现 `E_dense → C` reconstruction/compressor，也不调 K 或网络宽度。V7 的显式 SE(3) effect 路径仍是当前主方向。

**下一步**
优先完成 V7 action/descriptor/global-only 消融以定位 8.2% test ADE 改善来源；只有找到比 zero 更可预测的 interaction-specific target 后，再重启 dense bottleneck 研究。

## 实验：V9 global-action-only 对 V7 局部场归因

**假设**
若 V7 的 object-indexed spatiotemporal field 确实编码了不可替代的局部 interaction，移除空间 field 与显式 `[rho,a_n,a_t]` descriptor、只保留全局 temporal action 和 object context 后，验证与测试 ADE 应明显退化。

**观察到的失败 / 现象**
V7 测试 ADE 比 zero-flow 改善 8.2%，但 V8 已证明提高局部几何分辨率不能改善 relative target。尚不清楚 V7 收益来自 local field，还是来自更匹配刚体数据的 SE(3) head 与全局 motion correlation。

**改动**
增加 `field_ablation`。`global_action_only` 保留与 V7 相同的 Action Encoder、world cross-attention、object tokens、SE(3) head、loss 和 small20 配置；仅将 temporal action 在 64 个 hand patches 上取均值并投影，再复制到全部 object positions，同时把 field descriptor 清零。送入 SE(3) head 的 spatial feature variance 和 descriptor variance 因而严格为 0。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_112110`，5 epochs / 1920 steps，最佳 checkpoint 位于 step 1536。各轮验证 ADE 为 48.29、46.20、46.26、45.41、45.68 mm；最佳验证 ADE/FDE 为 45.41/77.51 mm，优于 full-field 的 46.70 mm。最佳 checkpoint 测试 ADE/FDE 为 42.29/73.38 mm，full-field 为 42.08/73.09 mm，仅退化 0.21/0.29 mm（ADE 约 0.5%）。测试 translation/rotation error 为 0.957 cm/2.628°，full-field 为 0.953 cm/2.572°。global-only 测试 field spatial variance 和 descriptor variance 均为 0。

**诊断**
局部场被完全消除后，验证反而改善，测试仅有远小于跨实验波动的轻微退化；因此没有证据表明 V7 使用了不可替代的 object-indexed local interaction。V7 的有效部分是逐步 SE(3) 刚体归纳偏置，以及包含 world context 的全局 temporal action conditioning。原 8.2% test 改善不能作为 mechanism latent 已成立的证据。

**决策**
保留 `global_action_only` 作为新的强 effect baseline，并保留 full-field 作对照；降低 full-field 的研究优先级。暂不做 descriptor-only 调权、dense compressor 或增加 field 分辨率。

**下一步**
先设计 interaction-rich sampling 或能区分“相同全局 hand transport、不同局部接触机制”的对照任务。只有局部模型能稳定超过 global-action-only 时，再研究可压缩的 interaction bottleneck `C`。

## 实验：V9 global C 监督的 MANO self-inverse

**假设**
即使 V9 的 C 主要表示全局 temporal action，它仍可能作为可微监督，从当前 MANO 手初始化恢复未来 8 帧动作，从而先打通 human self-inverse 到后续 robot FK 所需的优化链路。

**改动**
实现 `inverse_mano.py`。脚本根据 cache 的 `source_raw_file/raw_frame_id/side` 回到 GRAB raw 参数，加载对应 subject v_template 的左右手 MANO。冻结 V9 encoder，以原始 human future action 的 `[8,128]` global C 为 teacher；将未来 MANO pose、global orientation、translation 全部初始化为当前帧，优化 C MSE 加时间平滑先验。GT future hand points 不进入 loss，只计算 ADE/FDE。

**结果**
test sample 0、200 steps、Adam lr 0.003、smooth weight 0.01 时，C RMSE 从 0.2377 降至 0.00953；hand-point ADE 从静止基线 27.52 mm 降至 10.88 mm，FDE 从 50.43 mm 降至 10.63 mm。产物为 `output/InteractionDynamics/mano_inverse/test_000_smooth.npz`。高学习率 0.03、smooth weight 1e-4 的对照虽然将 C RMSE 降到 0.118，但 ADE/FDE 恶化到 82.97/252.04 mm。

**诊断**
C→MANO 的梯度链和 raw/cache 帧对应均已打通。C 监督含有可恢复动作的信息，但 inverse 明显欠约束：弱先验允许错误 MANO 轨迹匹配更低的 C loss。合理的时间先验不是可选鲁棒性，而是决定重建是否可信的必要约束。

**决策**
保留 MANO self-inverse 工具，并将稳定的 lr 0.003、smooth weight 0.01 设为默认。当前只通过单样本 sanity check，不宣称 split-level 泛化。

**下一步**
批量运行多个 test chunks，报告成功率和 ADE/FDE 分布；增加 C-only、C+smooth 和 oracle hand trajectory 三组对照后，再接 Robot FK。

## 实验：V10 Action token 的 global/local linear probe

**假设**
V9 只证明 object-indexed field 不必要，并不能证明 64 个 hand-patch Action tokens 已经坍缩。若 Action Encoder 仍保留手指局部动作，冻结表示上的线性 probe 应能在未见序列上预测去除全局均值后的 patch-local hand flow，并优于 zero predictor。

**改动**
新增 `action_probe.py`。在冻结 V9 best checkpoint 上一次性提取 small20 train/val/test 的 `action_context_tokens` 和稳定 KNN patch flow，将 target 精确分解为每步 global mean 与 zero-mean local residual。分别拟合 full、global、local 三个 affine ridge probe；ridge 从 0.001–100 中仅按 validation RMSE 选择，随后固定评估 test。主 encoder 不更新。

**结果**
Action token 的 train spatial variance 为 2.370，跨 patch pairwise cosine 为 0.721，未发生 collapse。global probe 选择 ridge 1，训练/验证/测试 RMSE 为 0.601/1.556/1.666 cm，对应相对 zero 改善 85.2%/67.0%/60.4%。local residual probe 选择 ridge 100，RMSE 为 0.297/0.427/0.452 cm，相对 zero 改善 44.9%/16.4%/22.8%。full-flow probe 选择 ridge 100，验证改善 20.1%，测试只改善 3.1%。结果保存在 `output/InteractionDynamics/action_probe/v9_small20.json`。

**诊断**
Action Encoder 明显保留全局运动，也保留较弱但可跨 sequence 线性读出的 patch-local residual。因此 V9 的结论应限定为“SE(3) effect head 不需要空间 field”，不能扩展为“Action representation 已全局坍缩”。full-flow 指标被大尺度 global motion 主导，无法单独衡量局部信息保存程度。

**决策**
保留 probe 工具。当前 checkpoint 已通过 local-information sanity check，不立即打开旧 `action_loss_weight=1`；旧 head 重建 raw flow 的目标不够针对性。若后续训练 action-information loss，应拆分 global mean head 和 local residual head，并分别报告相对 zero 的改善。

**下一步**
在 MANO self-inverse 中比较只监督 V9 global C 与同时监督 patch-level Action tokens，检查 local Action 信息能否进一步降低手指局部重建误差；这比直接重训旧 raw action head更接近 RL prior 需求。

## 实验：V11 object-centric body action field 辅助监督

**假设**
由全局时序 `C` 和 object tokens 重建不依赖 MANO 身份的 `[rho,d,v_n^local,v_t^local]` 场，可以保护局部执行体动作信息，同时不损害 SE(3) effect 泛化。

**观察到的失败 / 现象**
V9 的 Action local probe 在 test 上优于零预测 22.8%，但局部信息明显弱于 global 信息；直接重建 MANO 又会引入 human-specific 语义。

**改动**
保持 V9 global-action-only 与 SE(3) 路径不变。新增 `C + object world token → 6D field` MLP。target 使用完整 1538 点 future hand surface：5 cm Gaussian density、最近距离，以及将累计 hand displacement 减去全手均值后按 object normal 分解的局部法向/切向运动。三组 loss 等权，均不使用 future object motion、MANO joint、finger ID 或 hand point identity。

**结果**
训练产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_134059`。先训练 5 epochs / 1920 steps，再从 latest checkpoint 续训到总计 20 epochs / 7680 steps；实测耗时分别为 10:02 和 30:53，总计 40:55。按 object ADE 选择的 checkpoint 仍是 epoch 1：验证 48.42 mm，test ADE/FDE 45.71/79.40 mm。

local-motion 验证 RMSE 从 epoch 1 的 0.321 cm 下降，在 epoch 15 达到全程最好 0.242 cm，但仍差于 zero 0.223 cm；epoch 20 为 0.253 cm。使用 epoch 15 checkpoint 评估 test，density 为 0.136（zero 0.311）、distance 为 2.083 cm（zero 5.882 cm）、local-motion 为 0.277 cm（zero 0.243 cm），object ADE/FDE 为 44.41/79.28 mm。长训练比 epoch 1 的 test ADE 45.71 mm 有恢复，但仍弱于 V9 的 42.29/73.38 mm。

epoch 1 Action token spatial variance/cosine 为 1.030/0.760，local probe test 相对 zero 改善 34.0%。epoch 15 为 1.718/0.674，local probe 验证/test 改善 21.5%/26.1%，仍高于 V10 的 16.4%/22.8%，但优势明显缩小；global probe test 改善回到 60.5%，与 V10 的 60.4% 相同。

**诊断**
5 epochs 时训练尚未收敛，不能据此断言最终负优化；20-epoch 结果确认 local target 继续改善到 epoch 15，但随后平台波动，始终未超过 zero。辅助监督确实改变了 Action Encoder，并略微增加其线性可读局部信息；decoder 仍主要学会 density/distance 这类静态几何。object effect 长训练只部分恢复且始终弱于 V9，因此现有证据支持“固定 small20 下存在目标冲突”，但不外推为该监督在更大数据上必然负优化。

**决策**
保留 V11 实现和配置作为部分正向、整体未通过的对照。结论以完整 20 epochs 为准，不再称 5-epoch 结果已经收敛。当前不调 loss 权重：V11 的 field-zero 与 effect 两项验收仍失败，简单降权不能解决 target 可预测性。

**下一步**
先对 local-motion target 做 conditional-mean / 邻近区域 mask 诊断，确认远离物体处的 Gaussian 归一化是否把微弱、无意义的运动放大；只有 target 本身存在可超过零的强基线后，再测试近场 masked local loss 或与 SE(3) 梯度解耦。

## 实验：V9 SE(3) controlled overfit 与 Cm 误差口径对齐

**假设**
small20 上约 30–45 mm 的 train/test ADE 可能来自 SE(3) 坐标链或模型容量错误；若是如此，V9 在与原 V1 相同的 32 chunks、400 steps 下也无法充分过拟合。

**观察到的失败 / 现象**
V9 small20 训练 5 epochs 的 train ADE 为 29.03 mm，V11 为 43.60 mm；而 Cm full-GRAB 实验最终 train EPE 为 8.31 mm，容易被理解为 InteractionDynamics 连训练集也学不好。

**改动**
不改代码。使用 V9 `global_action_only`、SE(3) translation/rotation loss 和原 overfit 数据设置，固定 32 chunks、batch size 4，训练 50 epochs / 400 steps；无验证集和辅助 interaction loss。

**结果**
产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_140302`。约 200 steps 时单 batch ADE 已为 2.2–2.5 mm；400 steps 的 epoch ADE/FDE 为 1.277/1.598 mm，最后 batch 为 1.378/1.511 mm，translation error 为 0.040 cm、rotation error 为 0.444°。V9 因而能够比原 V1 的 6.71/12.43 mm 更充分地记住相同规模数据。

Cm 的可比 full-GRAB 结果使用 1068 条 train sequences、20 epochs / 226780 steps，最终 train EPE 为 8.31 mm；验证 stride 1–10 的 endpoint EPE 为 2.64–18.45 mm，平均 10.40 mm。它还只从当前手部 5 cm 内采 object points，并以校准尺度的 vector Huber 直接预测单个 endpoint flow。InteractionDynamics small20 只有 12 条 train sequences、5 epochs / 1920 steps，ADE 是未来 8 个累计时刻的平均，并由 8 个增量 SE(3) 复合得到。

**诊断**
SE(3) 解析链、尺度和容量没有阻止拟合；“无法过拟合”的判断来自把 small20 多序列训练均值当成 controlled overfit，并与 Cm 的大数据长训练 endpoint EPE 直接比较。剩余差距主要是未见 sequence 泛化：当前 V9 将 64 个 action patches 全局平均为 `C`，会丢掉 Cm 直接使用完整 1538 点 endpoint hand flow 与近场 object geometry 时保留的接触机制；同时逐步 SE(3) 的小误差会沿 8 步累计。V11 在 small20 上观察到目标冲突，但不是基础路径学不动。

**决策**
不修改 ADE 或 SE(3) 坐标实现，也不以扩大 decoder 容量为下一步。后续比较必须同时报告 controlled-overfit ADE、zero-relative improvement、逐步 EPE，并在相同 sequence 数与训练预算下对齐 Cm/InteractionDynamics；模型研究优先恢复近场局部 action 条件，而不是继续只用 global mean `C`。

## 实验：V12-A wrist 与局部关节动作分解监督

**假设**
现有 ActionToken 已包含手指局部动作，只是累计 raw flow 被 wrist 刚体运动主导；将监督拆成相邻 wrist SE(3) 和各帧 wrist frame 下的局部关节流，应同时超过各自零预测基线。

**观察到的失败 / 现象**
V10 local linear probe 只有 22.8% 改善，V11 的 object-centric local-motion 解码在 20 epochs 后仍差于零预测。现有 `action_reconstruction` 只预测累计 patch mean flow，且 V9/V11 配置将其权重设为零。

**改动**
Dataset 新增相邻 `hand_root_increment_pose_gt`，并把每帧完整 MANO surface 分别变换到自己的 wrist frame 后做稳定顶点差分，得到 `hand_articulation_increment_gt`。不修改 ActionEncoder 输入；从 `[B,8,64,D]` temporal ActionToken 逐 patch 解码 articulated flow，并从 patch 均值解码 root translation 与 rotation。三条 loss 独立配置和记录，物体 SE(3) 主路径保持 V9 global-action-only。

**结果**
运行 `outputs/interactiondynamics/interaction_dynamics_20260810_171704`，small20 共 5 epochs / 1920 steps，耗时 10:06。局部关节流验证 RMSE 从 epoch 1 的 0.0928 cm 降到 epoch 4 最佳 0.0586 cm，但仍差于 zero 0.0393 cm；epoch 5 回升到 0.0987 cm。最佳 checkpoint 的 test 局部流为 0.0599 cm，zero 为 0.0354 cm。wrist translation 验证/test 为 0.2648/0.2555 cm，明显优于 zero 0.6892/0.6637 cm；test wrist rotation error 为 2.75°。test object ADE/FDE 为 42.72/74.35 mm，略弱于 V9 的 42.29/73.38 mm。

**诊断**
目标分解是有效的：root motion 的可预测性与 local articulation 的失败被清楚隔离。现有 encoder 直接读取当前姿态和相对当前帧的 8 步累计位移，确实更容易编码全局刚体运动；只增加正确 local loss 能持续降低误差，但在相同训练预算下仍无法跨序列胜过极强的近零动作基线。结果支持 V12 文档提出的 static pose 与相邻 pose-pair 编码动机。

**决策**
保留 V12-A 作为诊断基线，不继续仅调整当前 cumulative encoder 的辅助 loss 权重。下一实验进入 V12-B/C：显式编码每帧 canonical static surface，再由相邻 pose representation 构造 incremental ActionToken。

## 实验：V12-B/C static pose-pair incremental ActionToken

**假设**
将每帧手表面放入各自 wrist frame，先编码 static local/global PoseToken，再由相邻 pose pair 构造 ActionToken，可以避免 cumulative flow 的 root-motion 主导，并使局部关节流跨序列优于零预测。

**观察到的失败 / 现象**
V12-A 的 wrist motion 可泛化，但 articulated flow test 0.0599 cm 仍差于 zero 0.0354 cm，说明仅给旧 cumulative encoder 增加正确监督不够。

**改动**
Dataset 输出当前加未来共 9 帧 wrist-local stable MANO surface 和 8 个 wrist SE(3) increment。共享 static encoder 在当前 World hand 的稳定 KNN 上产生 64 个 PoseToken 和 global mean code；pair encoder 融合前后 PoseToken、差分、global codes、wrist increment，以及相邻 patch 点差的 mean/max/min，再经 temporal Transformer 得到 8 步 ActionToken。static head 重建 64 个 wrist-local patch centers。由于 articulation target RMS 仅约 0.01–0.05 cm，decoder 使用零初始化并将该 loss 权重设为 100；static loss 权重为 0.01。

**结果**
最终 controlled overfit 为 `outputs/interactiondynamics/interaction_dynamics_20260810_174654`：32 chunks、400 steps、1:52。epoch 50 articulation 为 0.00990 cm，优于 zero 0.01197 cm（改善 17.3%）；static patch-center 为 2.910 cm，优于 centered baseline 2.969 cm；object ADE/FDE 为 3.11/4.15 mm。此前随机初始化或 raw-MSE 同权会让小幅 articulation 梯度被淹没，均未通过 overfit，已撤回。

small20 产物为 `outputs/interactiondynamics/interaction_dynamics_20260810_174924`，5 epochs / 1920 steps，耗时 10:03。按 validation articulation 选择 epoch 4：validation 为 0.01330 cm，zero 0.03930 cm，改善 66.2%；test 为 0.01173 cm，zero 0.03537 cm，改善 66.8%。test wrist translation 为 0.2698 cm，zero 0.6637 cm，改善 59.4%；rotation error 2.17°。validation static patch-center 为 2.763 cm，略优于 centered baseline 2.800 cm；test 为 3.184 cm，差于 baseline 2.943 cm。test object ADE/FDE 为 45.07/78.57 mm，弱于 V9 的 42.29/73.38 mm。

**诊断**
V12 的主要表示假设成立：相邻 wrist-frame surface motion 能形成跨 sequence 可读的详细 ActionToken，且比 V12-A 有大幅改善。static pose 在 validation 通过、test 失败，符合不同 subject morphology 混入 wrist-local surface 的风险。object effect 没同步改善，是因为当前仍使用 `global_action_only`，将 64 个详细 ActionToken 池化后送入 SE(3)；interaction tokens 的 test cosine 仍为 0.9993。

**决策**
保留 V12-B/C 作为新的 ActionEncoder 主候选。下一步不再调整 action reconstruction，而是分开处理两个问题：用 neutral deformation 或 MANO PCA pose 消除 static code 的 morphology；让 interaction/effect 显式消费 local incremental ActionToken，并用 object ADE 与 intervention 验证。

## 实验：V13 Local ActionToken 到 interaction slots

**假设**
V12 已学到可泛化的局部关节动作；用 object-query cross-attention 建立 object-indexed interaction，再压缩为 8 个 embodiment-independent slots，可以让 object SE(3) 真正依赖 local ActionToken identity。

**观察到的失败 / 现象**
V12 的 object effect 仍先对 64 个 ActionToken 取全局均值。V13 首版虽然产生 `[B,8,8,128]` slots，但 SE(3) 对 slots 再取均值；small20 test normal/mean/shuffle/cross-sample ADE 为 43.965/43.939/43.964/44.381 mm，mean 和 shuffle 几乎无影响，未通过核心 intervention。

**诊断**
仅引入 slot 形状不等于 decoder 使用局部结构。首版 object→action 可以退化为集合汇聚，slot→SE(3) 的硬均值又进一步削弱局部配对。需要显式提供 object-hand 空间对应，并移除最终硬均值。

**改动**
保留 V12 PosePairActionEncoder 与全部 action supervision。object token 查询带 object/hand patch 位置编码的 ActionToken，并加入 5 cm 距离 attention bias；64 个 object-indexed interaction tokens 由 8 个 learnable slots 聚合，再由单个 learned effect query 读取 slots 作为 SE(3) 条件。intervention 在进入 local fusion 前执行 normal、patch mean、patch reverse shuffle 和跨 sample 替换。

**结果**
修正版 controlled overfit 为 `outputs/interactiondynamics/interaction_dynamics_20260810_194117`：32 chunks、400 steps、1:58，object ADE 2.896 mm。small20 为 `outputs/interactiondynamics/interaction_dynamics_20260810_194350`：5 epochs / 1920 steps、10:27，最佳 epoch 5 验证 ADE/FDE 44.358/76.638 mm，articulation 0.0170 cm（zero 0.0393）。test normal ADE/FDE 为 43.584/75.845 mm；mean 43.732、shuffle 43.602、cross-sample 44.013 mm，相对 normal 分别退化 0.148、0.018、0.429 mm。

**决策**
保留 V13 结构作为下一步原型，但判定核心验收未通过：normal 虽优于 mean/shuffle，shuffle 差距远不够明显，尚不能证明 local ActionToken identity 真正影响 object effect。当前不进入 V14 multi-embodiment，也不修改已经成功的 V12 ActionToken。

**下一步**
导出并检查两级 attention 的空间分布；设计不会被 object/slot 集合池化抵消的局部 effect 读取方式，再以相同四组 intervention 验收。

## 实验：V14 pretrained Action → World → Cm

**假设**
独立预训练的 Pose/ActionToken 在逐 timestep query 当前 WorldToken 后，其输出本身即可作为 Cm 并预测 object SE(3)；破坏 detailed articulation tokens 应使物体预测明显恶化。

**改动**
Dataset 直接从 cache 的 canonical hand surface 构建独立 `64×32` action atlas，并限制右手样本。新增独立 `PretrainedTokenInteractionModel`：冻结 `pose_encoder_v2_morph.pt` 和 `action_encoder_v1_diagnostic.pt`，把 1 个 12D root SE(3) 投影 token 与 64 个 articulation ActionTokens 拼接；加入当前 object-frame patch 位置后逐 timestep query WorldToken，所得 `[B,8,65,384]` 直接作为 Cm。Effect Query 只读取 Cm 并预测增量 object SE(3)。Runner 使用独立 V14 loss/metrics 分支。

**结果**
运行 `outputs/interactiondynamics/interaction_dynamics_20260810_222251`。32 chunks、100 epochs / 800 steps，耗时 1:55；最终 train ADE/FDE 为 `1.0145/1.3907 mm`，SE(3) translation RMSE `0.02672 cm`、rotation error `0.4576°`，controlled-overfit 拟合验收通过。latest checkpoint 在 eval 模式下 normal ADE/FDE 为 `1.586/2.618 mm`；root-zero 为 `11.721/19.517 mm`；articulation-zero 为 `1.588/2.624 mm`；articulation-mean 为 `1.586/2.618 mm`；articulation-shuffle 为 `1.586/2.618 mm`；cross-sample 为 `2.796/4.756 mm`。

**诊断**
模型容量、SE(3) compose 和 Cm→Effect 路径均可拟合，cross-sample 也确认输出依赖当前动作。然而 root-zero 造成约 10.1 mm ADE 退化，而三种 articulation 干预变化不超过 0.003 mm，说明模型几乎完全依赖 RootToken；64 个 detailed pretrained ActionTokens 的内容和局部 identity 没有被 Effect 使用。仅有拟合结果不足以声称 detailed interaction dynamics 成立。

**决策**
保留 V14 最小实现和 intervention 工具，但判定核心科学验收未通过。按指导不运行 small20、不解冻 ActionEncoder、不增加 slots。下一步若继续，应先对 A→W 与 Effect Query 做 root/articulation 显式分支归因，寻找局部 token 被单 query 汇聚抵消的位置。

## 实验：V15 ActionToken V2 与终点接触监督

**假设**
冻结的 ActionToken V2 flow encoder 提供更纯粹的局部运动表示；从 endpoint articulation Cm 预测 object-patch/hand-patch soft contact，应在 32 个训练 chunk 上明显优于全零预测，并对 articulation token 的 patch identity 敏感。

**观察到的失败 / 现象**
V14 的物体 SE(3) 可充分拟合，但 articulation-zero/mean/shuffle 几乎不改变结果，Effect 主要依赖 RootToken。需要一个直接约束局部手物对应关系的监督。

**改动**
V14 主路径改接冻结的 ActionToken V2 flow encoder：对 9 帧 wrist-local surface 求 8 个相邻 dense flow，按预训练约定乘 100 并用固定 `64×32` atlas 编码。Dataset 返回未来终点手点；Runner 以当前 world object patch center 到未来 endpoint hand patch 内 32 点的最近距离构造 `sigma=1 cm` Gaussian soft contact。模型仅从 `Cm[:, -1, 1:]` 经 `384→384→64` sigmoid head 预测 `64×64` contact，使用 MSE 权重 1；V14 刚体 SE(3) 路径保持不变。

**结果**
controlled overfit 产物为 `outputs/interactiondynamics/interaction_dynamics_20260811_004602`。在最低预算 840 steps 后主动停止：contact MSE 为 `0.002946`，zero MSE 为 `0.003121`，只改善 `4.63%`，F1 为 0。latest checkpoint 的完整 train-set normal MSE 为 `0.00294766`、相对改善 `4.80%`；articulation-zero/mean/shuffle 分别为 `0.00294766/0.00294770/0.00294770`，cross-sample 为 `0.00294767`，没有可测的 identity 敏感性。目标 active fraction 约 `0.4%`，模型收敛到近零预测。

**诊断**
V15 当前 contact 定义极度稀疏，未加权 MSE 的最优捷径是预测全零；因此既未达到指导要求的 50% overfit 改善，也未通过 shuffle intervention。该结果不能支持“ActionToken V2 经 Cm 学到局部接触对应”的结论。

**决策**
保留最小实现作为可复查失败基线。按用户要求，无论门控结果仍启动全量 GRAB 观察性训练：`CUDA_VISIBLE_DEVICES=6,7` 的单个双卡 DDP run，W&B run `4d6a1wk7`，输出 `outputs/interactiondynamics/interaction_dynamics_20260811_005150`，`world_size=2`、global batch 8、计划 15030 steps。该运行不视为门控通过。

全量训练已正常完成 30 epoch / 15030 steps，共用时 2956 秒（约 49.3 分钟），训练集 4001 chunks、验证集 1595 chunks，按 sequence group 划分为 16/4 组。验证轨迹从 epoch 1 的 ADE/FDE `28.74/48.53 mm` 改善至最终 `20.88/34.87 mm`；最佳 ADE 为 epoch 16 的 `20.24 mm`，对应 FDE `33.96 mm`。最终训练 ADE/FDE 为 `6.96/11.25 mm`，说明刚体 effect 能拟合，但存在明显训练—验证差距。总验证 loss 最低点在 epoch 6（`0.22195`），之后最终回升至 `0.25302`。

接触分支没有随规模化训练变好。配置按 `val/contact/mse` 选择 best，因此 `best.pt` 停留在 epoch 1：MSE `0.00530083`，相对固定 zero MSE `0.00542484` 仅改善 `2.29%`；最终 MSE 为 `0.00534916`，改善缩小到 `1.40%`，precision/recall/F1 始终为 0。最终训练 contact MSE `0.00412905`，相对训练 zero MSE `0.00430433` 改善约 `4.07%`。这与 controlled overfit 的近零解一致：扩大数据和训练预算没有绕过 `0.59%` train / `0.73%` val active fraction 导致的监督不平衡。

因此全量结果只支持“V15 刚体轨迹分支可训练”，不支持“ActionToken V2 经 Cm 学到了局部接触对应”。W&B 在线记录：`https://wandb.ai/hitsz-oyx/ref2dex/runs/4d6a1wk7`。

**下一步**
先在同一 32-chunk controlled overfit 上处理接触监督不平衡（正样本加权或按每个 object patch 归一化），要求显著超过 zero baseline 且 F1 非零；通过后再重复 articulation shuffle 验收，不直接继续扩大训练。

## 实验：V15.1 endpoint 时间对齐、oracle 与 8-step readout

**假设**
V15 用 `H(t+8)` 对 `O(t)` 构造 contact，物体运动会使 GT 错位；修正为 `H(t+8)` 对 `O(t+8)` 后，若 endpoint oracle 可拟合而 Cm 仍失败，则瓶颈在不平衡和 Cm readout，而非 GT/loss 本身。

**改动**
Runner 使用 `world_obj_points_object + obj_disp_chunk_gt[:, -1]` 得到当前物体系下的 endpoint object points，再沿 World encoder 的稳定 `obj_knn_idx` 聚合 patch center。新增诊断脚本输出新旧 contact 热图和 endpoint-distance MLP oracle。为后续对照增加可关闭的 `contact_positive_weight`，以及对每个 hand patch 的 8 个 Cm 加入 time embedding 后做 softmax attention 的有序 readout；冻结 ActionToken 不变。

**结果**
32 chunks 中，旧/对齐 GT active fraction 为 `0.412%/0.481%`，zero MSE 为 `0.003180/0.003716`，object endpoint 平均移动 `9.20 mm`。500-step endpoint-distance oracle 的 MSE 为 `2.998e-6`，相对 zero 改善 `99.92%`，precision/recall/F1 为 `0.991/1.000/0.995`，证明对齐后 GT 和 MSE 可拟合。热图保存在 `output/research/InteractionDynamics/v15_1_contact_alignment/contact_alignment.png`。

仅修时间对齐的原 V15 运行 `interaction_dynamics_20260811_074943`，1504 steps 后 train contact MSE `0.003480`，zero `0.003716`，改善 `6.34%`，F1 仍为 0；ADE `1.48 mm`。`200×` 正区域加权运行 `interaction_dynamics_20260811_075726`，最佳 F1 为 `0.457`（precision `0.314`、recall `0.847`），但 raw MSE `0.004876` 差于 zero，说明过强权重把近零解改成了高 recall/低 precision 解。

8-step 有序 readout 运行 `interaction_dynamics_20260811_080156`，最佳 F1 `0.462`（precision `0.301`、recall `0.993`），raw MSE `0.005385`，相对只用最后一步的 F1 只增加 `0.004`。best checkpoint 的 normal/articulation-zero/mean/shuffle/cross-sample F1 为 `0.4615/0.4381/0.4560/0.4588/0.4595`；shuffle 只下降 `0.0027`，未通过 identity 验收。

**诊断**
时间错位是明确 bug，但不是 V15 失败的主要原因。正区域加权证明 Cm 可以输出非零 contact，但当前表示/readout 不能精确定位 patch correspondence；简单按 patch 汇聚 8 步也未弥合与 oracle 的差距。

**决策**
保留 endpoint 时间对齐修复、诊断/oracle 脚本和两个可关闭的对照开关。加权和 8-step readout 都不视为科学验收通过，不启动全量训练。

**下一步**
若继续，先在 oracle 和 Cm 之间加入 endpoint hand/object patch geometry 的受控对照，并对正区域权重做小范围定量扫描；验收同时要求 raw MSE 优于 zero 且 shuffle 明显退化。

## 实验：V16 object-centric interaction field

**假设**
让当前 object patch 逐步查询带动态手位置的冻结 ActionToken V2，并直接监督 endpoint object-contact field，可以绕过 V15 的 hand-indexed readout 瓶颈，并建立可测的 articulation 依赖。

**观察到的失败 / 现象**
首轮连续 32 chunks 在训练态达到 MSE `3.49e-4`、F1 `0.994`，但 eval running stats 下 MSE `0.08746`、F1 `0`。只把 world encoder 的两个 BatchNorm 切回 batch stats 即恢复训练指标，证明小批量 BatchNorm 形成了无效捷径。固定预训练 running stats 后，连续集 best MSE 为 `2.22e-4`、F1 `0.991`，但 articulation-zero 和 cross-sample 几乎不恶化。

**诊断**
连续集的 32 个窗口来自同一序列；将 GT 跨样本 roll 后 MSE 仅 `2.68e-4`、F1 仍为 `0.986`，因此干预没有信息量。每序列均匀取 2 个窗口后仍为 32 样本，cross-sample GT MSE 增至 `0.07237`、F1 降为 0；仅凭当前几何的 oracle 相对 zero 只改善 `0.77%`。

**改动**
新增 `[B,8,64,384]` object-centric field：object token 加当前位置编码后查询 articulation token，手 patch 位置使用当前位置加逐步 action displacement；RootToken 不进入接触分支。endpoint head 输出 `[B,64]`，target 为 V15.1 对齐接触矩阵沿 hand 维取 max，使用未加权 MSE。V16 配置固定 world BatchNorm running stats，并增加每序列最多 2 个窗口的多序列对照配置和两个诊断脚本。

**结果**
有效运行 `outputs/interactiondynamics/interaction_dynamics_20260811_111422` 使用 32 chunks、1200 steps，耗时 10:37。best checkpoint（step 1184）normal MSE `9.87e-5`，zero MSE `0.036186`，改善 `99.72%`，F1 `1.0`。articulation-zero MSE `2.01e-4`、F1 `0.961`，相对 normal 恶化 `103%`；cross-sample MSE `7.51e-4`、F1 `0.920`，恶化 `660%`。articulation-mean MSE `1.30e-4`；patch shuffle MSE `9.94e-5`，基本不变。

**决策**
保留 V16 最小结构、BatchNorm 固定和多序列 controlled sampling。V16 已通过 object-contact 可拟合性、非零 F1、articulation-zero 与 cross-sample 方向性门槛；但 patch shuffle 仍不敏感，因此只说明模型使用样本级 articulation 内容，尚不能声称使用了 hand-patch identity，也不据此启动全量训练。

**下一步**
在同一多序列集合上检查 object→action attention 与动态位置编码消融，解释为何 zero/cross 有效而 patch shuffle 无效，再决定是否加入空间 attention bias。

## 实验：V16.1 解析 Interaction Field 与 MANO self-inverse

**假设**
object-frame 的解析动态场 `Y=[r,u]` 比冻结 ActionToken 更适合约束物体附近交互；加入 wrist-object 相对位姿 `G` 后应能排除 MANO 梯度、坐标和优化实现错误。

**观察到的失败 / 现象**
learned contact/Cm 的监督与表示瓶颈互相耦合，无法判断 interaction target 本身是否足以驱动执行体优化。

**改动**
新增纯 PyTorch soft correspondence、relative geometry `r` 和由时刻 t 权重跟踪的 relative motion `u`，使用 128 个 object-frame FPS anchors 和 `tau=1.5 cm`。固定首帧与 GT beta，优化未来 8 帧 MANO pose/root；比较 `r/u/r+u/G+r+u` 和冻结 ActionToken V2。Adam 学习率 `0.003`、500 步、无 smoothness。另增加固定 beta 方向偏移的 cross-morphology 对照和结果汇总脚本。

**结果**
5 个不同序列 chunk 的 GT+有界噪声中，Full ADE 为 `0.312/0.620/0.241/0.717/0.326 mm`，平均 `0.443 mm`，5/5 通过 1 mm local-gradient gate。Interaction ADE 为 `0.705/2.290/0.668/2.636/1.052 mm`，对应 Y 的 r/u RMSE 均约 `0.004–0.010 cm`；表面未完全重合的样本仍可非常接近目标 Y。

current-frame repeat 的 stationary ADE 平均约 `109.3 mm`。Interaction 最终 ADE 为 `0.974/40.426/1.814/20.680/2.670 mm`，平均 `13.313 mm`，所有样本均改善超过 50%，但两个超远初始化在 500 步内未进入好 basin。Full 为 `0.657/9.107/1.023/5.143/1.333 mm`，平均 `3.453 mm`，3/5 达到 3 mm。Action baseline 平均 ADE `112.005 mm`，与 stationary 基本相同。单样本消融中 r-only ADE `2.675 mm` 但 FDE 保持 `16.726 mm`，因为当前定义的 `r_t` 不直接观察最后一帧；u-only ADE/FDE 为 `2.707/3.857 mm`。

candidate beta 固定方向偏移 `0.75/1.5/2.0` 时，Interaction 将初始 ADE `8.40/13.53/14.15 mm` 降至 `3.90/7.73/8.41 mm`，contact F1 为 `0.974/0.937/0.937`；Action 最终 ADE 为 `8.46/15.03/14.56 mm`，contact F1 为 `0.746/0.493/0.537`。

进一步先对 controlled set 的 32 chunks 解析筛选：13 个存在正接触，最高 active fraction 为 `12.01%`，其余 19 个为 0。选取 6 个不同序列的有接触样本，在统一 beta offset `1.5` 下成对比较。Interaction 最终 ADE 平均 `7.323 mm`、r/u RMSE `0.417/0.225 cm`、contact F1 平均/中位数 `0.918/0.939`；Action 最终 ADE `15.560 mm`、r/u RMSE `1.103/0.844 cm`、contact F1 `0.400/0.397`。6/6 样本的 Interaction F1 均高于 Action，且 Interaction 均改善初始 ADE，Action 则 6/6 恶化。

固定相同初始化、只把 beta 方向种子从 `43` 改为 `143` 后，6 个样本的 Interaction/Action 平均 ADE 为 `2.440/8.369 mm`，r RMSE 为 `0.131/0.636 cm`，u RMSE 为 `0.052/0.730 cm`，contact F1 为 `0.947/0.632`。两个 beta 方向合计 12 对实验中，Interaction contact F1 为 12/12 更高；方向 143 对两种方法都更容易，但没有改变排序。

**诊断**
Full 的局部结果确认 autograd、MANO forward、单位与 object-frame 坐标链有效。Y 对物体附近 interaction 的约束明显强于 ActionToken，且跨 morphology 的 contact 保真优势已在 6 个序列上重复。跨 morphology 的 surface ADE 同时包含不可消除的形态差异，不能替代 Y/contact 主指标。Y 已接近而 MANO 表面仍有差异，符合其主动忽略远离物体自由度的设计；repeat 的大误差则同时包含 500 步优化 basin 限制。无正接触样本上 F1=0 本身没有判别力，后续需同时报告 target active fraction。

**决策**
保留解析 Y、self-inverse 和 cross-beta 路径。判定 local-gradient gate 通过、Interaction 的 50% improvement gate 通过；Full repeat 的严格 3 mm gate 为 3/5，尚非全样本稳定。cross-morphology 优势已跨 6 个序列、2 个 beta 方向重复，当前证据支持继续到可视化和不同真实 subject，但不把 exact MANO reconstruction 当作 Y 的唯一成败标准。

**下一步**
在更多 subject 或 beta 方向上重复 cross-morphology，并可视化 Y matching 得到的等价手型；之后再决定是否进入 Robot FK。

## 实验：V16.2 Y-Teacher V1 收尾与公平 Action 对照

**假设**
补充 terminal `r_8` 和真实 soft distance `d` 后，解析 Y 不破坏局部 inverse，并在不同 morphology 下比完整动作模仿更稳定地保持目标 interaction。

**改动**
Teacher 冻结为 `r[9,128,3]`、`d[9,128]`、`u[8,128,3]`，统一使用 cm loss，contact 由 `exp(-d²/(2σ²))` 派生。新增 Direct Full Action（root 相邻 SE(3)+wrist-local surface increment）和 ActionToken+Root；冻结 Action encoder 参数但不在 candidate forward 使用 `no_grad`，保留输入梯度。

**结果**
同一样本 GT+noise 500 steps：`r/r+d/Y/Full` ADE 为 `0.577/0.665/0.697/0.168 mm`，FDE 为 `0.329/0.460/0.533/0.263 mm`。current repeat 的 stationary ADE 为 `11.389 mm`；Direct Action、ActionToken+Root、Y、Full 最终 ADE 为 `0.133/0.126/0.967/0.638 mm`。

candidate beta offset 1.5、6 个序列×2 个 beta 方向共 12 对中，Y / Direct Action / ActionToken+Root 的 contact F1 均值为 `0.619/0.228/0.228`，future ADE 为 `5.725/9.526/9.509 mm`，tip MPJPE 为 `3.838/12.102/12.070 mm`。Y 的 `d/r RMSE、ADE、tip MPJPE` 均 12/12 优于两种 Action，contact F1 为 9/12 更高、3/12 平局；平局来自按旧 `|r|` 筛选但按新 `d` 评价后没有正接触的样本。

**决策**
Y-Teacher V1 的定义、梯度和跨 morphology interaction 优势通过 Phase A 最小验证，予以冻结。后续 contact-active 样本必须按 `d` 重新筛选。

## 实验：V16.2 PCA 与 K=8 压缩原型

**假设**
coordinate-conditioned slot autoencoder 能以 8×128 scalars 压缩 Y，并超过忽略 anchor coordinate 对应关系的 flatten PCA。

**改动**
新增 sequence-disjoint GRAB Y 采样、PCA-256/512/1024 baseline，以及 1 层 latent cross-attention、1 层 self-attention、coordinate-conditioned decoder 的 K=8 原型。训练和验证只使用 `L_r+L_d+L_u`。

**结果**
PCA 使用 1280 train / 256 unseen-sequence chunks。256/512/1024 维的 `r RMSE` 为 `1.547/1.473/1.346 cm`，`d RMSE` 为 `1.632/1.579/1.494 cm`，`u RMSE` 为 `0.251/0.240/0.262 cm`，contact F1 为 `0.007/0.005/0.021`。

K=8 首版 512 chunks、20 epochs 的验证 `r/d/u=3.002/3.044/0.909 cm`、F1 `0.003`，差于 PCA。32-chunk controlled overfit 将学习率从 `1e-4` 提到 `1e-3` 并训练 500 epochs / 4000 steps 后，训练 `r/d/u=0.617/0.337/0.092 cm`、contact F1 `0.212`；仍未达到近似 oracle reconstruction，未通过压缩 gate。

**诊断**
PCA 暴露了跨物体 flatten anchor index 缺乏固定语义的问题。slot 原型梯度和输入梯度成立，但当前单层 decoder/优化尚不能在 32 chunks 上充分拟合；因此验证失败不能解释为“Y 不可压缩”，也不能声称 compact C 已成立。

**决策**
保留 PCA 和最小 autoencoder 作为失败基线，不运行 K=16/4 sweep、不做 reconstructed-Y MANO inverse、不进入 latent metric。下一步只先解决 K=8 controlled overfit。

## 实验：V16.3 Teacher 数据链 parity 与 sequence-balanced sampling

**假设**
Phase B 使用固定 current-object-frame 构造未来手点，与 inverse 使用逐帧动态 object frame 的 Y-Teacher V1 不一致；同时按 sequence 串接后截断会造成样本组成偏置。

**改动**
Dataset 新增 `action_hand_points_object_sequence[9,1538,3]`，每帧使用自身 `obj_root_pose_world`；PCA 和 autoencoder 只消费该字段。train/eval 先按 sequence disjoint 划分，再对每序列分别做 32/16 个时间均匀限额，最后 round-robin 组成总样本。新增 raw MANO↔Dataset parity 诊断。正式 `PretrainedActionAdapter` 移除冻结 encoder forward 外层的 `no_grad`，参数冻结不变、输入梯度保留。

**结果**
真实 sample 0 上，raw MANO 与 Dataset 的 object-frame surface 最大绝对差 `1.776e-7 m`；`r/d/u` 最大差分别 `2.198e-7/6.333e-8/6.359e-8 m`，通过 `<1e-5 m` parity gate。聚焦测试验证动态物体系能消除 hand-object 共同平移、均匀限额与 round-robin 顺序正确、冻结 shared flow adapter 后输入梯度非零。

full 配置在 80/20 sequence split 后实际为 16/4 个 train/eval sequences。请求 512/128 时，在每序列 32/16 上限下实际得到 492/64 chunks；train 每序列 12–32 个、eval 固定 16 个，sequence overlap 为 0。采样器不会为了凑满总数突破单序列上限。

**决策**
保留动态物体系字段、balanced sampling 和 shared adapter 梯度修复。V16.2 旧 compression 泛化结果因 teacher 定义与采样不一致而作废，不再用于判断 C。

## 实验：V16.3 K=8 的 1→4→32 controlled overfit

**假设**
统一 Teacher 后，如果 1-sample 仍无法拟合，则首要瓶颈在 decoder 表达或优化，而不是 K=8 latent capacity。

**改动**
固定 K=8、128 维 slots、`L_r+L_d+L_u` 和 `lr=1e-3`。先复跑原线性输出 head，再只把 geometry/motion head 改为 `128→512→output` pointwise MLP；不改变 encoder、slot 数或 loss。

**结果**
旧线性 head 的 1/4/32 训练 `r/d/u` 分别为 `0.352/0.199/0.0476`、`0.529/0.299/0.0683`、`0.522/0.416/0.101 cm`，未通过最小 gate。MLP head 的 1-sample 在 3000 steps 后为 `0.0485/0.0355/0.0138 cm`、contact F1 `0.800`；4-sample 为 `0.103/0.0741/0.0187 cm`、F1 `0.522`。32-sample 训练 500 epochs / 4000 steps 后为 `0.447/0.363/0.0621 cm`、F1 `0.0817`。

**诊断**
MLP 对 1-sample 带来明显改善，证明输出 head 表达是一个真实瓶颈；但误差随样本数快速上升，32-sample 仍远未充分拟合。当前证据不足以区分 K=8 容量、latent routing 和优化三者，也不能评价 unseen-sequence 泛化。

**决策**
保留 pointwise MLP head，但判定 32-chunk gate 未通过。按 V16.3 停止 512-chunk 泛化、PCA 重跑、K=16/4 sweep、reconstructed-Y inverse 和 C-space metric；下一实验只调查 K=8 在 32 chunks 上的 train reconstruction 瓶颈。

## 实验：V16.4 Object-Effect 条件交互场扩散

**假设**
冻结的解析 Y 可作为 knowledge target；在单物体 16 个 chunks 上，条件扩散应能学习 `p(Y|O,E)`，且 correct Effect 生成应优于 shuffled/zero Effect。

**改动**
暂停 Y compression。将 `r[9,M,3]、d[9,M]、u[8,M,3]` 按 anchor 整理为 `[M,60]` 并按通道标准化。Object encoder 对 128 个 FPS anchors 的 `[xyz,normal]` 做 pointwise MLP，同时 max-pool global token；Effect 使用 8 个相邻 object SE(3) 的 `[translation_cm, rotation_vector]`。6 层、256 维 Transformer 对 noisy Y 做 self-attention，并 cross-attend `[global object, 8 effect tokens]`。模型只读取 anchor coordinate/local geometry，不含 index embedding；训练时逐样本随机打乱 anchors。

**观察到的失败 / 现象**
1000 steps 后 noise MSE 已从 `1.411` 降至约 `0.05–0.13`，但 ancestral DDPM 采样累积误差，生成达到上千厘米。单变量改为确定性 DDIM 并裁剪标准化 clean prediction 后数值稳定，网络与 loss 不变。

**结果**
16 chunks 来自同一 sequence 的时间均匀位置；15/16 chunks 有正接触，contact active fraction `1.237%`。Effect translation 三轴 std 为 `1.210/0.925/1.196 cm`，rotation std 为 `0.0128/0.0106/0.0299 rad`，shuffle Effect RMSE `0.936`，因此 condition 并非近似常量；shuffle Y 自身的 `r/d/u` RMSE 为 `5.758/9.406/1.342 cm`、F1 `0.101`。

4000 steps 后训练 noise MSE 可达 `0.011–0.05`。DDIM correct Effect 生成 `r/d/u=4.725/7.628/1.168 cm`、contact F1 `0.0287`；shuffled Effect 为 `4.750/7.773/1.183 cm`、F1 `0.0366`；zero Effect 为 `4.823/7.777/1.179 cm`、F1 `0.0325`。固定噪声 denoise 在 `t=50` 的 correct/shuffle/zero MSE 为 `0.01948/0.02122/0.02080`，存在很弱的中噪声条件依赖；`t=99` 为 `0.01226/0.01216/0.01203`，纯噪声端没有 correct condition 优势。

**诊断**
epsilon prediction 训练问题本身可优化，permutation-equivariant 路径和采样数值也已打通；但网络在高噪声端学习近似 unconditional prior，无法从纯噪声依靠 Effect 选择对应 interaction mode。生成虽略优于 shuffle-Y 的 raw RMSE baseline，但 contact 与 condition sensitivity 都未通过。

**决策**
保留最小 conditional diffusion、DDIM 稳定采样和 intervention 诊断作为失败基线。Gate A（生成接近训练 Y）与 Gate B（correct condition 明显优于干预）均未通过；按 V16.4 不进行 Object intervention、unseen-object、generated-Y MANO inverse、Robot FK 或大规模训练。下一实验应只针对高噪声条件依赖，不同时修改 Object/Effect encoder 与 backbone。
## 实验：V16.5 当前状态条件的短时 Interaction Diffusion

**假设**
V16.4 从纯噪声生成完整交互场时容易退化为无条件先验；将干净的当前 `S_t=[r_t,d_t]` 作为 condition，只生成 `H=4` 的 `[u_t,S_{t+1},...,u_{t+3},S_{t+4}]`，应改善 controlled generation，并使 correct state 明显优于 shuffle/zero state。

**观察到的失败 / 现象**
V16.4 的 16-chunk 实验虽能把 noise MSE 降低，但 correct Effect 生成的 `r/d/u` 仍为 `4.725/7.628/1.168 cm`，且 Effect 干预差异很弱。

**诊断**
V16.5 的 state 干预表明模型确实读取当前交互状态，但 generated future 仍差于 persistence baseline；Effect 数据也具有足够变化，因此当前失败不能归因于常量条件。主要问题仍是 epsilon predictor 从纯噪声端恢复精确 future 的生成质量，而不是 state condition 完全被忽略。

**改动**
新增 `H` 可配置的 state/future packing、`32` 邻域局部 object PointMLP、无 anchor index embedding 的 permutation-equivariant self-attention，以及由 diffusion timestep 和 endpoint SE(3) 联合调制全部 block 的 AdaLN。训练随机同步置换 anchors，并加入 correct/shuffle/zero state 与 Effect 的固定噪声生成干预。新增 persistence 与条件多样性诊断。

**结果**
16 个单序列均匀 chunk、4000 steps 后 noise MSE 从 `1.468` 降至 `0.00879`。correct 生成 `u/r/d RMSE=0.793/2.424/3.497 cm`，contact F1 `0.0919`；state shuffle 为 `0.866/3.054/4.181 cm`、F1 `0.0680`，state zero 为 `0.862/3.405/5.134 cm`、F1 `0.0289`。Effect shuffle/zero 与 correct 基本相同。persistence baseline 为 `0.688/1.660/2.646 cm`、F1 `0.6968`，仍明显更好。Effect shuffle RMSE `3.852`，state/future shuffle RMSE 为 `1.383/2.853`，排除条件无变化的解释。

**决策**
保留 state-conditioned 数据定义、局部 object encoder 和 AdaLN 实现作为下一轮诊断基础。Gate B 的 current-state dependency 成立，但 Gate A 未超过 persistence，Gate C Effect dependency 未成立；按顺序停止，不做 object intervention、多 seed、MANO 或 Robot inverse。
## 实验：V16.6 Active + Next Meaningful Effect

**假设**
V16.5 endpoint Effect 缺少任务是否 active 以及 approach 阶段尚未发生的 future meaningful motion；改用 `G=[active,next meaningful 4-step SE(3)]` 后，模型应在 interaction-changing 时刻表现出 Goal dependency，并在 dynamic subset 超过 persistence。

**观察到的失败 / 现象**
首版按首次进入 10 cm 近场定义 active，把 `airplane_fly_1` 从 117 cm 持续接近物体的 approach 错标为 inactive；active 终点也遗漏 release 后 withdraw。该 sanity 未通过，未用于训练。

**诊断**
active interval 必须同时向前回溯持续接近段、向后延伸持续远离段。修正后选用 `cup_lift`：active `[25,872]`，frame 25 手物距离 `112.6 cm`、object 当前静止但 `k*=78`，正确表达 approach；inactive-before/after Goal 严格为零。3-frame object motion 分布支持 `0.2 cm / 1°` 阈值。32 样本的 future-persistence change 为 `0.069–0.489 cm`，故以 `δ=0.2 cm` 得到 16 dynamic/16 static。

**改动**
新增只读取完整 object pose 的 next meaningful motion extraction、active interval 几何标注、分阶段 32-frame sampler、25D Goal encoder 和 V16.5 同构 AdaLN diffusion。加入 state shuffle、motion shuffle/zero、active zero/flip、dynamic/static persistence 指标，以及在 Goal dependency 失败后按指导运行的 active-only 对 active+motion deterministic clean regression。

**结果**
32-chunk、4000-step diffusion noise MSE 从 `1.276` 降至 `0.01535`。correct dynamic `u/r/d=0.207/41.295/68.082 cm`、F1 `0`，远差于 persistence 的 `0.132/0.472/0.252 cm`、F1 `0.807`；motion shuffle/zero 和 active zero 与 correct 基本相同。State shuffle 的 dynamic `r/d=46.584/78.283 cm`，保留弱 state dependency。clean regression 的 active-only dynamic `r/d=1.605/2.755 cm`；active+motion 为 `1.590/2.332 cm`，motion shuffle 后退化至 `2.386/3.646 cm`，说明 `Xi` 具有可读辨识度，但 clean prediction 仍差于 persistence。

**决策**
保留 Goal extraction、active interval sanity 和分层评估。Gate B/C/D 未通过，V16.6 formulation 尚未成立；证据指向 absolute-future diffusion generation 是主要瓶颈，而非 motion Goal 完全无信息。按本轮范围停止，不实现 residual diffusion、inverse 或大规模训练。

## 实验：V16.7 Clean Persistence Residual Regression

**假设**
V16.6 的绝对未来目标含有大块已由当前状态确定的静态分量；固定相同数据、Goal 与 backbone，改为预测 `R=F_GT-F_persist`，应更容易在 dynamic subset 超过 persistence，并表现出明确 Goal 依赖。

**观察到的失败 / 现象**
V16.6 diffusion 的 dynamic `r/d=41.295/68.082 cm`；clean absolute regression 虽能读取 motion Goal，但 dynamic `r/d=1.590/2.332 cm`，仍差于 persistence 的 `0.472/0.252 cm`。

**诊断**
同一 `cup_lift` 32-frame set 仍为 16 dynamic / 16 static。Residual magnitude 的 min/median/mean/max 为 `0.069/0.194/0.224/0.489 cm`，dynamic/static mean 为 `0.323/0.126 cm`；`u/r/d` residual std 为 `0.103/0.357/0.188 cm`，未见尺度异常。远场诊断中 14 个 `<10 cm` 样本的 anchor-r variance / residual RMS 为 `3.364 cm² / 0.257 cm`，18 个 `>100 cm` 样本为 `7.480 cm² / 0.198 cm`；当前分层采样没有覆盖中间距离，本轮不据此修改输入。

**改动**
新增无 noisy future、timestep 或 sampler 的 6-layer AdaLN residual regressor，只读取当前 `S`、当前 object anchor/local patch 和 25D Goal。以 residual 自身统计量标准化并使用单一 MSE；最终指标统一在 `F_persist+R_hat` 上计算。加入 motion zero/shuffle/reverse、active zero、dynamic-active 交集、static predicted-residual RMS、精确恢复与 permutation-equivariance 测试。

**结果**
32 chunks、4000 steps 后，correct dynamic `u/r/d=0.085/0.244/0.109 cm`、F1 `0.884`，全面优于 persistence 的 `0.132/0.472/0.252 cm`、F1 `0.807`，Gate A 通过。static correct `r/d=0.146/0.091 cm`，也优于 persistence 的 `0.182/0.103 cm`；predicted residual RMS 为 `0.103 cm`，GT static residual mean 为 `0.126 cm`，Gate B 通过。

9 个 dynamic-active 样本的 correct `u/r/d=0.013/0.064/0.030 cm`；motion zero 为 `0.119/0.496/0.222`，shuffle 为 `0.127/0.501/0.228`，reverse 为 `0.183/0.686/0.325 cm`，active zero 为 `0.127/0.529/0.252 cm`。强干预均显著恶化，Gate C 通过。inactive correct predicted residual RMS 为 `0.096 cm`，尚未严格趋近零，但未阻碍本轮 controlled gate。

**决策**
保留 clean residual formulation。它在完全相同数据和 Goal 下首次同时超过 persistence、保持 static，并建立强 motion/active Goal dependency，支持“绝对未来 target 是 V16.6 clean regression 的主要瓶颈”。结论目前只限单序列 controlled overfit；按 V16.7 范围不进入 residual diffusion、扩大数据或 inverse。

## 实验：V16.8 v-prediction Conditional Residual Diffusion

**假设**
V16.7 已证明 residual 可预测；把同一 normalized residual 改为 v-prediction，并使用不除以小 `sqrt(alpha_bar)`、无 x0 clamp 的 deterministic sampler，应能稳定从纯噪声生成 residual，超过 persistence 且接近 clean regression。

**观察到的失败 / 现象**
旧 V16.6 epsilon diffusion 虽能降低 noise MSE，但 absolute-future 采样达到几十厘米并忽略 Goal。V16.8 训练期间 v-MSE 降至 `0.0543`，而固定四样本 sampling residual RMSE 在 `1.19–1.62` normalized units 波动，再次说明训练 loss 不能代替生成指标。

**诊断**
通过 V16.7 checkpoint 核对完全相同的 32 个 indices。v transform roundtrip 最大误差小于 `1e-6`；reverse 全程 finite，`x_t RMS` 从约 `1.005` 平稳到 `0.854`，最大 x0 RMS `0.855`，未出现旧 sampler 的尺度爆炸。Oracle denoising 的 t10/30/50/70/90 normalized RMSE 为 `0.062/0.122/0.167/0.214/0.368`：模型有去噪能力，高噪声端精度较弱；pure-noise sampling 的损失主要来自迭代生成而非数值公式错误。

**改动**
新增 6-layer AdaLN residual v-predictor、v transform/recovery、无 clamp 的 eta=0 sampler。训练每 500 steps 对固定四样本完整采样；最终使用相同 initial noise 比较 correct、motion zero/shuffle/reverse 与 active zero，并记录每 10 timestep 的 stability、oracle denoising 和四 seed 指标。未加入 CFG、加权 loss 或其他 V16.8 禁止项。

**结果**
4000 steps 后 correct dynamic `u/r/d=0.124/0.403/0.178 cm`、F1 `0.724`，优于 persistence 的 `0.132/0.472/0.252 cm`、F1 `0.807` 中三个 RMSE，Gate 2 通过；但相对 V16.7 `0.085/0.244/0.109 cm`，`r/d` 略高于 `1.5×` 工程线 `0.37/0.17 cm`，Gate 3 未通过。四 seed dynamic `r/d` 均值为 `0.377/0.171 cm`，std `0.055/0.020 cm`，仍位于门槛附近且存在可测随机波动。

dynamic-active correct `u/r/d=0.032/0.222/0.089 cm`；motion zero 为 `0.132/0.537/0.226`，shuffle 为 `0.145/0.536/0.241`，reverse 为 `0.182/0.724/0.329`，active zero 为 `0.137/0.551/0.236 cm`。所有强干预均恶化，表明 Goal dependency 被保留；因 Gate 3 已失败，该事实只作诊断，不改变停止顺序。

**决策**
保留 v-prediction、无 clamp sampler、训练中采样和 oracle/stability 诊断。Gate 0/1/2 通过，Gate 3 未通过，V16.8 尚不能认定达到 clean residual 的 controlled 精度；按指导停止，不扩大数据、不验证 diversity、不进入 inverse。下一步若继续，应只调查 high-noise denoising 与迭代 sampling 的精度损失。

## 实验：V16.9 高噪声训练与采样误差归因

**假设**
V16.8 与 clean V16.7 的差距主要来自 4000 steps 不足、高噪声条件训练不足和多步误差累积。按单变量顺序比较 uniform-10k、50% uniform + 50% `t∈[70,99]`、固定 checkpoint 的 100/50/25/10-step skipping；前三项仍不足时才加入 `t≥70` 的 x0 auxiliary loss。

**改动**
训练入口新增 uniform/half-high timestep sampling 和可选 high-noise x0 auxiliary。v sampler 支持使用原 100-step alpha schedule 的正规 timestep skipping，不通过截断循环近似；新增 oracle skipping 测试、训练配置对照和固定 seed sampling-step 诊断。所有实验继续核对 V16.7 的完全相同 32 indices。

**结果**
uniform 从 4k 延长到 10k 后，100-step dynamic `u/r/d` 从 `0.124/0.403/0.178` 改善到 `0.101/0.321/0.147 cm`，F1 从 `0.724` 到 `0.879`；t90 oracle 从 `0.368` 降到 `0.297`。四 seed dynamic `r/d=0.325±0.016/0.147±0.009 cm`，明显比 4k 的 `0.377±0.055/0.171±0.020` 更准且稳定。延长训练是本轮最大有效变量。

half-high-10k 的 t70/t90 oracle 为 `0.120/0.183`，优于 uniform-10k 的 `0.155/0.297`，确认高噪声训练比例会直接改善高噪声 oracle；但最终 dynamic `r/d=0.355/0.199 cm`，差于 uniform，说明 50% high-noise 牺牲了全程平衡。high-noise x0 auxiliary 的 100-step `r/d=0.309/0.164 cm`，r 略好、d 明显差于 uniform，也不构成整体收益。

固定 uniform-10k checkpoint 与 initial noise，100/50/25/10 steps 的 dynamic `r/d` 分别为 `0.321/0.147`、`0.315/0.146`、`0.314/0.145`、`0.307/0.150 cm`。half-high 对应 `0.355/0.199`、`0.353/0.196`、`0.354/0.193`、`0.325/0.161 cm`。减少调用次数仅小幅改善 uniform，说明迭代误差累积存在但不是主要瓶颈。

**决策**
保留 10k uniform 训练和正规 skipping，默认优先 uniform-10k；half-high 与 x0 auxiliary 仅作为失败/归因选项保留。当前最佳 uniform-10k + 10-step 为 `u/r/d=0.097/0.307/0.150 cm`，已接近但尚未达到指导建议的 `r≤0.30,d≤0.14 cm`，因此不扩大数据。下一步应继续聚焦高噪声条件映射与训练波动，而不是增加 sampler steps 或 contact loss。

## 实验：V17 Full-Data 启动审计与训练配置

**假设**
V16 的 residual/Goal formulation 在完整自然 demonstration 分布上可以跨未见 sequence 泛化；正式训练前先排除 chunk split 泄漏、数据阶段缺失和不合理 global batch。

**诊断**
旧 full manifest 含 25 个 dominant-hand 文件但只对应 20 个 demonstration 目录；按 hand 文件 split 会让同一 demonstration 的 left/right 泄漏。V17 改为以 `path.parent` 为 sequence 单元，固定 seed 42 得到 16/2/2 demonstration，hand 文件和 chunk 均无跨 split 重叠。manifest 仅用于确定 dominant hand 文件集合，不沿用其窗口筛选；每个文件恢复全部合法 frame。

**改动**
新增 full-frame Dataset、固定 sequence split、train-only normalization、BF16 四卡 DDP、DistributedSampler、deterministic/diffusion 联合训练、25-step validation、best/latest checkpoint 和 wandb online。因在线 NPZ/Y 构造使 smoke 极慢，增加语义不变的 tensor shard cache；cache 和 checkpoint 均放在忽略的 output 下。

**结果**
合法 chunks 为 train/val/test `7909/1471/1159`。各 split 均匀审计 512 个样本：train active/dynamic ratio `95.9%/60.4%`，val `97.5%/67.4%`，test `94.3%/89.3%`；train residual RMS median/mean/p90 为 `0.287/1.396/5.729 cm`。距离分桶覆盖 `<10、10–30、30–60、60–100、>100 cm`，未据此重采样。

单卡 BF16 纯模型 probe 从 batch 8 到 768 均可运行，1024 OOM；batch 512/768 reserved memory 为 `12.53/18.73 GB`，吞吐已接近饱和。真实四卡短优化比较 global batch 128/256 与 LR `1e-4/2e-4` 后，global batch 256、LR `2e-4` 的最佳 validation dynamic `r+d≈9.43 cm`，优于 global batch 128 的约 `10.88 cm` 和 LR `1e-4` 的约 `9.92 cm`。

**决策**
正式配置选 per-GPU batch 64、global batch 256、LR `2e-4`、BF16、30 epochs、uniform v-prediction、25-step DDIM。短优化尚未超过 validation persistence `r+d≈8.27 cm`，这不作为 500-step probe 的停止条件；正式训练将按 best validation checkpoint 判断 V17 Gate。

## 实验：V17 Full-Data 正式训练与 sequence-disjoint 测试

**假设**
controlled32 上成立的 residual 与 Goal dependency 能迁移到完整自然帧分布，并在未见 sequence 的 dynamic subset 上超过 persistence。

**观察到的失败 / 现象**
四卡 BF16 正式训练 30 epochs、930 optimizer steps，训练 MSE 持续下降，但 validation 指标在中后期恶化。按 diffusion validation dynamic `r+d` 选出的最佳 checkpoint 位于 epoch 21，而不是 epoch 30。

**改动**
使用 V17 固定 16/2/2 demonstration split 同时训练 deterministic residual 与 uniform v-diffusion；正式 W&B 使用 online 模式。新增独立 `eval_v17.py`，在 test split 汇报 overall/dynamic/static/active/inactive、30/60/100 cm 远场，并用同一 initial noise 比较 correct、motion zero/shuffle/reverse 和 active zero。

**结果**
正式训练每 epoch 约 `11.8 s`，总训练主体约 `5.9 min`；W&B run 为 `hzdly89j`。最佳 epoch 21 的 validation dynamic persistence `u/r/d=1.424/3.706/4.565 cm`、F1 `0.508`；deterministic 为 `1.854/4.675/5.103 cm`、F1 `0.351`；diffusion 为 `1.564/4.008/4.676 cm`、F1 `0.135`。扩散 `r+d=8.684 cm`，差于 persistence 的 `8.271 cm`。

独立 test 共 1159 帧，其中 dynamic 占约 `89.6%`。dynamic persistence `u/r/d=1.529/3.905/4.570 cm`、F1 `0.347`；deterministic correct 为 `1.808/4.636/5.935 cm`、F1 `0.188`；diffusion correct 为 `1.677/4.303/5.158 cm`、F1 `0.043`。30 cm 远场上 persistence/diffusion correct 的 `r/d` 分别为 `6.441/7.825` 与 `6.415/7.941 cm`，没有稳定优势。

Goal 干预也未通过。test dynamic-active 上 diffusion correct 的 `r/d=4.311/5.158 cm`；motion zero 为 `4.237/5.258`，shuffle 为 `4.345/5.178`，reverse 为 `4.836/5.600`，active zero 反而改善到 `3.998/4.703 cm`。模型对 reverse 有响应，但正确 Goal 并非性能必要条件，且 active 标志形成了有害依赖。deterministic 同样在 active zero 后由 `4.681/5.987` 改善到 `4.205/5.070 cm`。

**诊断**
V16 controlled set 的局部低误差没有转化为跨 demonstration 泛化。训练目标可被拟合，但模型在 validation/test 上既未超过 persistence，也没有学到可靠的自然分布 Goal dependency；active zero 改善排除了“只是生成随机性掩盖干预效果”的解释，因为 deterministic 分支也出现相同方向。当前主要问题是跨 sequence 分布与 Goal 条件泛化，不是继续增加 sampler steps。

**决策**
V17 Gate 未通过，不进入 inverse / C inference，不把当前 full-data checkpoint 作为可用 Interaction Knowledge。保留 sequence-disjoint full-frame 数据链路、DDP/W&B 训练和分层评估工具，供下一轮针对泛化失败做最小诊断。

## 实验：V17.1 time-to-effect 条件诊断

**假设**
V17 的 Goal 只描述下一段 meaningful object motion，没有描述它何时发生；追加 `τ=k*(t)-t` 的 `log(1+τ)` 标量可能改善跨 demonstration 的 deterministic residual 泛化，尤其是长时间 approach 帧。

**观察到的失败 / 现象**
V17 full-data 中 active ratio 接近常量，active zero 反而改善结果；同时 residual median/mean/p90 为 `0.287/1.396/5.729 cm`，存在重尾。为避免把问题继续归因于 diffusion，本轮只训练 deterministic 分支。

**诊断**
按 V17 完全相同的样本顺序重新计算 time-to-effect。train/val/test 的有效比例为 `87.2%/97.3%/82.2%`，有效值中位数均为 0，均值为 `4.88/4.26/2.42` 帧，p90 为 `20/18/3` 帧；三者都没有 `τ≥100` 样本。test 的 1159 帧中，`missing/0–10/10–30/30–100` 数量为 `206/877/40/36`。因此当前 meaningful-motion 阈值下，“相同 Goal 距离 effect 数百帧”的预设并不存在，时间分布本身还有明显 split variance。

**改动**
保持 V17 cache、固定 16/2/2 split、full-frame 分布、MSE、模型宽度和学习率不变。两个模型使用相同的 26 维 Goal 网络：baseline 的最后一维恒为 0，实验组输入由 train-only mean/std 标准化的 `log(1+τ)`，missing 时为 0。两组分别使用两卡 DDP、per-GPU batch 64、global batch 128、BF16、30 epochs，并行训练；没有启用或上传 W&B。checkpoint 均只按 validation dynamic `r+d` 选择。

**结果**
Baseline 最佳 epoch 4：validation dynamic `r/d=3.758/4.256 cm`，test 为 `3.835/4.466 cm`；均略优于 persistence 的 validation `3.706/4.565` 和 test `3.905/4.570 cm` 的 `r+d` 总分。该结果也说明 V17 原联合 diffusion 训练会影响 deterministic 对照，后续应把两类训练分开解释。

Time-to-effect 最佳 epoch 22：validation dynamic `r/d=3.818/4.110 cm`，`r+d=7.928 cm`，相比 baseline `8.013 cm` 改善 `1.1%`；test 为 `4.097/4.699 cm`，`r+d=8.795 cm`，相比 baseline `8.301 cm` 退化 `6.0%`，并差于 persistence `8.476 cm`。

时间分桶揭示了方向相反的效果。Test `10–30` 桶从 baseline `r/d=6.038/8.260` 改善到 `3.017/4.690 cm`，`30–100` 桶从 `5.372/6.709` 改善到 `4.617/4.224 cm`；但占动态有效样本大多数的 `0–10` 桶从 `2.829/2.632` 恶化到 `3.461/4.195 cm`。Validation 也在两个较长时间桶改善，但整体只得到小幅收益。

**决策**
`τ` 对稀少的较长等待帧确实携带信息，但没有形成跨 split 的整体收益；“缺少 time-to-effect 是 V17 整体失败主因”的假设不成立，当前实现不作为默认 Goal 保留。保留时间元数据、分桶评估和 deterministic-only 训练入口作为诊断工具。下一步若继续，应先调查为何 `τ=0` 占主导以及 active/meaningful-motion 的语义，而不是继续增加时间编码容量。

## 实验：V18 Stable-Grasp Interaction Diffusion

**假设**
V17 跨 demonstration 失败来自把 Goal、active 与 object effect 混入条件。若只保留当前 `Y=[r,d]`、物体局部几何和 stable-grasp 成功 demonstration 的数据分布，无 Goal residual v-diffusion 应能生成进入或维持稳定抓取的 H=8 Y trajectory。

**观察到的失败 / 现象**
V17 使用的 manifest 只覆盖约 20 个 demonstrations，不能代表 full GRAB；V18 改用 stage4 已缓存的全部左右手。原 stage4 shared schema 没有 object pose，因此利用 4096 个有稳定 correspondence 的刚体 object points 做 SVD 配准，恢复每帧动态 object frame。聚焦测试中跨帧刚体对齐误差小于 `1e-5 m`。

**诊断**
Gate 0 使用 `contact anchors≥4`、`u RMS<0.3 cm/frame` 连续 6 帧，并要求随后 30 帧内物体平移超过 1 cm 或旋转超过 5°。扫描 1335 个 demonstrations、2670 个手文件后得到 1857 个手事件，覆盖 1320 个 demonstrations；left/right 事件为 `619/1238`。`t_g` min/median/mean/p90/max 为 `25/60/85.3/158.4/714`；onset contact anchor 数为 `4/15/20.1/42/113`，stable `u` 为 `0.019/0.235/0.221/0.288/0.300 cm/frame`。随机检查 20 个 `t_g` 的接触与速度时序，大多数位于接触跃升且相对速度降到阈值的位置，未见普遍的短暂停触误标；少数裁剪序列开始时已处于接触。

**改动**
每个事件使用 `t_g-8...t_g-1` 八个进入/稳定窗口及 `t_g/t_g+2/t_g+4` 三个 maintenance 窗口，严格按 demonstration 80/10/10 split，得到 train/val/test `1487/185/185` events 和 `16357/2035/2035` samples。新增无 Goal 的 H=8 v-prediction AdaLN diffusion、10-step skipping、四卡 BF16 DDP、stable/persistence/K=8/diversity/初始未接触子集评估。本轮没有使用 active、time-to-effect、Action/PoseToken、effect、CFG、MANO inverse 或 W&B。

**结果**
Gate 1 controlled32 训练 3000 steps 后，single-sample RMSE 从 step 500 的 `0.580` 降到 `0.143 cm`；GT/persistence/diffusion stable success 为 `87.5%/43.8%/87.5%`，terminal `u RMS=0.200 cm/frame`，采样全程 finite，Gate 1 通过。

Gate 2 四卡训练 30 epochs、1920 steps，每 epoch 约 `8.8 s`。按 validation RMSE 选择的 epoch 29 在 test 上 single/best-of-8 RMSE 为 `0.970/0.874 cm`，相对 best-of-1 改善 `9.9%`；pairwise trajectory RMS 为 `0.562 cm`，说明 K=8 并非完全相同。GT/persistence/diffusion 单次 stable success 为 `77.3%/61.9%/65.5%`，diffusion any-of-8 为 `88.4%`。在 776 个初始 contact anchors 少于 4 的真正 formation 样本上，GT/persistence/diffusion 单次 success 为 `62.5%/0%/33.9%`，diffusion any-of-8 达 `70.1%`。因此模型确实能从未接触状态形成抓取，而不只是复制 persistence。

单独按 validation stable success 会选 epoch 1：test 单次/any-of-8 success 达 `92.7%/94.7%`，formation 为 `80.8%/86.2%`，但平均 terminal contact 达 `42.4` anchors、RMSE `1.949 cm`、pairwise RMS `1.935 cm`，明显通过过度制造接触投机 operational metric。相比之下 epoch 29 平均 terminal contact 为 `11.1`，更接近数据但成功率较低。

**决策**
Gate 2 的核心能力通过：best-RMSE 模型在初始未接触子集明显超过 persistence，best-of-8 提升且存在可测多样性，证明 stable-grasp 对齐后的 Y-transition 可以跨 sequence 学习。但 stable success 单指标不适合作为 checkpoint 唯一标准，必须与 trajectory RMSE/contact calibration 联合使用；当前不把 epoch 1 当作可用模型。按 V18 范围停止在 Y-space，不进入 MANO inverse 或物体轨迹 tracking。下一步应先定义不会奖励过度接触的联合选择指标，或把 formation/maintenance 分开报告。

## 实验：V18.1 长训、阶段评估与 Y 可实现性

**假设**
V18 仍处于欠训练；延长到约 10k steps 并联合轨迹 RMSE、稳定成功率和终端接触分布选择 checkpoint，可以提高抓取形成能力且避免过接触。单/双手事件分组和 MANO inverse 可进一步判断数据混合与 Y-space 生成是否可靠。

**改动**
在不改变 V18 模型和表示的前提下，从 epoch 30/step 1920 恢复优化器并训练到 epoch 160/step 10240，每 5 epochs 保存 checkpoint。评估新增 formation、transition、maintenance 三段，以及 GT/预测终端接触 mean/median/p10/p90；加入 validation Pareto 筛选、另一只手在 `t_g±2` 的接触审计和按单/双手分组测试。最后固定当前真实手帧，分别对 GT Y 和生成 Y 优化未来 8 帧 MANO 参数。

**结果**
长训未持续收敛：epoch 40 validation RMSE/stable/contact-mean-error 为 `0.972 cm/79.8%/0.190`，epoch 45 为 `0.947/69.7%/5.725`，epoch 50 为 `1.362/90.3%/1.883`；epoch 160 已退化到 test 前的 validation RMSE `3.404 cm` 且 stable 为 0。测试集 epoch 40 的 single/best-of-8 RMSE 为 `0.972/0.855 cm`，单次/any-of-8 stable 为 `85.7%/95.1%`；终端接触预测 mean/median/p10/p90 为 `29.0/24/9/56`，GT 为 `27.1/23/8/53`，在三个候选中最均衡。formation/transition/maintenance single RMSE 为 `1.136/0.960/0.699 cm`。

1857 个事件按另一只手最大 contact anchors `<4` 分为单手干净 1342 个（72.3%）和双手 515 个（27.7%）。epoch 40 test 单手/双手的 single RMSE 为 `0.898/1.132 cm`，stable 为 `86.6%/82.2%`；双手 GT/预测终端接触均值为 `18.0/22.8`，比单手的 `30.8/31.5` 校准更差，因此暂不删除双手数据，但后续必须保持分组报告。

3 个 test 样本、300-step MANO inverse 中，GT Y 从重复当前手初始化优化到平均 `0.119 cm` RMSE，生成 Y 仅到 `0.771 cm`；逐样本生成结果为 `0.702/0.658/0.953 cm`。同一优化器能近乎恢复 GT，说明生成 Y 的残余误差不能只归因于 inverse 优化失败。

**决策**
保留长训 checkpoint、阶段化/接触分布/单双手分组评估和 MANO 可实现性脚本；当前推荐 epoch 40，而非 epoch 45、50 或最后一轮。延长训练本身没有稳定收益，且生成 Y 与 MANO 流形仍有约 `0.65 cm` 的额外差距。下一步优先扩大 inverse 样本数并研究可实现性约束，不再盲目增加 epochs。

## 实验：V18.2 全量 Y 一致性与可实现性诊断

**假设**
V18.1 生成 Y 难以 MANO inverse，可能来自当前手重复初始化的 optimization basin，也可能来自独立生成的 `r/d/u` 内部冲突或 spatial Y 本身离开真实手流形。固定 epoch 40，通过全量解析约束、三种 inverse loss 和三种初始化区分原因。

**改动**
保持 V18 模型、数据、Y 定义和 checkpoint 不变，新建独立 V18.2 脚本。对 val/test 各 2035 windows 检查 `max(0,||r||-d)` 与 `||u-Δr||`；MANO inverse 支持 `r/rd/rdu`、current-repeat/GT-future/四路 multistart、独立 GPU shard、逐样本 component RMSE 和 contact precision/recall/F1。validation 的 GT-current p95 用作对应 loss mode 的可实现阈值。

**结果**
解析审计中 GT val/test 的 `||r||≤d` 违例率均严格为 0；epoch 40 生成 Y 的 val/test 违例率为 `31.9%/29.9%`，平均违例 `0.106/0.101 cm`。Test formation/transition/maintenance 分别为 `35.4%/26.4%/26.8%`，formation 平均违例 `0.140 cm`。`u-Δr` 的 test mean：GT/生成总体为 `0.344/0.523 cm`，formation 为 `0.420/0.607`，maintenance 为 `0.219/0.465 cm`；生成结果时序偏差更大，但该量只作相对统计。

按原 300-step 配置估算完整矩阵约需 40 GPU 小时。本轮为保持完整 val/test 覆盖，先以 50 steps、八卡独立 shard 跑完 val 2035 个 GT-current 和 test 2035 个完整 12-cell 矩阵。此时 GT-val p95 阈值仍宽达 r/rd/rdu `1.684/1.644/1.332 cm`，使 test direct rate 为 `94.7%/94.5%/94.4%`；该结果主要反映 50-step GT inverse 自身未收敛，不能解释为 generated Y 可实现。全量 test current-repeat 的 r-only r RMSE `0.847 cm`；rd 的 r/d RMSE `0.868/0.715 cm`；rdu 的 r/d/u RMSE `0.860/0.701/0.282 cm`、contact F1 `0.676`。GT-future 在 rd 下改善到 `0.596/0.447 cm`，但仍远高于充分优化的 GT 阈值；multistart 与 current-repeat 接近。

为校准步数效应，另用 300 steps 跑 16 个 val GT 与 16 个 test 完整矩阵。GT-val p95 收紧到 r/rd/rdu `0.133/0.139/0.084 cm`；16 个 generated test 在三种 loss 下 direct/basin-rescued 均为 0，persistent gap 均为 100%。对应 current-repeat r-only r RMSE `0.733 cm`；rd 的 r/d 为 `0.753/0.497 cm`；rdu 的 r/d/u 为 `0.744/0.496/0.161 cm`。GT-future 和 multistart 没有把任何样本救到真实阈值内。

**决策**
保留全量解析审计、分片 inverse 和统一汇总工具。解析恒等式已直接证明生成 `r,d` 冲突；充分优化子集又显示合理初始化不能消除 gap，因此当前主因更接近 generated spatial Y off-manifold，而不是单纯 inverse basin。50-step 全量分类只保留为预算/收敛诊断，不作为 realizability 结论。下一步优先尝试只生成 `r` 或从 `r` 解析/辅助派生 `d`；在解决 `r/d` 一致性前，不增加 inverse 初始化复杂度，也不先针对 `u` 加新 loss。

## 实验：V18.3 r/d 几何耦合与解析投影

**假设**
V18.2 的 `||r||>d` 违例若是生成误差的主要来源，把 `(r,d)` 投影回二阶锥 `||r||≤d` 后应明显改善对 GT 的误差和接触判定；若改善很小，则违例更可能只是整体 spatial Y 偏离流形的一个可观测症状。

**观察到的失败 / 现象**
V18.2 全量 test 约 29.9% anchor-step 违反几何恒等式，但尚未量化真实 margin、违例严重度以及最小解析修复能否恢复指标。

**诊断**
从 val/test 完整 split 各均匀抽取 512 个窗口，保持 epoch 40、seed 42 和 10-step sampler 不变。GT 的 `d-||r||` 均值为 `0.267/0.310 cm`，分别有 `47.4%/43.2%` 位于绝对 margin 小于 `0.1 cm` 的边界附近；`||r||` 与 `d` 的相关系数为 `0.996/0.993`。生成结果相关系数仍有 `0.982/0.976`，但违例率为 `31.6%/30.4%`，违例深度中位数 `0.233/0.236 cm`、p95 `0.978/0.984 cm`。Test formation 违例率最高，为 `35.5%`。

**改动**
新增中等规模诊断脚本，统一报告 margin、接触/远场条件违例、相关性，并比较三种不重训修复：只增大 `d`、只缩短 `r`、联合欧氏二阶锥投影。实现核对确认当前 correspondence 使用 `softmax(-dist²/τ²)`，与 V18.3 文档给出的标准高斯 `softmax(-dist²/(2τ²))` 相差因子 2；`τ=1.5 cm` 对应标准写法的有效尺度约 `1.06 cm`。该差异不影响 `||r||≤d` 恒等式，本轮不改冻结 Teacher 定义。

**结果**
Test 原始 `r/d` 联合 RMSE 为 `1.262 cm`；raise-d、shrink-r、联合锥投影分别为 `1.254/1.250/1.248 cm`，最佳仅改善约 `1.1%`。Contact F1 为 `0.6456/0.6419/0.6456/0.6459`，没有实际提升；formation F1 从 `0.4567` 下降到 raise-d 的 `0.4353` 或联合投影的 `0.4477`。Val 方向一致，联合 RMSE 从 `1.247` 降到 `1.234 cm`。原本违例位置的 anchor error 只比合法位置略高，test 为 `1.040` 对 `0.976 cm`；联合投影后违例位置降到 `0.998 cm`，仍未恢复到合法位置水平。

**决策**
保留解析锥投影作为诊断和可选安全约束，但不把它当作生成质量修复。`r/d` 违例是真实且稳定的 off-manifold 信号，不过单纯满足必要不等式不足以恢复 GT 或接触语义；下一实验应比较训练期 cone penalty 与保证约束的参数化，并继续以 trajectory/contact/MANO inverse 判断，而不能只看违例率归零。

## 实验：V18.4 MANO-H deterministic transition baseline

**假设**
把 future generation 从自由 Y 改成 object-frame MANO wrist/PCA residual，可由构造保证整条轨迹属于同一只合法 MANO 手；若 deterministic H regression 仍能保持 formation，则可以较小 interaction 代价消除 V18 的 off-manifold 问题。

**观察到的失败 / 现象**
V18 free-Y test RMSE 约 `0.972 cm` 且 stable success `85.7%`，但约 30% anchor-step 违反 `||r||≤d`，300-step MANO inverse 子集也全部存在 persistent gap。V18.3 的解析锥投影只改善约 1% RMSE，不能恢复共享手流形。

**诊断**
新 cache 沿用 V18 event、split、窗口和动态 object frame，并额外保存 `current_h[33]`、`future_delta_h[8,30]`、beta、side、raw frame ids 与 object transforms。GT H→MANO surface→Y 在 train/val/test 的最大绝对 parity error 分别为 `3.73e-4/3.51e-4/3.09e-4 cm`，均值约 `6–7e-5 cm`，Gate 0 通过。前 32 个 train 窗口训练 1500 steps 后，H translation/rotation/PCA RMSE 为 `0.0079 cm/3.85e-4 rad/0.0020`；回到 Y 后 `r/u/d=0.0099/0.0119/0.0075 cm`、contact F1 `0.996`、stable success `87.5%`，Gate 1 通过。

**改动**
新增独立 MANO-H cache builder、Dataset、复用 V18 anchor/state/object encoder 与 AdaLN block 的 deterministic transition model、仅 normalized H residual MSE 的训练入口，以及无需 inverse optimization 的 MANO-forward Y evaluator。没有修改旧 V18 cache，没有加入 diffusion、ActionToken、Y/contact/smoothness auxiliary 或训练时 MANO forward。按用户要求不使用全量训练，只取前 2048 个 train 窗口训练 1500 steps；val/test 各评估 512 个窗口。

**结果**
中等规模模型从 step 250 起 validation H error 未随 train loss 持续下降：最佳 translation/rotation/PCA RMSE 为 `2.743 cm/0.191 rad/0.493`，到 step 1500 仍为 `2.774/0.205/0.515`，存在明显跨 sequence 泛化差距。最佳 checkpoint 的 val `r/u/d=1.465/0.707/1.343 cm`、contact F1 `0.579`、stable success `0.59%`；test 为 `1.460/0.666/1.288 cm`、F1 `0.585`、stable success `2.34%`。Test formation 的 `r/u/d=1.526/0.618/1.295 cm`、F1 `0.479`、stable success `1.63%`。预测 terminal contact 均值仍为 `25.9`，但 stable success 极低，说明不是简单地完全无接触，而是 deterministic H trajectory 的接触与低相对速度没有同时成立。所有预测 Y 均由 MANO forward 构造，`r/d` violation 为 0 且天然共享同一手流形。

**决策**
保留 MANO-H 数据链、模型和 evaluator，但当前 deterministic checkpoint 不替换 V18。实验回答了核心权衡：完整 realizability 可以由构造获得，但 2048-window deterministic regression 相比 free-Y 明显损失 interaction accuracy 和 formation，符合 conditional mean / 多模态平均化，也伴随 sequence 泛化不足。下一步若继续，应固定同一 H 表示和数据链，最小比较 H-space stochastic model；不先加 Y auxiliary 掩盖问题，也不回到单纯 cone penalty。

## 实验：V18.5 MANO-constrained Y supervision

**假设**
V18.4 formation 失败不一定来自 MANO-H 表示，而可能来自逐维模仿唯一 GT human H 的错误目标。保留 H 作为合法手的内部结构变量，改为只优化 `H→MANO→Y` 的 interaction residual，应恢复 stable grasp，同时保持 MANO realizability。

**观察到的失败 / 现象**
V18.4 controlled32 可以把 Y 拟合到约 `0.01 cm`，但 2048-window test 的 stable/formation stable 只有 `2.34%/1.63%`，尽管 terminal contact 均值并非 0。这表明参数化和容量成立，而 H imitation 在跨 sequence 下产生了不满足 interaction 目标的平均轨迹。

**诊断**
新增与 frozen Teacher 完全同公式的 batched/chunked Y，按 16 个 anchors 分块。4 个 GT H 样本经 structured decoder 的最大/平均 Y parity error 为 `3.97e-4/2.45e-5 cm`；随机输出执行 Y loss backward 后模型梯度 finite 且 nonzero，Gate 0 通过。MANO layer 以 `subject:side` 为稳定 key 缓存，batch 内分组 forward/scatter，继续使用 sequence-specific GRAB template。

**改动**
保持 V18.4 的 current H、normalized `ΔH[8,30]`、模型、object frame、beta、数据和 1500-step 预算。删除 H-GT MSE；预测 H 经可微 MANO forward 与 batched interaction field 得到 future Y，再以 train-only V18 residual std 优化 Y residual MSE。没有加入 ActionToken、diffusion、contact/stable auxiliary、cone loss 或新 condition。实现了可选的 3σ barrier，但本轮权重保持 0；checkpoint 使用 validation `r+u+d` 与 terminal-contact mean error 的联合分数。

**结果**
Controlled32 最佳 checkpoint 的 `r/u/d=0.0580/0.0378/0.0437 cm`、contact F1 `0.968`、stable `90.6%`、formation stable `83.3%`，Gate 1 通过；normalized H 超过 3σ 仅 `0.065%`。

2048 train、1500 steps 的最佳 checkpoint 为 step 1250。均匀覆盖 512 test 窗口上，`r/u/d=1.595/0.508/1.479 cm`、F1 `0.606`、stable `51.0%`、formation stable `37.4%`。为与 V18.4 已记录的“前 512”口径严格对齐，另在相同前 512 test 上评估：V18.5 `r/u/d=1.324/0.411/1.202 cm`、F1 `0.622`、stable `57.2%`、formation stable `49.4%`；V18.4 分别为 `1.460/0.666/1.288 cm`、`0.585`、`2.34%`、`1.63%`。V18.5 terminal contact 均值为 `25.16`，且 `r/d` violation 严格为 0。Test normalized H mean/max abs 为 `0.261/5.10`，超过 3σ 的元素只占 `0.024%`，无需启用 prior。

**决策**
保留 V18.5 structured decoder 与纯 Y supervision。单变量替换监督后 formation 从约 2% 恢复到 49%，支持“H 应是内部合法性变量，而不是必须复制 GT human pose”的核心假设；V18.4 的失败不能简单归因于 MANO-H 过强。当前中等规模 V18.5 仍弱于 full-data free-Y epoch40 的 `85.7%` stable，且均匀 test 的 spatial RMSE 约 `1.5 cm`；因此尚不替换 V18。下一步可先扩大同一 deterministic baseline 判断数据效应，再决定是否需要 H-space stochastic model；不应重新加入强 H-GT loss。

## 实验：V18.6 逐 channel normalization 与全量 MANO-Y 训练

**假设**
V18.5 的 residual std 错把 anchor 与 56 个 channel 一起聚合成 scalar，实际接近 raw Y MSE。恢复 `[56]` 逐 channel normalization 应改善 interaction 与 terminal stable proxy；若中等规模结论保持，再公平扩展到完整 V18 数据。

**观察到的失败 / 现象**
代码审计确认 `torch.cat(...,0)` 后 tensor 为 `[N×128,56]`，旧 `std((0,1))` 只得到 scalar，而原设计要求 `std(0)`。该问题不否定 V18.5 的 Y-supervision 收益，但改变各 channel 权重，必须单变量复现。

**诊断与改动**
只把 residual std 改为 `[1,1,56]`，保持 V18.5 的 seed、前 2048 train、512 val/test、1500 steps、模型和 checkpoint criterion。新增 shape 测试。随后构建完整独立 MANO-H cache，覆盖与 V18 完全相同的 train/val/test `16357/2035/2035` windows；parity 仍在约 `1e-4 cm` 量级。新增四卡 DDP/BF16、在线 W&B、全量 validation 和 best/latest checkpoint 的正式训练入口。

**结果**
逐 channel 版在相同前 512 test 上达到 `r/u/d=1.289/0.373/1.117 cm`、contact F1 `0.630`、terminal 3-frame stable proxy `63.3%`、formation stable `51.8%`；scalar V18.5 对应为 `1.324/0.411/1.202 cm`、`0.622`、`57.2%/49.4%`。Normalized H mean/max abs 为 `0.210/2.525`，无元素超过 3σ，故 prior 继续为 0。

四卡真实 10-step 吞吐压测中，per-GPU batch 8/16/24/32 分别约为 `124/220/265/299 samples/s`，均未 OOM；正式选 batch 32/global 128。完整 epoch 128 optimizer steps、训练主体约 39 秒。按用户更新后的约一小时预算，启动 60 epochs、每 5 epochs 全量 validation 的 W&B online run `yi9wmzf7`。Epoch 5 train loss 已从 epoch 1 的 `1.492` 降到 `0.979`；validation `u/r/d=0.439/1.452/1.345 cm`，terminal contact mean `21.8`，GT `24.1`，训练数值正常。

**决策**
保留逐 channel normalization，并以它作为 V18.6 full-data 唯一正式配置。正式训练完成 60 epochs/7680 steps，训练主体约 39 分钟；train loss `1.492→0.360`，但 validation 在 epoch 10 最佳并在后续退化，因此采用 epoch 10 而不是最后一轮。完整 2035-window test 的 `u/r/d=0.440/1.392/1.224 cm`、terminal 3-frame stable proxy `72.8%`、formation stable `70.7%`、contact F1 `0.634`、`r/d` violation 0；normalized H 超过 3σ 的比例为 `0.029%`。W&B run `yi9wmzf7` 已完成同步。全量数据显著改善 formation，但仍弱于 free-Y epoch40 的 stable proxy；下一步需把 accuracy/realizability 权衡与下游行为一起判断。

## 实验：V19 通用轨迹抓取 Viewer v2

**假设**
可视化应面向“reference object trajectory 与 hand grasp/manipulation trajectory”这一通用任务，而不是暴露 `r/u/d` 等内部表示；viewer 与手参数化解耦后，未来可在不修改 UI 的情况下接入 Inspire/Allegro 等 backend。

**改动**
新增 `viewer_v2/backends.py` 的 `HandFrame/ObjectTrajectory/VisualizationSample/HandBackend` 接口和首个 `ManoBackend`。`prediction_v18_5.py` 独立负责加载 V18.5/V18.6 checkpoint、MANO-H cache、structured decoder，恢复 predicted/GT MANO meshes 和 GRAB canonical object mesh/9-frame world poses。`viewer.py` 只消费通用 mesh frames，使用 Viser 提供 sample/frame slider、Play/Stop、prediction/GT/reference trajectory 显隐和通用指标。当前模型未以任意 object trajectory 为 condition，界面明确标记 `Reference Object Trajectory`。

**结果**
使用 V18.6 epoch 10 best checkpoint 与 full test cache 的 sample 0 完成 provider smoke：得到 2035-sample provider、9 帧 predicted/GT hand、`19518 vertices/39044 faces` object mesh、`[9,4,4]` object poses。Viser 1.0.30 服务在 `127.0.0.1:8089` 成功启动并创建场景/GUI。当前仓库没有可信的 collision/penetration evaluator，第一版明确显示 N/A；grasp 指标沿用并命名为 terminal grasp proxy，避免冒充物理成功。

**决策**
保留通用 viewer shell、MANO provider 和最小交互。第一版不加入 ghost、视频导出、内部 loss/latent 热图，也不把 reference trajectory 称为模型输入；等真正引入 object-trajectory condition 或可靠物理 evaluator 后，再扩充对应字段。

## 实验：V19.1 MANO mesh 拓扑修复

**观察到的失败 / 现象**
V19 把 1538 个 MANO 三角面中心当作 mesh 顶点，却继续使用索引原始 778 个顶点的 MANO faces，导致预测与 GT 都出现错误连线和畸形拓扑。

**改动**
保持训练和 Y 构造中的 `face_centers` 不变；新增 viewer-only MANO 顶点恢复函数，沿用 structured decoder 的同一 `ΔH→world rotation/translation/pose` 坐标链，直接返回 `output.vertices`。GT 同样直接使用 MANO world vertices。GT 渲染改为绿色半透明实心网格。

**结果**
预测与 GT 均恢复为每帧 778 vertices、1538 faces，所有 face index 均落在原始顶点范围内；V18.6 test sample 0 的 provider 与 Viser 服务冒烟测试通过。

**决策**
保留修复。viewer 与训练/Y 链路职责明确分离，后续不能再将 interaction surface points 配合 MANO faces 当作可视化网格。

## 实验：V19.2 穿透诊断与 stable latch

**假设**
保持 V18.6 的 `Y={r,d,u}` 与 checkpoint 不变，解析 penetration 可区分正常接触和明显穿模；contact/u 连续事件可把 8 帧抓取形成自然接到完整 object trajectory。

**改动**
新增 watertight 校验、奇偶射线判内外、精确最近三角面距离与 anchor soft aggregation；非封闭/退化 mesh 明确返回无效。新增连续三帧 `contact anchors>=4 && u_rms<0.3 cm/frame` 的 latch 状态机，penetration 只显示、不进入 gate。provider 通过 hand cache 对应 `shared.npz/raw_frame_id` 恢复完整后续物体轨迹；有 latch 时冻结该帧 object-relative MANO mesh 并刚性跟随，无 latch 时停在第 8 帧。为保持交互加载速度，Pred/GT penetration 当前同口径统计 prediction horizon 末帧。

**结果**
解析四面体 sanity check 验证内部点为正穿透、外部/表面点无穿透，非 watertight mesh 返回 `valid=False`。V18.6 test sample 0–2 均未 latch，viewer 保持 9 帧；sample 3 在 frame 7 latch，扩展至 68 帧，末帧相对 latch 帧的 object-frame MANO 顶点最大漂移 `6.15e-8 m`。前四个 prediction 末帧最大穿透为 `8.53–11.49 mm`，对应 GT 为 `0–2.18 mm`，初步支持 p 对明显穿模敏感，但不足以确定 gate 阈值。

**决策**
保留解析诊断、latch 与 object-follow 系统链路；不修改 Y、不重训、不把 p 加入 stable gate。下一步若要加入 penetration loss 或 gate，应先扩大 GT/pred 分布统计并按 GT percentile 定阈值。

## 实验：V20 causal Field Dynamics controlled gate

**假设**
移除 `tokens.mean(1)`、`current_h` 与 MANO-H 输出，始终执行 `[B,N,*]→[B,N,*]`，可使 latent field 真正使用 object-anchor 空间身份。

**改动**
独立新增 causal `Y=[r,d,v,p]`、cache、field Transformer、normalized residual trainer 和 intervention evaluator，不修改 V18/V19。`v_t` 只使用 `h_t-h_{t-1}`；p 由相同 hand surface points 的 mesh penetration 聚合，invalid mesh 通过 mask 排除。网络以 current Y、anchor xyz、local object patch 为逐 anchor 输入，shared pointwise head 输出 `[B,N,8,8]`，无 pooling/global token/current H。新增 causal observed stable API，区别于 V19.2 predicted latch。

**结果**
单真实 stable event 共 13 个窗口，训练 1200 steps。normal `r/d/v/p MAE=0.167/0.137/0.052/0.023 cm`，persistence 为 `1.202/1.767/0.244/0.075 cm`；输出空间 variance `0.220`，未 collapse。mean-token 的 `r/d/v/p=0.564/0.483/0.086/0.083 cm`，token shuffle 为 `0.686/0.540/0.081/0.113 cm`，均显著退化。同步置换完整 field 后逆置换输出最大误差 `1.43e-6`。上述数字来自修正后的单次 soft aggregation p；初版双重聚合结果未采用。

**决策**
controlled overfit、空间身份 intervention 和 permutation equivariance Gate 通过，保留 V20 field-native 主干。当前只有单 event，不能声称 unseen sequence 优于 persistence；因此按版本纪律暂不实现 V20.1 MANO projector，下一步应构建小型 sequence-disjoint cache 并完成泛化 Gate。

## 实验：V20.1 MANO feasibility projector

**假设**
冻结 V20 后，从 repeat-current-hand 的零 `ΔH` 初始化优化未来 MANO trajectory，可显著缩小 predicted field 到 MANO 可实现流形的 projection gap，并可能纠正 free-field prediction。

**诊断与改动**
先修复 V20 p normalization 纳入 invalid 0、persistence p 未使用 valid mask 两个指标问题。为既有 controlled shard 补齐 V18.4 同语义的 MANO metadata；新增可微 `ΔH→MANO surface→causal [r,d,v]` decoder、Adam projector、free/projection/projected 三组指标、causal stable 与 post-hoc penetration。GT `ΔH` decoder parity 最大/平均误差 `4.21e-4/2.70e-5 cm`，低于 `1e-3 cm` Gate，坐标与时间语义通过。

**结果**
sample 0 的 free→GT `r/d/v RMSE=0.346/0.184/0.070 cm`。默认 100 steps、lr `1e-2` 时 normalized field loss `9.40→1.13`，但 projection gap 仍为 `1.391/1.293/0.251 cm`，projected→GT 恶化到 `1.421/1.250/0.276 cm`；free/projected/GT 均未 stable。projected terminal penetration 为 0，GT 最大/均值为 `0.824/0.0008 mm`，但 interaction 尚未投准，因此不能把无穿透视为成功。延长 predicted target 到 500 steps 后 gap 降至 `0.920/0.446/0.118 cm`，曲线仍缓慢下降。以 GT field 为 oracle target、500 steps、lr `0.05` 且关闭 smooth/prior 后，仍只达到 `r/d/v RMSE=0.835/0.396/0.118 cm`，尽管 GT `ΔH` 已证明是精确解。

**决策**
保留 decoder、projector evaluator 与负结果，但 V20.1 optimization Gate 未通过，停止扩展到 4–8 windows。当前证据只能说明零初始化 Adam 存在严重 basin/conditioning 问题，不能据此判定 predicted Y 不可实现，也不支持训练 learned projector。下一步若继续，应先在 oracle GT target 上研究更好的初始化或分阶段/二阶优化，直到能够接近已知 parity 解。
