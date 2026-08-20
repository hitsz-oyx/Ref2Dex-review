# Cm AI 自主决策记录

## 2026-08-18 — V1.2 object cache 首版采用转换器

- branch: 当前工作分支
- post-commit: `HEAD`

**未指定点**

V1.2 要求保留 mmap、ragged candidate 和 B=4 sampling，但未规定是否重算几何。

**实际选择**

复用既有 GRAB/ARCTIC Stage4 object-only NPZ，新增 NPZ 到 mmap/ragged 的转换器。

**其他合理选择**

重写 Stage4，或继续直接使用 NPZ。

**选择理由**

避免重复 MANO、物体采样和近邻计算，满足既定 cache 合同且易验证、易回退。

**对结果的影响**

转换依赖既有 Stage4 输出，不改变模型输入、GT 或 loss。

**可逆性**

高；保留原始 NPZ，删除派生 cache 即可回退。

## 2026-08-19 — V1.2.1 采用共享 epoch、LRU cache 和确定性 split

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

V1.2.1 指导要求修复 worker epoch、缓存泄漏和训练划分，但没有限定具体同步原语、cache 上限和 split 生成策略。

**实际选择**

1. `_MmapSequenceDataset` 用 `mp.Value("q", 0, lock=True)` 共享 epoch；
2. `_cache` 改为 `OrderedDict` LRU，打开序列上限设为 16；
3. 固定 split 采用 sequence 级确定性打散后按比例切分，并写入 `splits.json`。

**其他合理选择**

1. 用线程锁/共享内存封装替代 `mp.Value`；
2. 用无界缓存配合手工清理；
3. 用随机在线切分或单一全局列表切分。

**选择理由**

`mp.Value`、`OrderedDict` 和显式 `splits.json` 都是最小改动、最易验证、最易回退的实现；它们能直接对应指导中的可复现性要求，也更方便后续排查数据泄漏和 worker 状态漂移。

**对结果的影响**

只影响训练稳定性、缓存占用和实验可复现性，不改变模型语义、GT 定义或评估口径。

**可逆性**

完全可逆。

**建议用户确认**

否


## 2026-08-19 — V1.2.1 采用 3 GPU DDP + batch 48 作为正式长训起点

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

用户要求“用多卡一起训练”，但没有指定是 2 GPU 还是 3 GPU，也没有指定是按吞吐还是按效率选卡数。

**实际选择**

正式长训使用 3 GPU DDP（`CUDA_VISIBLE_DEVICES=0,1,5`）并把 mixed/object-v2 full training 的 batch size 设为 48。

**其他合理选择**

2 GPU DDP，或者 batch size 24 / 32 / 64。

**选择理由**

train-only benchmark 显示 mixed 的纯训练吞吐在 3 GPU 下随 batch 增长到 48 时达到峰值（约 366.7 samples/s），64 已回落到约 357.1 samples/s；因此 48 是当前能拿到的最优吞吐点，同时保留了 DDP 的并行收益。

**对结果的影响**

只影响训练时间和每步全局 batch，不改变模型语义、GT 或评估口径；DDP 需保持 `find_unused_parameters=true`。

**可逆性**

完全可逆。

**建议用户确认**

否

## 2026-08-20 — V2 GRAB 数据脚本采用 raw asset 解析与严格 subject template

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

用户要求修正 V2 数据脚本，但没有指定当 `v_template` 缺失时是继续生成还是停止。

**实际选择**

按正式 Cm/V2 数据的语义要求，默认停止并报错；保留显式 `--allow-default-mano` 作为调试/旧流程兼容开关，并让 adapter 同时支持 `dataset/GRAB` 与 `dataset/GRAB/data` 两种 root。

**其他合理选择**

继续静默回退平均 MANO，或在脚本中固定一种 root 布局。

**选择理由**

平均 MANO 会改变手点、手根和当前 candidate mask，结果不能作为 V2 正式实验；严格失败能尽早暴露路径/资产问题。

**对结果的影响**

只改变数据生成的资产解析和错误处理，不改变模型、loss、GT 定义或评估口径；已有错误 cache 不会自动修复。

**可逆性**

完全可逆；可用兼容开关恢复旧回退行为。

**建议用户确认**

否

是（建议时机：启动正式长训前）

## 2026-08-19 — GRAB gate+cm64 候选保持 time condition 并按 50 epochs 比较

- branch: `oyx`
- post-commit: `HEAD`

**未指定点**

用户指定 GRAB-only、开启 slot gate、把 slot embedding 维数从 256 改为 64，但未重复指定 time condition、训练 epoch 和 batch。

**实际选择**

保持 `use_time_condition=true` 和其余数据/GT/loss/split/calibration 不变；正式预算沿用当前确认的 50 epochs。batch 48 仅作为吞吐 sweep 起点，正式 batch 要在空闲多卡上重新测定。

**其他合理选择**

关闭 time condition；沿用 10000-step pilot；直接复用 mixed 的 batch 48 而不重新 benchmark。

**选择理由**

保持 time condition 可把用户要求之外的模型变化降到最少；按 epoch 控制可让 GRAB-only 完整遍历次数明确；`cm_dim=64` 会改变显存和计算瓶颈，因此需要重新确定 batch 峰值。

**对结果的影响**

该候选同时改变数据范围、gate 和 slot capacity，是复合候选，不应被解释为单独的 gate 或 cm_dim 消融。50 epochs 保证预算口径清晰，但与 mixed 的样本总量不同。

**可逆性**

完全可逆；使用独立配置，不修改现有 mixed 与 GRAB-only baseline。

**建议用户确认**

否；当前选择遵循用户刚确认的 50-epoch 正式训练口径，启动前仍需确认可用 GPU 不与其他任务冲突。

## 2026-08-20 — gate+cm64 采用全开后渐进启用的 warm-up

- branch: `oyx`
- post-commit: `453806a`

**未指定点**

用户要求增加 gate warm-up，但未指定 warm-up 时长以及从全 slot 到 hard gate 的切换方式。

**实际选择**

前 5 个 epoch 强制 16 个 slot 全部参与，随后 5 个 epoch 将 threshold 从 0 线性升至 0.85、count loss 权重从 0 线性升至配置值；第 11 个 epoch 起恢复完整 gate 配置。

**其他合理选择**

只关闭 gate 若干 epoch 后直接切换；或仅降低 threshold 而不渐进 count loss。

**选择理由**

Hard-Concrete 初始 nonzero probability 约 0.83，而原 threshold 为 0.85，直接切换会再次触发单 slot fallback。全开阶段先让 slot/decoder 学到有区分度的表示，渐进阶段再引入稀疏压力，可避免切换瞬间重新坍缩。

**对结果的影响**

改变 gate 的优化日程，不改变输入、GT、数据划分、评估指标或最终 gate 超参数；该版本应与原 gate+cm64 作为不同训练策略比较。

**可逆性**

完全可逆；warm-up 通过独立配置开关控制，默认关闭。

**建议用户确认**

否（方案已由用户确认）。
