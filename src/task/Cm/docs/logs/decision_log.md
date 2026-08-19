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
