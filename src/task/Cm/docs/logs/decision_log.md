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
