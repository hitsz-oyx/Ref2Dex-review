# AI 自主决策记录

## 2026-08-18：V1.2 object cache 首版实现

- 未明确点：V1.2 要求保留 mmap、ragged candidate 和 B=4 sampling，但没有规定转换器是否重算几何。
- 实际选择：复用已有 GRAB/ARCTIC Stage4 的 object-only NPZ，新增 NPZ 到 mmap/ragged 的转换器。
- 其他选择：重写 Stage4 或直接继续使用 NPZ。
- 原因：避免重复 MANO/物体采样和近邻计算，同时满足 V1.2 cache 合同，最易验证和回退。
- 影响：转换阶段依赖既有 Stage4 输出；不会改变模型输入、GT 或 loss。
- 可逆性：高；原始 NPZ 保留，删除新 cache 即可回退。
- 后续确认：混合采样比例仍由 E1 数据统计决定，首版统计输出记录 `p_d ∝ sqrt(N_d)` 候选。
