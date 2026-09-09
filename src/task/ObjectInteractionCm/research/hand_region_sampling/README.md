# Region 加权手点预览

本实验只生成 MANO/Inspire 的 2048 点 Region 加权采样 PLY，用于人工检查采样分布，不修改正式 ObjectInteractionCm cache。

入口：`run.py`

默认合同：

- 11 个 Region：掌部、五个 `body`、五个 `tip`；
- 总点数 2048；默认 `palm/body/tip = 512/1024/512`；
- Region 内按三角形面积加权、有放回三角形抽取，再作重心随机采样；
- `surface_seed=2024`；
- 输出 canonical MANO 和 Inspire 零姿态 PLY，按 Region 着色。

配额 profile：

- `--quota-profile balanced`：V1.2.6 基线，`palm/body/tip = 512/1024/512`；
- `--quota-profile tip_heavy`：保持总点数 2048，只将 body/tip 预算交换为 `512/512/1024`，用于检查指尖覆盖是否受比例限制。
- `--quota-profile link_stratified`：Inspire 对每个非空 visual link 独立分配 quota；MANO 使用对应的 joint-segment 分层。五个 Inspire 末端父 link 合计 1024 点，hand base 512 点，其余 finger link 合计 512 点。
- `--quota-profile uniform_surface`：取消所有 Region/link/segment 配额，直接在 MANO/Inspire 各自完整 mesh 上按单位表面积均匀采样 2048 点；PLY 使用统一灰色显示点密度。
- `--quota-profile uniform_surface_ratio`：MANO 保持 2048 点，Inspire 按完整 mesh 表面积比例使用 10135 点，以匹配两者的表面点密度；两份 PLY 仍使用统一灰色。

所有 profile 都保持相同的 seed 和 canonical 姿态；`tip_heavy`、`link_stratified`、`uniform_surface` 与 `uniform_surface_ratio` 只用于诊断对比，不覆盖基线 PLY。`link_stratified` 的 Inspire 点严格来自对应 link 自己的 mesh，不做跨 link 的面积竞争；`uniform_surface` 和 `uniform_surface_ratio` 则允许完整 mesh 上的全局面积竞争。

正式 cache/训练接入不属于本实验，需另行批准。
