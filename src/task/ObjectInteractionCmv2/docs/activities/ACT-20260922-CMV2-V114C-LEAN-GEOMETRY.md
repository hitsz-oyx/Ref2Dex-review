# ACT-20260922-CMV2-V114C-LEAN-GEOMETRY

- timestamp：2026-09-22T10:26:41+08:00
- activity_id：`ACT-20260922-CMV2-V114C-LEAN-GEOMETRY`
- work_version：`V1.14`
- git_commit：实施中；full build 前以最终提交替换
- base_commit：`6e4d96e`
- branch：`ai/ObjectInteractionCmv2/v1.13-io-acceleration`
- scope：本地 GRAB MANO/Inspire V1.14c lean geometry schema、reader、producer、配置与真实 parity pilot
- approval：用户明确批准仅用本地数据、lean cache 与随后一轮随机初始化训练
- rollback：停用 V1.14c schema/config；独立 v114c 数据产物可保留审计，旧 cache/split/checkpoint 未覆盖

## 完成内容与验证

- 固定使用 `data/processed_data/stage4/splits/full_grab_v1`，序列数为 train/val/test `1068/134/133`。
- 新 cache 只保存对象点/法向/位姿、帧时间/ID、对象点 ID、双手固定 4096 点/法向和 2 cm mask。
- MANO 从本地 raw GRAB 重建 mesh 后按冻结 correspondence 采样；Inspire 先生成每手 10135 点并计算 mask，
  再按 V1.14a 固定索引保存每手 2048 点。
- 定向 pytest：29 项通过（含 lean reader、rigid adapter、V1.14a/V1.14b 与 part-SE3 回归）。
- 真实 `grab/s9/cubelarge_pass_1` parity：MANO 点最大误差 `2.384e-7 m`、法向最大误差
  `4.706e-5`、2 cm mask 差异 0；Inspire 与完整 highres reference 的固定点/法向逐元素相同、mask 差异 0。
- Inspire 样例 geometry 字节数由 `114667334` 降至 `24713950`，比例 `0.215527`。
- 两组 lean sample 经 rigid part adapter 得到 `[4096,3]` 手点，随机 V1.14a 前向与 loss finite。

## 当前运行状态与科学结论

full cache 和训练尚未启动；pilot 只支持 schema/parity/接线与体积结论，不支持训练速度或科研效果结论。
正式生成必须使用干净 committed worktree，并在两组 `bad_count=0` 后才启动训练。
