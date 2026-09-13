# CmDecoderv2 架构记录

## V1.1.1 — Temporal-D2 首版合同

- 输入窗口为连续 30 Hz cache 帧：`K=4` 个 Cm transition 需要 `K+1=5` 个 geometry frame；第
  `tau` 个 Cm 的 hand flow 恒为 `tau -> tau+1`。
- 每帧冻结 OICM 输出 `16 x 32` Cm、`16 x 3` anchor position 和 normal。每帧 16 个 slot 作为无序
  set；只加入 time embedding，不加入 slot-ID embedding，也不做跨帧 slot tracking。
- Temporal-D2 memory 为 `K x 16` interaction tokens；query 为当前 Inspire global state、18 个 URDF
  link/state query 和 `h=1..K` future embedding，经 Transformer cross-attention 后输出多 horizon reference。
- 6 个独立 finger q 按 Dexplore native index `[6,8,10,12,14,15]` 排列，即
  `[index,middle,pinky,ring,thumb_yaw,thumb_pitch]`；native mimic `[7,9,11,13,16,17]` 由固定耦合恢复。
- wrist GT 是 Dexplore native q 通过右手 Inspire URDF FK 得到的 `hand_base_link` world pose；decoder
  输出当前 wrist 到未来 wrist 的相对 `SE(3)`，平移单位 m，旋转 head 为 rotvec/rad。
- 训练/验证只使用 `inspire_rl`；MANO test 只提供 source geometry/Cm 做定性 receding-horizon 可视化，
  不存在 paired Inspire test GT，也不报告 q/wrist test 定量指标。
