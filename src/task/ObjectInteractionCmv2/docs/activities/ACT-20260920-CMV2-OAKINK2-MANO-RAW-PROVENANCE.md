# OakInk2 最大手点流的原始 MANO 来源复核

- timestamp: 2026-09-20T05:08:00Z
- activity_id: ACT-20260920-CMV2-OAKINK2-MANO-RAW-PROVENANCE
- work_version: V1.11.1
- mode: inspect
- change_level: L0
- approval: user-requested diagnostic
- branch: cmv2
- git_commit: b9acd32b0bc7aa7aca73e1a6b3f0a512cbf8dadb
- scope: 复核 OakInk2 MANO cache 最大点流是否由原始官方 MANO annotation 忠实导出。
- run_id: oakink2_mano_raw_provenance_20260920T050800Z
- run_status: COMPLETED
- engineering_conclusion: SUPPORTED
- physical_ground_truth_conclusion: INCONCLUSIVE（未核对原始 RGB/深度视频）

## Provenance

目标序列：
`oakink2/scene_02__A008++seq__7a6ddf486bb514d6f1d2__2023-04-23-10-07-12/1009`

官方 annotation：
`/mnt/ugreen_nas/storage/Ref2Dex_storage/OakInk2/downloads/hf/OakInk-v2/anno_preview/scene_02__A008++seq__7a6ddf486bb514d6f1d2__2023-04-23-10-07-12.pkl`

对应 selection：
`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_raw/oakink2_inspire_selection_v1_4_23/full_20260917T091507Z/index.json`

MANO cache：
`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/oakink2_mano_20260919T075300Z`

该序列的 291 个 cache 帧与 selection 的 `selected_frame_ids` 完全一致，严格递增；cache `frame_time` 由
`selected_timeline_positions / 30` 写入，float32 序列化后的最大误差为约 `4.45e-7 s`。

## Reproduction

峰值关联的 source IDs 为 `1231, 1235, 1239, 1243`，cache index 为 `141, 142, 143, 144`。
使用 exporter 相同的：

- `annotation["raw_mano"][frame]` 中的 `lh/rh__pose_coeffs`、`__betas`、`__tsl`；
- `_ManoReconstructor`；
- `build_bilateral_mano_v1_4_cache._sample_surface_chunk`；
- 固定 MANO correspondence；

重新生成双手 4,096 点后，与 cache 逐点比较：

- 最大绝对误差：`8.94e-8 m`（约 `0.000089 mm`）；
- 平均绝对误差：`8.81e-9 m`；
- RMS 误差：`1.56e-8 m`；
- 三个相邻 flow 最大值：重建 `61.2554 / 76.9864 / 58.7593 mm`，cache 为 `61.2554 / 76.9864 / 58.7593 mm`。

官方 MANO translation 也显示同一运动：source `1231→1243` 的右手根平移为
`117.20 mm`，左手根平移为 `62.19 mm`；这不是 viewer 或 cache 产生的位移。

## Interpretation and limits

可以确认：viewer 展示的是 OakInk2 官方 `.pkl` MANO annotation 经固定重建/采样后的原始标注轨迹，不是 cache 生成的伪运动，也不是跨时间 gap 的那 201 mm 异常帧。

不能仅凭这项复核确认：官方 MANO 拟合是否完全符合原始录制中的真实手部物理运动。要确认这一层，需要进一步对照 OakInk2 原始 RGB/深度视频或官方拟合可视化。当前没有新增过滤或修改任何 cache。
