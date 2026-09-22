# OakInk2 MANO cache 短 stride 点流模长统计

- timestamp: 2026-09-20 +0000
- activity_id: ACT-20260920-CMV2-OAKINK2-MANO-FLOW-MAGNITUDE
- work_version: V1.11.1
- mode: inspect
- change_level: L0
- branch: cmv2
- git_commit: 0abbbca
- scope: 已完成 OakInk2 MANO 高分辨率 cache 的全量只读描述统计。

## Protocol

输入为 `/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/oakink2_mano_20260919T075300Z` 的 1,849 个 train 序列（379,591 帧）。对训练实际消费的 `knn_hand_points_world.npy`（双手 4,096 MANO 点）按 cache 帧索引计算 `||H[t+s] - H[t]||_2`，其中 `s` 为 1、2、3。范数在 world 与 current object frame 中相同；因此不改变训练的 flow 定义。

## Results

| Cache stride | Transitions | Point samples | Mean mm | Std mm | Max mm |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 377742 | 1547231232 | 2.2046 | 3.1616 | 201.2795 |
| 2 | 375893 | 1539657728 | 4.2825 | 6.3013 | 213.0104 |
| 3 | 374044 | 1532084224 | 6.3297 | 9.3973 | 223.3127 |

`source_frame_id` 的主众数分别为 `4/8/12`，对应 cache stride `1/2/3`；cache 的有效帧率仍为 30 Hz。故这里的 stride 指连续 cache 帧的索引间隔，不能解释成 raw source ID 增量为 1/2/3。

## Verification and rollback

全量扫描无写入、无失败；统计临时日志已从 `/tmp` 移除。回滚仅删除本 Activity 和索引条目，不影响任何 cache 或运行产物。
