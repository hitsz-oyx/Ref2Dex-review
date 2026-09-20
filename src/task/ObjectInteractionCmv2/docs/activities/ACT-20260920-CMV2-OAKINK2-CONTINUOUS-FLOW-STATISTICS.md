# OakInk2 MANO 时间连续性过滤后的点流模长统计

- timestamp: 2026-09-20T04:42:31Z
- activity_id: ACT-20260920-CMV2-OAKINK2-CONTINUOUS-FLOW-STATISTICS
- work_version: V1.11.1
- mode: inspect → run（只读统计）→ change（仅 L0 记录）
- change_level: L0
- approval: user-requested
- approval_basis: 用户要求恢复会话 `01a0b2c2-49f2-7870-b63c-cb765f7fd195` 并继续统计 OakInk2；沿用已批准的 V1.11d 连续性规则。
- branch: cmv2（续接既有工作，无新分支或提交）
- base_commit: b9acd32b0bc7aa7aca73e1a6b3f0a512cbf8dadb
- scope: OakInk2 MANO train cache 全量逐 stride 描述统计、峰值复核及本 Activity；不修改运行时实现或研究语义。
- run_id: oakink2_mano_continuous_flow_20260920T043800Z
- run_status: COMPLETED
- scientific conclusion: N/A（描述统计，不是模型效果或原始标注质量结论）

## 依据与输入

续接 [首次模长统计](ACT-20260920-CMV2-OAKINK2-MANO-FLOW-MAGNITUDE.md) 与
[V1.11d 时间连续性保护](ACT-20260920-CMV2-V111D-OAKINK2-TIMELINE-GUARD.md)，
适用计划为 [V1.11](../plan/V1.11.md) 与 [V1.11d](../plan/V1.11d.md)。

输入根：
`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/oakink2_mano_20260919T075300Z`。
输入 `index.json` 与 `cache_manifest.json` 确认 1,849 段、379,591 帧，全部为现有 train，
val/test 均为空；MANO 双手 4,096 点（左 2,048 后接右 2,048）。
扫描前后 index SHA-256 均为
`c4c635a5b584f83a23e1abe4abf3cc80b8cba91889e258e6e7b46ee9f580b751`。

## 口径

逐序列读取 `geometry/knn_hand_points_world.npy`、`frame_time.npy`、
`source_frame_id.npy` 与 geometry manifest。对每个固定 stride=1、2、3 的全部
`t → t+stride` 帧对求同一对应点的欧氏位移；不按随机 stride 分组、不额外启用 contact mask。
统计使用 world 坐标，位移范数等于两帧手点均变换至当前物体坐标系后的范数。
位移为 float32，转为 mm 后以 float64 累加，标准差为总体标准差。
均值对所有有效点样本等权，并非各序列均值的平均。

保留规则与现有 loader 相同：
`isclose(frame_time[t+s] - frame_time[t], s / effective_fps, rtol=0, atol=1e-4)`，
所有序列 `effective_fps=30`。本次不引入额外筛选条件。
分位数来自所有保留点样本的 0.01 mm 直方图分箱，不是抽样；表中给出分箱区间。
所有点位移有限；1,000 mm 以上溢出计数为零。

## 结果

| Stride | 原帧对 | 剔除 | 保留帧对 | 保留点样本 | 均值 mm | 标准差 mm | 最大值 mm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 377742 | 11 | 377731 | 1547186176 | 2.20377187 | 3.14922535 | 76.98639679 |
| 2 | 375893 | 18 | 375875 | 1539584000 | 4.28081662 | 6.28847930 | 137.73127747 |
| 3 | 374044 | 25 | 374019 | 1531981824 | 6.32724291 | 9.38419021 | 185.17768860 |

| Stride | P50 mm | P90 mm | P95 mm | P99 mm | P99.9 mm |
| ---: | --- | --- | --- | --- | --- |
| 1 | [0.94, 0.95) | [6.00, 6.01) | [8.62, 8.63) | [15.01, 15.02) | [23.97, 23.98) |
| 2 | [1.76, 1.77) | [11.91, 11.92) | [17.14, 17.15) | [29.82, 29.83) | [47.42, 47.43) |
| 3 | [2.57, 2.58) | [17.76, 17.77) | [25.58, 25.59) | [44.44, 44.45) | [70.25, 70.26) |

同时重算未过滤结果，均值 2.20461339 / 4.28245650 / 6.32967839 mm，
最大值 201.27947998 / 213.01036072 / 223.31274414 mm，
与上次记录的四位小数一致。过滤只使均值降低约 0.038%，但移除了之前的三档最大值。
11 个相邻时间断点分布在 9 个序列。

## 新最大值复核

三档保留最大值均来自：
`oakink2/scene_02__A008++seq__7a6ddf486bb514d6f1d2__2023-04-23-10-07-12/1009`。

| Stride | Cache 帧（从 0 计数） | Source ID | 点索引（双手合并） | 实际间隔 s |
| ---: | --- | --- | ---: | ---: |
| 1 | 142 → 143 | 1235 → 1239 | 3504（右手） | 0.0333337784 |
| 2 | 141 → 143 | 1231 → 1239 | 3504（右手） | 0.0666666031 |
| 3 | 141 → 144 | 1231 → 1243 | 2663（右手） | 0.0999994278 |

直接调用当前 `ThreeDomainTransitions._timeline_is_continuous`，三档保留峰值均通过，
三档历史 gap 峰值均拒绝；以 float64 坐标差独立复算六个峰值，误差均小于 0.0001 mm。
峰值 frame 142→143 的右手点平均位移为 63.3945 mm，中位数为 64.6783 mm，
右手质心位移为 63.2160 mm；并非只有一个表面点的大位移。
相邻 frame 141→142、142→143、143→144 的右手平均位移分别为
39.3616、63.3945、48.9923 mm，时间间隔均正常。

这些结果排除了该峰值由已知时间 gap 引起，但仅凭 cache 不能确认是实际快速运动还是原始
MANO 标注/拟合变化。未检查原始视频或重新拟合，不据此新增异常剔除规则。
也不能将手部输入流模长与模型物体预测 EPE/GT 流直接当作同一指标比较。

## 运行、产物与复现

输出目录：
`data/processed_data/oicmv2_oakink2_flow_statistics/v1/oakink2_mano_continuous_flow_20260920T043800Z/`，
通过既有 `data/processed_data` 软链接写入 NAS，属于目录合同允许的派生统计，Git 忽略。

- `config.json`：统计口径、输入、解释器、预算和完整执行脚本快照。
- `run_manifest.json`：身份、输入、时间、终态与 index 校验值。
- `statistics.json`：过滤前、过滤后、被剔除组的完整统计和最大值定位。
- `progress.jsonl`：扫描进度及完整终态结果。
- `verification_config.json`、`verification.json`：复核脚本与样本核算、真实 loader guard、float64 峰值和邻近轨迹结果。

命令为 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/wbcd/miniconda3/envs/p2v/bin/python -u -`，
标准输入为 `config.json` 的 `diagnostic_script` 原文。
复核命令另加 `PYTHONDONTWRITEBYTECODE=1`，标准输入见 `verification_config.json`。
重跑时必须新建独立输出目录并同步脚本中的 output；不覆盖本次产物。
CPU 单线程分块 128 帧，无 GPU；预算 900 s，实际 97.49 s；
2026-09-20T04:37:45Z 开始，04:39:23Z 完成。
last step/epoch、best metric、checkpoint、metrics.jsonl/train.log：N/A。

首次尝试 `oakink2_mano_continuous_flow_20260920T043600Z` 因遗漏路径中的
`geometry/` 层在首段读取前失败，未扫描或修改源数据。
该独立目录的 FAILED manifest/config/progress 保留，成功 run manifest 有回链；
修正读取路径后使用新 run_id，未覆盖失败记录。

## 验证、保护与回滚

- 全量 1,849 段、379,591 帧完成，保留/剔除帧对与 V1.11d 计数一致。
- 帧对 × 4,096 等于点样本数；原总和及平方和均等于保留与剔除两组之和。
- 直方图总数等于保留点数，扫描前后 index 校验值一致，六个峰值通过独立复算。
- 源数组均只读打开；数据 cache、selection、split、坐标、GT、KNN、loader、模型、
  checkpoint 和现有训练/导出进程保持不变。
- 本次只增加本 Activity、Activity 索引与 Git 忽略的独立统计产物，未提交。
- 统一验证基线在本次文档修改前已失败：
  `/home/wbcd/miniconda3/envs/openpi/bin/python tools/verify.py --changed`，
  governance 测试 9 passed / 1 failed，失败为既有
  `test_tracked_task_sources_and_configs_use_work_version` 的历史字段迁移遗留；
  不声称 VERIFY PASS。统计环境 p2v 无 pytest，验证使用既有 openpi 环境，未安装依赖。
- 回滚入口：移除本 Activity 与新增索引项；独立统计目录可保留审计，删除须遵循用户授权，
  不涉及源数据、cache 或运行时回滚。
- next step: 如需判断剩余大位移的原始标注质量，核对上述序列 source ID 1231–1243 的视频/MANO；
  本次统计本身已完成，不自动启动三域训练或新增过滤。
