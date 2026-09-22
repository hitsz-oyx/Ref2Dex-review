# OakInk2 MANO 最大点流帧查看器

- timestamp: 2026-09-20T04:54:27Z
- activity_id: ACT-20260920-CM-OAKINK2-MANO-VIEWER
- work_version: V1.4.22
- mode: change → run
- change_level: L1
- approval: user-requested
- branch: cmv2
- base_commit: b9acd32b0bc7aa7aca73e1a6b3f0a512cbf8dadb
- scope: 让既有 `visualize_grab.py` 只读识别 OakInk2 MANO cache，并启动点云查看器。
- run_id: oakink2_mano_max_view_20260920T045427Z
- run_status: RUNNING
- conclusion: N/A（可视化检查，不是科研效果结论）

## Change

`visualize_grab.py` 新增 OakInk2 index/schema 识别；OakInk2 使用 `knn_hand_points_world.npy` /
`knn_hand_normals_world.npy` 的双手 4,096 点合同。旧 GRAB/ARCTIC index 与 bilateral cache
路径保持原读取逻辑。OakInk2 当前只启用点云查看；该 cache 不含此脚本所需的 GRAB object mesh
和 MANO parent mesh，Mesh 关闭时不影响点云。

新增定向测试覆盖 OakInk2 schema、4096 点合同和现有 viewer 行为。

## Viewer run

- command: `/home/wbcd/miniconda3/envs/graspenv/bin/python -m src.task.ObjectInteractionCm.visualize_grab --index /mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/oicm_v1_4_5_highres_full/oakink2_mano_20260919T075300Z/index.json --split train --sequence oakink2/scene_02__A008++seq__7a6ddf486bb514d6f1d2__2023-04-23-10-07-12/1009 --frame 141 --future-delta 3 --point-display both --mesh-display off --port 8101`
- endpoint: `http://127.0.0.1:8101`
- pid: `3376712`
- initial frame: cache `141 → 144` (source `1231 → 1243`), the stride-3 retained maximum.
- point flow: hand max `185.1777 mm`; object max `172.0705 mm`.

## Protected and verification

- OakInk2 cache, source data, split, coordinates, units, GT, loader, training and checkpoints were not modified.
- `pytest -q src/task/ObjectInteractionCm/tests/test_visualize_grab_v1_4.py`: 6 passed.
- `py_compile` and `git diff --check`: passed.
- The viewer is point-cloud only; no mesh availability is inferred from this run.
- Rollback: revert the two viewer/test files; stop PID `3376712`. No data rollback is needed.
