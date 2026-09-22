# V1.11a 高分辨率 cache 恢复运行

- timestamp: 2026-09-19 16:27:00 +0000
- activity_id: ACT-20260919-CMV2-V111A-HIGHRES-RUNS
- work_version: V1.11.1
- mode: run
- change_level: L3
- approval: user-approved
- approval_basis: 用户确认 V1.11a 的 OakInk2 MANO、GRAB Inspire 和 ARCTIC Inspire 高分辨率 cache 范围。
- branch: cmv2
- git_commit: e2c94e4
- run_id: cmv2_v111a_oakink2_mano_resume_knnfb16_20260919T162700Z
- run_status: RUNNING

## Contract

先恢复 OakInk2 MANO output root；每侧 2,048、双侧 4,096 点、KNN32。已完成 geometry 由 `--resume` 验证并复用，新的 run manifest 使用 `work_version: V1.11.1`。完成后才顺序运行 GRAB/ARCTIC Inspire，每侧 10,135、双侧 20,270 点、KNN32。

## Command

`PYTHONPATH=/home/wbcd/workspace/oyx_ws/Ref2Dex PYTHONDONTWRITEBYTECODE=1 /home/wbcd/miniconda3/envs/graspenv/bin/python -u -m src.task.ObjectInteractionCmv2.tools.data.build_oakink2_mano_highres_v1_4 ... --device cuda:2 --mano-batch-size 128 --knn-frame-batch 16 --knn-object-chunk 512 --run-id cmv2_v111a_oakink2_mano_resume_knnfb16_20260919T162700Z --resume`

## Initial evidence

- GPU2 process 已启动；manifest 记录 `work_version: V1.11.1`。
- 校验阶段计数从零重新累计，首个检查为 6/1849 段、2743 帧、`failures=[]`；该计数不表示重写既有 geometry。

## Rollback

停止当前 PID 即可；保留已原子完成段和 manifest，不删除 output root 或任一 cache。
