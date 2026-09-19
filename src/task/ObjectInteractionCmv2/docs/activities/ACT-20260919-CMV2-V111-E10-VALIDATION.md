# V1.11 epoch 10 validation 复核

- timestamp: 2026-09-19 16:10:00 +0000
- activity_id: ACT-20260919-CMV2-V111-E10-VALIDATION
- work_version: V1.11.1
- mode: run
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求先评估当前中断训练结果。
- branch: cmv2
- git_commit: 2270051
- run_id: cmv2_v111_articulated_random_init_stride1to3_formal_20260919T075300Z
- run_status: UNKNOWN
- last_epoch: 10
- latest_checkpoint: `/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v111_articulated_random_init_stride1to3_formal_20260919T075300Z/latest.pt`

## Protocol

以 V1.11 固定 config、完整 validation split 和 `eval_stride=2` 加载 `latest.pt`，严格加载模型权重后调用既有 articulated validation。未改写训练输出、cache、checkpoint、split、模型或配置。

## Result and limits

- epoch 10 / step 76000：GRAB validation loss `0.0494743`，ARCTIC validation loss `0.0964230`，等权 selection metric `0.0729486`。
- 训练日志中的 epoch 9 `best.pt` selection metric 为 `0.0704961`，故 epoch 10 比当前最佳差 `3.48%`；应继续保留 `best.pt` 作为选择 checkpoint。
- 原训练 manifest 仍为 `RUNNING`，但训练进程已不存在且未写终态；因此此处标为 `UNKNOWN`，不将进程缺失臆断为用户停止或实现失败。
- 这是 validation 证据；ARCTIC 没有独立 test split，不能推断测试集或跨域泛化表现。

## Verification

使用当前 `2270051` 的 `train_articulated_formal._load`、`_datasets(..., validation=True)` 与 `_evaluate`；checkpoint architecture/version/seed 与 V1.11 config 严格匹配。

## Rollback

本 Activity 仅记录只读评估事实；删除该文档不会影响任何训练、cache 或 checkpoint。
