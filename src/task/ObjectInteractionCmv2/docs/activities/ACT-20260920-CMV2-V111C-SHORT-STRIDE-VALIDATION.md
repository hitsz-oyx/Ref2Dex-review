# V1.11c stride 1/2/3 二域 validation 评估

- timestamp: 2026-09-20 03:37:42 +0000
- activity_id: ACT-20260920-CMV2-V111C-SHORT-STRIDE-VALIDATION
- work_version: V1.11.1
- mode: run
- change_level: L3
- approval: user-approved
- approval_basis: 用户明确要求按 stride 1、2、3 分开统计 EPE、预测模长和 GT 模长。
- branch: cmv2
- git_commit: d33d6c3
- run_id: cmv2_v111c_articulated_best_e9_validation_stride1to3_20260920T033458Z
- run_status: COMPLETED
- last_epoch: 9
- best_metric: 0.0704961

## Protocol

GRAB 与 ARCTIC 都在固定 validation split 上按稳定规则分配 stride=1/2/3；checkpoint 为 V1.11 epoch-9 `best.pt`，batch=64、GPU1、seed=42。结果按域和实际 stride 分开统计。

## Results

| Domain | Stride | Samples | EPE mm | Pred magnitude mm | GT magnitude mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRAB | 1 | 13591 | 2.4432 | 6.7856 | 6.8371 |
| GRAB | 2 | 13244 | 4.4536 | 13.5573 | 13.7225 |
| GRAB | 3 | 13659 | 6.3954 | 20.4168 | 20.3857 |
| ARCTIC | 1 | 8625 | 3.3647 | 4.0266 | 4.7250 |
| ARCTIC | 2 | 8646 | 6.0829 | 7.0753 | 9.3648 |
| ARCTIC | 3 | 8496 | 8.3741 | 10.5956 | 13.7160 |

## Conclusion and limits

工程结论 `SUPPORTED`：sample accounting、每个 stride 分组和独立 run manifest 均完成。科学结论 `INCONCLUSIVE`：短 stride 范围内，GRAB 的预测/GT 流模长接近；ARCTIC 三档均低估 GT 流模长，且 EPE 随 stride 增大。该结果仅为 validation，不是 ARCTIC held-out test。

## Artifacts and rollback

评估目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v111c_articulated_best_e9_validation_stride1to3_20260920T033458Z`，包含 run manifest、metrics summary、metrics jsonl 与 eval log。删除该独立 output 或反转 V1.11c evaluator/config 提交不影响训练或 cache。
