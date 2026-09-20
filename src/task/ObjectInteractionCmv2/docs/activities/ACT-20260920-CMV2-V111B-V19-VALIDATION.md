# V1.11b 二域混合训练的 V1.9 口径 validation 评估

- timestamp: 2026-09-20 02:59:16 +0000
- activity_id: ACT-20260920-CMV2-V111B-V19-VALIDATION
- work_version: V1.11.1
- mode: run
- change_level: L3
- approval: user-approved
- approval_basis: 用户要求按此前方案一致的二域混合评估评估当前结果。
- branch: cmv2
- git_commit: 2a12812
- run_id: cmv2_v111b_articulated_best_e9_validation_grab1_arctic5to10_20260920T025629Z
- run_status: COMPLETED
- last_epoch: 9
- best_metric: 0.0704961（训练时的 stride=2 等权 validation selection metric）
- checkpoint: `/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v111_articulated_random_init_stride1to3_formal_20260919T075300Z/best.pt`

## Protocol

独立 output 使用 V1.9 冻结 validation protocol：固定 source split、GRAB stride=1、ARCTIC 稳定分配 stride=5..10、batch=64、GPU1、seed=42；严格加载 V1.11 epoch-9 best checkpoint。未修改训练、checkpoint、cache、GT 或 split。

## Results

- GRAB：40,762 samples，flow EPE `2.4704 mm`，预测/GT 流模长 `6.8527/6.9033 mm`，平均流向角 `58.19°`。
- ARCTIC：25,522 samples，flow EPE `21.8013 mm`，预测/GT 流模长 `24.6826/31.5577 mm`，平均流向角 `52.30°`。
- ARCTIC stride 5/6/7/8/9/10 EPE（mm）：`14.0623`、`17.4375`、`19.7285`、`22.9390`、`27.2737`、`29.3766`。
- 相比 V1.9 同口径已完成 V1.8 baseline：GRAB EPE 降 `19.7%`（3.0771→2.4704 mm），ARCTIC pooled EPE 升 `13.6%`（19.1902→21.8013 mm）；ARCTIC 各 stride 均变差，且恶化随 stride 增大。

## Conclusion and limits

工程结论为 `SUPPORTED`：V1.11 checkpoint 可由相同口径 evaluator 完整、可复现地评估。科学结论为 `INCONCLUSIVE`：短 stride=1..3 训练改善 GRAB，但在长 stride ARCTIC validation 上退化；ARCTIC 无独立 test split，不得解释为 held-out test 或泛化结论。

## Artifacts and rollback

评估目录：`/mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v111b_articulated_best_e9_validation_grab1_arctic5to10_20260920T025629Z`，含 `run_manifest.json`、`metrics_summary.json`、`metrics.jsonl`、`eval.log`。删除该独立 output 或反转 evaluator/config 提交不会影响训练或 cache。
