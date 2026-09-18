# ObjectInteractionCmv2 实验记录

## EXP-20260918-025859-CMV2-V146-TWO-DOMAIN-FLOW-EVAL

- timestamp: `2026-09-18 02:58:59 +0000`
- modification_version: `V1.4.6`
- operation: `experiment`
- run_id: `cmv2_v146_two_domain_flow_eval_best12k_20260918T025800Z`
- run_status: `COMPLETED`
- base_commit: `7a674228ce022e61a7ab6d00756dfb9c3a55d082`
- checkpoint: V1.4.4 `best.pt`，epoch `13`；该 checkpoint 使用同一 validation sources 的 selection metric 选出。
- hypothesis: 在固定 best checkpoint 与用户指定的非全量 transition 样本上，量化 GRAB 与不同 ARCTIC stride 的 object flow 误差、幅值校准和方向误差；不宣称 held-out 泛化。
- outcome: `INCONCLUSIVE`

### 固定合同

- GRAB validation：stride=`1`，确定性无放回 `12000` transition。
- ARCTIC trajectory validation：stride=`5,6,7,8,9,10`，每项确定性无放回 `2000` transition。
- 每 transition 的 `1024` object point 均在 `object_pose_t` 下评估；EPE/magnitude 为 point-micro `mm`。
- angle 为逐点 `degree`，仅当 prediction 与 GT norm 都 `>=1e-6 m` 时聚合；静止/排除计数独立报告。

### 结果

| domain / stride | transition | EPE (mm) | pred mag (mm) | GT mag (mm) | angle (°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRAB / 1 | 12000 | 3.9029 | 10.5788 | 10.8730 | 41.5272 |
| ARCTIC / 5 | 2000 | 19.4863 | 8.5650 | 24.3412 | 55.4258 |
| ARCTIC / 6 | 2000 | 23.1555 | 9.9442 | 28.6224 | 53.0425 |
| ARCTIC / 7 | 2000 | 27.1467 | 11.6339 | 33.8581 | 52.6015 |
| ARCTIC / 8 | 2000 | 29.4669 | 12.0179 | 36.3264 | 53.0096 |
| ARCTIC / 9 | 2000 | 32.7959 | 13.3911 | 40.4781 | 51.8048 |
| ARCTIC / 10 | 2000 | 36.2683 | 13.7914 | 44.0146 | 51.5517 |
| ARCTIC pooled | 12000 | 28.0533 | 11.5572 | 34.6068 | 52.9060 |
| all pooled | 24000 | 15.9781 | 11.0680 | 22.7399 | 47.2166 |

GRAB 的 `12288000` point 中有 `1` 个 GT-static / angle-excluded point；ARCTIC `12288000` point 中均无静止或排除点；全体合计 `24576000` point 中 prediction-static 和 both-static 均为 `0`。

### 解释与证据

在这一固定 validation diagnostic 上，ARCTIC 的 EPE 由 stride 5 的 `19.4863 mm` 单调升至 stride 10 的 `36.2683 mm`；prediction magnitude 相对 GT magnitude 在 ARCTIC pooled 中明显偏小（`11.5572` 对 `34.6068 mm`）。这些是描述性统计，尚不构成模型相对基线的效果结论，亦不构成 held-out 泛化结论。

- [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v146_two_domain_flow_eval_best12k_20260918T025800Z/run_manifest.json)
- [metrics summary](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v146_two_domain_flow_eval_best12k_20260918T025800Z/metrics_summary.json)
- [sample index](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v146_two_domain_flow_eval_best12k_20260918T025800Z/sample_index.json)
- [activity](activity_log.md)

## EXP-20260918-145725-CMV2-V192-ARTICULATED-VALIDATION-EVAL

- timestamp: `2026-09-18 15:05:33 +0000`
- modification_version: `V1.9.2`
- operation: `experiment / validation diagnostic`
- run_id: `cmv2_v192_articulated_best_e9_validation_grab1_arctic5to10_20260918T145838Z`
- run_status: `COMPLETED`
- base_commit: `896323d9932dc897dad9318e5e20d4b380804932`
- checkpoint: V1.8.1 `best.pt` 在评估读取时解析为 epoch `11`、step `88770`、selection metric `0.09729951618777996`，SHA-256 `a04bf59890003bad6dec2bef3e7e7d5b0eff25d9e741b9b1afa56e1cf7172ddf`。
- hypothesis: 在当前 ARCTIC test split 缺失时，量化冻结 V1.5 articulated checkpoint 于现有 validation split 的 GRAB stride=1 与 ARCTIC stable stride=5..10 flow 表现；不把该诊断视为 held-out 泛化。
- outcome: `INCONCLUSIVE`

### 固定合同

- split：全量现有 `val`，GRAB `40762` transition、ARCTIC `25522` transition。
- GRAB：固定 stride=`1`；ARCTIC：以 seed=42、sequence/current frame stable assignment 在 stride=`5..10` 选取，并按实际 stride 分组。
- batch=`64`、GPU1、每 transition `1024` object points；点级 EPE/magnitude 是 micro average（mm）；flow angle 仅在 prediction 和 GT norm 均 `>=1e-6 m` 时统计。
- 评估与训练并发执行，但只读加载 checkpoint；产物 manifest 固化 checkpoint hash、epoch 和 step。

### 结果

| group | transition | EPE (mm) | pred mag (mm) | GT mag (mm) | angle (°) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRAB / 1 | 40762 | 3.0771 | 6.9118 | 6.9033 | 61.2356 |
| ARCTIC / pooled 5..10 | 25522 | 19.1902 | 25.1008 | 31.5577 | 49.9500 |
| ARCTIC / 5 | 4310 | 13.7949 | 17.8528 | 22.6057 | 50.8774 |
| ARCTIC / 6 | 4259 | 16.2314 | 20.7260 | 26.3443 | 50.8965 |
| ARCTIC / 7 | 4169 | 17.5448 | 23.9141 | 29.5176 | 49.4200 |
| ARCTIC / 8 | 4228 | 20.1664 | 26.5476 | 33.5507 | 48.9000 |
| ARCTIC / 9 | 4300 | 23.2573 | 29.4201 | 37.6468 | 49.6366 |
| ARCTIC / 10 | 4256 | 24.1478 | 32.1797 | 39.7068 | 49.9423 |

### 解释与证据

GRAB 的预测/GT flow magnitude 接近（`6.9118` 对 `6.9033` mm），而 ARCTIC pooled 的预测 magnitude 低于 GT（`25.1008` 对 `31.5577` mm）。ARCTIC EPE 随实际 stride 从 5 到 10 整体升高（`13.7949` 到 `24.1478` mm）。这只是同一 validation split 上的描述性诊断；ARCTIC 当前没有 held-out test bucket，且正式训练尚在运行，不能解释为最终泛化或相对基线结论。

- [run manifest](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v192_articulated_best_e9_validation_grab1_arctic5to10_20260918T145838Z/run_manifest.json)
- [metrics summary](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v192_articulated_best_e9_validation_grab1_arctic5to10_20260918T145838Z/metrics_summary.json)
- [metrics](../../../../../../../../../../mnt/ugreen_nas/storage/Ref2Dex_storage/outputs/ObjectInteractionCmv2/cmv2_v192_articulated_best_e9_validation_grab1_arctic5to10_20260918T145838Z/metrics.jsonl)
- [activity](activity_log.md)
