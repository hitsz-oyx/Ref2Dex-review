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
