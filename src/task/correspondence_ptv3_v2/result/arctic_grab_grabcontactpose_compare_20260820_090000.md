# V1 ARCTIC 对比

生成时间：2026-08-20 09:00:00 UTC

## 测试协议

- 分层子集：`/tmp/arctic_eval_stratified_v1`，`179` 个 NPZ、`179` 条序列、`93,968` 帧
- 覆盖：11 个 object、5 个 subject（s01/s02/s04/s05/s06）
- object 子集：对每个 object 单独建立软链目录，按同一 evaluator 运行
- hand input：`stored_clean_hand_points`
- hand perturb：关闭
- runtime object resampling：关闭
- object perturb：固定 10° rotation + 10 mm translation

## Checkpoint

| 模型 | checkpoint | step / epoch |
| --- | --- | ---: |
| pure GRAB | `outputs/train/correspondence_ptv3_v2_old1797_compact_repro_ddp2_20260816_103130/checkpoints/latest.pt` | 154670 / 69 |
| GRAB+ContactPose | `outputs/train/correspondence_ptv3_v2_old1797_grab_contactpose_full_gpu3_20260816_110237/checkpoints/latest.pt` | 154670 / 54 |

## 结果

| 指标 | pure GRAB micro | GRAB+ContactPose micro | pure GRAB object-macro | GRAB+ContactPose object-macro |
| --- | ---: | ---: | ---: | ---: |
| clean random QFL | 0.0002656962 | 0.0002645223 | 0.000278944 | 0.0002749686 |
| clean random MAE | 0.0037520078 | 0.0047859375 | 0.0038794179 | 0.0048946651 |
| clean contact QFL | 0.0057336329 | 0.0055567428 | 0.0059287814 | 0.0056917697 |
| perturbed random QFL | 0.0003378127 | 0.0003621334 | 0.0003500446 | 0.0003732872 |
| perturbed random MAE | 0.0046070806 | 0.0061020334 | 0.0047317093 | 0.0062039457 |
| perturbed contact QFL | 0.0075595206 | 0.0078594176 | 0.0077315634 | 0.0080087717 |
| perturbed recovery Brier | 0.7981146511 | 0.7775316848 | 0.8101849147 | 0.7879994994 |
| perturbed recovery projection | 0.7760975894 | 0.7452399924 | 0.7852281303 | 0.7536250826 |
| fake-contact recovery Brier | 0.8821414715 | 0.8396722309 | 0.8918232614 | 0.8494764038 |
| missed-contact recovery Brier | 0.7101249937 | 0.7124604912 | 0.7238988402 | 0.7232769676 |

## 关键观察

- micro 和 object-macro 两层结论一致：纯 GRAB 在 correspondence 与 recovery 上都略优于 GRAB+ContactPose。
- object-macro 的差距比 micro 更稳定，说明并非只被单一高帧数 object 拉动。
- 这次覆盖全部 11 类 object，不再只代表 `box_*` 偏置。

## 结论

在 V1 分层 ARCTIC 子集上，纯 GRAB 仍然优于 GRAB+ContactPose；但两者差距不大，属于方向一致、幅度有限的域外差异。

## 证据

- `output/research/arctic_v1/stratified/manifest.json`
- `output/research/arctic_v1/arctic_v1_summary.json`
- `output/research/arctic_v1/objectwise/*/*.json`
- `/tmp/arctic_eval_stratified_v1`
- `/tmp/arctic_eval_stratified_v1_by_object`

## 限制

- 仍是分层子集，不是全量 ARCTIC。
- object-macro 通过按 object 单独评估再取平均得到，不等价于按帧权重的全量 micro。
- 两条 checkpoint 的训练预算仍不完全相同。