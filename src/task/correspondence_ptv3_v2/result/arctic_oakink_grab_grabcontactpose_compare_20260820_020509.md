# OakInk / GRAB / GRAB+ContactPose ARCTIC 对比

生成时间：2026-08-20 02:05:09 UTC

## 测试协议

- 数据：ARCTIC 子集 `tmp/arctic_eval_subset_20260817`
- 规模：20 个文件，12,389 帧
- 手输入：`stored_clean_hand_points`
- 手部扰动：关闭
- runtime object resampling：关闭
- object perturb：固定 10° rotation + 10 mm translation
- QFL / MAE：越低越好
- recovery Brier / projection：越高越好

## Checkpoint

| 模型 | checkpoint | step / epoch | 结果来源 |
| --- | --- | ---: | --- |
| OakInk | `outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_oakink_50ep_no_hand_perturb_runtime_20260818_145747/checkpoints/best.pt` | 140940 / 10 | 本次补测 |
| GRAB | `outputs/train/correspondence_ptv3_v2_old1797_compact_repro_ddp2_20260816_103130/checkpoints/latest.pt` | 154670 / 69 | 已有 ARCTIC 日志 |
| GRAB+ContactPose | `outputs/train/correspondence_ptv3_v2_old1797_grab_contactpose_full_gpu3_20260816_110237/checkpoints/latest.pt` | 154670 / 54 | 已有 ARCTIC 日志 |

## 训练配置摘要

| 项 | OakInk | GRAB | GRAB+ContactPose |
| --- | --- | --- | --- |
| 训练数据 | OakInk full `stage3_corr_oakink_mano_face_handroot_20260818` | GRAB compact repro | GRAB compact repro + ContactPose full |
| 训练入口 | `src/task/correspondence_ptv3_v2/configs/oakink_50ep_no_hand_perturb_runtime.yaml` | old1797 compact repro run 内嵌配置 | old1797 GRAB+ContactPose run 内嵌配置 |
| batch / GPU | 单卡 batch 16 | 双卡 global batch 128 | 单卡 batch 64 |
| 手部扰动 | 关闭 PCA/MANO hand perturb | 关闭 PCA/MANO hand perturb | 关闭 PCA/MANO hand perturb |
| runtime object resampling | 开启；stored-hand FPS-256 proxy | 旧 compact 训练协议 | 旧 compact 训练协议 |
| object perturb | 10° rotation / 10 mm translation | 10° rotation / 10 mm translation | 10° rotation / 10 mm translation |
| loss | cross-edge + contact auxiliary + hand-contact | cross-edge + contact auxiliary | cross-edge + contact auxiliary |
| scheduler / optimizer | cosine / AdamW，50 epoch 目标 | cosine-restart 续训 / AdamW | cosine-restart 续训 / AdamW |
| 备注 | OakInk 无 MANO pose/betas，训练协议不与 GRAB 严格同源 | 8 月中旬 no-PCA 默认对照 | 加入 ContactPose 后的联合训练对照 |

## 结果

| 指标 | OakInk | GRAB | GRAB+ContactPose |
| --- | ---: | ---: | ---: |
| clean random QFL | 0.0004892 | **0.0001431** | 0.0001669 |
| perturbed random QFL | 0.0007355 | **0.0002136** | 0.0002670 |
| clean random MAE | 0.026147 | **0.002711** | 0.004029 |
| perturbed random MAE | 0.026676 | **0.003699** | 0.005797 |
| clean contact auxiliary QFL | 0.010313 | **0.004611** | 0.004878 |
| perturbed contact auxiliary QFL | 0.018312 | **0.006532** | 0.007280 |
| perturbed recovery Brier | 0.5361 | **0.8383** | 0.8000 |
| perturbed recovery projection | 0.4246 | **0.8120** | 0.7728 |
| fake-contact recovery Brier | 0.6692 | **0.9066** | 0.8484 |
| missed-contact recovery Brier | 0.4005 | **0.7687** | 0.7507 |

## 结论

在这个 ARCTIC 子集上，纯 GRAB 的 correspondence 拟合最好，GRAB+ContactPose 次之，OakInk 明显落后。recovery 指标同样是纯 GRAB 最好，GRAB+ContactPose 略弱于纯 GRAB，OakInk 最差；这与旧日志里“recovery 略有改善”的表述不一致，当前以实际指标和任务架构中的高优定义为准。

OakInk 的 recovery 明显低于另外两者，但仍高于随机意义不大，说明它可以完成一定程度的 object perturb recovery。由于 OakInk checkpoint 的训练数据表示、训练协议和另外两套 checkpoint 不完全同源，当前结果应视为方向性比较，不能单独归因于数据集规模或联合训练本身。

clean recovery 指标没有列出，因为 clean 输入没有 changed edge，不定义 pseudo recovery。

## 来源与可复查文件

- 完整汇总：`src/task/correspondence_ptv3_v2/result/arctic_oakink_grab_grabcontactpose_compare_20260820_020509.json`
- OakInk 本次评测临时 JSON：`/tmp/oakink_arctic_eval_20260820.json`
- 纯 GRAB 已有结果：`output/research/contactpose_checkpoint_compare/arctic_pure_20260817_132348.json`
- GRAB+ContactPose 已有结果：`output/research/contactpose_checkpoint_compare/arctic_mixed_20260817_132348.json`
- 历史评测说明：`src/task/correspondence_ptv3_v2/research/log.md` 的“ARCTIC 子集独立评估”段落

## 限制

- 只评估了 20 个文件、12,389 帧，不代表全量 ARCTIC。
- 三套 checkpoint 不是同一训练初始化、同一训练预算和同一数据协议下的严格单变量消融。
- 纯 GRAB 与 GRAB+ContactPose 使用已有历史结果；OakInk 为本次相同协议补测。
