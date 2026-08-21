# ARCTIC min11 V1：noPCA 与 5mm object-only 快速评估

## 评测范围

- 物体：11/11 个 ARCTIC 物体类别。
- 文件：11 个 `.npz`，每个物体一个确定性样本；共约 4175 帧。
- subject：仅 `s01`，因此不是跨 subject 的最终 benchmark。
- 子集清单：`arctic_min11_v1_manifest.json`。
- 运行结果 JSON：`output/research/arctic_min11_v1/nopca.json`、`output/research/arctic_min11_v1/5mm.json`。

## 统一协议

- 输入：`stored_clean_hand_points`。
- hand/PCA perturbation：关闭。
- runtime object resampling：关闭。
- object perturbation：开启，rotation std `10 deg`、translation std `10 mm`，概率 `1.0`。
- evaluator：`research/contactpose_checkpoint_compare/evaluate.py`，无代码改动。

## 结果

数值越低越好；recovery 指标越高越好。

| 指标 | noPCA (step 317800) | 5mm (step 454000) |
| --- | ---: | ---: |
| clean random QFL | 0.0007073 | **0.0001345** |
| clean random MAE | 0.018176 | **0.008157** |
| clean contact auxiliary QFL | 0.017003 | **0.003130** |
| perturbed random QFL | **0.0008409** | 0.0009378 |
| perturbed random MAE | **0.018949** | 0.011547 |
| perturbed contact auxiliary QFL | **0.020374** | 0.022239 |
| perturbed recovery Brier | **0.5760** | 0.3404 |
| perturbed recovery projection | **0.4967** | 0.2447 |

## 解释与限制

在这个快速 object-only protocol 下，5mm checkpoint 的 clean 拟合更好，但 noPCA 的 recovery 明显更好；noPCA 的 perturbed correspondence QFL / contact auxiliary QFL 也略好。由于 checkpoint 训练步数不同、5mm 训练到 454k 而 noPCA 只保存到 317.8k，且子集只含 `s01`、每个物体只有一个文件，本结果只能作为方向性筛查，不能证明 5mm recipe 的单变量因果优势。

该评测有意不加入 PCA/hand perturb：它回答的是“相同 object-only 输入扰动下”的问题，因此对这两个 checkpoint 是公平的；它不回答 hand-only 或 hand+object robustness。后续若比较完整扰动鲁棒性，必须在两版上同时打开同一 PCA hand perturb 强度，并单独报告 hand-only、object-only、hand+object 三行。
