# ARCTIC min11 MANO：noPCA 与 5mm 三条件评估

## 数据

- NAS Stage 3：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1`
- 覆盖 11/11 个物体类别，11 个左手文件，共 4175 帧。
- 仅 subject `s01`，且 `ketchup` 经过 frame filtering 后只有 15 帧，因此这是方向性小样本，不是最终 benchmark。
- MANO：axis-angle45，`mano_use_pca=false`；hand noise 通过已有 ARCTIC 9mm geometry calibration 缩放到 5mm RMS。
- 清单：[arctic_min11_mano_v1_manifest.json](arctic_min11_mano_v1_manifest.json)

## 统一协议

- 两个 checkpoint 使用同一 NAS Stage 3 文件、同一 batch size、同一随机采样。
- runtime object resampling：关闭。
- `object_only`：object 10°/10 mm，hand clean。
- `hand_only`：axis-angle45 hand 5mm RMS，object clean。
- `hand_object`：同时开启 axis-angle45 hand 5mm RMS 和 object 10°/10 mm；关闭 exclusive gate。
- 结果 JSON：`output/research/arctic_min11_mano_v1/` 下的六个 condition 文件。

## 结果

recovery 指标越高越好，QFL 越低越好。

| 条件 | 指标 | noPCA | 5mm |
| --- | --- | ---: | ---: |
| object-only | clean random QFL | 0.0002265 | **0.0000392** |
| object-only | perturbed random QFL | **0.0002819** | 0.0003012 |
| object-only | recovery Brier | **0.5355** | 0.2929 |
| object-only | recovery projection | **0.4636** | 0.2127 |
| hand-only | perturbed random QFL | 0.0002322 | **0.0000516** |
| hand-only | recovery Brier | -0.5066 | **0.2112** |
| hand-only | recovery projection | **0.4498** | 0.3596 |
| hand+object | perturbed random QFL | 0.0002862 | 0.0003169 |
| hand+object | recovery Brier | **0.5356** | 0.2813 |
| hand+object | recovery projection | **0.4581** | 0.2033 |

## 解释

5mm 在 clean 拟合和 hand-only 输入扰动上明显更好；noPCA 在 object-only 和 hand+object recovery 上明显更好。这个结果支持“5mm hand perturb 路径本身有效，但当前互斥训练没有让模型学好 object/joint recovery”的解释：5mm 训练约为 80% hand-only、20% object-only、0% compound，而 noPCA 约为 100% object-only。

这仍不是最终因果结论：两条 checkpoint 训练步数不同（noPCA 317800、5mm 454000），子集只含一个 subject，且 axis-angle45 5mm 不是 GRAB PCA24 的完全同分布噪声。下一步应做 `H80/O100`（允许 compound）与当前 `H80/O20` 的 matched-budget 对照，并保留这三个评测条件。
