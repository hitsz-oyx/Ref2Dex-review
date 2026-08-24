# 协议 E：ARCTIC MANO min11 10 mm 三条件评估

- generated_at: 2026-08-23
- 数据：`/mnt/ugreen_nas/storage/Ref2Dex_storage/processed_data/stage3/arctic_min11_mano_v1`
- 规模：11 个 s01 左手序列、11 个物体、4,175 帧

## 协议

协议 E 沿用协议 D 的数据和三条件结构，但把 hand axis-angle45 几何扰动从 5 mm 提高到 10 mm RMS。评估端固定 `hand_perturb_prob=1.0`，不继承 checkpoint 的 H80/H50 训练门控比例。2026-08-23 追加评估历史 GRAB+ContactPose 两域 mixed latest.pt（step 154670 / epoch 54）。

| 条件 | hand | object |
| --- | --- | --- |
| object-only | clean | 10° / 10 mm，概率 1.0 |
| hand-only | 10 mm RMS，概率 1.0 | clean |
| hand+object | 10 mm RMS，概率 1.0 | 10° / 10 mm，概率 1.0 |

batch size 16、num_workers 0、runtime object resampling 关闭。object-only 与协议 D 完全相同，直接复用既有 JSON；另外两种条件在修复评估门控后重新运行。

本文同时报告：

```text
ΔQFL = perturbed random QFL - condition-matched clean random QFL
```

`perturbed QFL` 和 `ΔQFL` 衡量实际绝对质量与退化量；`pseudo_recovery_brier` 衡量相对伪几何 baseline 的恢复比例。三者不能相互替代。

## Object-only

| checkpoint | clean QFL ↓ | perturbed QFL ↓ | ΔQFL ↓ | recovery Brier ↑ | changed-edge fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| noPCA latest | 0.00022651 | 0.00028191 | +0.00005540 | **0.5355** | 0.004370 |
| H80 latest | 0.00003922 | 0.00030117 | +0.00026195 | 0.2929 | 0.004370 |
| H80 best | **0.00003749** | 0.00030386 | +0.00026637 | 0.2787 | 0.004370 |
| H50 best | 0.00008600 | **0.00025914** | +0.00017314 | 0.4119 | 0.004370 |
| GRAB+ContactPose latest | 0.00098074 | 0.00098546 | **+0.00000472** | 0.3341 | 0.004370 |

## Hand-only（10 mm）

| checkpoint | clean QFL ↓ | perturbed QFL ↓ | ΔQFL ↓ | recovery Brier ↑ | changed-edge fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| noPCA latest | 0.00022660 | 0.00025284 | +0.00002624 | 0.2295 | 0.003200 |
| H80 latest | 0.00003929 | 0.00010836 | +0.00006907 | 0.3598 | 0.003200 |
| H80 best | **0.00003747** | **0.00010732** | +0.00006986 | **0.3621** | 0.003200 |
| H50 best | 0.00008615 | 0.00014954 | +0.00006339 | 0.3117 | 0.003200 |
| GRAB+ContactPose latest | 0.00098094 | 0.00096150 | **-0.00001944** | -0.2130 | 0.003200 |

## Hand+object（10 mm + 10°/10 mm）

| checkpoint | clean QFL ↓ | perturbed QFL ↓ | ΔQFL ↓ | recovery Brier ↑ | changed-edge fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| noPCA latest | 0.00022710 | 0.00030224 | +0.00007514 | **0.5367** | 0.004462 |
| H80 latest | 0.00003924 | 0.00036099 | +0.00032175 | 0.2631 | 0.004462 |
| H80 best | **0.00003752** | 0.00036498 | +0.00032747 | 0.2535 | 0.004462 |
| H50 best | 0.00008606 | **0.00029481** | +0.00020876 | 0.4028 | 0.004462 |
| GRAB+ContactPose latest | 0.00098086 | 0.00097180 | **-0.00000905** | 0.3805 | 0.004462 |

## 三条件等权汇总

| checkpoint | Balanced perturbed QFL ↓ | Balanced ΔQFL ↓ | Balanced recovery Brier ↑ |
| --- | ---: | ---: | ---: |
| noPCA latest | 0.00027900 | +0.00005226 | **0.4339** |
| H80 latest | 0.00025684 | +0.00021759 | 0.3053 |
| H80 best | 0.00025872 | +0.00022123 | 0.2981 |
| H50 best | **0.00023450** | +0.00014843 | 0.3755 |
| GRAB+ContactPose latest | 0.00097292 | **-0.00000792** | 0.1672 |

Balanced 指标是三个条件的算术平均，每个条件等权。由于 joint recovery 仍可能由 object 扰动能量主导，Balanced recovery 不能代替三个分项。

## 结论

1. 10 mm hand-only 下，noPCA 的 recovery 为 0.2295，不再呈现旧 5 mm 表中的巨大负值；其绝对 ΔQFL 也是四者最小。这支持“小扰动能量会放大比值指标观感”的判断。
2. noPCA 的 perturbed QFL 仍然最高，原因是 clean 基线本身明显较差。因此“退化量小”不等于“最终预测最好”。
3. H80 best 的 hand-only 最终 QFL 和 recovery 最好；H50 best 的三条件等权 perturbed QFL 最好，表现为更均衡的绝对质量候选。
4. noPCA 在 joint recovery 上仍最高，但 joint changed-edge 能量包含 object 主导成分，不能据此宣称其 hand robustness 最好。
5. min11 仅有 s01，且 checkpoint 选择与训练预算不匹配；协议 E 仍是机制筛查，不是最终 benchmark。
6. 历史 GRAB+ContactPose 的 clean / perturbed QFL 都约为 0.001，明显高于另外四条新路线。其 ΔQFL 为负只表示扰动后指标略低于自身较差的 clean 基线，不能解释成最佳鲁棒性；它的 Balanced recovery 也是五者最低。

## 评估实现核验

本次核验发现旧 evaluator 会继承 checkpoint 的 `hand_perturb_prob`：H80/noPCA 为 0.8，H50 为 0.5，导致协议 D 的 hand-only / hand+object 输入并不一致。现已在 evaluator 中固定评估端概率为 1.0，并在输出协议字段写入 `hand_perturb_probability`。协议 D 的 object-only 结果不受影响；旧 hand-only / hand+object 数字应视为 `INVALID_IMPLEMENTATION`，不能继续用于公平排名。

## 产物

- JSON / log：`output/research/arctic_min11_mano_protocol_e_10mm_20260823/`
- evaluator：`src/task/correspondence_ptv3_v2/research/contactpose_checkpoint_compare/evaluate.py`
