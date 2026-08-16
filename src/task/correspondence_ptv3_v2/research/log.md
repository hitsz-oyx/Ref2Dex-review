# correspondence_ptv3_v2 研究日志

## 实验：新旧 checkpoint 的 ContactPose 跨数据集测试

**假设**
几何 9 mm 手扰动训练的新模型，相比旧 cosine-restart 模型，在未见过的 ContactPose 数据和物体位姿扰动下应表现出更好的恢复行为。

**观察到的失败 / 现象**
两次 GRAB 训练都没有配置独立 test split，仅比较各自 validation 指标会混入数据版本和划分差异。

**诊断**
使用本地 5 条 ContactPose Stage 3 v2.0 序列作为双方均未训练过的跨数据集测试集。该数据没有 MANO 参数，因此公平地关闭两边的 MANO forward、手部扰动与 runtime object resampling；分别测试 clean 输入和固定种子的 10° / 10 mm 物体扰动。

**改动**
新增 `research/contactpose_checkpoint_compare/evaluate.py`，从 checkpoint 内嵌配置恢复对应模型，并在相同测试协议下输出可复查 JSON。

**结果**
测试集包含 5 条序列、1728 帧；两边的 changed-edge fraction 均为 0.029910，说明使用了相同的确定性采样和扰动。best checkpoint 对比结果如下：

| 指标 | 旧 cosine-restart best | 新 geometry9mm best |
| --- | ---: | ---: |
| clean random QFL（低优） | 0.001129 | 0.001290 |
| perturbed random QFL（低优） | 0.001526 | 0.001814 |
| recovery Brier（高优） | 0.578640 | 0.478362 |
| recovery projection（高优） | 0.539929 | 0.402965 |
| fake-contact recovery Brier（高优） | 0.691990 | 0.653487 |
| missed-contact recovery Brier（高优） | 0.491841 | 0.344258 |

旧模型的 clean / perturbed QFL 分别低 12.4% / 15.9%，总体 recovery Brier 高 0.1003，差异主要来自 missed-contact recovery 高 0.1476。为排除 best checkpoint 选择偏差，又比较旧 latest（epoch 69）和新 epoch 41：旧模型的 clean / perturbed QFL 仍分别低 9.1% / 13.8%，recovery Brier / projection 分别高 0.0766 / 0.1378，排序保持一致。

分桶诊断显示新模型的输出发生概率收缩：clean 测试中 zero-target 预测均值由旧模型的 0.0154 升至 0.0280，nonzero-target 预测均值由 0.2356 降至 0.1894（GT 均值 0.2667）；新模型在 `y<0.25` 桶的 MAE 更低 0.0280，但在 `0.50<=y<0.75` 和 `y>=0.75` 桶的 MAE 分别高 0.0949 和 0.1209。新模型因此更少输出高置信接触，尤其损害 missed-contact recovery。

**决策**
当前证据不支持“9 mm 手扰动训练提高了 ContactPose 跨数据集恢复能力”的假设。保留新训练作为 GRAB 几何扰动实验，但若目标是当前 ContactPose 测试协议下的 correspondence 与 object-perturb recovery，应优先旧 cosine-restart checkpoint。该测试使用 ContactPose v2.0，无法验证 MANO hand perturbation 本身；需要 v2.1 ContactPose 才能单独回答手扰动恢复问题。

该对比不是单变量消融：新 run 同时改变了 MANO 手扰动、runtime object resampling、dense hand-contact 辅助头、global batch（旧 128、新 16）、学习率（旧 5e-5、新 1e-4）和 scheduler。且 ContactPose v2.0 迫使测试关闭 MANO forward/runtime resampling，输入协议更接近旧模型。因此不能据此断言 9 mm 噪声本身有害；下一步应在同一初始化、数据和优化配置下只切换 `apply_hand_perturb`，并用 ContactPose v2.1 测试。
