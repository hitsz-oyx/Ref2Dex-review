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

## 实验：旧 cosine-restart checkpoint 的 compact Stage 3 续训复现

**假设**
旧 checkpoint 在接近原配置下继续训练，可以复现原 run 后半段；旧 Stage 3 中没有训练消费者的 KNN 和诊断字段不应影响复现。

**观察到的失败 / 现象**
提交 `1797b28` 的 Dataset 强制要求旧 Stage 3 全字段，现有 compact Stage 3 因缺少 `obj_point_id`、`obj_to_hand_min_dist`、`hand_cano_points`、`hand_finger_id`、`hand_region_id` 而无法加载。尝试重建旧格式时即使将 KNN 降为 1，233 个文件已占 7.9 GB。

**诊断**
`gt_obj_to_hand_knn_idx` 在 v2 训练链路中完全不读取；object point ID 和 object-to-hand 距离只由 Dataset 输出，runner、模型和 loss 均无消费者。canonical hand 与语义字段仅在 `hand_context_dim>0` 时使用，而原配置为 0。现有 compact Stage 3 已包含几何、法线、采样 mask、hand-to-object 距离和稳定采样标识，足够执行原配置。

**改动**
停止旧 Stage 3 重建。在提交 `1797b28` 的独立 worktree 中增加受限 compact 读取，仅当 `hand_context_dim=0` 时跳过上述五类无消费者字段，不生成占位 tensor；严格旧格式模式保持不变。

**结果**
单文件 Dataset smoke 通过，无消费者字段未进入 batch。2026-08-16 启动全新双卡 run `correspondence_ptv3_v2_old1797_compact_repro_ddp2_20260816_103130`，从旧 best 的 step 113500 成功恢复，world size 2、global batch 128，cosine-restart phase 为 65830/107000；已运行到 step 113920，GPU 1、2 均为满负载。

**决策**
保留 compact 兼容方案并继续训练；待完成验证后与旧 run 同阶段指标对比，再判断是否复现。

## 实验：完整 GRAB 与 ContactPose 联合单卡续训

**假设**
在旧 cosine-restart checkpoint 上加入完整 ContactPose 数据继续训练，可能改善跨数据集 correspondence 表现；现有 compact Stage 3 已足够执行不含 MANO 手扰动的旧训练协议。

**观察到的失败 / 现象**
物理 GPU 2 正在执行双卡复现实验，无法另启 2、3 双卡。ContactPose 全量 Stage 3 为 v2.0，不含 MANO 参数，不能使用 9 mm 手部 PCA 扰动。

**诊断**
使用空闲物理 GPU 3 单卡，将完整 GRAB compact 数据的 1963 个文件与 ContactPose 的 885 个文件通过本地符号链接目录联合加载。两者均包含旧模型实际消费的几何、法线、contact distance 和采样字段。

**改动**
沿用旧模型的 10° / 10 mm object perturb、contact auxiliary 和 cosine-restart，从旧 best step 113500 续训；关闭 MANO reconstruction/hand perturb。单卡 batch 64，低于旧双卡实验的 global batch 128。

**结果**
混合 Dataset smoke 通过，GRAB 与 ContactPose 样本可组成同一 batch。2026-08-16 启动新 run `correspondence_ptv3_v2_old1797_grab_contactpose_full_gpu3_20260816_110237`，checkpoint 与 scheduler phase 均成功恢复，已运行至 step 113840，GPU 3 满负载。

**决策**
继续训练，首轮 validation 后分别结合联合验证指标和独立 ContactPose 测试判断效果。

## 实验：ContactPose 全量独立评估

**假设**
在完整 GRAB 中加入 ContactPose 后，ContactPose-specific correspondence 和 10° / 10 mm object recovery 应同时改善。

**改动**
使用纯 GRAB 复现 checkpoint 与 GRAB+ContactPose checkpoint，在相同 ContactPose v2.0 全量数据上评估。数据包含 885 个序列、403452 帧；两边均关闭 MANO/PCA hand perturb 和 runtime object resampling，只保留 clean 与固定 10° / 10 mm object perturb。

**结果**

| 指标 | 纯 GRAB | GRAB+ContactPose | 联合相对变化 |
| --- | ---: | ---: | ---: |
| clean random QFL | 0.002283 | 0.000604 | -73.6% |
| perturbed random QFL | 0.002659 | 0.000706 | -73.5% |
| clean random MAE | 0.02321 | 0.00908 | -60.9% |
| perturbed random MAE | 0.02387 | 0.01027 | -57.0% |
| clean contact auxiliary QFL | 0.02936 | 0.00887 | -69.8% |
| perturbed contact auxiliary QFL | 0.03721 | 0.01040 | -72.1% |
| perturbed recovery Brier | 0.5448 | 0.8092 | +48.5% |
| perturbed recovery projection | 0.5588 | 0.7957 | +42.4% |
| fake-contact recovery Brier | 0.6791 | 0.8566 | +26.1% |
| missed-contact recovery Brier | 0.4412 | 0.7727 | +75.1% |

**诊断**
联合训练明显提升 ContactPose 上的边分类拟合，但 recovery 指标整体恶化，尤其 missed-contact recovery。当前最合理解释是输出概率分布/校准发生变化，而不是几何 correspondence 本身变差；该联合训练不能直接作为 recovery 模型替代纯 GRAB 旧模型。

**产物**
完整 JSON 位于 `output/research/contactpose_checkpoint_compare/pure_20260817_101033.json` 和 `mixed_20260817_101033.json`。
