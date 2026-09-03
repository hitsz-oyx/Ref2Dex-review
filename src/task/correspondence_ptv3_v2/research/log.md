# correspondence_ptv3_v2 研究日志

## 更正：OakInk 官方 annotation 含 MANO pose/shape，已重导出 true hand-root（2026-08-24）

此前本日志中“OakInk annotation 未提供 MANO global orientation/pose/betas”的判断来自未检查 `general_info`，现已确认错误。官方 `general_info.hand_anno` 包含 root/local quaternion、`hand_shape` 和 wrist-position `hand_tsl`；`cam_extr` 可把 world root pose 变换到 camera。当前导出器已使用这些字段同时消除手、物体的 root rotation，并保存 axis-angle45 MANO 参数。

新 NAS 产物为 2596 个 NPZ / 252,172 帧，训练端 MANO 重建、接触距离和四视角一致性均通过。旧的 wrist-centered camera-frame 结论只描述 2026-08-18 产物，不再是当前 OakInk 数据事实；正式证据和兼容边界见 `docs/logs/experiment_log.md` 的 EXP-013。

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

## 实验：ARCTIC 子集独立评估

**假设**
在未参与训练的 ARCTIC 数据上进行相同协议评估，可以检查纯 GRAB 与 GRAB+ContactPose 联合训练 checkpoint 的跨数据集表现，避免只依据 ContactPose 训练分布下的结果判断。

**改动**
ARCTIC 原始 Stage 3 位于 NAS 的 `processed_data_backup/arctic/arctic_initonly_4096_hand_root_v2`，全量 297 个文件、约 22 GB。为控制时间，在仓库 `tmp/arctic_eval_subset_20260817` 建立符号链接子集：5 个 subject（s01、s02、s04、s05、s06）各取 4 个序列，共 20 文件、12389 帧。该数据为 schema 2.0.0，缺少显式 dataset descriptor，因此 Dataset/evaluator 增加 `dataset_id=arctic` 显式识别。

两份 checkpoint 使用相同评估协议：读取存储的 clean hand points，关闭 MANO/PCA hand perturb 与 runtime object resampling，分别评估 clean 和 10° rotation / 10 mm translation object perturb。

**结果**

| 指标 | 纯 GRAB | GRAB+ContactPose | 联合相对变化 |
| --- | ---: | ---: | ---: |
| clean random QFL | 0.0001431 | 0.0001669 | +16.7% |
| perturbed random QFL | 0.0002136 | 0.0002670 | +25.0% |
| clean random MAE | 0.002711 | 0.004029 | +48.6% |
| perturbed random MAE | 0.003699 | 0.005797 | +56.7% |
| clean contact auxiliary QFL | 0.004611 | 0.004878 | +5.8% |
| perturbed contact auxiliary QFL | 0.006532 | 0.007280 | +11.5% |
| perturbed recovery Brier | 0.8383 | 0.8000 | -4.6% |
| perturbed recovery projection | 0.8120 | 0.7728 | -4.8% |
| fake-contact recovery Brier | 0.9066 | 0.8484 | -6.4% |
| missed-contact recovery Brier | 0.7687 | 0.7507 | -2.3% |

**诊断**
在 ARCTIC 子集上，联合训练的 correspondence 拟合指标变差，且按任务架构定义（`pseudo_recovery_*` 越大越好）看，object perturb recovery 也从纯 GRAB 的 0.8383 / 0.8120 降到联合训练的 0.8000 / 0.7728。此前这段“recovery 指标小幅改善”的表述方向有误，当前以实际指标和架构定义为准。该结果与 ContactPose 全量上的“拟合明显改善、recovery 恶化”方向一致，但幅度和数据分布不同；ARCTIC 结果目前只能作为方向性证据，不能外推到全量 ARCTIC。

**产物**
结果 JSON：`output/research/contactpose_checkpoint_compare/arctic_pure_20260817_132348.json`、`arctic_mixed_20260817_132348.json`。

**决策**
保留 ARCTIC 子集评估作为公平外部测试；若需要最终结论，再扩展到全量 297 文件或按 subject 做分层统计。

## 实验：OakInk Stage 3 转换与分布先导统计

**假设**
OakInk 的手/物几何可以转换为当前 correspondence v2 的 compact Stage 3；其尺度、接触比例和物体尺寸分布可能解释联合增加数据后指标变化。

**改动**
新增 `research/oakink_conversion/convert_oakink_pilot.py`，从 OakInk per-view/per-frame pickle 和物体网格生成 4096 物体点、1538 手点及 hand-to-object 最短距离。OakInk 的 `obj_transf` 经过核验是 object-to-camera，转换采用物体 canonical 坐标、手点乘逆位姿。由于原始数据只有 778 个手顶点，pilot 以确定性索引扩展到 1538；手法线暂用 wrist radial proxy，因此当前产物只用于 schema/分布验证，不能直接作为最终训练数据。新增 `summarize_stage3_distribution.py` 统一统计距离、接触比例和点云半径。

**结果**
NAS pilot 目录 `OakInk/processed/stage3_corr_oakink_pilot_20260817` 已生成 20 个视角组、1000 帧，Dataset smoke 通过。OakInk pilot 的 hand-to-object 距离中位数为 23.98 mm，10 mm 内手点比例 20.65%，物体点半径中位数 74.4 mm。

与现有小样本统计的方向性对比：ContactPose 10 mm 比例 35.2%、物体半径中位数 55.8 mm；GRAB smoke 10 mm 比例 19.9%、物体半径 142.1 mm；ARCTIC 子集 10 mm 比例 17.6%、物体半径 269.2 mm。样本和坐标/采样版本尚未完全统一，不能据此下最终结论，但数据分布确实存在明显尺度与接触密度差异，可能导致大数据联合训练的 loss 权重和输出校准发生变化。

**下一步**
全量 OakInk 转换已在 NAS tmux `oakink_convert_20260817` 中运行；完成后重新统计全量，并补充按 object category、sequence 和 contact-distance 分桶的比较。最终训练前需要把手法线改为 MANO 面法线或明确关闭法线通道。

## 实验：GRAB 5 mm 手扰动与物体扰动互斥训练

**假设**
9 mm 手扰动与 10°/10 mm 物体扰动同时出现可能形成过强的复合噪声；把手扰动降到 5 mm，并让每个样本只接受手或物体一种扰动，可能改善 correspondence 与恢复指标的折中。

**改动**
新增稳定的 per-frame 互斥门控。`hand_perturb_prob=0.8` 时约 80% 样本只应用手扰动，其余样本按 `obj_perturb_prob=1.0` 只应用物体扰动。GRAB 9 mm 几何标定乘 `5/9` 得到 5 mm 目标。配置 `full_grab_50ep_geometry_5mm_exclusive_ddp2.yaml` 使用纯 GRAB、双卡和 W&B online，Stage 3 从 NAS 读取。

**结果**
200 帧门控 smoke 得到 164 帧仅手扰动、36 帧仅物体扰动、0 帧同时扰动、0 帧无扰动。2-step MANO 训练 smoke 通过。现有 NAS v2.1 文件权限为 `000`，已从可读 GRAB 原始数据启动新的 v2.1 Stage 2/3 重建；tmux `grab5mm_exclusive_ddp2_20260817` 会在数据生成完成后自动用 GPU 1、2 启动正式训练。

**下一步**
训练开始后记录 NAS 读取下的 step/s、GPU utilization 和 DataLoader 等待表现，并与本地旧 run 对比。

## 实验：OakInk / ContactPose / GRAB 混训前 schema 与启动缓存检查

**假设**
三套数据可以通过关闭 MANO/PCA 手部扰动后直接混训；持久化文件索引可以减少 NAS 大目录的启动扫描时间。

**观察到的失败 / 现象**
当前 OakInk 全量 Stage 3 样本为 `schema_version=2.0.0`、`coordinate_frame=object`，不含 MANO 字段；当前 ContactPose `use_stage3_v2` 也为 `schema_version=2.0.0`、`hand_root`，不含 MANO 字段。可读 GRAB v2.1 重建目录存在文件权限 `000`，无法完成内容读取。

**改动**
`CorrStaticDatasetV2` 新增可选 `data.cache_index_path`。该 JSON 缓存每个 NPZ 的绝对路径、mtime、大小和帧数；缓存命中前会校验文件状态，失效后自动重建，且不写入 NAS 数据目录。

**结果**
缓存代码通过 `graspenv` Python 语法检查。OakInk/ContactPose 的实际 schema 检查确认：二者不能直接进入当前 `use_mano_reconstruction=true` 的混合训练；OakInk 还与默认 `hand_root` 坐标系不一致。

**决策**
保留启动索引缓存。正式混训前必须重新导出 OakInk 与 ContactPose 为统一的 Stage 3 v2.1 `hand_root`，并确认 GRAB 文件可读；“无 PCA 扰动”应配置为 `meta.apply_hand_perturb=false`，不是把 PCA 改成另一种 MANO 噪声。

## 实验：关闭手部扰动时放宽 MANO/schema 约束

**假设**
`schema_version` 只是样本标识；当训练不执行 MANO 手部重建或扰动时，缺少 MANO 字段不应阻止旧 Stage 3 几何混训。

**改动**
Dataset 仅在 `use_mano_reconstruction && apply_hand_perturb` 时检查 MANO 字段，并移除 `schema_version >= 2.1.0` 的硬判断。Runner 只有在 `apply_hand_perturb=true` 时才触发 MANO forward。没有 MANO 描述的数据集 ID 推断为 `unknown`，不再因缺少描述失败。启动索引 cache 改为增量合并，并使用进程独立临时文件。

**结果**
OakInk `schema 2.0.0/object` 与 ContactPose `schema 2.0.0/hand_root` 均可在关闭手部扰动时加载；打开手部扰动时仍会对缺少 MANO 字段的文件 fail-fast。代码和 Dataset smoke 通过。

**决策**
保留该行为。三数据集正式混训仍需先统一 `coordinate_frame`；OakInk 不能直接与 `hand_root` 数据混合，必须重新导出或转换坐标。若关闭 runtime object resampling，则不需要 MANO 模型参与输入重建。

## 实验：OakInk 手点改为 MANO face-center

**假设**
OakInk 的 778 个 `hand_v` 顶点可以使用 MANO 拓扑转换为与 GRAB/ContactPose 一致的 1538 个面中心和真实面法线。

**改动**
`research/oakink_conversion/convert_oakink_pilot.py` 新增 MANO topology 读取：对每帧 778 顶点计算 1538 个三角面中心和 cross-product normals；使用 `hand_j[0]` wrist 将手和物体变换到 wrist-centered `hand_root`。由于 OakInk annotation 未提供 MANO global orientation/pose/betas，输出明确记录为 camera-rotation hand-root 近似，不能用于运行时 MANO 扰动。

**结果**
一组一帧转换 smoke 通过：`hand_points=(1,1538,3)`、`hand_normals=(1,1538,3)`，法线范数在 1.0 附近，所有坐标和距离均为有限值；Dataset 以 `coordinate_frame=hand_root` 读取通过。

**决策**
保留转换逻辑。需要重新运行全量 OakInk 导出到 NAS；在没有 OakInk MANO global orientation 的前提下，该版本适用于无手部扰动训练，不宣称与 GRAB 的 MANO 参数路径等价。

## 实验：OakInk 全量 MANO face-center / hand-root 导出

**假设**
统一为真实 MANO 1538 个三角面中心、面法线和 `hand_root` 后，OakInk 可以与 GRAB/ContactPose 在关闭手部扰动的配置下共同读取。

**改动**
从断点继续导出全部 3168 个 OakInk 视角组；缺少物体 OBJ 的组记录并跳过，不阻塞其余数据。输出目录为 `OakInk/processed/stage3_corr_oakink_mano_face_handroot_20260818`。

**结果**
成功导出 1700 组、158364 帧、2596 个 NPZ；572 个视角因缺失物体网格跳过。随机样本均为 `coordinate_frame=hand_root`，手点和法线形状为 `(T,1538,3)`，法线范数约为 1，坐标和距离均有限。Dataset smoke（`use_mano_reconstruction=true`、`apply_hand_perturb=false`、`runtime_resample_object=false`）通过，缓存索引可正常生成。

**决策**
保留该全量导出，作为无 PCA/无手部扰动联合训练的 OakInk 输入。OakInk 手根旋转来自 camera rotation 近似，且没有 MANO pose/betas；若未来重新开启 MANO 手扰动，仍需带完整 MANO 参数的数据版本。
