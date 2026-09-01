# CmDecoder 当前状态

## 2026-09-01 已停止 Cm，CmDecoder V1.1 计划待确认

- 当前 Cm 新训练 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222` 已按用户要求安全停止；其 `best.pt` 保留作为下一版 CmDecoder 的候选 frozen Cm checkpoint。
- 已新增草案计划：[`docs/plan/V1.1.md`](../plan/V1.1.md)。计划默认使用 Inspire-F1 object-disjoint v4 geometry、`object_pose_t` task cache、偶数 stride `{2,...,20}` 和 `CmPointFlowModel`；旧 hand-root/v2 task cache 与旧 token sidecar 禁止复用。
- 在计划定稿前不修改 CmDecoder 代码、不重建 cache、不启动 Decoder 长时训练；旧 cache、checkpoint 和输出均未覆盖。

## 2026-09-01 CmDecoder V1.1 已启动

- V1.1 计划已获用户确认并定稿；已新增 object-pose cache view、运行时偶数 stride Dataset 和当前 Cm 配置。
- 新训练：`outputs/cmdecoder/cm_decoder_20260901_151052`，GPU0/1/2、global batch=48、`CmPointFlowModel`、在线 frozen Cm token、初始 Cm 为 `outputs/cm/cm_grab_inspire_f1_hand_flow_cm64_additive_20260831_192222/checkpoints/best.pt`。
- 启动 smoke 已通过：step 100 hand-flow EPE=`8.136 mm`、zero-flow=`10.734 mm`，三卡显存/利用率正常；当前未见 OOM/NaN/NCCL 错误。
- 由于在线 token 与 10 个动态 stride 的输入语义，本 run 不复用旧 C=256 或 hand-root token sidecar。

## 当前操作 — V1.2.12：撤出 Task-local Component/data/registry（2026-09-01）

- 已删除 `src/task/CmDecoder/components/`、`data/`、`registry/` 以及根级 `components/ref2dex/cmdecoder_pointflow` 兼容入口。
- 配置不再声明 Component 清单；数据和 cache 继续直接使用根级 `data/`、`dataset/` 与 `outputs/`。
- 真实数据、cache、checkpoint、output 和运行中的 decoder 进程未移动、删除或停止。
- 计划：[`docs/plan/V1.md`](../plan/V1.md)（final）；历史 Component 修改记录保留，不作为当前入口。

- scope: task:CmDecoder
- last_updated: 2026-09-01
- last_verified: 2026-09-01
- related: [架构](architecture_log.md), [实验](experiment_log.md), [接手记忆](repo_memory.md)

## 当前状态

- 当前阶段 / 指导: wrist-aware baseline 保持相对 wrist SE(3)+6维手指 q 输出；已在修正腕部语义的3 Hz v2 cache 上完成纯固定对应点 loss 三组对照。逐手点 Cm flow → 联合 wrist+q fitting 路线也已实现。
- 历史 EXP-022 已停止：输出 `outputs/cmdecoder/cm_decoder_20260829_225518`；停止前约 step `86500/143110` / epoch `19`，无 OOM/NaN/NCCL 错误。EXP-023 rollout 已通过 Viser 在 `http://localhost:8096` 提供播放。
- 历史短暂启动的 decoder run `outputs/cmdecoder/cm_decoder_20260831_165400` 已停止；仅推进到 step `300` / epoch `1`，未生成 checkpoint。停止原因是发现其 v2 task cache 坐标与当前 Cm checkpoint 不一致，不能作为有效实验结果。
- 最近可靠结论: EXP-022 validation hand-flow EPE 从 epoch 1 的 `2.291 mm` 降至 epoch 15 的当前最佳 `1.425 mm`（step `71550`），相对 zero-flow=`12.118 mm` 改善约 `88.2%`。同 best 的 Inspire action-conditioned EXP-023 在固定 train episode 上 point EPE=`7.325/13.450 mm`，EXP-025 在 held-out test episode `bamboo_basket/5` 上为 `6.896/11.415 mm`（mean/final），两者均未超过 `20 mm`；test 结果支持短程状态反馈具有一定泛化，但仍有误差累积。本次 EXP-024 改为 GRAB hand-flow→当前 Cm tokens→Inspire F1，无 Inspire 侧 Cm 输入；32 帧 robot centroid-object 距离由 `27.9 mm` 漂至 `471.5 mm`，机器人质心累计位移 `297.1 mm`，说明跨手型无配对重定向仍明显发散。
- 阻塞 / 风险: v1 仅保留历史复现，不可用于 wrist-aware 点监督；旧版 Cm、EXP-018 Inspire checkpoint 与本次混合 Cm 的效果不能混作同一实验结论。当前三卡 decoder 与 Cm 不再并行，之前的 GPU1 慢卡竞争已解除；新的 global batch48 相比旧 global batch16 增加每次 optimizer step 的样本量，学习率仍保持 `3e-4`，需单独解释收敛速度与指标可比性。GRAB→Inspire F1 单序列有效窗口重定向仍出现约 `324 mm` object-relative wrist 漂移，当前无约束逐帧 fitting 不可作为可用重定向方案。
- 下一步: 先从当前 v4 geometry cache 重建与 `object_pose_t` 一致的 Inspire-F1 decoder task cache，再重新启动逐点 hand-flow decoder；重建前不复用旧 v2 的 wrist-frame task arrays 或 C=256 token sidecar。
- 证据与相关文档: [接手记忆](repo_memory.md), [实验记录](experiment_log.md)
## 2026-08-30 cache 语义更新

- 用户确认取消 5cm object candidate 查询；新 builder 保留兼容文件名但写入全 object surface pool 的全真 mask。
- 新 Inspire-F1 cache 正在会话中重建：`data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_20260830`。

## 2026-08-30 重新计算 Inspire-F1 5cm mask

- 用户要求恢复帧级 5cm candidate 过滤，但保留新的完整 object surface pool 与运行时 512 点随机采样。
- 对 `data/processed_data/cm_decoder/hrdexdb_inspire_f1_surface512_object_pose_fast_20260830/v4` 的 576/576 episode 已完成 GPU 计算；每个 episode 写入独立的 `geometry/obj_candidate_mask_5cm_recomputed.npy`，未覆盖训练当前读取的全真 `obj_candidate_mask_5cm.npy`。
- 全量统计：362,027 帧、mask 有效点总数 349,129,312，占 object-pool 点数 `23.5443%`。按 manifest：train 455 episode / 285,956 transitions 中 160,179 个有至少一个 5cm candidate（56.015%）；val 为 58.529%，test 为 59.248%。
- 当前 Cm 混合训练仍在 GPU 0/1/2 运行，尚未切换到 sidecar mask，避免在运行中改变 dataset 行空间。
