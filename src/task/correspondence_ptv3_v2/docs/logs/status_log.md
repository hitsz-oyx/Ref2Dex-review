# correspondence_ptv3_v2 当前状态

- scope: task:correspondence_ptv3_v2
- last_updated: 2026-08-24
- last_verified: 2026-08-24
- related: [架构](architecture_log.md) / [实验](experiment_log.md) / [修改](modification_log.md) / [当前指导](../指导/V1.md)

## 当前状态

- 当前阶段 / 指导: 三域等比例 mixed baseline 已从中断 checkpoint 续训；ARCTIC 外部评估仍遵循 V1。
- 当前进行中: GPU 1、2 双卡 DDP 续训 GRAB / ContactPose / OakInk 各 1/3 mixed baseline，总预算 154670 optimizer steps；2026-08-22 已从 step 22748 恢复并确认训练 step 继续写出。全量 ARCTIC MANO Stage 2/3 导出等待后续核验。HOCap subject_1 的 Stage 3 已完成首次纯 GRAB 外部测试。
- 最近可靠结论: 协议 E 已用统一 100% 的 10 mm hand perturb 评估 noPCA latest、H80 latest/best、H50 best 和历史 GRAB+ContactPose 两域 mixed latest。H80 best 的 hand-only 最终 QFL / recovery 最好；H50 best 的三条件等权 perturbed QFL 最低；历史两域 mixed 的 Balanced perturbed QFL 约 0.000973、recovery 0.1672，明显落后。旧协议 D 的 hand-only / hand+object 因 evaluator 继承训练门控概率而标记为 `INVALID_IMPLEMENTATION`，object-only 不受影响。三域 mixed 续训 W&B run ID 为 `d5yvor4z`。HOCap subject_1 已用纯 GRAB noPCA latest 跑完 28 文件 / 23,896 帧：clean random QFL 0.00016607，10°/10 mm object perturb 后 0.00020392，recovery Brier 0.5267。
- 阻塞 / 风险: 协议 E 仍仅覆盖 s01 min11，且 checkpoint 选择、global batch 与样本曝光量不统一。mixed 启动初期 NAS data wait 偏高；旧 run 日志 step 27980 后中断，但最近完整 checkpoint 仅到 step 22748，因此 5232 step 未保留为模型状态。全量 ARCTIC Stage 2/3 完成状态仍需核验。HOCap 当前只覆盖 subject_1，接触距离由几何派生且结果为 micro 聚合，不等同于正式多主体 benchmark。
- 下一步: 保持 mixed 配置不变继续训练并检查各域 validation；核验全量 ARCTIC Stage 2/3 的 schema、坐标系、MANO 字段、文件数和帧数。三域 mixed 获得可用 checkpoint 后，按 EXP-012 的相同 HOCap 协议做 matched 对照；HOCap 仍只作外部测试，不加入训练。
- 证据与相关文档: [EXP-012](experiment_log.md#exp-012--纯-grab-nopca-在-hocap-subject_1-的外部测试)；[EXP-011](experiment_log.md#exp-011--协议-e10-mm-mano-min11-三条件评估)；HOCap 结果 `output/research/hocap_subject1_grab_nopca_20260824/full_object_only.json`；[协议 E 结果](../../result/arctic_min11_mano_protocol_e_10mm_compare_20260823.md)；[训练日志](../../../../../outputs/correspondence_ptv3_v2/correspondence_ptv3_v2_mixed_grab_contactpose_oakink_equal_resume_gpu12_20260822_050212/train.log)。
