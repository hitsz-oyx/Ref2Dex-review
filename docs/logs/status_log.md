# Ref2Dex 当前状态

- scope: root
- last_updated: 2026-08-25
- last_verified: 2026-08-25
- related: [仓库记忆](repo_memory.md) / [修改记录](modification_log.md)

## 当前状态

- 当前阶段 / 指导: 已以 `oyx` 为主完成 `feature/hand-pca-perturbation` 的内容整合、递归 repo/machine memory 重构和全量回归验证。
- 当前进行中: correspondence_ptv3_v2 七域等比例 MANO/robot mixed 已在 GPU 2、3 以 W&B online 后台运行，随机初始化首个诊断到 step 20；旧三域进程仍保留在其原 GPU 上。ContactPose v2.1 全量 use 导出已完成；HRDexDB minimal allhands 的 2088 个 C2R-valid episode 已进入七域配置。
- 最近可靠结论: OakInk true-hand-root 全量重导出已生成 2596 个文件 / 252,172 帧，并通过严格 MANO loader、MANO 重建、接触距离和四视角一致性校验；当前在线 mixed 仍消费旧 OakInk 目录。ContactPose 原始包含 `mano_fits_15.json`，当前 v2.0 Stage3 只是未提取 MANO；单序列 v2.1 pilot 已完成重建核验。HOCap subject_1 Stage 3 已生成 28 个文件 / 23,896 帧并通过 loader 核验。HRDexDB minimal allhands Stage3 已完成 2088 个文件 / 1,265,425 帧：441 human、618 Inspire DFTP、576 Inspire F1、453 Allegro V5；四个 manifest 无错误，代表性 schema/shape/finite 检查通过。
- 阻塞 / 风险: 七域训练为随机初始化，不复用旧三域 checkpoint；机器人 10 mm 是按 FK 后 RMS 约束的经验目标，实际受限位和 link 几何影响。早期吞吐约 9.9–14.5 samples/s，data wait 较高，按当前 max_steps 粗估仍需数百小时；OakInk 仍有 572 个视角组因缺 object mesh 跳过；HOCap 仍只作外部测试。
- 下一步: 继续观察七域 DDP 首个 checkpoint、各域 validation 和稳定 step 吞吐。
- 证据与相关文档: [OakInk EXP-013](../../src/task/correspondence_ptv3_v2/docs/logs/experiment_log.md#exp-013--oakink-true-hand-root-全量重导出与-mano-重建校验)、全量 pytest `282 passed, 3 skipped`、[仓库修改记录](modification_log.md)、[Cm 状态](../../src/task/Cm/docs/logs/status_log.md)。
