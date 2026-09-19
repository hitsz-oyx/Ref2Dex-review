# CmResidual Task 入口

CmResidual 研究以几何重定向轨迹为 reference 的物理反馈 tracker，以及冻结该 tracker 后训练的独立
interaction residual。canonical owner 为 `src/task/CmResidual/`；IsaacGym vendor 路径只允许保留薄 adapter
和 Hydra 注册，不作为研究文档、数据合同或运行状态的事实源。

## 当前状态

- `work_version`: `V1.20`
- V1.20c：已验证 raw GRAB 的 airplane/table `mesh.obj` 与既有可运行 DExplore mesh 逐字节一致；两项均为上游刻意忽略的运行时资产。V1.20a launcher 在执行前以固定哈希物化它们，既不提交也不软链接；计划见 [V1.20c](plan/V1.20c.md)。
- V1.20：按用户授权建立 `s1_airplane_lift` 的隔离 DExplore 单序列 reconstructed baseline。原生 120 Hz contact、598-D schema、上游 reward/observation/action/physics 与坐标合同保持不变；公开 producer 缺失部分仅由 Task-local adapter 组装。DExplore 固定为 [vendor snapshot](../../../../third_party/DExplore/UPSTREAM.md)，多卡路径为 [V1.20a](plan/V1.20a.md) 定义的无 Horovod PyTorch/NCCL facade；vendor 迁移见 [V1.20b](plan/V1.20b.md)。工程接线证据见 [Activities](activities/README.md)，真实两 rank PhysX smoke 尚未开始，科研结论仍为 `INCONCLUSIVE`。
- V1.18：主线改为 `reference-conditioned PPO + frozen Cmv2 one-step planner + planner-to-actor distillation`。actor/critic 均接收 retargeted `+1/+16` 的 1442-D DExplore-layout future reference；Cmv2 仅在 pre-action active state 上为 K=8 local candidates 生成 stop-gradient teacher，PPO 仍执行自身 sampled action。初始实现及工程 smoke 的范围见 [V1.18 最终计划](plan/V1.18.md)，终态以 [活动记录](logs/activity_log.md) 为准。
- V1.17：DExplore 原始 teacher 仅训练与 CmResidual 对齐的 `s1_airplane_lift` GRAB 序列；输入仍取 `inspire_rl_object_dexplore` 中该序列的 RL rollout 手/物体状态，新 policy 从零初始化。GPU5 单卡容量探测固定 horizon 64、minibatch 256、1 PPO iteration、seed42，逐级确定稳定 env 数；该基线不改 CmResidual/Cmv2 合同。计划见 [V1.17](plan/V1.17.md)，运行状态以 [活动记录](logs/activity_log.md) 为准。
- V1.16.5：用户批准以已验证的 physical GPU0/1/3、3 rank×128 env 启动正式 PPO 长训；不设人为 epoch budget（配置保留 100000000 的安全上限），由用户人工停止。算法、输入、buffer、奖励与 critic 合同不变；运行状态以 [活动记录](logs/activity_log.md) 为准。
- V1.16.1：仅修正 DDP 启动器为 active Python 的 `torch.distributed.run` 加真实 `.py` bootstrap，并在 Task 正常退出/Ctrl-C 清理时 flush 未满 CmBuffer shard；不改 actor/Cmv2/reward/critic 或 buffer 字段。3×128 capacity probe 将作为单 epoch 工程测量运行；其运行状态以 [活动记录](logs/activity_log.md) 为准。
- V1.16：V1.15 actor 表示不变，但 nominal-base Cmv2 改为 current GPU hand link pose 到 next GPU reference hand link pose 的纯 GPU sweep；transition-only v2 CmBuffer 取消 executed-action Cmv2、在 GPU staging 后批量转存。base 68-D 恢复独立 RunningMeanStd，token/effect 语义保持。GPU5/1 env/8 step/1 epoch smoke 的 checkpoint/optimizer/action finite、reload diff=`0`、saturation=`0`，v2 buffer 有 8 个无点云 finite sample。三卡 DDP runner 已实现但未运行：GPU2 当前占用约 23 GiB，工程接线为 `SUPPORTED`、规模训练与科研效果仍为 `INCONCLUSIVE`。
- V1.15：冻结 Cmv2 的零 residual nominal-base effect 仅进入 PPO actor：raw transport 为 `68+16×40+18=726-D`，网络内以 masked self-attention/effect-conditioned pooling 形成 `214-D` actor 表示；critic 严格只读 `68-D` base prefix。GPU5/1 env/8 step/1 epoch smoke 的 checkpoint、optimizer 与动作均 finite，重载 action diff=`0`、saturation=`0`，CmBuffer 写入 8 个无点云 shard；这只支持工程接线，attention utility、Cmv2 微调资格与科研效果仍为 `INCONCLUSIVE`。
- V1.14.5：保持 68-D PPO actor/critic 与 frozen Cmv2；仅为已执行 residual 做 action-conditioned effect inference，并把无点云、可离线重建的 pre/post state、prediction、reference/PhysX effect 写入 per-rank CmBuffer。GPU5/1 env smoke 生成 1 条完整 shard、无 target saturation；buffer 未参与 PPO reward、actor/critic 或 Cmv2 更新。
- V1.14.4：用户批准的局部可行动作映射已将 reference-boundary 的外向 residual authority 收缩到实际可行余量，并保留原 18-D action、global scale、reward 与 mimic 合同。GPU5 单环境 zero/nonzero smoke 的 target saturation 为 `0`；其后 64-step/64-env/seed42 的 2+8 epoch PPO 完成 `40960` env-steps，10 个 epoch saturation 均为 `0`、checkpoint finite 且重载差为 `0`。这是工程协议 `SUPPORTED`，单 seed/单 reference 的 tracking、residual utility 与抓取效果仍为 `INCONCLUSIVE`。
- V1.14.3：固定 `sigma=0.1` 的无更新回放显示 saturation 集中于 native finger DOF 8、14、15；它们在部分 reference phase 的名义 target margin 已为零。该诊断支持“局部 reference-boundary 问题”，不支持直接缩小全局 scale；未改代码或重训。
- V1.14.2：已实现 reference-indexed reset、object pose/transition tracking reward 与独立 64-step curriculum runner。GPU5/64 env/seed42 的 64-step smoke 完成 2 epochs/8192 env-steps 和 checkpoint reload，但 epoch-1 residual saturation=`5.38%` 超过 `5%` 阈值，未续跑 8 epochs 或进入 128/366；协议为 `INVALID_IMPLEMENTATION`，科研效果 `INCONCLUSIVE`。详见 [活动记录](logs/activity_log.md)、[实验记录](logs/experiment_log.md)。
- V1.14.1：按用户定稿的 [V1.14 计划](plan/V1.14.md) 扩展真实物体 zero-residual 评估的 object pose/transition 指标，并已启动 GPU5、1 env、366-step 物理基线；运行终态、tracking 曲线和是否进入 reward/window 实现以 [活动记录](logs/activity_log.md) 为准。Cmv2 仍未接入策略。
- V1.13.4：用户明确批准越过未通过的追踪门禁，在 GRAB retargeted base 上以 GPU5、64 env、seed42 做探索性 PPO。2+8 epochs 完成，共 20480 env-steps；checkpoint 有限且可重载。但每环境仅 320 步，短于 366-step episode，未有完整 episode，最优 reward 为 `-inf`；epoch-10 residual 目标饱和率 `10.76%` 超过预定 `10%` 停止阈值。运行 `FAILED`，探索协议 `INVALID_IMPLEMENTATION`，科研效果 `INCONCLUSIVE`；未继续加预算。详见 [V1.13 计划](plan/V1.13.md)、[活动记录](logs/activity_log.md)、[实验记录](logs/experiment_log.md)。
- V1.13.3：用户批准的 GPU5 腕部 PD 增益对照 `200/20`、`300/30`、`400/40` 均完成 366 步无接触零 residual 评估。最优 `400/40` 在 GRAB 接触标记帧的腕部/指尖 p95 为 `0.02653/0.03648 m`，未达到 `0.015/0.020 m` 的预设门禁；三组最佳腕部对齐均为滞后两帧。按停止条件未进行有物体 Gate 或 PPO，下一步需单独协商速度前馈等低层控制方案。详见 [V1.13 最终计划](plan/V1.13.md)、[实验记录](logs/experiment_log.md)。
- V1.13.2：同一 GRAB 无接触 366 步协议补记实际腕部位置，诊断显示实际位置更接近两帧前的参考（接触帧 p95 `0.00740 m`，同帧 p95 `0.02762 m`）；这是时序滞后证据，不是修改固定参考时钟的许可。已在 V1.13 plan 中写入低层增益对照草案，尚未批准执行；PPO 仍未开始。
- V1.13.1：按用户确认切换到 GRAB 现成重定向轨迹作为新路线的确定性 base；腕部从 `wrist_pose_world_ref` 按 URDF 逆解，手指从 `q_native_ref` 读取，旧 DExplore 路径保留用于复现。366 步 CPU/单环境无接触零 residual 运行接线通过，未见接触或饱和；但 GRAB 接触帧指尖追踪误差 p95 约 `0.0375 m`，超过 Cmv2 `0.02 m` 交互半径，低层追踪是否足够仍为 `INCONCLUSIVE`，未启动 PPO。详见 [V1.13 最终计划](plan/V1.13.md)、[活动记录](logs/activity_log.md)。
- V1.12.5：用户确认的 [指导](指导/V1.12.md) 与 [最终计划](plan/V1.12.md) 已在当前 `oyx` 工作树落地 action-conditioned 第一阶段接口。真实 Cmv2 V1.3 `latest.pt` 已严格加载，reference FK 静态接触 frame 可产生非空 token；但 GPU5 的 53-step base warmup 后，实际手—物体表面距离为 `0.260–3.431 m`，并非 reference frame 53 的接触状态。该漂移已存在于 V1.11.2 base-only 运行；K=8 平行 env 还会因 PhysX 分叉而不是同一 counterfactual state。Cmv2 不拼接到 PPO observation，PPO 与样本内训练均未运行；当前为 `INVALID_IMPLEMENTATION`（物理 gate），科研效果 `INCONCLUSIVE`。详见 [活动记录](logs/activity_log.md)。
- 阶段：V1.11.2 已将 DexYCB actor 创建 DOF state 与 reference reset 对齐，并经用户批准在空闲 GPU5 完成
  4 env × 72 steps 修复后复跑；工程 Gate C（reset、finite、zero residual、正常退出）全部通过
- 行为结果：固定 `subject-10/20201022_110806/right` 单轨迹没有观察到成功抓取；最小平均 tip distance=
  `0.299875 m`，实际接触仅 12/72 steps、mean/max occupancy=`0.015972/0.20`，最大 lift 出现在无接触的 step 8，
  因而不能当作抓取。该单轨迹结果对“此序列成功抓取”为 `REFUTED`，对整个 DexYCB 泛化仍为 `INCONCLUSIVE`
- 资源状态：V1.10.1 fresh B0-Cm-path 已先在 GPU5 完成并通过；V1.11 未停止、重启或修改该 run，
  V1.11.2 仅在其结束后复用空闲 GPU5
- V1.10 状态：A-control 与 B-critic-Cm 均已从零完成 10 epochs / 20480 frames；各自 checkpoint 的 finite、
  frozen sigma、reload diff、matched actor initialization 和输入 provenance 工程 gate 全部通过。随后按最终计划
  完成 deterministic A-E10/B-E10（GPU5、64 env、seed42、367 steps）；两侧相对 B0 的 lift/contact ratio 分别为
  `1.000942/1.395830` 与 `1.002380/0.874114`，均通过双 80% preservation gate。B 相对 A 的 contact occupancy
  低 `37.38%`，lift 高 `0.14%`，其他行为指标方向混合；单 seed、单场景 critic-only Cm utility 仍为
  `INCONCLUSIVE`。详见 [V1.10 E10 实验记录](logs/experiment_log.md)。
- 计划状态：`final`（[V1.11 计划](plan/V1.11.md)；历史计划保留）
- 执行审批：用户批准隔离的 DexYCB 固定序列/物体、retarget、独立 config/runner 和单次 GPU6 gate；
  后续又批准 V1.11 reset 修复/GPU5 复跑及 V1.10 B0/A-T10/B-T10/A-E10/B-E10。V1.10 T20、多 seed、
  多 DexYCB 轨迹与任何 DexYCB 训练均未执行
- V1.9 matched ablation：A/B 都构造相同 2005-D observation 并加载同一冻结 OI-Cm；actor 均只读取
  1442-D base prefix，A critic 读取 1442-D，B critic 读取完整 2005-D。合同测试支持 actor 初始化完全一致
  且 Cm 后缀不影响 actor；Gate II 进一步确认两侧均能完成 2 epochs 和 checkpoint 重载，但 critic 首层宽度
  造成构造后 PyTorch RNG 状态不同；V1.9.2 通过固定顺序同时构造 1442/2005-D critic 候选消除了该差异，
  post-build CPU RNG 与首轮 stochastic action 最大差均为 `0.0`，两侧 2-epoch smoke 和 checkpoint reload
  均通过。epoch-1 success fraction 仍为 `0.1875/0.15625`，说明分进程 GPU rollout 不能视为逐 trajectory
  bitwise 配对；后续必须保持同 seed、多 seed 与效果边界，不用 smoke loss/success 判断 Cm utility，当前仍为 `INCONCLUSIVE`
- V1.8 safe 变体：保留 18D action，actor/critic 均关闭 OI-Cm、使用 1442-D base observation，
  residual mean 精确零初始化，state-independent `sigma=0.1` 且在 stability gate 中冻结
- 默认兼容路径：canonical task config 继续启用 OI-Cm 与 2005-D observation；旧训练配置未显式
  `learn_sigma=false` 时仍保持 sigma 可学习，不改变 V1.7 checkpoint 解释
- 当前实现：32-D 冻结 OI-Cm、2005-D observation、真实输入校验、net contact force，以及
  `0.015 m / 0.20 rad / 0.08 rad` 有界 residual 保持不变；pilot 使用 canonical GPU PhysX subscene 配置
- 当前 gate：V1.4 zero-residual、V1.5 fixed nonzero residual 和 V1.5.4 PPO wiring 均为工程 `SUPPORTED`
- 当前诊断：`num_subscenes=1` 会在 GPU `prepare_sim` 触发 SIGSEGV；删除 override 后 1-env smoke 和 64 env × 2 updates 均正常
- 当前诊断：V1.7 随机 residual rollout 相对 V1.4 zero-residual gate 明显退化；epoch 165 的
  `lift_mean=-0.5458 m`、`success_fraction=0`、`residual_rms=0.8521`，但运行协议不同且只有单 seed，
  不能归因于 residual 方法或 OI-Cm
- 当前结论上限：zero-residual 的单场景抓取/抬升行为为工程 `SUPPORTED`；V1.7 未完成训练和独立评估，
  residual 增益、收敛、Cm 因果与普遍抓取效果仍为 `INCONCLUSIVE`
- V1.8 修订 B0 在 GPU5/64 env/seed42/367 steps 下完成，zero residual target parity 为工程 `SUPPORTED`；
  `mean_env_max_lift_m=0.085096 m`、mean contact occupancy `0.193631`，对应 80% 阈值为 `0.068077 m`
  和 `0.154905`。E10 lift/contact ratio=`1.003871/1.164776`，E20=`0.973333/1.683597`，均通过双 gate；
  因此单 seed、单场景、20-epoch residual preservation stability 为工程 `SUPPORTED`，抓取改善、统计显著性、
  泛化和 Cm 因果仍为 `INCONCLUSIVE`
- 已确认 V1.3 主因是把原 DExplore 的空切片 `body_pos[..., 6:]` 误写成 body 维 remainder，制造了最高
  `188.4956 m/s` 的 reference 伪速度；同时 V1.4 对齐了 ReLU/RMS、scene reset 与 simulator 参数
- corrected reference 已切换到 `reference_tracking_v2/s1_airplane_lift`；source raw contact 与固定 URDF 表面距离门
  通过，manifest 为 `training_eligible=true`。旧 v1 保留为历史诊断证据
- 当前只证明 reference/data contract 具备训练准入；正式 PPO 的收敛、Cm 增益和抓取科研效果仍需独立训练与评估

## 文档入口

- [V1.1 用户指导](指导/V1.1.md)
- [V1.1 最终执行计划](plan/V1.1.md)
- [V1.3 用户指导](指导/V1.3.md)
- [V1.3 最终执行计划](plan/V1.3.md)
- [V1.4 用户指导](指导/V1.4.md)
- [V1.4 最终执行计划](plan/V1.4.md)
- [V1.5 用户指导](指导/V1.5.md)
- [V1.5 最终执行计划](plan/V1.5.md)
- [V1.6 用户指导](指导/V1.6.md)
- [V1.6 最终执行计划](plan/V1.6.md)
- [V1.7 用户指导](指导/V1.7.md)
- [V1.7 最终执行计划](plan/V1.7.md)
- [V1.8 用户指导](指导/V1.8.md)
- [V1.8 最终执行计划](plan/V1.8.md)
- [V1.9 用户指导](指导/V1.9.md)
- [V1.9 最终执行计划](plan/V1.9.md)
- [V1.10 用户指导](指导/V1.10.md)
- [V1.10 最终执行计划](plan/V1.10.md)
- [V1.11 用户指导](指导/V1.11.md)
- [V1.11 最终执行计划](plan/V1.11.md)
- [V1.12 用户指导](指导/V1.12.md)
- [V1.12 最终执行计划](plan/V1.12.md)
- [V1.13 用户指导](指导/V1.13.md)
- [V1.13 最终执行计划](plan/V1.13.md)
- [V1.14 用户指导](指导/V1.14.md)
- [V1.14 最终执行计划](plan/V1.14.md)
- [V1.15 用户指导](指导/V1.15.md)
- [V1.15 最终执行计划](plan/V1.15.md)
- [V1.16 用户指导](指导/V1.16.md)
- [V1.16 最终执行计划](plan/V1.16.md)
- [V1.18 用户指导](指导/V1.18.md)
- [V1.18 最终执行计划](plan/V1.18.md)
- [活动记录](logs/activity_log.md)
- [实验记录](logs/experiment_log.md)

V1.1 的 `D0` 已确认旧 reference 不能直接作为 corrected research GT：tensor columns `54:57` 的右腕 exp-map 与权威 global quaternion
不等价，且旧 18-DOF retarget 允许 wrist 逐帧补偿手指误差。计划现已改为读取 columns `51:54` + `309:313`
（`xyzw`），用整段唯一 `X_HB` 映射 wrist、只约束优化 q6 并生成新版本产物。主区间固定为 indices
`[44,410]` 的 367 帧，完整 432 帧仅作诊断。V1.4 只为发布 checkpoint parity 读取 legacy source，不改变该
corrected-reference 结论；V1.5 只允许显式 `ppo_pilot` 使用 legacy source，且失败未产生科研结论。
