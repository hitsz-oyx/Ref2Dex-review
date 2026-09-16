# CmResidual Task 入口

CmResidual 研究以几何重定向轨迹为 reference 的物理反馈 tracker，以及冻结该 tracker 后训练的独立
interaction residual。canonical owner 为 `src/task/CmResidual/`；IsaacGym vendor 路径只允许保留薄 adapter
和 Hydra 注册，不作为研究文档、数据合同或运行状态的事实源。

## 当前状态

- `modification_version`: `V1.11.2`
- 阶段：V1.11.2 已将 DexYCB actor 创建 DOF state 与 reference reset 对齐，并经用户批准在空闲 GPU5 完成
  4 env × 72 steps 修复后复跑；工程 Gate C（reset、finite、zero residual、正常退出）全部通过
- 行为结果：固定 `subject-10/20201022_110806/right` 单轨迹没有观察到成功抓取；最小平均 tip distance=
  `0.299875 m`，实际接触仅 12/72 steps、mean/max occupancy=`0.015972/0.20`，最大 lift 出现在无接触的 step 8，
  因而不能当作抓取。该单轨迹结果对“此序列成功抓取”为 `REFUTED`，对整个 DexYCB 泛化仍为 `INCONCLUSIVE`
- 资源状态：V1.10.1 fresh B0-Cm-path 已先在 GPU5 完成并通过；V1.11 未停止、重启或修改该 run，
  V1.11.2 仅在其结束后复用空闲 GPU5
- V1.10 状态：A-control 与 B-critic-Cm 均已从零完成 10 epochs / 20480 frames，checkpoint finite、
  frozen sigma、reload diff、matched actor initialization 和输入 provenance 工程 gate 全部通过；A-E10/B-E10
  尚未运行，因此 Cm utility 仍为 `INCONCLUSIVE`
- 计划状态：`final`（[V1.11 计划](plan/V1.11.md)；历史计划保留）
- 执行审批：用户批准隔离的 DexYCB 固定序列/物体、retarget、独立 config/runner 和单次 GPU6 gate；
  后续又批准 V1.11 reset 修复/GPU5 复跑及 V1.10 B0/A-T10/B-T10；V1.10 E10、多 DexYCB 轨迹与任何
  DexYCB 训练均未执行
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
- [活动记录](logs/activity_log.md)
- [实验记录](logs/experiment_log.md)

V1.1 的 `D0` 已确认旧 reference 不能直接作为 corrected research GT：tensor columns `54:57` 的右腕 exp-map 与权威 global quaternion
不等价，且旧 18-DOF retarget 允许 wrist 逐帧补偿手指误差。计划现已改为读取 columns `51:54` + `309:313`
（`xyzw`），用整段唯一 `X_HB` 映射 wrist、只约束优化 q6 并生成新版本产物。主区间固定为 indices
`[44,410]` 的 367 帧，完整 432 帧仅作诊断。V1.4 只为发布 checkpoint parity 读取 legacy source，不改变该
corrected-reference 结论；V1.5 只允许显式 `ppo_pilot` 使用 legacy source，且失败未产生科研结论。
