## 2026-09-15 09:40:00 +0800 — DExplore 1442 维观测与 Cm PPO 链路修复

- activity_id: `ACT-20260915-094000-CMRESIDUAL-FIX`
- timestamp: `2026-09-15 09:40:00 +0800`
- modification_version: `V1.1.3`
- operation_category: `code`、`documentation`
- task_mode: `change`
- change_level: `L2`（观测 schema、策略网络和训练注册）
- approval: `user-approved`（用户明确要求修复问题并开始训练）
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `bbb1ce28089fc202921d2660ddd45409f487a6cd`
- worktree_dirty: `true`（保留根目录及 CmDecoderv2 既有差异）
- scope: `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/`、`third_party/IsaacGymEnvs/isaacgymenvs/learning/cm_*.py`、`third_party/IsaacGymEnvs/isaacgymenvs/cfg/train/CmResidualPPO.yaml`、`third_party/IsaacGymEnvs/isaacgymenvs/train.py`
- run_id: `smoke_fixed_20260915d`
- run_status: `COMPLETED`
- conclusion: `SUPPORTED`（工程 smoke；不代表科研效果成立）

**修改**

- 观测改为显式构造两个 DExplore 721-D offset，输出严格为 1442-D；移除旧 71-D 向量加零尾逻辑。
- 新增 `cm_continuous`/`cm_actor_critic` 注册；Cm online 特征进入 actor/critic，target 以 EMA 更新并冻结，PPO 梯度回传 online 分支。
- 关闭 rl_games 自动 torch.compile，规避当前 PyTorch 2.4.1 环境的编译兼容错误。

**验证**

- `python3 -m py_compile ...`：通过。
- `PYTHONPATH=. python3 -m pytest -q src/task/CmResidual/tests/test_reference_contract.py`：`2 passed`。
- CPU 单环境 `max_iterations=1`：完成 epoch 1，生成 `runs/CmResidual_15-09-38-37/nn/last_CmResidual_ep_1_rew__2.47_.pth`，无 traceback。

**保护边界与回滚**

- 未修改 DExplore checkout、`inspire.pth`、原始数据、旧运行或既有用户 dirty diff；新训练使用独立输出目录。回滚入口为本次代码提交及本条 activity。

# CmResidual 活动记录

## 2026-09-15 09:18:52 +0800 — 三卡训练终态复核及实现无效更正

- activity_id: `ACT-20260915-091852-CMRESIDUAL-TERMINAL-AUDIT`
- timestamp: `2026-09-15 09:18:52 +0800`
- modification_version: `V1.1.2`（本次终态审计；被审计运行原标记为 `V1.1.1`）
- operation_category: `diagnostic`、`operation`、`documentation`
- task_mode: `run-only/operation`（终态查询与记录同步）
- change_level: `L0`（更正状态及记录实现缺陷，不修改训练实现、数据或指标定义）
- approval: `user-approved`（用户查询训练状态；沿用已有运行与记录授权）
- skills_used: `research-experiment-workflow`、`research-change-control`
- branch: `oyx`
- base_commit: `bbb1ce28089fc202921d2660ddd45409f487a6cd`
- worktree_dirty: `true`（保留既有四处根/CmDecoderv2 文档差异）
- scope: `src/task/CmResidual/docs/logs/activity_log.md`、`outputs/CmResidual/cmresidual_dexplore*/run_manifest.json` 及各原 manifest 的一次性备份；没有启动训练、修改代码或 checkpoint。
- run_status: `COMPLETED`（3 组最终尝试）；`FAILED`（6 组之前的尝试）
- conclusion: `INVALID_IMPLEMENTATION`（相对用户要求的完整 DExplore + Cm 残差链路）

**原因**

- 先前将“可完成 PPO iteration”当成可以启动三卡的依据不成立。实际输入只是旧 71D 状态加 1371 个零，没有 DExplore 的两个 reference offset/layout；Cm 只被实例化，没有参与实际 PPO 的特征输入、replay 监督、梯度或 EMA 更新。
- GPU1/GPU3 最终尝试共用 `runs/CmResidual_15-01-48-27/`，最佳 checkpoint、config 与 TensorBoard 不能视为按 seed 隔离。最初三个尝试也共用 `runs/CmResidual_15-01-39-36/`。本次保留所有文件，只按独立 stdout 路径定位唯一的终态 checkpoint。
- 三组最终尝试均出现 `MAX EPOCHS NUM!`，终态 checkpoint 为 epoch 1000；对应进程已退出。旧 manifest 的 `RUNNING` 是未更新状态，现已更正。
- CPU/GPU 仿真和 64/256 env 数量不同，回报不可用作受控三 seed 比较；没有独立评估。本批结果不能证明 Cm 有效或无效。

**验证**

- `ps -p 2510695,2535229,2535230 -o pid=,stat=,etime=,args=`：三个 PID 均不存在；全进程扫描没有 `train.py task=CmResidual`。
- 逐一重载 9 份 manifest，读取 `train.log` 的完成标记、traceback、最后 epoch/frame 和 checkpoint 文件名；使用 CPU `torch.load` 读取 3 个终态 checkpoint 的 epoch/frame/last_mean_rewards/model keys。
- 三个 model state_dict 均没有 Cm 参数；`rg` 检查 `CmResidualActor`、`begin_ppo_block`、`supervised_loss` 仅定义/导出，无训练调用；配置选择普通 `actor_critic`。
- checkpoint 的 `last_mean_rewards` 是训练期历史最高回报，终态 checkpoint 文件名中的回报是最后一期滑动平均；均不是任务成功率或独立评估。stdout 的 frames 在迭代开始打印，最后累计步数以 checkpoint 的 frame 为准。
- 原始训练未生成 `metrics.jsonl`；本次不将共享 TensorBoard 数据伪造为独立逐 seed 曲线。

**文件与终态**

- run_id: `cmresidual_dexplore_gpu0_s101`；run_status: `COMPLETED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `1000`；last_step: `8192000`；best_metric: `4.765340328216553`；退出依据：达到 max_epochs=1000；原启动器未记录操作系统退出码，按完成标记、checkpoint 内容及进程退出确认终态。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu0_s101](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu0_s101/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu0_s101/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/train.log)。
  最近 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/last_CmResidual_ep_1000_rew__-121.24_.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/last_CmResidual_ep_1000_rew__-121.24_.pth)；最终训练回报：`-121.24`。
  最佳 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/CmResidual.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-39-36/nn/CmResidual.pth)。
  GPU0 的最佳文件存在；目录也曾被同秒失败的 GPU1/3 初始化，不能把共享 config.yaml 当作本组独占配置。
- run_id: `cmresidual_dexplore_gpu1_s102`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu1_s102_cpu_sim`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：CPU 仿真 reset_idx 使用了 GPU 索引。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu1_s102_cpu_sim_retry2`；run_status: `COMPLETED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `1000`；last_step: `2048000`；best_metric: `2.997872829437256`；退出依据：达到 max_epochs=1000；原启动器未记录操作系统退出码，按完成标记、checkpoint 内容及进程退出确认终态。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/train.log)。
  最近 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-838.6_.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-838.6_.pth)；最终训练回报：`-838.6`。
  GPU1/3 的 CmResidual.pth、config.yaml、TensorBoard 路径相同；最佳文件归属不能仅由文件名确定。保留按本组 train.log 唯一文件名定位的终态 checkpoint。
- run_id: `cmresidual_dexplore_gpu1_s102_retry64`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_retry64/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu3_s103`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu3_s103_cpu_sim`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：CPU 仿真 reset_idx 使用了 GPU 索引。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。
- run_id: `cmresidual_dexplore_gpu3_s103_cpu_sim_retry2`；run_status: `COMPLETED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `1000`；last_step: `2048000`；best_metric: `5.319734573364258`；退出依据：达到 max_epochs=1000；原启动器未记录操作系统退出码，按完成标记、checkpoint 内容及进程退出确认终态。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/train.log)。
  最近 checkpoint：[third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-70.02_.pth](../../../../../third_party/IsaacGymEnvs/runs/CmResidual_15-01-48-27/nn/last_CmResidual_ep_1000_rew__-70.02_.pth)；最终训练回报：`-70.02`。
  GPU1/3 的 CmResidual.pth、config.yaml、TensorBoard 路径相同；最佳文件归属不能仅由文件名确定。保留按本组 train.log 唯一文件名定位的终态 checkpoint。
- run_id: `cmresidual_dexplore_gpu3_s103_retry64`；run_status: `FAILED`；conclusion: `INVALID_IMPLEMENTATION`。
  last_epoch: `0`；last_step: `0`；best_metric: `N/A`；退出依据：GPU PhysX illegal memory access；未确认根因，不视为显存不足或物理 GPU 故障。
  输出：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64)；manifest：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/run_manifest.json](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/run_manifest.json)；日志：[outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_retry64/train.log)。
  最佳/最近 checkpoint：未生成；last_epoch/last_step=0 表示没有完成的训练迭代记录。

**保护边界与回滚**

- 代码、训练配置、checkpoint、原始日志、外部 DExplore、用户四处既有 dirty diff 均未修改；未创建或恢复运行。回滚本次记录可恢复每组 `run_manifest.before_terminal_audit.json` 并删除本条活动，不删除训练产物。

**规范反馈**

- 已有版本指针和 Task README 仍为 `V1.0.1`，与历史运行 `V1.1.1` 不一致；V1.1 plan 还配对 V1.0 指导。当前根指针属于既有 dirty diff，本次只登记冲突，没有静默改写用户指导或治理文件。运行终态查询不因该冲突被阻塞；后续实现应同步修正受影响导航和计划配对。无需新增规则，现有隔离/manifest 规则执行不到位。

## 2026-09-15 01:49:51 +0800 — 三卡训练资源调整后保持运行

- activity_id: `ACT-20260915-014951-CMRESIDUAL-TRAIN-3GPU-RETRY`
- timestamp: `2026-09-15 01:49:51 +0800`
- modification_version: `V1.1.1`
- operation_category: `operation`、`experiment`
- change_level: `L3`
- approval: `user-approved`（沿用三卡并行训练授权）
- branch: `oyx`
- base_commit: `03ea264`
- scope: three-run DExplore residual PPO operation; manifests/logs under `outputs/CmResidual/`; no source/cache/checkpoint mutation.
- run_id: `cmresidual_dexplore_gpu0_s101`、`cmresidual_dexplore_gpu1_s102_cpu_sim_retry2`、`cmresidual_dexplore_gpu3_s103_cpu_sim_retry2`
- run_status: `RUNNING`
- conclusion: `INCONCLUSIVE`

**状态**

- GPU0 使用 GPU PhysX，已运行至约 epoch 110/1000。
- GPU1/GPU3 的 GPU PhysX 在初始化阶段触发 Isaac Gym allocator error 700；未继续使用故障模式，改为 CPU PhysX + 对应 GPU 上的 PPO 网络，两个 retry2 进程已分别运行至 epoch 26/30。
- 失败尝试保留在 `outputs/CmResidual/cmresidual_dexplore_gpu{1,3}_s{102,103}*` 的 manifest/log 中，未覆盖重试目录。
- 当前三组 PID 与完整命令记录在各自 `run_manifest.json`；终态前不得把运行标为完成。

**scope**

- 三组运行 manifest 和日志分别位于 [GPU0 run](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/run_manifest.json)、[GPU1 retry run](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/run_manifest.json) 和 [GPU3 retry run](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/run_manifest.json)。

**验证**

- 启动后检查确认三个 PID 仍在运行；GPU0/GPU1/GPU3 分别有训练进程和对应显存占用。GPU1/GPU3 的 CPU PhysX retry 已通过 30 个以上 epoch，无 CUDA allocator error。
- 各目录的 [GPU0 train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu0_s101/train.log)、[GPU1 train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu1_s102_cpu_sim_retry2/train.log)、[GPU3 train.log](../../../../../outputs/CmResidual/cmresidual_dexplore_gpu3_s103_cpu_sim_retry2/train.log) 持续写入；终态字段尚未产生。

**原因**

- GPU1/GPU3 的 GPU PhysX 在当前节点触发固定的 allocator error 700，继续重试同一模式没有证据会恢复；保留 GPU0 GPU PhysX，同时将另外两组切到 CPU PhysX、GPU PPO，以维持三卡并行且避免反复触发非法内存访问。

**验证与边界**

- GPU0、CPU smoke 和三组当前运行均使用 `inspire.pth` SHA256 `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- 本条记录的是运行状态，不把 epoch/reward 作为科研结论；正式结论须等待独立评估和完整指标。

## 2026-09-15 01:37:55 +0800 — 三卡并行训练启动

- activity_id: `ACT-20260915-013755-CMRESIDUAL-TRAIN-3GPU`
- timestamp: `2026-09-15 01:37:55 +0800`
- modification_version: `V1.1.1`
- operation_category: `operation`、`experiment`
- change_level: `L3`（三卡并行 PPO 长任务）
- approval: `user-approved`
- approval_basis: 用户明确要求“直接完成整条链路，然后用三张空闲的卡并行训练”。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `a22223e7c35570e79a53e165ff6ad5e1dc5f17d2`
- worktree_dirty: `true`（根目录既有文档差异未纳入运行）
- scope: vendor CmResidual DExplore package；外部 `inspire.pth` 只读；三次独立 seed、独立输出目录。
- run_id: `cmresidual_dexplore_gpu0_s101`、`cmresidual_dexplore_gpu1_s102`、`cmresidual_dexplore_gpu3_s103`
- run_status: `STARTED`
- conclusion: `INCONCLUSIVE`（运行尚未结束；训练结果不能由启动状态推断）

**运行合同**

- command template: `PYTHONPATH=/home2/wyy/isaac-gym/isaacgym/python:/home2/wyy/oyx_ws/Ref2Dex:/home2/wyy/oyx_ws/Ref2Dex/third_party/IsaacGymEnvs python3 isaacgymenvs/train.py task=CmResidual headless=True force_render=False pipeline=gpu sim_device=cuda:<gpu> rl_device=cuda:<gpu> graphics_device_id=<gpu> task.env.numEnvs=256 max_iterations=1000 seed=<seed> task.basePolicy.checkpoint=/home2/wyy/oyx_ws/dexplore/checkpoint/inspire.pth train.params.config.horizon_length=32 train.params.config.minibatch_size=2048`
- GPU/seed mapping: `cuda:0/101`、`cuda:1/102`、`cuda:3/103`；三次运行不共享 checkpoint 或 optimizer。
- base checkpoint SHA256: `8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- outputs: `outputs/CmResidual/cmresidual_dexplore_gpu{0,1,3}_s{101,102,103}/`；日志与 manifest 在各自目录，均为 `PENDING` 直至进程生成。

**启动证据**

- 三张卡启动前显存/利用率检查：GPU0 `2553 MiB/0%`、GPU1 `776 MiB/0%`、GPU3 `1997 MiB/0%`；GPU2/4/5/6 正在使用，GPU7 保留作故障回退。
- 单卡 GPU0 1-iteration smoke 已完成（8 envs）；CPU 1-iteration smoke 也完成。正式三卡运行不继承 smoke checkpoint。
- 终态必须补写 `last_step`/`last_epoch`、best/latest checkpoint、`train.log`、`metrics.jsonl`（若生成）和失败/停止原因。

**保护边界与回滚**

- 不修改外部 DExplore checkout、`inspire.pth`、旧 reference/cache、用户既有根/CmDecoderv2 dirty diff。
- 停止入口：按 run_id 单独终止对应 PID；删除各自 ignored output 目录即可回滚运行产物。

## 2026-09-15 01:08:43 +0800 — V1.1 DExplore 18D package contract

- activity_id: `ACT-20260915-010843-CMRESIDUAL-V11-PACKAGE`
- timestamp: `2026-09-15 01:08:43 +0800`
- modification_version: `V1.1.0`
- operation_category: `governance`、`architecture`、`code`、`documentation`
- change_level: `L3`（vendor Task package、公共 import、checkpoint/obs/action contract）
- approval: `user-approved`
- approval_basis: 用户明确确认取消 vendor `logs*` 忽略、使用冻结 DExplore `inspire.pth`、同步 1442D obs 和原生 18D action，并按既定 Cm online/target 方案执行。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `9654d657702a8a8da5f6c02c2ddd52dd1b2e7d54`
- worktree_dirty: `true`（保留既有根/CmDecoderv2 用户差异，未覆盖、未暂存）
- scope: `third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/`、vendor configs、vendor `.gitignore`、V1.1 plan；旧数据、checkpoint、output 和 `src/base/` 未修改。
- conclusion: `SUPPORTED`（工程接口 smoke；不代表 PPO 效果或科研结论）

**文件与变更**

- [V1.1 计划](../plan/V1.1.md) — 锁定 DExplore teacher、1442D obs、18D residual/action、Cm online/target、回滚和验证边界。
- [DExplore base policy](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/base_policy.py) — 加载并冻结 `inspire.pth`，校验网络形状与 running stats。
- [Cm adapter](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/cm_adapter.py) — target 冻结、PPO block EMA 和 replay feature 接口。
- [Residual actor](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/residual_policy.py) — 将冻结 teacher 与 Cm feature 拼接，输出有界 18D residual action。
- [Task package](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/task.py) — 18D residual 与 DExplore PD/mimic 合成；`__init__.py` 保持 `CmResidual` import 入口。
- [架构记录](../../../../../third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/docs/logs/architecture_log.md) — 追加 V1.1 用户确认快照；vendor `logs*` 忽略已删除。

**验证**

- `python3 -m py_compile third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual/*.py`：通过。
- 直接加载 `inspire.pth` 并前向零输入：输出 shape `(2,18)`，值域在 `[-1,1]`；checkpoint SHA256=`8f6823db752288f1bddd6d042981d33514e29dac5a68e58726e76215fea6d553`。
- `CmOnlineTarget(1442)` smoke：online/target feature shape 均为 `(2,128)`，通过。
- 未启动 Isaac Gym、PPO 或长时运行；因此本条不产生 `run_id`，也不宣称训练收益。

**原因**

- 旧入口是 12D CmDecoder residual，无法直接承载 DExplore teacher 的 18D wrist+finger action，也无法
  复用其 1442D policy observation。将同名脚本转换为 package 后，base checkpoint、Cm 生命周期和
  DExplore action mapping 有独立可测试边界，训练入口继续使用原 `tasks` 注册表。

**保护边界与回滚**

- 未提交外部 DExplore checkout、`inspire.pth`、旧 reference/cache/output/checkpoint、共享 `src/base/` 或用户已有 dirty diff。
- 回滚入口：回退本条 V1.1 文件/配置和 package 迁移提交，恢复旧 `tasks/cm_residual.py` 与 `.gitignore` 的 `logs*` 规则。

## 2026-09-14 18:41:47 +0800 — 建立 canonical Task 并完成首轮 reference 连续性诊断

- activity_id: `ACT-20260914-184147-CMRESIDUAL-GOVERNANCE-D0`
- timestamp: `2026-09-14 18:41:47 +0800`
- modification_version: `V1.0`
- operation_category: `governance`、`documentation`、`diagnostic`
- change_level: `L3`（新增 Task owner、根 Task 索引和版本指针），内含 `L0` 只读数据诊断
- approval: `user-approved`（Task owner 与计划方向）；`execution-pending`（代码、数据、仿真和训练）
- approval_basis: 用户明确回复“是的”，确认 V1.0 采用 `src/task/CmResidual` owner、单轨迹范围、累计
  base residual、联合 RSI、gate/pilot 优先且不直接长训
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留用户已有根/CmDecoderv2 activity diff，不覆盖、不暂存）
- scope: `docs/current_versions.yaml`、`docs/项目总览.md`、`src/task/CmResidual/`；不包含其他已有工作树差异
- conclusion: `INCONCLUSIVE`（Task 边界已明确；现有几何 reference 尚不足以安全冻结 PPO 数据与控制合同）

**范围与文件**

- 将用户原始 [指导/V1.0.md](../指导/V1.0.md) 原文迁入 canonical Task；迁移前后 SHA256 均为
  `bd2d555ae89f1b5a4d30393cb1bfaa776e7f83a7817cb36cc4e8fa30bb8df6a4`，未修改其内容。
- 将 [plan/V1.0.md](../plan/V1.0.md) 草案迁入 canonical Task，记录用户已批准方向和新增 `D0` 闸门；
  `status` 仍为 `draft`、`execution_approval` 仍为 `pending`。
- 新增 [Task 入口](../README.md)，并更新
  [根 Task 索引](../../../../../docs/项目总览.md) 与
  [当前版本指针](../../../../../docs/current_versions.yaml)。
- 未修改 `cm_residual.py`、Hydra 配置、共享 `src/base`、旧 cache/data/checkpoint/output 或用户指导；未启动
  Isaac Gym、PPO、数据生成或长时任务。

**D0 新证据**

- `s1/airplane_lift` sidecar 为 432 帧、30 Hz，`source_frame_id=0,4,...,1724`，映射连续且 object/q/wrist
  长度一致。独立 q6 单元素步差 p50/p95/p99/max 为 `0.000283/0.018418/0.071911/0.184409 rad`。
- wrist world 单步 translation p50/p95/p99/max 为 `5.90/53.68/85.08/100.14 mm`，rotation 为
  `0.109/6.520/11.542/16.946 deg`；object translation p95/max 为 `15.25/20.50 mm`。
- reference index `44..410` 为连续 `<=2 cm` hand-object 几何区间。排除远离物体的首尾后，腕姿 p95 明显
  降低，但 index `43/46/50/53/149/260/286/339/342` 等仍有局部旋转尖峰；index `342` 的 robot wrist
  为 `27.38 mm/10.55 deg`，同源原始 GRAB 右腕仅 `5.46 mm/1.81 deg`。
- 原始 GRAB 手指在 index `342` 同时存在约 `31.79 deg` 的单关节快速运动。producer 代码会把 6 个 virtual
  wrist 与 12 个 finger/mimic DOF 一起交给逐帧 warm-start retarget 优化，因此手指快速变化可能由 wrist
  补偿放大；后续 coupled fit 明确保持 native wrist `q[:,0:6]` 不变，不能修复该来源问题。
- 按 source URDF 的 joint velocity limit 和 30 Hz 差分，现有 native q 有 44 个 transition 至少一关节超限，
  其中 17 个位于 reference transition `43..409`。当前 legacy 环境又把全部 DOF velocity 覆盖为 `7.0`，
  因此不能把 legacy smoke 的可运行性当作 reference 物理可跟踪性。
- `inspire_geometric` 原始导出根没有 converter run manifest；筛选 manifest 和后续 coupled-fit manifest 能锁定
  筛选/拟合，但不能锁定首次 retarget 的精确命令、迭代参数和 producer commit。当前 provenance 不满足拟定的
  frozen reference 合同。

**原因**

- 开头/结尾的大位移在原始 GRAB 右手轨迹中同步出现，主要是实际接近/撤离，不应一概误报为 stage 拼接。
- interaction 区间内的部分腕姿放大和 joint-limit 超限仍未解决；静默裁剪、平滑或重采样会改变 GT、时间合同和
  checkpoint 解释，不能在 loader 中自行修复。
- 因此 `D0` 当前为 `INCONCLUSIVE`：可以继续只读定位和提出数据方案，但不得把计划标为 final，不得进入
  reference builder、physics playback 或 PPO。

**验证**

- 只读加载 sidecar/geometry/raw GRAB、producer 代码与 URDF，核对 shape、frame mapping、SE(3) 步差、接触距离、
  原始人体右腕/手指变化、native q 差分和 URDF velocity limit；未写入任何诊断 data/output。
- 使用 `sha256sum` 确认指导迁移前后内容一致；交接前使用 `audit_diff.py --check-links` 检查本条链接，并运行
  `git diff --check`。

**回滚**

- 回滚入口：只回退 [根 Task 索引](../../../../../docs/项目总览.md)、
  [当前版本指针](../../../../../docs/current_versions.yaml) 和本 Task 新增文档；若恢复迁移前布局，只移动指导/plan
  回原 vendor-local 文档目录。不得 reset 用户已有根/CmDecoderv2 activity diff。

## 2026-09-14 19:05:21 +0800 — 定位 wrist quaternion 合同并批准新 reference 方向

- activity_id: `ACT-20260914-190521-CMRESIDUAL-D0-WRIST-CONTRACT`
- timestamp: `2026-09-14 19:05:21 +0800`
- modification_version: `V1.0`
- operation_category: `diagnostic`、`architecture`、`documentation`
- change_level: `L2`（坐标、GT/reference、数据 schema 和验收合同）
- approval: `user-approved`（固定 wrist SE(3) mapping、finger-only constrained retarget、新版本产物且旧 sidecar
  不覆盖）；`execution-pending`（主帧区间、定量合同、代码、数据和物理 gate）
- approval_basis: 用户对“显式固定 GRAB 右腕→robot wrist SE(3) 映射，只优化 6 个独立手指 DOF 并施加
  速度/加速度约束，同时保留旧 sidecar 不覆盖”的推荐方案明确回复“是的”
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留用户已有根/CmDecoderv2 activity diff，不覆盖、不暂存）
- scope: `docs/current_versions.yaml`、`docs/项目总览.md`、`src/task/CmResidual/docs/指导/V1.0.md`、
  `src/task/CmResidual/docs/plan/V1.0.md`、`src/task/CmResidual/docs/README.md`、
  `src/task/CmResidual/docs/logs/activity_log.md`；本次新增实质编辑仅为 plan、Task 入口和本条 activity，前三项
  是同一未提交 Task diff 中已由上一条 activity 记录的版本指针、根导航和原文迁移
- conclusion: `REFUTED`（旧 reference 不能直接作为 training GT）；新 fixed-`X_HB` replacement 在 D1 生成和
  验证前仍为 `INCONCLUSIVE`

**新增坐标证据**

- DExplore tensor columns `51:57` 保存右腕 translation/exp-map，global joint quaternion block 位于
  `245:373`，其中右腕 quaternion 为 columns `309:313`、顺序 `xyzw`。对 432 帧比较两种旋转，SO(3)
  geodesic 差异 min/median/p95/max=`24.88/49.99/113.81/174.09 deg`，全部帧不等价。
- producer 的 `quat_to_exp_map` 对负 `w` 用 `abs(w)` 计算角度，却没有同步翻转 vector part；本序列右腕
  quaternion 的 `w` 全为负。新 schema 因此显式拒绝 columns `54:57`，只接受单位化的 columns `309:313`。
- 用同一 tensor 的 wrist/MCP landmarks 构造 human palm frame，得到其相对 global wrist quaternion 的旋转在
  432 帧内为数值常量，独立支持 quaternion 字段和 `xyzw` 解释。
- 采用正确 quaternion 后，原 indices `[44,410]` 的连续 `<=2 cm` interaction 区间共有 367 帧，对应 raw
  `source_frame_id=[176,1640]`；human wrist linear/angular speed 最大值=`1.104 m/s` / `2.248 rad/s`，低于
  source URDF virtual wrist limit `2.0 m/s` / `3.14 rad/s`。完整 432 帧最大值仍达到 `2.080 m/s` /
  `4.873 rad/s`，不推荐直接进入 V1.0 训练。
- 新合同将 robot wrist 固定为 `T_WB(t)=T_WH(t) X_HB`，整段只允许一个 `X_HB ∈ SE(3)`；逐帧变量仅 q6，
  mimic 固定，q6 写出前必须通过 position、`1.0 rad/s` velocity 和计划预注册的 `20.0 rad/s^2` acceleration
  硬门。不得从旧 robot wrist 拟合逐帧 offset。

**原因**

- 新坐标证据推翻了旧 exp-map 的 GT 身份；继续沿用旧 wrist 或用其拟合逐帧 offset 会把 converter 错误和
  finger compensation 固化进 base checkpoint。固定 `X_HB` 能表达一次性的 human→robot morphology 映射，
  同时阻断逐帧 wrist 泄漏；q6 约束则让新 reference 在进入 Isaac Gym 前先满足明确的运动学合同。

**计划与保护边界**

- 更新 [V1.0 计划草案](../plan/V1.0.md)，加入权威 tensor columns、fixed-`X_HB` 求解、q6 硬门、367 帧推荐
  区间、精确 action/reward/RSI/PPO 工程 gate 和稳定 CLI 合同；状态仍为 `draft`、执行审批仍为 `pending`。
- 旧 sidecar 与同源 [geometry manifest](../../../../../data/processed_data/coupled_geometric_cache_v1_20260912/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json)
  只读保留；未裁剪、平滑、重采样或重导出，未修改外部 DExplore checkout、vendor adapter、Hydra 配置、
  `src/base/`、checkpoint、output 或现有运行进程。
- 一次 CPU 内存 feasibility probe 未捕获到可复核 stdout，进程随后已不存在且未写文件；该尝试不登记为正式
  run、不作为阈值或方向证据，也未重复启动。

**验证**

- 只读加载 raw tensor、同源 geometry、producer 和 URDF，检查 quaternion/exp-map geodesic、palm-frame
  一致性、frame mapping 以及 30 Hz wrist velocity；所有数值仅用于 D0 诊断。
- 交接前运行 `audit_diff.py --check-links` 和 `git diff --check`。

**回滚**

- 文档改动的回滚入口为本条 activity、[Task 入口](../README.md) 和 [V1.0 计划草案](../plan/V1.0.md)；
  同一未提交 Task diff 还包括 [指导原文](../指导/V1.0.md)、
  [根 Task 索引](../../../../../docs/项目总览.md) 与 [当前版本指针](../../../../../docs/current_versions.yaml)。
  不得 reset 用户已有根/CmDecoderv2 activity diff。

## 2026-09-14 20:41:02 +0800 — V1.0 定稿后发现 pinky mimic 合同冲突并暂停 D1

- activity_id: `ACT-20260914-204102-CMRESIDUAL-D1-MIMIC-BLOCKER`
- timestamp: `2026-09-14 20:41:02 +0800`
- modification_version: `V1.0.1`
- operation_category: `governance`、`architecture`、`diagnostic`、`documentation`
- change_level: `L2`（mimic、native q、FK 与未来 checkpoint 解释）
- approval: `user-approved`（V1.0 final、indices `[44,410]`、D1 builder/artifact/offline validation）；
  `amendment-pending`（pinky mimic `1.05` 或 `1.18`）
- approval_basis: 用户对“367 帧主 reference、计划定稿、只执行 D1 且不运行 Isaac Gym/PPO”明确回复“同意”；
  mimic 冲突是在定稿后的实现前检查中新增发现，不能沿用原批准静默选择
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留用户已有根/CmDecoderv2 activity diff，不覆盖、不暂存）
- scope: `docs/current_versions.yaml`、`docs/项目总览.md`、`src/task/CmResidual/docs/指导/V1.0.md`、
  `src/task/CmResidual/docs/plan/V1.0.md`、`src/task/CmResidual/docs/README.md`、
  `src/task/CmResidual/docs/logs/activity_log.md`；未修改代码、配置、旧数据、cache、vendor 或外部 DExplore
- run_id: `N/A`（仅内存 feasibility probe，不是正式数据运行）
- run_status: `N/A`
- conclusion: `INCONCLUSIVE`（human target 与 fixed-`X_HB`/q6 可行性已有正证据，但 mimic 合同未决，D1
  尚不得生成）

**文件**

- [V1.0 最终计划](../plan/V1.0.md) — 记录用户的 final/D1-only 批准，并在发现 mimic 冲突后将执行状态设为
  `paused-pending-mimic-contract-amendment`。
- [Task 入口](../README.md) — 更新 `V1.0.1`、D1-only 边界和当前 blocker。
- [当前版本指针](../../../../../docs/current_versions.yaml) 与
  [根 Task 索引](../../../../../docs/项目总览.md) — 更新到 D1 `V1.0.1` 和 final 计划状态。
- [指导原文](../指导/V1.0.md) — 同一未提交 Task diff 的既有迁移文件，本轮未改内容。

**原因**

- legacy coupled/CmResidual contract 把 native pinky `q11` 展开为 `1.05*q10`；当前锁定 URDF 的 `<mimic>`
  明确为 `1.18*pinky_proximal`，且仓库与 DExplore 的三份同名 URDF SHA256 相同。两者不可同时作为
  “固定 mimic”真值；静默选择会改变 reference geometry、速度门和 future checkpoint 身份。
- `research-change-control` 要求 final plan 无法按原语义执行时暂停并重新协商，因此没有开始 builder 文件编辑
  或正式数据写出。

**验证**

- 在 GPU6 只用内存重建同源 432 帧 SMPL-X 五指尖；对齐后旧 DExplore robot→human tip EPE
  mean/p95/max=`11.585/22.807/51.109 mm`，支持 target 重建链可复核。
- 对批准的 367 帧做内存 fixed-`X_HB` + q6 feasibility probe：相对 neutral-q6 baseline 的 tip EPE mean
  从 `18.421 mm` 降至 `9.156 mm`（改善 `50.3%`），p95/max=`21.618/35.099 mm`；投影后 q6
  velocity/acceleration max=`1.000 rad/s` / `20.000 rad/s^2`。这些数值仅证明优化可行，不是正式 artifact、
  P0 smoke 或科研结论，未写输出文件。
- 核对锁定 URDF：index/middle/ring=`1.05`、pinky=`1.18`、thumb=`0.60/0.8`；核对 legacy 代码与 coupled
  manifest：四个非 thumb source 均硬编码 `1.05`。运行 `git diff --check` 通过。

**回滚**

- 当前仅有文档/版本指针变化；回滚入口为 [V1.0 最终计划](../plan/V1.0.md)、[Task 入口](../README.md)、
  [根 Task 索引](../../../../../docs/项目总览.md)、[当前版本指针](../../../../../docs/current_versions.yaml) 和
  本条 activity。未产生需要删除的数据或运行目录，不得 reset 用户已有根/CmDecoderv2 activity diff。

## 2026-09-14 21:53:28 +0800 — D1 fixed-wrist constrained-q6 reference artifact

- activity_id: `ACT-20260914-215328-CMRESIDUAL-D1-BUILD`
- timestamp: `2026-09-14 21:53:28 +0800`
- modification_version: `V1.0.1`
- operation_category: `code`、`data`、`operation`、`documentation`
- change_level: `L2`（reference、坐标、mimic、数据 schema 与离线验收）
- approval: `user-approved`
- approval_basis: 用户在恢复会话后要求继续；沿用 D1-only 批准并确认采用锁定 URDF pinky mimic `1.18`，不启动 Isaac Gym/PPO。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `ffdb9b30a92fda24894fba3964185d4e72fd6748`
- worktree_dirty: `true`（保留既有根/CmDecoderv2 文档差异；未覆盖或暂存无关改动）
- scope: `src/task/CmResidual/tools/data/build_reference.py`、`src/task/CmResidual/tests/test_reference_contract.py`、`src/task/CmResidual/docs/plan/V1.0.md`、`src/task/CmResidual/docs/README.md`；新数据产物位于 ignored `data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/`
- run_id: `s1_airplane_lift`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`（schema/SE(3)/mimic/q6 运动门和五指尖改善通过；源 cache 无 raw contact flag，接触保留门不可计算，故 artifact 暂不具备 training-eligible 状态）

**文件**

- [指导原文](../指导/V1.0.md) — 既有同一 Task dirty diff，本轮未改内容。
- [D1 builder](../../tools/data/build_reference.py) — 读取同源 quaternion/object pose，固定单一 `X_HB`，逐帧仅优化 q6，按 URDF mimic 展开并执行硬门；不修改旧 sidecar。
- [D1 contract tests](../../tests/test_reference_contract.py) — 验证 URDF-authoritative mimic 和 367 帧/source frame 合同。
- [D1 reference artifact](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/reference.npz) — 367 帧新 reference、wrist/object twist、link poses、mask。
- [D1 manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/manifest.json) — 输入 SHA、X_HB、mimic、优化器和 gate 状态。
- [run manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/run_manifest.json) — 运行命令、base commit 和终态。

**原因**

- 旧 sidecar 的 pinky `1.05` 与锁定 URDF 的 `1.18` 不兼容；新 artifact 明确记录 q13=`1.18*q12`，旧数据继续只作历史兼容。
- 固定 `X_HB` 后在批准的 `[44,410]` 区间生成 367 帧；五指尖 RMS `57.36 mm`，相对同一固定腕映射的 neutral baseline `88.88 mm` 改善 `35.46%`，超过计划的 `20%` 工程门。

**验证**

- `PYTHONPATH=. python3 src/task/CmResidual/tools/data/build_reference.py --output data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift`：完成，未覆盖既有输出。
- `PYTHONPATH=. python3 -m pytest -q src/task/CmResidual/tests/test_reference_contract.py`：`2 passed`。
- 独立 `numpy.load`/JSON 重载核对：所有字段 finite；367 帧，Task-local frame `0..366` 与 source frame `176..1640` 分离；X_HB 旋转行列式 `1.0`、正交误差 `<4e-16`、最后齐次行正确；mimic 最大误差 `<1.2e-7 rad`；独立 q6 速度最大 `1.000001 rad/s`（float32 保存舍入，容差 `1e-6`），加速度最大 `11.83 rad/s²`。
- `contact_flag=UNAVAILABLE`：输入 geometry manifest 未提供 raw contact flag，未用距离阈值伪造接触标签；因此 manifest/run manifest 均标记 `training_eligible=false` 与 `INCONCLUSIVE`。
- `git diff --check`：通过。

**保护边界与回滚**

- 未修改旧 coupled-geometric sidecar、CmDecoder/CmResidual legacy 配置、vendor adapter、`src/base/`、外部 DExplore checkout、checkpoint 或训练输出；未启动物理 gate/PPO。
- 回滚入口：删除新 ignored artifact 目录，并回退本条 builder/test、plan/README 和 activity 差异；旧输入与旧 reference 不受影响。


## 2026-09-14 22:02:00 +0800 — D1 contact flag source audit

- activity_id: `ACT-20260914-220200-CMRESIDUAL-CONTACT-AUDIT`
- timestamp: `2026-09-14 22:02:00 +0800`
- modification_version: `V1.0.1`
- operation_category: `diagnostic`、`documentation`
- change_level: `L0`
- approval: `auto`
- approval_basis: 用户要求继续；仅检查现有 source/parent cache 的 contact 字段，不修改数据或运行配置。
- skills_used: `research-change-control`、`research-experiment-workflow`
- branch: `oyx`
- base_commit: `6d54106`
- worktree_dirty: `true`
- scope: coupled geometric source、cm_object_v2 parent cache 和其 manifest；只读。
- run_id: `cmresidual_contact_flag_audit_20260914_220200`
- run_status: `COMPLETED`
- conclusion: `INCONCLUSIVE`

**文件**

- [D1 manifest](../../../../../data/processed_data/cm_residual/reference_tracking_v1/s1_airplane_lift/manifest.json) — 对照现有 artifact 的接触字段状态。
- [source geometry manifest](../../../../../data/processed_data/coupled_geometric_source_v1_20260912/sequences/train/inspire_rl/s1_airplane_lift/geometry/manifest.json) — 输入 schema 与字段清单。

**原因**

- D1 的接触保留门要求 raw contact flag；现有 source/parent cache 没有该标注，几何候选 mask 不能替代它，因此维持 `INCONCLUSIVE`。

**验证**

- source geometry manifest 只包含 pose、surface、frame mapping 和 q provenance，没有 raw contact/touch flag。
- parent cache 的 `right/candidate_active_5cm.npy` 语义为生成的 5 cm 几何候选 mask，不是传感器或标注 contact flag，不能代替接触真值。
- 未修改 cache、artifact、checkpoint、配置或运行进程。

**回滚**

- 本条仅为诊断记录；删除本条即可回滚。
