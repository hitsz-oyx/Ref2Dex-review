I recommend **training Cm during RL, but on a slower, separate learning loop from the residual policy**. I would not let PPO gradients update Cm directly at first.

The reason is that joint end-to-end learning creates a moving-target problem:

$$
C_m^{(k)}(s,a)
\rightarrow \pi_{\rm res}^{(k)}
$$

and after a Cm update,

$$
C_m^{(k+1)}(s,a)\neq C_m^{(k)}(s,a).
$$

Then the residual actor has to relearn what its own input representation means while simultaneously learning control. More seriously, PPO reward could push Cm toward task-specific features that help one Inspire task but destroy the embodiment-independent interaction semantics we actually care about.

## I would use two Cm networks

Maintain

$$
C_m^{online}
$$

and

$$
C_m^{target}.
$$

The residual policy always sees the slowly changing target model:

$$
\delta a_t
=
\pi_{\rm res}
\left(
s_t,\,
a_t^{base},\,
C_{m,t}^{target},\,
\hat e_t^{target}
\right).
$$

During one PPO rollout/update block, \(C_m^{target}\) is completely frozen.

Meanwhile every simulator transition gives us exactly the data Cm currently lacks:

$$
(s_t,a_t,s_{t+1})
$$

and therefore the actual hand flow and actual object effect. Put those transitions in a replay buffer and train

$$
C_m^{online}
$$

with **supervised dynamics losses**, not PPO reward:

$$
L_{Cm}
=
L_{\rm object\ effect}
+
\lambda_1L_{\rm effect/noeffect}
+
\lambda_2L_{\rm uncertainty}
+
\lambda_3L_{\rm representation\ anchor}.
$$

The replay should mix the original demonstration data with RL-generated data. This is crucial: otherwise Cm could catastrophically forget the MANO/Inspire interaction structure and specialize to whatever states the current policy happens to visit.

Periodically update the target model slowly:

$$
\theta_{target}
\leftarrow
(1-\tau)\theta_{target}
+
\tau\theta_{online}.
$$

So the actual loop becomes

$$
\boxed{
\text{RL produces hard states}
\rightarrow
\text{Cm learns those physics}
\rightarrow
\text{slow Cm target update}
\rightarrow
\text{residual policy receives better interaction features}.
}
$$

This is the arrangement I would choose.

### I would also change one thing from our earlier “Cm in critic” idea

If we continue using PPO, putting Cm **only into the critic is too weak** for what you want.

PPO normally uses a value function

$$
V(s),
$$

not an action-conditioned \(Q(s,a)\) that explicitly ranks candidate residuals. The critic helps estimate advantages, but it does not directly invert Cm into an action.

If Cm is supposed to alter the strategy, eventually it should enter the **actor**:

$$
\pi_{\rm res}
(
s,\,
a_{\rm base},\,
C_m^{ref},\,
C_m^{base},\,
\Delta e
)
\rightarrow
\delta a.
$$

Here \(C_m^{ref}\) represents the desired interaction from the demonstration, while \(C_m^{base}\) / \(\hat e^{base}\) describes what the base policy's candidate action is predicted to do.

Then the residual has a concrete job:

$$
\boxed{\text{correct the discrepancy between intended and current interaction}.}
$$

The value network can receive the same features, but it shouldn't be Cm's only route into control.

---

# Can we directly use the DExplore checkpoint as our base?

**Yes as a frozen base controller, but no as a literal drop-in checkpoint for the current `CmResidual` PPO network.**

DExplore now officially releases an Inspire state-based teacher checkpoint, `inspire.pth`. It is an 8.2M-parameter PPO actor-critic with a 1442-dimensional teacher observation and an **18-dimensional action**: 6 floating-wrist DOFs plus 12 Inspire finger DOFs. The actions are mapped into PD targets by DExplore's Inspire-specific action conversion. ([GitHub][1])

Your current `CmResidual`, by contrast, exposes a **12-dimensional residual action**. Six dimensions modify the six independent coupled finger coordinates, three modify wrist translation and three modify wrist rotation. Its observation is built from current 18-DOF position/velocity, the six-dimensional base finger target, wrist/object relative transforms and fingertip offsets. ([GitHub][2])

So

$$
\texttt{DExplore inspire.pth}
\not\equiv
\texttt{CmResidual PPO checkpoint}.
$$

You cannot simply call `load_state_dict()` on the existing residual policy.

What I would do is make DExplore a **separate frozen `BasePolicy`**:

$$
o_t^{DExplore}
\rightarrow
\pi_{DExplore}^{frozen}
\rightarrow
a_t^{18D}
\rightarrow
\text{DExplore PD mapping}
\rightarrow
a_t^{base}.
$$

Then our residual branch sits on top:

$$
a_t
=
a_t^{base}
+
M(\delta a_t^{12D}),
$$

where \(M\) expands our six independent finger corrections plus wrist correction into the actual 18-DOF PD-target representation.

That preserves DExplore exactly as the competent nominal policy while keeping our residual policy small.

---

## There is a compatibility issue we should take seriously

DExplore's checkpoint depends on much more than having an Inspire hand.

Its teacher expects a **1442-D proprioceptive + privileged reference observation**, including target hand pose, next-step object pose and contact targets. ([GitHub][1]) It was trained using DExplore's own modified Inspire asset: their repository says the Inspire URDF is a SolidWorks-based rebuild, with a 6-DOF floating wrist and their own collision meshes. ([GitHub][3])

Your current environment is already deliberately matching parts of DExplore—e.g. the current `cm_residual.py` explicitly says it matches DExplore's collision filters and uses the same 18-DOF position-control structure. ([GitHub][2]) That makes reuse quite plausible.

But before trusting the checkpoint, I would verify exact equality of:

$$
\text{URDF/mesh},
\quad
\text{DOF ordering},
\quad
\text{joint limits},
\quad
\text{PD gains},
$$

$$
\text{physics timestep/control frequency},
\quad
\text{collision filters},
\quad
\text{reference preprocessing},
\quad
\text{observation normalization}.
$$

If those differ, the checkpoint should be used as a **teacher for distillation/fine-tuning**, not directly as the base.

DExplore provides the exact evaluation command for `inspire.pth`, so I would first run the released checkpoint untouched in its native DExplore environment and reproduce its behavior. ([GitHub][4]) Then run the same checkpoint through our wrapper with *zero residual*. If those two behaviors diverge materially, fix environment compatibility before adding Cm.

Also note that the released checkpoint is under NVIDIA's non-commercial license. ([GitHub][1])

---

# I think this simplifies our project substantially

Instead of spending time retraining a generic tracker that DExplore has already trained well, we can begin from:

$$
\boxed{
\pi_{\rm DExplore}^{frozen}
}
$$

as the Inspire base.

Then our research question becomes much sharper:

$$
\boxed{
\text{Can interaction knowledge from Cm improve an already strong DExplore controller?}
}
$$

That is substantially harder—but also much more meaningful.

The architecture I would test is

$$
a_t^0
=
\pi_{\rm DExplore}(s_t,r_t)
$$

then obtain the base action's proposed hand-point flow through FK,

$$
a_t^0
\rightarrow\Delta H_t^0,
$$

and evaluate it with the slowly updated Cm target:

$$
(C_m^0,\hat e_t^0)
=
F_{Cm}^{target}(s_t,\Delta H_t^0).
$$

The demonstration provides the intended interaction/effect

$$
(C_m^\star,e_t^\star).
$$

Then

$$
\boxed{
\delta a_t
=
\pi_{\rm res}
(
s_t,
a_t^0,
C_m^\star,
C_m^0,
e_t^\star-\hat e_t^0
)
}
$$

and

$$
a_t=a_t^0+\delta a_t.
$$

The real simulator determines whether that correction was good.

Those transitions then improve \(C_m^{online}\).

---

## The training order I would use

1. **DExplore checkpoint frozen, Cm frozen, residual disabled.** Verify the base policy in our environment.

2. **DExplore frozen, Cm frozen, residual PPO enabled.** Establish whether ordinary residual RL can improve the base.

3. **DExplore frozen, Cm-target frozen inside each PPO block, Cm-online trained from replay.** Slowly synchronize online → target.

4. **Give Cm features to the residual actor**, not merely the value network, and compare against a residual actor receiving simple contact/geometric features.

5. Only after this is stable would I consider allowing a very small RL auxiliary gradient into the Cm encoder itself.

I would **not** begin with

$$
\nabla_{\theta_{Cm}}J_{\rm PPO}.
$$

Let Cm first learn from the simulator's actual dynamics. That preserves its interpretation as a world/inter-action model and keeps the control problem stationary enough for the residual policy to learn.

And using the released DExplore checkpoint is attractive precisely because it removes one major uncertainty: **we no longer need Cm or residual RL to learn basic dexterous tracking from scratch.** The burden on Cm becomes much cleaner—demonstrate useful interaction knowledge on top of an already competent controller.

[1]: https://github.com/NVlabs/dexplore/blob/main/MODEL_CARD.md?utm_source=chatgpt.com "dexplore/MODEL_CARD.md at main · NVlabs/dexplore · GitHub"
[2]: https://github.com/hitsz-oyx/Ref2Dex-review/blob/oyx/third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py "Ref2Dex-review/third_party/IsaacGymEnvs/isaacgymenvs/tasks/cm_residual.py at oyx · hitsz-oyx/Ref2Dex-review · GitHub"
[3]: https://github.com/NVlabs/dexplore/blob/main/NOTICE?utm_source=chatgpt.com "dexplore/NOTICE at main · NVlabs/dexplore · GitHub"
[4]: https://github.com/NVlabs/dexplore?utm_source=chatgpt.com "GitHub - NVlabs/dexplore · GitHub"

## Ref2Dex V1.1 locked amendment (2026-09-15)

用户已确认以 DExplore Inspire teacher `inspire.pth` 为冻结基础策略，观测同步为 DExplore 的 1442 维，动作
同步为原生 18 维（6 wrist + 12 finger）。CmResidual 已迁移为本目录 Python package；残差动作也是 18 维，
按 `clip(a_dexplore + delta, -1, 1)` 合成后复用 DExplore 的 PD/mimic 映射。Cm 采用 online/target 双模型，
PPO block 内 target 冻结并通过 EMA 更新。vendor `logs*` 忽略已取消，使本架构记录可审计；checkpoint 文件本身
仍由外部 DExplore 工作树提供，不进入 Ref2Dex Git。
