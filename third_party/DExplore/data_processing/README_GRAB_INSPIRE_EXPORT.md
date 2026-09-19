# GRAB -> Inspire export

This repository now contains the bridge used to export the raw GRAB clips
under `/home2/wyy/oyx_ws/Ref2Dex/data/raw_data/GRAB` into Dexplore's native
`interaction_hand_inspire.pt` format.

The intermediate geometric motions are in
`/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric`; the
corrected policy rollouts from `checkpoint/inspire.pth` are in
`/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_object_dexplore`
for the Dexplore-compatible 660-sequence set.

Each sequence directory contains one tensor with shape `(T, 598)` at
`interaction_hand_inspire.pt`.  The corrected RL exporter copies the human,
contact, and graph fields from the geometric tensor, replaces the 18-DOF
Inspire hand slice with the actual simulated DOF positions produced by the
deterministic checkpoint policy, and replaces object pose columns `198:205`
with the dynamic object's actual Isaac Gym position and rotation.

The 10-sequence corrected pilot was first checked at
`/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_object_pilot10`.
After visual approval, the complete corrected filtered export is at
`/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_object_dexplore`.
The older `inspire_rl` and `inspire_rl_dexplore` roots are retained for
backward comparison only; they predate object-state capture and still contain
reference object poses.

All 1335 geometric and RL files have been regenerated from the canonical
cache. The active directories now contain only the validated canonical export.

To reproduce the export from the repository root (with conda environment
`graspenv`):

```bash
PYTHONPATH=data_processing \
/home2/wyy/miniconda3/envs/graspenv/bin/python data_processing/adapt_interact_canonical.py ...

PYTHONPATH=data_processing \
/home2/wyy/miniconda3/envs/graspenv/bin/python data_processing/convert_grab.py ...

PYTHONPATH=data_processing \
/home2/wyy/miniconda3/envs/graspenv/bin/python data_processing/export_rl_batches.py ...
```

The exact source/output arguments used for the completed export are recorded
in the shell history/logs under the corresponding `processed_data` folders.

## Coordinate-frame warning

`convert_grab.py` expects the canonical InterAct motion representation. Do not
feed it the output of the older `prepare_grab.py` preprocessing path: that
cache has already applied the upright `90-degree X` transform, while the
converter applies the same transform again. The resulting tensor still has
the expected `(T, 598)` shape, but the hand and object can be separated by
30--40 cm. The canonical adapter is the safe input path:

```bash
PYTHONPATH=data_processing \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/adapt_interact_canonical.py \
    --interact-root /home2/wyy/oyx_ws/InterAct \
    --raw-root /home2/wyy/oyx_ws/Ref2Dex/data/raw_data/GRAB \
    --output-root /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/interact_grab_canonical
```

The full geometric and RL exports in the output directories were regenerated
from this canonical cache on 2026-09-05. The previous copies were
removed after validation.
The 42 successful RL batch logs are kept separately under
`logs_rl_canonical_full_20260905/` so they are not mistaken for motion
sequence directories by Dexplore.

## Dexplore-compatible filtered export

The complete 1335-sequence roots above intentionally preserve every GRAB
sequence.  Dexplore's normal loader additionally drops any sequence with a
left-hand contact label and excludes names containing `doorknob`.  A materialized
filtered view is available at:

```text
/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric_dexplore
/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_object_dexplore
```

These roots contain 660 paired sequences and can be passed directly as a
`motion_file` directory or to the Inspire trajectory viewer.  The filtering
manifest is stored as `manifest.json` in each root; the original full roots are
unchanged.

For example, browse only the Dexplore-compatible set with:

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/visualize_inspire_trajectory.py \
    --all \
    --source both \
    --geometric-root /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric_dexplore \
    --rl-root /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_object_dexplore
```

## InterAct-compatible cache

After cloning InterAct at `/home2/wyy/oyx_ws/InterAct`, its official GRAB
pipeline was run (with the unavailable SMPL-H/BodyPrior branch disabled for
GRAB only):

```bash
cd /home2/wyy/oyx_ws/InterAct
INTERACT_GRAB_ONLY=1 python process/process_grab.py
INTERACT_GRAB_ONLY=1 python process/canonicalize_human_multi_thread.py
cd simulation
INTERACT_GRAB_ONLY=1 python interact2mimic.py --dataset_name grab
```

All three stages are complete for all 1335 clips.  The processed and
canonical caches are at
`InterAct/data/grab/sequences` and `sequences_canonical`.  The official
InterMimic tensors are written to
`InterAct/simulation/intermimic/InterAct/grab` with shape `(T, 591)`; all
1335 tensors were checked for matching names, `float32` dtype, and finite
values.
`data_processing/adapt_interact_canonical.py` exposes the canonical cache as
Dexplore converter input under
`Ref2Dex/data/processed_data/interact_grab_canonical`.

## Offline visualization

The export can be inspected without opening an Isaac Gym window.  The
following command renders the canonical SMPL-X body and object to an H.264
preview, and writes a 6x3 plot comparing all 18 Inspire DOFs from geometric
retargeting with the `checkpoint/inspire.pth` rollout:

```bash
PYOPENGL_PLATFORM=egl PYTHONPATH=data_processing \
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/visualize_inspire_preview.py \
    --sequence s1_cup_lift \
    --max-frames 300
```

Outputs are written to
`/home2/wyy/oyx_ws/Ref2Dex/data/processed_data/visualization_preview/`:

* `<sequence>_human_object.mp4` — 640x480, 30 fps, canonical body/object;
* `<sequence>_qpos_compare.png` — all 18 native Inspire DOF trajectories.

For example, play the generated clip with `ffplay` or extract a still frame:

```bash
ffplay /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/visualization_preview/s1_cup_lift_human_object.mp4
ffmpeg -i /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/visualization_preview/s1_cup_lift_human_object.mp4 \
    -vf 'select=eq(n\,150)' -frames:v 1 /tmp/s1_cup_lift_frame150.png
```

## Inspire-only trajectory visualization

The native tensor keeps the full human/object context because Dexplore's
observation and reward code uses it.  The hand trajectory itself is the 18
values in columns `373:391` (`245 + 32 * 4` through `+18`).  Therefore SMPL-X
is needed while creating the retargeted data and while rolling out the RL
checkpoint, but it is not needed to inspect or replay an already-exported
Inspire trajectory.

Use the Viser viewer below to display the actual Inspire URDF and its native
joint trajectory.  The default Inspire-only mode does not import SMPL-X and
does not require `yourdfpy`; `--show-mano` enables the optional canonical MANO
reference and therefore needs the SMPL-X model files:
the current `graspenv` already contains Viser; in another environment install
it with `pip install viser`.

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/visualize_inspire_trajectory.py \
    --sequence s1_cup_lift \
    --source geometric
```

Open `http://localhost:8080` in a browser.  The GUI provides frame, play,
speed, loop, and restart controls.  To overlay the geometric and RL hands,
use:

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/visualize_inspire_trajectory.py \
    --sequence s1_cup_lift \
    --source both
```

To compare the canonical MANO reference with both retargeted hands in the
same object frame, add `--show-mano`.  The MANO surface is the 778-vertex hand
embedded in InterAct's canonical SMPL-X output (the same 24-D PCA hand blocks
used by the GRAB representation), so it remains aligned with the canonical
object instead of using a separate raw-hand coordinate frame:

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/visualize_inspire_trajectory.py \
    --sequence s1_cup_lift \
    --source both \
    --show-mano \
    --geometric-root /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric_dexplore \
    --rl-root /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_dexplore
```

The GUI has independent mouse-controlled checkboxes for MANO, geometric, and
RL hands.  Each checked layer is visible, so any combination can be shown.
There are separate controls for the geometric reference object and the RL
simulated object; these should not be conflated once an RL export contains the
actual physics trajectory.  Color legends and labels beside the rendered
hands/objects identify every layer.  Use `--mano-side left` or
`--mano-side both` when the left reference is needed.  The canonical SMPL-X
model root can be overridden with `--smplx-model-root`.

To browse all exported trajectories in one viewer, use `--all` instead of
`--sequence`:

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/visualize_inspire_trajectory.py \
    --all \
    --source both
```

The same browser can include MANO for every sequence (the current sequence is
the only one whose canonical SMPL-X mesh is loaded at a time):

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
data_processing/visualize_inspire_trajectory.py \
    --all \
    --source both \
    --show-mano \
    --geometric-root /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_geometric_dexplore \
    --rl-root /home2/wyy/oyx_ws/Ref2Dex/data/processed_data/inspire_rl_dexplore
```

The `Trajectory` dropdown and the previous/next buttons switch sequences with
the mouse.  Switching resets the frame to zero and updates the object mesh when
one is available.  Trajectories are loaded lazily,
so the browser can list the full export without loading all 1335 tensors into
memory at once.  The status line also reports whether the object is in contact
and how many left/right human-hand contact labels are active; this is useful for
`*_offhand_*` clips whose early object motion may be driven by the hand that is
not retargeted into the visible Inspire hand.

The object mesh is shown when the corresponding InterAct object is available;
pass `--hide-object` for a hand-only view.  `--check-only` validates the URDF
and tensor mapping without starting a server, and prints a FK/object-distance
sanity check. A large fingertip distance means the source trajectory has a
coordinate-frame problem rather than a Viser rendering problem.

For reference, the native `(T, 598)` layout is:

| columns | content |
| --- | --- |
| `0:102` | left/right human hand wrist and joint pose features |
| `102:198` | 32 human key-joint positions |
| `198:205` | object position and quaternion |
| `205:238` | object contact flag and 32 hand/body contact labels |
| `238:245` | table position and quaternion |
| `245:373` | 32 human-hand global quaternions |
| `373:391` | 18 native Inspire DOFs (the trajectory displayed by the Viser script) |
| `391:598` | reserved zero padding in this export |
