# G1 crawl in MJLab: what was built, how it is trained, how to retrain

This is the working record of the crawl project as of 2026-10-10. It is meant to be
enough to (a) understand every number in the configs, (b) reproduce the training,
and (c) retrain after changing the robot (for example a different hand or forearm).
File paths are relative to the `mjlab` checkout. The short task README
(`README.md` next to this file) is the quick-start; this file is the full story.

---

## 1. Goal and overall approach

Replicate the Isaac Lab `g1_crawl` project in MJLab: a Unitree G1 that locomotes on
hands and feet ("crawl"), commanded by forward speed and turn rate, then make it
robust on stairs and uneven ground with a commandable "high-step" mode.

The policy is a plain MLP trained with PPO (RSL-RL 5.5.1) on 8192 parallel MuJoCo
Warp environments. It is **blind**: no terrain sensing in the observations. The
training is a chain of resumes, each stage reusing the previous checkpoint:

1. Flat-ground crawl from animation resets (3-value command).
2. Flat polish ("Stage A"): wider turn range, pushes, smoothness terms.
3. Network widened to a 7-value command (step height + 3 spare slots), then
   terrain curriculum left to right over nine zones, then uniform spread.

Registered task ids (`config/g1/__init__.py`):

| Task id | Command | Terrain | Use |
|---|---|---|---|
| `Mjlab-Crawl-Flat-Unitree-G1` | 3 values | plane | original port, flat training |
| `Mjlab-Crawl-Rough-Unitree-G1` | 3 values | zones | terrain without step-height command |
| `Mjlab-Crawl-Flat-Cmd7-Unitree-G1` | 7 values | plane | flat play/eval of cmd7 checkpoints |
| `Mjlab-Crawl-Rough-Cmd7-Unitree-G1` | 7 values | zones | **current training task** |

---

## 2. Robot model (`robot.py`, `assets/g1_23dof_crawl.xml`, `assets/meshes/`)

MJLab's bundled G1 is 29-DoF with a long three-joint wrist. The source project
trained on a 23-DoF "simple forearm" G1 whose forearm is one rigid link ending in a
ball hand. We use the source's own MuJoCo export of that robot (its sim2sim
`scene.xml`, copied with its STL meshes). Reasons this matters for a hands-and-feet
gait, all measured, not guessed:

- MJLab's arm is about 7 cm longer and 0.7 kg heavier at the hand; its wrist
  pitch/yaw motors (5 Nm) back-drive under body weight; its wrist collision capsule
  counted as an "undesired contact" whenever a hand was planted. Together these made
  a feet-only gait the easier solution, which is the problem that started this work.
- MJLab's default pose is standing. With `use_default_offset=True`, action = 0 means
  "stand up", so every episode started with the robot launching itself out of the
  crawl pose and flipping onto its back; the crawl hip-roll angle was unreachable
  through the action clip (needed 6.9, clip 5.0).

What `get_crawl_spec()` does to the export at load time:

- Deletes the plane, light, camera, `<general>` actuators and keyframe (MJLab adds
  its own terrain, actuators and `init_state` keyframe).
- Adds `left/right_wrist_roll_joint` (hinge, axis X, range ±1.97222 rad, armature
  0.00360972) on the forearm bodies. The export has 21 joints; the Isaac USD, the
  animation and the policy layout have 23. The wrist roll does not move the hand
  contact point (the ball sits on the roll axis).
- Replaces the torso collision **box** with a **capsule** of the same extent
  (radius 0.09, half-length 0.165). MuJoCo caps box/heightfield contacts at 50 per
  pair and drops the rest; the box overflowed on rough tiles.

Joint order (`CRAWL_JOINTS`, also the model order and the policy's action order):
left leg (hip pitch, roll, yaw, knee, ankle pitch, roll), right leg (same), waist
yaw, left arm (shoulder pitch, roll, yaw, elbow, wrist roll), right arm (same).

Contact bodies: feet `left/right_ankle_roll_link` (7 thin capsules each, radius 1 cm,
lying flat on the sole), hands `left/right_forearm` (one sphere `*_hand_collision`,
radius 2 cm, 18 cm along the forearm). Sites used by sensors: `left_foot`,
`right_foot`, `left_palm`, `right_palm`.

Collision policy (`CRAWL_COLLISION`): all `*_collision*` geoms get `contype=0,
conaffinity=1` (robot-terrain only, **no self-collision**, matching the source's
`enabled_self_collisions=False`), `condim=3` (every contact frictional, as in
PhysX), `priority=1` so the robot's randomized friction wins over the terrain's
(mirrors the source's "multiply" combine against a unit-friction ground), base
friction 0.6.

Default pose (`CRAWL_INIT_STATE`, from the source `G1_CFG.init_state`): pelvis at
z = 0.22 m, orientation quaternion (0.721249, 0, 0.692676, 0) wxyz, i.e. pitched
about 1.53 rad so base +X points down; joints hip pitch -1.6796, hip roll ±2.4180,
hip yaw ±1.2084, knee 2.1130, ankle pitch 0.1941, ankle roll 0, waist yaw 0,
shoulder pitch 1.4579, shoulder roll ±1.5779, shoulder yaw ±1.4238, elbow -0.3125,
wrist roll 0 (left positive, right negative for roll/yaw). This pose is what
action = 0 commands, what `joint_pos_rel` observes relative to, and what
`joint_deviation_l1` pulls toward, exactly as in Isaac Lab.

Actuators (`CRAWL_ARTICULATION`, MJLab's `BuiltinPositionActuatorCfg`, i.e. MuJoCo
position actuators with kp/kd): arms 5020 (25 Nm), hip pitch/yaw and waist yaw
7520-14 (88 Nm), hip roll and knee 7520-22 (139 Nm), ankles 2 x 5020 (50 Nm).
kp = armature x (2 pi 10 Hz)^2, kd = 2 x 2.0 x armature x 2 pi 10 Hz (same formulas
as the source). `soft_joint_pos_limit_factor = 0.9`. Action: position target =
default pose + 0.5 x action, actions clipped to ±5 by the RSL-RL wrapper.

---

## 3. Simulation and physics

- MuJoCo Warp through MJLab 1.6.0; timestep 0.005 s, decimation 4, control at
  50 Hz; episode 20 s (1000 steps); `scale_rewards_by_dt = True` (Isaac Lab
  multiplies rewards by dt; the first port had this off, inflating every reward
  50x and breaking PPO's value scale).
- Contact budget `sim.nconmax`: 70 per world on flat, 140 on terrain (MJLab's
  velocity default of 35 overflowed on a single env on stairs: 14 foot capsules,
  2 hands, torso).
- Heightfield rule: MuJoCo Warp produces **one contact per capsule on a
  heightfield** and caps capsule/heightfield pairs at 50 contacts, so cell size must
  be 0.2 m (`horizontal_scale=0.2`; Perlin tiles use their own `resolution=0.2`).
  With 0.1 m cells the flat-ground policy flipped in 40% of envs on a *flat*
  heightfield and 0% on a flat box; with 0.05 m cells the feet lost grip and the
  robot slid in place. Near-flat heightfields are still worse than boxes for this
  robot, which is why flat ground is always a box tile.
- Zero-height tiles (stairs with 0 cm steps, grids with 0 cm) are degenerate nested
  boxes and also make the feet slide; `difficulty_range=(0.2, 1.0)` keeps every
  non-flat tile non-degenerate and the flat zone is a real plane box.

---

## 4. Observations and commands

Actor (policy input), all noisy during training (`enable_corruption=True`):

| Block | Size (cmd3 / cmd7) | Noise |
|---|---|---|
| projected gravity in base frame | 3 | U(-0.05, 0.05) |
| command | 3 / 7 | none |
| joint positions relative to default pose (23 joints) | 23 | U(-0.01, 0.01) |
| joint velocities | 23 | U(-1.5, 1.5) |
| previous action | 23 | none |
| **total** | **75 / 79** | |

Critic adds base linear velocity (3) and base angular velocity (3) in front: 81 / 85.

Command (`mdp/command.py`, `CrawlVelocityCommand`), resampled every 4 to 8 s:

| Slot | Meaning | Training range |
|---|---|---|
| 0 | forward speed along base Z (the forward axis when prone) | U(-1.0, 2.5) m/s |
| 1 | lateral speed | 0 (never used) |
| 2 | turn rate about base X (vertical axis when prone) | U(-3, 3) rad/s |
| 3 | commanded limb swing height (cmd7 only) | see section 6 |
| 4 to 6 | spare, always 0.0 (cmd7 only) | reserved |

15% of episodes are "idle": slots 0 to 2 forced to zero (slot 3 kept). The term also
accumulates, per episode, the distance crawled in the commanded direction and the
commanded distance; the terrain curriculum uses these.

Turn tracking compares the command with the **base-frame roll rate**. The source
compared against world yaw rate, which flips sign when prone; checkpoints trained
before 2026-10-04 17:00 have the mirrored turn direction (`--invert-turn` in teleop).

Left-right symmetry augmentation (`mdp/symmetry.py`): PPO sees each batch plus its
mirror (joint permutation left/right with sign flips on roll/yaw joints, negated
gravity Y, negated lateral and turn commands; step height and spares unchanged) and
a mirror loss with weight 0.5. The function is built per command size by
`make_crawl_symmetry_augmentation(num_cmd)`.

---

## 5. Rewards (`mdp/rewards.py`, weights in `config/g1/env_cfgs.py`)

All terms are multiplied by dt (0.02) per step; `Episode_Reward/*` in TensorBoard is
the episode sum divided by 20 s, so multiply by 20 for the per-episode value.

Shared by all crawl tasks:

| Term | Weight | What it does |
|---|---|---|
| `track_lin_vel_yz_exp` | +3.0 | exp(-err²/0.5²) on [forward, lateral] base velocity |
| `track_ang_vel_x_exp` | +3.0 | exp(-err²/0.5²) on base roll rate (turn) |
| `forward_progress` | +1.0 | speed along commanded direction, clipped at the command; non-zero gradient when stopped |
| `crawl_orientation` | +0.2 | 1 - 0.5·‖g_b - (1,0,0)‖², max when base +X points down |
| `base_height_l2` | -0.1 | (pelvis z - 0.22)² |
| `joint_deviation_l1` | -0.01 | Σ|q - q_default| |
| `joint_pos_limits` | -5.0 | soft-limit violation |
| `action_rate_l2` | -0.01 | Σ(a_t - a_{t-1})² |
| `joint_acc_l2` | -1e-7 | Σ q̈² |
| `base_wobble` | -0.02 | angular velocity about the two non-turning body axes |
| `base_bobbing` | -0.5 | base-frame X (vertical) velocity² |
| `stand_still` | -0.01 | Σ|q̇| while the command is below 0.1 |
| `joint_torques_l2` | -1e-4 | Σ τ² |
| `torque_limits` | -5.0 | torque beyond motor limits (139/88/50/25 Nm by joint) |
| `undesired_body_contacts` | -5.0 | number of bodies other than feet and forearms touching terrain (force > 1 N) |
| `feet_slide` | -0.2 | horizontal speed of feet/hands while in contact |
| `both_feet_air` / `both_hands_air` | -0.5 each | both feet (hands) airborne |
| `both_left_limbs_air` / `both_right_limbs_air` | -0.1 each | same per side |

Terrain tasks add:

| Term | Weight | What it does |
|---|---|---|
| `limb_clearance` (cmd7) | -0.5 (x2 after iter 20000) | Σ over 4 limbs of \|height above terrain - commanded height\| x limb horizontal speed |
| `limb_lift_bonus` (cmd7) | +0.5 (x2 after iter 20000) | Σ exp(-(h - target)²/0.03²) x min(speed, 1) |
| `limb_clearance` (cmd3) | -0.5 | same with a fixed 8 cm target |
| `termination_penalty` | -100 | on the step the episode is terminated by a roll-over |

Reward balance lesson: the first terrain run collapsed into standing still. At the
end of the flat phase tracking earned about +37 per episode against about -40 of
penalties, so when terrain made tracking harder the penalties for moving won, and
with tracking std 0.25 a stopped robot received no gradient to restart. The current
weights give the flat policy +80 per episode on flat ground, +79 on warm-up tiles
and +67 on random terrain (measured with `model_7000`), which is the margin that
keeps "moving" the better option.

---

## 6. Events, randomization, terminations

Startup domain randomization (per env, once): geom friction U(0.4, 1.0) absolute on
all collision geoms; body masses x U(0.9, 1.1); torso mass + U(-1, 1) kg; joint
friction x U(0.5, 2.0); joint armature x U(1.0, 1.05).

Reset: `reset_from_animation_frame` samples one of 80 frames of
`assets/animation_rc4.json` (the source's crawl motion capture, 23 joints plus
root), adds pose noise ±5 cm / ±0.1 rad, root velocity noise ±0.2 m/s / ±0.3 rad/s,
scales joint angles by U(0.9, 1.1), and places the root at the env origin (terrain
height included). Pushes: base velocity set to ±0.5 m/s in x/y every 5 to 10 s.

Terminations: time-out at 20 s; on terrain tasks also `crawl_flipped`
(projected gravity X below 0, i.e. more than 90 degrees from prone). Play configs
use an effectively infinite episode, no pushes, no observation noise.

Step-height command sampling (cmd7): per episode, with probability 0.8 from the
range of the tile the robot stands on, else uniform in (0.02, 0.15):
flat 2 to 6 cm, meadows/hill/valley 4 to 10, stairs up/down and box grid 6 to 12,
big stairs 8 to 14, random mix 6 to 14.

---

## 7. Terrain and curriculum (`CRAWL_ZONES_CFG`, `mdp/curriculums.py`)

Nine zones, declaration order = column order left to right, each with 5 difficulty
rows (difficulty 0.2, 0.4, 0.6, 0.8, 1.0), 8 x 8 m tiles:

| Zone | Tile | Range at difficulty 0.2 to 1.0 |
|---|---|---|
| flat | box | plane |
| meadows | Perlin heightfield, 0.2 m cells | mound height 8 to 25 cm |
| hill | pyramid slope heightfield | 15 to 35% grade |
| valley | inverted pyramid slope | 15 to 35% grade |
| stairs_up | box pyramid stairs, 0.45 m treads | 2 to 8 cm steps |
| stairs_down | inverted box pyramid stairs | 2 to 8 cm steps |
| box_grid | 0.4 m box grid | 1 to 6 cm height steps |
| big_stairs | inverted box pyramid stairs, 0.4 m treads | 4 to 12 cm steps |
| random_mix | (flat placeholder) | envs here are placed on a random other non-flat zone at the same row |

`terrain_zones_crawl` gives every env a progress index that walks through the rows
of a zone and then into the next zone to the right (flat counts as a single step,
so the path is 1 + 8 x 5 = 41 steps). Rules, applied at each episode reset:

- Promote when distance crawled in the commanded direction (accumulated over the
  episode) exceeds 80% of the accumulated commanded distance and at least 1 m was
  commanded; demote below 50%. Episodes ended by a roll-over never promote and
  always demote.
- The first 15% of envs are pinned to the flat zone for the whole run.
- Phases by **global** iteration (MJLab restores the env step counter on resume):
  warm-up (all in flat) until `FLAT_ITERATIONS = 7000`; curriculum until
  `SPREAD_AFTER_ITERATIONS = 15000`; then every reset lands on a uniformly random
  zone and row. Swing-height reward weights double at
  `STEP_HEIGHT_BOOST_ITERATION = 20000`.
- TensorBoard: `Curriculum/terrain_zones/zone_mean`, `zone_max`, `row_mean`,
  `unlocked`, `spread`, `frac_<zone>`.

Play mode uses the same nine zones with 4 rows at difficulty 0.4 to 1.0, no
curriculum, random spawn.

---

## 8. PPO (`config/g1/rl_cfg.py`)

| Setting | Flat tasks (source values) | Terrain cmd7 task |
|---|---|---|
| actor / critic | MLP 256-128-128, ELU, no obs normalization | same |
| action distribution | Gaussian, scalar std, init 1.0 | same |
| clip_param | 0.05 | **0.2** |
| entropy_coef | 0.008 | **0.005** |
| learning rate | 1e-3, adaptive on KL (desired 0.01) | same |
| epochs / minibatches | 5 / 4 | same |
| gamma / lambda | 0.99 / 0.95 | same |
| value loss coef, grad clip | 1.0, 1.0 | same |
| steps per env per iteration | 24 | same |
| symmetry | data augmentation + mirror loss 0.5 | same |
| envs | 4096 to 8192 | 8192 |

Why the terrain task differs: with clip 0.05 the adaptive schedule cut the
learning rate to its 1e-5 floor within 150 iterations of the curriculum changing the
data distribution, and it never recovered (a whole 2300-iteration run trained
nothing). Entropy had also drifted from 33 to 41 (std 1.0 to 1.7). RSL-RL restores
the stored learning rate on resume; `scripts/set_checkpoint_lr.py` rewrites it.

Throughput on the RTX 5090: about 1.05 s per iteration at 8192 envs on the plane,
1.1 to 1.3 s on terrain; 1000 iterations is roughly 18 to 20 minutes.

---

## 9. Run history and checkpoint lineage (`logs/rsl_rl/g1_crawl/`)

| Run | From | What | Verdict |
|---|---|---|---|
| `old_g1_29` | scratch | first port, 29-DoF G1, rewards unscaled | failed: feet-only, noise std grew to 4.5 |
| `2026-10-04_14-44-44` | scratch | 23-DoF robot, dt-scaled rewards | healthy, stopped at 1750 (verification) |
| `2026-10-04_15-07-17` | scratch | flat, 3000 iterations, 8192 envs | good flat crawler (turn sign mirrored) |
| `2026-10-04_18-12-42` | 15-07-17/2999 | Stage A rewards, first terrain attempt | **`model_7000.pt` is the best flat checkpoint**; collapsed to standing still after terrain unlock (reward balance) |
| `2026-10-04_18-15-31` | 15-07-17/2999 | parallel attempt | not used |
| `2026-10-05_cmd7_from_7000` | 18-12-42/7000 | widened to 7-value command (`expand_policy_obs.py`), identical actions | base for all cmd7 runs |
| `2026-10-05_11-15-08` | cmd7/7000 | first cmd7 terrain run | curriculum paced too slowly (fixed), `model_7750.pt` kept |
| `2026-10-05_12-10-16` | 11-15-08/7750 | after curriculum pacing fix | learning rate hit 1e-5 floor at 7900; nothing learned after |
| `2026-10-05_lrreset_from_7750` | 11-15-08/7750 | learning rate reset to 1e-3 | base for the current run |
| `2026-10-05_13-15-22` | lrreset/7750 | clip 0.2, entropy 0.005, flip rule, flat anchors | current terrain run |

---

## 10. Scripts

| Script | Purpose |
|---|---|
| `uv run train <task> ...` | MJLab trainer; resume with `--agent.resume True --agent.load-run <folder> --agent.load-checkpoint model_N.pt` |
| `uv run play <task> --checkpoint-file ... --viewer native|viser` | play; viser has a checkpoint dropdown |
| `scripts/play_teleop.py --task ... --checkpoint ...` | keyboard drive: W/S speed, A/D turn, X stop, H step height (cmd7) |
| `scripts/eval_crawl.py --task ... --checkpoint ... [--step-height h] [--render]` | contact fraction per limb, hand force share, orientation, speed, swing peaks, frame sheet + GIF |
| `scripts/crawl_tile_test.py --checkpoint ... --tile <zone|preset> --difficulty d [--render]` | flips/512 envs in 6 s on one tile type: the physics check for any new tile |
| `scripts/train_with_viewer.py --task ... --num-envs 100 [--checkpoint ...] [--spread-terrain]` | live training view of all envs in viser |
| `scripts/expand_policy_obs.py --checkpoint ... --out-dir ...` | widen a 3-command checkpoint to 7 (zero columns, identical behaviour) |
| `scripts/set_checkpoint_lr.py --checkpoint ... --out-dir ... --lr 1e-3` | rewrite the stored optimizer learning rate |
| `src/mjlab/scripts/visualize_terrain.py` | MJLab's terrain preset browser |
| `uv run tensorboard --logdir logs/rsl_rl/g1_crawl` | curves |

Health signals in TensorBoard: `Train/mean_reward` positive and not trending down
for long; `Train/mean_episode_length` near 1000 (a collapse toward tens of steps
means the policy is "committing suicide" through the flip termination);
`Loss/learning_rate` between 1e-4 and 3e-3 (1e-5 means frozen); `Loss/entropy`
roughly 30 to 40 (above 45 the noise is running away); `Episode_Termination/flipped`
low; `Episode_Reward/limb_lift_bonus` rising on cmd7 runs.

---

## 11. Retraining after changing the robot (hands, forearm, feet)

Checklist, in order. Items marked (joints) only apply if the joint set changes.

1. **Model.** Put the new MJCF and meshes in `assets/` and point `G1_CRAWL_XML` at
   it (`robot.py`). Keep the body names the sensors use, or update them:
   `FEET`, `HANDS`, `LEFT_LIMBS`, `RIGHT_LIMBS` in `robot.py` and the contact
   sensor regex in `env_cfgs.py` (`.*_(ankle_roll_link|forearm)$`), plus the four
   sites in `LIMB_SITES` (`left_foot`, `right_foot`, `left_palm`, `right_palm`;
   the height sensor and clearance rewards cast rays from these, so put a palm site
   at the new hand's contact point). Collision geom names must match
   `.*_collision.*`. Prefer spheres/capsules; avoid boxes on anything that can touch
   a heightfield (50-contact cap). Keep self-collision off unless the source changes.
2. **(joints)** Update `CRAWL_JOINTS`, the mirror permutation and signs in
   `mdp/symmetry.py`, `CRAWL_INIT_STATE.joint_pos`, actuator groups in
   `CRAWL_ARTICULATION`, and the torque limits table in `applied_torque_limits`.
   The animation reset requires every joint in `CRAWL_JOINTS` to exist in
   `animation_rc4.json` by name; a joint that is not in the animation needs either
   a new animation or a fallback value in `reset_from_animation_frame`.
3. **Sanity checks before any training** (all cheap):
   - `uv run play <flat task> --agent zero --viewer native`: with action 0 the robot
     must settle onto hands and feet, pelvis roughly 0.13 to 0.30 m, and never
     launch or flip. If it does, the default pose or action offset is wrong.
   - `uv run pytest tests/test_task_configs.py -q`.
   - `scripts/crawl_tile_test.py --tile flat` with an old checkpoint is only valid
     if observations and actions are unchanged; otherwise skip.
   - Watch for "height field collision overflow" or "nconmax overflow" messages;
     both mean dropped contacts (cell size, geom shape or `sim.nconmax`).
4. **Can an old checkpoint be reused?** Only if the observation and action sizes
   and orderings are identical (same 23 joints, same command size). A hand/forearm
   change that keeps the joints lets you resume `model_7000.pt` (or any cmd7
   checkpoint) directly; the policy adapts to the new contact geometry in a few
   hundred iterations. Any joint change means training the flat stage from scratch.
5. **Training sequence for a new robot:**
   1. Flat, 3-value command, from scratch, 8192 envs, 3000 iterations
      (`Mjlab-Crawl-Flat-Unitree-G1`). Expect mean reward to turn positive within a
      few hundred iterations; verify with teleop and `eval_crawl.py` (all four limbs
      cycling, hand force share 40 to 55%, exact speed tracking at 1 m/s).
   2. `expand_policy_obs.py` on the chosen checkpoint, then
      `Mjlab-Crawl-Rough-Cmd7-Unitree-G1` resumed from it with
      `--agent.max-iterations 25000`. Set `FLAT_ITERATIONS` to the checkpoint's
      iteration (curriculum starts immediately) or higher (extra flat polish), and
      `SPREAD_AFTER_ITERATIONS` 8000 later; `STEP_HEIGHT_BOOST_ITERATION` 5000
      after that.
   3. Judge with `eval_crawl.py` at `--step-height 0.03` and `0.12` on the rough
      task: swing peaks should differ by 5 cm or more; flips low; speed near command.
6. **If something goes wrong**, the failure modes seen so far and their signatures:
   - robot never uses hands: default pose / arm length / wrist compliance (section 2);
   - episodes collapse to tens of steps: environment is net-negative, policy ends
     episodes on purpose (check reward balance and tile physics);
   - robot stops moving on terrain: penalties exceed tracking, tracking kernel too
     narrow (section 5);
   - learning rate 1e-5: KL thrash after a distribution change (section 8);
   - sliding, "weird leg" poses on terrain: heightfield cell size / degenerate tiles
     / dropped contacts (section 3).

---

## 12. The braced 29-DoF robot (2026-10-10)

The robot on the bench is the stock 29-DoF G1, modified for crawling. Its model is
`robot29.py`; task ids end in `-29dof` and log to `logs/rsl_rl/g1_crawl_29dof/`.

**Hardware changes, and how they are modelled** (`get_crawl_spec_29()`; design
constants at the top of `robot29.py`):

| Change | Model |
|---|---|
| Rubber hands removed | Mesh and `*_hand_collision` deleted; the 149 g hand split out of the `*_wrist_yaw_link` inertial (motor link 105 g) |
| Printed wrist brace, from 10 mm past wrist roll to the wrist-yaw flange | Wrist pitch/yaw joints deleted (welded at 0); brace = cylinder body on `*_wrist_roll_link`, 40 mm radius, 150 g; replaces `*_wrist_collision` |
| Printed forearm extension, 100 mm, hemispherical tip | `*_forearm_extension` body on the flange: capsule r = 25 mm, 100 g, tip site `*_hand_contact` |

Elbow axis to tip is 325.5 mm (source robot 300 mm). Wrist roll stays actuated.
The policy controls **25 joints**: 12 leg, 3 waist (yaw, roll, pitch, all free
for the policy), 10 arm. On hardware, hold the four braced 4010 motors at 0 with
low gains or in damping mode so the brace carries the load.

**Crawl pose and reset animation.** `scripts/retarget_crawl_animation.py` solves the
five arm joints so the extension tip and forearm axis land where the source's ball
hand was (tip within 3 mm, axis within 0.6 deg; the residual exists because wrist
roll cannot move an on-axis tip). It writes `assets/crawl_pose_29dof.json` (the
default pose, mirror-symmetric) and `assets/animation_rc4_29dof.json` (80 reset
frames). Re-run it after changing `EXTENSION_LENGTH`; building the robot refuses a
pose solved for another length. Lowest tip point at pelvis z = 0.22: -0.0255
(source hand -0.0246); feet identical; nothing self-collides within 21 mm in any
frame.

**Torque safety** (`_add_torque_safety_29`, `mdp/rewards.py`). The source
`torque_limits` term cannot fire: it reads `qfrc_actuator`, already clamped to the
motor force range (still true on the 23-DoF tasks, left unchanged). The 29-DoF
tasks replace it with two terms that **start switched off** and are faded in by
the robustness curriculum once the robot already crawls well:

| Term | What it penalizes | Weight |
|---|---|---|
| `motor_torque_margin` | PD torque demand `kp(q_target - q) - kd qdot` above 80% of each motor's peak, in units of the peak, all 25 motors | 0 until level 0.2, -5 at 0.6 |
| `braced_wrist_torque` | Moment across each braced wrist joint about its axis (MuJoCo torque sensor at the welded joint, `robot29.BRACED_LOADS_29`) above 80% of 5 Nm | 0 until level 0.2, -1 at 0.8 |

Measured with the 23-DoF `model_7000.pt` driving this robot (unadapted, so on the
high side): elbows over their 25 Nm peak on 7% of steps; braced-wrist moment median
1-2 Nm, 95th percentile ~11 Nm, peak 36 Nm, mostly in turns. The tip is 19 cm from
the wrist axes, so propulsion forces at the tip become wrist moments. At the full
weights that gait pays about -22 (motors) and -27 (wrists) per episode against
+87 of tracking; at twice those weights the penalties matched tracking, which is
the standing-still failure of section 5.

**Elegance** (`_add_elegance_29`), on from the first iteration because a gait's
style is set while it is first learned. The reference is the rc4 capture: a
diagonal-sequence walk (each foot followed by the opposite hand), each limb down
~57% of a 0.4 s stride.

| Term | What it does | Weight |
|---|---|---|
| `gait_diagonal` | diagonal limbs share contact state while moving: trot phase 1, all four down 0.5, same-side pair 0 (reference crawl 0.48) | +0.3 |
| `gait_air_time` | each limb whose current swing is 0.10-0.40 s, while moving | +0.25 |
| `limb_clearance` | moving hands/feet away from 5 cm above the ground, times their speed (terrain tasks replace it) | -0.5 |
| `soft_landing` | contact force at each touchdown | -0.002 |
| `action_smoothness` | second difference of the actions | -0.005 |
| `posture` | `exp(-mean(err^2/std^2))` around the crawl pose; per-joint std from the reference crawl's RMS motion (left/right max, mirror-symmetric), x1.5 crawling, x2.25 fast, x0.5 standing; waist roll/pitch floor 0.25 rad | +0.5 |

With the borrowed gait the group nets about +5 per episode; a clean gait can earn
up to about +22, a quarter of tracking. The flat 29-DoF task now has the
`limb_height_scan` sensor too.

**Robustness curriculum** (`mdp/robustness.py`, `_add_robust_training_29`). All
physics randomization is re-drawn at every episode reset (not once at startup)
from ranges owned by one `level` in [0, 1]:

| Knob | Level 0 | Level 1 |
|---|---|---|
| Base velocity kick every 4-8 s | off | +/-1.0 m/s xy, +/-0.6 rad/s yaw |
| Robot friction | 0.5-1.0 | 0.25-1.25 |
| Body mass | x(1 +/- 0.05) | x(1 +/- 0.15) |
| Torso payload | +/-0.5 kg | -2 to +3 kg |
| Torso centre of mass | 0 | +/-3 cm |
| Joint dry friction | 0-0.1 Nm | 0-0.6 Nm (stock model has none) |
| Armature | x(1 +/- 0.02) | x(1 +/- 0.15) |
| Motor kp/kd | nominal | x(1 +/- 0.15) |
| Motor peak torque | nominal | x(0.85-1.0) |
| Actor observation noise | 0.5x source | 1.5x source |
| Motor command delay | 0 | 5-20 ms, held ~1 s at a time |
| Torque-limit penalties | off | see above (in from level 0.2) |

The level only moves after a 500-iteration warm-up and at least 150 iterations
since the last change. Up by 0.1 when the averaged crawled/commanded ratio is
>= 0.75 and roll-overs <= 5% of episodes; down by 0.1 below 0.45 or above 25%.
The level is not saved in checkpoints: a resumed run starts at `start_level`
(default 0) and climbs back. Roll-overs end the episode (`crawl_flipped`, -100)
on the flat 29-DoF task too.

PPO: clip 0.2, entropy 0.005 from the start (section 8: the source's clip 0.05
collapses the learning rate whenever a curriculum shifts the data). 8000
iterations at 8192 envs, about 1.75 s per iteration (~4 h).

**What to watch.** TensorBoard groups (`uv run tensorboard --logdir
logs/rsl_rl/g1_crawl_29dof`), or `scripts/plot_crawl_dashboard.py` for a one-page
PNG with limits drawn in:

- `Curriculum/robustness/*`: `level`, the gate (`gate_progress_ratio`,
  `gate_flip_rate`) and every knob's current value.
- `Episode_Metrics/track/*`: forward speed achieved, speed and turn errors.
- `Episode_Metrics/gait/*`: hands/feet on ground, hand share of load (healthy
  crawl 0.40-0.55), diagonal sync, swing height, pelvis height, prone alignment,
  torso and other contacts; `Metrics/air_time_mean`, `Metrics/landing_force_mean`.
- `Episode_Metrics/safety/*`: braced-wrist peak Nm and time over 5 Nm, motor
  peak demand / limit by group (legs, waist, arms), time any motor saturated,
  wrist-roll peak Nm, time near a joint limit.
- `Episode_Metrics/effort/*`: mechanical power, action jerk.

**Scripts.**

| Script | Purpose |
|---|---|
| `scripts/view_crawl_robot.py [--physics] [--dof23]` | MuJoCo viewer of the robot in its crawl pose |
| `scripts/retarget_crawl_animation.py [--render out.png]` | Re-solve crawl pose and reset animation |
| `scripts/crawl_torque_monitor.py [--checkpoint ...] [--mode hold]` | Live sim + bar chart of every motor's torque vs its limit; saves CSV + summary on exit. Without a 25-joint checkpoint it maps the 23-DoF policy across |
| `scripts/plot_crawl_dashboard.py [run_dir] [--watch 120]` | Dashboard PNGs from TensorBoard logs |

**Runs.**

| Run | From | What | Verdict |
|---|---|---|---|
| `g1_crawl_29dof/2026-10-10_18-26-14_braced_flat_elegant` | scratch | flat, 3 commands, robustness + elegance, 8000 iterations | good: level 1.0 by iteration 2000; braced wrists over 5 Nm 10% of the time (from 67%); motors saturated 0.7%; speed tracked -0.92 to +2.44 m/s; drifts left 2-9 deg per 10 s when told to go straight (fixed by heading hold, below) |

---

## 13. Second generation: gyro, lateral, straight lines, step height (2026-10-12)

Task `Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof` (and its terrain twin
`Mjlab-Crawl-Rough-Cmd7-Unitree-G1-29dof`); the 3-command 29-DoF task is unchanged
so its checkpoints keep loading.

**Why.** The first 29-DoF run veered left when commanded straight: 2-9 deg per 10 s
on the nominal robot, up to 25 deg on randomized ones, always the same way. The
model is symmetric (whole-body centre of mass 0.1 mm off centre), so it was a
small policy bias: 0.016 rad/s, which the turn reward scores 99.9% correct, and
which the policy could not feel without a gyro. A heading-hold loop removes it
with no retraining (`turn = -1.0 x wrap(target - heading)`, clamped to 0.5 rad/s;
positive turn commands are clockwise seen from above): 0.1-0.6 deg per 10 s,
worst 3 deg.

**What changed.**

| Change | Detail |
|---|---|
| Gyro in the policy inputs | `base_ang_vel` after gravity, noise U(-0.2, 0.2) rad/s scaled by the curriculum. The G1 pelvis IMU frame is aligned with the pelvis, so the real gyroscope reading maps directly. Actor 88 inputs (3 gravity + 3 gyro + 7 commands + 75 joints), critic 91 |
| Command modes | per resample: 10% stand, 25% straight (lateral 0), 10% sideways (forward 0), 10% spin in place, 45% mixed. Straight and sideways hold their starting heading (the loop above), so "go straight" is trained exactly as it will be used |
| Lateral command | slot 1, U(-0.5, 0.5) m/s, +Y = robot's left; tracked by `track_lin_vel_yz_exp`; `forward_progress` and the curriculum's progress score now use the planar [forward, lateral] direction (identical when lateral is 0) |
| Step height | slot 3, U(0.02, 0.15) m; flat clearance follows it (`limb_clearance_cmd` -0.5, `limb_lift_bonus` +0.5) instead of a fixed 5 cm. Slots 4-6 stay spare at 0 for future commands (body height, cadence) |
| Mirror fix (all tasks) | the critic's angular velocity flipped only Z under the left-right mirror; a pseudovector flips X and Z, and X is the turn rate, so every mirrored sample showed the critic the turn rate with the wrong sign |
| `stand_still` fix (all tasks) | it tested the norm of the whole command, which the step-height slot keeps non-zero, so it never fired on 7-command tasks; now velocity slots only |
| Run length | 50000 iterations |

New metrics: `track/lateral_speed_error_mps`, `gait/step_height_cmd_m`, and
`Metrics/base_velocity/error_heading` (mean heading error of heading-hold
commands, rad).

**Throughput (RTX 5090).** 8192 envs: 1.87 s per iteration, 12.9 GB peak; physics
is ~90% of it. 50000 iterations is about 26 h.

**Watching a checkpoint.** `scripts/crawl_torque_monitor.py --checkpoint ...` picks
the task from the checkpoint's input size and adds Q/E sideways, H step height and
G heading hold (on by default).

---

## 14. Standing, walking, and getting down into the crawl (2026-10-17)

Package `src/mjlab/tasks/g1_walk/`: the same braced robot as section 12
(`robot29.py`: hands off, wrist pitch/yaw welded, forearm extensions, 25 active
joints in the crawl policy's joint order), started upright. Two tasks and a
switcher script form the chain the real robot runs: walk around, stop, go down
into the crawl pose, hand over to the crawl policy.

### 14.1 `Mjlab-Walk-Flat-Unitree-G1-29dof`: stand and walk in every direction

MJLab's own G1 velocity task (`tasks/velocity`), transplanted onto `robot29`
with the sim-to-real additions of the crawl task:

| Piece | Detail |
|---|---|
| Actor inputs (84) | projected gravity (3), pelvis gyro (3), command `[vx, vy, wz]` (3), joint positions minus the standing pose (25, with the encoder-bias randomization applied), joint velocities (25), last action (25). No base linear velocity: the real robot has no clean estimate of it. Critic (97) adds the true joint positions, base linear velocity and foot contact state |
| Actions | PD targets = standing pose + per-motor scale x action (`G1_ACTION_SCALE`: 0.55 rad for hip pitch/yaw and waist yaw, 0.35 for hip roll/knee, 0.44 for the rest), raw actions clipped at 5 |
| Commands | `UniformVelocityCommand`: forward -0.6..1.0 m/s, lateral +-0.5 m/s, yaw +-0.8 rad/s; 15% of episodes stand still, 30% hold a heading, 20% walk straight; resampled every 3-8 s |
| Rewards | the velocity task's (tracking 2 + 2, upright, variable posture, foot clearance/swing/slip, soft landing, self-collision, action rate, joint limits, angular momentum) plus the crawl task's demand-based torque limits (`motor_torque_margin` on every motor, `braced_wrist_torque` on the welded wrists, both faded in by the curriculum from level 0.2), a penalty for anything but the feet on the ground, `stand_still` on zero commands, and -50 on a fall |
| Terminations | 20 s time-out, tilt over 70 deg, pelvis below 0.3 m |
| Robustness | `walk_robustness` (`mdp/robustness.py`): the crawl curriculum's ramp (section 12) gated on walking instead of crawling: distance made good along the commanded displacement over distance commanded (standing episodes are not scored), promote at >= 0.75 with <= 5% falls, demote below 0.45 or above 25% falls. Level 0 is already a deployable floor (friction 0.4-1.0, mass +-10%, gains +-5%, strength 0.9-1.0, 10 ms delay, 0.3 m/s pushes, full observation noise); level 1 is the crawl's end point (friction 0.25-1.25, payload -2..+3 kg, CoM +-3 cm, gains +-15%, strength 0.85, 5-20 ms delay, 1 m/s pushes, 1.5x noise) |
| PPO | the velocity task's network (512-256-128, observation normalization on, so the exported ONNX takes raw observations), clip 0.2, entropy 0.005, 24 steps x 4096 envs, 20000 iterations (~1.4 s each alone on the 5090) |

The run `g1_walk_29dof/2026-10-17_19-18-46_walk_v1` stood still for its first
~4000 iterations (the tracking kernel pays a motionless robot half credit on
these command ranges), then found the gait on its own, climbed the curriculum
to level 1 by ~10000 iterations and finished at 20000 (7.5 h). Final checkpoint
(`scripts/eval_walk.py`, 256 robots, deterministic actor): no falls on the
nominal robot and none with level-1 randomization plus a 0.5 m/s kick in every
phase; tracking error 5-10 cm/s nominal and 10-17 cm/s randomized at 0.5 and
1.0 m/s forward, 0.4 m/s backward and sideways; busiest motor at 0.7-0.86 of its
limit (median), braced wrists under 1 Nm. The weak command is turning in place:
0.6 rad/s commanded gave 0.1-0.3 rad/s (it turns fine while walking). Uniform
sampling of the three ranges almost never produces a pure turn, so the policy
twists on planted feet instead of stepping round. Hence, for the fine-tune:

* `WalkVelocityCommand` (`mdp/command.py`): 15% of episodes spin in place (no
  linear velocity, at least 0.3 rad/s), and the yaw tracking kernel is tighter
  (std 0.5 instead of sqrt(0.5), weight 2.5);
* `Mjlab-Walk-Flat-Sym-Unitree-G1-29dof`: the same task with the crawl's
  left-right mirrored data and mirror loss (`mdp/symmetry.py`, walk layouts), so
  both turn directions come out alike.

```bash
uv run train Mjlab-Walk-Flat-Sym-Unitree-G1-29dof --env.scene.num-envs 4096 --agent.resume True --agent.load-run 2026-10-17_19-18-46_walk_v1 --agent.load-checkpoint model_19999.pt --agent.max-iterations 4000 --agent.run-name walk_sym_spin
```

(`--agent.max-iterations` counts additional iterations on a resume; the
robustness level restarts at 0 and climbs back in ~1500 iterations.) The
fine-tune also carries the review fixes made after the first run started:
encoder bias in the actor's joint positions, the level-0 floor above, small
pushes from the start, one friction value per robot.

What the fine-tunes found: the spin episodes and the tighter kernel alone did
not make the policy step round from a standstill (it kept standing through pure
yaw commands); a positive `feet_air_time` term for any command above 0.15 did
(`walk_sym_spin2`, resumed again at 22250 with it). In-place turning then came
and went between checkpoints, so the checkpoint is chosen by
`scripts/eval_walk.py` rather than taken as the last one. **Current best:
`g1_walk_29dof/2026-10-18_07-45-18_walk_sym_spin2/model_25000.pt`** (ONNX in
its `exported/`): no falls nominal or randomized with kicks; tracking 6-10 cm/s
nominal, 11-17 cm/s randomized; a 0.6 rad/s turn in place delivers 0.44-0.50
rad/s both ways (0.4 rad/s spins still do not start stepping). A third
fine-tune (`walk_sym_spin3`, from 25000) adds a linear `yaw_progress` credit
toward the commanded rate (`mdp/rewards.py`), a stronger air-time term and
more spin episodes down to 0.2 rad/s; take its checkpoints only if the sweep
beats 25000.

### 14.2 `Mjlab-StandToCrawl-Unitree-G1-29dof`: the slow way down

The policy follows a kinematic schedule and only corrects it, the recipe of the
humanoid get-up papers (HumanUP's slowed-down tracking stage, "Learning to Get
Up"'s residual on a retimed reference), run in reverse and with the end state
fixed by the crawl task.

* **Keyframes** (`mdp/keyframes.py`, solved by least-squares IK on the robot
  model and checked with forward kinematics; the design notes are in the
  scratchpad `keyframes/`): stand (phi 0) -> crouch (0.15) -> deep squat, torso
  35 deg forward, extension tips 11 cm up (0.3) -> all fours, tips on the ground
  35-38 cm ahead, pelvis 0.42 m, base 52 deg (0.55) -> legs begin to splay
  (0.68) -> half splay, pelvis 0.28 m, base 82 deg (0.8) -> (0.9) -> the crawl
  pose (1). Feet flat and on the ground throughout; along the interpolated path
  nothing but soles and tips comes within 1.7 cm of the floor, the feet dip at
  most 1.6 cm, no self-collision.
* **Schedule** (`TransitionSchedule`): smoothstep between consecutive keyframes
  for the joints, pelvis height, base pitch and tip height, so the motion eases
  into a brief pause at every keyframe. The phase runs 0 -> 1 over a duration
  drawn per episode from 4-8 s, after 0.5-1.5 s of standing; the policy observes
  `[phi, 1/duration]`, so the speed is a run-time choice.
* **Action** (`ScheduledJointPositionAction`): target = schedule(phi) + 0.5 rad x
  residual, raw residual clipped at +-1.6, so the policy can never be more than
  0.8 rad from the reference: the reference carries the motion, the policy keeps
  the balance and picks the contacts.
* **Actor inputs (108)**: gravity, gyro, `[phi, 1/T]`, joint positions minus
  the standing pose, joint velocities, last residual, and the reference minus the
  measured joint positions (25).
* **Rewards**: pelvis height, base orientation (gravity direction), joint pose
  and tip height near their schedules (exponential kernels); a kernel on the
  crawl pose (std 0.35 rad, weight 3) plus a stillness penalty once phi = 1 (the
  state the crawl policy takes over); both feet on the ground up to the
  all-fours keyframe (phi 0.55) and the soles flat on the ground from phi 0.9:
  between the two the feet have to step some 0.6 m outward into the sprawl (the
  first run rewarded planted feet throughout and learned to cheat: knees
  over-bent, feet on their toes, 0.37 rad RMS off the pose); penalties on joint speed over 2.5 rad/s,
  pelvis speed over 0.5 m/s, any rise of a ground force between physics steps
  (impacts), contacts other than feet and tips, self-collision, joint limits,
  action rate; the torque-limit penalties from level 0 (-1 motors, -0.5 braced
  wrists, to -5/-1 by level 0.5); -100 on termination.
* **Terminations**: fallen (gravity toward base -X, sideways, or past prone), or
  the pelvis, torso, waist, shoulders, elbows, wrists or braces touching the
  ground (30 N).
* **Robustness**: `transition_robustness`, the same ramp gated on success
  (phase complete, joint RMS error to the crawl pose under 0.25 rad, no fall) with
  pushes only from level 0.2, up to 0.3 m/s.
* **Episode**: 12.5 s (standing wait + descent + hold). PPO as the walk task,
  raw-action clip 1.6, 10000 iterations.

Run `g1_stand_to_crawl_29dof/2026-10-18_00-40-32_stand_to_crawl_v2` (6 h with
the walk run sharing the GPU): the descent worked by iteration 1000, the
curriculum reached level 1 by 3000, and the end pose tightened to 0.06 rad RMS
by 7000. Final checkpoint through the whole chain (`play_switcher.py --eval 256`,
walk policy `walk_v1/model_19999`):

| Condition | Reached pose / fell / forbidden contact | Pose at hold | Descent peaks (median; p95) | Hand-over | Crawled |
|---|---|---|---|---|---|
| nominal, 6 s | 100% / 0% / 0% | 0.05 rad RMS, pelvis 0.218 m | motor 0.63 of limit; braced wrists 4.1 Nm (4.4); joint speed 2.8 rad/s (3.1); ground-force rise 97 N/step (151) | 100%, 7.0 s after C | 100% |
| level-1 randomization + 0.3 m/s push mid-descent, 6 s | 100% / 0% / 0% | 0.05 rad, 0.217 m | motor 0.62 (0.66); braced 3.9 Nm (5.0); speed 2.8 rad/s (3.4); rise 78 N/step (200) | 100%, 7.0 s | 100% |
| nominal, 4 s | 100% / 0% / 0% | 0.04 rad | motor 0.66; braced 3.1 Nm (6.4); speed 3.5 rad/s | 100%, 5.1 s | 100% |
| nominal, 8 s | 100% / 0% / 0% | 0.07 rad | motor 0.57; braced 4.0 Nm (4.2); speed 2.5 rad/s; rise 75 N/step | 100%, 9.0 s | 100% |

Use 6-8 s on the robot: at 4 s the braced wrists see over 5 Nm at the 95th
percentile. The schedule-only descent (no policy) falls every time, so the
residual is doing the balancing. The walk and transition ONNX exports reproduce
their torch actors to 1e-6 (`observation_names` in the ONNX metadata give the
input order).

### 14.3 `scripts/play_switcher.py`: the chain

One simulation, three trained policies, a small state machine:
`WALK --C--> TRANSITION --phase 1--> HOLD --gate--> CRAWL`. Each policy's
observation vector is rebuilt from the robot state in exactly its training
layout (`_WalkPolicy`, `_TransitionPolicy`, `_CrawlPolicy`: verified to match the
training environment's observations to float precision), every policy runs in
shadow mode before it is in charge so its own last action is always consistent,
and the PD targets are blended over 0.5 s at each hand-over. The gate before the
crawl policy takes over: joints still (max 0.5 rad/s), pelvis within 3 cm of
0.22 m, gravity direction within 0.15 of the crawl pose's, both feet and both
tips on the ground, all for 1 s. Keys: W/S, Q/E, A/D, X drive the active policy;
C starts the descent; H cycles the crawl step height; P pushes; F freezes.

`--eval N [--robust] [--push v]` runs N robots headless through stand -> C ->
hold -> crawl forward and prints the success rates and the peak loads (motor
demand/limit, braced-wrist moment, joint speed, ground-force rise). With the
schedule played without a policy (`--transition-checkpoint none`) every robot
falls: the kinematic path is not a balanced one; the residual policy is what makes
it so.

On the real robot the same three ONNX files and the same observation code apply
(the walk and transition exports carry their normalizers); the crawl policy is
loaded from its checkpoint as before. Run the four braced wrist motors in damping
mode in every state.
