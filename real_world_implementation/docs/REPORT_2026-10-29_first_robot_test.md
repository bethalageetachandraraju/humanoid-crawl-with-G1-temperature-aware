# G1 walk ↔ crawl chain: first real-robot test (2026-10-29)

For the agent maintaining the training code. This report says what ran on the
real braced G1, what went wrong, what the logs show, and what I suggest
changing in training. The deployment side (laptop controller) works. The
problem is in what the crawl and stand↔crawl policies learned.

**Short version.**
- The chain worked on hardware once end to end: walk → down → crawl (standing
  still) → up → walk.
- Twice the robot stopped itself: a motor hit **85 °C** while the robot held the
  crawl posture, and the controller's thermal stop switched to damping.
  - Run 3: the **right shoulder roll** (motor 23). It held about **8.8 Nm without a
    break**, with its PD target pinned at the joint limit, for 29 s of crawling
    with **zero command**.
  - Run 2, second descent: the robot settled **nose-down with its hips high**. The
    stand↔crawl policy never corrected it, so the hand-over gate never passed. It
    held about 50 Nm on the left knee for 25 s.
- The same static shoulder loading appears in simulation, so it is **learned
  behaviour**, not mainly a sim-to-real gap. Nothing in training penalises a
  steady 30–45 % load: there is no heat model, and the torque penalty only starts
  above 80 % of peak.
- **Crawl locomotion has not been tried on hardware yet.** Every crawl second on
  the robot had command 0. In simulation, moving loads the waist yaw and both
  shoulder rolls even more.

The request: fine-tune the **crawl** and **stand↔crawl** policies, keeping their
observation and action interfaces, so they hold and crawl inside the motors'
continuous ratings, without forgetting what they do now. The walk is fine.

---

## 1. What was deployed

| Role | Checkpoint (training machine, `mjlab/logs/rsl_rl/`) | Task id | ONNX in the bundle |
|---|---|---|---|
| Walk | `g1_walk_29dof/2026-10-23_06-41-09_walk_hard_track/model_55499.pt` | walk (hard) | `walk.onnx`, 84 in |
| Stand ↔ crawl | `g1_stand_to_crawl_29dof/2026-10-22_21-34-25_stand_crawl_v3/model_24999.pt` | `Mjlab-StandCrawl-Unitree-G1-29dof` | `stand_crawl.onnx`, 108 in, phase `[phi, dphi/dt]` |
| Crawl | `g1_crawl_29dof/2026-10-12_17-55-24_gen2_flat_16k/model_49999.pt` | `Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof` | `crawl.onnx`, 88 in |

The bundle is `mjlab/deploy/g1_chain_v2/`, as exported by
`scripts/export_chain_v2.py`, unchanged.

The laptop runs a numpy/onnxruntime port of `scripts/play_chain.py`'s
`ChainSwitcher`:
- `deploy/g1_chain_v2/chain_controller.py` and `sim2sim_mujoco.py` (laptop only so
  far);
- the Unitree SDK2 loop `unitree_sdk_python/g1_chain_real.py`.

It is the same state machine and the same observation layouts. Everything runs
at 50 Hz, with all three policies in shadow mode and a 0.5 s target blend at
every hand-over. Transitions were set to 8 s. Gains are as in `chain.json`: kp =
armature·(2π·10)², damping ratio 2. The braced wrist motors (20, 21, 27, 28) are
in damping.

The robot's gates use the IMU, joint speeds and forward kinematics of
`mujoco/scene.xml` instead of contacts:
- HOLD → CRAWL needs max |qd| < 0.5, |g − g_crawl| < 0.15 and pelvis over the
  mean limb-link height 0.151 ± 0.03 m, all for 1 s.

Two deployment details matter here:
- **PD targets are clamped to the joint ranges** before they are sent (training
  never clamped). The status counter `clamp 150` in run 3 is this clamp, almost
  all of it on the right shoulder roll.
- **Thermal stop:** if any motor's temperature reading reaches 85 °C, the
  controller switches every motor to damping. This is our controller's limit,
  not Unitree's firmware cut-off.

Before the robot, the laptop passed CPU MuJoCo sim-to-sim (7 variants: delay up
to 20 ms, pushes up to 0.5 m/s, transitions of 4–8 s) and a DDS sim-to-sim with
1 kHz motor PD.

## 2. What happened on the robot

The robot was on a harness, with braces and 100 mm extensions fitted. Commands
came from the keyboard.

| Run (log) | Duration | Sequence | Hottest motor reading | Ended by |
|---|---|---|---|---|
| 1 `chain_real_20261029_140936` | 185 s | stand, walk only (in place, sideways 0.3, turn 0.1) | 39 → 46 °C | operator |
| 2 `chain_real_20261029_153228` | 664 s | walk 226 s → C → down 8 s → HOLD 10 s → **CRAWL 53 s, command 0** → U → up 8 s → walk 283 s → C → down 8 s → **HOLD 25 s, gate never passed** | 85 °C | thermal stop in HOLD |
| 3 `chain_real_20261029_153830` (6 min after run 2) | 108 s | walk 41 s → C → down 8 s → HOLD 4.6 s → **CRAWL 29 s, command 0** | 85 °C | thermal stop: "motor 23 at 85 C" |

After the stop the robot sagged onto its torso: the tilt reading went from 96°
to 68°. The crawl command was **exactly 0 for all 82 s of crawling** (checked in
the logs). The walk was mostly in place.

Heating rate of the hottest motor:

| Segment | Rate | Note |
|---|---|---|
| Run 2 walk, 226 s | ~0 °C/s (49 → 50) | walking is thermally fine |
| Run 2 walk after the crawl, 283 s | −0.05 °C/s (67 → 52) | walking cools |
| Run 2 CRAWL, 53 s | +0.35 °C/s (53 → 71) | |
| Run 2 HOLD #2, stuck 25 s | +1.4 °C/s (51 → 85) | |
| Run 3 HOLD, 4.6 s | +1.25 °C/s | |
| Run 3 CRAWL, 29 s | +1.1 °C/s (54 → 85) | |

The run 3 motor was probably still warm inside from run 2: the user's
motor-safety model estimates the housing time constant at about 5 min, and the
break was 6 min.

## 3. Measurements: real vs simulation, per phase

**Simulation used for comparison:** CPU MuJoCo with the training model
(`deploy/g1_chain_v2/mujoco/scene.xml`) and the same deploy controller. Delay was
10 ms and transitions 8 s. The sequence was: walk in place 20 s, C, down, HOLD,
crawl in place 30 s, U, up, walk in place 10 s. Torque here is the actuator force
(PD, clipped at peak).

**Real torque** is `tau_est` from LowState. It matches kp·(target − q) − kd·qd
computed from the logged targets within 0.5 Nm.

"×cont" is the RMS torque divided by the **estimated continuous rating** from the
project's own `src/mjlab/tasks/g1_motor_safety/README.md`:

| Motor class | Joints | Estimated continuous rating |
|---|---|---|
| 7520-22 | knees, hip roll | 36 Nm |
| 7520-14 | hip pitch/yaw, waist yaw | 23 Nm |
| 5020 ×2 | ankles, waist roll/pitch | 13.5 Nm |
| 5020 | shoulders, elbows, wrist roll | 6.8 Nm |

These ratings are stand-ins, not measured on a G1. Treat ×cont as relative heat
load: heat goes roughly with its square.

| Phase | Source | Duration | Pelvis tilt | Most loaded joints: RMS Nm (×cont) |
|---|---|---|---|---|
| WALK in place | sim | 30 s | 4° | R/L shoulder roll 3.0 (0.44), L ankle pitch 5.0 (0.37) |
| | real, run 2 | 510 s | 5° | L/R ankle pitch 8.6 / 8.0 (0.64 / 0.59), shoulder roll 3.2 (0.47) |
| DOWN (8 s) | sim | 8 s | 49° | waist pitch 15.9 (1.18), R knee 23.2 (0.65) |
| | real, run 2 | 16 s | 46° | waist pitch 15.2 (1.13), **L/R knee 34.1 / 30.6 (0.95 / 0.85)** |
| HOLD (phi = 1) | sim | 4 s | 88° | **R shoulder roll 11.2 (1.65)**, waist yaw 20.5 (0.89), L hip roll 25.8 (0.72) |
| | real, run 3 | 5 s | 88° | **L knee 46.4 (1.29)**, R knee 35.7 (0.99), R shoulder roll 6.5 (0.95) |
| | real, run 2 #2 (stuck) | 25 s | **101°** | **L knee 49.6 (1.38)**, R elbow 5.5 (0.81), R shoulder yaw 5.2 (0.77) |
| CRAWL, command 0 | sim | 30 s | 96° | **R shoulder roll 7.4 (1.09)**, L shoulder roll 3.4 (0.49), knees 9.5 (0.26) |
| | real, run 2 | 53 s | 93° | **L shoulder roll 7.5 (1.10)**, R shoulder roll 5.6 (0.83), knees 26–27 (0.75) |
| | real, run 3 | 29 s | 96° | **R shoulder roll 8.8 (1.29)**, L shoulder roll 6.3 (0.92), knees 25 (0.70) |
| UP (8 s) | sim | 8 s | 51° | waist pitch 14.0 (1.04), R hip yaw 23.9 (1.04) |
| | real, run 2 | 8 s | 56° | R hip yaw 27.8 (1.21), L hip yaw 21.8 (0.95), L shoulder roll 6.3 (0.93) |

**Crawl while moving: simulation only.** Same setup, 21 s of crawling after the
hand-over. This has not been tried on the robot.

| Crawl command | Speed reached | Most loaded joints: RMS Nm (×cont) |
|---|---|---|
| fwd 0.5 | 0.6 m/s | **waist yaw 34.1 (1.48)**, L/R shoulder roll 8.6 (1.27), knees 22 (0.62) |
| fwd 0.25 | 0.33 m/s | waist yaw 31.7 (1.38), shoulder roll 8.1–8.2 (1.21) |
| turn 0.5 | | waist yaw 32.9 (1.43), shoulder roll 7.7–7.8 (1.15) |
| lateral 0.3 | | waist yaw 32.7 (1.42), shoulder roll 7.7–7.9 (1.15) |

So moving will heat the waist yaw and both shoulder rolls continuously, more than
holding still does.

### Joint positions and targets in the crawl at command 0 (rad)

| Joint | Crawl pose (`chain.json`) | sim q | real q (run 2 / run 3) | real target, mean |
|---|---|---|---|---|
| L shoulder roll | +1.443 | +1.53 | +1.65 / +1.75 | +2.19 / +2.21 |
| R shoulder roll | −1.443 | −1.52 | −1.67 / −1.64 | −2.06 / **−2.25 = joint limit, 100 % of ticks** |
| L knee | 2.113 | 1.63 | 1.99 / 1.91 | 1.72 / 1.66 |
| R knee | 2.113 | 1.72 | 1.98 / 2.15 | 1.70 / 1.89 |
| L elbow | −0.438 | −0.50 | +0.04 / −0.01 | |
| R elbow | −0.438 | −0.31 | −0.13 / +0.15 | |

Shoulder roll joint range: L [−1.5882, 2.2515], R [−2.2515, 1.5882].

## 4. Findings

1. **The policy holds the shoulder roll against a stall at zero command, and the
   simulator does it too.**
   - The crawl policy keeps one shoulder-roll target 0.4–0.6 rad beyond where the
     joint actually is, at or past the joint limit. That is 6–9 Nm of steady PD
     torque with no motion.
   - Sim: R shoulder roll 7.4 Nm, target −2.04 vs q −1.52. Real: up to 8.8 Nm.
   - At phi = 1 the stand↔crawl policy does the same, up to 11.2 Nm on the right
     shoulder roll in sim.
   - It is asymmetric (sim R 7.4 vs L 3.4 Nm), and the side flips between real
     runs.
   - In the simulator this is free. On the robot a 5020 shoulder motor heats about
     1 °C/s and trips in under 30 s.

2. **Training does not penalise sustained moderate torque.** In the 2026-10-29
   copy of the code:
   - `src/mjlab/tasks/g1_crawl/config/g1/env_cfgs.py` → `_add_torque_safety_29`:
     `motor_torque_margin` only penalises demand **above 80 % of peak**, i.e. 20 Nm
     on a shoulder.
   - `joint_torques_l2` has weight −1e-4, which is negligible.
   - There is no thermal model in the crawl or stand↔crawl tasks.
   - The stand↔crawl task is likely similar; check its reward set.

3. **Targets beyond the joint range: training and deployment disagree.**
   - Training never clamps PD targets, so the policy learned to use "target past
     the limit" as a force source: torque = kp·(target − q), with no clamp.
   - On the robot the targets are clamped to the joint range (they have to be, to
     keep the joints off their hard stops). So the robot gets less torque than the
     policy expects at exactly those joints.
   - In the DDS-sim log this also happened for: waist pitch (DOWN/UP, at its −0.52
     limit 18–51 % of ticks), ankle roll/pitch and shoulder roll.

4. **The knees carry about 3× more on the real robot.**
   - Crawl at command 0: about 26 Nm real vs 9 Nm sim.
   - Real knees are more bent (q ≈ 1.9–2.15 vs 1.63–1.72).
   - DOWN/HOLD knees: 30–50 Nm real vs about 23 Nm sim.
   - The walk's ankle pitch is about 2× sim (8–11.6 vs 4–5 Nm).
   - Possible causes, unverified:
     - total mass or mass distribution: the braces are modelled at 0.15 kg each and
       the extension tips at 0.1 kg, but the real parts have not been weighed;
     - pelvis/CoM offset;
     - joint friction;
     - the harness.
   - The knees were not the limit in run 3, but the stuck HOLD (L knee 49.6 Nm,
     1.38× its estimated rating) is consistent with a knee trip in run 2. The log
     records only the hottest reading, not which motor; the run 3 motor came from
     the terminal message.

5. **The stand↔crawl policy can rest in a second, wrong equilibrium at phi = 1.**
   - On run 2's second descent it settled **nose-down, pelvis pitch about 101–103°
     instead of 88°**, so |g − g_crawl| ≈ 0.25–0.27 against the gate's 0.15.
   - Its hips were **4 cm high**: pelvis over limbs 0.19 m against 0.151 ± 0.03.
   - It was otherwise still (|qd| ≈ 0.1 rad/s), and stayed like that for 25 s
     without correcting.
   - The other two HOLDs (run 2 #1, run 3) converged within 4–10 s: tilt 88–89°,
     pelvis over limbs 0.136–0.151.
   - Harness tension may have contributed; it wasn't recorded.

6. **The walk is fine thermally.** It walked in place for 4.7 min with the hottest
   motor at 50 °C or below, and cooled the robot after crawling. No walk changes
   needed.

7. **The crawl command is roughly m/s.** fwd 0.5 gives 0.6 m/s in sim. The robot
   controller allows up to 0.5 by default; first moving tests should stay at
   0.1–0.2.

## 5. Suggestions for training

Constraint: **keep both interfaces**:
- crawl: 88 inputs, target = crawl_pose + 0.5·a;
- stand↔crawl: 108 inputs, target = ref(phi) + 0.5·clip(a, ±1.6);
- the same joint order, action scales and phase semantics.

Then only the two ONNX files change. `chain.json`, the robot controller and its
gates stay as they are. The walk does not need retraining.

### 5.1 Fine-tune, don't retrain (against forgetting)

- **Start from the deployed checkpoints:** crawl `gen2_flat_16k/model_49999`,
  stand↔crawl `stand_crawl_v3/model_24999`.
- **Keep the final training conditions:** every existing reward term, the command
  distribution, the randomization and the robustness curriculum at the level they
  ended at (stand↔crawl reached `TRANSITION_RAMP_HARD` level 1.0). Do not let
  curricula restart at 0.
- **Small steps:** low learning rate (about 1e-4 to 3e-4) and a small action std
  (about 0.2–0.3; `scripts/set_checkpoint_lr.py` already does lr and min-std).
- **Ease the new terms in:** start them at weight 0 and ramp them in over a few
  hundred iterations, or warm up the critic first (actor frozen for about 50–100
  iterations), so the first updates after the reward change aren't driven by a
  stale value function.
- **Keep the observation normalizer fixed,** or at least monitor its drift. Its
  statistics are baked into the ONNX.
- **Anchor to the old policy:** add a behaviour-cloning / KL term toward the frozen
  original actor on the rollout states, e.g. β·‖μ_new(s) − μ_old(s)‖², and decay β
  over the fine-tune. This is the most direct guard against forgetting, together
  with the gates below.
- **Use the symmetry augmentation / mirror loss** (`g1_crawl/mdp/symmetry.py`
  exists). The shoulder load is asymmetric, and its side flips between runs.

### 5.2 New objective terms (crawl and stand↔crawl)

1. **Penalise sustained load, not peaks.**
   - Option (a), a thermal proxy: per joint, an exponential average of torque²
     with roughly the winding time constant (16 s for 5020, 25 s for 7520).
     Penalise relu(√E_j / τ_cont,j − 0.6)², summed over joints, including the
     shoulders, elbows, waist yaw/pitch and knees.
   - Option (b), the project's own plug-in:
     `g1_motor_safety/plugin.py::motor_safe(cfg, mode="penalty" | "plugin",
     init_temp="mixed", observe=False)`. With `observe=False` the network inputs
     don't change. That is the "PluginNoObs" route the dance experiments used to
     fine-tune a finished policy inside the heat model and filter.
2. **Train with the deployment's target clamp,** or penalise targets outside the
   range:
   - clamp the PD targets to the `chain.json` joint ranges in the action term, so
     the policy can't rely on preload against a limit;
   - or penalise Σ relu(|target − clip(target)|).
   - Either removes the stall strategy and the train/deploy mismatch from
     finding 3.
3. **A rest posture at zero command.**
   - Make holding still cheap in torque. A stronger version of term 1, applied
     only when the command is 0 (crawl stand-still episodes and the stand↔crawl
     hold), should push the policy toward a posture where bones, not the shoulder
     roll motors, carry the weight: arms more under the shoulders, elbows near
     straight.
   - A small |target − q| penalty on the arm joints at rest is a simple extra.

### 5.3 Training distribution

- **Longer and more frequent stand-still in the crawl.** On the robot the crawl
  sits at command 0 for tens of seconds. Use more zero-command episodes and
  segments of 30–60 s.
- **Longer holds at phi = 1** for stand↔crawl. Currently 1.5–4 s; on the robot the
  hold lasts until the gate passes, so use up to about 20 s.
- **Recovery from the wrong rest pose** (finding 5).
  - Add hold starts at phi = 1 that are nose-down (pitch 95–110°), hips 3–6 cm
    high, and with asymmetric shoulders.
  - Reward converging to the crawl pose's pitch and pelvis-over-limbs height.
  - Track the deployment gate as a metric: share of holds that satisfy
    max |qd| < 0.5, |g − g_crawl| < 0.15 and pelvis over limbs 0.151 ± 0.03
    within 3 s.
- **Mass.** Check that payload, CoM and arm-end-mass randomization cover the real
  robot (finding 4). I've asked the user to weigh the braces and extensions. If the
  real robot is heavier, also train at that mass.

### 5.4 Acceptance checks before export (simulation)

1. **No regressions.** `scripts/play_chain.py --eval 256` nominal, `--hard` and
   `--hard --push 0.3` stay at today's level: 0 % / about 3 % / about 4 % any
   failure. Reach-crawl, crawled > 1 m, stood up and walking-again stay unchanged.
   Crawl velocity tracking within about 10 % of today.
2. **Stationary crawl, 60 s at command 0.** Every joint's RMS torque ≤ 0.6× its
   estimated continuous rating: shoulder roll ≤ about 4 Nm, knee ≤ about 20 Nm.
   With the heat model, starting at 50 °C, no winding passes 70 °C.
3. **Hold at phi = 1, 20 s.** Same budget, and the deployment gate is met within
   3 s in ≥ 99 % of episodes, including the perturbed starts.
4. **Crawl moving at 0.3 m/s, 60 s.** Waist yaw and shoulder rolls ≤ 0.8× their
   continuous rating, or the winding stays < 80 °C in the heat model.
5. **Targets in range.** PD targets outside the joint ranges on < 1 % of ticks
   (or zero, if clamped in training).
6. **Braces.** Braced-wrist moment p95 ≤ 5 Nm, as now.
7. **Export.** Re-export with `scripts/export_chain_v2.py`. The input dims must
   stay 84/108/88. Then run the laptop's `deploy/g1_chain_v2/sim2sim_mujoco.py
   --auto` (PASS/FAIL, plus peak motor load and braced-wrist moment per phase).

### 5.5 Bigger change, only if 5.1–5.3 aren't enough

Give the policies the motor temperatures as inputs (the plug-in's `plugin` or
`budget` modes; `scripts/motor_safety/widen_checkpoint.py` widens the input
layer). The robot provides two temperature readings per motor, so this is
possible on hardware. It changes the observation layout and needs a controller
update, which I'd do on the laptop side.

## 6. Data limitations

- **Only the hottest motor reading is logged** (`temp_max`), not each motor's
  temperatures, so which motor tripped in run 2 isn't recorded. Logging all 29×2
  readings is in place from the next session on (section 8). That will also give
  `scripts/motor_safety/sysid_thermal.py` real data to identify the heat model.
- **The continuous ratings are estimates** from the motor-safety README, not
  measurements.
- **Mass, CoM and harness:** the brace and extension masses are not weighed yet,
  and the harness tension was not recorded.
- **One robot, three runs, one afternoon.**

## 7. Files

In `real_robot_test_2026-10-29/`:

**`logs/chain_real_*.npz`.** Every array is sampled at 50 Hz, policy joint order
(`joints`):

| Array | Meaning |
|---|---|
| `t` | time |
| `mode` | index into `modes`: DAMP, STANDUP, STAND, CHAIN, SCHEDULE |
| `chain_mode` | WALK 0, STOPPING 1, DOWN 2, HOLD 3, CRAWL 4, CRAWL_STOP 5, UP 6, STAND_HOLD 7; −1 outside CHAIN |
| `phase` | phi |
| `q`, `dq` | joint positions and speeds |
| `target` | sent, after the clamp |
| `action` | the active policy's clipped action |
| `tau_est` | the driver's torque estimate |
| `gravity` | pelvis frame |
| `gyro` | |
| `cmd` | walk command |
| `ccmd` | crawl command [fwd, lat, turn] |
| `temp_max` | °C, max over motors |
| `above_limbs` | forward-kinematics pelvis over mean limb-link height |
| `pelvis_height` | forward-kinematics pelvis over the soles |
| `gate_ok` | |

Motor index = policy index for 0–19, and +2 for policy 20–24. So motor 23 is
policy joint 21, `right_shoulder_roll_joint`.

**`analysis/`:**
- `analyze.py`: the real-vs-sim tables above;
- `moving.py`: the moving-crawl table;
- `stats.json`: every number, for every joint and phase.

**Laptop-side code:** `mjlab/deploy/g1_chain_v2/chain_controller.py`,
`sim2sim_mujoco.py`, and `~/humanoid/unitree_sdk_python/g1_chain_real.py`.

## 8. Deployment-side changes (laptop, not needed from training)

Done after this test, in `unitree_sdk_python/g1_chain_real.py` (DDS-sim tested):
- **Temperatures:** all 29×2 motor readings are logged (`temps`), and the status
  line names the hottest motor.
- **Auto stand-up:** down on the floor (DOWN / HOLD / CRAWL / CRAWL_STOP), the
  robot stands up by itself, as if `u` was pressed, once a motor reaches 72 °C
  (`--hot-stand-up`). It goes through the normal gated path, so a stuck HOLD
  still needs the operator.
- **Crawl refused when warm:** `c` is refused while any motor is at 60 °C or more
  (`--crawl-below-c`).
- **Gate warning:** a gate that has waited 5 s prints a warning with what it waits
  for, and how to override it.
- **Keys:** w/s, a/d, q/e change forward, sideways and rotation by ±0.1 per
  press (walk and crawl); x zeroes them.

Logs from the next session will include per-motor temperatures, which are usable
for `sysid_thermal.py`.
