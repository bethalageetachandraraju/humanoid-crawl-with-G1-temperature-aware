# Real-world implementation: G1 walk ↔ crawl chain (sim-to-real)

This folder runs policies trained in simulation on the real Unitree G1 (29-DoF,
wrist braces and forearm extensions). It is self-contained: it does not import
mjlab, torch or the training code. It needs numpy, plus Unitree's DDS stack to
talk to the robot. It runs the policies in two ways:

| | **A. Laptop over Ethernet** | **B. Onboard, Unitree remote** |
|---|---|---|
| Where the controller runs | Your laptop, cable to the robot | The G1's own computer (PC2, `unitree@192.168.123.164`) |
| How you drive it | Keyboard in the terminal (or `--remote`) | The Unitree wireless remote |
| Policy runtime | onnxruntime or numpy | numpy only (`.npz` weights) |
| Motor loop | 500 Hz, every state message decoded | 100 Hz, newest state polled once per tick |
| Start it with | `python controller/g1_chain_real.py --real --net <iface> --wrists braced` | `./run_robot.sh` in `~/g1_onboard` (built by `controller/tools/make_onboard_package.py`) |
| Use it for | First tests of a new bundle, debugging, logging | Untethered runs (the ~2 h crawl session) |

Both use the same code: `controller/g1_chain_real.py`, which runs the bundle's
`chain_controller.py`. A policy that works in mode A works unchanged in mode B.

In this repo the folder sits next to `training/` (the mjlab tasks, rewards and
heat model) and `deploy/` (bundles exported from training). Its own
`bundles/g1_chain_v3/` is the bundle that ran on the robot. The newer
`deploy/g1_chain_v3_1` has the same interface but a different stand → walk
hand-over; see [§4.1](#41-the-repos-v31-bundle).

**If you are an agent working on training:** start with
[The deployment contract](#3-the-deployment-contract-what-a-trained-policy-must-match).
It is the interface between training and this folder. Then read [Making a new
bundle](#4-making-a-new-bundle-from-training) and [What the hardware taught
us](#10-what-the-hardware-taught-us-feed-this-back-into-training).

---

## 1. Folder map

```
real_world_implementation/
├── README.md                     this file
├── requirements-laptop.txt       pip packages for mode A and the sim tests
├── controller/                   runs the robot (laptop, or copied onto the robot by the packager)
│   ├── g1_chain_real.py          the walk↔crawl controller: modes, keys, remote, thermal guard, logging
│   ├── g1_walk_real.py           base layer: DDS I/O, LowCmd, safety stops, damping, run loop
│   ├── g1_config.py              joint order, motor map, kp/kd, stand pose, joint limits, safety numbers
│   ├── thermal_guard.py          motor-temperature guard (warn 65 °C, stand up 72 °C, stop 85 °C); same file as ../deploy/thermal_guard.py
│   ├── inputs.py                 terminal keys and the Unitree wireless remote decoder
│   ├── walk_policy.py            walk-only ONNX runner (used by g1_walk_real.py on its own)
│   ├── check_robot.py            read-only readiness check (SDK, network, bundle, policy self-test, timing)
│   ├── sim_bridge.py             MuJoCo stand-in for the robot over the same DDS messages (loopback only)
│   ├── assets/g1_29dof_braced_chain_sim.xml   the bridge's robot model (from the bundle's scene.xml)
│   ├── tools/
│   │   ├── onnx_to_npz.py        bundle ONNX → numpy .npz weights + selftest.npz (for the robot computer)
│   │   ├── make_onboard_package.py   builds build/g1_onboard/ for mode B
│   │   └── make_sim_model.py     bundle scene.xml → assets/…_chain_sim.xml for the bridge
│   ├── third_party/unitree_sdk2_python/   Unitree's Python SDK 1.0.1 (BSD-3), unchanged
│   ├── vendor_src/cyclonedds-0.10.2.tar.gz   CycloneDDS Python bindings source (compiled on the robot)
│   └── vendor_extra/typing_extensions.py     for Python 3.8 on the robot
├── bundles/
│   └── g1_chain_v3/              the bundle that ran on the robot (2026-10-30), see §3 and §4
│       ├── chain.json            the contract: joints, motors, gains, poses, limits, obs/action, gates
│       ├── walk.onnx  stand_crawl.onnx  crawl.onnx (= crawl_creepy.onnx)  crawl_scuttle.onnx
│       ├── *.npz  selftest.npz   numpy copies of the ONNX files + reference outputs
│       ├── chain_controller.py   state machine, observations, targets, gates (numpy)
│       ├── sim2sim_mujoco.py     CPU sim-to-sim on the training model (+ --test suite)
│       ├── replay_log.py         replays a robot log in MuJoCo (video, PD-target ghost)
│       ├── mujoco/scene.xml      the exact training model (also used for forward kinematics)
│       └── README.md             the training team's notes for v3
├── analysis/
│   ├── plot_motors.py            29 per-motor torque/temperature pages for one log
│   └── plot_session.py           hottest motor, temperature/torque/current heatmaps for logs
├── example_logs/chain_real_20261030_093943.npz   a real v3 run: walk → crawl → stand → walk
├── docs/REPORT_2026-10-29_first_robot_test.md    first hardware test (v2): heat, clamp, findings
└── build/                        (generated) build/g1_onboard/ for mode B
```

The controller writes logs to `controller/logs/`. `build/` and
`controller/logs/` are generated; `.gitignore` excludes them.

---

## 2. The robot as this code assumes it

- **Unitree G1, 29 DoF.** Wrist pitch and yaw (motors 20, 21, 27, 28) are
  locked by 3D-printed braces and have **100 mm forearm extensions** (the
  crawl stands on them). Those four motors get kp 0, kd 0.5 (damping). The
  policies drive the other **25 joints**.
- **Waist unlocked.** The policies drive waist roll and pitch.
- **DDS:** `rt/lowstate` (unitree_hg `LowState_`, about 1040 Hz on this robot)
  and `rt/lowcmd` (`LowCmd_`, 35 slots, 29 used). Domain 0 on the robot's
  192.168.123.x network. Every LowCmd has `mode_pr = 0` (ankles and waist as
  pitch/roll joints), `mode_machine` copied from LowState (this robot reports
  5), per motor `mode 1, q, dq 0, tau 0, kp, kd`, then the CRC.
- **The built-in controller must be released** (`MotionSwitcherClient
  .ReleaseMode()`) before low-level control. The controller does it after you
  type `release`. **The robot goes limp at that moment, so it must hang on the
  gantry.**
- **IMU:** `imu_state.quaternion` (w, x, y, z) and `imu_state.gyroscope`
  (rad/s) are the pelvis IMU, the same frame as the training model's
  `imu_in_pelvis` site.
- **Temperatures:** `motor_state[i].temperature` is two readings per motor. We
  treat them as winding and housing, and every guard uses the hotter one.
- **Network:** the robot's computers are 192.168.123.161 (Unitree's motion
  computer) and 192.168.123.164 (PC2, ours, `eth0`). The laptop needs a free
  192.168.123.x/24 address on its wired interface (this laptop: `enp3s0`,
  192.168.123.26).
- **The robot computer's clock runs 12 h ahead (China time).** Log names
  written on the robot are in that time.

---

## 3. The deployment contract (what a trained policy must match)

Everything here is read from `bundles/<name>/chain.json`, and
`g1_chain_real.load_bundle` **asserts** most of it against `g1_config.py` at
start-up. A bundle that disagrees refuses to load. If training changes any of
these on purpose, change `g1_config.py` with it and re-run every test in §5.

### 3.1 Timing
- Policies run at **exactly 50 Hz** (`control_hz: 50`, dt 0.02 s). Training
  used physics dt 0.005 s with decimation 4.
- Each target is held between policy steps: 10 motor commands in mode A
  (500 Hz), 2 in mode B (100 Hz).
- The policies are **sensitive to late steps**. On the robot computer the
  policy once ran at 38 Hz (7–8 ms steps) and the walk went unstable within a
  second. The controller now goes to damping if 5 of the last 50 policy
  intervals are over 26 ms.
- Sim-to-sim passes with up to 20 ms of action delay (`--delay-steps 4`). Keep
  delay randomization of at least 0–20 ms in training.

### 3.2 Joints, motors, braced wrists
Policy joint order (25) is MuJoCo order of the training model:

```
 0 left_hip_pitch   1 left_hip_roll   2 left_hip_yaw   3 left_knee   4 left_ankle_pitch   5 left_ankle_roll
 6 right_hip_pitch  7 right_hip_roll  8 right_hip_yaw  9 right_knee 10 right_ankle_pitch 11 right_ankle_roll
12 waist_yaw       13 waist_roll     14 waist_pitch
15 left_shoulder_pitch  16 left_shoulder_roll  17 left_shoulder_yaw  18 left_elbow  19 left_wrist_roll
20 right_shoulder_pitch 21 right_shoulder_roll 22 right_shoulder_yaw 23 right_elbow 24 right_wrist_roll
```

- Policy joint `i` → motor `i` for `i < 20`; policy joints 20–24 → motors
  22–26.
- Motors 20, 21, 27, 28 (wrist pitch/yaw) are braced. They are not in the
  policy and are damped on the robot. In the training model they are welded,
  and their load is measured at the `*_wrist_*_load` sites (`braced_joints` in
  `chain.json`, 5 Nm rating).

### 3.3 PD gains and torque limits
The G1 driver computes `τ = kp·(q* − q) + kd·(0 − q̇)` (no velocity target, no
feed-forward), limited by the motor's peak torque. The gains are mjlab's:
`kp = armature·(2π·10 Hz)²`, `kd = 4·armature·2π·10` (damping ratio 2).

| Joints | Motors | kp | kd | Peak Nm | Walk scale |
|---|---|---|---|---|---|
| hip pitch, hip yaw, waist yaw | 0, 2, 6, 8, 12 | 40.179 | 2.558 | 88 | 0.5476 |
| hip roll, knee | 1, 3, 7, 9 | 99.098 | 6.309 | 139 | 0.3507 |
| ankle pitch/roll, waist roll/pitch (two motors via linkage) | 4, 5, 10, 11, 13, 14 | 28.501 | 1.814 | 50 | 0.4386 |
| shoulders, elbow, wrist roll | 15–19, 22–26 | 14.251 | 0.907 | 25 | 0.4386 |
| wrist pitch/yaw (braced) | 20, 21, 27, 28 | 0 | 0.5 | 5 | – |

These are softer than Unitree's example gains on purpose. `python
controller/g1_config.py` prints the full per-joint table and checks it against
`chain.json`.

### 3.4 Observations
No scaling, noise or clipping outside the network. Any normalizer must be
baked into the exported ONNX (see 3.7). Frames and units:

- `gravity_b` = Rᵀ·(0, 0, −1) from the IMU quaternion (unit vector, pelvis
  frame).
- `gyro` = `imu_state.gyroscope` (rad/s, pelvis frame).
- `q`, `qd` = motor encoders in PR mode, in policy order (rad, rad/s).
- `last_action` = **this policy's own clipped output** from the previous tick.

| Policy | Dim | Layout |
|---|---|---|
| walk | 84 | `gravity_b(3), gyro(3), [vx, vy, wz](3), q − stand_pose(25), qd(25), last_action(25)` |
| stand ↔ crawl | 108 | `gravity_b(3), gyro(3), q − stand_pose(25), qd(25), last_action(25), [φ, dφ/dt](2), ref(φ) − q(25)` |
| crawl (every style) | 88 | `gravity_b(3), gyro(3), [fwd, lat, turn, step_h, 0, 0, 0](7), q − crawl_pose(25), qd(25), last_action(25)` |

**Shadow mode:** every policy runs every tick on the live state, also while
another policy drives the robot. Each one sees its own last action. When
control passes to a policy, its target blends in over 0.5 s (smoothstep) from
the target that was applied. At the start of the chain all last actions are
zero.

### 3.5 Actions → PD targets
| Policy | Clip | Target |
|---|---|---|
| walk | ±5.0 | `stand_pose + scale ⊙ a`, scale = 0.25·peak/kp (table above) |
| stand ↔ crawl | ±1.6 | `ref(φ) + 0.5·a` |
| crawl | ±5.0 | `crawl_pose + 0.5·a` |

**Every target is then clamped to `joint_limits`** (the joint's hard range).
v3 was trained with this clamp. v2 was not, and on the robot its shoulder target
sat on the clamp for 30 s at 8.8 Nm (§10). **Train with the clamp.**

### 3.6 Commands, phase and the state machine
- **Walk command** `[vx, vy, wz]`: ranges vx −0.6…1.0, vy ±0.5, wz ±0.8 (wz > 0
  turns left). The controller limits how fast it changes (0.5 m/s², 0.5 m/s²,
  1 rad/s²). The operator's default caps are vx −0.3…0.5, vy ±0.3, wz ±0.3.
- **Crawl command** `[fwd, lat, turn, step_height]`:
  - creepy: fwd −0.2…0.5, lat ±0.25, turn ±0.7, step height 0.02…0.15 m (starts
    at 0.03);
  - scuttle: fwd −0.3…1.2, lat ±0.4, turn ±1.0;
  - **turn > 0 is clockwise seen from above** (the opposite of the walk's wz);
  - slots 4–6 are always 0;
  - command change rate is limited to 0.5, 0.25 and 1.0 per second.
- **Phase** φ: 0 = standing, 1 = crawl pose. Going down, φ rises at 1/T and
  dφ/dt = +1/T; going up it falls at −1/T. Holding either end, dφ/dt = 0. **T =
  8 s on the robot** (trained 4–8 s). `ref(φ)` is a per-joint smoothstep
  between the 8 keyframes in `chain.json → policies.stand_crawl.keyframes`.
- **States:**
  `WALK —down→ STOPPING —still→ DOWN —φ=1→ HOLD —crawl gate→ CRAWL —up→
  CRAWL_STOP —still on 4 limbs→ HOLD → UP —φ=0→ STAND_HOLD —upright, still→
  WALK`.
  - The walk command is 0 outside WALK; the crawl command is 0 outside CRAWL.
  - Crawl styles switch only when the crawl command is 0 and the robot is
    still; the switch blends over 0.5 s.
  - **HOLD stands back up** if the crawl gate hasn't passed after 8 s.
- **Robot gates** (sim uses contacts and base velocity; the robot uses the IMU,
  joint speeds and numpy forward kinematics of `mujoco/scene.xml`):

  | Gate | Conditions |
  |---|---|
  | STOPPING / STAND_HOLD | max \|qd\| < 0.6 rad/s, \|gyro\| < 0.3 and \|yaw rate\| < 0.2 rad/s, feet height difference < 2 cm |
  | STAND_HOLD, also | gravity_z < −0.97, pelvis > 0.70 m over the soles |
  | HOLD (crawl gate) | max \|qd\| < 0.5, \|g − g_crawl\| < 0.15, all four limbs within 4 cm of the lowest, pelvis 0.151 ± 0.03 m above the mean limb origin |
  | CRAWL_STOP | same as HOLD, but \|g − g_crawl\| < 0.25 and no height check |
  | Dwell | 1.0 s in HOLD, 0.5 s elsewhere |

  **The crawl pose the stand ↔ crawl policy ends in must meet the HOLD gate on
  the robot**, or HOLD times out and stands up. The 10-29 v2 run found a
  nose-down equilibrium (tilt 99–103° instead of 88°) that never did.

### 3.7 Export format
- One ONNX file per policy: input named **`obs`**, shape `[1, N]` float32,
  output `[1, 25]` actions (before the clip).
- **Stateless MLP only.** The robot computer has no onnxruntime, so
  `controller/tools/onnx_to_npz.py` converts each file to numpy. It supports
  just these ONNX ops: `Sub` (normalizer mean), `Div` (normalizer std), `Gemm`
  (alpha = beta = 1) and `Elu` (alpha = 1).
- Anything else (LayerNorm, Tanh, recurrent layers, history buffers) needs
  `NumpyMLP` in `chain_controller.py` and `onnx_to_npz.py` extended, plus a
  re-run of the selftest. A history or recurrent policy also needs the
  observation builders changed.
- The current networks are 4-layer ELU MLPs. The exporter is
  `training/scripts/export_chain_v2.py` in this repo (run inside mjlab with the
  `training/` overlay, see the top-level README; examples in
  `deploy/VERSIONS.md`).

### 3.8 What else the controller reads from the bundle
- **`chain.json`:** `version` (must be 2), `control_hz`, `joints`,
  `motor_index`, `pd` (kp, kd, peak_nm per joint), `joint_limits`,
  `braced_joints`, `stand_pose`, `crawl_pose`, `crawl_pelvis_above_limbs_m`,
  `policies.{walk,stand_crawl,crawl}` (file, input_dim, action_clip, scale /
  action_scale, keyframes, command_ranges) and `crawl_variants` (optional; the
  first one is the default style).
- **`mujoco/scene.xml`:** used for forward kinematics. Keep these names:
  `robot/pelvis`, `robot/{left,right}_ankle_roll_link`,
  `robot/{left,right}_forearm_extension`, contact capsules named `*_collision`
  containing `foot` or `forearm_tip`, hinge joints at body origins, and the
  `robot/imu_in_pelvis` site (the bridge model's IMU).

---

## 4. Making a new bundle from training

1. **Export** with `training/scripts/export_chain_v2.py` into
   `bundles/<new_name>/` (or into `deploy/<new_name>/` and pass `--bundle`
   everywhere below): the ONNX files, `chain.json` and `mujoco/` (scene.xml +
   assets). Use v3 as the template, and keep everything in §3 the same unless
   you change `g1_config.py` with it.
2. **Copy the runtime files** from `bundles/g1_chain_v3/`:
   `chain_controller.py`, `sim2sim_mujoco.py`, `replay_log.py`.
3. **Numpy weights and selftest** (any env with onnx + onnxruntime):
   `python controller/tools/onnx_to_npz.py bundles/<new_name>`. Every policy
   must print `max |numpy − onnxruntime|` below 1e-4.
4. **If the robot model changed**, rebuild the bridge model:
   `python controller/tools/make_sim_model.py --bundle bundles/<new_name>`
   (needs MuJoCo ≥ 3.5 to read the training scene).
5. **Run the tests in §5** against it (`--bundle bundles/<new_name>`).
6. **Mode A on the gantry first**, then mode B: `python
   controller/tools/make_onboard_package.py --bundle bundles/<new_name>`.
7. Never overwrite a bundle that has been on the robot. Make a new folder, so
   rolling back is `--bundle bundles/g1_chain_v3`.

### 4.1 The repo's v3.1 bundle
`deploy/g1_chain_v3_1` (2026-10-31) is v3 with a new `stand_crawl.onnx`. It
was hardened in simulation for 40 ms lag and weak or heavy robots. Its other
ONNX files and `mujoco/` are byte-identical to `bundles/g1_chain_v3`, and so
are every observation, action, gain and joint limit. It passes the
controller's start-up checks.

What differs is the STAND_HOLD → WALK hand-over:

| | Hand-over condition |
|---|---|
| v3.1 (its README) | upright (gravity_z < −0.94) and tall (pelvis > 0.66 m) for 0.3 s, **without** waiting for the robot to be still |
| `chain_controller.py` here (v3) | gravity_z < −0.97, pelvis > 0.70 m, still, for 0.5 s |

The relaxed hand-over is not in the robot controller yet. To run v3.1 on the
robot:
1. Add it to `chain_controller.py` as an option.
2. Copy that file, `sim2sim_mujoco.py` and `replay_log.py` into a copy of the
   bundle, and run `onnx_to_npz.py`.
3. Pass every test in §5, then go to the gantry.

---

## 5. Testing before the robot (in this order)

All of these are offline or loopback-only. Commands run from
`real_world_implementation/` with the laptop env (§6.1) unless noted.

| # | Test | Command | Pass when |
|---|---|---|---|
| 1 | Config matches bundle | `python controller/g1_config.py` | `chain.json check: OK` |
| 2 | Policies, timing, SDK | `python controller/check_robot.py` | all `[OK]` except the robot network when no cable is plugged in |
| 3 | CPU sim-to-sim suite | `python bundles/<b>/sim2sim_mujoco.py --test all` | `moving`, `hold`, `cycles`, `switch` all PASS (about 2 min on a laptop CPU) |
| 4 | Robustness | `python bundles/<b>/sim2sim_mujoco.py --auto --delay-steps 4 --push 0.3` | PASS |
| 5 | Interactive look | `python bundles/<b>/sim2sim_mujoco.py` (viewer; W/S/A/D/Q/E, C down, U up, V style) | looks like training |
| 6 | DDS loopback, laptop settings | two terminals, see below | full loop, no damping stop, policy at 50 Hz |
| 7 | DDS loopback, onboard settings | same with the onboard package (see below) | same, `pol 50Hz` in the status |

**What the suite checks:**
- **moving:** crawls at 0.2, 0.35 and 0.5 m/s, sideways and turning, then
  stands up.
- **hold:** the crawl gate opens within 3.5 s, and the hold and the crawl at
  rest stay ≤ 0.7× each motor's continuous rating.
- **cycles:** 5 × (walk 30 s, down, crawl 0.3 m/s for 60 s, up).
- **switch:** creepy ↔ scuttle.

It reports each motor's load as RMS torque divided by its continuous rating
(the heat model: `training/src/mjlab/tasks/g1_motor_safety/motor_model.py`).

DDS loopback, mode A settings (the bridge only runs on `lo`, DDS domain 1, so
it can never reach the robot):
```bash
cd controller
python sim_bridge.py --headless --lower-at 6 --release-at 12 --duration 74 &
python g1_chain_real.py --sim --duration 68 \
  --script '1:1,7:2,13:w,13.3:w,13.6:w,16:q,18:x,22:c,36:w,36.3:w,39:x,42:u,58:w,58.3:w,58.6:w,61:x'
```
Mode B settings (built package, fake Unitree remote from the bridge):
```bash
python controller/tools/make_onboard_package.py
cd build/g1_onboard
python ../../controller/sim_bridge.py --headless --lower-at 6 --release-at 12 --duration 80 \
  --remote-script '1:Start,7:R1+X,13:ly=0.6,17:ly=0,22:R1+Down,37:ly=0.5,41:ly=0,44:R1+Up,62:ly=0.6,65:ly=0,70:Select' &
python g1_chain_real.py --sim --wrists braced --remote --loop-hz 100 --poll-state --duration 76
```
Drop `--headless` from the bridge to watch it. Use `--heat MOTOR:RATE:START:SECONDS`
on the bridge to test the thermal guard (a fake heating motor). Each run saves a
`chain_sim_*.npz` log.

---

## 6. Mode A: laptop over Ethernet

### 6.1 Laptop environment (once)
Use Python 3.10 (tested: conda env `asap_deploy`, numpy 1.26, onnxruntime
1.20, MuJoCo 3.3.4):
```bash
pip install -r requirements-laptop.txt
pip install --no-deps -e controller/third_party/unitree_sdk2_python
```
`--no-deps` skips the SDK's opencv requirement, which this code doesn't use.
CycloneDDS **0.10.2** is the version Unitree's SDK and the robot use.

### 6.2 Network (once per laptop)
Plug the cable into the robot. Give the laptop's wired interface a free
address on 192.168.123.0/24 (for example 192.168.123.26/24, no gateway). Check
it, read-only (publishes nothing):
```bash
python controller/check_robot.py
python controller/g1_chain_real.py --real --net enp3s0 --monitor
```
`--monitor` prints the state rate (about 1040 Hz), `mode_machine`, gravity,
joints and temperatures. Tilt the robot forward: gravity x goes positive.

### 6.3 Run (operator procedure)
1. Robot **hanging on the gantry**, braces and extensions on, waist unlocked,
   someone ready to cut power.
2. **Gantry check without policies** (first time with a new bundle). This
   moves the PD targets along the 8 keyframes:
   `python controller/g1_chain_real.py --real --net enp3s0 --wrists braced --schedule-only`
3. **Run:**
   `python controller/g1_chain_real.py --real --net enp3s0 --wrists braced`.
   - Type `yes` (checklist) and `release` (the robot goes limp, in DAMP).
   - `--dry-run` computes the policies without applying them.
   - `--remote` adds the Unitree remote mapping from §7.3.
4. `1` stands up slowly to the stand pose (3 s), still hanging.
5. **Lower the gantry until the feet carry the weight.** Then `2` starts the
   policies; the walk balances the robot. Slack the harness.
6. Drive:
   - `w`/`s` forward ±0.1, `a`/`d` sideways, `q`/`e` turn; each press adds to
     the command, `x` zeroes it. The walk stands still below about 0.25 m/s
     forward.
   - `c` goes down to the crawl (the harness must be slack enough for a prone
     robot, pelvis at 0.22 m). The same keys drive the crawl.
   - `v` switches crawl style; `h` cycles the step height.
   - `u` stands up again.
   - `g` passes a gate the controller is waiting on. Only use it when the robot
     is visibly settled.
7. **Space** (or `0`) = damping at any time. Ctrl+C quits and saves the log.

The status line (twice a second) shows:
- the mode and phase, and the policy rate (`pol 50Hz`);
- the command and caps;
- what a gate is waiting for;
- the hottest motor;
- the guard state.

---

## 7. Mode B: onboard on the G1's computer, Unitree remote

### 7.1 Why it differs
PC2 runs Ubuntu 20.04 on ARM (aarch64), Python 3.8, numpy 1.17, has **no
internet**, and should not get system-wide installs. So the package:
- runs the policies in numpy (`.npz` from `onnx_to_npz.py`, checked against
  onnxruntime to 3e-5);
- copies Unitree's SDK into `vendor/`;
- compiles the CycloneDDS Python bindings into `vendor/`, against the
  CycloneDDS 0.10.2 C library Unitree already installed at
  `~/cyclonedds_ws/install/cyclonedds`.

Nothing outside `~/g1_onboard` changes.

**The loop must run at 100 Hz with `--poll-state`.** The laptop's default
(500 Hz, every one of ~1000 state messages a second decoded in Python) made the
policy run at 38 Hz there, and the legs went wild. `run_robot.sh` sets both.

### 7.2 Build, copy, check (from `real_world_implementation/` on the laptop)
1. Build the package:
   ```bash
   python controller/tools/make_onboard_package.py   # → build/g1_onboard (6 MB)
   ```
2. Copy it to the robot (laptop on the cable; a person types the robot's
   password):
   ```bash
   scp -r build/g1_onboard unitree@192.168.123.164:~/
   ```
3. On the robot (`ssh unitree@192.168.123.164`), the first time only:
   ```bash
   cd ~/g1_onboard && ./build_vendor.sh
   ```
   This compiles the bindings into `vendor/` (about a minute) and runs
   `check_robot.py`, which must end with **READY** (control step about 1.7 ms).
4. Read-only check, which publishes nothing:
   ```bash
   python3 g1_chain_real.py --real --net eth0 --monitor --remote
   ```
   Press remote buttons and watch them in the output.

To update the package later: rebuild, then copy over `~/g1_onboard`, keeping
its `vendor/` (it holds the compiled bindings), or re-run `./build_vendor.sh`.

### 7.3 Run
```bash
cd ~/g1_onboard && NET=eth0 ./run_robot.sh            # extra options pass through, e.g. --max-crawl-fwd 0.35
```
Type `yes` and `release` (or add `--yes` once you trust the checklist), then
unplug the cable if you want. If ssh or WiFi drops, the controller keeps
running and the remote still works. `kill <pid>` stops it cleanly (damping,
log saved).

| Remote | Action |
|---|---|
| **B or F1** | **DAMPING (e-stop)** |
| Start | stand up slowly to the stand pose (hanging) |
| R1 + X | start the policies (stand / walk); feet must carry the weight first |
| left stick | forward / back / left / right (walk and crawl) |
| right stick | rotate (right = clockwise) |
| R1 + Down | stand → crawl |
| R1 + Up | crawl → stand |
| tap R1 / R2 | crawl step height +/− 0.02 m |
| L1 / L2 | speed cap +/− 0.1 m/s (walk or crawl, whichever runs; never past the trained range) |
| Select | quit: damping, log saved |

Hold R1, then press X / Up / Down. A tap of R1 on its own changes the step
height. The ssh terminal's keys still work too (Space = damping).

Copy the logs back with
`scp unitree@192.168.123.164:~/g1_onboard/logs/*.npz controller/logs/`. The
names use the robot's China-time clock.

---

## 8. Safety layer (both modes)

| Trigger | Action |
|---|---|
| No `rt/lowstate` for 0.1 s | damping |
| Tilt > 45° from where the current mode should have the pelvis (upright, schedule pitch, or prone) | damping |
| Any joint > 30 rad/s | damping |
| Any motor reading ≥ 85 °C | damping |
| 5 of the last 50 policy intervals > 26 ms | damping ("policy steps late") |
| Non-finite policy output | damping |
| Crawl side (HOLD/CRAWL/CRAWL_STOP) and a motor ≥ 72 °C, or rising > 1 °C/s over 5 s above 55 °C | stands up by itself (like `u`) |
| After such a stand-up | no new descent until every motor is < 60 °C |
| HOLD without the crawl gate for 8 s | stands up by itself |
| ≥ 65 °C | warning in the status |
| Operator: Space / 0 / remote B or F1 | damping |

"Damping" means kp 0 with the training kd on every motor, so the robot slumps
into the harness. The controller keeps sending damping for 1 s at exit, and
leaves the robot in low-level mode. Restart Unitree's controller (remote or
power cycle) to get the built-in modes back.

---

## 9. Logs (`controller/logs/chain_{sim,real}_YYYYmmdd_HHMMSS.npz`)

Recorded every policy tick (50 Hz) in every mode, saved at exit. The stamp is
the end time. `np.load(path)`; an example is in `example_logs/`.

| Field | Shape | Meaning |
|---|---|---|
| `t` | (T,) | monotonic time, s |
| `mode` / `modes` | (T,) / names | controller mode: DAMP, STANDUP, STAND, CHAIN, SCHEDULE |
| `chain_mode` | (T,) | −1 outside CHAIN; else WALK 0, STOPPING 1, DOWN 2, HOLD 3, CRAWL 4, CRAWL_STOP 5, UP 6, STAND_HOLD 7 |
| `joints` | (25,) | policy joint names |
| `q`, `dq` | (T, 25) | joint position / speed, policy order |
| `gravity`, `gyro` | (T, 3) | gravity unit vector and angular velocity, pelvis frame |
| `cmd`, `ccmd` | (T, 3) | walk command; crawl command [fwd, lat, turn] |
| `target` | (T, 25) | PD targets actually sent |
| `action` | (T, 25) | the active policy's clipped action |
| `tau_est` | (T, 25) | motor torque estimate, policy order, Nm |
| `phase` | (T,) | φ |
| `above_limbs`, `pelvis_height`, `gate_ok` | (T,) | forward-kinematics heights (m, NaN outside CHAIN) and the gate |
| `temp` | (T, 29, 2) | both temperature readings of every motor, motor order, °C |
| `tau_motor`, `q_motor`, `dq_motor` | (T, 29) | every motor, motor order (incl. braced wrists) |
| `temp_max` | (T,) | hottest reading |
| `guard` | (T,) | thermal guard: ok / warn / stand_up / damp |
| `style` | (T,) | active crawl style |

Logs from before 2026-10-30 lack the per-motor fields.

Tools:
- `python bundles/g1_chain_v3/replay_log.py <log> --start 30 --end 100 --ghost --video out.mp4`
  rebuilds the motion in MuJoCo (posture exact; position from leg odometry)
  with the PD targets as a blue ghost.
- `python analysis/plot_motors.py <log>` and `python analysis/plot_session.py
  <logs…>` plot torque, temperature and estimated current against each motor's
  limits.

These logs are the data for fitting the real motor heat model and for
comparing sim against real.

---

## 10. What the hardware taught us (feed this back into training)

Measured on the robot, 2026-10-29 (v2) and 2026-10-30 (v3). Details are in
`docs/REPORT_2026-10-29_first_robot_test.md` and the bundle's README.

**Heat is the binding constraint, and it comes from torque held at a
standstill.**
- v2 resting in the crawl held the right shoulder roll at 8.8 Nm (1.29× its
  ~6.8 Nm continuous rating) for 29 s. Its target sat on the joint-limit clamp
  and the arm stayed 0.6 rad short. It went 51 → 85 °C, hit the stop and
  collapsed.
- A second v2 descent stuck nose-down with a knee at its end stop (50 Nm,
  1.38×), 52 → 85 °C in 33 s.
- v2's only torque terms were `joint_torques_l2` (−1e-4) and a margin that
  charged torque above 80% of peak, so neither cost anything at 35% of peak.
- v3 added sustained-heat terms (2 s average of (τ/τ_cont)², budget 0.6²), a
  shoulder-roll and a waist-yaw term, the target clamp, and a three-limbs-down
  "creepy" gait. On the robot its shoulders rest at about 3.1 Nm (0.46×), and
  the hottest motor stayed at 46–56 °C through up to 148 s of crawling.
- The operator later ran about 2 h of continuous crawling with repeated
  stand ↔ crawl cycles in mode B. That log has not been analysed yet.

**Things that are different on the robot (open):**

| Finding | Real | Sim | Suggestion |
|---|---|---|---|
| Stand-up (UP) hip-yaw tracking error | 1.26–1.62 rad, every stand-up; one v3 stand-up tipped over (log 09:47) | 0.94 rad | v3.1's stand ↔ crawl was hardened for this in sim (§4.1), not yet tried on the robot; otherwise randomize hip-yaw friction/armature and foot friction, penalize hip-yaw deviation in UP, or slow UP |
| Stand-up peak load | hip yaw 1.4–1.5× for 8 s | – | short, not a heat problem yet; watch it |
| Creepy crawl limb clearance | 1.5–2 cm; step-height command barely changes it | feet ~4 cm, hands ~2 cm | reward tracking of the step-height command more strongly; randomize floor compliance/friction |
| Left/right symmetry | right side 5–6 clear steps vs 17–19 on the left in one run | symmetric | check symmetry augmentation; mirrored commands |
| Crawl turn tracking | 0.11 rad/s for a 0.2 command | tracks | |
| Knee load resting in the crawl | v2: ~26 Nm | ~9 Nm | the model under-predicts knee load; verify with the 50 Hz logs |
| Walk at low speed | stands still below ~0.25 m/s forward | | dead zone in the hardened walk |
| Walk ankle temperature | ankles are now the hottest motors (61–70 °C while walking/standing) | | the walk was never in the thermal fine-tune; add the heat terms |
| Motor heat model | shoulder heated as modelled; a knee heated ~2× faster | stand-in parameters | fit the model to the per-motor temperature and torque logs |
| Policy timing | 38 Hz made the walk unstable within ~1 s | | keep 50 Hz exact; randomize delay 0–20 ms |
| State dropout | a lost state stream (10-29, ~16:57) meant damping, then a fall | | nothing for training; operator procedure |

**Do not change silently:**
- the observation layouts, clips, scales and poses;
- the joint and motor order;
- the gains;
- the braced-wrist setup;
- the target clamp;
- the 50 Hz rate;
- the scene.xml names in §3.8.

The controller asserts most of them, and changing one means updating
`controller/g1_config.py` and re-running §5.

---

## 11. Acceptance checklist for a new bundle

- [ ] §5 tests 1–7 pass with `--bundle bundles/<new>` (and the onboard package built from it).
- [ ] Sim-to-sim hold test: crawl gate < 3.5 s; hold and rest ≤ 0.7× continuous on every motor.
- [ ] Sim-to-sim with `--delay-steps 4 --push 0.3` passes.
- [ ] `onnx_to_npz.py` errors < 1e-4 for every policy; `selftest.npz` regenerated.
- [ ] Gantry: `--schedule-only` moves through the keyframes without collisions.
- [ ] Mode A first, on the gantry, with someone on the e-stop. Then a short mode B run.
- [ ] Logs copied back and plotted (`analysis/plot_session.py`). Hottest motor, stand-up tracking and limb clearance compared with §10.

---

## 12. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no rt/lowstate on <iface> after 5 s` | wrong `--net`, no 192.168.123.x address, cable, or robot not on |
| `--real needs --wrists …` | say what is fitted: `--wrists braced` (braces on) or `hold` (no braces; no crawl) |
| `bundle joint order / gains / … differs` (assert at start) | the bundle disagrees with `g1_config.py`; see §3 |
| Status shows `pol 38Hz`, then "policy steps late" | slow computer: `--loop-hz 100 --poll-state` (default in `run_robot.sh`) |
| A gate waits forever (`waiting: …` in the status) | the robot is not where the gate expects; HOLD stands up after 8 s; `g` passes a gate only when visibly settled |
| `c` refused: "thermal guard: no crawl until every motor is < 60 C" | walk or stand to cool |
| Robot limp after the program ends | expected (low-level damping); restart Unitree's controller from the remote or power cycle |
| MuJoCo viewer slow on a hybrid-GPU laptop | the scripts force the NVIDIA GPU (PRIME offload); `--igpu` disables that |
| `make_sim_model.py` fails to read scene.xml | needs MuJoCo ≥ 3.5 (contact sensors); run it with the training env |
