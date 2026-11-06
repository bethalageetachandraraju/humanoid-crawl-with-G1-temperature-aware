# Sim-to-real: the chain on the real G1

Runs the trained walk ↔ crawl policies on the real Unitree G1 (29-DoF, wrist
braces with forearm extensions). It needs numpy and Unitree's DDS stack, and no
mjlab or torch. One controller (`controller/g1_chain_real.py`), two ways to run
it:

| | A. Laptop over Ethernet | B. Onboard, Unitree remote |
|---|---|---|
| Runs on | laptop wired to the robot | the G1's computer (PC2, `unitree@192.168.123.164`) |
| Driven by | keyboard (or `--remote`) | Unitree remote, no cable |
| Policies | ONNX or numpy | numpy `.npz` only |
| Motor loop | 500 Hz | 100 Hz, `--poll-state` |

`bundles/g1_chain_v3/` is the bundle that ran on the robot. The repo's
`deploy/g1_chain_v3_1` has the same interface (see [New bundle](#new-bundle)).

## Folder

```
controller/   g1_chain_real.py (modes, keys, remote, thermal guard, logging), g1_walk_real.py (DDS I/O,
              safety stops), g1_config.py (joints, motors, gains, limits), thermal_guard.py, check_robot.py,
              sim_bridge.py (MuJoCo robot over DDS, loopback only), tools/ (onnx_to_npz, make_onboard_package,
              make_sim_model), third_party/unitree_sdk2_python, vendor_src/ (CycloneDDS bindings for the robot)
bundles/      g1_chain_v3: chain.json, ONNX + .npz policies, chain_controller.py, sim2sim_mujoco.py,
              replay_log.py, mujoco/scene.xml
analysis/     plot_motors.py, plot_session.py (torque / temperature per motor from a log)
example_logs/ a real v3 run: walk -> crawl -> stand -> walk
docs/         report of the first robot test (2026-10-29)
```

## A. Laptop over Ethernet

```bash
pip install -r requirements-laptop.txt                                # Python 3.10
pip install --no-deps -e controller/third_party/unitree_sdk2_python
# wired interface on 192.168.123.x/24, e.g. 192.168.123.26
python controller/check_robot.py                                      # read-only: SDK, network, policies, timing
python controller/g1_chain_real.py --real --net enp3s0 --monitor      # read-only: state rate, IMU, temperatures
python controller/g1_chain_real.py --real --net enp3s0 --wrists braced --schedule-only   # gantry: keyframes, no policy
python controller/g1_chain_real.py --real --net enp3s0 --wrists braced                   # run
```

Type `yes`, then `release`: the built-in controller is released and **the
robot goes limp**, so it must hang on the gantry. Then:

1. `1` stands up slowly (hanging).
2. Lower the gantry until the feet carry the weight.
3. `2` starts the policies.
4. Drive:
   - `w`/`s`, `a`/`d`, `q`/`e` add ±0.1 to forward, sideways and turn; `x` zeroes.
   - `c` goes down to crawl, `u` stands up.
   - `v` switches crawl style, `h` changes step height.
5. **Space = damping** at any time. Ctrl+C quits and saves the log.

## B. Onboard, Unitree remote

PC2 runs Ubuntu 20.04 aarch64 with Python 3.8 and has no internet. The package
installs nothing system-wide: the policies run in numpy, and Unitree's SDK and
the CycloneDDS bindings live in `~/g1_onboard/vendor`.

```bash
python controller/tools/make_onboard_package.py          # laptop: builds build/g1_onboard
scp -r build/g1_onboard unitree@192.168.123.164:~/       # laptop on the cable
# on the robot:
cd ~/g1_onboard && ./build_vendor.sh                     # first time only; must end with READY
NET=eth0 ./run_robot.sh                                  # 100 Hz, --poll-state, remote
scp unitree@192.168.123.164:~/g1_onboard/logs/*.npz controller/logs/   # laptop: logs back
```

| Remote | Action |
|---|---|
| **B / F1** | **damping (e-stop)** |
| Start | stand pose (hanging) |
| R1 + X | start the policies (feet carrying the weight) |
| left stick / right stick | move / rotate (right = clockwise) |
| R1 + Down / R1 + Up | stand → crawl / crawl → stand |
| tap R1 / R2 | step height ± 2 cm |
| L1 / L2 | speed cap ± 0.1 m/s |
| Select | quit (damping, log saved) |

Keep 100 Hz with `--poll-state` (`run_robot.sh` sets both). At 500 Hz the
robot computer ran the policy at 38 Hz and the legs went wild. If ssh drops,
the controller keeps running.

## Test before the robot

All of these are offline or on loopback (DDS domain 1, which can't reach the
robot):

| Test | Command | Pass |
|---|---|---|
| config vs bundle | `python controller/g1_config.py` | `chain.json check: OK` |
| readiness | `python controller/check_robot.py` | all OK (network fails without a cable) |
| sim-to-sim suite | `python bundles/g1_chain_v3/sim2sim_mujoco.py --test all` | moving, hold, cycles, switch PASS (about 2 min) |
| lag + push | `python bundles/g1_chain_v3/sim2sim_mujoco.py --auto --delay-steps 4 --push 0.3` | PASS |
| DDS loopback | `cd controller && python sim_bridge.py --lower-at 6 --release-at 12` (one terminal), `python g1_chain_real.py --sim` (another) | full loop, `pol 50Hz` |

In the loopback test:
- Press `1` at once and `2` after the bridge lowers the robot (6 s).
- For the onboard settings, add `--remote-script
  '1:Start,7:R1+X,22:R1+Down,44:R1+Up'` to the bridge and `--remote --loop-hz
  100 --poll-state` to the controller.
- The bridge's `--heat MOTOR:RATE:START:SECONDS` fakes a hot motor to test the
  thermal guard.

## Contract with training

`chain.json` holds all of this, and the controller checks it against
`controller/g1_config.py` at start-up (a mismatch refuses to load). If you
change any of it on purpose, change `g1_config.py` and re-run the tests.

- **Rate:** policies run at exactly 50 Hz (physics 5 ms × 4). Late steps
  destabilize the walk. Keep 0–20 ms lag randomization in training.
- **Joints:** 25 in MuJoCo order. Policy 0–19 → motors 0–19, policy 20–24 →
  motors 22–26. Wrist pitch/yaw motors 20, 21, 27, 28 are braced (kp 0, kd 0.5).
- **Gains:** the driver does `τ = kp(q* − q) − kd·q̇`.

  | Joints | kp | kd | Peak Nm |
  |---|---|---|---|
  | hip pitch / yaw, waist yaw | 40.18 | 2.56 | 88 |
  | hip roll, knee | 99.10 | 6.31 | 139 |
  | ankles, waist roll / pitch | 28.50 | 1.81 | 50 |
  | shoulders, elbow, wrist roll | 14.25 | 0.91 | 25 |

- **Observations:** raw SI units, one frame, normalizer baked into the ONNX.
  - `g` = gravity in the pelvis frame, from the IMU quaternion.
  - `ω` = IMU gyro (rad/s).
  - `last` = the policy's own clipped output from the previous tick (shadow
    mode: every policy runs every tick).

  | Policy | Dim | Layout |
  |---|---|---|
  | walk | 84 | `g, ω, [vx vy wz], q − stand, qd, last` |
  | stand ↔ crawl | 108 | `g, ω, q − stand, qd, last, [φ, dφ/dt], ref(φ) − q` |
  | crawl | 88 | `g, ω, [fwd lat turn step_h 0 0 0], q − crawl, qd, last` |

- **Targets:**
  - walk: `stand + 0.25·peak/kp · clip(a, ±5)`;
  - stand ↔ crawl: `ref(φ) + 0.5·clip(a, ±1.6)`;
  - crawl: `crawl + 0.5·clip(a, ±5)`.

  **All are then clamped to the joint ranges. Train with that clamp.** v2
  didn't, and its shoulder sat on the clamp at 8.8 Nm until the 85 °C stop.
- **Commands:**
  - Walk: vx −0.6…1.0, vy ±0.5, wz ±0.8.
  - Crawl (creepy): fwd −0.2…0.5, lat ±0.25, turn ±0.7, step height
    0.02–0.15. **Crawl turn > 0 is clockwise.**
  - φ: 0 = stand, 1 = crawl. It moves at ±1/T, with T = 8 s on the robot.
- **Gates** (IMU, joint speeds, numpy forward kinematics):
  - The crawl gate needs the pelvis 0.151 ± 0.03 m above the limbs, all four
    limbs down, still, and gravity within 0.15 of the crawl pose's.
  - HOLD stands up after 8 s without it.
  - The stand gate (v3) is gravity_z < −0.97, pelvis > 0.70 m, still, for 0.5 s.
- **Export:** one ONNX per policy, input `obs` `[1, N]`, output `[1, 25]`.
  Stateless MLP using only the ops `Sub`, `Div`, `Gemm` and `Elu` (that's all
  `tools/onnx_to_npz.py` converts for the robot computer).
- **scene.xml names used by the forward kinematics:** `robot/pelvis`,
  `*_ankle_roll_link`, `*_forearm_extension`, `*_collision` capsules with
  `foot` / `forearm_tip`, and the `imu_in_pelvis` site.

## New bundle

1. Export with `training/scripts/export_chain_v2.py` into `bundles/<name>/`
   (or `deploy/<name>/`, then use `--bundle`).
2. Copy `chain_controller.py`, `sim2sim_mujoco.py` and `replay_log.py` from
   `bundles/g1_chain_v3/` into it.
3. Run `python controller/tools/onnx_to_npz.py bundles/<name>`. Every policy's
   error must be under 1e-4.
4. Run the tests above with `--bundle bundles/<name>`, then the gantry, then
   mode A, then mode B.
5. Never overwrite a bundle that has run on the robot. Rolling back is
   `--bundle bundles/g1_chain_v3`.

**v3.1** (`deploy/g1_chain_v3_1`) is v3 with a new `stand_crawl.onnx` and a
relaxed stand hand-over: gravity_z < −0.94 and pelvis > 0.66 m for 0.3 s,
without waiting for the robot to be still. It passes the controller's start-up
checks. Its relaxed hand-over isn't in `chain_controller.py` yet; add it before
running v3.1 on the robot.

## Safety

| Trigger | Action |
|---|---|
| no state for 0.1 s · tilt > 45° from the mode's expected pose · joint > 30 rad/s · motor ≥ 85 °C · 5 of 50 policy steps > 26 ms · non-finite output | damping |
| crawl side and a motor ≥ 72 °C, or rising > 1 °C/s above 55 °C | stands up by itself; no new descent until every motor < 60 °C |
| HOLD without the crawl gate for 8 s | stands up by itself |
| Space / 0 / remote B or F1 | damping |

Damping is kp 0 with the training kd: the robot slumps into the harness. It
stays in low-level mode afterwards; restart Unitree's controller from the remote
or power-cycle it.

## Logs

`controller/logs/chain_{sim,real}_<end time>.npz`, 50 Hz, saved at exit. The
robot's clock runs 12 h ahead (China time).
- Fields:
  - `t`, `mode`, `chain_mode` (−1, or WALK 0 … STAND_HOLD 7);
  - `q`, `dq`, `target`, `action`, `tau_est` (policy order);
  - `gravity`, `gyro`, `cmd`, `ccmd`, `phase`, `above_limbs`,
    `pelvis_height`, `gate_ok`;
  - `temp` (29 motors × 2 readings, °C), `tau_motor`, `q_motor`, `dq_motor`
    (motor order);
  - `guard`, `style`.
- Replay in MuJoCo with the targets as a ghost:
  `python bundles/g1_chain_v3/replay_log.py <log> --ghost --video out.mp4`
- Plot torque and temperature per motor: `python analysis/plot_session.py
  <logs>`.

## What the hardware showed (for training)

| Finding (robot, 2026-10-29/30) | Suggestion |
|---|---|
| Heat comes from torque held still. v2 resting in the crawl: shoulder 8.8 Nm (1.3× rating), 51 → 85 °C in 41 s. v3 (heat terms + clamp + creepy gait): shoulder 3.1 Nm, hottest motor ≤ 56 °C over 148 s of crawling | keep the heat terms and the clamp |
| Stand-up hip yaw 1.3–1.6 rad off target (sim 0.94); one v3 stand-up tipped over | v3.1's hardened stand ↔ crawl, untested on the robot |
| Creepy crawl lifts limbs 1.5–2 cm (sim ~4 cm); step-height command barely changes it | reward clearance tracking |
| Right side steps less than left; turns reach ~half the command | check symmetry |
| Walk ankles are now the hottest motors (61–70 °C) | add the heat terms to the walk |
| A knee heated ~2× faster than the motor model | fit the model to the `temp` / `tau_motor` logs |
| 38 Hz policy steps → unstable walk within ~1 s | keep 50 Hz exact, randomize lag |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no rt/lowstate … after 5 s` | wrong `--net`, no 192.168.123.x address, or robot off |
| `bundle … differs` at start-up | the bundle and `g1_config.py` disagree (see the contract) |
| `pol 38Hz`, then "policy steps late" | use `--loop-hz 100 --poll-state` |
| a gate waits forever | the robot isn't in the pose the gate expects; HOLD stands up after 8 s; `g` passes a gate only if visibly settled |
| `c` refused by the thermal guard | walk or stand until every motor is < 60 °C |
| robot limp after exit | expected; restart Unitree's controller |
