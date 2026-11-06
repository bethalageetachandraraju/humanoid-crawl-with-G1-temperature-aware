# Motor-safe G1 walking (proof of concept)

Can a G1 walking policy be kept safe against two failures at once, **falling** and
**per-joint motor overload**, by giving the simulator an identified motor model and
putting a motor safety filter in the loop? This package is the smallest experiment
that answers that question in simulation. Results: [`REPORT.md`](REPORT.md).

## What is in it

| File | Contents |
|---|---|
| `motor_model.py` | `ThermalDcMotorActuatorCfg`: mjlab's DC motor actuator (PD + torque-speed envelope, `dc_motor_clip`) extended with a two-node winding/housing heat model per joint, per-env plant parameters drawn around the identification, the G1 firmware overheat cut-off, quantized temperature readings, and the motor safety filter. Pure functions for the heat model and the filter bounds. |
| `mdp.py` | Observations (two temperature readings and thermal headroom per motor), the heat penalty, the filter-intervention penalty, metrics; the HJ motor margin `g(s)`. |
| `env_cfgs.py` | G1 flat velocity walking with a 0-8 kg torso payload; the three training arms; the HJ safety task. |
| `safety_ppo.py` | Hamilton-Jacobi safety learning on rsl_rl: `SafetyPPO` (PPO with the avoid Bellman backup from `safety_sb3.backups`), the pilot wrapper for handover states, the runner. |
| `__init__.py` | Tasks `Mjlab-MotorSafety-G1-{Ideal,Penalty,Filter}` and `Mjlab-MotorSafety-G1-{HjSafety,HjMotorOnly}`, experiment `g1_motor_safety`. |
| `../../../../scripts/motor_safety/sysid_thermal.py` | Identifies the heat model from joint logs (torque estimate, speed, the two temperature readings); `--synthetic` makes the logs from a hidden "true" motor. |
| `../../../../scripts/motor_safety/eval_motor_safety.py` | 8-minute cold-start evaluation on the realistic motors, payload groups, with or without the filter at deployment. |
| `../../../../scripts/motor_safety/plot_motor_safety.py` | Figures and tables. |
| `../../../../scripts/motor_safety/train_governor.py` | Learns the governor's command-conditioned safety value `V(s, c)` for a fixed walking policy + clip filter (policy evaluation with the avoid backup). |
| `../../../../scripts/motor_safety/eval_extreme.py` | The extreme test: every controller side by side in one simulation, patrol mission without rest, governor and derating arms. |
| `../../../../scripts/motor_safety/gpu_gate.sh` | Starts a GPU job only when >= 8 GB are free (the GPU is shared). |
| `../../../../scripts/motor_safety/run_hj_compare.sh` | Normal RL vs clip filter vs HJ safety filter on the cooled plant, room and hot room. |
| `../../../../scripts/motor_safety/render_motor_safety.py` | Videos, robots tinted by winding temperature; HJ rows show `V(s)` and when the fallback drives. |
| `../../../../tests/test_g1_motor_safety.py` | CPU tests: steady state = continuous rating, barrier invariance under model mismatch, envelope, copper loss. |

## The motor model

Per joint, every 5 ms physics step:

1. `tau_pd = kp (q* - q) - kd dq` (the G1 driver's PD, mjlab's gains).
2. Torque-speed envelope: stall torque 1.4 x peak at zero speed, falling linearly to
   zero at the no-load speed, capped at the driver current limit (peak torque).
3. Firmware cut-off: winding >= 105 C -> zero torque until it cools to 85 C.
4. Heat: `I = tau / (n gear kt)`, `P = R25 (1 + 0.00393 (T_w - 25)) I^2 + k_iron dq^2`,
   `C_w dT_w = P - (T_w - T_h)/R_wh`,
   `C_h dT_h = (T_w - T_h)/R_wh - G_ha (T_h - T_amb) + Q_drv + sum_nb G_c (T_h,nb - T_h)`.
5. Cooling (`cooling=True`, the default):
   - **Fan.** The G1's joint motors are air cooled. Each motor gets a fan whose duty
     follows its winding reading, off below 40 C and full at 60 C. At full duty it
     raises the housing-to-air conductance by `k_fan = 0.6` (x1.6).
   - **Forced convection while walking.** `G_ha` is also multiplied by
     `(1 + 0.3 |v_xy|)`.
   - **Driver heat.** The motor driver puts a constant 0.5-1.5 W into the housing.
   - **Neighbour coupling.** Housings in the same cluster exchange heat through
     0.15 W/K: per side, the three hip motors; knee and ankles (the ankle motors sit
     in the shank); the three shoulder motors; elbow and wrists; plus the waist.
   - Sources: the speed-dependent convection, driver heat and inter-motor coupling
     come from the lumped whole-body thermal network identified on a quadruped by
     Lin et al. (ROBOT 2025), which Wan et al. (arXiv 2605.27046) and Qian et al.
     (arXiv 2603.01631) use for thermal-aware RL. The fan curve is ours.
   - Every env draws its own `k_fan`, `k_vel`, `G_c` and `Q_drv` (+-20-30 %).

Motor classes (mjlab's G1 grouping; gears, peak torques, speeds, armatures and PD gains
are mjlab's constants; the electrical and thermal values are the "identification"):

| class | joints | gear | peak Nm | continuous Nm (25 C amb) | winding / housing time const |
|---|---|---|---|---|---|
| 7520-22 | knees, hip roll | 22.5 | 139 | 36 | 25 s / 7 min |
| 7520-14 | hip pitch, hip yaw, waist yaw | 14.3 | 88 | 23 | 25 s / 7 min |
| 5020 x2 | ankles, waist roll/pitch | 16 | 50 | 13.5 | 16 s / 5 min |
| 5020 | shoulders, elbows, wrist roll | 16 | 25 | 6.8 | 16 s / 5 min |
| 4010 | wrist pitch/yaw | 25 | 5 | 1.8 | 10 s / 3 min |

Every env draws its own true values within +-4..20 % of these (`PLANT_SPREAD`) and an
ambient of 20-30 C; the readings are quantized to 1 C.

## The safety filter

Clamps the PD torque each physics step to `[lo, hi]`:

* **Feasibility** (hard action constraint): 90 % of the identified torque-speed
  envelope, so the policy never relies on driver saturation.
* **Thermal barrier**: with `h = 90 C - T_w`, allow at most
  `P = C_w h / 2 s + (T_w - T_h)/R_wh - k_iron dq^2` of heat, evaluated at the worst
  corner of the identification's uncertainty box and the sensor quantization, i.e.
  `|tau| <= n gear kt sqrt(P / R(T_w))`. Then `h(t+dt) >= (1 - dt/2 s) h(t)`: no
  winding ever crosses 90 C for any plant in the box (tested in
  `test_barrier_keeps_winding_below_limit_under_model_mismatch`). Cold motors get
  their full peak torque; at the limit the bound is the continuous torque the housing
  can carry away, which shrinks as the housing heats.

The barrier only needs the winding node and the measured housing temperature, so the
cooling model does not change the filter. Fans and airflow cool the housing, the
filter sees that in `T_h`, and the bound is still sound.

The filter is myopic: it guarantees the motor, not the robot. Falling is the second
failure, and handling it is the policy's job: the filter arm is trained *inside* the
filter, observing the temperatures and the headroom, so it can learn gaits whose heat
the motors can sustain.

## The Hamilton-Jacobi safety filter (HJSF)

The clip filter above is a one-step rule: it never lets a winding cross 90 C, but it
does not know whether the torque it allows today leaves enough budget to keep walking
in a minute. HJ reachability asks the long-horizon question instead: *from this state,
can some policy keep every motor under its rating from now on?*

- **Failure set, motors only.** `g(s) = min_j (90 C - T_w,j) / 10 C`, clamped to
  +-3: `g < 0` as soon as any winding passes its rating. There is no tilt or pelvis
  height term in `g`.
- **Safety Bellman equation** (avoid; Fisac et al. ICRA 2019), backup from
  `safety_sb3.backups.avoid_target`:
  `V(s) = (1 - gamma) g(s) + gamma min(g(s), V(s'))`, `V(terminal) = g`.
  `V(s) >= 0` means the fallback keeps every motor safe from `s` on.
- **Learner.** `SafetyPPO`: rsl_rl's PPO with returns and advantages from that backup
  (safety_sb3's `TensorSafetyRolloutBuffer` recursion), no timeout bootstrapping,
  gamma 0.999 (about 20 s, the winding time constant). The env reward is `g` itself.
  Actor warm-started from the walking policy; the value net is fresh and reads only
  what the robot measures, so the same `V(s)` runs on the robot.
- **Handover states.** At deployment the fallback takes over mid-gait with hot motors.
  So in training a frozen task policy (the pilot, normal RL) drives each episode for a
  random 0-15 s first (70 % of episodes). The fallback learns only from what happens
  after the handover; pilot transitions carry no gradient.
- **Falls.** A fall ends the episode (the env's standard terminations), and the
  terminal step is scored `min(g, -1)`. Without that, the motor-safest thing is to go
  limp and drop to the floor (zero torque, no heat), and `V` would certify it.
  `Mjlab-MotorSafety-G1-HjMotorOnly` drops that rule, as an ablation.
- **The filter** (least restrictive, with hysteresis): the task policy acts while
  `V(s) > 0.2`. Below that, the fallback acts until `V(s) > 0.5`. `V` is in margin units,
  so 0.2 is 2 C of winding headroom the fallback is predicted to keep.

| | clip (CBF) filter | HJ safety filter |
|---|---|---|
| what it acts on | torque, every 5 ms physics step | which policy drives, every 20 ms |
| horizon | one step (barrier decay 2 s) | about 20 s of future (gamma 0.999) |
| model | identified heat model, worst case | learned from simulation on the identified plant |
| guarantee | hard, under the uncertainty box | learned: only as good as `V` |
| failure it can foresee | the next step's heat | a gait that will overheat in the next tens of seconds |

## The HJ reference governor (ours, v2)

HJSF v1 fails in practice. It swaps one walking policy for another mid-stride, and
those handovers knock the robot over: at room temperature 14 % of robots fall,
switching about 10 times a minute at 8 kg. Its value net was also trained almost
only on states the fallback drives, so it misreads states the task policy drives.

The governor keeps the HJ idea and drops the switching. Inspiration:
- **Explicit reference governors** (Garone, Di Cairano and Kolmanovsky, Automatica
  2017): never touch the controller; shape the *reference* it tracks so the closed
  loop stays inside a safe set.
- **Value-based shielding** from the HJ papers (Fisac et al. 2019; ISAACS;
  Gameplay Filters): the safe set is the zero super-level set of a learned avoid value.

How it works:
- **One value for the whole closed loop**: `V(s, c)`, the avoid value of "the
  walking policy, the clip filter and the real motors, holding command `c` from
  state `s` on". Failure is a fall; the clip filter already keeps every winding
  under 90 C, so the danger left is running the torque budget down until the robot
  falls. `train_governor.py` learns it by policy evaluation (no policy update)
  with the `safety_sb3` avoid backup, gamma 0.9995 (about 40 s of foresight):
  - Each robot holds a random command for 5-30 s, then draws a new one.
  - At a switch, the backup bootstraps from the next state with the old command
    still in it, so `V` is the value of *holding* `c`.
  - Training covers hot starts, 0-16 kg and rooms at 20-45 C.
- **The governor**: every 0.1 s, evaluate `V(s, alpha c_user)` for alpha in
  {1, 0.8, 0.6, 0.4, 0.2, 0} and send the largest alpha with `V >= eps` (and
  `>= eps_up` to raise alpha again; 0.3 / 0.6, picked by a threshold sweep). If nothing clears it, send
  the alpha with the highest value.
- **Smooth by construction**: the walking policy never changes, so slowing down or
  stopping is just another command, which it was trained to follow.
- **Layered**: the clip filter stays on as the hard per-step guarantee. The
  governor adds the foresight the clip filter lacks: it slows down or stops before
  the torque budget gets too tight to keep walking.

The baseline it has to beat besides the learned ones is **temperature derating**,
the obvious engineering rule: scale the command by `(T1 - T) / (T1 - T0)` on the
hottest winding reading.

## Known simplifications

* The electrical/thermal values are plausible, not measured on a G1. They set *when*
  motors overheat; the experiment's conclusions are about *what the filter does once
  they do*. A real identification (`sysid_thermal.py` on G1 logs) replaces them.
* The ankle and waist roll/pitch linkages are modelled as two independent motors per
  joint. On the G1 the same two motors serve pitch and roll, so their real heating is
  higher than modelled here (the ankles overheat first even so).
* One heat node per winding and per housing, coupled only within a cluster; no sensor
  lag. A real filter would run on an observer that also predicts the hot spot.
* The fan curve and the airflow gain are assumptions, not G1 data.
* Firmware behaviour after a cut-off is simplified to zero torque until 85 C.

## Run it

```bash
# base policy on ideal motors, then fine-tune each arm from it (same iterations)
uv run train Mjlab-MotorSafety-G1-Ideal --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.run-name base_ideal
for ARM in Ideal Penalty Filter; do
  uv run train Mjlab-MotorSafety-G1-$ARM --env.scene.num-envs 4096 --agent.max-iterations 1500 \
    --agent.resume True --agent.load-run <base run dir name> --agent.run-name ft_${ARM,,}
done
# evaluate (8 min, cold start, payload 0-8 kg)
uv run python scripts/motor_safety/eval_motor_safety.py --name ideal --checkpoint <ckpt>
uv run python scripts/motor_safety/eval_motor_safety.py --name ideal+filter --checkpoint <ckpt> --filter
uv run python scripts/motor_safety/plot_motor_safety.py --names ideal ideal+filter penalty filter
# HJ safety: value + fallback (needs safety_sb3: uv pip install -e ~/humanoid/safety-stable-baselines)
uv run train Mjlab-MotorSafety-G1-HjSafety --env.scene.num-envs 4096 --agent.max-iterations 2000 \
  --agent.resume True --agent.load-run 2026-10-24_21-36-42_base_ideal \
  --agent.load-checkpoint model_2999.pt --agent.run-name hj_safety
bash scripts/motor_safety/run_hj_compare.sh <hj_safety checkpoint>
uv run python scripts/motor_safety/sysid_thermal.py --synthetic
uv run pytest tests/test_g1_motor_safety.py
```
