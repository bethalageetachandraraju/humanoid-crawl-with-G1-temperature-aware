# g1_chain_v3_1: v3 with a hardened stand <-> crawl (2026-10-31)

After v3, the robot needed a hand to finish standing up. In this bundle only
`stand_crawl.onnx` is new. `walk.onnx`, `crawl.onnx` (= `crawl_creepy.onnx`),
`crawl_scuttle.onnx` and `mujoco/` are byte-identical to `g1_chain_v3/`. Every
input, output, gain and joint limit is the same.

| File | Policy | Checkpoint (`logs/rsl_rl/`) |
|---|---|---|
| `stand_crawl.onnx` | stand <-> crawl, **new** | `g1_stand_to_crawl_29dof/2026-10-31_09-27-40_robust_v5/model_35197.pt` |
| `walk.onnx` | walk, as v3 | `g1_walk_29dof/2026-10-23_06-41-09_walk_hard_track/model_55499.pt` |
| `crawl.onnx` | creepy crawl, as v3 | `g1_crawl_29dof/2026-10-30_03-24-17_creepy_hard/model_59499.pt` |
| `crawl_scuttle.onnx` | scuttle crawl, as v3 | `g1_crawl_29dof/2026-10-30_03-25-17_scuttle_hard/model_59299.pt` |

## Two controller changes go with it

1. **Relaxed stand hand-over.** STAND_HOLD -> WALK once the robot is upright
   (gravity_z < -0.94) and tall (pelvis over the feet > 0.66 m) for 0.3 s. Don't
   wait for it to be still: the walk policy settles the stand better than the
   transition policy at phi 0.
2. **HOLD timeout, 8 s.** If the crawl gate hasn't passed 8 s into HOLD, go UP
   anyway. This was already in v3's DEPLOY.md.

`scripts/sim2sim_chain_cpu.py --relaxed-handover --hold-timeout 8` implements
both; `scripts/play_chain.py --relaxed-handover` implements the first.

## What was wrong with v3's stand-up (simulation)

`scripts/standup_stress.py` runs the whole loop with the deploy target clamp and
8 s transitions. Each group of robots gets one fixed perturbation, most of them
beyond the training ranges.

v3 survives every single factor on its own: motors at 50 %, +8 kg, centre of mass
8 cm off, joint friction 1.5 Nm, gains x0.7, carpet twist, 0.4 m/s pushes. It
breaks in two places:
- **40 ms of motor command lag.** The robot tilts over in STAND_HOLD and only
  45 % stand. v2 does the same.
- **Several factors at once** (weak + heavy + slippery + lag): 48 % stand.

From the 2026-10-29 robot logs:
- The real lag is under 20 ms: the motor torque matches the previous tick's
  target.
- The legs came in from the sprawl against about 1.5x the simulated resistance
  (hip yaw at 0.69 vs 0.54 of peak).
- The knees folded ~0.3 rad past target, which looks like a softer robot than
  the model.

## Training: StandCrawl-Robust -> Robust2 -> Robust5

Continued from v3's `push_v1/model_30200` with wider margins:
- motor command lag 5-40 ms (was 0-20);
- motors 65-100 % (was 75-100 %);
- payload -3..+6 kg and torso centre of mass +-6 cm;
- motor stiffness and damping drawn once per robot (0.7-1.3) times a small
  per-motor draw, so whole-robot-soft worlds are trained;
- joint friction up to 1.5 Nm, arm ends -0.05..+0.5 kg;
- carpet contacts with twist friction, floor friction 0.25-1.4;
- shoves every 2-4 s, and the heat terms;
- smooth rewards for the hand-over states: base still at the top, crawl height
  and orientation at the bottom;
- 35 % of episodes start on all fours. Half of those start in a recorded creepy
  or scuttle hand-over pose (`scripts/record_crawl_stops.py`), the state the
  chain actually hands over in.

## Results (mjlab, 8 s transitions, creepy crawl at 0.3 m/s, relaxed hand-over)

Share of robots walking again after the whole loop (64-96 robots per row):

| World | v3 | v3.1 |
|---|---|---|
| nominal; 0.4 m/s pushes; slippery floor 0.25 | 100 % | 100 % |
| motors 60 % / +8 kg / all gains x0.7, x0.8 or x1.3 | 100 % | 100 % |
| carpet, friction 1.0-1.5 with twist (also with 80 % motors and +3 kg) | 100 % | 100 % |
| 40 ms lag | 45-70 % (22-33 % fall) | 93 % (1 % fall) |
| front-heavy (+2 kg, COM +4 cm, arm ends +0.5 kg, 85 % motors, 20 ms) | 81-85 % | 94 % (no falls) |
| combined heavy (70 % motors, +5 kg, gains 0.85, friction 0.35, 20-40 ms, pushes) | 44-58 % | 95 % (failures in the crawl) |

v3's ranges cover runs with the strict and the relaxed hand-over.

The hard world with 0.3 m/s pushes, 1024 robots:

| | v3 | v3.1 |
|---|---|---|
| failures | 4.0 % | 3.8 % |
| walking again | 95.2 % | 96.1 % |
| brace touches in HOLD | 13 | 3 |

Heat while holding the crawl pose:
- busiest motor: 0.50x its continuous rating (v3 0.64x);
- every robot stays within the 0.6 budget (v3: 27 % do);
- waist pitch through the transitions: 0.77x (v3 0.90x).

## CPU MuJoCo sim-to-sim of this bundle

`scripts/sim2sim_chain_cpu.py` runs the ONNX files on `mujoco/scene.xml` with
the deploy clamp and gates from the IMU, encoders and forward kinematics. With
the relaxed hand-over and the HOLD timeout, all 14 runs pass:
- nominal, and 1 kHz explicit PD;
- lag of 20, 30 and 40 ms (40 ms twice, once with pushes), and 40 ms with 1 kHz PD;
- 0.5 m/s pushes;
- 70 % motors with +5 kg (a slower first step: 2.18 m in 10 s vs v3's 2.43 m);
- 6 s transitions, crawl at 0.5 m/s, and scuttle (with 1 kHz PD too).

v3 failed all three 40 ms runs.

## Alternates

- `robust_v2/model_33698` (Robust2): the same world without the hand-over
  starts and per-robot stiffness. 4.7 % hard-loop failures, 27 brace touches in
  HOLD.
- `robust_v3` and `robust_v4`: best hard loop (2.4 %), but the pelvis sits down
  in the hold when every motor is 30 % softer. Not for the robot.

All are listed in `deploy/VERSIONS.md`.
