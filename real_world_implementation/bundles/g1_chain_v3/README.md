# Chain v3: walk -> creepy / scuttle crawl -> walk (braced 29-DoF G1)

Built 2026-10-30 after the first robot test (2026-10-29), where two motors hit
the controller's 85 C stop while the robot held the crawl (right shoulder roll,
8.8 Nm for 29 s; a knee, ~50 Nm in a stand<->crawl hold stuck nose-down on
carpet). Same inputs, outputs, gains and `chain.json` layout as v2: a v2
controller runs v3 by swapping the ONNX files. `deploy/g1_chain_v2/` (the trot
crawl) is unchanged; `deploy/VERSIONS.md` lists every version and checkpoint.

| File | Policy | Checkpoint (`mjlab/logs/rsl_rl/`) |
|---|---|---|
| `walk.onnx` | hardened walk (unchanged from v2) | `g1_walk_29dof/2026-10-23_06-41-09_walk_hard_track/model_55499.pt` |
| `stand_crawl.onnx` | two-way stand <-> crawl, thermal + carpet fine-tune | `g1_stand_to_crawl_29dof/2026-10-30_05-29-27_push_v1/model_30200.pt` |
| `crawl.onnx` = `crawl_creepy.onnx` | **creepy ghost crawl** (default): slow, three limbs down | `g1_crawl_29dof/2026-10-30_03-24-17_creepy_hard/model_59499.pt` |
| `crawl_scuttle.onnx` | **scuttle**: fast crawl, 0.5-1 m/s | `g1_crawl_29dof/2026-10-30_03-25-17_scuttle_hard/model_59299.pt` |
| `mujoco/scene.xml` | the simulated robot | |

## What changed and why

- **Heat.** Every policy was fine-tuned with a sustained-heat penalty: each
  motor's torque over ~2 s against its continuous rating, budget 0.6x, plus
  dedicated terms on the waist yaw and shoulder rolls (the old trot twisted the
  torso on the waist yaw every step, 1.4x its rating; the old crawl held the
  shoulder rolls at 1.5x at rest).
- **The robot's target clamp is trained in.** PD targets are clipped to
  `chain.json`'s `joint_limits` during training, as the controller does. Keep
  the clamp: the policies expect it.
- **Carpet.** Feet and extension tips train with twist friction (condim 4) and
  floor friction up to ~1.5 (tile to carpet, per episode). On carpet physics the
  old stand <-> crawl reached the crawl pose 58 % of the time, this one 94-98 %.
- **Robustness.** The crawls trained at the chain test's hard randomization
  (+-20 % mass, -3..+5 kg payload, motors at 75-100 %, gains +-25 %, 1 m/s
  pushes, heavier arm ends); a torso / pelvis / brace touch ends an episode.

## Crawl styles: speeds and switching

| Style | Command ranges (fwd / lat / turn) | Notes |
|---|---|---|
| creepy | -0.2..0.5 m/s / +-0.25 / +-0.7 rad/s | stalk at 0.2-0.35; 0.5 reaches ~0.43 |
| scuttle | -0.3..1.2 m/s / +-0.4 / +-1.0 rad/s | 0.75 -> 0.70 m/s, 1.0 -> 0.87 m/s |

Both styles take the same observation and give the same action as `crawl.onnx`.
To switch while crawling: set the crawl command to 0, wait until the robot is
still (as the crawl-stop gate), swap the active policy and blend the targets
over 0.5 s; run every policy every tick on the live observation so each keeps
its own last action. A switch mid-stride jolts the robot (motor demand to
~100 % of peak, brace moments 8-11 Nm in simulation); at rest it is clean.
In mjlab: `scripts/play_chain.py --crawl-variants creepy=... scuttle=...`, V
switches.

## Simulation results (robot's target clamp, carpet contacts)

Busiest motor = its RMS torque / its estimated continuous rating
(`g1_motor_safety/motor_model.py`; 0.6x is the long-run budget, 1x the rating).
"Randomized" = the chain test's hard randomization plus pushes; falls are per
60 s robot. 256 robots per line unless noted.

**Creepy crawl**

| Scenario | Busiest motor | Speed | Falls |
|---|---|---|---|
| rest | 0.52x (knees) | - | 0 |
| 0.2 m/s | 0.53x | 0.23 | 0 |
| 0.35 m/s | 0.58x | 0.37 | 0.4 % |
| 0.5 m/s | 0.63x | 0.43 | 0 |
| 0.35 m/s, randomized | 0.70x | 0.33 | 0.8 % |
| 0.35 m/s, tile (floor 0.35-0.5) | 0.74x | 0.35 | 1.6 % |
| 0.35 m/s, carpet (floor 0.9-1.2) | 0.69x | 0.32 | 2.3 % |
| old trot at 0.3 m/s, for reference | 1.38x (waist yaw) | 0.32 | |

**Scuttle crawl**

| Scenario | Busiest motor | Speed | Falls |
|---|---|---|---|
| rest | 0.60x | - | 0.4 % |
| 0.5 m/s | 0.84x | 0.45 | 0 |
| 0.75 m/s | 0.92x | 0.70 | 0.4 % |
| 1.0 m/s | 0.97x (waist pitch) | 0.87 | 0.4 % |
| 0.8 m/s, randomized / tile / carpet | 1.02x / 1.10x / 1.02x | 0.76 | 2.7 / 4.3 / 3.1 % |

**Stand <-> crawl** (this bundle vs v2's)

| Scenario | v2 (stand_crawl_v3/24999) | v3 (push_v1/30200) |
|---|---|---|
| hold the crawl pose 20 s, busiest motor | 1.72x | 0.65x |
| descent / stand-up, busiest motor | waist pitch 1.10x | waist pitch 0.90x |
| carpet physics: reaches the crawl pose | 58 % | 96 % |
| stuck nose-down start: gate within 3 s | 88 % (no carpet) | 75 %, 89 % reached (carpet) |
| knee peak going down | 2.70 rad | 2.46 rad |

**Whole chain loop** (walk -> down -> hold -> crawl 10 s -> up -> walk)

| | normal | hard | hard + pushes 0.3 m/s |
|---|---|---|---|
| v2 (trot crawl) | 0 % | 3.1 % | 3.9-4.7 % |
| v3, creepy | 0 % | 2.3 % (none while crawling) | 2.7 % |
| v3, scuttle at 0.8 m/s | 0 % | 5.5 % | 8.6 % |
| v3, creepy <-> scuttle switched at rest every 6 s (64 robots) | | 6.2 % (none at a switch) | |

**Heat over a long session** (two-node motor model with fans, room 30 C, start
40 C; "x2" = copper loss doubled, the pessimistic case from the robot test)

| | hottest winding, model x1 | model x2 |
|---|---|---|
| 37 min: 5 x (walk 1 min, down, crawl 6 min, up), old trot chain | trips the 85 C stop | 283 C (model) |
| same, creepy (rests, 0.2-0.5 m/s) | 50 C | 64 C |
| same, scuttle (rests, 0.5-1 m/s) | 66 C | 107 C |
| creepy 0.5 m/s, 60 min non-stop | 51 C | 66 C |
| scuttle 1.0 m/s, non-stop | 70 C at 7.5 min, 74 C at 60 min | 85 C at 2 min |

The walk is unchanged (0.44-0.55x at 0-0.6 m/s).

## Limits

- Heat-model uncertainty: the robot test heated a shoulder about as the model
  predicts and a knee about 2x faster. Estimates are given for both.
- Scuttle at 0.75-1 m/s runs its waist motors near 0.9-1.0x their rating: fine
  for long stretches if the motors match the model, a few minutes if they heat
  2x faster. Use it in bursts (<= 2-3 min) between creepy crawling or rest
  until all-motor temperature logs from the robot say otherwise. It is also the
  less robust style under heavy randomization and shoves (5-9 % loop failures
  vs 2-3 % for creepy): smooth floors, no hard pushes at speed.
- Scuttle's speed tracking is short of the command (1.0 -> 0.87 m/s).
- The real knees carried ~3x the simulated load in the crawl (26 vs 9 Nm),
  unexplained (mass, CoM, knee contact, harness). Weigh the braces and log.
- One robot test so far. First runs on the harness, someone on the e-stop.

## Controller checklist (laptop)

- Clamp PD targets to `joint_limits` (trained with it).
- Braced motors 20, 21, 27, 28 in damping (unchanged).
- Thermal: keep the 85 C stop; stand up automatically from crawl or hold at
  ~72 C; log every motor's two temperatures (`scripts/motor_safety/
  sysid_thermal.py` can then identify the real heat model).
- Crawl command limits per style from `chain.json` (`crawl_variants`).
- Switch crawl styles only at zero command and still.
