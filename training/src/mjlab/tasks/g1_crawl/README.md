# G1 crawl in MJLab

This task ports the active `g1-crawl` Isaac Lab setup to the local MJLab task
registry as `Mjlab-Crawl-Flat-Unitree-G1`. It uses the checked-in
`assets/animation_rc4.json` resets, the 23-DoF policy interface, 50 Hz control,
20 second episodes, noisy actor observations, separate critic observations,
symmetry augmentation, domain randomization, and the active reward weights from
`g1_crawl_env_cfg.py`.

The source repository says that the precise rewards behind its final crawl
policies are unknown. This port follows the checked-in active config; it cannot
recover undocumented reward sweeps that may have produced the distributed
`.pt` policies. The checked-in active config also sets `USE_ROUGH_TERRAIN=False`,
so its terrain curriculum is disabled. Pose resets sample uniformly from rc4,
as in that active config.

The active source config samples an angular-X crawl command but wires it to a
reward that compares against world angular velocity Z, which flips the turn sign
when the body is prone. This port tracks the command against base-frame roll rate
(`track_ang_vel_x_exp`) instead, so the turn direction is fixed.

## Stage A (flat-ground polish) changes

Relative to the source reward set: turn command range widened to ±3 rad/s and
idle (zero-command) envs raised to 15%; pushes every 5 to 10 s; `action_rate_l2`
doubled to -0.02 plus a joint-acceleration penalty (-2.5e-7); new terms
`base_wobble` (angular velocity about the non-turning body axes), `base_bobbing`
(vertical base velocity) and `stand_still` (joint motion while the command is
zero). All are resume-safe: the policy inputs and outputs are unchanged, so a
checkpoint from the plain config can be continued with `--agent.resume True`.

## Robot model

The task does not use MJLab's bundled 29-DoF G1. It loads the source project's
own 23-DoF "simple forearm" G1 (`assets/g1_23dof_crawl.xml`, the sim2sim MuJoCo
export of the robot the Isaac Lab task trained on, with its meshes). `robot.py`
strips the export's plane, light, camera, actuators and keyframe, and adds back
the two wrist-roll hinges that the export lacks but the USD, the animation and
the policy layout have. The entity default pose is the source's crawl pose, so
`use_default_offset=True`, `joint_pos_rel` and `joint_deviation_l1` are all
relative to the crawl pose exactly as in Isaac Lab. Self-collisions are off and
all contacts are frictional, matching the source articulation settings.

Why this matters: an earlier version of this port used the 29-DoF G1 with its
standing default pose. With action `0` the PD targets were the standing pose, so
every episode started with the robot launching itself out of the crawl pose and
flipping onto its back, and the hip-roll angle of the crawl pose was unreachable
through the `clip_actions=5.0` limit. The 29-DoF arm is also 7 cm longer and
0.7 kg heavier at the hand, its wrist pitch/yaw motors (5 Nm) back-drive under
body weight, and its wrist collision capsule counted as an undesired contact
whenever a hand was planted. Together these made a feet-only gait the easier
solution. Rewards were also unscaled by `dt` while Isaac Lab scales them, which
inflated every reward 50x.

### Reward balance (after the first terrain run)

The first terrain run collapsed into standing still: at the end of the flat
phase tracking earned about +37 per episode against about -40 of penalties, so
once terrain made tracking harder the penalties for moving outweighed the gain,
and with `std=0.25` a stopped robot received no tracking gradient to restart.
The shared config now uses tracking weights 3.0 with `std=0.5`, a linear
`forward_progress` term (speed along the commanded direction, clipped at the
command), and lighter motion penalties (`action_rate_l2` -0.01, `joint_acc_l2`
-1e-7, `base_wobble` -0.02, `stand_still` -0.01); the terrain task uses
`limb_clearance` -0.5 and a -100 termination penalty.

## Stage B: stairs and uneven terrain (`Mjlab-Crawl-Rough-Unitree-G1`)

Same policy interface (still blind, no height scan in the observations), so any
flat-task checkpoint can be resumed into it. The scene is a generated grid of
box-only tiles: flat, a 0.4 m box grid with 1 to 6 cm height steps, and pyramid
stairs up and down with 2.4 to 12 cm steps on 0.4 m treads
(`difficulty_range=(0.2, 1.0)` so no tile is degenerate). Heightfield tiles are
deliberately absent: MuJoCo Warp gives one contact per capsule on a heightfield
and the crawl's flat-lying foot capsules slide on them; the flat-ground policy
flipped in 40% of envs on a *flat* heightfield versus 0% on a flat box.

`terrain_levels_crawl` runs three phases on global iteration counts (MJLab
restores the env step counter on resume, so a run resumed from iteration 2999
sees the same global numbers):

1. Up to `FLAT_ITERATIONS` (7000): every env is held on the flat tile at level
   0, which is physically a plane.
2. Then the curriculum: each env gets a random tile type at level 0 and is
   promoted when it crawls 80% of the commanded distance, demoted below 50%.
3. From `SPREAD_AFTER_ITERATIONS` (15000): promotion stops and every resetting
   env is placed on a uniformly random level and tile type, so the final phase
   trains on all terrains equally (`Curriculum/terrain_levels/spread` = 1).

The task adds a limb-clearance penalty on a per-limb terrain height sensor, a
termination when the body rolls past 90 degrees from prone, a -200 termination
penalty, and a larger contact budget. The export's torso collision box is
swapped for a capsule because MuJoCo caps box/heightfield contacts at 50 per
pair.

```bash
uv run train Mjlab-Crawl-Rough-Unitree-G1 --env.scene.num-envs 8192 \
  --agent.max-iterations 25000 --agent.resume True \
  --agent.load-run <flat run> --agent.load-checkpoint model_2999.pt \
  --agent.logger tensorboard
```

Watch `Curriculum/terrain_levels/unlocked`, `.../spread`, `.../mean` and
`Episode_Termination/flipped` in TensorBoard. Play mode uses a small 5x5 grid
at difficulty 0.4 to 1.0 with the curriculum disabled.

## Zone layout and left-to-right curriculum

The rough tasks now use `CRAWL_ZONES_CFG`: nine zones laid out left to right
(flat, meadows, hill, valley, stairs up 2 to 8 cm, stairs down 2 to 8 cm, box
grid 1 to 6 cm, big stairs 4 to 12 cm, random mix), each with 5 difficulty
rows front to back (flat counts as a single step).
`terrain_zones_crawl` gives every env a progress index that walks through the
rows of a zone and then into the next zone to the right (promote when the
distance crawled in the commanded direction, accumulated over the episode,
exceeds 80% of the accumulated commanded distance; demote below 50%); the random-mix zone places its envs on a
random non-flat zone at the same row. Warm-up keeps everything in the flat zone,
the spread phase scatters envs over all zones and rows. TensorBoard logs
`Curriculum/terrain_zones/zone_mean`, `zone_max`, `row_mean` and the fraction of
envs per zone. Meadows are a Perlin heightfield with a height floor; check it
with `scripts/crawl_tile_test.py --tile meadows` before relying on it, since
near-flat heightfields make the foot capsules slide. Play mode keeps the same
six zones with 4 rows and no curriculum.

## High-step mode: 7-value command (`*-Cmd7-*` tasks)

`Mjlab-Crawl-Flat-Cmd7-Unitree-G1` and `Mjlab-Crawl-Rough-Cmd7-Unitree-G1` use
the command `[forward, lateral, turn, step_height, spare, spare, spare]`
(actor 79 values, critic 85). `step_height` (m) is the commanded limb swing
height, sampled per episode from the tile type 80% of the time (flat 2 to 6 cm,
box grid 6 to 12 cm, stairs 8 to 15 cm) and uniformly 2 to 15 cm otherwise, and
tracked by `limb_clearance` (velocity-weighted deviation) plus `limb_lift_bonus`
(kernel), both doubled after global iteration `STEP_HEIGHT_BOOST_ITERATION`
(20000). The three spare slots are always 0.0 during training; deployment code
must send 0.0 for them until a slot is given a meaning. Zero inputs leave their
weights untrained, so a spare can be activated later without surgery.

Convert a 3-command checkpoint once with `scripts/expand_policy_obs.py`
(inserts four zero input columns; the expanded policy is numerically identical
to the original while the new slots are zero), then resume from the new folder.
`scripts/play_teleop.py` cycles the step height with `H`; `scripts/eval_crawl.py
--step-height 0.12` reports measured swing peaks per limb.

## Train

From the MJLab checkout:

```bash
cd mjlab
uv run train Mjlab-Crawl-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 1000 \
  --agent.logger tensorboard
```

The runner writes checkpoints under `logs/rsl_rl/g1_crawl/<run>/`. For a
smaller initial run, lower `--env.scene.num-envs` (for example, 256); training
throughput and convergence will differ from the source's 4096 environments.

## Watch training live

`uv run train` has no live viewer. `scripts/train_with_viewer.py` runs the same
RSL-RL loop in a background thread and streams every environment to a Viser page
(default http://localhost:8080). Use a small environment count for this; it is a
sanity check, not the real run:

```bash
uv run python scripts/train_with_viewer.py --num-envs 100 --max-iterations 2000
```

## Play

Choose a checkpoint from the run directory:

```bash
uv run play Mjlab-Crawl-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_crawl/<run>/model_1000.pt \
  --viewer native
```

For a short rendered video:

```bash
uv run play Mjlab-Crawl-Flat-Unitree-G1 \
  --checkpoint-file logs/rsl_rl/g1_crawl/<run>/model_1000.pt \
  --video --video-length 500
```

To inspect the reset motion before training, use the zero agent (action `0`
now holds the crawl pose):

```bash
uv run play Mjlab-Crawl-Flat-Unitree-G1 --agent zero --viewer native
```

## Policy interface and transfer

The actor receives 75 values in this order: projected gravity (3), command
`[linear-z, linear-y, angular-x]` (3), the 23 joint position offsets from the
crawl pose, the 23 joint velocities, and the 23 previous actions. Its output is
23 position targets, scaled by 0.5 and added to the crawl pose, in the order
legs (left 6, right 6), waist yaw, arms (left 5, right 5). The critic receives
the actor state plus base linear and angular velocity, for 81 values.

The joint set, order, default pose, action scale, actuator gains and effort
limits now match the source task. Before deploying, still verify the real
robot's joint ordering, sign conventions, observation estimator, control rate,
and exported checkpoint interface. MJLab training checkpoints can be played in
MJLab; they are not automatically compatible with the Isaac-based
`g1_crawl/deployment` runner.
