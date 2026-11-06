from mjlab.tasks.g1_crawl.config.g1.rl_cfg import (
  AnchoredPpoAlgorithmCfg,
  anchored_finetune,
)
from mjlab.tasks.g1_walk.mdp.symmetry import make_stand_crawl_symmetry_augmentation
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  TRANSITION_ACTION_CLIP,
  unitree_g1_kneel_env_cfg,
  unitree_g1_stand_crawl_carpet_env_cfg,
  unitree_g1_stand_crawl_cool_env_cfg,
  unitree_g1_stand_crawl_push_env_cfg,
  unitree_g1_stand_crawl_robust_env_cfg,
  unitree_g1_stand_to_crawl_env_cfg,
  unitree_g1_walk_env_cfg,
)
from .rl_cfg import unitree_g1_walk_ppo_runner_cfg

# The velocity runner also exports every saved checkpoint to ONNX (with the
# observation normalizer baked in), which is what runs on the real robot.
register_mjlab_task(
  task_id="Mjlab-Walk-Flat-Unitree-G1-29dof",
  env_cfg=unitree_g1_walk_env_cfg(),
  play_env_cfg=unitree_g1_walk_env_cfg(play=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Same task with left-right mirrored data and a mirror loss: for fine-tuning a
# walk checkpoint whose turning came out asymmetric (same observations and
# actions, so `--agent.resume` from a plain walk run works).
register_mjlab_task(
  task_id="Mjlab-Walk-Flat-Sym-Unitree-G1-29dof",
  env_cfg=unitree_g1_walk_env_cfg(),
  play_env_cfg=unitree_g1_walk_env_cfg(play=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(symmetry=True),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-StandToCrawl-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_to_crawl_env_cfg(),
  play_env_cfg=unitree_g1_stand_to_crawl_env_cfg(play=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(
    experiment_name="g1_stand_to_crawl_29dof",
    max_iterations=10_000,
    clip_actions=TRANSITION_ACTION_CLIP,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Rough-terrain variants: the same observations and actions as the flat tasks, so
# a flat checkpoint resumes into them (`--agent.resume`, same experiment folders
# as the flat tasks; tag a run with `--agent.run-name rough`). Generated terrain with
# the crawl task's zone curriculum, on top of the pushes, randomization, noise
# and command delay the flat tasks already train with.
register_mjlab_task(
  task_id="Mjlab-Walk-Rough-Unitree-G1-29dof",
  env_cfg=unitree_g1_walk_env_cfg(rough=True),
  play_env_cfg=unitree_g1_walk_env_cfg(play=True, rough=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(experiment_name="g1_walk_29dof"),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Walk-Rough-Sym-Unitree-G1-29dof",
  env_cfg=unitree_g1_walk_env_cfg(rough=True),
  play_env_cfg=unitree_g1_walk_env_cfg(play=True, rough=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(experiment_name="g1_walk_29dof", symmetry=True),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-StandToCrawl-Rough-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_to_crawl_env_cfg(rough=True),
  play_env_cfg=unitree_g1_stand_to_crawl_env_cfg(play=True, rough=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(
    experiment_name="g1_stand_to_crawl_29dof",
    max_iterations=10_000,
    clip_actions=TRANSITION_ACTION_CLIP,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Take a knee (right knee, by a slow reverse lunge), hold, stand back up: one
# policy for both directions. Same observation layout as the stand-to-crawl.
register_mjlab_task(
  task_id="Mjlab-StandKneel-Unitree-G1-29dof",
  env_cfg=unitree_g1_kneel_env_cfg(),
  play_env_cfg=unitree_g1_kneel_env_cfg(play=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(
    experiment_name="g1_kneel_29dof",
    max_iterations=10_000,
    clip_actions=TRANSITION_ACTION_CLIP,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Hardened walk: resume a walk that finished the normal curriculum (same
# observations and actions). Harder randomization ramp, IMU tilt, arm-end mass.
register_mjlab_task(
  task_id="Mjlab-Walk-Rough-Sym-Hard-Unitree-G1-29dof",
  env_cfg=unitree_g1_walk_env_cfg(rough=True, hard=True),
  play_env_cfg=unitree_g1_walk_env_cfg(play=True, rough=True, hard=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(experiment_name="g1_walk_29dof", symmetry=True),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Walk-Flat-Sym-Hard-Unitree-G1-29dof",
  env_cfg=unitree_g1_walk_env_cfg(hard=True),
  play_env_cfg=unitree_g1_walk_env_cfg(play=True, hard=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(experiment_name="g1_walk_29dof", symmetry=True),
  runner_cls=VelocityOnPolicyRunner,
)

# Stand <-> crawl both ways in one policy (the stand-to-crawl's observation and
# action layout, command [phi, dphi/dt]); resumes from a stand-to-crawl run.
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_to_crawl_env_cfg(bidirectional=True),
  play_env_cfg=unitree_g1_stand_to_crawl_env_cfg(play=True, bidirectional=True),
  rl_cfg=unitree_g1_walk_ppo_runner_cfg(
    experiment_name="g1_stand_to_crawl_29dof",
    max_iterations=10_000,
    clip_actions=TRANSITION_ACTION_CLIP,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Thermal fine-tune of the two-way stand <-> crawl after the 2026-10-29 robot
# test: resume from a ``Mjlab-StandCrawl-Unitree-G1-29dof`` run.
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Cool-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_cool_env_cfg(),
  play_env_cfg=unitree_g1_stand_crawl_cool_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_walk_ppo_runner_cfg(
      experiment_name="g1_stand_to_crawl_29dof",
      clip_actions=TRANSITION_ACTION_CLIP,
    )
  ),
  runner_cls=VelocityOnPolicyRunner,
)


def _stand_crawl_cool2_rl_cfg(max_iterations: int = 1500):
  cfg = anchored_finetune(
    unitree_g1_walk_ppo_runner_cfg(
      experiment_name="g1_stand_to_crawl_29dof",
      clip_actions=TRANSITION_ACTION_CLIP,
    ),
    max_iterations=max_iterations,
    actor_freeze_updates=20,
  )
  # Mirror loss only: the policy is not yet symmetric, so mirrored rollout
  # samples would be off-policy for the PPO ratio.
  assert isinstance(cfg.algorithm, AnchoredPpoAlgorithmCfg)
  cfg.algorithm.symmetry_cfg = {
    "data_augmentation_func": make_stand_crawl_symmetry_augmentation(),
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.2,
  }
  return cfg


# Continuation of a StandCrawl-Cool run: the hold's heat penalty doubled and a
# left-right mirror loss (the hold leaned on one shoulder).
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Cool2-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_cool_env_cfg(heat_weight=-0.8, ramp=(0, 1)),
  play_env_cfg=unitree_g1_stand_crawl_cool_env_cfg(play=True, heat_weight=-0.8),
  rl_cfg=_stand_crawl_cool2_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


# Continuation of StandCrawl-Cool2: the transitions' own heat at half weight
# (repeated stand <-> crawl cycles in a long session).
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Cool3-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_cool_env_cfg(
    heat_weight=-0.8, ramp=(0, 1), moving_scale=0.5
  ),
  play_env_cfg=unitree_g1_stand_crawl_cool_env_cfg(
    play=True, heat_weight=-0.8, moving_scale=0.5
  ),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=2500),
  runner_cls=VelocityOnPolicyRunner,
)


# Stand <-> crawl for carpet: twist friction on the limbs and a penalty on
# planted feet sliding or spinning. Resumes from a StandCrawl-Cool2/3 run.
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Carpet-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_carpet_env_cfg(),
  play_env_cfg=unitree_g1_stand_crawl_carpet_env_cfg(play=True),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=2000),
  runner_cls=VelocityOnPolicyRunner,
)


# StandCrawl-Push: Cool2 hardened against shoves in the descent and stand-up
# (transitions' heat free, pushes every 2-4 s, touches > 10 N end the episode).
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Push-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_push_env_cfg(),
  play_env_cfg=unitree_g1_stand_crawl_push_env_cfg(play=True),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=1500),
  runner_cls=VelocityOnPolicyRunner,
)


# StandCrawl-Robust: StandCrawl-Push with wider margins (lag to 40 ms, weaker
# motors, heavier, carpet twist, slippery floors) and bonuses for passing the
# switcher's gates. Resumes from a StandCrawl-Push run.
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Robust-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_robust_env_cfg(),
  play_env_cfg=unitree_g1_stand_crawl_robust_env_cfg(play=True),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=2000),
  runner_cls=VelocityOnPolicyRunner,
)


# Robust2: Robust with smooth gate rewards (base settled at the top, crawl
# height and orientation at the bottom). Resumes from a Robust run.
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Robust2-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_robust_env_cfg(settle_weight=2.0),
  play_env_cfg=unitree_g1_stand_crawl_robust_env_cfg(play=True, settle_weight=2.0),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=1500),
  runner_cls=VelocityOnPolicyRunner,
)


# Robust3: Robust2 starting a quarter of the episodes from recorded creepy and
# scuttle hand-over poses, and stricter about forbidden touches (> 3 N).
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Robust3-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_robust_env_cfg(settle_weight=2.0, stop_frames=True),
  play_env_cfg=unitree_g1_stand_crawl_robust_env_cfg(
    play=True, settle_weight=2.0, stop_frames=True
  ),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=1500),
  runner_cls=VelocityOnPolicyRunner,
)


# Robust4: Robust3 with 35 % crawl-pose starts (was 50 %): the descent trains
# as much as before. Resumes from a Robust3 run.
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Robust4-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_robust_env_cfg(
    settle_weight=2.0, stop_frames=True, crawl_start_prob=0.35
  ),
  play_env_cfg=unitree_g1_stand_crawl_robust_env_cfg(
    play=True, settle_weight=2.0, stop_frames=True, crawl_start_prob=0.35
  ),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=1000),
  runner_cls=VelocityOnPolicyRunner,
)


# Robust5: from robust_v2, hand-over starts at 0.35, the 10 N touch limit, and
# one motor-stiffness draw per robot (whole-robot-soft worlds).
_ROBUST5 = dict(
  settle_weight=2.0,
  stop_frames=True,
  crawl_start_prob=0.35,
  strict_touch=False,
  correlated_gains=True,
)
register_mjlab_task(
  task_id="Mjlab-StandCrawl-Robust5-Unitree-G1-29dof",
  env_cfg=unitree_g1_stand_crawl_robust_env_cfg(**_ROBUST5),
  play_env_cfg=unitree_g1_stand_crawl_robust_env_cfg(play=True, **_ROBUST5),
  rl_cfg=_stand_crawl_cool2_rl_cfg(max_iterations=1500),
  runner_cls=VelocityOnPolicyRunner,
)
