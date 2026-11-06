from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_crawl_cool_env_cfg,
  unitree_g1_crawl_creepy_carpet_env_cfg,
  unitree_g1_crawl_creepy_env_cfg,
  unitree_g1_crawl_creepy_hard_env_cfg,
  unitree_g1_crawl_creepy_shoulder_env_cfg,
  unitree_g1_crawl_env_cfg,
  unitree_g1_crawl_fast_carpet_env_cfg,
  unitree_g1_crawl_fast_env_cfg,
  unitree_g1_crawl_rough_env_cfg,
  unitree_g1_crawl_scuttle_cool_env_cfg,
  unitree_g1_crawl_scuttle_env_cfg,
  unitree_g1_crawl_scuttle_fast_env_cfg,
  unitree_g1_crawl_scuttle_free_env_cfg,
  unitree_g1_crawl_scuttle_hard_env_cfg,
  unitree_g1_crawl_walkgait_env_cfg,
)
from .rl_cfg import anchored_finetune, unitree_g1_crawl_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Unitree-G1",
  env_cfg=unitree_g1_crawl_env_cfg(),
  play_env_cfg=unitree_g1_crawl_env_cfg(play=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Crawl-Rough-Unitree-G1",
  env_cfg=unitree_g1_crawl_rough_env_cfg(),
  play_env_cfg=unitree_g1_crawl_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(),
)

# 7-value command variants: [fwd, lat, turn, step_height, spare, spare, spare].
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Unitree-G1",
  env_cfg=unitree_g1_crawl_env_cfg(cmd7=True),
  play_env_cfg=unitree_g1_crawl_env_cfg(play=True, cmd7=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(num_cmd=7),
)
register_mjlab_task(
  task_id="Mjlab-Crawl-Rough-Cmd7-Unitree-G1",
  env_cfg=unitree_g1_crawl_rough_env_cfg(cmd7=True),
  play_env_cfg=unitree_g1_crawl_rough_env_cfg(play=True, cmd7=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, terrain=True),
)

##
# 29-DoF G1: the robot Unitree ships, and the one a real-hardware policy has to
# be trained on. Separate task ids because the observation and action layouts
# differ (29 joints instead of 23), so no 23-DoF checkpoint loads into them.
##

register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_env_cfg(dof29=True),
  play_env_cfg=unitree_g1_crawl_env_cfg(play=True, dof29=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(dof29=True),
)
register_mjlab_task(
  task_id="Mjlab-Crawl-Rough-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_rough_env_cfg(dof29=True),
  play_env_cfg=unitree_g1_crawl_rough_env_cfg(play=True, dof29=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(dof29=True),
)
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_env_cfg(cmd7=True, dof29=True),
  play_env_cfg=unitree_g1_crawl_env_cfg(play=True, cmd7=True, dof29=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
)
register_mjlab_task(
  task_id="Mjlab-Crawl-Rough-Cmd7-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_rough_env_cfg(cmd7=True, dof29=True),
  play_env_cfg=unitree_g1_crawl_rough_env_cfg(play=True, cmd7=True, dof29=True),
  rl_cfg=unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, terrain=True, dof29=True),
)

# Thermal fine-tune of the gen2 crawl after the 2026-10-29 robot test: resume
# from a ``Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof`` run (same layouts).
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Cool-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_cool_env_cfg(),
  play_env_cfg=unitree_g1_crawl_cool_env_cfg(play=True),
  rl_cfg=anchored_finetune(unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True)),
)

# Continuation of a Cool run with the moving heat at half weight (see
# ``unitree_g1_crawl_cool_env_cfg``): the critic already knows the terms, so a
# short ramp and warm-up.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Cool2-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_cool_env_cfg(moving_scale=0.5, ramp=(20, 100)),
  play_env_cfg=unitree_g1_crawl_cool_env_cfg(play=True, moving_scale=0.5),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True), actor_freeze_updates=20
  ),
)

# Continuation of Cool2: a stronger braced-wrist penalty and a waist-yaw heat
# term at full weight while moving (heat terms already known: no ramp). The
# anchor (to Cool2's actor) is looser so the gait can change.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Cool3-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_cool_env_cfg(
    moving_scale=0.5, ramp=(0, 1), braced_weight=-4.0, waist_yaw_weight=-0.5
  ),
  play_env_cfg=unitree_g1_crawl_cool_env_cfg(play=True, moving_scale=0.5),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=1500,
    actor_freeze_updates=20,
    anchor_coef=0.5,
    anchor_decay_updates=1000,
  ),
)

# Cool4: a walking crawl gait (three limbs down) at the robot's speeds, every
# heat term at full weight while moving; continues a Cool3 run with a loose
# anchor so the gait can change.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Cool4-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_walkgait_env_cfg(),
  play_env_cfg=unitree_g1_crawl_walkgait_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=4000,
    actor_freeze_updates=20,
    anchor_coef=0.2,
    anchor_decay_updates=500,
  ),
)

# The creepy crawl (Halloween): a slow, deliberate, three-limbs-down crawl that
# also stays cool for long crawls. Same observations and actions as the trot.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Creepy-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_creepy_env_cfg(),
  play_env_cfg=unitree_g1_crawl_creepy_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=4000,
    actor_freeze_updates=20,
    anchor_coef=0.2,
    anchor_decay_updates=500,
  ),
)

# The fast crawl: mostly 0.75-1 m/s, thermally priced, robust. Continues the
# trot (Cool3) with a loose anchor.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Fast-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_fast_env_cfg(),
  play_env_cfg=unitree_g1_crawl_fast_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=4000,
    actor_freeze_updates=20,
    anchor_coef=0.3,
    anchor_decay_updates=1000,
  ),
)

# Fast and creepy crawls continued with carpet contacts (twist friction on the
# limbs, floor friction to 1.3): resume from a Fast / Creepy run.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Fast2-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_fast_carpet_env_cfg(),
  play_env_cfg=unitree_g1_crawl_fast_carpet_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=3600,
    actor_freeze_updates=20,
    anchor_coef=0.3,
    anchor_decay_updates=1000,
  ),
)
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Creepy2-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_creepy_carpet_env_cfg(),
  play_env_cfg=unitree_g1_crawl_creepy_carpet_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=3200,
    actor_freeze_updates=20,
    anchor_coef=0.2,
    anchor_decay_updates=500,
  ),
)

# Fast3: the fast crawl on carpet with the waist-yaw heat term tripled and a
# looser anchor, for a gait that does not twist the torso every step.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Fast3-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_fast_carpet_env_cfg(waist_yaw_weight=-1.5),
  play_env_cfg=unitree_g1_crawl_fast_carpet_env_cfg(play=True, waist_yaw_weight=-1.5),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=3200,
    actor_freeze_updates=20,
    anchor_coef=0.1,
    anchor_decay_updates=300,
  ),
)

# Creepy3: the creepy crawl on carpet with the shoulder rolls priced on their
# own (they hold the splayed arms). Resumes from a Creepy2 run.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Creepy3-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_creepy_shoulder_env_cfg(),
  play_env_cfg=unitree_g1_crawl_creepy_shoulder_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=2800,
    actor_freeze_updates=20,
    anchor_coef=0.1,
    anchor_decay_updates=300,
  ),
)

# Scuttle: the fast crawl grown from the creepy (three-limb) gait; resumes from
# a Creepy2/3 run with a loose anchor.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Scuttle-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_scuttle_env_cfg(),
  play_env_cfg=unitree_g1_crawl_scuttle_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=2500,
    actor_freeze_updates=20,
    anchor_coef=0.05,
    anchor_decay_updates=200,
  ),
)

# Scuttle2: Scuttle with the commanded speed paid harder (it stopped at 0.75 m/s
# for 1 m/s commands). Resumes from a Scuttle run.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Scuttle2-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_scuttle_fast_env_cfg(),
  play_env_cfg=unitree_g1_crawl_scuttle_fast_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=2000,
    actor_freeze_updates=20,
    anchor_coef=0.05,
    anchor_decay_updates=200,
  ),
)

# Scuttle3: Scuttle2 without the creepy crawl's joint-speed cap (it held the
# speed at 0.75 m/s). Resumes from a Scuttle2 run.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-Scuttle3-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_scuttle_free_env_cfg(),
  play_env_cfg=unitree_g1_crawl_scuttle_free_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=1800,
    actor_freeze_updates=20,
    anchor_coef=0.05,
    anchor_decay_updates=200,
  ),
)

# Hardened creepy and scuttle crawls: the chain test's hard randomization and a
# forbidden-body touch ends the episode. Resume from Creepy3 / Scuttle3 runs.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-CreepyHard-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_creepy_hard_env_cfg(),
  play_env_cfg=unitree_g1_crawl_creepy_hard_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=1600,
    actor_freeze_updates=20,
    anchor_coef=0.2,
    anchor_decay_updates=600,
  ),
)
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-ScuttleHard-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_scuttle_hard_env_cfg(),
  play_env_cfg=unitree_g1_crawl_scuttle_hard_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=1600,
    actor_freeze_updates=20,
    anchor_coef=0.2,
    anchor_decay_updates=600,
  ),
)

# ScuttleCool: ScuttleHard plus a waist-pitch heat term. Resume from ScuttleHard.
register_mjlab_task(
  task_id="Mjlab-Crawl-Flat-Cmd7-ScuttleCool-Unitree-G1-29dof",
  env_cfg=unitree_g1_crawl_scuttle_cool_env_cfg(),
  play_env_cfg=unitree_g1_crawl_scuttle_cool_env_cfg(play=True),
  rl_cfg=anchored_finetune(
    unitree_g1_crawl_ppo_runner_cfg(num_cmd=7, dof29=True),
    max_iterations=1200,
    actor_freeze_updates=20,
    anchor_coef=0.2,
    anchor_decay_updates=600,
  ),
)
