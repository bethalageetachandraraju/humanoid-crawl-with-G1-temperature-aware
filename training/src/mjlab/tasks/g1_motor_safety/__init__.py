"""Motor-safe G1 walking: identified heating DC motors, firmware overheat cut-off,
and a motor safety filter (torque-speed feasibility + thermal barrier).

Tasks (``env_cfgs.py``): ``Mjlab-MotorSafety-G1-{Ideal,Penalty,Filter}``. All three
log to the experiment ``g1_motor_safety`` so the arms can warm-start from one
base run (``--agent.resume True --agent.load-run <run>``).
"""

from dataclasses import dataclass

from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  ARMS,
  EXT_AMBIENT,
  EXT_PAYLOAD,
  motor_hj_env_cfg,
  motor_safety_env_cfg,
)
from .plugin import MODES, motor_safe
from .safety_ppo import SafetyRunner

EXPERIMENT = "g1_motor_safety"
#: The frozen task policy whose states the fallback takes over from (normal RL,
#: ideal-motor training). Paths are relative to the mjlab repository root.
PILOT = "logs/rsl_rl/g1_motor_safety/2026-10-24_22-48-36_ft_ideal/model_4498.pt"


@dataclass(kw_only=True)
class SafetyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  actor_warmup_updates: int = 50
  """Updates during which only the (fresh) value function learns."""


@dataclass(kw_only=True)
class SafetyRunnerCfg(RslRlOnPolicyRunnerCfg):
  pilot_checkpoint: str = PILOT
  pilot_prob: float = 0.7
  pilot_max_s: float = 15.0
  init_actor_only: bool = True


def _rl_cfg():
  cfg = unitree_g1_ppo_runner_cfg()
  cfg.experiment_name = EXPERIMENT
  cfg.save_interval = 250
  return cfg


def _safety_rl_cfg() -> SafetyRunnerCfg:
  base = unitree_g1_ppo_runner_cfg()
  return SafetyRunnerCfg(
    actor=base.actor,
    critic=base.critic,
    # The value net reads only what the robot can measure, so the same V(s)
    # can be the deployed monitor.
    obs_groups={"actor": ("actor",), "critic": ("actor",)},
    algorithm=SafetyPpoAlgorithmCfg(
      class_name="mjlab.tasks.g1_motor_safety.safety_ppo:SafetyPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # No entropy bonus: the margin barely penalizes action noise, so any bonus
      # inflates the std (0.37 -> 0.54 in 500 updates at 0.002) and with it falls.
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.999,  # ~20 s horizon, the winding time constant
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      actor_warmup_updates=50,
    ),
    experiment_name=EXPERIMENT,
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=1500,
  )


for _task, _fall in (("HjSafety", True), ("HjMotorOnly", False)):
  register_mjlab_task(
    task_id=f"Mjlab-MotorSafety-G1-{_task}",
    env_cfg=motor_hj_env_cfg(fall_is_failure=_fall),
    play_env_cfg=motor_hj_env_cfg(play=True, fall_is_failure=_fall),
    rl_cfg=_safety_rl_cfg(),
    runner_cls=SafetyRunner,
  )

for _arm in ARMS:
  register_mjlab_task(
    task_id=f"Mjlab-MotorSafety-G1-{_arm.capitalize()}",
    env_cfg=motor_safety_env_cfg(_arm),
    play_env_cfg=motor_safety_env_cfg(_arm, play=True, init_temp="ambient"),
    rl_cfg=_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )

# The clip-filter arm on the extended envelope (heavier, hotter): the walking
# policy under the HJ reference governor, and the strongest baseline without it.
register_mjlab_task(
  task_id="Mjlab-MotorSafety-G1-FilterExt",
  env_cfg=motor_safety_env_cfg("filter", payload=EXT_PAYLOAD, ambient=EXT_AMBIENT),
  play_env_cfg=motor_safety_env_cfg(
    "filter", play=True, init_temp="ambient", payload=EXT_PAYLOAD, ambient=EXT_AMBIENT
  ),
  rl_cfg=_rl_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Plug-in proof of concept: the same transform on a task it was never built for,
# tracking a 131 s LAFAN1 dance. Only the plug-in mode differs between the arms.
DANCE_CLIP = (
  "/home/smoky/humanoid/supermimic/data/motions/lafan1_g1/dance1_subject1.npz"
)


def _dance_env_cfg(
  mode: str,
  play: bool = False,
  weight: float = 1.0,
  init: str = "random",
  observe: bool | None = None,
):
  cfg = unitree_g1_flat_tracking_env_cfg(play=play)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.motion_file = DANCE_CLIP
  return motor_safe(
    cfg,
    mode,
    init_temp="ambient" if play else init,
    weight=weight,
    observe=observe,
  )


def _dance_rl_cfg():
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.experiment_name = "g1_motor_safe_dance"
  return cfg


for _mode in MODES:
  register_mjlab_task(
    task_id=f"Mjlab-MotorSafe-Dance-G1-{_mode.capitalize()}",
    env_cfg=_dance_env_cfg(_mode),
    play_env_cfg=_dance_env_cfg(_mode, play=True),
    rl_cfg=_dance_rl_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )

# The heat penalty at weight 1 collapses when tracking is learned from scratch:
# early on the penalty outweighs the tracking reward, so ending the episode
# (falling) pays best. Weight 0.1 is the fair soft-constraint baseline.
register_mjlab_task(
  task_id="Mjlab-MotorSafe-Dance-G1-PenaltyLow",
  env_cfg=_dance_env_cfg("penalty", weight=0.1),
  play_env_cfg=_dance_env_cfg("penalty", play=True, weight=0.1),
  rl_cfg=_dance_rl_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# Round 2: the same arms, fine-tuned with per-robot thermal states ("mixed"). In
# round 1 every motor drew its own temperature, so a cold robot never appeared in
# training and the temperature-observing policies failed from a cold start.
for _mode in MODES:
  _w = 0.1 if _mode == "penalty" else 1.0
  _name = "PenaltyLow" if _mode == "penalty" else _mode.capitalize()
  register_mjlab_task(
    task_id=f"Mjlab-MotorSafe-Dance2-G1-{_name}",
    env_cfg=_dance_env_cfg(_mode, weight=_w, init="mixed"),
    play_env_cfg=_dance_env_cfg(_mode, play=True, weight=_w),
    rl_cfg=_dance_rl_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
  )

# Round 3: the plug-in applied the way it worked for walking, by fine-tuning the
# finished stock dance policy inside the filter. "NoObs" keeps the stock network
# (no temperature observations); "Plugin" itself is reached by widening the stock
# network's input (scripts/motor_safety/widen_checkpoint.py).
register_mjlab_task(
  task_id="Mjlab-MotorSafe-Dance2-G1-PluginNoObs",
  env_cfg=_dance_env_cfg("plugin", init="mixed", observe=False),
  play_env_cfg=_dance_env_cfg("plugin", play=True, observe=False),
  rl_cfg=_dance_rl_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
