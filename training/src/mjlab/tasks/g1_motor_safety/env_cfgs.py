"""G1 flat velocity walking with a payload, on identified heating DC motors.

One environment, three training arms that differ only in the motor model and
one reward term:

  ideal    today's practice: ideal PD motors with a flat torque clamp, no heat
           effects (the thermal state is still simulated and observed, it just
           has no consequence), no overheat cut-off.
  penalty  the realistic motor (torque-speed envelope, heat, firmware cut-off)
           and a reward penalty on hot windings: the soft-constraint approach of
           thermal-aware locomotion work.
  filter   the realistic motor with the motor safety filter in the loop
           (feasibility envelope + thermal barrier, a hard torque constraint)
           and a small penalty on how much the filter holds back.

Every arm observes the same thing (stock velocity observations plus the two
temperature readings and the thermal headroom of each motor), so all policies
share one network shape and warm-start from each other.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.g1_motor_safety import mdp
from mjlab.tasks.g1_motor_safety.motor_model import ThermalDcMotorActuatorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

ARMS = ("ideal", "penalty", "filter")
PAYLOAD_RANGE = (0.0, 8.0)  # kg on the torso, per env
#: The extended operating envelope of the governor experiment: heavier loads and
#: hotter rooms than the walking policies were first trained for.
EXT_PAYLOAD = (0.0, 16.0)
EXT_AMBIENT = (20.0, 45.0)
MIN_PELVIS_HEIGHT = 0.45  # m; below this the robot has collapsed

MOTOR_FLAGS = {
  "ideal": dict(envelope=False, protection=False, filter_enabled=False),
  "penalty": dict(envelope=True, protection=True, filter_enabled=False),
  "filter": dict(envelope=True, protection=True, filter_enabled=True),
}


def motor_safety_env_cfg(
  arm: str,
  play: bool = False,
  init_temp: str = "random",
  payload: tuple[float, float] = PAYLOAD_RANGE,
  heat_penalty: float = 1.0,
  intervention_penalty: float = 1.0,
  ambient: tuple[float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  if arm not in ARMS:
    raise ValueError(f"arm {arm!r} not in {ARMS}")
  cfg = unitree_g1_flat_env_cfg(play=play)

  robot = get_g1_robot_cfg()
  assert robot.articulation is not None
  flags = MOTOR_FLAGS[arm]
  motor = ThermalDcMotorActuatorCfg(
    init_temp=init_temp,
    envelope=flags["envelope"],
    protection=flags["protection"],
    filter_enabled=flags["filter_enabled"],
  )
  if ambient is not None:
    motor.ambient_range = ambient
  robot.articulation = replace(robot.articulation, actuators=(motor,))
  cfg.scene.entities = {"robot": robot}

  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["motor_temperatures"] = ObservationTermCfg(func=mdp.motor_temperatures)
    terms["motor_headroom"] = ObservationTermCfg(func=mdp.motor_headroom)

  cfg.events["payload"] = EventTermCfg(
    func=dr.body_mass,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
      "operation": "add",
      "ranges": payload,
    },
  )

  # Fixed command box (no velocity curriculum), so every arm and every
  # fine-tuning stage trains on the same commands.
  cfg.curriculum.pop("command_vel", None)

  cfg.terminations["collapsed"] = TerminationTermCfg(
    func=vel_mdp.root_height_below_minimum,
    params={"minimum_height": MIN_PELVIS_HEIGHT},
  )

  if arm == "penalty":
    cfg.rewards["winding_heat"] = RewardTermCfg(
      func=mdp.winding_heat_penalty, weight=-heat_penalty
    )
  if arm == "filter":
    cfg.rewards["filter_intervention"] = RewardTermCfg(
      func=mdp.filter_intervention, weight=-intervention_penalty
    )

  cfg.metrics["max_winding_temp"] = MetricsTermCfg(
    func=mdp.max_winding_temp, reduce="max"
  )
  cfg.metrics["over_rating"] = MetricsTermCfg(func=mdp.over_rating)
  cfg.metrics["motor_cut"] = MetricsTermCfg(func=mdp.motor_cut, reduce="max")
  cfg.metrics["torque_held_back"] = MetricsTermCfg(func=mdp.torque_held_back)
  cfg.metrics["min_headroom"] = MetricsTermCfg(func=mdp.mean_headroom)
  return cfg


def motor_hj_env_cfg(
  play: bool = False, fall_is_failure: bool = True
) -> ManagerBasedRlEnvCfg:
  """The Hamilton-Jacobi motor-safety task (``safety_ppo.py``).

  The realistic motor (no filter), and a single reward term: the motor margin
  g(s) itself, unscaled, which the safety Bellman backup turns into V(s). The
  episode ends when a winding passes its rating (g < 0) or the robot falls,
  and lasts 30 s so a pilot-driven prefix still leaves time to act.
  """
  cfg = motor_safety_env_cfg("penalty", play=play)
  cfg.rewards = {
    "safety_margin": RewardTermCfg(
      func=mdp.safety_margin, weight=1.0, params={"fall_is_failure": fall_is_failure}
    )
  }
  cfg.scale_rewards_by_dt = False
  cfg.terminations["motor_overheat"] = TerminationTermCfg(func=mdp.motor_overheat)
  if not play:
    cfg.episode_length_s = 30.0
  return cfg


def motor_governor_env_cfg(
  payload: tuple[float, float] = EXT_PAYLOAD,
  ambient: tuple[float, float] = EXT_AMBIENT,
) -> ManagerBasedRlEnvCfg:
  """Value learning for the HJ reference governor (``train_governor.py``).

  The closed loop being certified: a frozen walking policy on the realistic
  motors with the clip filter on. The reward channel is the safety margin
  g(s) (motor headroom; a fall is a terminal failure). Commands are set by the
  trainer every step, so the command manager never resamples. No pushes: the
  value certifies the deployed system, which the evaluations run without them.
  """
  cfg = motor_safety_env_cfg("filter", payload=payload, ambient=ambient)
  cfg.rewards = {
    "safety_margin": RewardTermCfg(func=mdp.safety_margin, weight=1.0),
  }
  cfg.scale_rewards_by_dt = False
  cfg.terminations["motor_overheat"] = TerminationTermCfg(func=mdp.motor_overheat)
  cfg.events.pop("push_robot", None)
  cfg.episode_length_s = 60.0
  twist = cfg.commands["twist"]
  assert isinstance(twist, vel_mdp.UniformVelocityCommandCfg)
  twist.resampling_time_range = (1.0e9, 1.0e9)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.debug_vis = False
  return cfg
