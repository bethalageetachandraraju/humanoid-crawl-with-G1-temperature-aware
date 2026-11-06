"""Plug-in motor safety for any mjlab G1 task: one config transform, no task code.

    cfg = motor_safe(any_g1_env_cfg)                  # ours: filter in the loop
    cfg = motor_safe(any_g1_env_cfg, mode="penalty")  # soft-constraint baseline
    cfg = motor_safe(any_g1_env_cfg, mode="ideal")    # today's practice
    cfg = motor_safe(any_g1_env_cfg, mode="budget")   # ours + thermal foresight
    cfg = motor_safe(any_g1_env_cfg, mode="lagrangian")  # PID-Lagrangian baseline
    cfg = motor_safe(any_g1_env_cfg, mode="cat")      # constraints-as-terminations
    cfg = motor_safe(any_g1_env_cfg, mode="caps")     # ours, budget-conditioned

What the transform does, whatever the task (walking, tracking a dance, ...):

1. **Motors.** Every joint gets the identified motor (``ThermalDcMotorActuatorCfg``):
   torque-speed envelope, winding/housing heat with fan cooling, firmware cut-off.
   ``ideal`` keeps the heat simulated but without consequence (flat torque clamp,
   no cut-off), which is how policies are trained today.
2. **The clip filter** (``plugin`` only): the per-step thermal barrier + 90 %
   envelope clamp on the torque. It sits in the environment, so PPO trains on the
   filtered closed loop and needs no change: the action it stores is the policy's
   own, and the filter is part of the dynamics it learns.
3. **Observations** (``penalty`` and ``plugin``): the two temperature readings and
   the thermal headroom of every motor, appended to the ``actor`` and ``critic``
   groups, so the policy can learn to keep its budget. ``budget`` adds the horizon
   budget: the torque each motor can sustain for 2, 10, 30 and 120 s from its
   readings (``ThermalDcMotorActuator.horizon_budget``), the foresight the one-step
   filter lacks.
4. **Reward** (one term, task-independent): ``plugin`` pays a small penalty on the
   torque the filter holds back; ``penalty`` pays a squared hinge on hot windings.
5. **Temperatures at reset** are drawn anywhere in the safe range, so short
   episodes still see hot motors; rooms are 20-30 C.

The task's own rewards, terminations, commands and observations are untouched.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.g1_motor_safety import mdp
from mjlab.tasks.g1_motor_safety.motor_model import ThermalDcMotorActuatorCfg

MODES = ("ideal", "penalty", "plugin", "budget", "lagrangian", "cat", "caps")
_FLAGS = {
  "ideal": dict(envelope=False, protection=False, filter_enabled=False),
  "penalty": dict(envelope=True, protection=True, filter_enabled=False),
  "plugin": dict(envelope=True, protection=True, filter_enabled=True),
  "budget": dict(envelope=True, protection=True, filter_enabled=True),
  "lagrangian": dict(envelope=True, protection=True, filter_enabled=False),
  "cat": dict(envelope=True, protection=True, filter_enabled=False),
  "caps": dict(envelope=True, protection=True, filter_enabled=True),
}


def motor_safe(
  cfg: ManagerBasedRlEnvCfg,
  mode: str = "plugin",
  init_temp: str = "random",
  ambient: tuple[float, float] | None = None,
  weight: float = 1.0,
  observe: bool | None = None,
) -> ManagerBasedRlEnvCfg:
  """Return ``cfg`` (modified in place) with the motor-safety plug-in applied.

  ``observe`` defaults to True except in ``ideal`` mode, whose policy sees only
  the task's own observations (a stock policy).
  """
  if mode not in MODES:
    raise ValueError(f"mode {mode!r} not in {MODES}")
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  flags = _FLAGS[mode]
  motor = ThermalDcMotorActuatorCfg(
    init_temp=init_temp,
    envelope=flags["envelope"],
    protection=flags["protection"],
    filter_enabled=flags["filter_enabled"],
  )
  if ambient is not None:
    motor.ambient_range = ambient
  robot.articulation = replace(robot.articulation, actuators=(motor,))

  if observe if observe is not None else mode != "ideal":
    for group in ("actor", "critic"):
      terms = cfg.observations[group].terms
      terms["motor_temperatures"] = ObservationTermCfg(func=mdp.motor_temperatures)
      terms["motor_headroom"] = ObservationTermCfg(func=mdp.motor_headroom)
      if mode == "budget":
        terms["motor_budget"] = ObservationTermCfg(func=mdp.motor_budget)

  if mode in ("plugin", "budget", "caps"):
    cfg.rewards["filter_intervention"] = RewardTermCfg(
      func=mdp.filter_intervention, weight=-weight
    )
  elif mode == "penalty":
    cfg.rewards["winding_heat"] = RewardTermCfg(
      func=mdp.winding_heat_penalty, weight=-weight
    )
  elif mode == "lagrangian":
    cfg.rewards["pid_lagrangian"] = RewardTermCfg(func=mdp.pid_lagrangian, weight=1.0)
  elif mode == "cat":
    cfg.terminations["thermal_cat"] = TerminationTermCfg(func=mdp.thermal_cat)
  if mode == "caps":
    cfg.events["torque_caps"] = EventTermCfg(
      func=mdp.randomize_torque_caps, mode="interval", interval_range_s=(2.0, 10.0)
    )

  cfg.metrics["max_winding_temp"] = MetricsTermCfg(
    func=mdp.max_winding_temp, reduce="max"
  )
  cfg.metrics["over_rating"] = MetricsTermCfg(func=mdp.over_rating)
  cfg.metrics["motor_cut"] = MetricsTermCfg(func=mdp.motor_cut, reduce="max")
  cfg.metrics["torque_held_back"] = MetricsTermCfg(func=mdp.torque_held_back)
  return cfg
