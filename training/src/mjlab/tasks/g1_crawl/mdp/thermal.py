"""Motor-heat terms for the thermal fine-tune of the crawl and stand <-> crawl.

The first real-robot test (2026-10-29) tripped the controller's 85 C thermal stop
twice while the robot held still: 8.8 Nm on a shoulder roll for 29 s, 50 Nm on a
knee for 25 s. Nothing in training priced a steady 30-45 % load. These terms do:

* ``motor_heat_rate``: copper loss goes with torque^2, so the instant heat of
  each motor in units of its continuous rating is ``(tau / tau_cont)^2``. Small,
  always on; being convex it also prefers sharing a load between motors.
* ``motor_heat_sustained``: the same quantity averaged over the last couple of
  seconds (so a gait's torque peaks average out), penalised above a budget
  (``budget`` x continuous rating, RMS). Strongest while holding still.
* ``joint_pos_above``: keeps a joint off its end stop (the real knees reached
  2.88 rad at the bottom of every descent).
* ``floor_friction``: one floor per episode (tile to carpet), each limb's
  material on top, instead of every geom drawn on its own.
* ``reward_ramp``: fades new terms in after a fine-tune starts, so the first
  updates are not driven by a value function that has never seen them.

Continuous ratings come from ``g1_motor_safety.motor_model`` (estimates, not
measurements; the real motors heated about 2x faster than that model).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg
  from mjlab.managers.reward_manager import RewardTermCfg


def continuous_ratings(joints: tuple[str, ...]) -> dict[str, float]:
  """Joint name -> estimated continuous torque (Nm, 25 C ambient, 90 C winding)."""
  from mjlab.tasks.g1_motor_safety.motor_model import motor_for_joint

  return {name: motor_for_joint(name).continuous_torque() for name in joints}


def _rest_mask(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """True where the robot is meant to hold still: a zero crawl command, or a
  phase command that is holding either end (``rate == 0``)."""
  term = env.command_manager.get_term(command_name)
  rate = getattr(term, "rate", None)
  if rate is not None:
    return rate == 0.0
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.linalg.vector_norm(command[:, :3], dim=1) < 0.05


class _Ratings:
  """Actuator ids and continuous ratings, in the order of ``ratings``."""

  def __init__(self, env: ManagerBasedRlEnv, ratings: dict[str, float]):
    robot: Entity = env.scene["robot"]
    ids, names = robot.find_actuators(tuple(ratings), preserve_order=True)
    assert tuple(names) == tuple(ratings), names
    self.robot = robot
    self.ids = torch.tensor(ids, device=env.device)
    self.tau_cont = torch.tensor([ratings[n] for n in names], device=env.device)

  def load(self) -> torch.Tensor:
    """(tau / tau_cont)^2 per motor: instant heat in units of the rating."""
    tau = self.robot.data.actuator_force[:, self.ids]
    return (tau / self.tau_cont).square()


_RATINGS: dict[tuple[int, tuple[str, ...]], _Ratings] = {}


def motor_heat_rate(env: ManagerBasedRlEnv, ratings: dict[str, float]) -> torch.Tensor:
  """Penalty: sum over motors of (tau / tau_cont)^2."""
  key = (id(env), tuple(ratings))
  if key not in _RATINGS:
    _RATINGS[key] = _Ratings(env, ratings)
  return _RATINGS[key].load().sum(dim=1)


class motor_heat_sustained:  # noqa: N801 -- MJLab reward terms are lowercase.
  """Penalty: averaged (tau / tau_cont)^2 above ``budget``^2, summed over motors.

  The average is exponential with time constant ``tau_s``. At an episode's first
  step it starts at the instant value, as if the robot had been doing this for a
  while, so a short episode still pays for a held load. Scaled by
  ``moving_scale`` unless the command asks the robot to hold still.

  Params:
    ratings: Joint name -> continuous torque (``continuous_ratings``).
    command_name: Crawl velocity command or stand <-> crawl phase command.
    budget: Allowed RMS torque as a fraction of the continuous rating.
    tau_s: Averaging time constant (s).
    moving_scale: Weight multiplier while the robot is commanded to move.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.motors = _Ratings(env, cfg.params["ratings"])
    n, k = env.num_envs, len(self.motors.ids)
    self.avg = torch.zeros(n, k, device=env.device)
    self.fresh = torch.ones(n, dtype=torch.bool, device=env.device)
    self.alpha = env.step_dt / cfg.params.get("tau_s", 2.0)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    ratings: dict[str, float],
    command_name: str,
    budget: float = 0.6,
    tau_s: float = 2.0,
    moving_scale: float = 0.25,
  ) -> torch.Tensor:
    del ratings, tau_s
    load = self.motors.load()
    self.avg += self.alpha * (load - self.avg)
    self.avg[self.fresh] = load[self.fresh]
    self.fresh[:] = False
    excess = torch.relu(self.avg - budget**2).sum(dim=1)
    scale = torch.where(_rest_mask(env, command_name), 1.0, moving_scale)
    return excess * scale

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.fresh[ids] = True
    self.avg[ids] = 0.0


def motor_heat_rms_ratio(
  env: ManagerBasedRlEnv, ratings: dict[str, float]
) -> torch.Tensor:
  """Metric: the most loaded motor's |tau| / tau_cont (instant)."""
  key = (id(env), tuple(ratings))
  if key not in _RATINGS:
    _RATINGS[key] = _Ratings(env, ratings)
  return _RATINGS[key].load().max(dim=1).values.sqrt()


def limb_support(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  limb_names: tuple[str, ...],
  command_name: str,
  min_down: int = 3,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward: at least ``min_down`` limbs on the ground while moving.

  A trot stands on one diagonal pair at a time, so the torso hangs between a
  hand on one side and a foot on the other and the waist yaw carries the twist
  every step (35 Nm RMS, 1.5x its rating, at 0.3 m/s). A walk keeps three limbs
  down most of the time.
  """
  from mjlab.sensor import ContactSensor

  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  names = list(sensor.primary_names)
  ids = [names.index(name) for name in limb_names]
  down = (sensor.data.force[:, ids].norm(dim=-1) > 1.0).sum(dim=1)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = command[:, :3].abs().sum(dim=1) > command_threshold
  return (down >= min_down).float() * moving.float()


def relative_progress(
  env: ManagerBasedRlEnv, command_name: str, min_command: float = 0.05
) -> torch.Tensor:
  """Reward: share of the commanded speed achieved, in [0, 1], each part.

  Mean over the planar part (speed along [forward, lateral] / its magnitude)
  and the turn part (turn rate / command), each only when commanded above
  ``min_command``. The exponential kernels and ``forward_progress`` pay almost
  nothing for a slow command, so with heat priced a policy stands through 0.1
  m/s (the Cool2 crawl did). This pays a slow command as much as a fast one.
  """
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_xy = command[:, :2]
  speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
  vel_xy = robot.data.root_link_lin_vel_b[:, [2, 1]]
  along = (vel_xy * cmd_xy).sum(dim=-1) / speed.clamp(min=1e-6)
  planar = (along / speed.clamp(min=1e-6)).clamp(0.0, 1.0)
  turn_cmd = command[:, 2]
  turn = (
    robot.data.root_link_ang_vel_b[:, 0] * turn_cmd.sign()
  ) / turn_cmd.abs().clamp(min=1e-6)
  turn = turn.clamp(0.0, 1.0)
  use_p = (speed > min_command).float()
  use_t = (turn_cmd.abs() > min_command).float()
  total = planar * use_p + turn * use_t
  return total / (use_p + use_t).clamp(min=1.0)


def foot_slip_twist(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
  twist_weight: float = 0.2,
) -> torch.Tensor:
  """Penalty: planted feet sliding (m/s) or twisting (rad/s, x ``twist_weight``).

  Summed over the feet in contact. On a high-friction carpet a foot cannot be
  dragged or spun into place; it has to be lifted and set down.
  """
  from mjlab.sensor import ContactSensor

  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  names = list(sensor.primary_names)
  down = sensor.data.force[:, [names.index(b) for b in body_names]].norm(dim=-1) > 1.0
  robot: Entity = env.scene["robot"]
  ids, _ = robot.find_bodies(body_names, preserve_order=True)
  slip = robot.data.body_link_lin_vel_w[:, ids, :2].norm(dim=-1)
  twist = robot.data.body_link_ang_vel_w[:, ids, 2].abs()
  return ((slip + twist_weight * twist) * down.float()).sum(dim=1)


def bodies_touch_ground(
  env: ManagerBasedRlEnv, sensor_name: str, body_pattern: str, threshold: float = 10.0
) -> torch.Tensor:
  """Termination: any body matching ``body_pattern`` presses on the ground with
  more than ``threshold`` N (the chain test counts any such touch as a failure:
  torso, pelvis, arm links, the wrist braces)."""
  from mjlab.sensor import ContactSensor

  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  ids = [i for i, n in enumerate(sensor.primary_names) if re.fullmatch(body_pattern, n)]
  assert ids, body_pattern
  return (sensor.data.force[:, ids].norm(dim=-1) > threshold).any(dim=1)


def joint_speed_over(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, limit: float
) -> torch.Tensor:
  """Penalty: joint speed above ``limit`` (rad/s), summed over joints. Slow,
  deliberate limbs: the creepy crawl, and less iron loss and impact."""
  asset: Entity = env.scene[asset_cfg.name]
  qd = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.relu(qd.abs() - limit).sum(dim=1)


def joint_pos_above(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
  """Penalty: how far (rad) the selected joints are past ``threshold``, summed."""
  asset: Entity = env.scene[asset_cfg.name]
  q = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.relu(q - threshold).sum(dim=1)


_ROBOT = SceneEntityCfg("robot")


@requires_model_fields("geom_friction")
def floor_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  ranges: tuple[float, float],
  materials: dict[str, tuple[float, float]],
  clamp: tuple[float, float] = (0.15, 1.5),
  asset_cfg: SceneEntityCfg = _ROBOT,
  torsional: tuple[float, float] | None = None,
) -> None:
  """One floor per episode: friction ``floor x material``.

  ``floor`` is drawn per env in ``ranges`` (shared by every contact, as one floor
  is: glazed tile at the low end, carpet at the high end). Each geom matching a
  ``materials`` pattern multiplies it by its own draw from that pattern's range
  (rubber soles vs printed plastic tips). The robot's limb geoms have contact
  priority, so this is the contact's friction. ``ranges`` is the knob the
  robustness curriculum writes (its ``friction_low/high``).

  ``torsional``: also draw the twist friction (m, condim-4 contacts) per
  episode, shared by the floor and scaled by each geom's material draw: low on
  tile, high on carpet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  floor = torch.empty(n, 1, device=env.device).uniform_(*ranges)
  twist = (
    torch.empty(n, 1, device=env.device).uniform_(*torsional)
    if torsional is not None
    else None
  )
  names = list(asset.geom_names)
  field = env.sim.model.geom_friction
  for pattern, (lo, hi) in materials.items():
    local = [i for i, name in enumerate(names) if re.fullmatch(pattern, name)]
    assert local, f"no geom matches {pattern!r}"
    gids = asset.indexing.geom_ids[torch.tensor(local, device=env.device)]
    mat = torch.empty(n, len(local), device=env.device).uniform_(lo, hi)
    mu = (floor * mat).clamp(*clamp)
    rows, cols = torch.meshgrid(
      env_ids.to(torch.long), gids.to(torch.long), indexing="ij"
    )
    field[rows, cols, 0] = mu
    if twist is not None:
      field[rows, cols, 1] = (twist * mat).clamp(min=1e-4)


class reward_ramp:  # noqa: N801 -- MJLab curriculum terms are lowercase.
  """Fade reward terms in after a (resumed) run starts.

  Each term's weight is 0 for ``delay_iterations`` and reaches its target
  linearly over the next ``ramp_iterations``. Counted from the first iteration
  of this run: mjlab restores the global step counter on resume (after the
  curriculum's first call), so a jump of over 1000 iterations restarts the count.

  Params:
    weights: Reward term name -> target weight.
    delay_iterations: Iterations at weight 0 (critic catches up first).
    ramp_iterations: Iterations from 0 to the target.
    steps_per_iteration: Env steps per learning iteration.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    self.terms = {
      name: env.reward_manager.get_term_cfg(name) for name in cfg.params["weights"]
    }
    self.start: int | None = None
    self.last = -1
    for term in self.terms.values():
      term.weight = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    weights: dict[str, float],
    delay_iterations: int = 50,
    ramp_iterations: int = 500,
    steps_per_iteration: int = 24,
  ) -> dict[str, torch.Tensor]:
    del env_ids
    iteration = env.common_step_counter // steps_per_iteration
    jumped = iteration - self.last > 1000 or iteration < self.last
    if self.start is None or jumped:
      self.start = iteration
    self.last = iteration
    t = (iteration - self.start - delay_iterations) / max(ramp_iterations, 1)
    frac = min(max(t, 0.0), 1.0)
    for name, term in self.terms.items():
      term.weight = weights[name] * frac
    return {"fraction": torch.tensor(frac, device=env.device)}
