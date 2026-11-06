"""Observation, reward and metric terms that read the G1 motor model."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.g1_motor_safety.motor_model import (
  T_LIMIT,
  T_REF,
  ThermalDcMotorActuator,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TEMP_SCALE = 50.0  # observation: (T - 25) / 50


def motor(env: ManagerBasedRlEnv) -> ThermalDcMotorActuator:
  for act in env.scene["robot"].actuators:
    if isinstance(act, ThermalDcMotorActuator):
      return act
  raise RuntimeError("no ThermalDcMotorActuator on the robot")


##
# Observations.
##


def motor_temperatures(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The two quantized temperature readings of every joint motor (58)."""
  t_w, t_h = motor(env).measured()
  return torch.cat([t_w - T_REF, t_h - T_REF], dim=-1) / TEMP_SCALE


def motor_headroom(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Thermal torque bound / peak torque per joint (29), computed from the readings
  and the identified model: how much torque the motor can still give before its
  winding would have to be held back."""
  return motor(env).headroom()


##
# Rewards.
##


def motor_budget(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Torque each motor can sustain for 2, 10, 30 and 120 s, as a fraction of its
  peak torque (4 x 29): the thermal foresight the one-step headroom lacks."""
  return motor(env).horizon_budget()


def thermal_cost(env: ManagerBasedRlEnv, t_soft: float = 88.0) -> torch.Tensor:
  """Constraint cost per step: winding degrees above ``t_soft``, summed over the
  motors, in units of the 2 C band up to the rating."""
  return ((motor(env).t_w - t_soft).clamp(min=0.0) / (T_LIMIT - t_soft)).sum(-1)


class pid_lagrangian:
  """PPO-Lagrangian as a reward term, with Stooke et al.'s PID multiplier (ICML
  2020): returns ``-lambda * c(s)`` with c = ``thermal_cost``. Every ``period``
  steps (one PPO rollout) lambda is updated from the batch-mean cost J against
  the allowed mean cost ``limit``: lambda = max(0, kp e + ki sum(e) + kd de+),
  e = J - limit. Task-agnostic: it reads only the motors."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    p = cfg.params
    self.kp, self.ki, self.kd = p.get("kp", 1.0), p.get("ki", 0.1), p.get("kd", 0.0)
    self.limit = p.get("limit", 0.02)
    self.period = int(p.get("period", 24))
    self.t_soft = p.get("t_soft", 88.0)
    self.lam = torch.zeros((), device=env.device)
    self.integral = torch.zeros((), device=env.device)
    self.prev = torch.zeros((), device=env.device)
    self.acc = torch.zeros((), device=env.device)
    self.n = 0

  def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
    del kwargs
    c = thermal_cost(env, self.t_soft)
    self.acc += c.mean()
    self.n += 1
    if self.n >= self.period:
      j = self.acc / self.n
      e = j - self.limit
      self.integral = (self.integral + e).clamp(min=0.0)
      deriv = (j - self.prev).clamp(min=0.0)
      self.prev = j
      self.lam = (self.kp * e + self.ki * self.integral + self.kd * deriv).clamp(
        min=0.0
      )
      self.acc = torch.zeros_like(self.acc)
      self.n = 0
    return -self.lam * c


def thermal_cat(
  env: ManagerBasedRlEnv, t_soft: float = 88.0, p_max: float = 0.25
) -> torch.Tensor:
  """Constraints as terminations (Chane-Sane et al. 2024): end the episode with
  probability p_max * clip((hottest winding - t_soft) / (T_LIMIT - t_soft), 0, 1)."""
  t = motor(env).t_w.max(-1).values
  delta = p_max * ((t - t_soft) / (T_LIMIT - t_soft)).clamp(0.0, 1.0)
  return torch.rand_like(delta) < delta


#: Limb groups for torque-cap randomization.
CAP_GROUPS = (
  r"^left_(hip|knee|ankle)",
  r"^right_(hip|knee|ankle)",
  r"^waist",
  r"^left_(shoulder|elbow|wrist)",
  r"^right_(shoulder|elbow|wrist)",
)


def randomize_torque_caps(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  p_none: float = 0.5,
  p_global: float = 0.25,
  global_range: tuple[float, float] = (0.4, 1.0),
  group_range: tuple[float, float] = (0.3, 1.0),
) -> None:
  """Interval event: new per-motor torque caps (fractions of peak). Half the time
  none; a quarter one global factor; a quarter one limb group capped. The policy
  sees the cap through ``motor_headroom`` and learns how to degrade under it."""
  m = motor(env)
  dev = m.t_w.device
  ids = torch.arange(env.num_envs, device=dev) if env_ids is None else env_ids
  n, n_j = ids.numel(), m.t_w.shape[1]
  if n == 0:
    return
  masks = torch.tensor(
    [[re.search(g, nm) is not None for nm in m.target_names] for g in CAP_GROUPS],
    device=dev,
  )
  caps = torch.ones(n, n_j, device=dev)
  u = torch.rand(n, device=dev)
  glob = (u >= p_none) & (u < p_none + p_global)
  grp = u >= p_none + p_global
  lo, hi = global_range
  caps[glob] = (lo + (hi - lo) * torch.rand(int(glob.sum()), 1, device=dev)).expand(
    -1, n_j
  )
  k = int(grp.sum())
  if k:
    g = torch.randint(0, len(CAP_GROUPS), (k,), device=dev)
    lo, hi = group_range
    val = lo + (hi - lo) * torch.rand(k, 1, device=dev)
    caps[grp] = torch.where(masks[g], val, torch.ones_like(val))
  m.torque_cap[ids] = caps


def winding_heat_penalty(env: ManagerBasedRlEnv, t_soft: float = 75.0) -> torch.Tensor:
  """Squared hinge on the winding readings above ``t_soft``, normalised so it is 1
  per joint at the rating T_LIMIT (the soft-constraint baseline)."""
  t_w, _ = motor(env).measured()
  x = ((t_w - t_soft) / (T_LIMIT - t_soft)).clamp(min=0.0)
  return x.square().sum(-1)


def filter_intervention(env: ManagerBasedRlEnv) -> torch.Tensor:
  """How much the motor model held back the PD torque this step, squared and
  normalised by the peak torque (filter clamp, envelope and cut-off together)."""
  m = motor(env)
  return (((m.tau_pd - m.tau) / m.ident["peak"]).square()).sum(-1)


##
# Metrics.
##


def max_winding_temp(env: ManagerBasedRlEnv) -> torch.Tensor:
  return motor(env).t_w.max(dim=-1).values


def over_rating(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 while any winding is above the rating T_LIMIT."""
  return (motor(env).t_w > T_LIMIT).any(dim=-1).float()


def motor_cut(env: ManagerBasedRlEnv) -> torch.Tensor:
  """1 while any motor is cut by the firmware overheat protection."""
  return motor(env).tripped.any(dim=-1).float()


def torque_held_back(env: ManagerBasedRlEnv) -> torch.Tensor:
  return filter_intervention(env)


def mean_headroom(env: ManagerBasedRlEnv) -> torch.Tensor:
  return motor(env).headroom().min(dim=-1).values


##
# Events.
##


@requires_model_fields("body_mass", recompute=RecomputeLevel.set_const)
def payload_groups(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  payloads: tuple[float, ...],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Startup event: env i carries ``payloads[i % len(payloads)]`` kg on the body
  (evaluation groups). Model-mass edits must happen as an event, before the
  simulator captures its CUDA graphs."""
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  group = torch.tensor(payloads, device=env.device)[ids % len(payloads)]
  for p in payloads:
    sub = ids[group == p]
    if sub.numel():
      dr.body_mass(env, sub, ranges=(p, p), operation="add", asset_cfg=asset_cfg)


##
# Hamilton-Jacobi motor safety (safety_ppo.py).
##

G_SCALE = 10.0  # C of winding headroom per unit of margin
G_CLAMP = 3.0
FALL_ANCHOR = -1.0


def motor_margin(env: ManagerBasedRlEnv) -> torch.Tensor:
  """g(s) = min_j (T_LIMIT - T_w,j) / G_SCALE, clamped: < 0 once any winding is
  past its rating. True winding temperatures (the physical failure set)."""
  g = (T_LIMIT - motor(env).t_w).min(dim=-1).values / G_SCALE
  return g.clamp(-G_CLAMP, G_CLAMP)


def fallen(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The env's own fall terminations: torso tilt past 70 deg or pelvis below 0.45 m."""
  d = env.scene["robot"].data
  up = -d.projected_gravity_b[:, 2]
  return (up < math.cos(math.radians(70.0))) | (d.root_link_pos_w[:, 2] < 0.45)


def safety_margin(env: ManagerBasedRlEnv, fall_is_failure: bool = True) -> torch.Tensor:
  """The reward channel of the HJ safety task: the motor margin g(s).

  With ``fall_is_failure`` a fall, which ends the episode, is scored as a
  failure (``FALL_ANCHOR``) at that terminal step. No tilt or height term enters
  the margin anywhere else; without it the motor-safest behaviour is to go limp
  and collapse (zero torque, no heat), which the value would certify as safe.
  """
  g = motor_margin(env)
  if fall_is_failure:
    g = torch.where(fallen(env), torch.minimum(g, torch.full_like(g, FALL_ANCHOR)), g)
  return g


def motor_overheat(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Termination: any winding at or past the rating (g < 0)."""
  return (motor(env).t_w >= T_LIMIT).any(dim=-1)
