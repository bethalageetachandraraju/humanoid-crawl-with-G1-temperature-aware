"""G1 joint motors as identified DC motors with heat, overheat protection and a
motor safety filter.

THE PLANT (what the simulated robot does, per joint, every 5 ms physics step):

1. PD law on the set-point, as the G1 motor driver runs it:
   tau_pd = kp (q* - q) - kd dq.
2. Torque-speed envelope of the DC motor (mjlab's ``dc_motor_clip``): stall
   torque at zero speed falling linearly to zero at no-load speed, capped by the
   driver's current limit (``peak_torque``).
3. Firmware overheat protection: once the winding reaches ``T_TRIP`` the driver
   cuts the motor (zero torque) until it cools to ``T_RELEASE``. On the real G1
   this is the "winding overheated" motor error; a cut knee or ankle is a fall.
4. Two-node thermal model, winding (w) and housing (h), with cooling:
     I       = tau / (n_motors * gear * kt)          (current of each motor)
     P       = R25 (1 + alpha (T_w - 25)) I^2 + k_iron dq^2
     C_w dT_w/dt = P - (T_w - T_h) / R_wh
     C_h dT_h/dt = (T_w - T_h) / R_wh - G_ha (T_h - T_amb) + Q_drv
                   + sum_j G_c (T_h,j - T_h)       (neighbouring motors)
     G_ha = (1 + k_fan u_fan) (1 + k_vel |v_xy|) / R_ha
   Integrated with explicit Euler at the physics step. The G1 reports two
   temperatures per motor; we read them as T_w and T_h, quantized to 1 C.

   The cooling terms follow the whole-body lumped thermal networks identified on
   a quadruped by Lin et al. (ROBOT 2025) and used for thermal-aware RL by Wan et
   al. (arXiv 2605.27046) and Qian et al. (arXiv 2603.01631): a constant motor-
   driver heat, heat exchange between neighbouring motors, and a forced-
   convection correction that grows with the robot's speed through the air. On
   top of that the G1's "local air cooling" is modelled as a fan per motor whose
   duty ramps with the winding reading (off below ``FAN_ON``, full at
   ``FAN_FULL``) and raises the housing-to-air conductance by up to ``k_fan``.

Each env draws its own "true" parameters around the identified ones
(``PLANT_SPREAD``) and its own ambient temperature, so the policy and the filter
never know the plant exactly.

THE SAFETY FILTER (optional, ``filter_enabled``): clamps the PD torque every
physics step to [lo, hi] = feasibility bound intersected with a thermal bound.

* Feasibility: ``ENVELOPE_MARGIN`` times the identified torque-speed envelope.
  The policy never relies on driver saturation, which is the least well
  modelled part of a real actuator.
* Thermal: a discrete-time control barrier function on h = T_LIMIT - T_w. The
  allowed copper loss is
     P_allow = C_w (T_LIMIT - T_w) / TAU_CBF + (T_w - T_h) / R_wh - k_iron dq^2
  evaluated with the worst-case parameters of the identified uncertainty box
  (``FILTER_CONSERVATISM``) and the sensor quantization. Then
  h' >= (1 - dt / TAU_CBF) h, so T_w never crosses T_LIMIT for any plant inside
  the box. When the motor is cool the bound is far above the peak torque; at
  the limit it shrinks to the torque whose heat the housing can carry away,
  i.e. the motor's continuous rating at the current housing temperature.

The parameter values are a plausible identification of the four G1 motor
classes (see ``docs`` in this package's README): gears, peak torques, speeds,
armatures and PD gains are mjlab's G1 constants; torque constants, resistances
and thermal resistances/capacities are chosen so that each class has a
continuous torque of about a quarter to a third of its peak and winding / housing
time constants of tens of seconds / several minutes, typical of small air-cooled
joint actuators. They stand in for a real system identification
(``scripts/motor_safety/sysid_thermal.py`` fits them from logs).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCfg, ActuatorCmd
from mjlab.actuator.dc_actuator import (
  DcMotorActuator,
  DcMotorActuatorCfg,
  dc_motor_clip,
)
from mjlab.actuator.pd_actuator import pd_torque
from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1c
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  from mjlab.entity import Entity

# Temperatures, C.
T_REF = 25.0  # resistance reference temperature
T_LIMIT = 90.0  # winding rating: the safety constraint
T_TRIP = 105.0  # firmware cuts the motor
T_RELEASE = 85.0  # ... until the winding cools to this
TEMP_QUANT = 1.0  # sensor resolution
ALPHA_CU = 0.00393  # copper resistance temperature coefficient, 1/K

# Safety filter.
TAU_CBF = 2.0  # s, barrier decay time constant
ENVELOPE_MARGIN = 0.9  # fraction of the torque-speed envelope the filter allows

# Plant parameter spread around the identification: (relative half-width).
PLANT_SPREAD = {
  "r25": 0.08,
  "kt": 0.04,
  "c_w": 0.10,
  "r_wh": 0.10,
  "c_h": 0.10,
  "r_ha": 0.15,
  "k_iron": 0.20,
}
AMBIENT_RANGE = (20.0, 30.0)

# Cooling (plant only; the filter needs none of it, see thermal_torque_bound).
FAN_ON = 40.0  # C, winding reading where the motor fan starts
FAN_FULL = 60.0  # C, ... and where it runs at full duty
COOLING = {
  "k_fan": 0.6,  # full fan: housing-to-air conductance x 1.6
  "k_vel": 0.3,  # forced convection while moving: x (1 + 0.3 |v| / (m/s))
  "g_couple": 0.15,  # W/K between the housings of neighbouring motors
}
COOLING_SPREAD = {"k_fan": 0.2, "k_vel": 0.3, "g_couple": 0.3, "q_drv": 0.2}
# Motors that sit together and share heat through the structure, per side.
MOTOR_CLUSTERS = (
  ("hip_pitch", "hip_roll", "hip_yaw"),
  ("knee", "ankle_pitch", "ankle_roll"),  # the ankle motors sit in the shank
  ("waist_yaw", "waist_roll", "waist_pitch"),
  ("shoulder_pitch", "shoulder_roll", "shoulder_yaw"),
  ("elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"),
)


@dataclass(frozen=True)
class MotorParams:
  """One G1 motor class, joint-side quantities unless noted."""

  name: str
  gear: float
  n_motors: int  # motors driving the joint (2 for ankle and waist roll/pitch)
  peak_torque: float  # driver current limit, Nm
  stall_torque: float  # zero-speed torque at bus voltage, Nm
  no_load_speed: float  # rad/s
  kt: float  # motor torque constant incl. gearbox efficiency, Nm/A (motor side)
  r25: float  # winding resistance at 25 C, Ohm (I^2 R copper loss, per motor)
  c_w: float  # winding heat capacity, J/K
  r_wh: float  # winding -> housing, K/W
  c_h: float  # housing heat capacity, J/K
  r_ha: float  # housing -> ambient, K/W
  k_iron: float  # speed-dependent loss, W / (rad/s)^2 (joint speed, per motor)
  q_drv: float  # constant motor-driver heat into the housing, W
  armature: float
  stiffness: float
  damping: float

  def continuous_torque(
    self, t_amb: float = T_REF, t_w: float = T_LIMIT, cooling: float = 1.0
  ) -> float:
    """Joint torque that holds the winding at ``t_w`` in steady state (no speed,
    no driver heat). ``cooling`` multiplies the housing-to-air conductance (1 =
    passive, 1 + k_fan = fan at full duty)."""
    p = (t_w - t_amb) / (self.r_wh + self.r_ha / cooling)
    r = self.r25 * (1.0 + ALPHA_CU * (t_w - T_REF))
    return self.n_motors * self.gear * self.kt * math.sqrt(p / r)

  @property
  def tau_winding(self) -> float:
    return self.c_w * self.r_wh

  @property
  def tau_housing(self) -> float:
    return self.c_h * self.r_ha


def _gear(g: tuple[float, float, float]) -> float:
  return g[1] * g[2]


_M5020 = dict(
  kt=0.055, r25=0.25, c_w=18.0, r_wh=0.9, c_h=120.0, r_ha=2.6, k_iron=0.004, q_drv=1.0
)
_M7520 = dict(
  kt=0.10, r25=0.11, c_w=45.0, r_wh=0.55, c_h=320.0, r_ha=1.3, k_iron=0.012, q_drv=1.5
)
_M4010 = dict(
  kt=0.025, r25=0.9, c_w=6.0, r_wh=1.6, c_h=40.0, r_ha=5.0, k_iron=0.002, q_drv=0.5
)

MOTOR_5020 = MotorParams(
  name="5020",
  gear=_gear(g1c.GEARS_5020),
  n_motors=1,
  peak_torque=g1c.ACTUATOR_5020.effort_limit,
  stall_torque=1.4 * g1c.ACTUATOR_5020.effort_limit,
  no_load_speed=g1c.ACTUATOR_5020.velocity_limit,
  armature=g1c.ARMATURE_5020,
  stiffness=g1c.STIFFNESS_5020,
  damping=g1c.DAMPING_5020,
  **_M5020,
)
MOTOR_5020_DUAL = MotorParams(
  name="5020x2",
  gear=_gear(g1c.GEARS_5020),
  n_motors=2,
  peak_torque=2 * g1c.ACTUATOR_5020.effort_limit,
  stall_torque=2 * 1.4 * g1c.ACTUATOR_5020.effort_limit,
  no_load_speed=g1c.ACTUATOR_5020.velocity_limit,
  armature=2 * g1c.ARMATURE_5020,
  stiffness=2 * g1c.STIFFNESS_5020,
  damping=2 * g1c.DAMPING_5020,
  **_M5020,
)
MOTOR_7520_14 = MotorParams(
  name="7520-14",
  gear=_gear(g1c.GEARS_7520_14),
  n_motors=1,
  peak_torque=g1c.ACTUATOR_7520_14.effort_limit,
  stall_torque=1.4 * g1c.ACTUATOR_7520_14.effort_limit,
  no_load_speed=g1c.ACTUATOR_7520_14.velocity_limit,
  armature=g1c.ARMATURE_7520_14,
  stiffness=g1c.STIFFNESS_7520_14,
  damping=g1c.DAMPING_7520_14,
  **_M7520,
)
MOTOR_7520_22 = MotorParams(
  name="7520-22",
  gear=_gear(g1c.GEARS_7520_22),
  n_motors=1,
  peak_torque=g1c.ACTUATOR_7520_22.effort_limit,
  stall_torque=1.4 * g1c.ACTUATOR_7520_22.effort_limit,
  no_load_speed=g1c.ACTUATOR_7520_22.velocity_limit,
  armature=g1c.ARMATURE_7520_22,
  stiffness=g1c.STIFFNESS_7520_22,
  damping=g1c.DAMPING_7520_22,
  **_M7520,
)
MOTOR_4010 = MotorParams(
  name="4010",
  gear=_gear(g1c.GEARS_4010),
  n_motors=1,
  peak_torque=g1c.ACTUATOR_4010.effort_limit,
  stall_torque=1.4 * g1c.ACTUATOR_4010.effort_limit,
  no_load_speed=g1c.ACTUATOR_4010.velocity_limit,
  armature=g1c.ARMATURE_4010,
  stiffness=g1c.STIFFNESS_4010,
  damping=g1c.DAMPING_4010,
  **_M4010,
)

# Joint-name regex -> motor class, the same grouping as mjlab's G1 actuators.
G1_MOTOR_MAP: tuple[tuple[str, MotorParams], ...] = (
  (r".*_elbow_joint", MOTOR_5020),
  (r".*_shoulder_pitch_joint", MOTOR_5020),
  (r".*_shoulder_roll_joint", MOTOR_5020),
  (r".*_shoulder_yaw_joint", MOTOR_5020),
  (r".*_wrist_roll_joint", MOTOR_5020),
  (r".*_hip_pitch_joint", MOTOR_7520_14),
  (r".*_hip_yaw_joint", MOTOR_7520_14),
  (r"waist_yaw_joint", MOTOR_7520_14),
  (r".*_hip_roll_joint", MOTOR_7520_22),
  (r".*_knee_joint", MOTOR_7520_22),
  (r".*_wrist_pitch_joint", MOTOR_4010),
  (r".*_wrist_yaw_joint", MOTOR_4010),
  (r"waist_pitch_joint", MOTOR_5020_DUAL),
  (r"waist_roll_joint", MOTOR_5020_DUAL),
  (r".*_ankle_pitch_joint", MOTOR_5020_DUAL),
  (r".*_ankle_roll_joint", MOTOR_5020_DUAL),
)

_THERMAL_FIELDS = ("r25", "kt", "c_w", "r_wh", "c_h", "r_ha", "k_iron")


def motor_for_joint(name: str) -> MotorParams:
  for pattern, motor in G1_MOTOR_MAP:
    if re.fullmatch(pattern, name):
      return motor
  raise KeyError(f"no G1 motor class for joint {name!r}")


##
# Pure functions (CPU-testable).
##


def copper_resistance(r25: torch.Tensor, t_w: torch.Tensor) -> torch.Tensor:
  return r25 * (1.0 + ALPHA_CU * (t_w - T_REF))


def motor_power(
  tau: torch.Tensor, dq: torch.Tensor, t_w: torch.Tensor, p: dict[str, torch.Tensor]
) -> torch.Tensor:
  """Heat into each motor's winding, W."""
  current = tau / (p["n"] * p["gear"] * p["kt"])
  return copper_resistance(p["r25"], t_w) * current.square() + p["k_iron"] * dq.square()


def thermal_step(
  t_w: torch.Tensor,
  t_h: torch.Tensor,
  t_amb: torch.Tensor,
  power: torch.Tensor,
  p: dict[str, torch.Tensor],
  dt: float,
  ha_gain: torch.Tensor | float = 1.0,
  q_housing: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
  """One explicit-Euler step of the two-node model.

  ``ha_gain`` multiplies the housing-to-air conductance (fan, forced
  convection); ``q_housing`` is extra heat into the housing (driver heat, heat
  from neighbouring motors), W.
  """
  q_wh = (t_w - t_h) / p["r_wh"]
  q_ha = ha_gain * (t_h - t_amb) / p["r_ha"]
  t_w_next = t_w + dt * (power - q_wh) / p["c_w"]
  return t_w_next, t_h + dt * (q_wh - q_ha + q_housing) / p["c_h"]


def fan_duty(t_w_reading: torch.Tensor) -> torch.Tensor:
  """Fan duty in [0, 1] from the winding reading (a linear firmware fan curve)."""
  return ((t_w_reading - FAN_ON) / (FAN_FULL - FAN_ON)).clamp(0.0, 1.0)


def cluster_adjacency(names: list[str], device) -> torch.Tensor:
  """(n, n) 0/1 matrix: motors in the same ``MOTOR_CLUSTERS`` group and side."""

  def key(name: str):
    side = (
      "left"
      if name.startswith("left")
      else "right"
      if name.startswith("right")
      else "mid"
    )
    for c, group in enumerate(MOTOR_CLUSTERS):
      if any(f"{g}_joint" in name for g in group):
        return side, c
    return side, -1

  keys = [key(n) for n in names]
  a = torch.zeros(len(names), len(names), device=device)
  for i, ki in enumerate(keys):
    for j, kj in enumerate(keys):
      if i != j and ki == kj and ki[1] >= 0:
        a[i, j] = 1.0
  return a


def thermal_torque_bound(
  t_w_meas: torch.Tensor,
  t_h_meas: torch.Tensor,
  dq: torch.Tensor,
  p: dict[str, torch.Tensor],
  t_limit: float = T_LIMIT,
  tau_cbf: float = TAU_CBF,
  quant: float = TEMP_QUANT,
) -> torch.Tensor:
  """Largest |joint torque| that keeps h = t_limit - T_w a discrete-time barrier.

  ``p`` holds the CONSERVATIVE filter parameters (``filter_params``); ``t_*_meas``
  are quantized sensor readings, so the true winding is within quant / 2 of
  them. Returns a non-negative bound per joint.
  """
  half = 0.5 * quant
  t_w_hi = t_w_meas + half
  dt_wh = (t_w_meas - half) - (t_h_meas + half)
  # The worst case heat flow out of the winding: the larger resistance when the
  # winding is hotter than the housing, the smaller one when it is colder.
  q_out = torch.where(dt_wh >= 0, dt_wh / p["r_wh_hi"], dt_wh / p["r_wh_lo"])
  p_allow = p["c_w"] * (t_limit - t_w_hi) / tau_cbf + q_out - p["k_iron"] * dq.square()
  r = copper_resistance(p["r25"], t_w_hi)
  current = torch.sqrt(p_allow.clamp(min=0.0) / r)
  return p["n"] * p["gear"] * p["kt"] * current


BUDGET_HORIZONS = (2.0, 10.0, 30.0, 120.0)  # s


def budget_tables(
  p: dict[str, torch.Tensor], times: torch.Tensor
) -> dict[str, torch.Tensor]:
  """Winding response of the two-node model on a grid of ``times`` (n_t,), per
  joint, for parameters ``p`` (each (1, n_j)). With no fan and no neighbours,

      T_w(t) = ew T_w0 + eh T_h0 + ea (T_amb / R_ha + Q_drv) + sp P

  for a constant copper power P from t = 0. The model is linear with a
  non-negative impulse response (a Metzler, Hurwitz A), so the largest constant
  P keeping T_w <= T_lim on [0, H] also bounds any power profile below it.
  Returns ew, eh, ea, sp, each (n_t, n_j).
  """
  cw, ch, rwh, rha = (p[k][0] for k in ("c_w", "c_h", "r_wh", "r_ha"))
  n_j = cw.numel()
  a = torch.zeros(n_j, 2, 2, device=cw.device, dtype=cw.dtype)
  a[:, 0, 0] = -1.0 / (cw * rwh)
  a[:, 0, 1] = 1.0 / (cw * rwh)
  a[:, 1, 0] = 1.0 / (ch * rwh)
  a[:, 1, 1] = -(1.0 / rwh + 1.0 / rha) / ch
  at = a[None] * times[:, None, None, None]  # (n_t, n_j, 2, 2)
  e = torch.linalg.matrix_exp(at)
  eye = torch.eye(2, device=cw.device, dtype=cw.dtype)
  s = torch.linalg.solve(a[None].expand_as(e), e - eye)  # A^-1 (e^{At} - I)
  return {
    "ew": e[..., 0, 0],
    "eh": e[..., 0, 1],
    "ea": s[..., 0, 1] / ch,
    "sp": s[..., 0, 0] / cw,
  }


def horizon_power_budget(
  t_w: torch.Tensor,
  t_h: torch.Tensor,
  t_amb: torch.Tensor,
  q_drv: torch.Tensor,
  r_ha: torch.Tensor,
  tables: dict[str, torch.Tensor],
  masks: torch.Tensor,
  t_limit: float = T_LIMIT,
) -> torch.Tensor:
  """Largest constant copper power per motor that keeps the winding under
  ``t_limit`` for each horizon. ``masks`` (n_h, n_t) selects the grid times up
  to each horizon. Returns (n_env, n_h, n_j), clamped at 0."""
  free = (
    tables["ew"][None] * t_w[:, None]
    + tables["eh"][None] * t_h[:, None]
    + tables["ea"][None] * (t_amb / r_ha + q_drv)[:, None]
  )  # (n_env, n_t, n_j)
  ratio = (t_limit - free) / tables["sp"][None]
  big = torch.full_like(ratio, float("inf"))
  per_h = [torch.where(m[None, :, None], ratio, big).amin(1) for m in masks]
  return torch.stack(per_h, 1).clamp(min=0.0)


def envelope_bounds(
  dq: torch.Tensor, p: dict[str, torch.Tensor], margin: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
  """(lo, hi) of the torque-speed envelope at joint speed ``dq``, times margin.

  The same curve as mjlab's ``dc_motor_clip``.
  """
  big = torch.full_like(dq, 1.0e9)
  hi = dc_motor_clip(big, p["stall"], p["no_load"], p["peak"], dq)
  lo = dc_motor_clip(-big, p["stall"], p["no_load"], p["peak"], dq)
  return margin * lo, margin * hi


##
# The actuator.
##


@dataclass(kw_only=True)
class ThermalDcMotorActuatorCfg(DcMotorActuatorCfg):
  """All 29 G1 joints as identified DC motors (one actuator, per-joint params).

  The four switches select what the simulated motor does; the arms of the
  experiment differ only in them (``env_cfgs.py``).
  """

  envelope: bool = True
  """Apply the torque-speed envelope. False = a flat +-peak clamp (stock mjlab)."""
  protection: bool = True
  """Firmware overheat cut-off at ``T_TRIP``."""
  filter_enabled: bool = False
  """Run the motor safety filter (feasibility + thermal barrier)."""
  plant_spread: bool = True
  """Draw per-env plant parameters around the identification."""
  cooling: bool = True
  """Fans, forced convection, driver heat and motor-to-motor coupling."""
  init_temp: str = "random"
  """Initial temperatures at a reset: "random" (every motor anywhere in the safe
  set), "mixed" (one heat level per robot, 30 % cold: covers cold starts),
  "ambient" (a cold robot), or "keep" (no reset; long evaluations)."""
  ambient_range: tuple[float, float] = AMBIENT_RANGE
  """Room temperature drawn per env at a reset, C."""
  t_limit: float = T_LIMIT
  tau_cbf: float = TAU_CBF
  envelope_margin: float = ENVELOPE_MARGIN
  # DcMotorActuatorCfg fields, unused (per-joint values come from G1_MOTOR_MAP).
  stiffness: float = 0.0
  damping: float = 0.0
  saturation_effort: float = 1.0
  velocity_limit: float = 1.0
  effort_limit: float = 1.0
  target_names_expr: tuple[str, ...] = (".*",)

  def __post_init__(self) -> None:
    # Skip DcMotorActuatorCfg's scalar-limit warnings: limits are per joint.
    ActuatorCfg.__post_init__(self)
    if self.init_temp not in ("random", "mixed", "ambient", "keep"):
      raise ValueError(f"init_temp {self.init_temp!r}")

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> ThermalDcMotorActuator:
    return ThermalDcMotorActuator(self, entity, target_ids, target_names)


class ThermalDcMotorActuator(DcMotorActuator[ThermalDcMotorActuatorCfg]):
  """See the module docstring. State tensors are (num_envs, num_joints)."""

  def __init__(self, cfg, entity, target_ids, target_names) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.motors = [motor_for_joint(n) for n in target_names]

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    for name, motor in zip(target_names, self.motors, strict=True):
      self._mjs_actuators.append(
        create_motor_actuator(
          spec,
          name,
          effort_limit=motor.peak_torque,
          armature=motor.armature,
          transmission_type=self.cfg.transmission_type,
        )
      )

  def _col(self, attr: str) -> torch.Tensor:
    return torch.tensor(
      [float(getattr(m, attr)) for m in self.motors], device=self._dev
    )

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    self._dev = device
    n_env, n_j = data.nworld, len(self.motors)

    def full(attr: str) -> torch.Tensor:
      return self._col(attr).expand(n_env, n_j).clone()

    self.stiffness = full("stiffness")
    self.damping = full("damping")
    self.force_limit = full("peak_torque")
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self.default_force_limit = self.force_limit.clone()
    self.saturation_effort = full("stall_torque")
    self.velocity_limit_motor = full("no_load_speed")
    # Identified parameters (what the filter and the policy may know).
    ident = {k: self._col(k) for k in _THERMAL_FIELDS}
    ident["n"] = self._col("n_motors")
    ident["gear"] = self._col("gear")
    ident["peak"] = self._col("peak_torque")
    ident["stall"] = self._col("stall_torque")
    ident["no_load"] = self._col("no_load_speed")
    self.ident = ident
    # Conservative corner of the uncertainty box for the filter.
    s = PLANT_SPREAD
    self.filter_params = {
      "n": ident["n"],
      "gear": ident["gear"],
      "kt": ident["kt"] * (1 - s["kt"]),
      "r25": ident["r25"] * (1 + s["r25"]),
      "c_w": ident["c_w"] * (1 - s["c_w"]),
      "r_wh_hi": ident["r_wh"] * (1 + s["r_wh"]),
      "r_wh_lo": ident["r_wh"] * (1 - s["r_wh"]),
      "k_iron": ident["k_iron"] * (1 + s["k_iron"]),
    }
    # Horizon budget (worst corner, no fan): fastest-heating, worst-cooled motor.
    self.budget_params = {
      "c_w": ident["c_w"] * (1 - s["c_w"]),
      "c_h": ident["c_h"] * (1 - s["c_h"]),
      "r_wh": ident["r_wh"] * (1 + s["r_wh"]),
      "r_ha": ident["r_ha"] * (1 + s["r_ha"]),
    }
    times = torch.logspace(-1.3, math.log10(BUDGET_HORIZONS[-1]), 48, device=device)
    self.budget_table = budget_tables(self.budget_params, times)
    self.budget_masks = torch.stack([times <= h * (1 + 1e-6) for h in BUDGET_HORIZONS])
    # Per-env plant parameters, ambient and state.
    self.plant = {k: ident[k].expand(n_env, n_j).clone() for k in _THERMAL_FIELDS}
    for k in ("n", "gear"):
      self.plant[k] = ident[k].expand(n_env, n_j)
    self.adjacency = cluster_adjacency(list(self.target_names), device)
    self.degree = self.adjacency.sum(-1)
    self.cool = {
      k: torch.full((n_env, 1), v, device=device) for k, v in COOLING.items()
    }
    self.q_drv = self._col("q_drv").expand(n_env, n_j).clone()
    self.fan = torch.zeros(n_env, n_j, device=device)
    self.t_amb = torch.full((n_env, 1), T_REF, device=device)
    self.t_w = torch.full((n_env, n_j), T_REF, device=device)
    self.t_h = torch.full((n_env, n_j), T_REF, device=device)
    self.tripped = torch.zeros(n_env, n_j, dtype=torch.bool, device=device)
    self.tau_pd = torch.zeros(n_env, n_j, device=device)
    self.tau = torch.zeros(n_env, n_j, device=device)
    self.dq = torch.zeros(n_env, n_j, device=device)
    self.bound_lo = -self.force_limit.clone()
    self.bound_hi = self.force_limit.clone()
    self.thermal_bound = self.force_limit.clone()
    # Envs whose torque the filter clamps (all, by default). Lets one simulation
    # compare filtered and unfiltered controllers side by side.
    self.filter_mask = torch.ones(n_env, 1, dtype=torch.bool, device=device)
    # Per-motor torque cap as a fraction of peak (budget-conditioned policies):
    # set by a governor or randomized in training; 1 = no cap.
    self.torque_cap = torch.ones(n_env, n_j, device=device)
    # Episode accumulators read by metrics (reset with the env).
    self.clip_energy = torch.zeros(n_env, device=device)
    self.reset(None)

  # -- sensing ------------------------------------------------------------------

  def measured(self) -> tuple[torch.Tensor, torch.Tensor]:
    """The two temperature readings per motor, quantized like the G1's."""
    q = TEMP_QUANT
    return torch.round(self.t_w / q) * q, torch.round(self.t_h / q) * q

  def horizon_budget(self) -> torch.Tensor:
    """Torque each motor can sustain for 2, 10, 30 and 120 s without its winding
    passing the limit, as a fraction of peak torque, from the readings (worst
    case: rounded up, worst parameter corner, no fan). (n_env, n_h * n_j)."""
    t_w_m, t_h_m = self.measured()
    half = 0.5 * TEMP_QUANT
    p_bar = horizon_power_budget(
      t_w_m + half,
      t_h_m + half,
      self.t_amb,
      self.q_drv,
      self.budget_params["r_ha"],
      self.budget_table,
      self.budget_masks,
      self.cfg.t_limit,
    )
    fp = self.filter_params
    p_bar = (p_bar - fp["k_iron"] * self.dq.square()[:, None]).clamp(min=0.0)
    r = copper_resistance(fp["r25"], torch.full_like(self.t_w, self.cfg.t_limit))
    tau = fp["n"] * fp["gear"] * fp["kt"] * torch.sqrt(p_bar / r[:, None])
    return (tau / self.ident["peak"]).clamp(0.0, 1.0).flatten(1)

  def headroom(self) -> torch.Tensor:
    """Torque the filter allows as a fraction of peak, in [0, 1]: the thermal
    bound, or the torque cap when that is lower."""
    h = self.thermal_bound / self.ident["peak"]
    return torch.minimum(h, self.torque_cap).clamp(0.0, 1.0)

  # -- reset -------------------------------------------------------------------

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if not hasattr(self, "t_w"):
      return
    if env_ids is None:
      env_ids = slice(None)
    if isinstance(env_ids, slice):
      idx = torch.arange(self.t_w.shape[0], device=self._dev)[env_ids]
    else:
      idx = env_ids
    n, n_j = idx.numel(), self.t_w.shape[1]
    if n == 0:
      return
    dev = self._dev
    if self.cfg.plant_spread:
      for k, w in PLANT_SPREAD.items():
        u = torch.rand(n, n_j, device=dev) * 2 - 1
        self.plant[k][idx] = self.ident[k] * (1 + w * u)
      for k, v in COOLING.items():
        u = torch.rand(n, 1, device=dev) * 2 - 1
        self.cool[k][idx] = v * (1 + COOLING_SPREAD[k] * u)
      u = torch.rand(n, n_j, device=dev) * 2 - 1
      self.q_drv[idx] = self._col("q_drv") * (1 + COOLING_SPREAD["q_drv"] * u)
      lo, hi = self.cfg.ambient_range
      self.t_amb[idx] = lo + (hi - lo) * torch.rand(n, 1, device=dev)
    if self.cfg.init_temp == "ambient":
      self.t_h[idx] = self.t_amb[idx].expand(n, n_j)
      self.t_w[idx] = self.t_amb[idx].expand(n, n_j)
    elif self.cfg.init_temp == "random":
      amb = self.t_amb[idx]
      top = self.cfg.t_limit - 2.0
      t_h = amb + torch.rand(n, n_j, device=dev) * (top - 10.0 - amb)
      t_w = t_h + torch.rand(n, n_j, device=dev) * (top - t_h)
      self.t_h[idx], self.t_w[idx] = t_h, t_w
    elif self.cfg.init_temp == "mixed":
      # A robot that has worked is warm all over: one heat level per robot, motor
      # spread around it, and 30 % of robots cold (independent per-motor draws
      # never produce a cold robot, so a cold start was out of distribution).
      amb = self.t_amb[idx]
      top = self.cfg.t_limit - 2.0
      level = torch.rand(n, 1, device=dev) * (torch.rand(n, 1, device=dev) >= 0.3)
      spread = 0.7 + 0.3 * torch.rand(n, n_j, device=dev)
      t_h = amb + level * spread * (top - 10.0 - amb)
      t_w = t_h + level * torch.rand(n, n_j, device=dev) * (top - t_h)
      self.t_h[idx], self.t_w[idx] = t_h, t_w
    self.tripped[idx] = False
    self.torque_cap[idx] = 1.0
    self.clip_energy[idx] = 0.0

  # -- control -----------------------------------------------------------------

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    cfg = self.cfg
    assert self.stiffness is not None and self.damping is not None
    assert self.saturation_effort is not None and self.force_limit is not None
    assert self.velocity_limit_motor is not None
    dq = cmd.vel
    tau_pd = pd_torque(self.stiffness, self.damping, cmd)
    t_w_m, t_h_m = self.measured()
    self.thermal_bound = thermal_torque_bound(
      t_w_m, t_h_m, dq, self.filter_params, cfg.t_limit, cfg.tau_cbf
    )
    tau = tau_pd
    if cfg.filter_enabled:
      lo, hi = envelope_bounds(dq, self.ident, cfg.envelope_margin)
      cap = torch.minimum(self.thermal_bound, self.torque_cap * self.ident["peak"])
      lo = torch.maximum(lo, -cap)
      hi = torch.minimum(hi, cap)
      hi = torch.maximum(hi, lo)  # empty set only if beyond the envelope at speed
      self.bound_lo, self.bound_hi = lo, hi
      tau = torch.where(self.filter_mask, torch.clamp(tau, lo, hi), tau)
    if cfg.envelope:
      tau = dc_motor_clip(
        tau, self.saturation_effort, self.velocity_limit_motor, self.force_limit, dq
      )
    else:
      tau = torch.clamp(tau, -self.force_limit, self.force_limit)
    if cfg.protection:
      tau = torch.where(self.tripped, torch.zeros_like(tau), tau)
    self.tau_pd, self.tau, self.dq = tau_pd, tau, dq
    return tau

  def update(self, dt: float) -> None:
    super().update(dt)
    power = motor_power(self.tau, self.dq, self.t_w, self.plant)
    gain, q_h = 1.0, 0.0
    if self.cfg.cooling:
      t_w_m, _ = self.measured()
      self.fan = fan_duty(t_w_m)
      speed = self.entity.data.root_link_lin_vel_w[:, :2].norm(dim=-1, keepdim=True)
      gain = (1 + self.cool["k_fan"] * self.fan) * (1 + self.cool["k_vel"] * speed)
      q_couple = self.cool["g_couple"] * (
        self.t_h @ self.adjacency.T - self.t_h * self.degree
      )
      q_h = self.q_drv + q_couple
    self.t_w, self.t_h = thermal_step(
      self.t_w, self.t_h, self.t_amb, power, self.plant, dt, gain, q_h
    )
    if self.cfg.protection:
      self.tripped = (self.tripped & (self.t_w > T_RELEASE)) | (self.t_w >= T_TRIP)
    peak = self.ident["peak"]
    self.clip_energy += (((self.tau_pd - self.tau) / peak).square()).sum(-1) * dt
