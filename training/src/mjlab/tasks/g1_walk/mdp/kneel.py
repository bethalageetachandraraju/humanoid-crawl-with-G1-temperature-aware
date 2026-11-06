"""Take a knee and stand back up: one policy, a phase that goes both ways.

The same residual-on-a-schedule recipe as the stand-to-crawl transition
(``transition.py``), for a one-knee kneel reached by a slow reverse lunge: shift
the weight onto the left foot, step the right foot back onto its toes, lower
the right knee to the floor, hold (``kneel_keyframes.py``, checked with forward
kinematics, the centre of mass inside the support area at every keyframe).
Standing up is the same path backwards.

The phase ``phi`` (0 standing, 1 kneeling) and its signed rate are the command.
Each episode draws a timeline: stand, go down over 4-8 s, hold the kneel, and
then (most episodes) stand back up over 4-8 s and hold the stand. Some episodes
start kneeling, so standing up and long kneeling holds get their share of
samples. Pushes come from the robustness curriculum, holds included.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.g1_crawl.mdp.robustness import RobustnessRamp, crawl_robustness
from mjlab.tasks.g1_walk.mdp.kneel_keyframes import (
  KNEEL_KEYFRAME_TABLE,
  KNEEL_SWING_WINDOW,
)
from mjlab.tasks.g1_walk.mdp.transition import smoothstep
from mjlab.tasks.g1_walk.robot import JOINTS_29, STAND_HEIGHT, STAND_POSE_29
from mjlab.utils.lab_api.math import quat_mul, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

KNEE_BODY = "right_knee_link"
"""The kneeling knee; its shin capsule's upper end rests on the floor."""
RIGHT_FOOT_SITE = "right_foot"
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class KneelKeyframe:
  phi: float
  height: float
  """Pelvis height (m)."""
  pitch_deg: float
  """Base pitch about +Y (deg)."""
  knee_height: float
  """Right knee joint origin height (m); ~0.045 with the knee on the floor."""
  rfoot_height: float
  """Right foot sole site height (m)."""
  joint_pos: dict[str, float]


KNEEL_KEYFRAMES: tuple[KneelKeyframe, ...] = tuple(
  KneelKeyframe(**k) for k in KNEEL_KEYFRAME_TABLE
)
assert KNEEL_KEYFRAMES[0].phi == 0.0 and KNEEL_KEYFRAMES[-1].phi == 1.0
assert all(
  a.phi < b.phi for a, b in zip(KNEEL_KEYFRAMES, KNEEL_KEYFRAMES[1:], strict=False)
)
assert math.isclose(KNEEL_KEYFRAMES[0].height, STAND_HEIGHT, abs_tol=1e-4)
assert all(
  math.isclose(KNEEL_KEYFRAMES[0].joint_pos[j], STAND_POSE_29[j], abs_tol=1e-5)
  for j in JOINTS_29
), "the first kneel keyframe must be the stand pose"
KNEEL_POSE_29: dict[str, float] = dict(KNEEL_KEYFRAMES[-1].joint_pos)
KNEEL_HEIGHT = KNEEL_KEYFRAMES[-1].height
KNEEL_PITCH_DEG = KNEEL_KEYFRAMES[-1].pitch_deg


class KneelSchedule:
  """References as functions of the phase: smoothstep between keyframes."""

  def __init__(
    self,
    device: torch.device | str,
    keyframes: tuple[KneelKeyframe, ...] = KNEEL_KEYFRAMES,
    joints: tuple[str, ...] = JOINTS_29,
  ) -> None:
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    self.joints = joints
    self.phis = t([k.phi for k in keyframes])
    self.heights = t([k.height for k in keyframes])
    self.pitches = t([math.radians(k.pitch_deg) for k in keyframes])
    self.knees = t([k.knee_height for k in keyframes])
    self.rfeet = t([k.rfoot_height for k in keyframes])
    self.q = t([[k.joint_pos[j] for j in joints] for k in keyframes])
    self.q_stand, self.q_kneel = self.q[0], self.q[-1]

  def _lerp(self, table: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    idx = (torch.searchsorted(self.phis, phi.contiguous(), right=True) - 1).clamp(
      0, len(self.phis) - 2
    )
    span = self.phis[idx + 1] - self.phis[idx]
    s = smoothstep((phi - self.phis[idx]) / span)
    a, b = table[idx], table[idx + 1]
    return a + (b - a) * (s[:, None] if table.dim() == 2 else s)

  def joint_pos(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.q, phi)

  def height(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.heights, phi)

  def gravity(self, phi: torch.Tensor) -> torch.Tensor:
    pitch = self._lerp(self.pitches, phi)
    zeros = torch.zeros_like(pitch)
    return torch.stack([torch.sin(pitch), zeros, -torch.cos(pitch)], dim=1)

  def knee_height(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.knees, phi)

  def rfoot_height(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.rfeet, phi)


# Phase windows (from the keyframes): the right foot is in the air between the
# weight shift and its placement behind; the knee is down at the end.
SWING_WINDOW = KNEEL_SWING_WINDOW
KNEE_DOWN_PHASE = 0.97


@dataclass(kw_only=True)
class KneelPhaseCommandCfg(CommandTermCfg):
  entity_name: str
  wait_s: tuple[float, float] = (0.5, 1.5)
  """Standing before going down."""
  down_s: tuple[float, float] = (4.0, 8.0)
  hold_s: tuple[float, float] = (1.5, 5.0)
  """Kneeling hold before standing up (also the first hold of kneel starts)."""
  up_s: tuple[float, float] = (4.0, 8.0)
  stand_up_prob: float = 0.8
  """Fraction of episodes that stand back up; the rest stay kneeling."""

  def build(self, env: ManagerBasedRlEnv) -> KneelPhaseCommand:
    return KneelPhaseCommand(self, env)


class KneelPhaseCommand(CommandTerm):
  """Command ``[phi, dphi/dt]`` plus the reference schedules at ``phi``."""

  cfg: KneelPhaseCommandCfg

  def __init__(self, cfg: KneelPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    n, dev = self.num_envs, self.device
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    self.phase, self.rate = z(), z()
    self.t_wait, self.t_down, self.t_hold, self.t_up = z(), z() + 1, z(), z() + 1
    self.start_kneel = torch.zeros(n, dtype=torch.bool, device=dev)
    """Set by ``reset_kneel_start`` before the command resamples."""
    self.will_stand = torch.ones(n, dtype=torch.bool, device=dev)
    self.command_buf = torch.zeros(n, 2, device=dev)
    self.manual = False
    self.manual_phase, self.manual_rate = z(), z()

    ids, names = self.robot.find_joints(JOINTS_29, preserve_order=True)
    assert tuple(names) == JOINTS_29, names
    self.joint_ids = torch.tensor(ids, device=dev)
    self.knee_id = self.robot.find_bodies([KNEE_BODY])[0][0]
    self.rfoot_id = self.robot.find_sites([RIGHT_FOOT_SITE])[0][0]
    self.schedule = KneelSchedule(dev)
    self.ref_joint_pos = self.schedule.q_stand.repeat(n, 1)
    self.ref_height = torch.full((n,), STAND_HEIGHT, device=dev)
    self.ref_gravity = self.schedule.gravity(z())
    self.ref_knee_height = self.schedule.knee_height(z())
    self.ref_rfoot_height = self.schedule.rfoot_height(z())
    self.err_stand, self.err_kneel = z(), z()
    """RMS joint error (rad) to the stand and kneel poses, refreshed every step."""
    for name in ("phase_end", "end_err_rad", "height_err_m", "knee_err_m"):
      self.metrics[name] = z()

  @property
  def command(self) -> torch.Tensor:
    return self.command_buf

  @property
  def hold_kneel(self) -> torch.Tensor:
    return (self.phase >= 1.0) & (self.rate == 0.0)

  @property
  def hold_stand(self) -> torch.Tensor:
    return (self.phase <= 0.0) & (self.rate == 0.0)

  @property
  def end_err(self) -> torch.Tensor:
    """Joint error to the pose being held (stand or kneel), else to the nearer."""
    return torch.where(self.phase >= 0.5, self.err_kneel, self.err_stand)

  def manual_control(self) -> None:
    """``manual_phase`` / ``manual_rate`` drive the command (play, switcher)."""
    self.manual = True

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.manual:
      return
    r = torch.empty(len(env_ids), device=self.device)
    self.t_wait[env_ids] = r.uniform_(*self.cfg.wait_s)
    self.t_down[env_ids] = r.uniform_(*self.cfg.down_s)
    self.t_hold[env_ids] = r.uniform_(*self.cfg.hold_s)
    self.t_up[env_ids] = r.uniform_(*self.cfg.up_s)
    self.will_stand[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.stand_up_prob

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids  # Pure function of the state; refreshing every env is safe.
    if self.manual:
      self.phase = self.manual_phase.clamp(0.0, 1.0)
      self.rate = self.manual_rate.clone()
    else:
      t = self._env.episode_length_buf.float() * self._env.step_dt
      kneel0 = self.start_kneel
      down = ((t - self.t_wait) / self.t_down).clamp(0.0, 1.0)
      down = torch.where(kneel0, torch.ones_like(down), down)
      t_up0 = torch.where(kneel0, self.t_hold, self.t_wait + self.t_down + self.t_hold)
      up = ((t - t_up0) / self.t_up).clamp(0.0, 1.0) * self.will_stand.float()
      self.phase = down - up
      going_down = ~kneel0 & (t >= self.t_wait) & (t < self.t_wait + self.t_down)
      going_up = self.will_stand & (t >= t_up0) & (t < t_up0 + self.t_up)
      self.rate = going_down.float() / self.t_down - going_up.float() / self.t_up
    self.command_buf[:, 0] = self.phase
    self.command_buf[:, 1] = self.rate
    s = self.schedule
    self.ref_joint_pos = s.joint_pos(self.phase)
    self.ref_height = s.height(self.phase)
    self.ref_gravity = s.gravity(self.phase)
    self.ref_knee_height = s.knee_height(self.phase)
    self.ref_rfoot_height = s.rfoot_height(self.phase)
    q = self.robot.data.joint_pos[:, self.joint_ids]
    self.err_stand = (q - s.q_stand).square().mean(dim=1).sqrt()
    self.err_kneel = (q - s.q_kneel).square().mean(dim=1).sqrt()

  def _update_metrics(self) -> None:
    self.metrics["phase_end"][:] = self.phase
    self.metrics["end_err_rad"][:] = self.end_err
    self.metrics["height_err_m"][:] = (
      self.robot.data.root_link_pos_w[:, 2] - self.ref_height
    ).abs()
    self.metrics["knee_err_m"][:] = (
      self.robot.data.body_link_pos_w[:, self.knee_id, 2] - self.ref_knee_height
    ).abs()


def _term(env: ManagerBasedRlEnv, command_name: str) -> KneelPhaseCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, KneelPhaseCommand)
  return term


##
# Reset into the kneel.
##


def reset_kneel_start(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  prob: float,
  joint_noise: float = 0.03,
) -> None:
  """Start a fraction of episodes kneeling (after the standing reset events)."""
  cmd = _term(env, command_name)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  r = torch.rand(len(env_ids), device=env.device)
  kneel = env_ids[r < prob]
  cmd.start_kneel[env_ids] = False
  if len(kneel) == 0:
    return
  cmd.start_kneel[kneel] = True
  robot = cmd.robot
  env.sim.forward()  # The standing reset wrote qpos; refresh the root pose.
  state = torch.cat(
    [
      robot.data.root_link_pos_w[kneel],
      robot.data.root_link_quat_w[kneel],
      torch.zeros(len(kneel), 6, device=env.device),
    ],
    dim=1,
  )
  half = math.radians(KNEEL_PITCH_DEG) / 2
  pitch = torch.tensor(
    [math.cos(half), 0.0, math.sin(half), 0.0], device=env.device
  ).expand(len(kneel), 4)
  state[:, 2] = env.scene.env_origins[kneel, 2] + KNEEL_HEIGHT + 0.01
  state[:, 3:7] = quat_mul(yaw_quat(state[:, 3:7]), pitch)
  state[:, 7:] = 0.0
  robot.write_root_state_to_sim(state, env_ids=kneel)
  q = cmd.schedule.q_kneel.expand(len(kneel), -1)
  q = q + joint_noise * (2 * torch.rand_like(q) - 1)
  robot.write_joint_state_to_sim(
    q, torch.zeros_like(q), joint_ids=cmd.joint_ids, env_ids=kneel
  )


##
# Rewards.
##


def ref_height_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  cmd = _term(env, command_name)
  err = cmd.robot.data.root_link_pos_w[:, 2] - cmd.ref_height
  return torch.exp(-err.square() / std**2)


def ref_gravity_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  cmd = _term(env, command_name)
  err = (cmd.robot.data.projected_gravity_b - cmd.ref_gravity).square().sum(dim=1)
  return torch.exp(-err / std**2)


def ref_joint_pose_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  cmd = _term(env, command_name)
  q = cmd.robot.data.joint_pos[:, cmd.joint_ids]
  return torch.exp(-(q - cmd.ref_joint_pos).square().mean(dim=1) / std**2)


def ref_knee_height_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """The kneeling knee's height near its schedule (down only at the end)."""
  cmd = _term(env, command_name)
  z = cmd.robot.data.body_link_pos_w[:, cmd.knee_id, 2]
  return torch.exp(-(z - cmd.ref_knee_height).square() / std**2)


def ref_rfoot_height_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """The stepping foot's height near its schedule (the lift while it steps)."""
  cmd = _term(env, command_name)
  z = cmd.robot.data.site_pos_w[:, cmd.rfoot_id, 2]
  return torch.exp(-(z - cmd.ref_rfoot_height).square() / std**2)


def end_pose_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """While holding the stand or the kneel: a tight kernel on that pose."""
  cmd = _term(env, command_name)
  stand = cmd.hold_stand.float() * torch.exp(-cmd.err_stand.square() / std**2)
  kneel = cmd.hold_kneel.float() * torch.exp(-cmd.err_kneel.square() / std**2)
  return stand + kneel


def hold_still(env: ManagerBasedRlEnv, command_name: str):
  """Penalty: joint motion while holding either end."""
  cmd = _term(env, command_name)
  holding = (cmd.hold_stand | cmd.hold_kneel).float()
  return holding * cmd.robot.data.joint_vel[:, cmd.joint_ids].abs().sum(dim=1)


def ref_joint_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Observation: scheduled joint pose minus the measured one (rad)."""
  cmd = _term(env, command_name)
  return cmd.ref_joint_pos - cmd.robot.data.joint_pos[:, cmd.joint_ids]


def _contact(env: ManagerBasedRlEnv, sensor_name: str, body: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  i = list(sensor.primary_names).index(body)
  return (sensor.data.found[:, i] > 0).float()


def support_contacts(env: ManagerBasedRlEnv, command_name: str, sensor_name: str):
  """The planned contacts: left foot always; right foot except while it steps;
  the right knee once the phase says it is down."""
  cmd = _term(env, command_name)
  left = _contact(env, sensor_name, FEET[0])
  right = _contact(env, sensor_name, FEET[1])
  knee = _contact(env, sensor_name, KNEE_BODY)
  a, b = SWING_WINDOW
  planted = ((cmd.phase <= a) | (cmd.phase >= b)).float()
  down = (cmd.phase >= KNEE_DOWN_PHASE).float()
  return left + planted * right + down * knee


def planted_slip(env: ManagerBasedRlEnv, command_name: str, sensor_name: str):
  """Penalty: horizontal speed of feet and knee while they touch the floor."""
  del command_name
  robot: Entity = env.scene["robot"]
  total = torch.zeros(env.num_envs, device=env.device)
  for body in (*FEET, KNEE_BODY):
    i = robot.find_bodies([body])[0][0]
    speed = robot.data.body_link_lin_vel_w[:, i, :2].norm(dim=1)
    total = total + _contact(env, sensor_name, body) * speed
  return total


##
# Curriculum gate: success = holding the end pose at the end, no fall.
##

KNEEL_RAMP = RobustnessRamp(
  push_xy_mps=(0.0, 0.5, 0.2, 1.0),
  push_yaw_radps=(0.0, 0.3, 0.2, 1.0),
  friction_low=(0.4, 0.3),
  friction_high=(1.0, 1.2),
  mass_scale=(0.10, 0.15),
  payload_low_kg=(-0.5, -2.0),
  payload_high_kg=(0.5, 3.0),
  com_offset_m=(0.01, 0.03),
  joint_friction_nm=(0.1, 0.6),
  armature_scale=(0.05, 0.15),
  pd_gain_scale=(0.05, 0.15),
  effort_scale_low=(0.9, 0.85),
  obs_noise_scale=(1.0, 1.5),
  delay_max_steps=(2.0, 4.0),
  delay_min_steps=(0.0, 1.0),
  motor_penalty_weight=(-1.0, -5.0, 0.0, 0.5),
  braced_penalty_weight=(-0.5, -1.0, 0.0, 0.5),
)


class kneel_robustness(crawl_robustness):  # noqa: N801
  """``crawl_robustness`` gated on ending the episode in the held pose."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    ramp: RobustnessRamp,
    events: dict[str, str],
    noisy_obs: tuple[str, ...],
    motor_reward: str | None = None,
    braced_reward: str | None = None,
    start_level: float = 0.0,
    step: float = 0.1,
    warmup_iterations: int = 300,
    min_iterations_per_level: int = 150,
    promote_progress: float = 0.8,
    promote_max_flip_rate: float = 0.05,
    demote_progress: float = 0.5,
    demote_flip_rate: float = 0.25,
    min_commanded_m: float = 1.0,
    ema_decay: float = 0.995,
    steps_per_iteration: int = 24,
    success_pose_err: float = 0.25,
  ) -> dict[str, torch.Tensor]:
    del min_commanded_m
    if len(env_ids) > 0:
      cmd = _term(env, command_name)
      fell = env.termination_manager.terminated[env_ids].float()
      holding = (cmd.hold_stand | cmd.hold_kneel)[env_ids]
      ok = holding & (cmd.end_err[env_ids] < success_pose_err)
      success = ok.float() * (1.0 - fell)
      self.prog_sum = self.prog_sum * ema_decay + success.sum().item()
      self.prog_n = self.prog_n * ema_decay + float(success.numel())
      self.flip_sum = self.flip_sum * ema_decay + fell.sum().item()
      self.flip_n = self.flip_n * ema_decay + float(fell.numel())
    success_rate = self.prog_sum / self.prog_n if self.prog_n > 50 else 0.0
    fall_rate = self.flip_sum / self.flip_n if self.flip_n > 50 else 1.0
    iteration = env.common_step_counter // steps_per_iteration
    settled = iteration - self.last_change_iter >= min_iterations_per_level
    if iteration >= warmup_iterations and settled:
      old = self.level
      if success_rate >= promote_progress and fall_rate <= promote_max_flip_rate:
        self.level = min(1.0, round(self.level + self.step, 6))
      elif success_rate < demote_progress or fall_rate > demote_flip_rate:
        self.level = max(0.0, round(self.level - self.step, 6))
      if self.level != old:
        self.last_change_iter = iteration
        self._apply()
    r = self.ramp
    values = {
      "level": self.level,
      "gate_success_rate": success_rate,
      "gate_fall_rate": fall_rate,
      "push_mps": self._at(r.push_xy_mps),
      "friction_min": self._at(r.friction_low),
      "payload_max_kg": self._at(r.payload_high_kg),
      "motor_gain_scale_pm": self._at(r.pd_gain_scale),
      "motor_strength_min": self._at(r.effort_scale_low),
      "obs_noise_scale": self._at(r.obs_noise_scale),
      "delay_max_ms": round(self._at(r.delay_max_steps)) * env.physics_dt * 1000,
      "motor_penalty_weight": self._at(r.motor_penalty_weight),
      "braced_penalty_weight": self._at(r.braced_penalty_weight),
    }
    return {k: torch.tensor(v, device=env.device) for k, v in values.items()}
