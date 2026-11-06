"""Stand-to-crawl transition: a phase command with scheduled references.

The policy does not follow a recorded motion. It gets a phase ``phi`` that runs
from 0 (standing) to 1 (the crawl task's initial pose) over a duration drawn
per episode, and is rewarded for keeping four low-dimensional quantities near
smooth schedules of ``phi``:

* pelvis height, standing 0.76 m down to the crawl pose's 0.22 m;
* gravity direction in the base frame, upright to "+X down" (prone);
* the height of the forearm-extension tips, down to the ground early, so the
  robot is on all fours before it lowers and splays the legs;
* the joint pose, each joint travelling from its standing to its crawl value in
  its own window of the phase (arms first, hip abduction last).

The schedules make the descent slow and predictable; the policy keeps balance
and chooses the contacts. After ``phi`` reaches 1 it must hold the crawl pose
still until the episode ends: that is the state the crawl policy takes over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.g1_crawl.mdp.events import reset_from_animation_frame
from mjlab.tasks.g1_crawl.mdp.robustness import RobustnessRamp, crawl_robustness
from mjlab.tasks.g1_walk.mdp.keyframes import KEYFRAME_TABLE
from mjlab.tasks.g1_walk.robot import (
  CRAWL_HEIGHT,
  CRAWL_POSE_29,
  CRAWL_ROT,
  JOINTS_29,
  STAND_HEIGHT,
  STAND_POSE_29,
)
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, yaw_quat

CRAWL_ANIMATION_29 = (
  Path(__file__).resolve().parents[2]
  / "g1_crawl"
  / "assets"
  / "animation_rc4_29dof.json"
)
# Where the creepy and scuttle crawls hand back to the stand <-> crawl policy
# (``scripts/record_crawl_stops.py``, hard randomization).
CRAWL_STOP_FRAMES_29 = tuple(
  str(Path(__file__).resolve().parents[1] / "assets" / f"crawl_stops_{name}.json")
  for name in ("creepy", "scuttle")
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def smoothstep(x: torch.Tensor) -> torch.Tensor:
  x = x.clamp(0.0, 1.0)
  return x * x * (3.0 - 2.0 * x)


CRAWL_PITCH_DEG = math.degrees(2.0 * math.atan2(CRAWL_ROT[2], CRAWL_ROT[0]))
"""Base pitch about +Y in the crawl pose (quaternion w, 0, y, 0)."""


@dataclass(frozen=True)
class Keyframe:
  """A whole-body pose on the way down, checked with forward kinematics."""

  phi: float
  height: float
  """Pelvis height (m)."""
  pitch_deg: float
  """Base pitch about +Y (deg): gravity in the base frame is (sin, 0, -cos)."""
  hand_height: float
  """Extension tip height (m); 0 on the ground."""
  joint_pos: dict[str, float]


KEYFRAMES: tuple[Keyframe, ...] = tuple(Keyframe(**k) for k in KEYFRAME_TABLE)
"""Stand, squat and lean, all fours, splay, the crawl pose (``keyframes.py``)."""
assert KEYFRAMES[0].phi == 0.0 and KEYFRAMES[-1].phi == 1.0
assert all(a.phi < b.phi for a, b in zip(KEYFRAMES, KEYFRAMES[1:], strict=False))
for _frame, _pose, _height in (
  (KEYFRAMES[0], STAND_POSE_29, STAND_HEIGHT),
  (KEYFRAMES[-1], CRAWL_POSE_29, CRAWL_HEIGHT),
):
  assert math.isclose(_frame.height, _height, abs_tol=1e-4), _frame.height
  assert all(
    math.isclose(_frame.joint_pos[j], _pose[j], abs_tol=1e-5) for j in JOINTS_29
  ), "the first/last keyframe must be the stand/crawl pose"
del _frame, _pose, _height


class TransitionSchedule:
  """The references as functions of the phase: smoothstep between keyframes.

  Shared by the command term (training) and the policy switcher (deployment),
  so both evaluate exactly the same references. Each segment eases in and out,
  so the motion pauses briefly at every keyframe, a statically held pose.
  """

  def __init__(
    self,
    device: torch.device | str,
    keyframes: tuple[Keyframe, ...] | None = None,
    joints: tuple[str, ...] = JOINTS_29,
  ) -> None:
    keyframes = keyframes or KEYFRAMES
    self.joints = joints
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    self.phis = t([k.phi for k in keyframes])
    self.heights = t([k.height for k in keyframes])
    self.pitches = t([math.radians(k.pitch_deg) for k in keyframes])
    self.hands = t([k.hand_height for k in keyframes])
    self.q = t([[k.joint_pos[j] for j in joints] for k in keyframes])
    self.q_stand, self.q_crawl = self.q[0], self.q[-1]
    self.g_stand = self.gravity(torch.zeros(1, device=device))[0]
    self.g_crawl = self.gravity(torch.ones(1, device=device))[0]

  def _segment(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = (torch.searchsorted(self.phis, phi.contiguous(), right=True) - 1).clamp(
      0, len(self.phis) - 2
    )
    span = self.phis[idx + 1] - self.phis[idx]
    return idx, smoothstep((phi - self.phis[idx]) / span)

  def _lerp(self, table: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    idx, s = self._segment(phi)
    a, b = table[idx], table[idx + 1]
    return a + (b - a) * (s[:, None] if table.dim() == 2 else s)

  def joint_pos(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.q, phi)

  def height(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.heights, phi)

  def pitch(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.pitches, phi)

  def gravity(self, phi: torch.Tensor) -> torch.Tensor:
    pitch = self.pitch(phi)
    zeros = torch.zeros_like(pitch)
    return torch.stack([torch.sin(pitch), zeros, -torch.cos(pitch)], dim=1)

  def hand_height(self, phi: torch.Tensor) -> torch.Tensor:
    return self._lerp(self.hands, phi)


@dataclass(kw_only=True)
class TransitionPhaseCommandCfg(CommandTermCfg):
  entity_name: str
  duration_s: tuple[float, float] = (4.0, 8.0)
  """Time for the phase to run from 0 to 1, drawn per episode."""
  delay_s: tuple[float, float] = (0.5, 1.5)
  """Standing time before the phase starts moving, drawn per episode."""
  bidirectional: bool = False
  """Also stand back up: after a hold at phi = 1 the phase runs back to 0, and
  the command's second entry is the signed rate dphi/dt (0 while holding)
  instead of 1/duration. Episodes flagged by ``reset_crawl_start`` begin in the
  crawl pose (phase 1)."""
  hold_s: tuple[float, float] = (1.5, 4.0)
  """Bidirectional: crawl-pose hold before standing up."""
  up_s: tuple[float, float] = (4.0, 8.0)
  """Bidirectional: time for the phase to run from 1 back to 0."""
  stand_up_prob: float = 0.8
  """Bidirectional: fraction of episodes that stand back up."""

  def build(self, env: ManagerBasedRlEnv) -> TransitionPhaseCommand:
    return TransitionPhaseCommand(self, env)


class TransitionPhaseCommand(CommandTerm):
  """Command ``[phi, 1/duration]`` (bidirectional: ``[phi, dphi/dt]``) plus the
  reference schedules at ``phi``."""

  cfg: TransitionPhaseCommandCfg

  def __init__(self, cfg: TransitionPhaseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    n, dev = self.num_envs, self.device
    self.phase = torch.zeros(n, device=dev)
    self.duration = torch.ones(n, device=dev)
    self.delay = torch.zeros(n, device=dev)
    self.command_buf = torch.zeros(n, 2, device=dev)
    self.manual = False
    self.manual_phase = torch.zeros(n, device=dev)
    # Bidirectional timeline (see the config).
    self.rate = torch.zeros(n, device=dev)
    self.manual_rate = torch.zeros(n, device=dev)
    self.t_hold = torch.zeros(n, device=dev)
    self.t_up = torch.ones(n, device=dev)
    self.will_stand = torch.zeros(n, dtype=torch.bool, device=dev)
    self.start_crawl = torch.zeros(n, dtype=torch.bool, device=dev)
    """Set by ``reset_crawl_start`` before the command resamples."""

    ids, names = self.robot.find_joints(JOINTS_29, preserve_order=True)
    assert tuple(names) == JOINTS_29, names
    self.joint_ids = torch.tensor(ids, device=dev)
    self.schedule = TransitionSchedule(dev)
    self.q_crawl = self.schedule.q_crawl

    self.ref_joint_pos = self.schedule.q_stand.repeat(n, 1)
    self.ref_height = torch.full((n,), STAND_HEIGHT, device=dev)
    self.ref_gravity = self.schedule.g_stand.repeat(n, 1)
    self.ref_hand_height = self.schedule.hands[0].repeat(n)
    self.ground_z = torch.zeros(n, device=dev)
    """Terrain height at the spawn point: the descent happens in place."""
    self.pose_err = torch.zeros(n, device=dev)
    """RMS joint error to the crawl pose (rad), refreshed every step."""
    self.stand_err = torch.zeros(n, device=dev)
    """RMS joint error to the stand pose (rad), refreshed every step."""
    for name in ("phase_end", "pose_err_rad", "height_err_m", "gravity_err"):
      self.metrics[name] = torch.zeros(n, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return self.command_buf

  @property
  def hold_crawl(self) -> torch.Tensor:
    return (self.phase >= 1.0) & (self.rate == 0.0)

  @property
  def hold_stand(self) -> torch.Tensor:
    return (self.phase <= 0.0) & (self.rate == 0.0)

  def manual_control(self) -> None:
    """Freeze sampling: ``manual_phase`` drives the phase (play, the switcher)."""
    self.manual = True
    self.delay[:] = 0.0
    self.manual_phase[:] = 0.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.manual:
      return
    r = torch.empty(len(env_ids), device=self.device)
    self.duration[env_ids] = r.uniform_(*self.cfg.duration_s)
    self.delay[env_ids] = r.uniform_(*self.cfg.delay_s)
    if self.cfg.bidirectional:
      self.t_hold[env_ids] = r.uniform_(*self.cfg.hold_s)
      self.t_up[env_ids] = r.uniform_(*self.cfg.up_s)
      self.will_stand[env_ids] = r.uniform_(0.0, 1.0) < self.cfg.stand_up_prob

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids  # Pure function of the state; refreshing every env is safe.
    bidir = self.cfg.bidirectional
    if self.manual:
      self.phase = self.manual_phase.clamp(0.0, 1.0)
      self.rate = self.manual_rate.clone() if bidir else 1.0 / self.duration
    elif bidir:
      # wait -> down -> hold -> (will_stand) up -> stand; crawl starts skip down.
      t = self._env.episode_length_buf.float() * self._env.step_dt
      crawl0 = self.start_crawl
      down = ((t - self.delay) / self.duration).clamp(0.0, 1.0)
      down = torch.where(crawl0, torch.ones_like(down), down)
      t_up0 = torch.where(crawl0, self.t_hold, self.delay + self.duration + self.t_hold)
      up = ((t - t_up0) / self.t_up).clamp(0.0, 1.0) * self.will_stand.float()
      self.phase = down - up
      going_down = ~crawl0 & (t >= self.delay) & (t < self.delay + self.duration)
      going_up = self.will_stand & (t >= t_up0) & (t < t_up0 + self.t_up)
      self.rate = going_down.float() / self.duration - going_up.float() / self.t_up
    else:
      t = self._env.episode_length_buf.float() * self._env.step_dt
      self.phase = ((t - self.delay) / self.duration).clamp(0.0, 1.0)
      self.rate = 1.0 / self.duration
    self.command_buf[:, 0] = self.phase
    self.command_buf[:, 1] = self.rate
    self.ground_z = self._env.scene.env_origins[:, 2]
    self.ref_height = self.schedule.height(self.phase)
    self.ref_gravity = self.schedule.gravity(self.phase)
    self.ref_joint_pos = self.schedule.joint_pos(self.phase)
    self.ref_hand_height = self.schedule.hand_height(self.phase)
    q = self.robot.data.joint_pos[:, self.joint_ids]
    self.pose_err = (q - self.q_crawl).square().mean(dim=1).sqrt()
    self.stand_err = (q - self.schedule.q_stand).square().mean(dim=1).sqrt()

  def _update_metrics(self) -> None:
    # Last-step values: at reset they describe where the episode ended.
    self.metrics["phase_end"][:] = self.phase
    self.metrics["pose_err_rad"][:] = self.pose_err
    self.metrics["height_err_m"][:] = (
      self.robot.data.root_link_pos_w[:, 2] - self.ground_z - self.ref_height
    ).abs()
    self.metrics["gravity_err"][:] = (
      self.robot.data.projected_gravity_b - self.ref_gravity
    ).norm(dim=1)


def _term(env: ManagerBasedRlEnv, command_name: str) -> TransitionPhaseCommand:
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, TransitionPhaseCommand)
  return term


##
# Reference tracking.
##


def ref_height_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """Pelvis height near its schedule."""
  cmd = _term(env, command_name)
  err = cmd.robot.data.root_link_pos_w[:, 2] - cmd.ground_z - cmd.ref_height
  return torch.exp(-err.square() / std**2)


def ref_gravity_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """Base orientation near its schedule (gravity direction in the base frame)."""
  cmd = _term(env, command_name)
  err = (cmd.robot.data.projected_gravity_b - cmd.ref_gravity).square().sum(dim=1)
  return torch.exp(-err / std**2)


def ref_joint_pose_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """Joints near their scheduled values (mean square over joints)."""
  cmd = _term(env, command_name)
  q = cmd.robot.data.joint_pos[:, cmd.joint_ids]
  err = (q - cmd.ref_joint_pos).square().mean(dim=1)
  return torch.exp(-err / std**2)


def ref_hand_height_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg
):
  """Extension tips near their scheduled height (mean square over the tips)."""
  cmd = _term(env, command_name)
  z = cmd.robot.data.site_pos_w[:, asset_cfg.site_ids, 2] - cmd.ground_z[:, None]
  err = (z - cmd.ref_hand_height[:, None]).square().mean(dim=1)
  return torch.exp(-err / std**2)


def final_pose_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """Once the phase is complete: a tight kernel on the crawl pose itself."""
  cmd = _term(env, command_name)
  done = (cmd.phase >= 1.0).float()
  return done * torch.exp(-cmd.pose_err.square() / std**2)


def stand_pose_exp(env: ManagerBasedRlEnv, command_name: str, std: float):
  """Bidirectional: while holding phi = 0 (before going down, and once stood
  back up), a tight kernel on the stand pose: what the walk policy takes over."""
  cmd = _term(env, command_name)
  return cmd.hold_stand.float() * torch.exp(-cmd.stand_err.square() / std**2)


def hold_still(env: ManagerBasedRlEnv, command_name: str):
  """Penalty: joint motion once the phase is complete (hold for the hand-over)."""
  cmd = _term(env, command_name)
  done = (cmd.phase >= 1.0).float()
  return done * cmd.robot.data.joint_vel[:, cmd.joint_ids].abs().sum(dim=1)


def before_start_still(env: ManagerBasedRlEnv, command_name: str):
  """Penalty: joint motion before the phase starts (stand and wait)."""
  cmd = _term(env, command_name)
  waiting = (cmd.phase <= 0.0).float()
  return waiting * cmd.robot.data.joint_vel[:, cmd.joint_ids].abs().sum(dim=1)


def stand_gate(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_joint_speed: float = 0.6,
  max_gyro: float = 0.3,
  min_gravity_z: float = 0.97,
) -> torch.Tensor:
  """Bonus: holding phi = 0 and passing the switcher's stand gate (upright,
  joints still, body calm), the state the walk policy takes over in."""
  cmd = _term(env, command_name)
  d = cmd.robot.data
  ok = (
    (d.projected_gravity_b[:, 2] < -min_gravity_z)
    & (d.joint_vel[:, cmd.joint_ids].abs().amax(dim=1) < max_joint_speed)
    & (d.root_link_ang_vel_b.norm(dim=1) < max_gyro)
  )
  return (cmd.hold_stand & ok).float()


# Pelvis above the mean height of the four limb links in the crawl pose (the
# switcher's crawl gate, ``chain.json`` crawl_pelvis_above_limbs_m).
CRAWL_ABOVE_LIMBS = 0.151
GATE_LIMBS = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_forearm_extension",
  "right_forearm_extension",
)


def crawl_gate(
  env: ManagerBasedRlEnv,
  command_name: str,
  max_joint_speed: float = 0.5,
  max_gravity_err: float = 0.15,
  height_tol: float = 0.03,
) -> torch.Tensor:
  """Bonus: holding phi = 1 and passing the switcher's crawl gate (joints
  still, prone like the crawl pose, pelvis at the crawl height over the limbs),
  the state the crawl policy takes over in."""
  cmd = _term(env, command_name)
  d = cmd.robot.data
  limb_ids = cmd.robot.find_bodies(GATE_LIMBS, preserve_order=True)[0]
  limbs = d.body_link_pos_w[:, limb_ids, 2].mean(dim=1)
  ok = (
    (d.joint_vel[:, cmd.joint_ids].abs().amax(dim=1) < max_joint_speed)
    & ((d.projected_gravity_b - cmd.schedule.g_crawl).norm(dim=1) < max_gravity_err)
    & ((d.root_link_pos_w[:, 2] - limbs - CRAWL_ABOVE_LIMBS).abs() < height_tol)
  )
  return (cmd.hold_crawl & ok).float()


def stand_settle_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_std: float = 0.1,
  gyro_std: float = 0.3,
) -> torch.Tensor:
  """Holding phi = 0: the base still (no drift, no sway). A smooth version of
  ``stand_gate`` that the policy's exploration noise does not zero out."""
  cmd = _term(env, command_name)
  d = cmd.robot.data
  v = d.root_link_lin_vel_w[:, :2].norm(dim=1)
  w = d.root_link_ang_vel_b.norm(dim=1)
  k = torch.exp(-((v / speed_std) ** 2) - (w / gyro_std) ** 2)
  return cmd.hold_stand.float() * k


def crawl_settle_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  height_std: float = 0.03,
  gravity_std: float = 0.15,
) -> torch.Tensor:
  """Holding phi = 1: pelvis at the crawl gate's height over the limbs and
  prone like the crawl pose. A smooth version of ``crawl_gate``."""
  cmd = _term(env, command_name)
  d = cmd.robot.data
  limb_ids = cmd.robot.find_bodies(GATE_LIMBS, preserve_order=True)[0]
  limbs = d.body_link_pos_w[:, limb_ids, 2].mean(dim=1)
  h = d.root_link_pos_w[:, 2] - limbs - CRAWL_ABOVE_LIMBS
  g = (d.projected_gravity_b - cmd.schedule.g_crawl).norm(dim=1)
  k = torch.exp(-((h / height_std) ** 2) - (g / gravity_std) ** 2)
  return cmd.hold_crawl.float() * k


def ref_joint_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Observation: scheduled joint pose minus the measured one (rad)."""
  cmd = _term(env, command_name)
  return cmd.ref_joint_pos - cmd.robot.data.joint_pos[:, cmd.joint_ids]


##
# Slowness and softness.
##


def joint_vel_cap(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, limit: float):
  """Penalty: joint speed above ``limit`` rad/s, squared."""
  robot: Entity = env.scene[asset_cfg.name]
  over = torch.relu(robot.data.joint_vel[:, asset_cfg.joint_ids].abs() - limit)
  return over.square().sum(dim=1)


def base_speed_cap(env: ManagerBasedRlEnv, limit: float):
  """Penalty: pelvis speed above ``limit`` m/s, squared."""
  robot: Entity = env.scene["robot"]
  speed = robot.data.root_link_lin_vel_w.norm(dim=1)
  return torch.relu(speed - limit).square()


def contact_impact(env: ManagerBasedRlEnv, sensor_name: str, scale: float = 300.0):
  """Penalty: rise of any body's ground force between physics steps, in units
  of ``scale`` N (about the robot's weight)."""
  sensor: ContactSensor = env.scene[sensor_name]
  history = sensor.data.force_history
  assert history is not None, "contact_impact needs history_length >= 2"
  force = history.norm(dim=-1)  # [B, N, H], oldest first.
  rise = torch.relu(force[:, :, 1:] - force[:, :, :-1]).amax(dim=2)
  return rise.sum(dim=1) / scale


def feet_both_down(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names,
  command_name: str | None = None,
  phase_max: float = 1.0,
):
  """One while both feet touch the ground (only up to ``phase_max``: from the
  all-fours keyframe on, the feet have to step outward into the sprawl)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  lookup = {name: i for i, name in enumerate(sensor.primary_names)}
  ids = [lookup[name] for name in body_names]
  down = (sensor.data.found[:, ids] > 0).all(dim=1).float()
  if command_name is not None:
    down = down * (_term(env, command_name).phase <= phase_max).float()
  return down


def feet_flat_end(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  std: float = 0.03,
  phase_min: float = 0.9,
):
  """Near the end: the soles (foot sites) at ground level, as in the crawl pose."""
  cmd = _term(env, command_name)
  z = cmd.robot.data.site_pos_w[:, asset_cfg.site_ids, 2]
  late = (cmd.phase >= phase_min).float()
  return late * torch.exp(-(z / std).square()).mean(dim=1)


##
# Terminations.
##


def fell(
  env: ManagerBasedRlEnv,
  back_limit: float = 0.5,
  side_limit: float = 0.7,
  over_limit: float = 0.5,
) -> torch.Tensor:
  """Fallen backward (gravity toward base -X), sideways, or past prone."""
  g = env.scene["robot"].data.projected_gravity_b
  return (g[:, 0] < -back_limit) | (g[:, 1].abs() > side_limit) | (g[:, 2] > over_limit)


##
# Curriculum: the crawl's robustness ramp gated on transition success.
##

TRANSITION_RAMP = RobustnessRamp(
  # A deployable floor at level 0 (see WALK_RAMP); pushes only once it works.
  push_xy_mps=(0.0, 0.3, 0.2, 1.0),
  push_yaw_radps=(0.0, 0.2, 0.2, 1.0),
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
  # Torque limits matter from the start here: hands take load as they go down.
  motor_penalty_weight=(-1.0, -5.0, 0.0, 0.5),
  braced_penalty_weight=(-0.5, -1.0, 0.0, 0.5),
)


TRANSITION_RAMP_HARD = RobustnessRamp(
  # For the two-way stand <-> crawl policy: level 0 as TRANSITION_RAMP's, level
  # 1 well past its level 1 (pushes 0.3 -> 0.6 m/s, gains and rotor inertia off
  # by a quarter, weaker motors, sticky joints, more payload, noisier sensors).
  push_xy_mps=(0.0, 0.6, 0.2, 1.0),
  push_yaw_radps=(0.0, 0.4, 0.2, 1.0),
  friction_low=(0.4, 0.2),
  friction_high=(1.0, 1.4),
  mass_scale=(0.10, 0.2),
  payload_low_kg=(-0.5, -3.0),
  payload_high_kg=(0.5, 5.0),
  com_offset_m=(0.01, 0.05),
  joint_friction_nm=(0.1, 1.0),
  armature_scale=(0.05, 0.3),
  pd_gain_scale=(0.05, 0.25),
  effort_scale_low=(0.9, 0.75),
  obs_noise_scale=(1.0, 2.0),
  delay_max_steps=(2.0, 4.0),
  delay_min_steps=(0.0, 1.0),
  motor_penalty_weight=(-1.0, -5.0, 0.0, 0.5),
  # Standing up pushes on the hands: at TRANSITION_RAMP's -1 the braced wrists
  # took 7 Nm peaks (limit 5) for a penalty worth ~0.01% of the return.
  braced_penalty_weight=(-2.0, -6.0, 0.0, 0.5),
)


def reset_crawl_start(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  prob: float,
  joint_noise: float = 0.05,
  drop_m: float = 0.02,
  gait_frac: float = 0.5,
  nose_down_frac: float = 0.0,
  nose_down_deg: tuple[float, float] = (5.0, 15.0),
  nose_down_lift_m: tuple[float, float] = (0.03, 0.06),
  shoulder_roll_skew: float = 0.3,
  stop_frames: tuple[str, ...] = (),
  stop_frac: float = 0.0,
) -> None:
  """Start a fraction of episodes on all fours (after the standing reset
  events), about to stand up: in the crawl pose with joint noise, or, for
  ``gait_frac`` of them, in a frame of the crawl gait (the crawl task's reset
  animation), which is where the crawl policy hands over (it stands ~14 cm
  higher than the crawl pose). The hold before standing up settles either into
  the crawl pose.

  ``nose_down_frac`` of the crawl-pose starts begin in the posture the real
  robot got stuck in (2026-10-29): pitched ``nose_down_deg`` further nose-down,
  pelvis ``nose_down_lift_m`` higher, and both shoulder rolls shifted by the
  same random amount in ``+-shoulder_roll_skew`` rad (one arm out, one in).

  ``stop_frac`` of them (before the others are split) start in a pose drawn
  from ``stop_frames``: files of real hand-overs from the deployed crawls
  (``scripts/record_crawl_stops.py``), one file picked at random per robot."""
  cmd = _term(env, command_name)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  r = torch.rand(len(env_ids), device=env.device)
  crawl = env_ids[r < prob]
  cmd.start_crawl[env_ids] = False
  if len(crawl) == 0:
    return
  cmd.start_crawl[crawl] = True
  robot = cmd.robot
  env.sim.forward()  # The standing reset wrote qpos; refresh the root pose.
  state = torch.cat(
    [
      robot.data.root_link_pos_w[crawl],
      robot.data.root_link_quat_w[crawl],
      torch.zeros(len(crawl), 6, device=env.device),
    ],
    dim=1,
  )
  rot = torch.tensor(CRAWL_ROT, device=env.device).expand(len(crawl), 4)
  state[:, 2] = env.scene.env_origins[crawl, 2] + CRAWL_HEIGHT + drop_m
  state[:, 3:7] = quat_mul(yaw_quat(state[:, 3:7]), rot)
  robot.write_root_state_to_sim(state, env_ids=crawl)
  q = cmd.q_crawl.expand(len(crawl), -1)
  q = q + joint_noise * (2 * torch.rand_like(q) - 1)
  robot.write_joint_state_to_sim(
    q, torch.zeros_like(q), joint_ids=cmd.joint_ids, env_ids=crawl
  )
  is_stop = torch.rand(len(crawl), device=env.device) < (
    stop_frac if stop_frames else 0.0
  )
  is_gait = ~is_stop & (torch.rand(len(crawl), device=env.device) < gait_frac)
  nose = crawl[
    ~is_gait & ~is_stop & (torch.rand(len(crawl), device=env.device) < nose_down_frac)
  ]
  if len(nose) > 0:
    k = len(nose)
    pitch = torch.empty(k, device=env.device).uniform_(*nose_down_deg)
    zeros = torch.zeros(k, device=env.device)
    extra = quat_from_euler_xyz(zeros, torch.deg2rad(pitch), zeros)
    rows = torch.isin(crawl, nose)
    root = state[rows]
    root[:, 3:7] = quat_mul(root[:, 3:7], extra)
    root[:, 2] += torch.empty(k, device=env.device).uniform_(*nose_down_lift_m)
    robot.write_root_state_to_sim(root, env_ids=nose)
    q_nose = q[rows].clone()
    skew = torch.empty(k, device=env.device).uniform_(
      -shoulder_roll_skew, shoulder_roll_skew
    )
    for name in ("left_shoulder_roll_joint", "right_shoulder_roll_joint"):
      q_nose[:, JOINTS_29.index(name)] += skew
    robot.write_joint_state_to_sim(
      q_nose, torch.zeros_like(q_nose), joint_ids=cmd.joint_ids, env_ids=nose
    )
  gait = crawl[is_gait]
  if len(gait) > 0:
    reset_from_animation_frame(
      env,
      gait,
      asset_cfg=SceneEntityCfg("robot", joint_names=JOINTS_29),
      position_range=(0.95, 1.05),
      animation_path=CRAWL_ANIMATION_29,
    )
  stop = crawl[is_stop]
  if len(stop) > 0:
    pick = torch.randint(len(stop_frames), (len(stop),), device=env.device)
    for k, path in enumerate(stop_frames):
      ids = stop[pick == k]
      if len(ids) > 0:
        reset_from_animation_frame(
          env,
          ids,
          asset_cfg=SceneEntityCfg("robot", joint_names=JOINTS_29),
          position_range=(0.97, 1.03),
          animation_path=path,
        )


class transition_robustness(crawl_robustness):  # noqa: N801
  """``crawl_robustness`` gated on reaching the crawl pose without falling
  (bidirectional: on ending in the pose the timeline asked for)."""

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
    del min_commanded_m  # The crawl/walk gate's; success is pose-based here.
    if len(env_ids) > 0:
      cmd = _term(env, command_name)
      fell_now = env.termination_manager.terminated[env_ids].float()
      if cmd.cfg.bidirectional:
        # Ended holding the pose its timeline asked for: crawl or stand.
        crawl = cmd.hold_crawl & (cmd.pose_err < success_pose_err)
        stand = cmd.hold_stand & (cmd.stand_err < success_pose_err)
        reached = (crawl | stand)[env_ids]
      else:
        reached = (cmd.phase[env_ids] >= 1.0) & (
          cmd.pose_err[env_ids] < success_pose_err
        )
      success = reached.float() * (1.0 - fell_now)
      self.prog_sum = self.prog_sum * ema_decay + success.sum().item()
      self.prog_n = self.prog_n * ema_decay + float(success.numel())
      self.flip_sum = self.flip_sum * ema_decay + fell_now.sum().item()
      self.flip_n = self.flip_n * ema_decay + float(fell_now.numel())
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
      "mass_scale_pm": self._at(r.mass_scale),
      "payload_max_kg": self._at(r.payload_high_kg),
      "joint_friction_max_nm": self._at(r.joint_friction_nm),
      "motor_gain_scale_pm": self._at(r.pd_gain_scale),
      "motor_strength_min": self._at(r.effort_scale_low),
      "obs_noise_scale": self._at(r.obs_noise_scale),
      "delay_max_ms": round(self._at(r.delay_max_steps)) * env.physics_dt * 1000,
      "motor_penalty_weight": self._at(r.motor_penalty_weight),
      "braced_penalty_weight": self._at(r.braced_penalty_weight),
    }
    return {k: torch.tensor(v, device=env.device) for k, v in values.items()}
