"""Per-step training metrics for the crawl task (no weight, logged only).

MJLab averages each term over the episode (or takes the max, per its
``MetricsTermCfg.reduce``) and logs it as ``Episode_Metrics/<term name>``. The
29-DoF task names its terms ``track/...``, ``gait/...``, ``safety/...`` and
``effort/...`` so they sort into readable groups in TensorBoard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor, TerrainHeightSensor
from mjlab.tasks.g1_crawl.mdp.rewards import braced_joint_moment, motor_torque_ratio

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_CONTACT_N = 1.0  # A limb or body "touches" above this net force.


def _robot(env: ManagerBasedRlEnv) -> Entity:
  return env.scene["robot"]


def _limb_forces(
  env: ManagerBasedRlEnv, sensor_name: str, body_names: tuple[str, ...]
) -> torch.Tensor:
  """Net contact force magnitude per named body, (num_envs, len(body_names))."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  names = list(sensor.primary_names)
  ids = [names.index(name) for name in body_names]
  assert sensor.data.force is not None
  return sensor.data.force[:, ids].norm(dim=-1)


# Tracking.


def fwd_speed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Actual forward speed (m/s): base-frame Z, the forward axis when prone."""
  return _robot(env).data.root_link_lin_vel_b[:, 2]


def fwd_speed_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """|commanded - actual| forward speed (m/s)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (command[:, 0] - fwd_speed(env)).abs()


def lateral_speed_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """|commanded - actual| sideways speed (m/s): base-frame Y."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (command[:, 1] - _robot(env).data.root_link_lin_vel_b[:, 1]).abs()


def command_value(
  env: ManagerBasedRlEnv, command_name: str, index: int
) -> torch.Tensor:
  """One command slot, e.g. the commanded step height (index 3)."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, index]


def turn_rate_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """|commanded - actual| turn rate (rad/s): base-frame X, vertical when prone."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return (command[:, 2] - _robot(env).data.root_link_ang_vel_b[:, 0]).abs()


# Gait and posture.


def limb_contact_frac(
  env: ManagerBasedRlEnv, sensor_name: str, body_names: tuple[str, ...]
) -> torch.Tensor:
  """Fraction of the named limbs on the ground (0..1)."""
  touching = _limb_forces(env, sensor_name, body_names) > _CONTACT_N
  return touching.float().mean(dim=1)


def hand_load_share(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  hand_names: tuple[str, ...],
  foot_names: tuple[str, ...],
) -> torch.Tensor:
  """Share of the limbs' ground force carried by the hands (0..1)."""
  hands = _limb_forces(env, sensor_name, hand_names).sum(dim=1)
  feet = _limb_forces(env, sensor_name, foot_names).sum(dim=1)
  return hands / (hands + feet).clamp(min=1e-6)


def body_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Pelvis height above the env origin (m)."""
  return _robot(env).data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def prone_alignment(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Gravity along base +X: 1 = perfectly prone, 0 = on its side, <0 = flipped."""
  return _robot(env).data.projected_gravity_b[:, 0]


def bodies_touching(
  env: ManagerBasedRlEnv, sensor_name: str, body_names: tuple[str, ...]
) -> torch.Tensor:
  """1 when any of the named bodies touches the ground."""
  return (_limb_forces(env, sensor_name, body_names) > _CONTACT_N).any(dim=1).float()


def other_bodies_touching(
  env: ManagerBasedRlEnv, sensor_name: str, excluded_body_names: tuple[str, ...]
) -> torch.Tensor:
  """How many bodies other than hands and feet touch the ground."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  keep = [i for i, n in enumerate(sensor.primary_names) if n not in excluded_body_names]
  return (sensor.data.force[:, keep].norm(dim=-1) > _CONTACT_N).sum(dim=1).float()


def diagonal_sync(
  env: ManagerBasedRlEnv, sensor_name: str, limb_names: tuple[str, str, str, str]
) -> torch.Tensor:
  """Diagonal-pair contact agreement, as in the ``diagonal_gait`` reward (ungated)."""
  down = _limb_forces(env, sensor_name, limb_names) > _CONTACT_N
  lh, rh, lf, rf = down.unbind(dim=1)
  sync = 0.5 * ((lh == rf).float() + (rh == lf).float())
  return sync * (1.0 - 0.5 * down.all(dim=1).float())


def swing_height(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  site_names: tuple[str, ...],
  min_speed: float = 0.2,
) -> torch.Tensor:
  """Mean height above terrain (m) of limbs moving faster than ``min_speed``."""
  sensor = env.scene[height_sensor_name]
  assert isinstance(sensor, TerrainHeightSensor)
  robot = _robot(env)
  ids = robot.find_sites(site_names, preserve_order=True)[0]
  speed = robot.data.site_lin_vel_w[:, ids, :2].norm(dim=-1)
  swinging = (speed > min_speed).float()
  height = sensor.data.heights * swinging
  return height.sum(dim=1) / swinging.sum(dim=1).clamp(min=1.0)


# Torque safety.


def motor_peak_ratio(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> torch.Tensor:
  """Most loaded of the given motors: torque demand / peak (1 = at peak)."""
  return motor_torque_ratio(env, motors).max(dim=1).values


def motor_peak_nm(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> torch.Tensor:
  """Largest torque demand among the given motors (Nm)."""
  peak = torch.tensor([m[2] for m in motors.values()], device=env.device)
  return (motor_torque_ratio(env, motors) * peak).max(dim=1).values


def motor_any_saturated(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> torch.Tensor:
  """1 when any of the given motors is asked for more than its peak."""
  return (motor_torque_ratio(env, motors) > 1.0).any(dim=1).float()


def braced_over_limit(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  axes: tuple[tuple[float, float, float], ...],
  limit: float,
) -> torch.Tensor:
  """1 when any braced joint carries more than ``limit`` (Nm) about its axis."""
  return (braced_joint_moment(env, sensor_names, axes) > limit).any(dim=1).float()


_JOINT_IDS: dict[tuple[int, tuple[str, ...]], list[int]] = {}


def _joint_ids(env: ManagerBasedRlEnv, names: tuple[str, ...]) -> list[int]:
  key = (id(env), names)
  if key not in _JOINT_IDS:
    _JOINT_IDS[key] = _robot(env).find_joints(names, preserve_order=True)[0]
  return _JOINT_IDS[key]


def joint_near_limit(
  env: ManagerBasedRlEnv, joint_names: tuple[str, ...], margin: float = 0.05
) -> torch.Tensor:
  """1 when any joint is within ``margin`` (fraction of range) of a soft limit."""
  robot = _robot(env)
  ids = _joint_ids(env, joint_names)
  q = robot.data.joint_pos[:, ids]
  limits = robot.data.soft_joint_pos_limits[:, ids]
  span = limits[..., 1] - limits[..., 0]
  low = q < limits[..., 0] + margin * span
  high = q > limits[..., 1] - margin * span
  return (low | high).any(dim=1).float()


# Effort.


def mechanical_power(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> torch.Tensor:
  """Sum over motors of |torque x joint speed| (W), torque clamped to peak."""
  robot = _robot(env)
  ids = _joint_ids(env, tuple(motors))
  peak = torch.tensor([m[2] for m in motors.values()], device=env.device)
  torque = motor_torque_ratio(env, motors).clamp(max=1.0) * peak
  return (torque * robot.data.joint_vel[:, ids].abs()).sum(dim=1)
