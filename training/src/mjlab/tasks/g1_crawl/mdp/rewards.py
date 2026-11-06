"""Reward terms translated from the active Isaac Lab G1 crawl config."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, TerrainHeightSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def track_lin_vel_yz_exp(
  env: ManagerBasedRlEnv, std: float, command_name: str
) -> torch.Tensor:
  """Exponential tracking of base-frame [vertical, lateral] velocity."""
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  error = command[:, :2] - robot.data.root_link_lin_vel_b[:, [2, 1]]
  return torch.exp(-torch.sum(error.square(), dim=1) / std**2)


def track_ang_vel_z_world_exp(
  env: ManagerBasedRlEnv, std: float, command_name: str
) -> torch.Tensor:
  """Preserve the source reward's comparison against world-frame yaw rate."""
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.exp(
    -torch.square(command[:, 2] - robot.data.root_link_ang_vel_w[:, 2]) / std**2
  )


def track_ang_vel_x_exp(
  env: ManagerBasedRlEnv, std: float, command_name: str
) -> torch.Tensor:
  """Track the commanded turn rate as base-frame roll rate (body X is vertical).

  The source compared this command against world yaw rate, which flips its sign
  when prone. Tracking body X directly makes the turn command direction fixed.
  """
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.exp(
    -torch.square(command[:, 2] - robot.data.root_link_ang_vel_b[:, 0]) / std**2
  )


def _boost(
  env: ManagerBasedRlEnv, after_iterations: int | None, factor: float
) -> float:
  if after_iterations is None or env.common_step_counter < after_iterations * 24:
    return 1.0
  return factor


def limb_clearance_cmd(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  cmd_index: int = 3,
  boost_after_iterations: int | None = None,
  boost: float = 1.0,
) -> torch.Tensor:
  """Velocity-weighted deviation of limb height from the commanded swing height.

  Same shape as mjlab's ``feet_clearance`` but the target is per env, read from
  the command vector (slot ``cmd_index``). ``boost`` multiplies the term after
  ``boost_after_iterations`` global iterations (one-run curriculum).
  """
  robot: Entity = env.scene[asset_cfg.name]
  sensor = env.scene[height_sensor_name]
  assert isinstance(sensor, TerrainHeightSensor)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = command[:, cmd_index : cmd_index + 1]
  height = sensor.data.heights
  speed = torch.norm(robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  cost = torch.sum(torch.abs(height - target) * speed, dim=1)
  return cost * _boost(env, boost_after_iterations, boost)


def limb_lift_bonus(
  env: ManagerBasedRlEnv,
  height_sensor_name: str,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  cmd_index: int = 3,
  std: float = 0.03,
  boost_after_iterations: int | None = None,
  boost: float = 1.0,
) -> torch.Tensor:
  """Reward moving limbs for being near the commanded swing height (kernel)."""
  robot: Entity = env.scene[asset_cfg.name]
  sensor = env.scene[height_sensor_name]
  assert isinstance(sensor, TerrainHeightSensor)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  target = command[:, cmd_index : cmd_index + 1]
  height = sensor.data.heights
  speed = torch.norm(robot.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1)
  kernel = torch.exp(-torch.square(height - target) / std**2)
  bonus = torch.sum(kernel * torch.clamp(speed, max=1.0), dim=1)
  return bonus * _boost(env, boost_after_iterations, boost)


def forward_progress(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Linear credit for speed along the commanded planar direction.

  Unlike the exponential tracking kernel, this is non-zero (and has a gradient)
  when the robot is far from the target speed, so a stopped robot is always
  pulled toward moving. Clipped at the commanded speed so overshoot earns nothing.
  The direction is [forward, lateral]; with lateral 0 (every task before the
  lateral command) this is the original forward-only term.
  """
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  cmd_xy = command[:, :2]
  speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
  vel_xy = robot.data.root_link_lin_vel_b[:, [2, 1]]
  along = (vel_xy * cmd_xy).sum(dim=-1) / speed.clamp(min=1e-6)
  return torch.clamp(along, min=torch.zeros_like(speed), max=speed)


def crawl_ang_vel_yz_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize base wobble: angular velocity about the non-turning body axes."""
  ang_vel = env.scene["robot"].data.root_link_ang_vel_b
  return torch.sum(torch.square(ang_vel[:, 1:]), dim=1)


def crawl_lin_vel_x_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Penalize vertical bobbing: base-frame X velocity (gravity axis when prone)."""
  return torch.square(env.scene["robot"].data.root_link_lin_vel_b[:, 0])


def stand_still_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
  threshold: float = 0.1,
) -> torch.Tensor:
  """Penalize joint motion while the command is (near) zero: hold a rest stance."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  # Velocity slots only: gait-parameter slots (step height) are never zero.
  idle = torch.linalg.vector_norm(command[:, :3], dim=1) < threshold
  motion = torch.sum(torch.abs(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
  return motion * idle.float()


def align_projected_gravity_plus_x_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Reward lying orientation with gravity projected onto the base +X axis."""
  gravity = env.scene["robot"].data.projected_gravity_b
  target = torch.zeros_like(gravity)
  target[:, 0] = 1.0
  return 1.0 - 0.5 * torch.sum((gravity - target).square(), dim=1)


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  height = robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
  return torch.sum((height - target_height).square(), dim=1)


def joint_deviation_l1(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  deviation = (
    robot.data.joint_pos[:, asset_cfg.joint_ids]
    - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.abs(deviation), dim=1)


def both_limbs_air(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """Return one when both selected contact bodies are airborne."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.current_air_time is not None
  lookup = {name: i for i, name in enumerate(sensor.primary_names)}
  ids = [lookup[name] for name in body_names]
  return (sensor.data.current_air_time[:, ids] > 0.0).all(dim=1).float()


def feet_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  body_names: tuple[str, ...],
) -> torch.Tensor:
  """Penalize horizontal foot/hand speed while the corresponding body contacts."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  lookup = {name: i for i, name in enumerate(sensor.primary_names)}
  sensor_ids = [lookup[name] for name in body_names]
  robot: Entity = env.scene["robot"]
  body_ids, _ = robot.find_bodies(body_names)
  if sensor.data.force_history is not None:
    contacts = (
      torch.linalg.vector_norm(sensor.data.force_history[:, sensor_ids], dim=-1)
      .max(dim=2)
      .values
      > 1.0
    )
  else:
    in_contact = sensor.data.found[:, sensor_ids] > 0
    contacts = in_contact
  speed = torch.linalg.vector_norm(
    robot.data.body_link_lin_vel_w[:, body_ids, :2], dim=-1
  )
  return torch.sum(speed * contacts, dim=1)


def undesired_ground_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  excluded_body_names: tuple[str, ...],
  threshold: float = 1.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  ids = [
    i for i, name in enumerate(sensor.primary_names) if name not in excluded_body_names
  ]
  if sensor.data.force_history is not None:
    force = (
      torch.linalg.vector_norm(sensor.data.force_history[:, ids], dim=-1)
      .max(dim=2)
      .values
    )
  else:
    force = torch.linalg.vector_norm(sensor.data.force[:, ids], dim=-1)
  return (force > threshold).sum(dim=1).float()


def diagonal_gait(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  limb_names: tuple[str, str, str, str],
  command_name: str,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward diagonal limbs touching down and lifting off together while moving.

  ``limb_names`` = (left hand, right hand, left foot, right foot). Per step:
  0.5 for each diagonal pair (left hand + right foot, right hand + left foot)
  whose two limbs share contact state, halved when all four are down. So a trot
  phase (one diagonal planted, the other swinging) scores 1, standing on all
  four 0.5, and a same-side pair 0. The rc4 reference crawl -- a
  diagonal-sequence walk, each foot followed by the opposite hand -- averages
  0.48. Zero when the command asks for less than ``command_threshold``.
  """
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  names = list(sensor.primary_names)
  ids = [names.index(name) for name in limb_names]
  down = sensor.data.force[:, ids].norm(dim=-1) > 1.0
  lh, rh, lf, rf = down.unbind(dim=1)
  sync = 0.5 * ((lh == rf).float() + (rh == lf).float())
  score = sync * (1.0 - 0.5 * down.all(dim=1).float())
  command = env.command_manager.get_command(command_name)
  assert command is not None
  moving = command[:, :3].abs().sum(dim=1) > command_threshold
  return score * moving.float()


# Nominal motor limit per G1 joint, first match wins. Hip roll and knee are
# 7520-22; ankles and waist roll/pitch are two 5020s through a four-bar; wrist
# pitch and yaw are 4010; shoulders, elbows and wrist roll are single 5020s;
# everything left (hip pitch/yaw, waist yaw) is a 7520-14. The last four entries
# only ever match on the 29-DoF robot.
_MOTOR_LIMITS_NM: tuple[tuple[tuple[str, ...], float], ...] = (
  (("hip_roll", "knee"), 139.0),
  (("ankle_", "waist_roll", "waist_pitch"), 50.0),
  (("wrist_pitch", "wrist_yaw"), 5.0),
  (("shoulder_", "elbow", "wrist_roll"), 25.0),
)


def _motor_limit_nm(name: str) -> float:
  for keys, limit in _MOTOR_LIMITS_NM:
    if any(key in name for key in keys):
      return limit
  return 88.0


##
# Torque safety. ``applied_torque_limits`` below cannot fire: it reads
# ``qfrc_actuator``, which MuJoCo has already clamped to each motor's force
# range. These terms read what the joints are actually asked to carry instead.
##

_MOTOR_PARAMS: dict[tuple[int, tuple[str, ...]], tuple[torch.Tensor, ...]] = {}


def _motor_params(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Joint ids and kp, kd, peak torque tensors in matching order (cached)."""
  key = (id(env), tuple(motors))
  if key not in _MOTOR_PARAMS:
    robot: Entity = env.scene["robot"]
    ids, names = robot.find_joints(tuple(motors), preserve_order=True)
    table = torch.tensor([motors[n] for n in names], device=env.device)
    _MOTOR_PARAMS[key] = (
      torch.tensor(ids, device=env.device),
      table[:, 0],
      table[:, 1],
      table[:, 2],
    )
  return _MOTOR_PARAMS[key]  # type: ignore[return-value]


def motor_torque_ratio(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> torch.Tensor:
  """|PD torque demand| / peak torque per joint, shape (num_envs, num_joints).

  The demand ``kp (q_target - q) - kd qdot`` is what the position servo asks
  for before MuJoCo clamps it, so values above 1 mean the motor is saturated and
  the load is winning.
  """
  robot: Entity = env.scene["robot"]
  ids, kp, kd, peak = _motor_params(env, motors)
  demand = (
    kp * (robot.data.joint_pos_target[:, ids] - robot.data.joint_pos[:, ids])
    - kd * robot.data.joint_vel[:, ids]
  )
  return demand.abs() / peak


def motor_torque_margin(
  env: ManagerBasedRlEnv,
  motors: dict[str, tuple[float, float, float]],
  soft_fraction: float = 0.8,
) -> torch.Tensor:
  """Penalty: torque demand above ``soft_fraction`` of each motor's peak.

  Summed over joints, in units of each motor's peak torque so a 139 Nm knee and
  a 25 Nm shoulder count alike. Zero inside the margin; keeps growing past
  saturation, so the gradient points back inside even when the motor is pinned.
  """
  ratio = motor_torque_ratio(env, motors)
  return torch.sum(torch.relu(ratio - soft_fraction), dim=1)


def motor_torque_ratio_max(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> torch.Tensor:
  """Metric: the most loaded motor's demand / peak."""
  return motor_torque_ratio(env, motors).max(dim=1).values


def motor_saturated(
  env: ManagerBasedRlEnv, motors: dict[str, tuple[float, float, float]]
) -> torch.Tensor:
  """Metric: 1 when any motor's demand exceeds its peak torque."""
  return (motor_torque_ratio(env, motors) > 1.0).any(dim=1).float()


def braced_joint_moment(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  axes: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """|Moment| (Nm) across each braced joint about its axis, (num_envs, k).

  Each sensor is a MuJoCo torque sensor at the welded joint's origin, in the
  child body's frame; projecting onto the joint axis gives the load the motor
  and the brace carry together.
  """
  moments = []
  for name, axis in zip(sensor_names, axes, strict=True):
    torque = env.scene[name].data
    axis_t = torch.tensor(axis, dtype=torque.dtype, device=torque.device)
    moments.append((torque @ axis_t).abs())
  return torch.stack(moments, dim=1)


def braced_joint_torque(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  axes: tuple[tuple[float, float, float], ...],
  limit: float,
  soft_fraction: float = 0.8,
) -> torch.Tensor:
  """Penalty: braced-joint moment above ``soft_fraction * limit``, in limits."""
  moments = braced_joint_moment(env, sensor_names, axes)
  return torch.sum(torch.relu(moments / limit - soft_fraction), dim=1)


def braced_joint_moment_max(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
  axes: tuple[tuple[float, float, float], ...],
) -> torch.Tensor:
  """Metric: the largest braced-joint moment (Nm)."""
  return braced_joint_moment(env, sensor_names, axes).max(dim=1).values


def applied_torque_limits(
  env: ManagerBasedRlEnv,
  joint_names: tuple[str, ...],
) -> torch.Tensor:
  """Penalize joint-space actuator forces above the G1's nominal motor limits."""
  robot: Entity = env.scene["robot"]
  joint_ids, names = robot.find_joints(joint_names)
  limits = [_motor_limit_nm(name) for name in names]
  dof_forces = robot.data.qfrc_actuator[:, joint_ids]
  max_force = torch.tensor(limits, dtype=dof_forces.dtype, device=env.device)
  return torch.sum(torch.relu(torch.abs(dof_forces) - max_force), dim=1)
