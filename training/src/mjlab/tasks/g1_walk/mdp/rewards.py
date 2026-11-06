"""Walk-task rewards beyond MJLab's velocity set."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def yaw_progress(
  env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.1
) -> torch.Tensor:
  """Linear credit for yaw rate in the commanded direction, capped at the command.

  The exponential tracking kernel pays a robot that stands through a turn
  command most of the reward, so from a standstill it never finds out that it
  has to step round. This term keeps a gradient toward turning until the
  commanded rate is reached (the crawl task's ``forward_progress``, for yaw).
  """
  robot: Entity = env.scene["robot"]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  wz_cmd = command[:, 2]
  active = (wz_cmd.abs() > threshold).float()
  toward = robot.data.root_link_ang_vel_b[:, 2] * torch.sign(wz_cmd)
  return active * torch.minimum(toward.clamp(min=0.0), wz_cmd.abs())
