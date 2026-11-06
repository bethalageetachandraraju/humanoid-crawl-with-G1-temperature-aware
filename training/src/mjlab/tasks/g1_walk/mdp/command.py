"""The velocity command with explicit spin-in-place episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass(kw_only=True)
class WalkVelocityCommandCfg(UniformVelocityCommandCfg):
  rel_spin_envs: float = 0.0
  """Fraction of episodes that turn in place: no linear velocity, a yaw rate of
  at least ``spin_min`` rad/s. Uniform sampling of three ranges almost never
  produces a pure turn, so without these the policy never learns to step round."""
  spin_min: float = 0.3

  def build(self, env: ManagerBasedRlEnv) -> WalkVelocityCommand:
    return WalkVelocityCommand(self, env)


class WalkVelocityCommand(UniformVelocityCommand):
  cfg: WalkVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    r = torch.empty(len(env_ids), device=self.device)
    spin = env_ids[r.uniform_(0.0, 1.0) <= self.cfg.rel_spin_envs]
    if len(spin) == 0:
      return
    self.vel_command_b[spin, :2] = 0.0
    self.vel_command_w[spin, :2] = 0.0
    yaw = self.vel_command_b[spin, 2]
    hi = self.cfg.ranges.ang_vel_z[1]
    self.vel_command_b[spin, 2] = torch.sign(yaw) * yaw.abs().clamp(
      min=min(self.cfg.spin_min, hi)
    )
    self.is_heading_env[spin] = False  # The sampled rate is the command.
    self.is_standing_env[spin] = False
    self.is_forward_env[spin] = False
