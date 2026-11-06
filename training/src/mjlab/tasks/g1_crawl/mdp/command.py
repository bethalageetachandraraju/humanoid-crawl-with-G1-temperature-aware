"""Velocity commands for the G1 crawl gait (base Z, base Y, and base roll)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class CrawlVelocityCommandCfg(CommandTermCfg):
  """Configuration matching the original three-axis crawl command."""

  entity_name: str = "robot"
  lin_vel_z: tuple[float, float] = (-1.0, 2.5)
  lin_vel_y: tuple[float, float] = (0.0, 0.0)
  ang_vel_x: tuple[float, float] = (-3.0, 3.0)
  rel_standing_envs: float = 0.05
  """Fraction of commands that are all zero (rest stance)."""
  standing_time_range: tuple[float, float] | None = None
  """How long a rest-stance command lasts (s); None: ``resampling_time_range``.
  On the robot the crawl rests at zero command for tens of seconds."""
  # Command modes. Each resample draws one mode; whatever probability is left
  # over after the four below is "mixed": forward, lateral and turn all drawn
  # independently. With all of them 0 (the default) every non-standing command
  # is mixed, which is the original behaviour.
  rel_straight_envs: float = 0.0
  """Pure forward/backward: lateral 0, turn 0 (or heading hold, below)."""
  straight_fwd_range: tuple[float, float] | None = None
  """Forward speed of straight commands (m/s); None: ``lin_vel_z``. A cruise
  speed band, for a crawl that should spend most of its time near it."""
  rel_sideways_envs: float = 0.0
  """Pure sideways crawl: forward 0, turn 0 (or heading hold)."""
  rel_spin_envs: float = 0.0
  """Turn in place: forward and lateral 0."""
  heading_stiffness: float = 0.0
  """Heading hold for straight and sideways commands: the turn slot becomes
  ``-k * wrap(target - heading)`` (clamped to ``heading_max_turn``), holding the
  world heading the command started at. That is what an operator's "go straight"
  looks like on the robot (the same loop runs at deployment), and it turns slow
  heading drift into a turn command the policy is rewarded for following.
  Positive turn commands are clockwise seen from above (base X points down)."""
  heading_max_turn: float = 0.5
  # Optional gait-parameter slots appended to the 3 velocity entries. When
  # ``step_height`` is set the command is [fwd, lat, turn, step_height, spares...].
  step_height: tuple[float, float] | None = None
  """Global range (m) of the commanded limb swing height; None disables it."""
  step_height_by_terrain: dict[str, tuple[float, float]] | None = None
  """Per sub-terrain-name ranges used with probability ``terrain_prob``."""
  terrain_prob: float = 0.8
  num_spare: int = 0
  """Reserved command slots, always 0.0 (keeps the network input fixed)."""

  def build(self, env: ManagerBasedRlEnv) -> CrawlVelocityCommand:
    return CrawlVelocityCommand(self, env)


class CrawlVelocityCommand(CommandTerm):
  """Sample [linear-z, linear-y, angular-x] in the robot base frame."""

  cfg: CrawlVelocityCommandCfg

  def __init__(self, cfg: CrawlVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    self.num_cmd = 3 + (0 if cfg.step_height is None else 1 + cfg.num_spare)
    self.vel_command_b = torch.zeros(self.num_envs, self.num_cmd, device=self.device)
    self._terrain_ranges: torch.Tensor | None = None
    terrain = env.scene.terrain
    if (
      cfg.step_height_by_terrain is not None
      and terrain is not None
      and terrain.cfg.terrain_generator is not None
    ):
      names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
      lo_hi = [cfg.step_height_by_terrain.get(n, cfg.step_height) for n in names]
      self._terrain_ranges = torch.tensor(lo_hi, device=self.device)
    self.is_standing_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Heading hold state (straight and sideways commands).
    self.is_heading_env = torch.zeros_like(self.is_standing_env)
    self._heading_unset = torch.zeros_like(self.is_standing_env)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yz"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_roll"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
    self.manual = False
    # Per-episode accumulators for the terrain curriculum: distance crawled in
    # the commanded direction and total commanded distance (both in metres).
    self.progress_acc = torch.zeros(self.num_envs, device=self.device)
    self.commanded_acc = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def manual_control(self) -> None:
    """Hand the command to an outside driver (keyboard, joystick, deployment).

    Stops resampling and every automatic override -- the rest stance and the
    heading hold of straight/sideways modes -- so the slots hold exactly what
    the driver writes. Without this, a heading-hold mode drawn at startup keeps
    replacing the turn slot every step and the robot ignores turn commands.
    """
    self.manual = True
    self.is_standing_env[:] = False
    self.is_heading_env[:] = False
    self._heading_unset[:] = False
    self.vel_command_b[:] = 0.0

  def heading(self) -> torch.Tensor:
    """World heading (rad) of the base forward axis (+Z, horizontal when prone)."""
    w, x, y, z = self.robot.data.root_link_quat_w.unbind(dim=-1)
    return torch.atan2(2 * (y * z - w * x), 2 * (x * z + w * y))

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    if self.manual:
      return
    n = len(env_ids)
    cmd = torch.empty(n, 3, device=self.device)
    cmd[:, 0].uniform_(*self.cfg.lin_vel_z)
    cmd[:, 1].uniform_(*self.cfg.lin_vel_y)
    cmd[:, 2].uniform_(*self.cfg.ang_vel_x)
    # One mode per command: standing, straight, sideways, spin, else mixed.
    u = torch.rand(n, device=self.device)
    edges = torch.cumsum(
      torch.tensor(
        [
          self.cfg.rel_standing_envs,
          self.cfg.rel_straight_envs,
          self.cfg.rel_sideways_envs,
          self.cfg.rel_spin_envs,
        ],
        device=self.device,
      ),
      dim=0,
    )
    standing = u < edges[0]
    straight = (u >= edges[0]) & (u < edges[1])
    sideways = (u >= edges[1]) & (u < edges[2])
    spin = (u >= edges[2]) & (u < edges[3])
    cmd[straight, 1:] = 0.0
    if self.cfg.straight_fwd_range is not None and straight.any():
      cmd[straight, 0] = torch.empty(int(straight.sum()), device=self.device).uniform_(
        *self.cfg.straight_fwd_range
      )
    cmd[sideways, 0] = 0.0
    cmd[sideways, 2] = 0.0
    cmd[spin, :2] = 0.0
    self.vel_command_b[env_ids, :3] = cmd
    self.is_standing_env[env_ids] = standing
    if self.cfg.standing_time_range is not None and standing.any():
      rest = env_ids[standing]
      self.time_left[rest] = self.time_left[rest].uniform_(
        *self.cfg.standing_time_range
      )
    heading = (straight | sideways) & (self.cfg.heading_stiffness > 0.0)
    self.is_heading_env[env_ids] = heading
    self._heading_unset[env_ids] = heading  # Latched on the next update.
    if self.cfg.step_height is not None:
      height = torch.empty(len(env_ids), device=self.device).uniform_(
        *self.cfg.step_height
      )
      terrain = self._env.scene.terrain
      if self._terrain_ranges is not None and terrain is not None:
        rng = self._terrain_ranges[terrain.terrain_types[env_ids]]
        by_terrain = rng[:, 0] + torch.rand_like(height) * (rng[:, 1] - rng[:, 0])
        use_terrain = torch.rand_like(height) < self.cfg.terrain_prob
        height = torch.where(use_terrain, by_terrain, height)
      self.vel_command_b[env_ids, 3] = height
      self.vel_command_b[env_ids, 4:] = 0.0

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    out = super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.progress_acc[ids] = 0.0
    self.commanded_acc[ids] = 0.0
    return out

  def _update_metrics(self) -> None:
    max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
    vel = self.robot.data.root_link_lin_vel_b
    dt = self._env.step_dt
    # Planar [forward, lateral]: distance crawled along the commanded direction.
    cmd_xy = self.vel_command_b[:, :2]
    speed = torch.linalg.vector_norm(cmd_xy, dim=-1)
    along = (vel[:, [2, 1]] * cmd_xy).sum(dim=-1) / speed.clamp(min=1e-6)
    self.progress_acc += torch.clamp(along, min=0.0) * dt
    self.commanded_acc += speed * dt
    self.metrics["error_vel_yz"] += (
      torch.linalg.vector_norm(self.vel_command_b[:, :2] - vel[:, [2, 1]], dim=-1)
      / max_steps
    )
    self.metrics["error_vel_roll"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 0])
      / max_steps
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids
    if self.manual:
      return
    standing = self.is_standing_env
    self.vel_command_b[standing, :3] = 0.0  # Keep gait-parameter slots.
    if self.cfg.heading_stiffness > 0.0:
      heading = self.heading()
      # Latch the target after the reset has placed the robot.
      unset = self._heading_unset
      self.heading_target[unset] = heading[unset]
      self._heading_unset[:] = False
      error = _wrap(self.heading_target - heading)
      hold = self.is_heading_env
      turn = (-self.cfg.heading_stiffness * error).clamp(
        -self.cfg.heading_max_turn, self.cfg.heading_max_turn
      )
      self.vel_command_b[hold, 2] = turn[hold]
      max_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
      self.metrics["error_heading"] += error.abs() * hold.float() / max_steps


def _wrap(angle: torch.Tensor) -> torch.Tensor:
  return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi
