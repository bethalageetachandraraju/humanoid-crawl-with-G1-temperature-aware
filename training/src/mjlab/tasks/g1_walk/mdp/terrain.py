"""Terrain-aware pieces for the rough-ground walk and stand-to-crawl tasks.

The flat tasks measure heights against the world origin. On generated terrain
the ground is wherever the tile puts it, so the termination here reads the
ground under the feet, and the two scores below feed the crawl task's zone
curriculum (``terrain_zones_crawl``) with this task's notion of progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import TerrainHeightSensor
from mjlab.tasks.g1_crawl.mdp.curriculums import terrain_zones_crawl
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# A walk that commands more than a couple of tiles of travel is judged on the
# first few metres: the robot leaves its tile long before the episode ends.
MAX_SCORED_DISTANCE_M = 4.0


def root_height_above_ground_below(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  height_sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Pelvis lower than ``minimum_height`` over the ground under the feet.

  Ground height under a foot is its site height minus the height-scan reading.
  Terminates a robot that has sat or knelt down, on any terrain.
  """
  robot = env.scene[asset_cfg.name]
  sensor = env.scene[height_sensor_name]
  assert isinstance(sensor, TerrainHeightSensor)
  site_z = robot.data.site_pos_w[:, asset_cfg.site_ids, 2]
  ground = (site_z - sensor.data.heights).mean(dim=1)
  return robot.data.root_link_pos_w[:, 2] - ground < minimum_height


def walk_progress(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str = "twist"
) -> tuple[torch.Tensor, torch.Tensor]:
  """(distance made good along the command, distance commanded) per episode."""
  cmd = env.command_manager.get_term(command_name)
  assert isinstance(cmd, UniformVelocityCommand)
  commanded = cmd.commanded_displacement_w[env_ids]
  actual = (
    cmd.robot.data.root_link_pos_w[env_ids, :2] - cmd.episode_start_pos_w[env_ids]
  )
  length = torch.linalg.vector_norm(commanded, dim=1)
  along = (actual * commanded).sum(dim=1) / length.clamp(min=1e-6)
  scale = (MAX_SCORED_DISTANCE_M / length.clamp(min=1e-6)).clamp(max=1.0)
  return along.clamp(min=0.0) * scale, length * scale


def transition_progress(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "phase",
  success_pose_err: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
  """1 of 1 for an episode that reached the crawl pose, 0 of 1 for one that did not."""
  cmd = env.command_manager.get_term(command_name)
  reached = (cmd.phase[env_ids] >= 1.0) & (cmd.pose_err[env_ids] < success_pose_err)  # type: ignore[attr-defined]
  return reached.float(), torch.ones_like(reached, dtype=torch.float)


def terrain_zones_walk(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  flat_iterations: int = 0,
  spread_after_iterations: int | None = None,
  mix_zone_name: str | None = "random_mix",
) -> dict[str, torch.Tensor]:
  """The crawl's zone curriculum, scored on distance walked over commanded."""
  return terrain_zones_crawl(
    env,
    env_ids,
    command_name="twist",
    flat_iterations=flat_iterations,
    spread_after_iterations=spread_after_iterations,
    mix_zone_name=mix_zone_name,
    score_fn=walk_progress,
    relative_to_start=True,
  )


def terrain_zones_transition(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  flat_iterations: int = 0,
  spread_after_iterations: int | None = None,
  mix_zone_name: str | None = None,
) -> dict[str, torch.Tensor]:
  """The crawl's zone curriculum, scored on reaching the crawl pose."""
  return terrain_zones_crawl(
    env,
    env_ids,
    command_name="phase",
    flat_iterations=flat_iterations,
    spread_after_iterations=spread_after_iterations,
    mix_zone_name=mix_zone_name,
    score_fn=transition_progress,
    relative_to_start=True,
  )
