"""Terrain curriculum for the crawl task (port of the source's crawl variant)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def terrain_levels_crawl(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  flat_iterations: int = 0,
  spread_after_iterations: int | None = None,
  steps_per_iteration: int = 24,
  flat_terrain_name: str = "flat",
  promotion_fraction: float = 0.8,
  min_commanded_distance: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> dict[str, torch.Tensor]:
  """Promote/demote envs by distance crawled versus distance commanded.

  The crawl command's first entry is the forward (base-Z) speed, so the
  commanded distance is ``|cmd_z| * episode_length``. Levels are frozen at their
  initial value for the first ``flat_iterations`` learning iterations so the
  policy first settles on the ``flat_terrain_name`` tile type at level 0, which is
  physically identical to a plane; at unlock every env gets a random tile type
  at level 0 and the usual promotion rule runs. Iteration counts are global:
  MJLab restores ``common_step_counter`` when resuming from a checkpoint.
  After ``spread_after_iterations`` (if set) promotion stops and every resetting
  env is placed on a uniformly random level and tile type, so the final phase
  trains on all terrains with equal frequency regardless of performance.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None
  generator = terrain.cfg.terrain_generator
  assert generator is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None

  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )
  commanded = torch.abs(command[env_ids, 0]) * env.max_episode_length_s
  move_up = distance > torch.clamp(
    commanded * promotion_fraction, max=generator.size[0] / 2
  )
  move_up &= commanded >= min_commanded_distance
  move_down = (distance < commanded * 0.5) & ~move_up

  step = env.common_step_counter
  warmup = step < flat_iterations * steps_per_iteration
  spread = (
    spread_after_iterations is not None
    and step >= spread_after_iterations * steps_per_iteration
  )
  assert terrain.terrain_origins is not None
  if spread:
    terrain.randomize_env_origins(env_ids)
  elif warmup or step == 0:
    flat_type = list(generator.sub_terrains.keys()).index(flat_terrain_name)
    terrain.terrain_types[env_ids] = flat_type
    terrain.terrain_levels[env_ids] = 0
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    env._crawl_terrain_unlocked = False  # type: ignore[attr-defined]
  else:
    if not getattr(env, "_crawl_terrain_unlocked", False):
      # First reset after the warm-up: give every env a random tile type.
      num_types = len(generator.sub_terrains)
      all_ids = torch.arange(env.num_envs, device=env.device)
      terrain.terrain_types[:] = torch.randint(
        0, num_types, (env.num_envs,), device=env.device
      )
      terrain.terrain_levels[:] = 0
      terrain.env_origins[:] = terrain.terrain_origins[
        terrain.terrain_levels[all_ids], terrain.terrain_types[all_ids]
      ]
      env._crawl_terrain_unlocked = True  # type: ignore[attr-defined]
      move_up = torch.zeros_like(move_up)
      move_down = torch.zeros_like(move_down)
    terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  return {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
    "unlocked": torch.tensor(0.0 if warmup else 1.0, device=levels.device),
    "spread": torch.tensor(1.0 if spread else 0.0, device=levels.device),
  }


def terrain_zones_crawl(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  flat_iterations: int = 0,
  spread_after_iterations: int | None = None,
  steps_per_iteration: int = 24,
  mix_zone_name: str | None = "random_mix",
  promotion_fraction: float = 0.8,
  min_commanded_distance: float = 1.0,
  flat_anchor_fraction: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  score_fn: Callable[
    [ManagerBasedRlEnv, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
  ]
  | None = None,
  relative_to_start: bool = False,
) -> dict[str, torch.Tensor]:
  """Left-to-right zone curriculum over the sub-terrain columns.

  The first ``flat_anchor_fraction`` of envs stay in the flat zone for the whole
  run so the base gait is never forgotten. Episodes that ended by a termination
  (rolled over) never promote; they demote.

  Each env carries a progress index ``p`` in ``[0, num_zones * num_rows)``:
  ``zone = p // num_rows`` (sub-terrain in declaration order, left to right) and
  ``row = p % num_rows`` (difficulty within the zone). Promotion/demotion moves
  ``p`` by one. The optional ``mix_zone_name`` zone places its envs on a random
  one of the other non-flat zones at the same row. Phases follow
  ``terrain_levels_crawl``: warm-up in zone 0 row 0, then the curriculum, then
  uniform spread over all zones and rows. Iteration counts are global unless
  ``relative_to_start`` counts them from the first call (fine-tuning a resumed
  checkpoint, whose step counter is restored).

  ``score_fn(env, env_ids)`` replaces the crawl command's progress with
  ``(distance made good, distance commanded)`` per finished episode, for tasks
  whose command term is not the crawl's.
  """
  asset: Entity = env.scene[asset_cfg.name]
  terrain = env.scene.terrain
  assert terrain is not None and terrain.terrain_origins is not None
  generator = terrain.cfg.terrain_generator
  assert generator is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None
  names = list(generator.sub_terrains.keys())
  num_zones = len(names)
  num_rows = terrain.terrain_origins.shape[0]
  mix_zone = names.index(mix_zone_name) if mix_zone_name in names else -1
  device = env.device
  # Flat tiles are identical at every difficulty: the flat zone is one step.
  rows_per_zone = torch.tensor(
    [1 if n == "flat" else num_rows for n in names], device=device
  )
  zone_offset = torch.cumsum(rows_per_zone, dim=0) - rows_per_zone
  total_steps = int(rows_per_zone.sum())
  cmd_term = env.command_manager.get_term(command_name)

  if not hasattr(env, "_crawl_zone_progress"):
    env._crawl_zone_progress = torch.zeros(  # type: ignore[attr-defined]
      env.num_envs, dtype=torch.long, device=device
    )
  progress: torch.Tensor = env._crawl_zone_progress  # type: ignore[attr-defined]

  step = env.common_step_counter
  if relative_to_start:
    if not hasattr(env, "_zone_start_step"):
      env._zone_start_step = step  # type: ignore[attr-defined]
    step -= env._zone_start_step  # type: ignore[attr-defined]
  warmup = step < flat_iterations * steps_per_iteration
  spread = (
    spread_after_iterations is not None
    and step >= spread_after_iterations * steps_per_iteration
  )

  if spread:
    progress[env_ids] = torch.randint(0, total_steps, (len(env_ids),), device=device)
  elif warmup or step == 0:
    progress[env_ids] = 0
  else:
    # Distance crawled in the commanded direction versus distance commanded,
    # both accumulated over the episode (robust to command sign changes/idle).
    if score_fn is not None:
      distance, commanded = score_fn(env, env_ids)
    elif hasattr(cmd_term, "progress_acc"):
      distance = cmd_term.progress_acc[env_ids]  # type: ignore[attr-defined]
      commanded = cmd_term.commanded_acc[env_ids]  # type: ignore[attr-defined]
    else:
      distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
      )
      commanded = torch.abs(command[env_ids, 0]) * env.max_episode_length_s
    move_up = distance > commanded * promotion_fraction
    move_up &= commanded >= min_commanded_distance
    move_down = (distance < commanded * 0.5) & ~move_up
    terminated = env.termination_manager.terminated[env_ids]
    move_up &= ~terminated
    move_down |= terminated
    progress[env_ids] = torch.clamp(
      progress[env_ids] + move_up.long() - move_down.long(), 0, total_steps - 1
    )
  anchors = env_ids < int(flat_anchor_fraction * env.num_envs)
  progress[env_ids[anchors]] = 0

  p_ids = progress[env_ids]
  zone = torch.searchsorted(zone_offset, p_ids, right=True) - 1
  row = p_ids - zone_offset[zone]
  tile = zone.clone()
  if mix_zone >= 0:
    candidates = [i for i in range(num_zones) if i not in (0, mix_zone)]
    pick = torch.tensor(candidates, device=device)[
      torch.randint(0, len(candidates), (len(env_ids),), device=device)
    ]
    tile = torch.where(zone == mix_zone, pick, zone)
  terrain.terrain_types[env_ids] = tile
  terrain.terrain_levels[env_ids] = row
  terrain.env_origins[env_ids] = terrain.terrain_origins[row, tile]

  all_zone_i = torch.searchsorted(zone_offset, progress, right=True) - 1
  all_zone = all_zone_i.float()
  out: dict[str, torch.Tensor] = {
    "zone_mean": all_zone.mean(),
    "zone_max": all_zone.max(),
    "row_mean": (progress - zone_offset[all_zone_i]).float().mean(),
    "unlocked": torch.tensor(0.0 if warmup else 1.0, device=device),
    "spread": torch.tensor(1.0 if spread else 0.0, device=device),
  }
  for i, name in enumerate(names):
    out[f"frac_{name}"] = (all_zone == i).float().mean()
  return out
