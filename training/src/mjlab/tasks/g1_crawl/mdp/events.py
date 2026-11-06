"""Reset the MJLab G1 from the checked-in crawl motion capture frames."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ASSETS = Path(__file__).resolve().parents[1] / "assets"
_ANIMATION_PATH = _ASSETS / "animation_rc4.json"
_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@lru_cache(maxsize=4)
def _load_animation(path: Path) -> tuple[torch.Tensor, dict[str, int]]:
  data = json.loads(path.read_text())
  frames = torch.tensor(data["frames"], dtype=torch.float32)
  joint_indices = {
    joint["name"]: int(joint["qposadr"])
    for joint in data["metadata"]["joints"]
    if joint["type"] != "free"
  }
  return frames, joint_indices


def _sample_range(
  ranges: dict[str, tuple[float, float]] | None,
  count: int,
  device: str,
) -> torch.Tensor:
  ranges = ranges or {}
  bounds = torch.tensor(
    [ranges.get(key, (0.0, 0.0)) for key in _POSE_KEYS], device=device
  )
  return sample_uniform(bounds[:, 0], bounds[:, 1], (count, 6), device=device)


def reset_from_animation_frame(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  pose_range: dict[str, tuple[float, float]] | None = None,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  position_range: tuple[float, float] | None = (0.9, 1.1),
  joint_velocity_range: tuple[float, float] | None = (0.0, 0.0),
  animation_path: str | Path | None = None,
) -> None:
  """Sample crawl qpos frames and reset the root plus the selected joints.

  ``animation_path`` defaults to the 23-DoF rc4 capture; the 29-DoF task passes
  the retargeted file next to it. Joints are resolved by name, so an animation
  only has to contain the joints named in ``asset_cfg``; every one of them must
  be present or this raises.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  asset: Entity = env.scene[asset_cfg.name]
  frames, animation_joint_indices = _load_animation(
    Path(animation_path) if animation_path is not None else _ANIMATION_PATH
  )
  frame_ids = torch.randint(len(frames), (len(env_ids),), device="cpu")
  frame = frames[frame_ids].to(device=env.device)

  root_pos = frame[:, :3].clone()
  root_quat = frame[:, 3:7].clone()
  pose_noise = _sample_range(pose_range, len(env_ids), env.device)
  root_pos += pose_noise[:, :3]
  root_quat = quat_mul(
    root_quat,
    quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
  )
  root_pos += env.scene.env_origins[env_ids]

  velocity_noise = _sample_range(velocity_range, len(env_ids), env.device)
  root_vel = velocity_noise
  asset.write_root_link_pose_to_sim(torch.cat((root_pos, root_quat), dim=-1), env_ids)
  asset.write_root_link_velocity_to_sim(root_vel, env_ids)

  joint_ids, joint_names = asset.find_joints(asset_cfg.joint_names or (".*",))
  missing = [name for name in joint_names if name not in animation_joint_indices]
  if missing:
    raise ValueError(f"Crawl animation is missing joints: {missing}")
  qpos_indices = [animation_joint_indices[name] for name in joint_names]
  joint_pos = frame[:, qpos_indices].clone()
  if position_range is not None:
    scale = torch.empty_like(joint_pos).uniform_(*position_range)
    joint_pos *= scale

  joint_vel = torch.zeros_like(joint_pos)
  if joint_velocity_range is not None:
    joint_vel *= torch.empty_like(joint_vel).uniform_(*joint_velocity_range)

  limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
  joint_pos = torch.maximum(torch.minimum(joint_pos, limits[..., 1]), limits[..., 0])
  asset.write_joint_state_to_sim(
    joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids
  )
