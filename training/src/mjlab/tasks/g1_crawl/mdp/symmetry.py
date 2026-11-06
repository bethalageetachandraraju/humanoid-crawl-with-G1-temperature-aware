"""Left-right reflection of the crawl policy's observations and actions.

The mirror map is derived from the joint-name tuple so the 23-DoF and 29-DoF
layouts share one implementation: a joint swaps with its opposite-side twin, and
picks up a sign flip if it rotates about the robot's roll (X) or yaw (Z) axis.
Pitch joints -- knee, elbow, hip/ankle/shoulder/wrist/waist pitch -- keep their
sign. ``_JOINT_PERMUTATION_23`` / ``_JOINT_SIGN_23`` are the tables the task
shipped with and are kept as a regression check on that rule.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from tensordict import TensorDict

from mjlab.tasks.g1_crawl.robot import CRAWL_JOINTS

_JOINT_PERMUTATION_23 = (
  6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 18, 19, 20, 21, 22, 13, 14, 15, 16, 17,
)  # fmt: skip
_JOINT_SIGN_23 = (
  1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1,
)  # fmt: skip


def _mirror_name(name: str) -> str:
  if name.startswith("left_"):
    return "right_" + name[len("left_") :]
  if name.startswith("right_"):
    return "left_" + name[len("right_") :]
  return name


def mirror_map(joints: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
  """Index permutation and sign flip that reflect a joint vector left-right."""
  index = {name: i for i, name in enumerate(joints)}
  permutation, sign = [], []
  for name in joints:
    twin = _mirror_name(name)
    if twin not in index:
      raise ValueError(f"{name!r} has no left-right counterpart in the joint set")
    permutation.append(index[twin])
    sign.append(-1 if ("roll" in name or "yaw" in name) else 1)
  return tuple(permutation), tuple(sign)


assert mirror_map(CRAWL_JOINTS) == (_JOINT_PERMUTATION_23, _JOINT_SIGN_23)


@lru_cache(maxsize=8)
def _mirror_tensors(
  permutation: tuple[int, ...],
  sign: tuple[int, ...],
  device: torch.device,
  dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
  return (
    torch.tensor(permutation, device=device),
    torch.tensor(sign, dtype=dtype, device=device),
  )


def _mirror_joint_block(
  block: torch.Tensor, permutation: tuple[int, ...], sign: tuple[int, ...]
) -> torch.Tensor:
  index, flip = _mirror_tensors(permutation, sign, block.device, block.dtype)
  return block.index_select(-1, index) * flip


# Under the left-right reflection (base Y -> -Y) a velocity keeps X and Z and
# flips Y, while an angular velocity -- a pseudovector -- flips X and Z and keeps
# Y. Base X is vertical when prone, so angular X is the turn rate.
_ANG_VEL_SIGN = (-1.0, 1.0, -1.0)


def make_crawl_symmetry_augmentation(
  num_cmd: int = 3, joints: tuple[str, ...] = CRAWL_JOINTS, gyro: bool = False
):
  """Build the RSL-RL data-augmentation function for one task layout.

  Actor:  gravity(3), [base angular velocity(3) if ``gyro``], command(num_cmd),
          then joint pos / vel / previous action.
  Critic: base lin vel(3), ang vel(3), gravity(3), command(num_cmd), joints.
  """
  permutation, sign = mirror_map(joints)
  num_joints = len(joints)

  def mirror_blocks(obs: torch.Tensor, start: int) -> torch.Tensor:
    out = obs.clone()
    for block in range(3):
      a = start + num_joints * block
      out[..., a : a + num_joints] = _mirror_joint_block(
        obs[..., a : a + num_joints], permutation, sign
      )
    return out

  def mirror_ang_vel(obs: torch.Tensor, at: int) -> None:
    for axis, flip in enumerate(_ANG_VEL_SIGN):
      if flip < 0:
        obs[..., at + axis] *= -1

  def mirror_command(obs: torch.Tensor, at: int) -> None:
    obs[..., at + 1] *= -1  # Commanded lateral velocity.
    obs[..., at + 2] *= -1  # Commanded turn rate.
    # Slots 3+ (step height, spares) are left-right symmetric.

  actor_cmd = 6 if gyro else 3

  def mirror_actor(actor: torch.Tensor) -> torch.Tensor:
    mirrored = mirror_blocks(actor, actor_cmd + num_cmd)
    mirrored[..., 1] *= -1  # Projected gravity Y.
    if gyro:
      mirror_ang_vel(mirrored, 3)
    mirror_command(mirrored, actor_cmd)
    return mirrored

  def mirror_critic(critic: torch.Tensor) -> torch.Tensor:
    mirrored = mirror_blocks(critic, 9 + num_cmd)
    mirrored[..., 1] *= -1  # Base-frame lateral velocity.
    # Angular velocity. Before 2026-10-12 only Z was flipped here, so in every
    # mirrored sample the critic saw the turn rate with the wrong sign.
    mirror_ang_vel(mirrored, 3)
    mirrored[..., 7] *= -1  # Projected gravity Y.
    mirror_command(mirrored, 9)
    return mirrored

  def crawl_symmetry_augmentation(
    env: object,
    obs: TensorDict | None,
    actions: torch.Tensor | None,
  ) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append mirrored actor/critic observations and action targets."""
    del env
    mirrored_obs = None
    if obs is not None:
      actor, critic = obs["actor"], obs["critic"]
      assert isinstance(actor, torch.Tensor) and isinstance(critic, torch.Tensor)
      mirrored_obs = obs.clone()
      mirrored_obs["actor"] = mirror_actor(actor)
      mirrored_obs["critic"] = mirror_critic(critic)
      mirrored_obs = TensorDict.cat([obs, mirrored_obs], dim=0)

    mirrored_actions = None
    if actions is not None:
      mirrored_actions = torch.cat(
        (actions, _mirror_joint_block(actions, permutation, sign)), dim=0
      )
    return mirrored_obs, mirrored_actions

  return crawl_symmetry_augmentation


crawl_symmetry_augmentation = make_crawl_symmetry_augmentation(3)
crawl_symmetry_augmentation_cmd7 = make_crawl_symmetry_augmentation(7)
