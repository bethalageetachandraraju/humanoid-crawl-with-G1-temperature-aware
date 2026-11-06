"""Left-right reflection of the walk policy's observations and actions.

Same rule as the crawl task's (``g1_crawl/mdp/symmetry.py``: a joint swaps with
its opposite-side twin and flips sign if it rotates about roll or yaw), applied
to the walk task's layouts:

Actor (84):  gravity(3), gyro(3), command [vx, vy, wz](3), joint pos / vel /
             last action (3 x 25).
Critic (97): base lin vel(3), gravity(3), gyro(3), command(3), the three joint
             blocks, foot air time(2), foot contact(2), foot contact forces
             (2 x 3), feet in the order (left, right).

Under the reflection a velocity flips Y, an angular velocity (pseudovector)
flips X and Z, the lateral and turn commands flip, and the feet swap.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from mjlab.tasks.g1_crawl.mdp.symmetry import _mirror_joint_block, mirror_map
from mjlab.tasks.g1_walk.robot import JOINTS_29

_ANG_VEL_SIGN = (-1.0, 1.0, -1.0)


def make_walk_symmetry_augmentation(joints: tuple[str, ...] = JOINTS_29):
  permutation, sign = mirror_map(joints)
  n = len(joints)

  def mirror_blocks(obs: torch.Tensor, start: int) -> torch.Tensor:
    out = obs.clone()
    for block in range(3):
      a = start + n * block
      out[..., a : a + n] = _mirror_joint_block(obs[..., a : a + n], permutation, sign)
    return out

  def mirror_ang_vel(obs: torch.Tensor, at: int) -> None:
    for axis, flip in enumerate(_ANG_VEL_SIGN):
      if flip < 0:
        obs[..., at + axis] *= -1

  def mirror_command(obs: torch.Tensor, at: int) -> None:
    obs[..., at + 1] *= -1  # Lateral velocity.
    obs[..., at + 2] *= -1  # Turn rate.

  def mirror_actor(actor: torch.Tensor) -> torch.Tensor:
    out = mirror_blocks(actor, 9)
    out[..., 1] *= -1  # Projected gravity Y.
    mirror_ang_vel(out, 3)
    mirror_command(out, 6)
    return out

  def mirror_critic(critic: torch.Tensor) -> torch.Tensor:
    out = mirror_blocks(critic, 12)
    out[..., 1] *= -1  # Base-frame lateral velocity.
    out[..., 4] *= -1  # Projected gravity Y.
    mirror_ang_vel(out, 6)
    mirror_command(out, 9)
    feet = 12 + 3 * n
    # Air time (L, R), contact (L, R): swap. Forces (L xyz, R xyz): swap, flip Y.
    out[..., feet : feet + 2] = critic[..., [feet + 1, feet]]
    out[..., feet + 2 : feet + 4] = critic[..., [feet + 3, feet + 2]]
    f = feet + 4
    out[..., f : f + 3] = critic[..., f + 3 : f + 6]
    out[..., f + 3 : f + 6] = critic[..., f : f + 3]
    out[..., f + 1] *= -1
    out[..., f + 4] *= -1
    return out

  def walk_symmetry_augmentation(
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
      assert actor.shape[-1] == 9 + 3 * n and critic.shape[-1] == 12 + 3 * n + 10
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

  return walk_symmetry_augmentation


def make_stand_crawl_symmetry_augmentation(joints: tuple[str, ...] = JOINTS_29):
  """The stand <-> crawl layouts: the walk's, without the velocity command, with
  the phase command [phi, dphi/dt] (unchanged by the reflection) and the
  reference-minus-measured joint block (a joint block: the schedule's keyframes
  are left-right symmetric) appended.

  Actor (108):  gravity(3), gyro(3), joint pos / vel / last action (3 x 25),
                phase(2), ref - q(25).
  Critic (131): base lin vel(3), gravity(3), gyro(3), the three joint blocks,
                limb air time(4), limb contact(4), limb contact forces (4 x 3),
                phase(2), ref - q(25). The limbs are (left foot, right foot,
                left extension tip, right extension tip): the four-limb sensor.
  """
  permutation, sign = mirror_map(joints)
  n = len(joints)

  def mirror_joints(obs: torch.Tensor, out: torch.Tensor, start: int) -> None:
    out[..., start : start + n] = _mirror_joint_block(
      obs[..., start : start + n], permutation, sign
    )

  def mirror_ang_vel(obs: torch.Tensor, at: int) -> None:
    for axis, flip in enumerate(_ANG_VEL_SIGN):
      if flip < 0:
        obs[..., at + axis] *= -1

  def mirror_actor(actor: torch.Tensor) -> torch.Tensor:
    out = actor.clone()
    out[..., 1] *= -1  # Projected gravity Y.
    mirror_ang_vel(out, 3)
    for block in range(3):
      mirror_joints(actor, out, 6 + n * block)
    mirror_joints(actor, out, 6 + 3 * n + 2)
    return out

  def mirror_critic(critic: torch.Tensor) -> torch.Tensor:
    out = critic.clone()
    out[..., 1] *= -1  # Base-frame lateral velocity.
    out[..., 4] *= -1  # Projected gravity Y.
    mirror_ang_vel(out, 6)
    for block in range(3):
      mirror_joints(critic, out, 9 + n * block)
    limbs = 9 + 3 * n
    swap = [1, 0, 3, 2]  # Left <-> right foot, left <-> right tip.
    for block in (limbs, limbs + 4):  # Air time, contact.
      out[..., block : block + 4] = critic[..., [block + i for i in swap]]
    f = limbs + 8
    for i, j in enumerate(swap):
      out[..., f + 3 * i : f + 3 * i + 3] = critic[..., f + 3 * j : f + 3 * j + 3]
      out[..., f + 3 * i + 1] *= -1  # Force Y.
    mirror_joints(critic, out, f + 12 + 2)
    return out

  def stand_crawl_symmetry_augmentation(
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
      assert actor.shape[-1] == 6 + 4 * n + 2, actor.shape
      assert critic.shape[-1] == 9 + 4 * n + 22, critic.shape
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

  return stand_crawl_symmetry_augmentation
