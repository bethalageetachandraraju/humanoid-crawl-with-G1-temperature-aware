"""Terminations for the crawl task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def crawl_flipped(env: ManagerBasedRlEnv, limit: float = 0.0) -> torch.Tensor:
  """Terminate when gravity no longer points along base +X (robot rolled over).

  Prone crawl has ``projected_gravity_b[:, 0] ~ 1``. Below ``limit`` the body is
  more than 90 degrees from prone (on its back or standing on its head).
  """
  return env.scene["robot"].data.projected_gravity_b[:, 0] < limit
