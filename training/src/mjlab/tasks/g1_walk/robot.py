"""The braced 29-DoF crawl G1 (``g1_crawl/robot29.py``), standing up.

The walk and transition tasks use the robot the crawl task trains on -- hands
removed, wrist pitch/yaw braced, printed forearm extensions, 25 active joints
in ``JOINTS_29`` order -- so one robot model serves every policy that runs on
the bench, and a transition can hand over to the crawl policy in its own pose.
Only the initial state differs: the asset zoo's knees-bent standing keyframe.
"""

from __future__ import annotations

import re

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import KNEES_BENT_KEYFRAME
from mjlab.entity import EntityCfg
from mjlab.tasks.g1_crawl.robot29 import (
  CRAWL_ARTICULATION_29,
  CRAWL_INIT_STATE_29,
  CRAWL_JOINTS_29,
  CRAWL_SELF_COLLISION_29,
  get_crawl_spec_29,
)
from mjlab.utils.spec_config import CollisionCfg

JOINTS_29 = CRAWL_JOINTS_29
"""Policy joint order, shared with the crawl policy."""


def resolve_pose(
  joint_pos: dict[str, float], joints: tuple[str, ...] = JOINTS_29
) -> dict[str, float]:
  """Keyframe (regex -> value, later patterns win) to one value per joint."""
  pose = {}
  for name in joints:
    value = 0.0
    for pattern, v in joint_pos.items():
      if re.fullmatch(pattern, name):
        value = v
    pose[name] = value
  return pose


assert KNEES_BENT_KEYFRAME.joint_pos is not None
STAND_POSE_29: dict[str, float] = resolve_pose(KNEES_BENT_KEYFRAME.joint_pos)
"""Standing rest pose, one value per policy joint."""
assert CRAWL_INIT_STATE_29.joint_pos is not None
CRAWL_POSE_29: dict[str, float] = resolve_pose(CRAWL_INIT_STATE_29.joint_pos)
"""The crawl task's initial pose: where the transition must end."""

STAND_HEIGHT = 0.76
"""Pelvis height standing (the keyframe's)."""
CRAWL_HEIGHT = float(CRAWL_INIT_STATE_29.pos[2])
"""Pelvis height in the crawl pose."""
CRAWL_ROT = tuple(float(v) for v in CRAWL_INIT_STATE_29.rot)
"""Base orientation (w, x, y, z) in the crawl pose: ~88 deg pitch, +X down."""

STAND_INIT_STATE_29 = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, STAND_HEIGHT),
  rot=(1.0, 0.0, 0.0, 0.0),
  joint_pos=dict(STAND_POSE_29),
  joint_vel={".*": 0.0},
)


def get_g1_walk_robot_cfg_29() -> EntityCfg:
  """Fresh config for the modified 29-DoF G1, standing (new instance each call)."""
  return EntityCfg(
    init_state=STAND_INIT_STATE_29,
    collisions=(CRAWL_SELF_COLLISION_29,),
    spec_fn=get_crawl_spec_29,
    articulation=CRAWL_ARTICULATION_29,
  )


# Kneeling puts the knee on the floor, so the shin capsules (whose upper end is
# the knee) get friction like the feet and the extension tips; every other
# robot geom stays frictionless against the robot itself.
_KNEEL_LIMB_GEOMS = r"^(left|right)_(foot[1-7]|forearm_tip|shin)_collision$"
KNEEL_COLLISION_29 = CollisionCfg(
  geom_names_expr=(r".*_collision.*",),
  contype=1,
  conaffinity=1,
  condim={_KNEEL_LIMB_GEOMS: 3, ".*_collision.*": 1},
  priority={_KNEEL_LIMB_GEOMS: 1, ".*": 0},
  friction={_KNEEL_LIMB_GEOMS: (0.6,)},
)


def get_g1_kneel_robot_cfg_29() -> EntityCfg:
  """As ``get_g1_walk_robot_cfg_29`` with frictional knee (shin) contacts."""
  return EntityCfg(
    init_state=STAND_INIT_STATE_29,
    collisions=(KNEEL_COLLISION_29,),
    spec_fn=get_crawl_spec_29,
    articulation=CRAWL_ARTICULATION_29,
  )
