"""The 23-DoF "simple forearm" G1 used by the original Isaac Lab crawl task.

The source project trained on a 23-DoF G1 USD whose forearms are a single rigid
link ending in a small ball hand. Its sim2sim MuJoCo export of that robot is
checked in here as ``assets/g1_23dof_crawl.xml`` together with its meshes. MJLab's
bundled 29-DoF G1 differs in ways that matter for a hands-and-feet gait: a longer
and heavier three-joint wrist whose pitch/yaw motors (5 Nm) back-drive under body
load, compliant waist roll/pitch, extra wrist/elbow collision capsules, and
self-collisions. Using the source model removes those differences.

The export has 21 joints (no wrist roll). The Isaac Lab USD, the rc4 animation,
and the policy interface all have 23, so the two wrist-roll hinges are added back
on the forearm bodies. They do not move the hand contact point.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

G1_CRAWL_XML: Path = Path(__file__).resolve().parent / "assets" / "g1_23dof_crawl.xml"
assert G1_CRAWL_XML.exists(), G1_CRAWL_XML

# Policy layout shared by the Isaac Lab USD, the rc4 animation and the mirror
# function: legs (12), waist yaw (1), arms (10). This is also the MuJoCo joint
# order of the model below.
CRAWL_JOINTS = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
# The forearm body carries the ball hand (the source robot's ``*_wrist_link``).
HANDS = ("left_forearm", "right_forearm")
LEFT_LIMBS = ("left_ankle_roll_link", "left_forearm")
RIGHT_LIMBS = ("right_ankle_roll_link", "right_forearm")

_WRIST_ROLL_RANGE = [-1.97222, 1.97222]
_WRIST_ROLL_ARMATURE = 0.00360972  # Same value the export uses for the arm joints.


def get_crawl_spec() -> mujoco.MjSpec:
  """Load the source robot and strip the scene-level elements MJLab provides."""
  spec = mujoco.MjSpec.from_file(str(G1_CRAWL_XML))
  for geom in list(spec.geoms):
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
      spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  for camera in list(spec.cameras):
    spec.delete(camera)
  # MJLab adds its own position actuators and ``init_state`` keyframe.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  for key in list(spec.keys):
    spec.delete(key)
  # MuJoCo caps box-vs-heightfield contacts at 50 per pair and drops the rest,
  # which the export's torso box overflows on rough tiles. Use a capsule of the
  # same extent (same name, so contact penalties are unchanged).
  torso = spec.geom("torso_collision_main")
  torso.type = mujoco.mjtGeom.mjGEOM_CAPSULE
  torso.size[0] = 0.09  # Radius; the box was 0.075 x 0.108 half-extents.
  torso.size[1] = 0.165  # Half-length along the box's Z (0.178 minus radius).
  # The export compiles with ``angle="radian"``, so the range is in radians.
  for side in ("left", "right"):
    spec.body(f"{side}_forearm").add_joint(
      name=f"{side}_wrist_roll_joint",
      type=mujoco.mjtJoint.mjJNT_HINGE,
      axis=[1.0, 0.0, 0.0],
      range=_WRIST_ROLL_RANGE,
      armature=_WRIST_ROLL_ARMATURE,
    )
  return spec


# Crawl pose from the source ``G1_CFG.init_state``. Making it the entity default
# matters: ``JointPositionActionCfg(use_default_offset=True)`` centres the action
# on it, ``joint_pos_rel`` observes relative to it and ``joint_deviation_l1``
# pulls towards it, exactly as in the Isaac Lab task.
CRAWL_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.22),
  rot=(0.721249, -0.000001, 0.692676, -0.000012),  # ~1.53 rad pitch: base +X down.
  joint_pos={
    ".*_hip_pitch_joint": -1.6796101123595506,
    "left_hip_roll_joint": 2.4180011235955052,
    "right_hip_roll_joint": -2.4180011235955052,
    "left_hip_yaw_joint": 1.2083865168539325,
    "right_hip_yaw_joint": -1.2083865168539325,
    ".*_knee_joint": 2.1130298764044944,
    ".*_ankle_pitch_joint": 0.194143033707865,
    ".*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    ".*_shoulder_pitch_joint": 1.4578526315789473,
    "left_shoulder_roll_joint": 1.5778684210526317,
    "right_shoulder_roll_joint": -1.5778684210526317,
    "left_shoulder_yaw_joint": 1.4238245614035088,
    "right_shoulder_yaw_joint": -1.4238245614035088,
    ".*_elbow_joint": -0.3124709677419355,
    ".*_wrist_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

# The source articulation has ``enabled_self_collisions=False`` and every PhysX
# contact is frictional, so: robot-terrain only, condim 3 everywhere. Priority 1
# makes the (randomised) robot friction win over the terrain's, which mirrors
# the source's "multiply" combine against a unit-friction terrain.
CRAWL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision.*",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

# Same motors and gains as the source: arms 5020, hip pitch/yaw + waist 7520-14,
# hip roll + knee 7520-22, ankles 2x5020. No waist roll/pitch or wrist pitch/yaw.
CRAWL_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_ANKLE,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_g1_crawl_robot_cfg() -> EntityCfg:
  """Fresh config for the 23-DoF crawl G1 (new instance each call)."""
  return EntityCfg(
    init_state=CRAWL_INIT_STATE,
    collisions=(CRAWL_COLLISION,),
    spec_fn=get_crawl_spec,
    articulation=CRAWL_ARTICULATION,
  )
