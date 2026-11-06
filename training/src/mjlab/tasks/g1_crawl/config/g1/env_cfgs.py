"""MJLab configuration for the active flat-ground Isaac G1 crawl task."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.dr import (
  body_com_offset,
  body_mass,
  effort_limits,
  geom_friction,
  joint_armature,
  joint_friction,
  pd_gains,
)
from mjlab.envs.mdp.observations import (
  base_ang_vel,
  base_lin_vel,
  generated_commands,
  joint_pos_rel,
  joint_vel_rel,
  last_action,
  projected_gravity,
)
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  BuiltinSensorCfg,
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.g1_crawl import mdp
from mjlab.tasks.g1_crawl.mdp import metrics
from mjlab.tasks.g1_crawl.robot import (
  CRAWL_JOINTS,
  FEET,
  HANDS,
  LEFT_LIMBS,
  RIGHT_LIMBS,
  get_g1_crawl_robot_cfg,
)
from mjlab.tasks.g1_crawl.robot29 import (
  BRACED_LOADS_29,
  BRACED_TORQUE_LIMIT,
  CONTACT_BODIES_29,
  CRAWL_ANIMATION_29,
  CRAWL_INIT_STATE_29,
  CRAWL_JOINTS_29,
  CRAWL_SELF_COLLISION_29_TORSION,
  FEET_29,
  HANDS_29,
  LEFT_LIMBS_29,
  LIMB_SITES_29,
  MOTORS_29,
  RIGHT_LIMBS_29,
  deploy_target_clip,
  get_g1_crawl_robot_cfg_29,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import (
  box_random_grid,
  flat,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  perlin_noise,
  pyramid_stairs,
  pyramid_stairs_inv,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg

# Commanded limb swing height (m): global range and per-tile ranges (cmd7 tasks).
STEP_HEIGHT_RANGE = (0.02, 0.15)
STEP_HEIGHT_BY_TERRAIN = {
  "flat": (0.02, 0.06),
  "box_grid": (0.06, 0.12),
  "pyramid_stairs": (0.08, 0.15),
  "pyramid_stairs_inv": (0.08, 0.15),
}
# Second-generation command distribution (29-DoF, 7 commands). Each resample
# picks a mode: 10% stand, 25% straight, 10% sideways, 10% spin in place, 45%
# mixed. Straight and sideways hold their starting heading (turn = -1.0 x
# heading error, at most 0.5 rad/s), as a joystick "go straight" will on the
# robot. Lateral (crab) speed up to 0.5 m/s either way.
_GEN2_COMMAND = {
  "lin_vel_y": (-0.5, 0.5),
  "rel_standing_envs": 0.10,
  "rel_straight_envs": 0.25,
  "rel_sideways_envs": 0.10,
  "rel_spin_envs": 0.10,
  "heading_stiffness": 1.0,
  "heading_max_turn": 0.5,
}

# Global iteration after which the swing-height terms are weighted 2x (cmd7).
STEP_HEIGHT_BOOST_ITERATION = 20000

# Bodies the crawl contact sensor treats as "limbs" on the 23-DoF robot. The
# forearm body carries that model's ball hand.
CONTACT_BODIES = r".*_(ankle_roll_link|forearm)$"

# 29-DoF robot: torque sensors across the braced wrist joints (scene names carry
# the entity prefix) and the joint axis each one is read along.
BRACED_TORQUE_SENSORS = tuple(
  f"robot/{joint.removesuffix('_joint')}_torque" for joint in BRACED_LOADS_29
)
BRACED_AXES = tuple(axis for _, axis in BRACED_LOADS_29.values())
# Torque safety (29-DoF). Penalties start at this fraction of each limit: the
# trained 23-DoF gait keeps its busiest motor (the elbow) at 78% of peak at the
# 99th percentile, so 0.8 costs a good gait nothing.
TORQUE_SOFT_FRACTION = 0.8

__all__ = [
  "CRAWL_JOINTS",
  "FEET",
  "HANDS",
  "LEFT_LIMBS",
  "RIGHT_LIMBS",
  "unitree_g1_crawl_env_cfg",
  "unitree_g1_crawl_cool_env_cfg",
  "add_thermal_finetune",
  "unitree_g1_crawl_rough_env_cfg",
  "CRAWL_TERRAINS_CFG",
  "CRAWL_ZONES_CFG",
  "FLAT_ITERATIONS",
  "SPREAD_AFTER_ITERATIONS",
]


def unitree_g1_crawl_env_cfg(
  play: bool = False, cmd7: bool = False, dof29: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build training or play config for the G1 crawl task.

  ``cmd7`` appends a commanded limb swing height plus three spare slots to the
  command: ``[fwd, lat, turn, step_height, spare, spare, spare]``.

  ``dof29`` swaps the source project's 23-DoF "simple forearm" G1 for the 29-DoF
  robot on the bench, modified for crawling (hands removed, wrist pitch/yaw
  braced, printed forearm extensions; 25 active joints). See ``robot29.py``.
  """
  joints = CRAWL_JOINTS_29 if dof29 else CRAWL_JOINTS
  feet = FEET_29 if dof29 else FEET
  hands = HANDS_29 if dof29 else HANDS
  left_limbs = LEFT_LIMBS_29 if dof29 else LEFT_LIMBS
  right_limbs = RIGHT_LIMBS_29 if dof29 else RIGHT_LIMBS

  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {
    "robot": get_g1_crawl_robot_cfg_29() if dof29 else get_g1_crawl_robot_cfg()
  }
  cfg.scene.num_envs = 4096
  cfg.viewer.body_name = "torso_link"

  # The checked-in active Isaac task is flat: USE_ROUGH_TERRAIN=False.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = (
    ContactSensorCfg(
      name="crawl_contacts",
      primary=ContactMatch(
        mode="body",
        pattern=CONTACT_BODIES_29 if dof29 else CONTACT_BODIES,
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=3,
      track_air_time=True,
    ),
    ContactSensorCfg(
      name="body_ground_contacts",
      primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=3,
    ),
  )

  if dof29:
    # Load across each braced wrist joint (see ``robot29.BRACED_LOADS_29``).
    cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
      BuiltinSensorCfg(
        name=f"{joint.removesuffix('_joint')}_torque",
        sensor_type="torque",
        obj=ObjRef(type="site", name=site, entity="robot"),
      )
      for joint, (site, _) in BRACED_LOADS_29.items()
    )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = joints
  # Source: ``JointPositionActionCfg(scale=0.5, use_default_offset=True)`` on all
  # joints, with the entity default being the crawl pose (see ``robot.py``).
  action.scale = 0.5
  action.use_default_offset = True

  cfg.commands = {
    "base_velocity": mdp.CrawlVelocityCommandCfg(
      resampling_time_range=(4.0, 8.0),
      entity_name="robot",
      # Stage A: more idle envs (rest stance) and a wider, variable turn range.
      rel_standing_envs=0.15,
      lin_vel_z=(-1.0, 2.5),
      lin_vel_y=(0.0, 0.0),
      ang_vel_x=(-3.0, 3.0),
      step_height=STEP_HEIGHT_RANGE if cmd7 else None,
      num_spare=3 if cmd7 else 0,
    )
  }
  # The 29-DoF 7-command tasks are the second-generation setup: a gyro in the
  # policy's inputs, a lateral (crab) command, explicit straight / sideways /
  # spin commands with heading hold, and step height. See ``_GEN2_COMMAND``.
  gen2 = dof29 and cmd7
  if gen2:
    cmd_cfg = cfg.commands["base_velocity"]
    assert isinstance(cmd_cfg, mdp.CrawlVelocityCommandCfg)
    for key, value in _GEN2_COMMAND.items():
      setattr(cmd_cfg, key, value)

  actor_terms = {
    "projected_gravity": ObservationTermCfg(
      func=projected_gravity,
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "velocity_commands": ObservationTermCfg(
      func=generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=last_action),
  }
  critic_terms = {
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=projected_gravity),
    "velocity_commands": ObservationTermCfg(
      func=generated_commands, params={"command_name": "base_velocity"}
    ),
    "joint_pos": ObservationTermCfg(
      func=joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
    ),
    "joint_vel": ObservationTermCfg(
      func=joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
    ),
    "actions": ObservationTermCfg(func=last_action),
  }
  if gen2:
    # Pelvis gyro. The G1's pelvis IMU frame is aligned with the pelvis, so the
    # real robot's gyroscope reading is this value directly.
    gyro = ObservationTermCfg(
      func=base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)
    )
    actor_terms = {
      "projected_gravity": actor_terms["projected_gravity"],
      "base_ang_vel": gyro,
      **{k: v for k, v in actor_terms.items() if k != "projected_gravity"},
    }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  cfg.events = {
    "randomize_contact_friction": EventTermCfg(
      func=geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=r".*_collision.*"),
        "ranges": (0.4, 1.0),
        "operation": "abs",
      },
    ),
    "randomize_body_mass": EventTermCfg(
      func=body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=r".*"),
        "ranges": (0.9, 1.1),
        "operation": "scale",
      },
    ),
    "add_torso_mass": EventTermCfg(
      func=body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "ranges": (-1.0, 1.0),
        "operation": "add",
      },
    ),
    "randomize_joint_friction": EventTermCfg(
      func=joint_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=r".*"),
        "ranges": (0.5, 2.0),
        "operation": "scale",
      },
    ),
    "randomize_joint_armature": EventTermCfg(
      func=joint_armature,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=r".*"),
        "ranges": (1.0, 1.05),
        "operation": "scale",
      },
    ),
    "reset_from_crawl_motion": EventTermCfg(
      func=mdp.reset_from_animation_frame,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=joints),
        # The rc4 capture is a 23-DoF robot; the 29-DoF task resets from the
        # version retargeted onto its braced forearm and extension.
        "animation_path": str(CRAWL_ANIMATION_29) if dof29 else None,
        "pose_range": {
          "x": (-0.05, 0.05),
          "y": (-0.05, 0.05),
          "z": (-0.05, 0.05),
          "roll": (-0.10, 0.10),
          "pitch": (-0.10, 0.10),
          "yaw": (-0.10, 0.10),
        },
        "velocity_range": {
          "x": (-0.20, 0.20),
          "y": (-0.20, 0.20),
          "z": (-0.20, 0.20),
          "roll": (-0.30, 0.30),
          "pitch": (-0.30, 0.30),
          "yaw": (-0.30, 0.30),
        },
        "position_range": (0.9, 1.1),
        "joint_velocity_range": (0.0, 0.0),
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(5.0, 10.0),  # Stage A: pushes on (source rough cfg).
      params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    ),
  }

  cfg.rewards = {
    # Tracking: wider kernels (partial credit at partial speed) plus a linear
    # progress term so a stopped robot still has a gradient toward moving.
    "track_lin_vel_yz_exp": RewardTermCfg(
      func=mdp.track_lin_vel_yz_exp,
      weight=3.0,
      params={"command_name": "base_velocity", "std": 0.5},
    ),
    "track_ang_vel_x_exp": RewardTermCfg(
      func=mdp.track_ang_vel_x_exp,
      weight=3.0,
      params={"command_name": "base_velocity", "std": 0.5},
    ),
    "forward_progress": RewardTermCfg(
      func=mdp.forward_progress,
      weight=1.0,
      params={"command_name": "base_velocity"},
    ),
    "crawl_orientation": RewardTermCfg(
      func=mdp.align_projected_gravity_plus_x_l2, weight=0.2
    ),
    "base_height_l2": RewardTermCfg(
      func=mdp.base_height_l2,
      weight=-0.1,
      params={
        "target_height": 0.22,
        "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
      },
    ),
    "joint_deviation_l1": RewardTermCfg(
      func=mdp.joint_deviation_l1,
      weight=-0.01,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-1e-2),
    "joint_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-1e-7,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
    ),
    # Stage A grace/stability terms.
    "base_wobble": RewardTermCfg(func=mdp.crawl_ang_vel_yz_l2, weight=-0.02),
    "base_bobbing": RewardTermCfg(func=mdp.crawl_lin_vel_x_l2, weight=-0.5),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still_joint_vel,
      weight=-0.01,
      params={
        "command_name": "base_velocity",
        "asset_cfg": SceneEntityCfg("robot", joint_names=joints),
        "threshold": 0.1,
      },
    ),
    "joint_torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-1e-4,
      params={"asset_cfg": SceneEntityCfg("robot", actuator_names=list(joints))},
    ),
    "torque_limits": RewardTermCfg(
      func=mdp.applied_torque_limits,
      weight=-5.0,
      params={"joint_names": joints},
    ),
    "undesired_body_contacts": RewardTermCfg(
      func=mdp.undesired_ground_contacts,
      weight=-5.0,
      params={
        "sensor_name": "body_ground_contacts",
        "excluded_body_names": feet + hands,
        "threshold": 1.0,
      },
    ),
    "feet_slide": RewardTermCfg(
      func=mdp.feet_slide,
      weight=-0.2,
      params={"sensor_name": "crawl_contacts", "body_names": feet + hands},
    ),
    "both_feet_air": RewardTermCfg(
      func=mdp.both_limbs_air,
      weight=-0.5,
      params={"sensor_name": "crawl_contacts", "body_names": feet},
    ),
    "both_hands_air": RewardTermCfg(
      func=mdp.both_limbs_air,
      weight=-0.5,
      params={"sensor_name": "crawl_contacts", "body_names": hands},
    ),
    "both_left_limbs_air": RewardTermCfg(
      func=mdp.both_limbs_air,
      weight=-0.1,
      params={"sensor_name": "crawl_contacts", "body_names": left_limbs},
    ),
    "both_right_limbs_air": RewardTermCfg(
      func=mdp.both_limbs_air,
      weight=-0.1,
      params={"sensor_name": "crawl_contacts", "body_names": right_limbs},
    ),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True)
  }
  # The source's active config assigns terrain curriculum only on rough terrain;
  # flat-task curriculum is absent. Remove the generic velocity curricula.
  cfg.curriculum = {}
  cfg.metrics = {}

  if dof29:
    _add_torque_safety_29(cfg)
    _add_elegance_29(cfg, feet=feet, hands=hands, step_height_cmd=cmd7)
    _add_robust_training_29(cfg, feet=feet, hands=hands, play=play)
  cfg.decimation = 4
  cfg.sim.mujoco.timestep = 0.005
  # Contacts per world (pooled across envs). Hands, 14 foot capsules and the
  # torso can exceed the velocity task's default of 35 on a single env. The
  # 29-DoF robot has more collision geoms and self-collisions enabled.
  cfg.sim.nconmax = 120 if dof29 else 70
  cfg.episode_length_s = 20.0
  # Isaac Lab's RewardManager multiplies every term by ``step_dt``; keep parity so
  # returns, value targets and logged ``Episode_Reward/*`` match the source.
  cfg.scale_rewards_by_dt = True

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


def _add_torque_safety_29(cfg: ManagerBasedRlEnvCfg) -> None:
  """Torque limits that actually bind, for the modified 29-DoF robot.

  ``torque_limits`` reads ``qfrc_actuator``, which MuJoCo has already clamped to
  each motor's force range, so it can never fire; it is replaced here. Measured
  by driving this robot with the trained 23-DoF policy (unadapted, so on the
  high side): the elbows were asked for more than their 25 Nm peak on 7% of
  steps, and the moment across the braced wrist joints had a median of 1-2 Nm, a
  95th percentile of ~11 Nm and peaks of 36 Nm.

  Both start at weight 0: a crawl learned under torque limits from the first
  iteration tends to be timid and awkward. ``crawl_robustness`` fades them in
  once the robot already crawls well, to -5 (motors, by level 0.6) and -1
  (braced wrists, by level 0.8). At those weights the unadapted gait above pays
  about -22 and -27 per episode against +87 of tracking, so moving always stays
  the better option (twice that and the penalties matched tracking -- the
  standing-still failure of section 5).
  """
  cfg.rewards.pop("torque_limits")
  # Every motor: PD torque demand above 80% of its peak, in units of the peak.
  cfg.rewards["motor_torque_margin"] = RewardTermCfg(
    func=mdp.motor_torque_margin,
    weight=0.0,
    params={"motors": MOTORS_29, "soft_fraction": TORQUE_SOFT_FRACTION},
  )
  # Braced wrist pitch/yaw: moment about the joint axis above 80% of the limit.
  cfg.rewards["braced_wrist_torque"] = RewardTermCfg(
    func=mdp.braced_joint_torque,
    weight=0.0,
    params={
      "sensor_names": BRACED_TORQUE_SENSORS,
      "axes": BRACED_AXES,
      "limit": BRACED_TORQUE_LIMIT,
      "soft_fraction": TORQUE_SOFT_FRACTION,
    },
  )


def _posture_tolerances_29() -> tuple[dict, dict, dict]:
  """Per-joint posture tolerance (rad) when standing / crawling / fast.

  Read off the reference crawl: each joint's RMS deviation from the crawl pose
  over the 80 retargeted frames, the larger of each left/right pair so the
  reward stays mirror-symmetric (the animation itself is not). Crawling allows
  1.5x that, fast motion 2.25x, standing 0.5x. Joints the reference never moves
  get a floor: waist roll/pitch stay free for the policy to use, just not to
  flop around.
  """
  data = json.loads(CRAWL_ANIMATION_29.read_text())
  frames = np.asarray(data["frames"])
  qposadr = {j["name"]: j["qposadr"] for j in data["metadata"]["joints"]}
  default = CRAWL_INIT_STATE_29.joint_pos
  assert default is not None
  rms = {
    n: float(np.sqrt(np.mean((frames[:, qposadr[n]] - default[n]) ** 2)))
    for n in CRAWL_JOINTS_29
  }
  floor = {"waist_roll_joint": 0.25, "waist_pitch_joint": 0.25}
  walking: dict[str, float] = {}
  for name in CRAWL_JOINTS_29:
    twin = (
      name.replace("left_", "right_")
      if "left_" in name
      else name.replace("right_", "left_")
    )
    base = max(rms[name], rms.get(twin, 0.0))
    walking[name] = max(1.5 * base, floor.get(name, 0.15))
  standing = {n: max(0.5 * v, 0.08) for n, v in walking.items()}
  running = {n: 1.5 * v for n, v in walking.items()}
  return standing, walking, running


def _add_elegance_29(
  cfg: ManagerBasedRlEnvCfg,
  feet: tuple[str, ...],
  hands: tuple[str, ...],
  step_height_cmd: bool = False,
) -> None:
  """Rewards for a clean, natural crawl rather than any crawl that tracks.

  On from the first iteration, because the gait's style is set while it is
  first learned. The reference is the rc4 capture: a diagonal-sequence walk
  (each foot followed by the opposite hand), each limb down ~57% of a 0.4 s
  stride. Weights keep the whole group below about a quarter of what tracking
  pays, so style shapes the gait without competing with moving at all.
  """
  limbs = (hands[0], hands[1], feet[0], feet[1])
  moving = {"command_name": "base_velocity", "command_threshold": 0.1}
  if not any(s.name == "limb_height_scan" for s in cfg.scene.sensors or ()):
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
      _limb_height_sensor(LIMB_SITES_29),
    )
  standing, walking, running = _posture_tolerances_29()
  cfg.rewards.update(
    {
      # Diagonal limbs move together: trot-like, the classic smooth crawl.
      "gait_diagonal": RewardTermCfg(
        func=mdp.diagonal_gait,
        weight=0.3,
        params={"sensor_name": "crawl_contacts", "limb_names": limbs, **moving},
      ),
      # Real steps: each swing lasts 0.10-0.40 s (reference ~0.17 s), no
      # shuffling or tapping.
      "gait_air_time": RewardTermCfg(
        func=velocity_mdp.feet_air_time,
        weight=0.25,
        params={
          "sensor_name": "crawl_contacts",
          "threshold_min": 0.10,
          "threshold_max": 0.40,
          **moving,
        },
      ),
      # Lift hands and feet ~5 cm while they move instead of dragging them.
      # (The terrain tasks replace this with their own clearance term.)
      # Lift hands and feet while they move instead of dragging them: to 5 cm,
      # or to the commanded step height on 7-command tasks (plus a bonus for
      # being near it). The terrain tasks replace this with their own terms.
      "limb_clearance": RewardTermCfg(
        func=mdp.limb_clearance_cmd if step_height_cmd else velocity_mdp.feet_clearance,
        weight=-0.5,
        params={
          "height_sensor_name": "limb_height_scan",
          "asset_cfg": SceneEntityCfg("robot", site_names=LIMB_SITES_29),
          **(
            {"command_name": "base_velocity"}
            if step_height_cmd
            else {"target_height": 0.05, **moving}
          ),
        },
      ),
      # Place limbs, do not stomp: impact force at each touchdown.
      "soft_landing": RewardTermCfg(
        func=velocity_mdp.soft_landing,
        weight=-0.002,
        params={"sensor_name": "crawl_contacts", **moving},
      ),
      **(
        {
          "limb_lift_bonus": RewardTermCfg(
            func=mdp.limb_lift_bonus,
            weight=0.5,
            params={
              "height_sensor_name": "limb_height_scan",
              "command_name": "base_velocity",
              "asset_cfg": SceneEntityCfg("robot", site_names=LIMB_SITES_29),
              "std": 0.03,
            },
          )
        }
        if step_height_cmd
        else {}
      ),
      # Smooth motion: second difference of the actions.
      "action_smoothness": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.005),
      # Stay within the reference crawl's joint envelope, looser when faster.
      "posture": RewardTermCfg(
        func=velocity_mdp.variable_posture,
        weight=0.5,
        params={
          "asset_cfg": SceneEntityCfg("robot", joint_names=CRAWL_JOINTS_29),
          "command_name": "base_velocity",
          "std_standing": standing,
          "std_walking": walking,
          "std_running": running,
          "walking_threshold": 0.1,
          "running_threshold": 1.5,
        },
      ),
    }
  )


# Robustness curriculum ramps (level 0 -> level 1); see ``mdp/robustness.py``.
ROBUSTNESS_RAMP_29 = mdp.RobustnessRamp()


def _add_robust_training_29(
  cfg: ManagerBasedRlEnvCfg,
  feet: tuple[str, ...],
  hands: tuple[str, ...],
  play: bool,
) -> None:
  """Per-episode randomization, a robustness curriculum, metrics (29-DoF).

  The source randomizes physics once at startup with fixed ranges. Here every
  disturbance is re-drawn at each episode reset from ranges the
  ``crawl_robustness`` curriculum owns: mild while the crawl is learned, widened
  step by step once the robot tracks its command and stops rolling over, and
  narrowed again if it stops coping. Roll-overs end the episode (as on the
  terrain tasks) so a pushed-over robot does not waste the rest of it.
  """
  ramp = ROBUSTNESS_RAMP_29
  # Level-0 values; the curriculum overwrites them in place as the level moves.
  for name in (
    "randomize_contact_friction",
    "randomize_body_mass",
    "add_torso_mass",
    "randomize_joint_friction",
    "randomize_joint_armature",
  ):
    cfg.events.pop(name)
  robot_all = SceneEntityCfg("robot")
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  dr = {
    "dr_friction": (
      geom_friction,
      {
        "asset_cfg": SceneEntityCfg("robot", geom_names=r".*_collision.*"),
        "ranges": (ramp.friction_low[0], ramp.friction_high[0]),
        "operation": "abs",
      },
    ),
    "dr_body_mass": (
      body_mass,
      {
        "asset_cfg": SceneEntityCfg("robot", body_names=r".*"),
        "ranges": (1.0 - ramp.mass_scale[0], 1.0 + ramp.mass_scale[0]),
        "operation": "scale",
      },
    ),
    "dr_torso_payload": (
      body_mass,
      {
        "asset_cfg": torso,
        "ranges": (ramp.payload_low_kg[0], ramp.payload_high_kg[0]),
        "operation": "add",
      },
    ),
    "dr_torso_com": (
      body_com_offset,
      {"asset_cfg": torso, "ranges": (0.0, 0.0), "axes": [0, 1, 2]},
    ),
    "dr_joint_friction": (
      joint_friction,
      {
        "asset_cfg": SceneEntityCfg("robot", joint_names=r".*"),
        "ranges": (0.0, ramp.joint_friction_nm[0]),
        "operation": "abs",
      },
    ),
    "dr_armature": (
      joint_armature,
      {
        "asset_cfg": SceneEntityCfg("robot", joint_names=r".*"),
        "ranges": (1.0 - ramp.armature_scale[0], 1.0 + ramp.armature_scale[0]),
        "operation": "scale",
      },
    ),
    "dr_motor_gains": (
      pd_gains,
      {"asset_cfg": robot_all, "kp_range": (1.0, 1.0), "kd_range": (1.0, 1.0)},
    ),
    "dr_motor_strength": (
      effort_limits,
      {"asset_cfg": robot_all, "effort_limit_range": (1.0, 1.0)},
    ),
  }
  for name, (func, params) in dr.items():
    cfg.events[name] = EventTermCfg(func=func, mode="reset", params=params)
  cfg.events["push_robot"] = EventTermCfg(
    func=envs_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(4.0, 8.0),
    params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
  )

  cfg.terminations["flipped"] = TerminationTermCfg(
    func=mdp.crawl_flipped, params={"limit": 0.0}
  )
  cfg.rewards["termination_penalty"] = RewardTermCfg(
    func=envs_mdp.is_terminated, weight=-100.0
  )

  _add_metrics_29(cfg, feet=feet, hands=hands)

  if play:
    # Nominal robot for viewing; delay keeps its actuator default (0-20 ms).
    for name in dr:
      cfg.events.pop(name)
    return
  cfg.curriculum["robustness"] = CurriculumTermCfg(
    func=mdp.crawl_robustness,
    params={
      "command_name": "base_velocity",
      "ramp": ramp,
      "events": {
        "push": "push_robot",
        "friction": "dr_friction",
        "mass": "dr_body_mass",
        "payload": "dr_torso_payload",
        "com": "dr_torso_com",
        "joint_friction": "dr_joint_friction",
        "armature": "dr_armature",
        "pd_gains": "dr_motor_gains",
        "effort": "dr_motor_strength",
      },
      "noisy_obs": tuple(
        name
        for name in ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel")
        if name in cfg.observations["actor"].terms
      ),
      "motor_reward": "motor_torque_margin",
      "braced_reward": "braced_wrist_torque",
    },
  )


def _add_metrics_29(
  cfg: ManagerBasedRlEnvCfg, feet: tuple[str, ...], hands: tuple[str, ...]
) -> None:
  """TensorBoard ``Episode_Metrics/<group>/<name>``: means unless marked peak."""
  legs = {k: v for k, v in MOTORS_29.items() if re.search("hip|knee|ankle", k)}
  waist = {k: v for k, v in MOTORS_29.items() if k.startswith("waist")}
  arms = {k: v for k, v in MOTORS_29.items() if k not in legs and k not in waist}
  wrist_roll = {k: v for k, v in MOTORS_29.items() if "wrist_roll" in k}
  cmd = {"command_name": "base_velocity"}
  limbs = {"sensor_name": "crawl_contacts"}
  bodies = {"sensor_name": "body_ground_contacts"}
  braced = {"sensor_names": BRACED_TORQUE_SENSORS, "axes": BRACED_AXES}
  m = metrics
  terms: dict[str, tuple] = {
    # Is it doing what it is told?
    "track/fwd_speed_mps": (m.fwd_speed, {}, "mean"),
    "track/fwd_speed_error_mps": (m.fwd_speed_error, cmd, "mean"),
    "track/lateral_speed_error_mps": (m.lateral_speed_error, cmd, "mean"),
    "track/turn_rate_error_radps": (m.turn_rate_error, cmd, "mean"),
    # Is it crawling on four limbs?
    "gait/hands_on_ground_frac": (
      m.limb_contact_frac,
      {**limbs, "body_names": hands},
      "mean",
    ),
    "gait/feet_on_ground_frac": (
      m.limb_contact_frac,
      {**limbs, "body_names": feet},
      "mean",
    ),
    "gait/hand_load_share": (
      m.hand_load_share,
      {**limbs, "hand_names": hands, "foot_names": feet},
      "mean",
    ),
    "gait/diagonal_sync": (
      m.diagonal_sync,
      {**limbs, "limb_names": (hands[0], hands[1], feet[0], feet[1])},
      "mean",
    ),
    "gait/swing_height_m": (
      m.swing_height,
      {"height_sensor_name": "limb_height_scan", "site_names": LIMB_SITES_29},
      "mean",
    ),
    "gait/body_height_m": (m.body_height, {}, "mean"),
    "gait/prone_alignment": (m.prone_alignment, {}, "mean"),
    "gait/torso_on_ground_frac": (
      m.bodies_touching,
      {**bodies, "body_names": ("torso_link", "pelvis")},
      "mean",
    ),
    "gait/other_bodies_touching": (
      m.other_bodies_touching,
      {**bodies, "excluded_body_names": feet + hands},
      "mean",
    ),
    # Are the motors and the braced wrists inside their limits?
    "safety/braced_wrist_peak_nm": (mdp.braced_joint_moment_max, braced, "max"),
    "safety/braced_wrist_over_limit_frac": (
      m.braced_over_limit,
      {**braced, "limit": BRACED_TORQUE_LIMIT},
      "mean",
    ),
    "safety/motor_peak_ratio": (m.motor_peak_ratio, {"motors": MOTORS_29}, "max"),
    "safety/motor_saturated_frac": (
      m.motor_any_saturated,
      {"motors": MOTORS_29},
      "mean",
    ),
    "safety/leg_motor_peak_ratio": (m.motor_peak_ratio, {"motors": legs}, "max"),
    "safety/waist_motor_peak_ratio": (m.motor_peak_ratio, {"motors": waist}, "max"),
    "safety/arm_motor_peak_ratio": (m.motor_peak_ratio, {"motors": arms}, "max"),
    "safety/wrist_roll_peak_nm": (m.motor_peak_nm, {"motors": wrist_roll}, "max"),
    "safety/joint_near_limit_frac": (
      m.joint_near_limit,
      {"joint_names": tuple(MOTORS_29)},
      "mean",
    ),
    # How hard is it working?
    "effort/mech_power_w": (m.mechanical_power, {"motors": MOTORS_29}, "mean"),
    "effort/action_jerk": (envs_mdp.mean_action_acc, {}, "mean"),
  }
  command = cfg.commands["base_velocity"]
  assert isinstance(command, mdp.CrawlVelocityCommandCfg)
  if command.step_height is not None:
    terms["gait/step_height_cmd_m"] = (
      m.command_value,
      {**cmd, "index": 3},
      "mean",
    )
  for name, (func, params, reduce) in terms.items():
    cfg.metrics[name] = MetricsTermCfg(func=func, params=params, reduce=reduce)


##
# Stage B: stairs and uneven terrain with a gated curriculum.
##

# Curriculum gates in learning iterations. They are GLOBAL: MJLab restores the
# env step counter when resuming, so a run resumed from iteration 2999 with
# FLAT_ITERATIONS=7000 spends 4000 iterations on flat ground first.
FLAT_ITERATIONS = 7000
# After this global iteration the curriculum stops and every env is re-spawned
# on a random level and tile type at each reset (phase C).
SPREAD_AFTER_ITERATIONS = 15000

# Box-only tiles. MuJoCo Warp gives a single contact per capsule on heightfields
# and the crawl's long, flat-lying foot capsules slide on them (measured: the
# flat-ground policy flipped in 40% of envs on a *flat* heightfield versus 0% on a
# flat box). Nested boxes at zero height are degenerate too, so the difficulty
# floor keeps every non-flat tile non-degenerate (steps >= 2.4 cm). The warm-up
# phase forces all envs onto the flat tile, which is physically a plane.
CRAWL_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  curriculum=True,
  difficulty_range=(0.2, 1.0),
  sub_terrains={
    "flat": flat(proportion=0.25),
    "box_grid": box_random_grid(
      proportion=0.25, grid_width=0.4, grid_height_range=(0.0, 0.06)
    ),
    "pyramid_stairs": pyramid_stairs(
      proportion=0.25, step_height_range=(0.0, 0.12), step_width=0.4
    ),
    "pyramid_stairs_inv": pyramid_stairs_inv(
      proportion=0.25, step_height_range=(0.0, 0.12), step_width=0.4
    ),
  },
  add_lights=True,
)

# Six zones laid out left to right (declaration order = column order). Each zone
# has 10 difficulty rows (front to back). "random_mix" has no tiles of its own:
# the zone curriculum places its envs on a random non-flat zone at the same row.
# Meadows are a heightfield (Perlin) with a height floor: near-flat heightfields
# make the crawl's foot capsules slide, so verify this tile with
# scripts/crawl_tile_test.py before relying on it.
CRAWL_ZONES_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=5,  # Difficulty 0.2, 0.4, 0.6, 0.8, 1.0 per zone.
  num_cols=20,
  curriculum=True,
  difficulty_range=(0.2, 1.0),
  sub_terrains={
    "flat": flat(proportion=0.12),
    # Perlin uses ``resolution`` (not horizontal_scale) for its cell size; 5 cm
    # cells overflow MuJoCo's 50-contact cap per capsule/heightfield pair.
    "meadows": perlin_noise(
      proportion=0.11,
      height_range=(0.08, 0.25),
      scale=6.0,
      octaves=3,
      horizontal_scale=0.2,
      resolution=0.2,
    ),
    "hill": hf_pyramid_slope(
      proportion=0.11, slope_range=(0.15, 0.35), horizontal_scale=0.2
    ),
    "valley": hf_pyramid_slope_inv(
      proportion=0.11, slope_range=(0.15, 0.35), horizontal_scale=0.2
    ),
    "stairs_up": pyramid_stairs(
      proportion=0.11, step_height_range=(0.0, 0.08), step_width=0.45
    ),
    "stairs_down": pyramid_stairs_inv(
      proportion=0.11, step_height_range=(0.0, 0.08), step_width=0.45
    ),
    "box_grid": box_random_grid(
      proportion=0.11, grid_width=0.4, grid_height_range=(0.0, 0.06)
    ),
    "big_stairs": pyramid_stairs_inv(
      proportion=0.11, step_height_range=(0.04, 0.12), step_width=0.4
    ),
    "random_mix": flat(proportion=0.11),
  },
  add_lights=True,
)
STEP_HEIGHT_BY_ZONE = {
  "flat": (0.02, 0.06),
  "meadows": (0.04, 0.10),
  "hill": (0.04, 0.10),
  "valley": (0.04, 0.10),
  "stairs_up": (0.06, 0.12),
  "stairs_down": (0.06, 0.12),
  "box_grid": (0.06, 0.12),
  "big_stairs": (0.08, 0.14),
  "random_mix": (0.06, 0.14),
}

LIMB_SITES = ("left_foot", "right_foot", "left_palm", "right_palm")


def _limb_height_sensor(limb_sites: tuple[str, ...]) -> TerrainHeightSensorCfg:
  """Height of each hand/foot site above the terrain (clearance rewards)."""
  return TerrainHeightSensorCfg(
    name="limb_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in limb_sites),
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
  )


def unitree_g1_crawl_rough_env_cfg(
  play: bool = False, cmd7: bool = False, dof29: bool = False
) -> ManagerBasedRlEnvCfg:
  """Stage A crawl task on generated terrain with a gated level curriculum."""
  cfg = unitree_g1_crawl_env_cfg(play=play, cmd7=cmd7, dof29=dof29)
  limb_sites = LIMB_SITES_29 if dof29 else LIMB_SITES

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = CRAWL_ZONES_CFG
  cfg.scene.terrain.max_init_terrain_level = 0
  # Stairs and heightfields produce many more contacts per robot. The 29-DoF
  # robot has more collision geoms and its self-collisions are on.
  cfg.sim.nconmax = 220 if dof29 else 140

  if not any(s.name == "limb_height_scan" for s in cfg.scene.sensors or ()):
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
      _limb_height_sensor(limb_sites),
    )

  if cmd7:
    # Swing height follows the command: terrain-correlated sampling, tracking
    # penalty plus a kernel bonus, both doubled after the boost iteration.
    cmd = cfg.commands["base_velocity"]
    assert isinstance(cmd, mdp.CrawlVelocityCommandCfg)
    cmd.step_height_by_terrain = STEP_HEIGHT_BY_ZONE
    cmd.terrain_prob = 0.8
    cfg.rewards["limb_clearance"] = RewardTermCfg(
      func=mdp.limb_clearance_cmd,
      weight=-0.5,
      params={
        "height_sensor_name": "limb_height_scan",
        "command_name": "base_velocity",
        "asset_cfg": SceneEntityCfg("robot", site_names=limb_sites),
        "boost_after_iterations": STEP_HEIGHT_BOOST_ITERATION,
        "boost": 2.0,
      },
    )
    cfg.rewards["limb_lift_bonus"] = RewardTermCfg(
      func=mdp.limb_lift_bonus,
      weight=0.5,
      params={
        "height_sensor_name": "limb_height_scan",
        "command_name": "base_velocity",
        "asset_cfg": SceneEntityCfg("robot", site_names=limb_sites),
        "std": 0.03,
        "boost_after_iterations": STEP_HEIGHT_BOOST_ITERATION,
        "boost": 2.0,
      },
    )
  else:
    # Lift hands and feet over edges while moving (fixed 8 cm target).
    cfg.rewards["limb_clearance"] = RewardTermCfg(
      func=velocity_mdp.feet_clearance,
      weight=-0.5,
      params={
        "target_height": 0.08,
        "height_sensor_name": "limb_height_scan",
        "command_name": "base_velocity",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=limb_sites),
      },
    )
  # Rolling over on a step wastes the rest of the episode: end it instead.
  cfg.terminations["flipped"] = TerminationTermCfg(
    func=mdp.crawl_flipped, params={"limit": 0.0}
  )
  cfg.rewards["termination_penalty"] = RewardTermCfg(
    func=envs_mdp.is_terminated, weight=-100.0
  )

  cfg.curriculum["terrain_zones"] = CurriculumTermCfg(
    func=mdp.terrain_zones_crawl,
    params={
      "command_name": "base_velocity",
      "flat_iterations": FLAT_ITERATIONS,
      "spread_after_iterations": SPREAD_AFTER_ITERATIONS,
      "steps_per_iteration": 24,
      "mix_zone_name": "random_mix",
    },
  )

  if play:
    cfg.curriculum = {}
    # Same six zones left to right, 4 rows, no curriculum: envs spawn anywhere.
    cfg.scene.terrain.terrain_generator = replace(
      CRAWL_ZONES_CFG,
      curriculum=True,
      num_rows=4,
      border_width=10.0,
      difficulty_range=(0.4, 1.0),
    )
    cfg.scene.terrain.max_init_terrain_level = None

  return cfg


##
# Thermal fine-tune (after the 2026-10-29 robot test): hold still and crawl
# inside the motors' continuous ratings, on any floor, without relearning.
##

# Each limb's material on top of the floor's friction: rubber soles, and the
# printed (PETG) forearm extension tips, which slide more on tiles.
LIMB_MATERIALS_29 = {
  r"^(left|right)_foot[1-7]_collision$": (0.85, 1.1),
  r"^(left|right)_forearm_tip_collision$": (0.55, 1.0),
}
# One floor per episode: glazed tile (low end) to carpet (high end).
FLOOR_FRICTION = (0.35, 1.2)
# Allowed RMS torque, as a fraction of each motor's continuous rating.
HEAT_BUDGET = 0.6
# Real knees reached their 2.88 rad stop at the bottom of every descent.
KNEE_MARGIN_RAD = 2.5
# Target weights of the new terms, faded in by ``reward_ramp``. Sized so the
# deployed crawl (gen2_flat_16k/49999) pays about 15 % of its return at the
# start (measured with scripts/thermal_eval.py --scenario train: -15 and -3
# per 20 s episode against ~121), as it does at rest -- where it holds both
# shoulder rolls at ~1.5x their rating -- and, at a quarter, while moving
# (waist yaw 1.4x, shoulder rolls 1.25x).
THERMAL_WEIGHTS = {
  "motor_heat_sustained": -0.35,
  "motor_heat_rate": -0.015,
  "knee_margin": -5.0,
}


def add_thermal_finetune(
  cfg: ManagerBasedRlEnvCfg,
  command_name: str,
  play: bool,
  weights: dict[str, float] | None = None,
  moving_scale: float = 0.25,
  ramp: tuple[int, int] = (50, 500),
  braced_weight: float | None = None,
  waist_yaw_weight: float | None = None,
) -> None:
  """Heat terms, the robot's target clamp, floor friction, and a locked
  robustness level for fine-tuning a converged crawl or stand <-> crawl policy.

  - Rewards: ``motor_heat_sustained`` (full weight while holding still,
    ``moving_scale`` of it while moving), ``motor_heat_rate``,
    ``knee_margin``. They fade in over ``ramp[1]`` iterations after a
    ``ramp[0]``-iteration delay (the PPO's actor is frozen for the first
    updates, so the critic learns them first).
  - PD targets clamped to the joint ranges, as the robot controller does.
  - Floor friction: shared per episode, limb material on top.
  - The robustness curriculum starts and stays at level 1 (where both
    deployed policies finished): no demotion while the new terms fade in.
    ``braced_weight`` replaces its braced-wrist penalty weight there.
  - ``waist_yaw_weight``: a second sustained-heat term on the waist yaw alone,
    at full weight while moving too (the crawl's diagonal gait twists the torso
    on it; summed with the others it was traded for the shoulders).
  """
  weights = dict(THERMAL_WEIGHTS if weights is None else weights)
  joints = CRAWL_JOINTS_29
  ratings = mdp.continuous_ratings(joints)
  knees = SceneEntityCfg("robot", joint_names=(r".*_knee_joint",))
  cfg.rewards["motor_heat_sustained"] = RewardTermCfg(
    func=mdp.motor_heat_sustained,
    weight=weights["motor_heat_sustained"],
    params={
      "ratings": ratings,
      "command_name": command_name,
      "budget": HEAT_BUDGET,
      "tau_s": 2.0,
      "moving_scale": moving_scale,
    },
  )
  if waist_yaw_weight is not None:
    weights["waist_yaw_heat"] = waist_yaw_weight
    cfg.rewards["waist_yaw_heat"] = RewardTermCfg(
      func=mdp.motor_heat_sustained,
      weight=waist_yaw_weight,
      params={
        "ratings": {"waist_yaw_joint": ratings["waist_yaw_joint"]},
        "command_name": command_name,
        "budget": HEAT_BUDGET,
        "tau_s": 2.0,
        "moving_scale": 1.0,
      },
    )
  cfg.rewards["motor_heat_rate"] = RewardTermCfg(
    func=mdp.motor_heat_rate,
    weight=weights["motor_heat_rate"],
    params={"ratings": ratings},
  )
  cfg.rewards["knee_margin"] = RewardTermCfg(
    func=mdp.joint_pos_above,
    weight=weights["knee_margin"],
    params={"asset_cfg": knees, "threshold": KNEE_MARGIN_RAD},
  )
  cfg.metrics["heat/max_motor_ratio"] = MetricsTermCfg(
    func=mdp.motor_heat_rms_ratio, params={"ratings": ratings}, reduce="mean"
  )
  cfg.metrics["heat/load_sum"] = MetricsTermCfg(
    func=mdp.motor_heat_rate, params={"ratings": ratings}, reduce="mean"
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.clip = deploy_target_clip()

  if "dr_friction" in cfg.events:
    cfg.events["dr_friction"] = EventTermCfg(
      func=mdp.floor_friction,
      mode="reset",
      params={"ranges": FLOOR_FRICTION, "materials": LIMB_MATERIALS_29},
    )

  if play or "robustness" not in cfg.curriculum:
    return
  robust = cfg.curriculum["robustness"]
  robust.params["ramp"] = replace(
    robust.params["ramp"],
    friction_low=(FLOOR_FRICTION[0], FLOOR_FRICTION[0]),
    friction_high=(FLOOR_FRICTION[1], FLOOR_FRICTION[1]),
  )
  if braced_weight is not None:
    robust.params["ramp"] = replace(
      robust.params["ramp"], braced_penalty_weight=(braced_weight, braced_weight)
    )
  robust.params["start_level"] = 1.0
  robust.params["demote_progress"] = 0.0
  robust.params["demote_flip_rate"] = 1.01
  cfg.curriculum["thermal_ramp"] = CurriculumTermCfg(
    func=mdp.reward_ramp,
    params={
      "weights": weights,
      "delay_iterations": ramp[0],
      "ramp_iterations": ramp[1],
    },
  )


def unitree_g1_crawl_cool_env_cfg(
  play: bool = False,
  moving_scale: float = 0.25,
  ramp: tuple[int, int] = (50, 500),
  braced_weight: float | None = None,
  waist_yaw_weight: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """Thermal fine-tune of the gen2 crawl (``Mjlab-Crawl-Flat-Cmd7-...-29dof``).

  Same observations, actions and rewards, plus ``add_thermal_finetune``. Rest
  commands are 15 % of resamples and last 6-20 s (were 10 %, 4-8 s): about 30 %
  of the time at zero command, as on the robot, where the crawl rested for up
  to a minute.

  cool_v1 (moving_scale 0.25) brought the rest load down (shoulder rolls 10 ->
  4 Nm in 700 iterations) but not the moving one (waist yaw 1.5x its rating at
  0.3 m/s); the Cool2 task continues from it at 0.5 with a short ramp. Cool2
  shifted load into the braced wrists (loop p95 4.7 -> 6.5 Nm while crawling):
  Cool3 continues from it with the braced penalty at -4 (was -1). Cool2 also
  traded shoulder heat for waist yaw heat while moving (waist yaw 1.38x ->
  1.50x its rating at 0.3 m/s, shoulder rolls 1.24x -> 0.83x): Cool3 adds a
  waist-yaw-only heat term at full weight while moving.
  """
  cfg = unitree_g1_crawl_env_cfg(play=play, cmd7=True, dof29=True)
  command = cfg.commands["base_velocity"]
  assert isinstance(command, mdp.CrawlVelocityCommandCfg)
  command.rel_standing_envs = 0.15
  command.standing_time_range = (6.0, 20.0)
  add_thermal_finetune(
    cfg,
    "base_velocity",
    play,
    moving_scale=moving_scale,
    ramp=ramp,
    braced_weight=braced_weight,
    waist_yaw_weight=waist_yaw_weight,
  )
  return cfg


def unitree_g1_crawl_walkgait_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Cool4: a cooler crawl gait for long crawls (30 min and more).

  Cool2/3 kept the trot: one diagonal pair down at a time, so the waist yaw
  carries the torso's twist every step (35 Nm RMS, 1.5x its rating, at 0.3 m/s;
  limbs down 2.05 on average). Here:
  - commands as the robot uses them: forward -0.3..0.5 m/s, sideways +-0.3,
    turn +-0.8 rad/s (were -1..2.5, +-0.5, +-3);
  - no trot reward (``gait_diagonal``); ``limb_support`` pays for three or more
    limbs down while moving, a walk;
  - every heat term at full weight while moving, the waist-yaw term on top, the
    braced-wrist penalty at -4;
  - ``relative_progress`` pays the share of a slow command achieved: Cool2
    stood still through 0.1 m/s commands.
  """
  cfg = unitree_g1_crawl_cool_env_cfg(
    play=play,
    moving_scale=1.0,
    ramp=(20, 300),
    braced_weight=-4.0,
    waist_yaw_weight=-0.5,
  )
  command = cfg.commands["base_velocity"]
  assert isinstance(command, mdp.CrawlVelocityCommandCfg)
  command.lin_vel_z = (-0.3, 0.5)
  command.lin_vel_y = (-0.3, 0.3)
  command.ang_vel_x = (-0.8, 0.8)
  cfg.rewards.pop("gait_diagonal", None)
  cfg.rewards["limb_support"] = RewardTermCfg(
    func=mdp.limb_support,
    weight=0.6,
    params={
      "sensor_name": "crawl_contacts",
      "limb_names": (HANDS_29[0], HANDS_29[1], FEET_29[0], FEET_29[1]),
      "command_name": "base_velocity",
    },
  )
  cfg.rewards["relative_progress"] = RewardTermCfg(
    func=mdp.relative_progress,
    weight=1.0,
    params={"command_name": "base_velocity"},
  )
  return cfg


def unitree_g1_crawl_creepy_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The creepy crawl (Halloween): slow, deliberate, one limb at a time.

  ``unitree_g1_crawl_walkgait_env_cfg`` (three limbs down, heat priced while
  moving, slow commands paid in full) made creepier, which also keeps it cool:
  - stalking speeds: forward -0.2..0.5 m/s, sideways +-0.25, turn +-0.7
    (creepy_v1 trained to 0.35; the user's range for this style is <= 0.5);
  - long, slow swings: air time paid between 0.15 and 0.8 s (was 0.10-0.40);
  - joints above 4 rad/s penalised (the trot's p95 is ~11), and a doubled
    action-rate penalty: smooth, unhurried limbs.
  The trot crawl stays as its own version; this one starts from it.
  """
  cfg = unitree_g1_crawl_walkgait_env_cfg(play=play)
  command = cfg.commands["base_velocity"]
  assert isinstance(command, mdp.CrawlVelocityCommandCfg)
  command.lin_vel_z = (-0.2, 0.5)
  command.lin_vel_y = (-0.25, 0.25)
  command.ang_vel_x = (-0.7, 0.7)
  air = cfg.rewards["gait_air_time"]
  air.params["threshold_min"] = 0.15
  air.params["threshold_max"] = 0.8
  cfg.rewards["action_rate_l2"].weight = -2e-2
  cfg.rewards["joint_speed_over"] = RewardTermCfg(
    func=mdp.joint_speed_over,
    weight=-0.1,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=CRAWL_JOINTS_29),
      "limit": 4.0,
    },
  )
  return cfg


def unitree_g1_crawl_fast_env_cfg(
  play: bool = False, waist_yaw_weight: float = -0.5
) -> ManagerBasedRlEnvCfg:
  """The fast crawl: most of the time at 0.75-1 m/s, cool and robust.

  From the thermal fine-tune (``unitree_g1_crawl_cool_env_cfg``: heat terms,
  the robot's target clamp, floor friction, robustness level 1 locked):
  - commands: 45 % of resamples cruise straight at 0.7-1.05 m/s (heading
    hold), 10 % rest (6-20 s), 10 % sideways, 10 % spin; the rest mixed, forward
    -0.3..1.2 m/s, sideways +-0.4, turn +-1.0 rad/s;
  - every heat term at full weight while moving, the waist-yaw term on top
    (the trot's twist), the braced-wrist penalty at -4;
  - ``relative_progress`` pays the share of every command achieved, so heat is
    never saved by going slower than asked;
  - not a trot: ``gait_diagonal`` off and a light ``limb_support`` (three limbs
    down) -- the user wants crawl styles other than the trot, and three-limb
    support also takes the torso twist off the waist yaw.
  """
  cfg = unitree_g1_crawl_cool_env_cfg(
    play=play,
    moving_scale=1.0,
    ramp=(20, 300),
    braced_weight=-4.0,
    waist_yaw_weight=waist_yaw_weight,
  )
  command = cfg.commands["base_velocity"]
  assert isinstance(command, mdp.CrawlVelocityCommandCfg)
  command.lin_vel_z = (-0.3, 1.2)
  command.lin_vel_y = (-0.4, 0.4)
  command.ang_vel_x = (-1.0, 1.0)
  command.rel_standing_envs = 0.10
  command.rel_straight_envs = 0.45
  command.rel_sideways_envs = 0.10
  command.rel_spin_envs = 0.10
  command.straight_fwd_range = (0.7, 1.05)
  cfg.rewards.pop("gait_diagonal", None)
  cfg.rewards["limb_support"] = RewardTermCfg(
    func=mdp.limb_support,
    weight=0.3,
    params={
      "sensor_name": "crawl_contacts",
      "limb_names": (HANDS_29[0], HANDS_29[1], FEET_29[0], FEET_29[1]),
      "command_name": "base_velocity",
    },
  )
  cfg.rewards["relative_progress"] = RewardTermCfg(
    func=mdp.relative_progress,
    weight=1.0,
    params={"command_name": "base_velocity"},
  )
  return cfg


# Carpet (2026-10-29 robot test: a foot caught toe-down in the carpet during
# stand-to-crawl). Twist friction on the limbs, drawn per episode with the floor.
CARPET_FLOOR_FRICTION = (0.35, 1.3)
LIMB_TORSIONAL_FRICTION = (0.003, 0.03)


def add_carpet_contacts(cfg: ManagerBasedRlEnvCfg) -> None:
  """Limbs with condim-4 contacts (twist friction), floor friction to 1.3 (feet
  up to ~1.5 with the rubber's draw), twist friction 3-30 mm per episode."""
  robot = cfg.scene.entities["robot"]
  robot.collisions = (CRAWL_SELF_COLLISION_29_TORSION,)
  materials = {
    r"^(left|right)_foot[1-7]_collision$": (0.85, 1.15),
    r"^(left|right)_forearm_tip_collision$": (0.55, 1.0),
  }
  event = cfg.events.get("dr_friction")
  if event is not None:
    event.params.update(
      ranges=CARPET_FLOOR_FRICTION,
      materials=materials,
      torsional=LIMB_TORSIONAL_FRICTION,
    )
  robust = cfg.curriculum.get("robustness")
  if robust is not None:
    hi = CARPET_FLOOR_FRICTION[1]
    robust.params["ramp"] = replace(robust.params["ramp"], friction_high=(hi, hi))
  # A condim-4 contact takes six constraint rows instead of four.
  cfg.sim.njmax = (cfg.sim.njmax or 300) + 100


def continue_thermal_finetune(cfg: ManagerBasedRlEnvCfg) -> None:
  """Resuming a run that already learned the heat terms: no fade-in."""
  ramp = cfg.curriculum.get("thermal_ramp")
  if ramp is not None:
    ramp.params["delay_iterations"] = 0
    ramp.params["ramp_iterations"] = 1


def unitree_g1_crawl_fast_carpet_env_cfg(
  play: bool = False, waist_yaw_weight: float = -0.5
) -> ManagerBasedRlEnvCfg:
  """Fast crawl continued with carpet contacts (``add_carpet_contacts``).

  Fast2 (waist-yaw term -0.5) left the waist yaw at 1.35x its rating at
  0.75-1 m/s while everything else was <= 0.8x: Fast3 triples that term."""
  cfg = unitree_g1_crawl_fast_env_cfg(play=play, waist_yaw_weight=waist_yaw_weight)
  add_carpet_contacts(cfg)
  continue_thermal_finetune(cfg)
  return cfg


def unitree_g1_crawl_creepy_carpet_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Creepy crawl continued with carpet contacts (``add_carpet_contacts``)."""
  cfg = unitree_g1_crawl_creepy_env_cfg(play=play)
  add_carpet_contacts(cfg)
  continue_thermal_finetune(cfg)
  return cfg


def unitree_g1_crawl_creepy_shoulder_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Creepy3: the creepy crawl on carpet with a shoulder-roll heat term.

  Creepy2 brought the waist yaw to 0.5-0.7x its rating but holds the arms
  splayed, the shoulder rolls at 0.8-0.95x while moving (0.52x at rest): too
  much for a 30-minute crawl. The new term prices the two shoulder rolls alone,
  at full weight while moving, as the waist-yaw term did for the trot."""
  cfg = unitree_g1_crawl_creepy_carpet_env_cfg(play=play)
  ratings = mdp.continuous_ratings(CRAWL_JOINTS_29)
  rolls = ("left_shoulder_roll_joint", "right_shoulder_roll_joint")
  cfg.rewards["shoulder_roll_heat"] = RewardTermCfg(
    func=mdp.motor_heat_sustained,
    weight=-1.0,
    params={
      "ratings": {j: ratings[j] for j in rolls},
      "command_name": "base_velocity",
      "budget": HEAT_BUDGET,
      "tau_s": 2.0,
      "moving_scale": 1.0,
    },
  )
  ramp = cfg.curriculum.get("thermal_ramp")
  if ramp is not None:
    ramp.params["weights"]["shoulder_roll_heat"] = -1.0
  return cfg


def unitree_g1_crawl_scuttle_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Scuttle: the fast crawl grown from the creepy gait instead of the trot.

  Fast3 (from the trot) kept twisting the torso every step: waist yaw 1.3x its
  rating at 0.75-1 m/s. The creepy gait (three limbs down) keeps the waist yaw
  at 0.5-0.7x. Here it is asked for the fast crawl's speeds (45 % of commands
  cruise at 0.7-1.05 m/s, up to 1.2), with faster swings allowed (air time
  0.10-0.40 s, joint-speed cap 7 rad/s), a lighter three-limbs-down reward, and
  the waist-yaw and shoulder-roll heat terms; carpet contacts as the others.
  """
  cfg = unitree_g1_crawl_creepy_shoulder_env_cfg(play=play)
  command = cfg.commands["base_velocity"]
  assert isinstance(command, mdp.CrawlVelocityCommandCfg)
  command.lin_vel_z = (-0.3, 1.2)
  command.lin_vel_y = (-0.4, 0.4)
  command.ang_vel_x = (-1.0, 1.0)
  command.rel_standing_envs = 0.10
  command.rel_straight_envs = 0.45
  command.rel_sideways_envs = 0.10
  command.rel_spin_envs = 0.10
  command.straight_fwd_range = (0.7, 1.05)
  command.standing_time_range = (6.0, 20.0)
  air = cfg.rewards["gait_air_time"]
  air.params["threshold_min"] = 0.10
  air.params["threshold_max"] = 0.40
  cfg.rewards["joint_speed_over"].params["limit"] = 7.0
  cfg.rewards["limb_support"].weight = 0.3
  cfg.rewards["action_rate_l2"].weight = -1e-2
  return cfg


def unitree_g1_crawl_scuttle_fast_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Scuttle2: Scuttle settled at 0.75 m/s for 1 m/s commands (the shortfall paid
  too little against the heat of going faster). ``relative_progress`` at 2.5
  (was 1.0): a 25 % shortfall costs ~0.6/s, more than the heat it saves."""
  cfg = unitree_g1_crawl_scuttle_env_cfg(play=play)
  cfg.rewards["relative_progress"].weight = 2.5
  return cfg


def unitree_g1_crawl_scuttle_free_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Scuttle3: Scuttle2 still topped out at 0.75 m/s for 1 m/s commands. The
  creepy crawl's slow-motion limits held it back: no joint-speed cap (it was 7
  rad/s; the 1 m/s trot needs ~11) and the three-limbs-down reward at a token
  0.1. Heat terms and the 2.5 progress weight unchanged."""
  cfg = unitree_g1_crawl_scuttle_fast_env_cfg(play=play)
  cfg.rewards.pop("joint_speed_over", None)
  cfg.rewards["limb_support"].weight = 0.1
  return cfg


# The chain test's forbidden bodies (scripts/play_chain.py): any ground touch
# by these is a failure there.
FORBIDDEN_BODIES_29 = (
  r"^(pelvis|torso_link|waist_.*_link|.*_shoulder_.*_link|.*_elbow_link"
  r"|.*_wrist_.*_link|.*_wrist_brace)$"
)


def add_hard_crawl_robustness(cfg: ManagerBasedRlEnvCfg) -> None:
  """The chain test's hard randomization, which the crawl never trained at.

  The creepy crawl failed 15.6 % of hard chain loops (the trot chain 3-4 %):
  half while crawling, most with a brace, the torso or the pelvis on the
  ground. Level 1 here matches ``TRANSITION_RAMP_HARD`` (+-20 % mass, payload
  -3..+5 kg, CoM +-5 cm, joint friction <= 1 Nm, rotor inertia and gains +-25-30
  %, motors at 75-100 %, sensor noise x2; pushes stay at the crawl's 1 m/s),
  arm ends -0.05..+0.2 kg as on the hard walk, and a forbidden-body touch
  (> 10 N) ends the episode (-100) instead of only costing a little.
  """
  robust = cfg.curriculum.get("robustness")
  if robust is not None:
    r = robust.params["ramp"]
    robust.params["ramp"] = replace(
      r,
      mass_scale=(0.2, 0.2),
      payload_low_kg=(-3.0, -3.0),
      payload_high_kg=(5.0, 5.0),
      com_offset_m=(0.05, 0.05),
      joint_friction_nm=(1.0, 1.0),
      armature_scale=(0.3, 0.3),
      pd_gain_scale=(0.25, 0.25),
      effort_scale_low=(0.75, 0.75),
      obs_noise_scale=(2.0, 2.0),
    )
  if "dr_friction" in cfg.events:
    cfg.events["dr_arm_end_mass"] = EventTermCfg(
      func=body_mass,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=r"^(left|right)_(wrist_roll_link|forearm_extension)$"
        ),
        "ranges": (-0.05, 0.2),
        "operation": "add",
      },
    )
  cfg.terminations["forbidden_touch"] = TerminationTermCfg(
    func=mdp.bodies_touch_ground,
    params={
      "sensor_name": "body_ground_contacts",
      "body_pattern": FORBIDDEN_BODIES_29,
      "threshold": 10.0,
    },
  )


def unitree_g1_crawl_creepy_hard_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Creepy3 hardened (``add_hard_crawl_robustness``)."""
  cfg = unitree_g1_crawl_creepy_shoulder_env_cfg(play=play)
  add_hard_crawl_robustness(cfg)
  return cfg


def unitree_g1_crawl_scuttle_hard_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Scuttle3 hardened (``add_hard_crawl_robustness``)."""
  cfg = unitree_g1_crawl_scuttle_free_env_cfg(play=play)
  add_hard_crawl_robustness(cfg)
  return cfg


def unitree_g1_crawl_scuttle_cool_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """ScuttleHard with a sustained-heat term on the waist pitch at full weight
  while moving: at 1 m/s it ran at 0.97x its rating (waist yaw 0.85x), the
  limit on how long the scuttle can run. The heat terms fade back in quickly
  (the resumed policy already trained with them)."""
  cfg = unitree_g1_crawl_scuttle_hard_env_cfg(play=play)
  ratings = mdp.continuous_ratings(CRAWL_JOINTS_29)
  cfg.rewards["waist_pitch_heat"] = RewardTermCfg(
    func=mdp.motor_heat_sustained,
    weight=-0.5,
    params={
      "ratings": {"waist_pitch_joint": ratings["waist_pitch_joint"]},
      "command_name": "base_velocity",
      "budget": HEAT_BUDGET,
      "tau_s": 2.0,
      "moving_scale": 1.0,
    },
  )
  ramp = cfg.curriculum.get("thermal_ramp")
  if ramp is not None:
    ramp.params["weights"]["waist_pitch_heat"] = -0.5
    ramp.params["delay_iterations"] = 20
    ramp.params["ramp_iterations"] = 150
  return cfg
