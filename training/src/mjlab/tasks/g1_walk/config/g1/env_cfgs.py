"""Walk/stand and stand-to-crawl configurations for the braced 29-DoF G1."""

from __future__ import annotations

import math
import re
from dataclasses import replace

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
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
from mjlab.tasks.g1_crawl import mdp as crawl_mdp
from mjlab.tasks.g1_crawl.config.g1.env_cfgs import (
  add_carpet_contacts,
  add_thermal_finetune,
  continue_thermal_finetune,
)
from mjlab.tasks.g1_crawl.mdp import metrics
from mjlab.tasks.g1_crawl.robot29 import (
  BRACED_LOADS_29,
  BRACED_TORQUE_LIMIT,
  FEET_29,
  HANDS_29,
  MOTORS_29,
)
from mjlab.tasks.g1_walk import mdp
from mjlab.tasks.g1_walk.robot import (
  JOINTS_29,
  get_g1_kneel_robot_cfg_29,
  get_g1_walk_robot_cfg_29,
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
  random_rough,
  wave_terrain,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg

# Torque sensors across the braced wrist joints (scene names carry the entity
# prefix) and the joint axis each one is read along; see ``robot29``.
BRACED_TORQUE_SENSORS = tuple(
  f"robot/{joint.removesuffix('_joint')}_torque" for joint in BRACED_LOADS_29
)
BRACED_AXES = tuple(axis for _, axis in BRACED_LOADS_29.values())
TORQUE_SOFT_FRACTION = 0.8

FOOT_SITES = ("left_foot", "right_foot")
TORSO = ("torso_link",)


def _action_scale_29() -> dict[str, float]:
  """The asset zoo's per-motor action scale, resolved onto the 25 active joints."""
  scale = {}
  for joint in JOINTS_29:
    matches = [v for k, v in G1_ACTION_SCALE.items() if re.fullmatch(k, joint)]
    assert len(matches) == 1, (joint, matches)
    scale[joint] = matches[0]
  return scale


def _braced_sensors() -> tuple[BuiltinSensorCfg, ...]:
  return tuple(
    BuiltinSensorCfg(
      name=f"{joint.removesuffix('_joint')}_torque",
      sensor_type="torque",
      obj=ObjRef(type="site", name=site, entity="robot"),
    )
    for joint, (site, _) in BRACED_LOADS_29.items()
  )


def _ground_sensors(
  feet: tuple[str, ...], limbs_pattern: str
) -> tuple[ContactSensorCfg, ...]:
  """Feet (air time), every body (undesired contacts) and self-collisions."""
  del feet
  return (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(mode="body", pattern=limbs_pattern, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
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
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )


def _foot_height_sensor(sites: tuple[str, ...]) -> TerrainHeightSensorCfg:
  return TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in sites),
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )


def _observations(joints: tuple[str, ...], command: str) -> dict:
  """Actor: what the real robot measures. Critic: plus privileged state."""
  jcfg = SceneEntityCfg("robot", joint_names=joints)
  actor = {
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
    ),
    # Pelvis gyro: the G1's pelvis IMU frame is aligned with the pelvis.
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)
    ),
    "velocity_commands": ObservationTermCfg(
      func=envs_mdp.generated_commands, params={"command_name": command}
    ),
    # Biased: the real encoders carry a calibration offset (encoder_bias event).
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": jcfg, "biased": True},
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": jcfg},
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }
  critic = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    **{k: ObservationTermCfg(func=v.func, params=v.params) for k, v in actor.items()},
    # The critic gets the true joint positions.
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel, params={"asset_cfg": jcfg}
    ),
    "foot_air_time": ObservationTermCfg(
      func=velocity_mdp.foot_air_time, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact": ObservationTermCfg(
      func=velocity_mdp.foot_contact, params={"sensor_name": "feet_ground_contact"}
    ),
    "foot_contact_forces": ObservationTermCfg(
      func=velocity_mdp.foot_contact_forces,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }
  return {
    "actor": ObservationGroupCfg(
      terms=actor, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic, concatenate_terms=True, enable_corruption=False
    ),
  }


def _posture_std() -> tuple[dict, dict, dict]:
  """MJLab's G1 walking posture tolerances (rad); wrists here are wrist roll."""
  standing = {".*": 0.05}
  walking = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    r".*ankle_roll.*": 0.1,
    r".*waist_yaw.*": 0.2,
    r".*waist_roll.*": 0.08,
    r".*waist_pitch.*": 0.1,
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.15,
    r".*wrist.*": 0.3,
  }
  running = {k: v * 1.6 for k, v in walking.items()}
  running[r".*waist_roll.*"] = 0.08
  return standing, walking, running


def _add_torque_safety(cfg: ManagerBasedRlEnvCfg) -> None:
  """Demand-based torque penalties (see the crawl task): off until the
  robustness curriculum fades them in, once the gait exists."""
  cfg.rewards["motor_torque_margin"] = RewardTermCfg(
    func=crawl_mdp.motor_torque_margin,
    weight=0.0,
    params={"motors": MOTORS_29, "soft_fraction": TORQUE_SOFT_FRACTION},
  )
  cfg.rewards["braced_wrist_torque"] = RewardTermCfg(
    func=crawl_mdp.braced_joint_torque,
    weight=0.0,
    params={
      "sensor_names": BRACED_TORQUE_SENSORS,
      "axes": BRACED_AXES,
      "limit": BRACED_TORQUE_LIMIT,
      "soft_fraction": TORQUE_SOFT_FRACTION,
    },
  )


def _add_robust_training(
  cfg: ManagerBasedRlEnvCfg,
  command: str,
  play: bool,
  ramp: mdp.RobustnessRamp,
  curriculum=mdp.walk_robustness,
  push_interval_s: tuple[float, float] = (4.0, 8.0),
  **gate,
) -> None:
  """Per-episode randomization owned by the robustness curriculum (crawl's)."""
  robot_all = SceneEntityCfg("robot")
  torso = SceneEntityCfg("robot", body_names=TORSO)
  dr = {
    "dr_friction": (
      geom_friction,
      {
        "asset_cfg": SceneEntityCfg("robot", geom_names=r".*_collision.*"),
        "ranges": (ramp.friction_low[0], ramp.friction_high[0]),
        "operation": "abs",
        "shared_random": True,  # One surface per robot: the ends of the range happen.
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
    interval_range_s=push_interval_s,
    params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
  )
  if play:
    for name in dr:
      cfg.events.pop(name)
    cfg.events.pop("push_robot")
    return
  cfg.curriculum["robustness"] = CurriculumTermCfg(
    func=curriculum,
    params={
      "command_name": command,
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
      "noisy_obs": ("projected_gravity", "base_ang_vel", "joint_pos", "joint_vel"),
      "motor_reward": "motor_torque_margin",
      "braced_reward": "braced_wrist_torque",
      **gate,
    },
  )


# The brace and extension masses are estimates (``BRACE_MASS``,
# ``EXTENSION_MASS``): the hard tasks add -50..+200 g at each arm end.
ARM_END_BODIES = r"^(left|right)_(wrist_roll_link|forearm_extension)$"
IMU_TILT_MAX_DEG = 2.0


def _add_hard_robustness(cfg: ManagerBasedRlEnvCfg, play: bool) -> None:
  """On top of the hard ramp: what system identification would have pinned
  down. The actor's gravity comes from an IMU tilted up to IMU_TILT_MAX_DEG
  (fixed per episode), and the arm-end masses are uncertain."""
  term = cfg.observations["actor"].terms["projected_gravity"]
  term.func = mdp.projected_gravity_tilted
  term.params = {"max_deg": IMU_TILT_MAX_DEG}
  if play:
    term.params = {"max_deg": 0.0}
    return
  cfg.events["dr_arm_end_mass"] = EventTermCfg(
    func=body_mass,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=ARM_END_BODIES),
      "ranges": (-0.05, 0.2),
      "operation": "add",
    },
  )


def _add_metrics(cfg: ManagerBasedRlEnvCfg, feet: tuple[str, ...]) -> None:
  """``Episode_Metrics/<group>/<name>``: means unless marked peak."""
  legs = {k: v for k, v in MOTORS_29.items() if re.search("hip|knee|ankle", k)}
  waist = {k: v for k, v in MOTORS_29.items() if k.startswith("waist")}
  arms = {k: v for k, v in MOTORS_29.items() if k not in legs and k not in waist}
  bodies = {"sensor_name": "body_ground_contacts"}
  braced = {"sensor_names": BRACED_TORQUE_SENSORS, "axes": BRACED_AXES}
  m = metrics
  terms: dict[str, tuple] = {
    "gait/feet_on_ground_frac": (
      m.limb_contact_frac,
      {"sensor_name": "feet_ground_contact", "body_names": feet},
      "mean",
    ),
    "gait/torso_on_ground_frac": (
      m.bodies_touching,
      {**bodies, "body_names": ("torso_link", "pelvis")},
      "mean",
    ),
    "gait/other_bodies_touching": (
      m.other_bodies_touching,
      {**bodies, "excluded_body_names": feet},
      "mean",
    ),
    "gait/body_height_m": (m.body_height, {}, "mean"),
    "safety/braced_wrist_peak_nm": (
      crawl_mdp.braced_joint_moment_max,
      braced,
      "max",
    ),
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
    "safety/joint_near_limit_frac": (
      m.joint_near_limit,
      {"joint_names": tuple(MOTORS_29)},
      "mean",
    ),
    "effort/mech_power_w": (m.mechanical_power, {"motors": MOTORS_29}, "mean"),
  }
  for name, (func, params, reduce) in terms.items():
    cfg.metrics[name] = MetricsTermCfg(func=func, params=params, reduce=reduce)


##
# Rough terrain.
##

# Same layout as the crawl task's CRAWL_ZONES_CFG (one sub-terrain per column,
# difficulty rising along the rows) so ``terrain_zones_crawl`` drives it. The
# walking policy is blind, so the tiles stay within what a biped can feel its
# way over: steps up to 10 cm, slopes up to about 14 degrees.
WALK_ZONES_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=5,  # Difficulty 0.2, 0.4, 0.6, 0.8, 1.0 per zone.
  num_cols=20,
  curriculum=True,
  difficulty_range=(0.2, 1.0),
  sub_terrains={
    "flat": flat(proportion=0.12),
    "gravel": random_rough(
      proportion=0.11, noise_range=(0.005, 0.04), noise_step=0.005
    ),
    "hill": hf_pyramid_slope(
      proportion=0.11, slope_range=(0.05, 0.25), horizontal_scale=0.2
    ),
    "valley": hf_pyramid_slope_inv(
      proportion=0.11, slope_range=(0.05, 0.25), horizontal_scale=0.2
    ),
    "stairs_up": pyramid_stairs(
      proportion=0.11, step_height_range=(0.02, 0.10), step_width=0.35
    ),
    "stairs_down": pyramid_stairs_inv(
      proportion=0.11, step_height_range=(0.02, 0.10), step_width=0.35
    ),
    "box_grid": box_random_grid(
      proportion=0.11, grid_width=0.4, grid_height_range=(0.0, 0.06)
    ),
    "waves": wave_terrain(proportion=0.11, amplitude_range=(0.0, 0.10)),
    "random_mix": flat(proportion=0.11),
  },
  add_lights=True,
)

# The robot goes down where it stands, so what matters is the ground under it
# and under the limbs that sprawl out: no stairs or slopes with a flat centre.
TRANSITION_ZONES_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=5,
  num_cols=20,
  curriculum=True,
  difficulty_range=(0.2, 1.0),
  sub_terrains={
    "flat": flat(proportion=0.2),
    "gravel": random_rough(
      proportion=0.2, noise_range=(0.005, 0.03), noise_step=0.005, border_width=0.0
    ),
    "meadows": perlin_noise(
      proportion=0.2,
      height_range=(0.02, 0.08),
      scale=6.0,
      octaves=3,
      horizontal_scale=0.2,
      resolution=0.2,
    ),
    "box_grid": box_random_grid(
      proportion=0.2,
      grid_width=0.4,
      grid_height_range=(0.0, 0.04),
      platform_width=0.6,
    ),
    "waves": wave_terrain(proportion=0.2, amplitude_range=(0.0, 0.04)),
  },
  add_lights=True,
)

# Curriculum gates, in iterations counted from the start of the run (a resumed
# run starts counting at its first iteration): flat ground while the robustness
# curriculum settles, then the zones, then every env on a random tile.
WALK_FLAT_ITERATIONS = 200
WALK_SPREAD_AFTER_ITERATIONS = 8000
TRANSITION_FLAT_ITERATIONS = 200
TRANSITION_SPREAD_AFTER_ITERATIONS = 4000


def _make_rough(
  cfg: ManagerBasedRlEnvCfg,
  zones: TerrainGeneratorCfg,
  curriculum: CurriculumTermCfg,
  play: bool,
) -> None:
  """Swap the plane for generated terrain with the zone curriculum on top."""
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = zones
  cfg.scene.terrain.max_init_terrain_level = 0
  # Stairs and heightfields produce many more contacts per robot.
  cfg.sim.nconmax = 220
  cfg.sim.njmax = 600
  if play:
    # The same zones left to right, no curriculum: envs spawn anywhere.
    cfg.scene.terrain.terrain_generator = replace(
      zones, num_rows=4, border_width=10.0, difficulty_range=(0.4, 1.0)
    )
    cfg.scene.terrain.max_init_terrain_level = None
  else:
    cfg.curriculum["terrain_zones"] = curriculum


def unitree_g1_walk_env_cfg(
  play: bool = False, rough: bool = False, hard: bool = False
) -> ManagerBasedRlEnvCfg:
  """Flat-ground omnidirectional walking and standing for the braced G1.

  MJLab's G1 velocity task on the crawl robot, with the sim-to-real additions
  of the crawl task: an actor that sees only gravity, gyro, joints and its own
  actions; demand-based torque limits on every motor and on the braced wrists;
  and the performance-gated robustness curriculum (randomization re-drawn each
  episode, pushes, noise, command delay) that only hardens the policy once it
  tracks its commands without falling.

  ``rough=True`` puts it on generated terrain (``WALK_ZONES_CFG``) under the
  crawl task's zone curriculum, with the same pushes, randomization and noise.

  ``hard=True`` trains under ``WALK_RAMP_HARD`` (it starts where ``WALK_RAMP``
  ends) plus an IMU tilt and arm-end mass uncertainty (``_add_hard_robustness``),
  for resuming a walk that already finished the normal curriculum.
  """
  joints = JOINTS_29
  command = "twist"
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_g1_walk_robot_cfg_29()}
  cfg.scene.num_envs = 4096
  cfg.viewer.body_name = "torso_link"

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = (
    (_foot_height_sensor(FOOT_SITES),)
    + _ground_sensors(FEET_29, r"^(left|right)_ankle_roll_link$")
    + _braced_sensors()
  )

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = joints
  action.scale = _action_scale_29()
  action.use_default_offset = True

  cfg.commands = {
    command: mdp.WalkVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.15,
      rel_heading_envs=0.3,
      rel_forward_envs=0.2,
      rel_spin_envs=0.25,
      spin_min=0.2,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=mdp.WalkVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.6, 1.0),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-0.8, 0.8),
        heading=(-math.pi, math.pi),
      ),
    )
  }
  cfg.commands[command].viz.z_offset = 1.15  # type: ignore[attr-defined]
  cfg.observations = _observations(joints, command)

  cfg.events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.03),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "yaw": (-0.2, 0.2),
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "encoder_bias": cfg.events["encoder_bias"],
  }

  standing, walking, running = _posture_std()
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=velocity_mdp.track_linear_velocity,
      weight=2.0,
      params={"command_name": command, "std": math.sqrt(0.25)},
    ),
    # A tighter yaw kernel than MJLab's sqrt(0.5): at that width a robot that
    # twists on planted feet instead of stepping round still earns most of it.
    "track_angular_velocity": RewardTermCfg(
      func=velocity_mdp.track_angular_velocity,
      weight=2.5,
      params={"command_name": command, "std": 0.5},
    ),
    # Stepping pays whenever anything is commanded, a turn in place included:
    # from a standstill the policy otherwise never finds out that it has to step
    # round, and stands through pure yaw commands.
    "air_time": RewardTermCfg(
      func=velocity_mdp.feet_air_time,
      weight=0.5,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": command,
        "command_threshold": 0.15,
      },
    ),
    # Linear credit toward the commanded yaw rate (see ``mdp.yaw_progress``).
    "yaw_progress": RewardTermCfg(
      func=mdp.yaw_progress, weight=1.0, params={"command_name": command}
    ),
    "upright": RewardTermCfg(
      func=velocity_mdp.upright,
      weight=1.0,
      params={
        "std": math.sqrt(0.2),
        "asset_cfg": SceneEntityCfg("robot", body_names=TORSO),
      },
    ),
    "pose": RewardTermCfg(
      func=velocity_mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=joints),
        "command_name": command,
        "std_standing": standing,
        "std_walking": walking,
        "std_running": running,
        "walking_threshold": 0.05,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=velocity_mdp.body_angular_velocity_penalty,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=TORSO)},
    ),
    "angular_momentum": RewardTermCfg(
      func=velocity_mdp.angular_momentum_penalty,
      weight=-0.02,
      params={"sensor_name": "robot/root_angmom"},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=joints)},
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "foot_clearance": RewardTermCfg(
      func=velocity_mdp.feet_clearance,
      weight=-2.0,
      params={
        "target_height": 0.1,
        "height_sensor_name": "foot_height_scan",
        "command_name": command,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
    "foot_swing_height": RewardTermCfg(
      func=velocity_mdp.feet_swing_height,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "height_sensor_name": "foot_height_scan",
        "target_height": 0.1,
        "command_name": command,
        "command_threshold": 0.05,
      },
    ),
    "foot_slip": RewardTermCfg(
      func=velocity_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": command,
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
    "soft_landing": RewardTermCfg(
      func=velocity_mdp.soft_landing,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": command,
        "command_threshold": 0.05,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    # Anything but the feet on the ground (the extensions are not hands here).
    "undesired_body_contacts": RewardTermCfg(
      func=crawl_mdp.undesired_ground_contacts,
      weight=-2.0,
      params={
        "sensor_name": "body_ground_contacts",
        "excluded_body_names": FEET_29,
        "threshold": 1.0,
      },
    ),
    "stand_still": RewardTermCfg(
      func=crawl_mdp.stand_still_joint_vel,
      weight=-0.005,
      params={
        "command_name": command,
        "asset_cfg": SceneEntityCfg("robot", joint_names=joints),
        "threshold": 0.1,
      },
    ),
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  _add_torque_safety(cfg)

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}
    ),
    "too_low": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum, params={"minimum_height": 0.3}
    ),
  }
  cfg.curriculum = {}
  cfg.metrics = {"mean_action_acc": cfg.metrics["mean_action_acc"]}
  if hard:
    _add_robust_training(
      cfg,
      command,
      play,
      mdp.WALK_RAMP_HARD,
      min_commanded_m=1.0,
      warmup_iterations=200,
      min_iterations_per_level=200,
      promote_max_flip_rate=0.08 if rough else 0.05,
      # Resumes a walk that had climbed (the level is not in checkpoints).
      start_level=0.7,
    )
    _add_hard_robustness(cfg, play)
    # The rough/hard lineage over-turns (0.6 rad/s commanded, ~0.95 delivered):
    # the capped yaw_progress pays under-turning twice and over-turning not at
    # all. Make the symmetric tracking kernel the main yaw term. (Tripling
    # yaw_progress instead drove it to 1.7 rad/s; see G1_PROJECT_CONTEXT.md.)
    cfg.rewards["yaw_progress"].weight = 0.5
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.35
  else:
    _add_robust_training(cfg, command, play, mdp.WALK_RAMP, min_commanded_m=1.0)
  _add_metrics(cfg, FEET_29)
  if rough:
    # Heights are measured over the ground under the feet, not the world origin.
    cfg.terminations["too_low"] = TerminationTermCfg(
      func=mdp.root_height_above_ground_below,
      params={
        "minimum_height": 0.3,
        "height_sensor_name": "foot_height_scan",
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    )

  cfg.sim.nconmax = 120
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  cfg.episode_length_s = 20.0

  if rough:
    _make_rough(
      cfg,
      WALK_ZONES_CFG,
      CurriculumTermCfg(
        func=mdp.terrain_zones_walk,
        params={
          "flat_iterations": WALK_FLAT_ITERATIONS,
          "spread_after_iterations": WALK_SPREAD_AFTER_ITERATIONS,
        },
      ),
      play,
    )

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


##
# Stand-to-crawl transition.
##

HAND_SITES = ("left_hand_contact", "right_hand_contact")
TRANSITION_ACTION_SCALE = 0.5  # rad per unit of residual action.
TRANSITION_ACTION_CLIP = 1.6  # Raw-action clip in the runner: +-0.8 rad residual.
# Bodies that must never touch the ground during the transition (episode ends).
FORBIDDEN_BODIES = (
  r"^(pelvis|torso_link|waist_.*_link|.*_shoulder_.*_link|.*_elbow_link"
  r"|.*_wrist_.*_link|.*_wrist_brace)$"
)
TRANSITION_EPISODE_S = 12.5  # Up to 1.5 s standing + 8 s descent + 3 s hold.
# Two-way: up to 1.5 s wait + 8 s down + 4 s hold + 8 s up, then standing.
STAND_CRAWL_EPISODE_S = 25.0


def unitree_g1_stand_to_crawl_env_cfg(
  play: bool = False, rough: bool = False, bidirectional: bool = False
) -> ManagerBasedRlEnvCfg:
  """Slow, scheduled descent from standing into the crawl task's initial pose.

  See ``mdp/transition.py`` for the phase command and its schedules. The
  episode starts standing, the phase runs over 4-8 s (its rate is observed, so
  the switcher can pick the speed), and the policy then holds the crawl pose
  until the episode ends. Falls, and any contact of the torso, pelvis or arm
  links with the ground, end the episode. Torque limits apply from the start.

  ``rough=True`` descends on uneven ground (``TRANSITION_ZONES_CFG``), with the
  schedule's heights measured from the terrain height at the spawn point.

  ``bidirectional=True``: one policy for both ways (stand -> crawl and crawl ->
  stand). The phase runs down, holds, and in most episodes runs back up; a third
  of the episodes start in the crawl pose. The command is ``[phi, dphi/dt]``.
  It trains under ``TRANSITION_RAMP_HARD`` with the IMU tilt and arm-end mass
  uncertainty of the hard walk, behind an achievable gate (<= 10% falls).
  """
  joints = JOINTS_29
  command = "phase"
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_g1_walk_robot_cfg_29()}
  cfg.scene.num_envs = 4096
  cfg.viewer.body_name = "torso_link"

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  limbs = r"^(left|right)_(ankle_roll_link|forearm_extension)$"
  cfg.scene.sensors = (
    _ground_sensors(FEET_29, limbs)
    + (
      ContactSensorCfg(
        name="forbidden_contact",
        primary=ContactMatch(mode="body", pattern=FORBIDDEN_BODIES, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
      ),
    )
    + _braced_sensors()
  )

  # Residual on the scheduled reference pose: with the runner clipping raw
  # actions at TRANSITION_ACTION_CLIP, the policy can depart from the schedule
  # by at most TRANSITION_ACTION_CLIP * TRANSITION_ACTION_SCALE rad.
  cfg.actions = {
    "joint_pos": mdp.ScheduledJointPositionActionCfg(
      entity_name="robot",
      actuator_names=joints,
      scale=TRANSITION_ACTION_SCALE,
      command_name=command,
    )
  }

  cfg.commands = {
    command: mdp.TransitionPhaseCommandCfg(
      entity_name="robot",
      resampling_time_range=(1e3, 1e3),  # Drawn once per episode.
      duration_s=(4.0, 8.0),
      delay_s=(0.5, 1.5),
      bidirectional=bidirectional,
    )
  }
  cfg.observations = _observations(joints, command)
  # Actor layout (108): gravity, gyro, joint_pos, joint_vel, actions, then the
  # phase command [phi, 1/T] and the reference minus the measured joint pose.
  # (``play_switcher._TransitionPolicy`` rebuilds exactly this order.)
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["phase"] = terms.pop("velocity_commands")
    # Where the schedule wants each joint: what the residual acts on.
    terms["ref_joint_err"] = ObservationTermCfg(
      func=mdp.ref_joint_error, params={"command_name": command}
    )

  cfg.events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.02),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "encoder_bias": cfg.events["encoder_bias"],
  }
  if bidirectional:
    cfg.events["reset_crawl_start"] = EventTermCfg(
      func=mdp.reset_crawl_start,
      mode="reset",
      params={"command_name": command, "prob": 0.0 if play else 0.35},
    )

  jcfg = SceneEntityCfg("robot", joint_names=joints)
  cfg.rewards = {
    # Follow the schedules.
    "ref_height": RewardTermCfg(
      func=mdp.ref_height_exp,
      weight=1.0,
      params={"command_name": command, "std": 0.08},
    ),
    "ref_orientation": RewardTermCfg(
      func=mdp.ref_gravity_exp,
      weight=1.0,
      params={"command_name": command, "std": 0.35},
    ),
    "ref_joint_pose": RewardTermCfg(
      func=mdp.ref_joint_pose_exp,
      weight=1.5,
      params={"command_name": command, "std": 0.45},
    ),
    "ref_hand_height": RewardTermCfg(
      func=mdp.ref_hand_height_exp,
      weight=1.0,
      params={
        "command_name": command,
        "std": 0.12,
        "asset_cfg": SceneEntityCfg("robot", site_names=HAND_SITES),
      },
    ),
    "final_pose": RewardTermCfg(
      func=mdp.final_pose_exp,
      weight=3.0,
      params={"command_name": command, "std": 0.35},
    ),
    # Feet planted through the squat and lean; from all fours on they must
    # step outward into the sprawl, and end flat on the ground.
    "feet_down": RewardTermCfg(
      func=mdp.feet_both_down,
      weight=0.3,
      params={
        "sensor_name": "feet_ground_contact",
        "body_names": FEET_29,
        "command_name": command,
        "phase_max": 0.55,
      },
    ),
    "feet_flat_end": RewardTermCfg(
      func=mdp.feet_flat_end,
      weight=1.0,
      params={
        "command_name": command,
        "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
      },
    ),
    # Slow, soft, still.
    "hold_still": RewardTermCfg(
      func=mdp.hold_still, weight=-0.02, params={"command_name": command}
    ),
    "wait_still": RewardTermCfg(
      func=mdp.before_start_still, weight=-0.02, params={"command_name": command}
    ),
    "joint_vel_cap": RewardTermCfg(
      func=mdp.joint_vel_cap,
      weight=-0.5,
      params={"asset_cfg": jcfg, "limit": 2.5},
    ),
    "base_speed_cap": RewardTermCfg(
      func=mdp.base_speed_cap, weight=-1.0, params={"limit": 0.5}
    ),
    "contact_impact": RewardTermCfg(
      func=mdp.contact_impact,
      weight=-2.0,
      params={"sensor_name": "body_ground_contacts"},
    ),
    "undesired_body_contacts": RewardTermCfg(
      func=crawl_mdp.undesired_ground_contacts,
      weight=-1.0,
      params={
        "sensor_name": "body_ground_contacts",
        "excluded_body_names": FEET_29 + HANDS_29,
        "threshold": 1.0,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": jcfg}
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  if bidirectional:
    # Stood back up (or still waiting): the stand pose, for the walk hand-over.
    cfg.rewards["stand_pose"] = RewardTermCfg(
      func=mdp.stand_pose_exp,
      weight=3.0,
      params={"command_name": command, "std": 0.35},
    )
  _add_torque_safety(cfg)

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell": TerminationTermCfg(func=mdp.fell),
    "forbidden_contact": TerminationTermCfg(
      func=velocity_mdp.illegal_contact,
      params={"sensor_name": "forbidden_contact", "force_threshold": 30.0},
    ),
  }
  cfg.curriculum = {}
  cfg.metrics = {"mean_action_acc": cfg.metrics["mean_action_acc"]}
  if bidirectional:
    _add_robust_training(
      cfg,
      command,
      play,
      mdp.TRANSITION_RAMP_HARD,
      curriculum=mdp.transition_robustness,
      push_interval_s=(3.0, 6.0),
      success_pose_err=0.3,
      min_iterations_per_level=200,
      promote_max_flip_rate=0.10,
      demote_flip_rate=0.25,
      # The level is not saved in checkpoints; a resume of a policy that had
      # climbed starts here (the gate still demotes if it is too hard).
      start_level=0.3,
    )
    _add_hard_robustness(cfg, play)
  else:
    _add_robust_training(
      cfg,
      command,
      play,
      mdp.TRANSITION_RAMP,
      curriculum=mdp.transition_robustness,
      push_interval_s=(3.0, 6.0),
      success_pose_err=0.3,
    )
  _add_metrics(cfg, FEET_29)
  cfg.metrics["gait/hands_on_ground_frac"] = MetricsTermCfg(
    func=metrics.limb_contact_frac,
    params={"sensor_name": "feet_ground_contact", "body_names": HANDS_29},
    reduce="mean",
  )

  cfg.sim.nconmax = 120
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  cfg.episode_length_s = (
    STAND_CRAWL_EPISODE_S if bidirectional else TRANSITION_EPISODE_S
  )

  if rough:
    _make_rough(
      cfg,
      TRANSITION_ZONES_CFG,
      CurriculumTermCfg(
        func=mdp.terrain_zones_transition,
        params={
          "flat_iterations": TRANSITION_FLAT_ITERATIONS,
          "spread_after_iterations": TRANSITION_SPREAD_AFTER_ITERATIONS,
        },
      ),
      play,
    )

  if play:
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


def unitree_g1_stand_crawl_cool_env_cfg(
  play: bool = False,
  heat_weight: float = -0.4,
  ramp: tuple[int, int] = (50, 500),
  moving_scale: float = 0.25,
) -> ManagerBasedRlEnvCfg:
  """Thermal fine-tune of the two-way stand <-> crawl
  (``Mjlab-StandCrawl-Unitree-G1-29dof``), after the 2026-10-29 robot test.

  ``add_thermal_finetune`` (heat terms, the robot's target clamp, floor
  friction, level locked at 1), plus what the robot showed:
  - holds at phi = 1 last 1.5-20 s (were 1.5-4 s): on the robot the hold lasts
    until the hand-over gate passes;
  - 40 % of the crawl-pose starts begin nose-down with the hips high and the
    shoulders skewed, the posture it got stuck in for 25 s.

  cool_v1 brought the hold's right shoulder roll from 13.6 to 9.9 Nm, still
  1.5x its rating and one-sided; the Cool2 task continues with
  ``heat_weight`` doubled and a left-right mirror loss (hold 0.66x its rating,
  falls 17 -> 6 %). Cool3 continues it with the heat of the transitions
  themselves priced at half (``moving_scale``): repeated stand <-> crawl cycles.
  """
  cfg = unitree_g1_stand_to_crawl_env_cfg(play=play, bidirectional=True)
  command = cfg.commands["phase"]
  assert isinstance(command, mdp.TransitionPhaseCommandCfg)
  command.hold_s = (1.5, 20.0)
  # Up to 1.5 s wait + 8 s down + 20 s hold + 8 s up, then standing.
  cfg.episode_length_s = 40.0
  if "reset_crawl_start" in cfg.events:
    cfg.events["reset_crawl_start"].params["nose_down_frac"] = 0.4
  # The deployed policy (stand_crawl_v3/24999) at these weights pays about
  # -30 and -6 per 40 s episode against ~257 (thermal_eval --scenario
  # hold_train); its knees reach ~2.70 rad on the way down in simulation, the
  # real robot's 2.88.
  weights = {
    "motor_heat_sustained": heat_weight,
    "motor_heat_rate": -0.03,
    "knee_margin": -10.0,
  }
  add_thermal_finetune(
    cfg, "phase", play, weights=weights, ramp=ramp, moving_scale=moving_scale
  )
  return cfg


def unitree_g1_stand_crawl_carpet_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Stand <-> crawl for carpet (robot test, 2026-10-29: a foot caught toe-down
  in the carpet on the way down and had to be freed by hand).

  Continues the Cool3 setup with ``add_carpet_contacts`` (twist friction on the
  limbs, floor friction to 1.3) and ``foot_slip_twist``: a planted foot that
  slides or spins is penalised, so the feet are lifted and set into the sprawl
  instead of dragged or twisted into it.
  """
  cfg = unitree_g1_stand_crawl_cool_env_cfg(
    play=play, heat_weight=-0.8, ramp=(0, 1), moving_scale=0.5
  )
  add_carpet_contacts(cfg)
  continue_thermal_finetune(cfg)
  cfg.rewards["foot_slip_twist"] = RewardTermCfg(
    func=crawl_mdp.foot_slip_twist,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "body_names": FEET_29,
      "twist_weight": 0.2,
    },
  )
  return cfg


def unitree_g1_stand_crawl_push_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """StandCrawl-Push: Cool2 (the shipped stand <-> crawl) hardened against shoves.

  Under the chain test's hard randomization with pushes in the descent and the
  stand-up, Cool2 failed 9.4 % of loops (the original, stand_crawl_v3, 2.3 %):
  saving heat, it resisted shoves less, the pelvis touching down. Here the
  transitions' heat is free (only holds are priced: ``moving_scale`` 0), the
  robot is pushed every 2-4 s (was 3-6), and a forbidden-body touch above 10 N
  ends the episode (was 30 N; the chain test counts any touch).
  """
  cfg = unitree_g1_stand_crawl_cool_env_cfg(
    play=play, heat_weight=-0.8, ramp=(0, 1), moving_scale=0.0
  )
  if "push_robot" in cfg.events:
    cfg.events["push_robot"].interval_range_s = (2.0, 4.0)
  cfg.terminations["forbidden_contact"].params["force_threshold"] = 10.0
  return cfg


# StandCrawl-Robust: motor command lag up to 40 ms (5 ms physics steps).
STAND_CRAWL_LAG_STEPS = 8


def unitree_g1_stand_crawl_robust_env_cfg(
  play: bool = False,
  settle_weight: float = 0.0,
  stop_frames: bool = False,
  crawl_start_prob: float = 0.5,
  strict_touch: bool = True,
  correlated_gains: bool = False,
) -> ManagerBasedRlEnvCfg:
  """StandCrawl-Robust: StandCrawl-Push (the v3 stand <-> crawl) with wider
  margins, after the stand-up needed a hand on the robot (2026-10-30).

  The stress test (``scripts/standup_stress.py``) found the v3 policy solid
  against each single factor, but losing the top of the stand-up at 40 ms of
  motor command lag (STAND_HOLD tilts, 45 % stood) and under combined weak +
  heavy + slippery + lag worlds (48 %); the robot's legs also came in from the
  sprawl against ~1.5x the simulated resistance. Here, with the policy's
  interfaces unchanged:
  - lag 5-40 ms (was 0-20), motors down to 65 % (75 %), payload to +6 kg (+5),
    torso COM +-6 cm (5), joint friction to 1.5 Nm (1), gains +-30 % (25),
    arm ends to +0.5 kg (+0.2);
  - carpet contacts always (twist friction), floor friction 0.25-1.4;
  - bonuses for passing the switcher's gates while holding (``stand_gate`` at
    phi = 0, ``crawl_gate`` at phi = 1): the hand-over states themselves.
    In training they stay ~0: the exploration noise keeps some joint above the
    gates' speed limits.

  robust_v1 then fell far less (40 ms lag: 2 vs 33 %) but drifted at the top
  of the stand-up (base > 0.1 m/s), missing the stand gate. ``settle_weight``
  (Robust2) adds smooth versions of the gates: ``stand_settle`` (base still)
  and ``crawl_settle`` (crawl height and orientation).

  robust_v2 then brushed the right wrist brace on the floor in the hold after
  the creepy crawl stopped (27 of 1024 hard loops; v3 13), a start it never
  trained from: the crawl-pose starts were the crawl pose or trot-gait frames,
  20 cm below where the creepy crawl rests. ``stop_frames`` (Robust3): half of
  the episodes start on all fours, half of those in a recorded creepy or
  scuttle hand-over pose, and a forbidden touch above 3 N (was 10) ends the
  episode.

  robust_v3 then failed 2.4 % of the hard loops (v3 4.0, robust_v2 4.7) but
  sagged onto the torso going down in the weakest, heaviest worlds (36 % of
  "combo heavy" vs 2 %): with half the episodes starting on all fours, the
  descent trained less. Robust4 continues it with ``crawl_start_prob`` back at
  0.35.

  Both robust_v3 and robust_v4 sat the pelvis on the floor in the hold when
  every motor was 30 % softer (kp and kd x0.7: 74-79 % of loops; robust_v2
  1 %), the real robot's likely direction (its knees folded ~0.3 rad past
  target on 2026-10-29). Robust5 restarts from robust_v2 with the hand-over
  starts at 0.35, the 10 N touch limit (``strict_touch`` False), and
  ``correlated_gains``: one stiffness draw per robot for every motor
  (``pd_gains_correlated``), so whole-robot-soft worlds are trained.
  """
  cfg = unitree_g1_stand_crawl_push_env_cfg(play=play)
  add_carpet_contacts(cfg)
  continue_thermal_finetune(cfg)
  robot = cfg.scene.entities["robot"]
  art = robot.articulation
  assert art is not None
  robot.articulation = replace(
    art,
    actuators=tuple(
      replace(a, delay_max_lag=STAND_CRAWL_LAG_STEPS) for a in art.actuators
    ),
  )
  if "dr_arm_end_mass" in cfg.events:
    cfg.events["dr_arm_end_mass"].params["ranges"] = (-0.05, 0.5)
  robust = cfg.curriculum.get("robustness")
  if robust is not None:

    def const(v: float) -> tuple[float, float]:
      return (v, v)

    robust.params["ramp"] = replace(
      robust.params["ramp"],
      friction_low=const(0.25),
      friction_high=const(1.4),
      payload_low_kg=const(-3.0),
      payload_high_kg=const(6.0),
      com_offset_m=const(0.06),
      joint_friction_nm=const(1.5),
      pd_gain_scale=const(0.3),
      effort_scale_low=const(0.65),
      delay_max_steps=const(float(STAND_CRAWL_LAG_STEPS)),
      delay_min_steps=const(1.0),
    )
  cfg.rewards["stand_gate"] = RewardTermCfg(
    func=mdp.stand_gate, weight=1.0, params={"command_name": "phase"}
  )
  cfg.rewards["crawl_gate"] = RewardTermCfg(
    func=mdp.crawl_gate, weight=1.0, params={"command_name": "phase"}
  )
  if stop_frames and "reset_crawl_start" in cfg.events:
    cfg.events["reset_crawl_start"].params.update(
      prob=0.0 if play else crawl_start_prob,
      stop_frames=mdp.CRAWL_STOP_FRAMES_29,
      stop_frac=0.5,
    )
    if strict_touch:
      cfg.terminations["forbidden_contact"].params["force_threshold"] = 3.0
  if correlated_gains and "dr_motor_gains" in cfg.events:
    cfg.events["dr_motor_gains"].func = mdp.pd_gains_correlated
  if settle_weight:
    cfg.rewards["stand_settle"] = RewardTermCfg(
      func=mdp.stand_settle_exp,
      weight=settle_weight,
      params={"command_name": "phase"},
    )
    cfg.rewards["crawl_settle"] = RewardTermCfg(
      func=mdp.crawl_settle_exp,
      weight=settle_weight,
      params={"command_name": "phase"},
    )
  return cfg


##
# Take a knee and stand back up.
##

KNEE_LIMBS = r"^(left|right)_(ankle_roll_link|knee_link)$"
# Bodies that must never touch the ground while kneeling (episode ends): all but
# the feet and the kneeling (right) knee.
KNEEL_FORBIDDEN_BODIES = (
  r"^(pelvis|torso_link|waist_.*_link|.*_hip_.*_link|left_knee_link"
  r"|.*_shoulder_.*_link|.*_elbow_link|.*_wrist_.*_link|.*_wrist_brace"
  r"|.*_forearm_extension)$"
)
KNEEL_EPISODE_S = 25.0  # Stand + down + hold + up + stand: up to ~22.5 s.


def unitree_g1_kneel_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Kneel on the right knee by a slow reverse lunge, hold, and stand back up.

  See ``mdp/kneel.py``: a residual policy on a keyframe schedule whose phase
  runs both ways (0 standing, 1 kneeling) at a commanded, observed rate. The
  observation layout is the stand-to-crawl transition's (108): gravity, gyro,
  joints, last action, ``[phi, dphi/dt]``, reference minus joint positions.
  """
  joints = JOINTS_29
  command = "phase"
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_g1_kneel_robot_cfg_29()}
  cfg.scene.num_envs = 4096
  cfg.viewer.body_name = "torso_link"
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = (
    _ground_sensors(FEET_29, KNEE_LIMBS)
    + (
      ContactSensorCfg(
        name="forbidden_contact",
        primary=ContactMatch(
          mode="body", pattern=KNEEL_FORBIDDEN_BODIES, entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        history_length=3,
      ),
    )
    + _braced_sensors()
  )
  cfg.actions = {
    "joint_pos": mdp.ScheduledJointPositionActionCfg(
      entity_name="robot",
      actuator_names=joints,
      scale=TRANSITION_ACTION_SCALE,
      command_name=command,
    )
  }
  cfg.commands = {
    command: mdp.KneelPhaseCommandCfg(
      entity_name="robot", resampling_time_range=(1e3, 1e3)
    )
  }
  cfg.observations = _observations(joints, command)
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["phase"] = terms.pop("velocity_commands")
    terms["ref_joint_err"] = ObservationTermCfg(
      func=mdp.kneel.ref_joint_error, params={"command_name": command}
    )

  cfg.events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (0.0, 0.02),
          "yaw": (-3.14, 3.14),
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # After the standing reset: some episodes start on the knee.
    "reset_kneel_start": EventTermCfg(
      func=mdp.reset_kneel_start,
      mode="reset",
      params={"command_name": command, "prob": 0.0 if play else 0.3},
    ),
    "encoder_bias": cfg.events["encoder_bias"],
  }

  jcfg = SceneEntityCfg("robot", joint_names=joints)
  k = mdp.kneel
  c = {"command_name": command}
  cfg.rewards = {
    # Follow the schedules.
    "ref_height": RewardTermCfg(
      func=k.ref_height_exp, weight=1.0, params={**c, "std": 0.06}
    ),
    "ref_orientation": RewardTermCfg(
      func=k.ref_gravity_exp, weight=1.0, params={**c, "std": 0.3}
    ),
    "ref_joint_pose": RewardTermCfg(
      func=k.ref_joint_pose_exp, weight=1.5, params={**c, "std": 0.4}
    ),
    "ref_knee_height": RewardTermCfg(
      func=mdp.ref_knee_height_exp, weight=1.0, params={**c, "std": 0.05}
    ),
    "ref_rfoot_height": RewardTermCfg(
      func=mdp.ref_rfoot_height_exp, weight=0.5, params={**c, "std": 0.04}
    ),
    # The planned contacts, and the end poses held.
    "support_contacts": RewardTermCfg(
      func=mdp.support_contacts,
      weight=0.3,
      params={**c, "sensor_name": "feet_ground_contact"},
    ),
    "end_pose": RewardTermCfg(
      func=mdp.end_pose_exp, weight=3.0, params={**c, "std": 0.3}
    ),
    # Slow, soft, still, no sliding.
    "hold_still": RewardTermCfg(func=k.hold_still, weight=-0.02, params=c),
    "planted_slip": RewardTermCfg(
      func=mdp.planted_slip,
      weight=-0.5,
      params={**c, "sensor_name": "feet_ground_contact"},
    ),
    "joint_vel_cap": RewardTermCfg(
      func=mdp.joint_vel_cap, weight=-0.5, params={"asset_cfg": jcfg, "limit": 2.5}
    ),
    "base_speed_cap": RewardTermCfg(
      func=mdp.base_speed_cap, weight=-1.0, params={"limit": 0.4}
    ),
    "contact_impact": RewardTermCfg(
      func=mdp.contact_impact,
      weight=-2.0,
      params={"sensor_name": "body_ground_contacts"},
    ),
    "undesired_body_contacts": RewardTermCfg(
      func=crawl_mdp.undesired_ground_contacts,
      weight=-1.0,
      params={
        "sensor_name": "body_ground_contacts",
        "excluded_body_names": FEET_29 + ("right_knee_link",),
        "threshold": 1.0,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": jcfg}
    ),
    "action_rate_l2": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  _add_torque_safety(cfg)

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "tilted": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    ),
    "forbidden_contact": TerminationTermCfg(
      func=velocity_mdp.illegal_contact,
      params={"sensor_name": "forbidden_contact", "force_threshold": 30.0},
    ),
  }
  cfg.curriculum = {}
  cfg.metrics = {"mean_action_acc": cfg.metrics["mean_action_acc"]}
  _add_robust_training(
    cfg,
    command,
    play,
    mdp.KNEEL_RAMP,
    curriculum=mdp.kneel_robustness,
    push_interval_s=(2.0, 5.0),
    success_pose_err=0.25,
    # A 25 s episode with two transitions: the walk's 5% fall gate kept the
    # level at 0 (falls plateaued at 10-14% with exploration noise).
    promote_max_flip_rate=0.15,
    demote_flip_rate=0.30,
  )
  _add_metrics(cfg, FEET_29)
  cfg.metrics["gait/knee_on_ground_frac"] = MetricsTermCfg(
    func=metrics.limb_contact_frac,
    params={"sensor_name": "feet_ground_contact", "body_names": ("right_knee_link",)},
    reduce="mean",
  )

  cfg.sim.nconmax = 120
  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  cfg.episode_length_s = KNEEL_EPISODE_S
  if play:
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg
