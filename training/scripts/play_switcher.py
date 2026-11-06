"""Walk, go down into the crawl pose, crawl: three policies, one robot, one sim.

The braced 29-DoF G1 (``g1_crawl/robot29.py``) runs under a small state machine
that mirrors what the real robot's controller has to do:

  WALK  --C-->  TRANSITION  --phase 1-->  HOLD  --gate-->  CRAWL

* WALK: the velocity policy (``Mjlab-Walk-Flat-Unitree-G1-29dof``). Keys W/S
  forward speed, Q/E sideways, A/D turn, X stop.
* TRANSITION: the stand-to-crawl policy (``Mjlab-StandToCrawl-Unitree-G1-29dof``)
  tracks its slow schedule; the phase advances over ``--transition-s`` seconds.
  Without a transition checkpoint the schedule alone is played (zero residual).
* HOLD: the transition policy holds the crawl pose while the gate is checked:
  joints still, pelvis height and orientation at the crawl pose, both feet and
  both extension tips on the ground, for ``--dwell-s`` seconds.
* CRAWL: the crawl policy (``Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof``), with the
  same keys plus H to cycle the commanded step height.

Every hand-over blends the PD joint targets over ``--blend-s`` seconds, and
every policy runs in shadow mode (computing its own actions from the live
state) before it is in charge, so its "last action" input is always its own.
Other keys: P push the robot, F freeze (hold the current pose), Enter reset.

Each policy's observations are built here from the robot state, in the layout
its task trained with (see ``_WalkPolicy``, ``_TransitionPolicy``,
``_CrawlPolicy``): the same code a deployment needs, against the same ONNX
inputs. The action term of this script's environment is a raw joint-target
pass-through.

Headless evaluation of the whole chain, ``N`` robots in parallel::

  uv run python scripts/play_switcher.py --eval 256 --robust

reports how many reached the crawl pose, passed the gate and crawled off, with
peak motor and braced-wrist loads, joint speeds and impacts along the way.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch
from tensordict import TensorDict

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.g1_crawl.config.g1.env_cfgs import CRAWL_ZONES_CFG
from mjlab.tasks.g1_crawl.robot29 import BRACED_LOADS_29, MOTORS_29
from mjlab.tasks.g1_walk.config.g1.env_cfgs import (
  FORBIDDEN_BODIES,
  TRANSITION_ACTION_CLIP,
  TRANSITION_ACTION_SCALE,
  _action_scale_29,
  _add_robust_training,
  unitree_g1_walk_env_cfg,
)
from mjlab.tasks.g1_walk.mdp import TRANSITION_RAMP, TransitionSchedule
from mjlab.tasks.g1_walk.robot import (
  CRAWL_POSE_29,
  JOINTS_29,
  STAND_POSE_29,
)
from mjlab.tasks.registry import load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import (
  KEY_A,
  KEY_C,
  KEY_D,
  KEY_E,
  KEY_F,
  KEY_H,
  KEY_P,
  KEY_Q,
  KEY_S,
  KEY_W,
  KEY_X,
)

WALK_TASK = "Mjlab-Walk-Flat-Unitree-G1-29dof"
TRANSITION_TASK = "Mjlab-StandToCrawl-Unitree-G1-29dof"
CRAWL_TASK = "Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof"
CRAWL_CHECKPOINT = (
  "logs/rsl_rl/g1_crawl_29dof/2026-10-12_17-55-24_gen2_flat_16k/model_49999.pt"
)
LIMBS = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_forearm_extension",
  "right_forearm_extension",
)
# Pelvis height over the mean of the four limb links at the crawl pose (flat).
CRAWL_ABOVE_LIMBS = 0.151
WALK, TRANSITION, HOLD, CRAWL, FREEZE = range(5)
MODE_NAMES = ("WALK", "TRANSITION", "HOLD", "CRAWL", "FREEZE")


def latest_checkpoint(experiment: str) -> Path | None:
  runs = sorted(Path("logs/rsl_rl", experiment).glob("*/model_*.pt"))
  if not runs:
    return None
  return max(runs, key=lambda p: (p.stat().st_mtime, int(p.stem.split("_")[1])))


def load_actor(task_id: str, checkpoint: str | Path, device: str):
  """The trained actor (with its observation normalizer) as obs -> mean action."""
  from rsl_rl.models import MLPModel

  agent_cfg = load_rl_cfg(task_id)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg), task_id
  state = torch.load(checkpoint, map_location=device, weights_only=False)
  sd = state["actor_state_dict"]
  obs_dim = sd["mlp.0.weight"].shape[1]
  act_dim = sd["distribution.std_param"].shape[0]
  dummy = TensorDict({"actor": torch.zeros(1, obs_dim, device=device)}, batch_size=[1])
  actor = MLPModel(
    dummy,
    {"actor": ["actor"]},
    "actor",
    act_dim,
    hidden_dims=agent_cfg.actor.hidden_dims,
    activation=agent_cfg.actor.activation,
    obs_normalization=agent_cfg.actor.obs_normalization,
    distribution_cfg=agent_cfg.actor.distribution_cfg,
  ).to(device)
  actor.load_state_dict(sd, strict=True)
  actor.eval()
  clip = agent_cfg.clip_actions

  def run(obs: torch.Tensor) -> torch.Tensor:
    assert obs.shape[1] == obs_dim, (task_id, obs.shape, obs_dim)
    act = actor(TensorDict({"actor": obs}, batch_size=[obs.shape[0]]))
    return act.clamp(-clip, clip) if clip else act

  return run, obs_dim


class _Policy:
  """A trained policy with its own observation layout and action mapping."""

  def __init__(self, n: int, device: str) -> None:
    self.last_action = torch.zeros(n, len(JOINTS_29), device=device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.last_action[env_ids] = 0.0

  def base(self, robot) -> tuple[torch.Tensor, torch.Tensor]:
    return robot.data.projected_gravity_b, robot.data.root_link_ang_vel_b

  def targets(self, robot) -> torch.Tensor:
    raise NotImplementedError


class _WalkPolicy(_Policy):
  """Actor obs: gravity, gyro, [vx, vy, wz], q - stand, qd, last action."""

  def __init__(self, n, device, actor):
    super().__init__(n, device)
    self.actor = actor
    self.command = torch.zeros(n, 3, device=device)
    self.q_default = torch.tensor([STAND_POSE_29[j] for j in JOINTS_29], device=device)
    scale = _action_scale_29()
    self.scale = torch.tensor([scale[j] for j in JOINTS_29], device=device)

  def targets(self, robot):
    g, w = self.base(robot)
    obs = torch.cat(
      [
        g,
        w,
        self.command,
        robot.data.joint_pos - self.q_default,
        robot.data.joint_vel,
        self.last_action,
      ],
      dim=1,
    )
    self.last_action = self.actor(obs)
    return self.q_default + self.scale * self.last_action


class _TransitionPolicy(_Policy):
  """Actor obs: gravity, gyro, q - stand, qd, last action, [phi, 1/T], ref - q.

  (The training config appends the phase and the reference error after the
  velocity-task terms; see ``unitree_g1_stand_to_crawl_env_cfg``.)
  Target = scheduled reference(phi) + TRANSITION_ACTION_SCALE * residual.
  """

  def __init__(self, n, device, actor, duration_s: float):
    super().__init__(n, device)
    self.actor = actor  # None: play the schedule with zero residual.
    self.schedule = TransitionSchedule(device)
    self.phase = torch.zeros(n, device=device)
    self.inv_t = torch.full((n,), 1.0 / duration_s, device=device)
    self.q_stand = self.schedule.q_stand

  def reset(self, env_ids):
    super().reset(env_ids)
    self.phase[env_ids] = 0.0

  def reference(self) -> torch.Tensor:
    return self.schedule.joint_pos(self.phase)

  def targets(self, robot):
    ref = self.reference()
    if self.actor is None:
      return ref
    g, w = self.base(robot)
    q = robot.data.joint_pos
    obs = torch.cat(
      [
        g,
        w,
        q - self.q_stand,
        robot.data.joint_vel,
        self.last_action,
        self.phase[:, None],
        self.inv_t[:, None],
        ref - q,
      ],
      dim=1,
    )
    self.last_action = self.actor(obs).clamp(
      -TRANSITION_ACTION_CLIP, TRANSITION_ACTION_CLIP
    )
    return ref + TRANSITION_ACTION_SCALE * self.last_action


class _CrawlPolicy(_Policy):
  """Actor obs: gravity, gyro, 7 commands, q - crawl pose, qd, last action."""

  STEP_HEIGHTS = (0.03, 0.08, 0.12)

  def __init__(self, n, device, actor):
    super().__init__(n, device)
    self.actor = actor
    self.command = torch.zeros(n, 7, device=device)
    self.command[:, 3] = self.STEP_HEIGHTS[0]
    self.q_default = torch.tensor([CRAWL_POSE_29[j] for j in JOINTS_29], device=device)

  def targets(self, robot):
    g, w = self.base(robot)
    obs = torch.cat(
      [
        g,
        w,
        self.command,
        robot.data.joint_pos - self.q_default,
        robot.data.joint_vel,
        self.last_action,
      ],
      dim=1,
    )
    self.last_action = self.actor(obs)
    return self.q_default + 0.5 * self.last_action


class Switcher:
  """The state machine; acts as the viewer's policy (obs ignored, state read)."""

  def __init__(self, env: ManagerBasedRlEnv, args, device: str) -> None:
    self.env = env
    self.robot = env.scene["robot"]
    assert tuple(self.robot.joint_names) == JOINTS_29, self.robot.joint_names
    n = env.num_envs
    self.n, self.device, self.dt = n, device, env.step_dt
    self.args = args
    walk_actor, _ = load_actor(WALK_TASK, args.walk_checkpoint, device)
    crawl_actor, _ = load_actor(CRAWL_TASK, args.crawl_checkpoint, device)
    tr_actor = (
      load_actor(TRANSITION_TASK, args.transition_checkpoint, device)[0]
      if args.transition_checkpoint
      else None
    )
    self.walk = _WalkPolicy(n, device, walk_actor)
    self.transition = _TransitionPolicy(n, device, tr_actor, args.transition_s)
    self.crawl = _CrawlPolicy(n, device, crawl_actor)
    self.mode = torch.full((n,), WALK, dtype=torch.long, device=device)
    self.time = torch.zeros(n, device=device)
    self.mode_t0 = torch.zeros(n, device=device)
    self.dwell = torch.zeros(n, device=device)
    self.blend_t0 = torch.full((n,), -1e9, device=device)
    self.blend_from = torch.zeros(n, len(JOINTS_29), device=device)
    self.applied = torch.zeros(n, len(JOINTS_29), device=device)
    self.frozen = torch.zeros(n, len(JOINTS_29), device=device)
    self.g_crawl = self.transition.schedule.g_crawl
    sensor = env.scene["limb_contacts"]
    lookup = {name: i for i, name in enumerate(sensor.primary_names)}
    self.limb_ids = [lookup[name] for name in LIMBS]
    self.limb_body_ids = self.robot.find_bodies(LIMBS, preserve_order=True)[0]
    self.forbidden_names = list(env.scene["forbidden_contact"].primary_names)
    kp_kd_peak = torch.tensor([MOTORS_29[j] for j in JOINTS_29], device=device)
    self.kp, self.kd, self.peak = kp_kd_peak[:, 0], kp_kd_peak[:, 1], kp_kd_peak[:, 2]
    self.braced = [
      (
        env.scene[f"robot/{j.removesuffix('_joint')}_torque"],
        torch.tensor(a, device=device),
      )
      for j, (_, a) in BRACED_LOADS_29.items()
    ]
    self.stats = self._new_stats()

  # -- state ----------------------------------------------------------------

  def _new_stats(self) -> dict[str, torch.Tensor]:
    n, dev = self.n, self.device
    peaks = ("max_motor_ratio", "max_braced_nm", "max_joint_speed", "max_force_rise")
    flags = ("fell", "forbidden")
    return {
      # Peaks and failures, separately while going down (+hold) and crawling.
      **{f"{k}_{g}": torch.zeros(n, device=dev) for k in peaks for g in ("tr", "cr")},
      **{
        f"{k}_{g}": torch.zeros(n, dtype=torch.bool, device=dev)
        for k in flags
        for g in ("tr", "cr")
      },
      "fell": torch.zeros(n, dtype=torch.bool, device=dev),
      "forbidden": torch.zeros(n, dtype=torch.bool, device=dev),
      "forbidden_body": torch.full((n,), -1, dtype=torch.long, device=dev),
      "hold_pose": torch.zeros(n, len(JOINTS_29), device=dev),
      "hold_height": torch.zeros(n, device=dev),
      "forbidden_phase": torch.zeros(n, device=dev),
      "reached_hold": torch.zeros(n, dtype=torch.bool, device=dev),
      "switched": torch.zeros(n, dtype=torch.bool, device=dev),
      "switch_time": torch.zeros(n, device=dev),
      "crawl_start_x": torch.zeros(n, 2, device=dev),
    }

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
    for p in (self.walk, self.transition, self.crawl):
      p.reset(ids)
    self.walk.command[ids] = 0.0
    self.crawl.command[ids] = 0.0
    self.crawl.command[ids, 3] = _CrawlPolicy.STEP_HEIGHTS[0]
    self.mode[ids] = WALK
    self.time[ids] = 0.0
    self.mode_t0[ids] = 0.0
    self.dwell[ids] = 0.0
    self.blend_t0[ids] = -1e9
    self.applied[ids] = self.robot.data.joint_pos[ids]
    for v in self.stats.values():
      v[ids] = 0

  def set_mode(self, env_ids: torch.Tensor, mode: int) -> None:
    if len(env_ids) == 0:
      return
    self.mode[env_ids] = mode
    self.mode_t0[env_ids] = self.time[env_ids]
    self.dwell[env_ids] = 0.0
    self.blend_t0[env_ids] = self.time[env_ids]
    self.blend_from[env_ids] = self.applied[env_ids]
    if mode == TRANSITION:
      self.transition.phase[env_ids] = 0.0
    if mode == FREEZE:
      self.frozen[env_ids] = self.robot.data.joint_pos[env_ids]
    if mode == CRAWL:
      self.stats["switched"][env_ids] = True
      self.stats["switch_time"][env_ids] = self.time[env_ids]
      self.stats["crawl_start_x"][env_ids] = self.robot.data.root_link_pos_w[
        env_ids, :2
      ]

  def start_transition(self, env_ids: torch.Tensor | None = None) -> None:
    ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
    ids = ids[self.mode[ids] == WALK]
    self.walk.command[ids] = 0.0
    self.set_mode(ids, TRANSITION)

  def push(self, speed: float) -> None:
    vel = self.robot.data.root_link_vel_w.clone()
    kick = torch.randn(self.n, 2, device=self.device)
    kick = kick / kick.norm(dim=1, keepdim=True).clamp(min=1e-6) * speed
    vel[:, :2] += kick
    self.robot.write_root_link_velocity_to_sim(vel)

  # -- gate ------------------------------------------------------------------

  def pelvis_above_limbs(self) -> torch.Tensor:
    """Pelvis height over the mean of the four limb links: ground-relative."""
    d = self.robot.data
    limbs = d.body_link_pos_w[:, self.limb_body_ids, 2].mean(dim=1)
    return d.root_link_pos_w[:, 2] - limbs

  def gate(self) -> torch.Tensor:
    d = self.robot.data
    still = d.joint_vel.abs().amax(dim=1) < self.args.gate_joint_speed
    height = (self.pelvis_above_limbs() - CRAWL_ABOVE_LIMBS).abs() < 0.03
    orient = (d.projected_gravity_b - self.g_crawl).norm(dim=1) < 0.15
    found = self.env.scene["limb_contacts"].data.found
    assert found is not None
    contacts = (found[:, self.limb_ids] > 0).all(dim=1)
    return still & height & orient & contacts

  # -- step ------------------------------------------------------------------

  def __call__(self, obs) -> torch.Tensor:
    del obs
    robot, d = self.robot, self.robot.data
    self.time += self.dt
    # Every policy observes and acts every step (shadow mode when not in charge).
    t_walk = self.walk.targets(robot)
    t_tr = self.transition.targets(robot)
    t_crawl = self.crawl.targets(robot)
    target = torch.where(
      (self.mode == WALK)[:, None],
      t_walk,
      torch.where((self.mode == CRAWL)[:, None], t_crawl, t_tr),
    )
    target = torch.where((self.mode == FREEZE)[:, None], self.frozen, target)
    # Blend at hand-overs.
    s = ((self.time - self.blend_t0) / self.args.blend_s).clamp(0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    self.applied = self.blend_from + (target - self.blend_from) * s[:, None]

    # Advance the state machine.
    in_tr = self.mode == TRANSITION
    self.transition.phase[in_tr] += self.dt * self.transition.inv_t[in_tr]
    done = in_tr & (self.transition.phase >= 1.0)
    self.transition.phase.clamp_(max=1.0)
    ids = done.nonzero().squeeze(1)
    self.stats["reached_hold"][ids] = True
    self.mode[ids] = HOLD  # No blend: the same policy keeps control.
    self.mode_t0[ids] = self.time[ids]
    self.dwell[ids] = 0.0
    hold = self.mode == HOLD
    self.stats["hold_pose"][hold] = d.joint_pos[hold]
    self.stats["hold_height"][hold] = self.pelvis_above_limbs()[hold]
    ok = self.gate()
    self.dwell = torch.where(
      hold & ok, self.dwell + self.dt, torch.zeros_like(self.dwell)
    )
    self.set_mode(
      (hold & (self.dwell >= self.args.dwell_s)).nonzero().squeeze(1), CRAWL
    )

    # Safety bookkeeping, attributed to the descent (+hold) or to the crawl.
    demand = self.kp * (self.applied - d.joint_pos) - self.kd * d.joint_vel
    ratio = (demand.abs() / self.peak).amax(dim=1)
    braced = torch.zeros(self.n, device=self.device)
    for sensor, axis in self.braced:
      braced = torch.maximum(braced, (sensor.data @ axis).abs())
    speed = d.joint_vel.abs().amax(dim=1)
    rise = torch.zeros(self.n, device=self.device)
    hist = self.env.scene["body_ground_contacts"].data.force_history
    if hist is not None:
      force = hist.norm(dim=-1)
      rise = torch.relu(force[:, :, 1:] - force[:, :, :-1]).amax(dim=2).amax(dim=1)
    g = d.projected_gravity_b
    fell = (g[:, 2] > 0.5) | (g[:, 1].abs() > 0.8) | (g[:, 0] < -0.6)
    forbidden = self.env.scene["forbidden_contact"].data.found
    assert forbidden is not None
    hit = forbidden > 0
    st = self.stats
    first = hit.any(dim=1) & ~st["forbidden"]
    st["forbidden_body"][first] = hit[first].float().argmax(dim=1)
    st["forbidden_phase"][first] = self.transition.phase[first]
    st["fell"] |= fell
    st["forbidden"] |= hit.any(dim=1)
    going_down = (self.mode == TRANSITION) | (self.mode == HOLD)
    for tag, mask in (("tr", going_down), ("cr", self.mode == CRAWL)):
      for key, value in (
        ("max_motor_ratio", ratio),
        ("max_braced_nm", braced),
        ("max_joint_speed", speed),
        ("max_force_rise", rise),
      ):
        name = f"{key}_{tag}"
        st[name] = torch.where(mask, torch.maximum(st[name], value), st[name])
      st[f"fell_{tag}"] |= fell & mask
      st[f"forbidden_{tag}"] |= hit.any(dim=1) & mask
    return self.applied

  def status(self) -> str:
    i = 0
    mode = MODE_NAMES[int(self.mode[i])]
    txt = f"\r[{mode:10s}] t={float(self.time[i]):6.1f}s"
    if mode == "WALK":
      c = self.walk.command[i].tolist()
      txt += f"  cmd fwd {c[0]:+.2f} lat {c[1]:+.2f} turn {c[2]:+.2f}"
    elif mode in ("TRANSITION", "HOLD"):
      txt += f"  phase {float(self.transition.phase[i]):.2f}  gate {'ok' if bool(self.gate()[i]) else '--'}  dwell {float(self.dwell[i]):.1f}s"
    elif mode == "CRAWL":
      c = self.crawl.command[i].tolist()
      txt += f"  cmd fwd {c[0]:+.2f} lat {c[1]:+.2f} turn {c[2]:+.2f} step {c[3]:.2f}"
    st = self.stats
    tag = "cr" if mode == "CRAWL" else "tr"
    txt += (
      f"  | peak motor {float(st[f'max_motor_ratio_{tag}'][i]):.2f}x"
      f"  braced {float(st[f'max_braced_nm_{tag}'][i]):.1f}Nm"
      f"  qd {float(st[f'max_joint_speed_{tag}'][i]):.1f}rad/s"
    )
    return txt + "   "


def make_env_cfg(n: int, robust: bool, no_delay: bool, rough: bool = False):
  cfg = unitree_g1_walk_env_cfg(play=True)
  cfg.scene.num_envs = n
  if rough:
    # The crawl task's six zones (stairs, slopes, boxes, meadows), spawned anywhere.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = replace(
      CRAWL_ZONES_CFG,
      curriculum=True,
      num_rows=4,
      border_width=10.0,
      difficulty_range=(0.4, 1.0),
    )
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.sim.nconmax = 220
  cfg.commands = {}
  cfg.rewards = {}
  cfg.terminations = {}
  cfg.curriculum = {}
  cfg.metrics = {}
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={"projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity)},
      concatenate_terms=True,
      enable_corruption=False,
    )
  }
  # Raw joint targets in: the switcher computes every policy's mapping itself.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINTS_29,
      scale=1.0,
      offset=0.0,
      use_default_offset=False,
    )
  }
  cfg.events = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}},
    ),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }
  cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="limb_contacts",
      primary=ContactMatch(
        mode="body",
        pattern=r"^(left|right)_(ankle_roll_link|forearm_extension)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=3,
    ),
    ContactSensorCfg(
      name="forbidden_contact",
      primary=ContactMatch(mode="body", pattern=FORBIDDEN_BODIES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=3,
    ),
  )
  if robust:
    # Level-1 randomization of the training curricula, re-drawn every reset.
    _add_robust_training(cfg, "none", play=False, ramp=TRANSITION_RAMP)
    cfg.curriculum = {}
    cfg.events.pop("push_robot")
    r = TRANSITION_RAMP
    at = lambda knob: knob[1]  # noqa: E731  Level 1.
    p = {k: cfg.events[k].params for k in cfg.events if k.startswith("dr_")}
    p["dr_friction"]["ranges"] = (at(r.friction_low), at(r.friction_high))
    p["dr_body_mass"]["ranges"] = (1 - at(r.mass_scale), 1 + at(r.mass_scale))
    p["dr_torso_payload"]["ranges"] = (at(r.payload_low_kg), at(r.payload_high_kg))
    p["dr_torso_com"]["ranges"] = (-at(r.com_offset_m), at(r.com_offset_m))
    p["dr_joint_friction"]["ranges"] = (0.0, at(r.joint_friction_nm))
    p["dr_armature"]["ranges"] = (1 - at(r.armature_scale), 1 + at(r.armature_scale))
    p["dr_motor_gains"]["kp_range"] = (1 - at(r.pd_gain_scale), 1 + at(r.pd_gain_scale))
    p["dr_motor_gains"]["kd_range"] = (1 - at(r.pd_gain_scale), 1 + at(r.pd_gain_scale))
    p["dr_motor_strength"]["effort_limit_range"] = (at(r.effort_scale_low), 1.0)
  if no_delay:
    robot_cfg = cfg.scene.entities["robot"]
    assert robot_cfg.articulation is not None
    robot_cfg.articulation = replace(
      robot_cfg.articulation,
      actuators=tuple(
        replace(a, delay_min_lag=0, delay_max_lag=0)
        for a in robot_cfg.articulation.actuators
      ),
    )
  return cfg


def evaluate(switcher: Switcher, wenv, args) -> None:
  """Stand 2 s, go down, hold, switch, crawl forward 5 s; report."""
  n, dt = switcher.n, switcher.dt
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  switcher.reset()
  # Optional walking prelude (video): forward, turn, sideways, stop, then C.
  prelude = (
    (
      ((0.5, 0.0, 0.0), 3.0),
      ((0.5, 0.0, 0.6), 2.0),
      ((0.0, 0.4, 0.0), 2.0),
      ((0.0, 0.0, -0.6), 2.0),
      ((0.0, 0.0, 0.0), 1.5),
    )
    if args.walk_first
    else ()
  )
  t_c = 2.0 + sum(seconds for _, seconds in prelude)
  steps_stand = int(t_c / dt)
  steps_total = int((t_c + args.transition_s + args.hold_timeout_s + 5.0) / dt)
  pushed = False
  frames: list = []
  video: list = []
  frame_every = max(1, int(0.5 / dt))
  with torch.inference_mode():
    for step in range(steps_total):
      t = step * dt
      if args.video and step % 2 == 0:  # 25 fps
        frame = wenv.unwrapped.render()
        if frame is not None:
          video.append(frame)
      if args.frames and step >= steps_stand and step % frame_every == 0:
        frame = wenv.unwrapped.render()
        if frame is not None:
          frames.append((t - t_c, frame))
      if prelude and step < steps_stand:
        elapsed, cmd = t - 2.0, (0.0, 0.0, 0.0)
        for c, seconds in prelude:
          if elapsed < seconds:
            cmd = c if elapsed >= 0 else (0.0, 0.0, 0.0)
            break
          elapsed -= seconds
        (
          switcher.walk.command[:, 0],
          switcher.walk.command[:, 1],
          switcher.walk.command[:, 2],
        ) = cmd
      if step == steps_stand:
        switcher.start_transition()
      if args.push and not pushed and t >= t_c + 0.5 * args.transition_s:
        switcher.push(args.push)
        pushed = True
      crawling = switcher.mode == CRAWL
      since = switcher.time - switcher.stats["switch_time"]
      switcher.crawl.command[:, 0] = torch.where(
        crawling & (since > 1.0), torch.full_like(since, 0.5), torch.zeros_like(since)
      )
      obs = wenv.step(switcher(obs))[0]
  st = switcher.stats
  crawled = (switcher.robot.data.root_link_pos_w[:, :2] - st["crawl_start_x"]).norm(
    dim=1
  )
  ok = st["switched"] & ~st["fell"] & ~st["forbidden"]
  moved = ok & (crawled > 0.5)
  pct = lambda m: 100.0 * float(m.float().mean())  # noqa: E731
  bodies = st["forbidden_body"][st["forbidden"]].tolist()
  phases = st["forbidden_phase"][st["forbidden"]].tolist()
  by_body: dict[str, list[float]] = {}
  for b, ph in zip(bodies, phases, strict=True):
    by_body.setdefault(switcher.forbidden_names[b], []).append(ph)
  held = st["reached_hold"] & ~st["fell"]
  q_crawl = switcher.crawl.q_default
  joint_err = (
    (st["hold_pose"][held] - q_crawl).abs().mean(dim=0) if held.any() else None
  )
  worst_joints = ""
  if joint_err is not None:
    order = joint_err.argsort(descending=True)[:6].tolist()
    worst_joints = ", ".join(
      f"{JOINTS_29[i].removesuffix('_joint')} {float(joint_err[i]):.2f}" for i in order
    )
    pose_rms = float(
      (st["hold_pose"][held] - q_crawl).square().mean(dim=1).sqrt().median()
    )
    height = float(st["hold_height"][held].median())
  touched = ", ".join(
    f"{name} x{len(ph)} (phase {min(ph):.2f}-{max(ph):.2f})"
    for name, ph in sorted(by_body.items(), key=lambda kv: -len(kv[1]))
  )

  def peaks(tag: str) -> str:
    def q(key: str, p: float) -> float:
      return float(st[f"{key}_{tag}"].quantile(p))

    return (
      f"motor demand/limit {q('max_motor_ratio', 0.5):.2f} (p95 "
      f"{q('max_motor_ratio', 0.95):.2f}), braced wrist {q('max_braced_nm', 0.5):.1f} Nm "
      f"(p95 {q('max_braced_nm', 0.95):.1f}), joint speed {q('max_joint_speed', 0.5):.1f} "
      f"rad/s (p95 {q('max_joint_speed', 0.95):.1f}), ground-force rise "
      f"{q('max_force_rise', 0.5):.0f} N/step (p95 {q('max_force_rise', 0.95):.0f})"
    )

  if args.video and video:
    import imageio

    imageio.mimwrite(args.video, video, fps=25, codec="libx264", quality=8)
    print(f"video: {args.video} ({len(video)} frames)")
  if args.frames and frames:
    from PIL import Image, ImageDraw

    tiles = []
    for t, frame in frames[:32]:
      im = Image.fromarray(frame)
      draw = ImageDraw.Draw(im)
      draw.rectangle([0, 0, 90, 18], fill=(20, 20, 20))
      draw.text((4, 3), f"{t:+.1f} s", fill=(255, 255, 255))
      tiles.append(im)
    cols = 8
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h))
    for i, im in enumerate(tiles):
      sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(args.frames)
    print(f"frames: {args.frames}")
  switched = st["switched"]
  switch_time = (
    float((st["switch_time"][switched] - t_c).median())
    if switched.any()
    else float("nan")
  )
  hold_line = (
    f"  pose at hold: RMS err {pose_rms:.2f} rad, pelvis {height:.3f} m; worst joints: "
    f"{worst_joints}\n"
    if worst_joints
    else ""
  )
  print(
    f"\n=== chain evaluation, {n} robots, {'randomized' if args.robust else 'nominal'}"
    f"{', pushed' if args.push else ''} ===\n"
    "going down (descent + hold):\n"
    f"  schedule completed (phase 1):        {pct(st['reached_hold']):5.1f}%\n"
    f"  fell:                                {pct(st['fell_tr']):5.1f}%\n"
    f"  forbidden body touched the ground:   {pct(st['forbidden_tr']):5.1f}%"
    f"{'  first: ' + touched if touched else ''}\n"
    f"{hold_line}"
    f"  peaks (median over robots): {peaks('tr')}\n"
    "hand-over:\n"
    f"  gate passed, crawl policy took over: {pct(switched):5.1f}%   median "
    f"{switch_time:.1f} s after 'C'\n"
    "crawling (4 s at 0.5 m/s after the hand-over):\n"
    f"  crawled > 0.5 m:                     {pct(moved):5.1f}%\n"
    f"  fell / forbidden contact:            {pct(st['fell_cr']):5.1f}% / "
    f"{pct(st['forbidden_cr']):5.1f}%\n"
    f"  peaks (median over robots): {peaks('cr')}\n"
  )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--walk-checkpoint", default=None)
  ap.add_argument(
    "--transition-checkpoint", default="latest", help="'none' plays the schedule"
  )
  ap.add_argument("--crawl-checkpoint", default=CRAWL_CHECKPOINT)
  ap.add_argument("--transition-s", type=float, default=6.0, help="descent duration")
  ap.add_argument("--blend-s", type=float, default=0.5)
  ap.add_argument("--dwell-s", type=float, default=1.0)
  ap.add_argument("--hold-timeout-s", type=float, default=6.0)
  ap.add_argument("--gate-joint-speed", type=float, default=0.5)
  ap.add_argument(
    "--eval", type=int, default=0, help="headless chain test with N robots"
  )
  ap.add_argument("--robust", action="store_true", help="level-1 randomization")
  ap.add_argument(
    "--push", type=float, default=0.0, help="eval: kick (m/s) mid-descent"
  )
  ap.add_argument("--no-delay", action="store_true", help="zero motor command delay")
  ap.add_argument(
    "--terrain",
    choices=("flat", "rough"),
    default="flat",
    help="rough: the crawl task's stairs, slopes, boxes and meadows",
  )
  ap.add_argument("--frames", default=None, help="eval: save a frame strip (PNG)")
  ap.add_argument("--video", default=None, help="eval: save an MP4 of robot 0")
  ap.add_argument(
    "--walk-first", action="store_true", help="eval: walk, turn and sidestep before C"
  )
  args = ap.parse_args()
  for name, exp in (
    ("walk_checkpoint", "g1_walk_29dof"),
    ("transition_checkpoint", "g1_stand_to_crawl_29dof"),
  ):
    value = getattr(args, name)
    if value in (None, "latest"):
      setattr(args, name, latest_checkpoint(exp))
    elif value == "none":
      setattr(args, name, None)
  assert args.walk_checkpoint, (
    "no walk checkpoint found under logs/rsl_rl/g1_walk_29dof"
  )
  print(
    f"walk: {args.walk_checkpoint}\ntransition: {args.transition_checkpoint or 'schedule only'}\ncrawl: {args.crawl_checkpoint}"
  )

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  n = args.eval or 1
  cfg = make_env_cfg(n, args.robust, args.no_delay, args.terrain == "rough")
  cfg.viewer.height, cfg.viewer.width = 360, 480
  env = ManagerBasedRlEnv(
    cfg=cfg,
    device=device,
    render_mode="rgb_array" if (args.frames or args.video) else None,
  )
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  switcher = Switcher(env, args, device)
  if args.eval:
    evaluate(switcher, wenv, args)
    env.close()
    return

  def on_key(key: int) -> None:
    w, c = switcher.walk.command, switcher.crawl.command
    mode = int(switcher.mode[0])
    if key == KEY_C:
      switcher.start_transition()
    elif key == KEY_F:
      switcher.set_mode(torch.arange(n, device=device), FREEZE)
    elif key == KEY_P:
      switcher.push(0.5)
    elif key == KEY_X:
      w[:] = 0.0
      c[:, :3] = 0.0
    elif mode == WALK:
      if key == KEY_W:
        w[:, 0] = (w[:, 0] + 0.1).clamp(max=1.0)
      elif key == KEY_S:
        w[:, 0] = (w[:, 0] - 0.1).clamp(min=-0.6)
      elif key == KEY_Q:
        w[:, 1] = (w[:, 1] + 0.1).clamp(max=0.5)
      elif key == KEY_E:
        w[:, 1] = (w[:, 1] - 0.1).clamp(min=-0.5)
      elif key == KEY_A:
        w[:, 2] = (w[:, 2] + 0.2).clamp(max=0.8)
      elif key == KEY_D:
        w[:, 2] = (w[:, 2] - 0.2).clamp(min=-0.8)
    elif mode == CRAWL:
      if key == KEY_W:
        c[:, 0] = (c[:, 0] + 0.25).clamp(max=2.5)
      elif key == KEY_S:
        c[:, 0] = (c[:, 0] - 0.25).clamp(min=-1.0)
      elif key == KEY_Q:
        c[:, 1] = (c[:, 1] + 0.1).clamp(max=0.5)
      elif key == KEY_E:
        c[:, 1] = (c[:, 1] - 0.1).clamp(min=-0.5)
      elif key == KEY_A:
        c[:, 2] = (c[:, 2] + 0.25).clamp(max=3.0)
      elif key == KEY_D:
        c[:, 2] = (c[:, 2] - 0.25).clamp(min=-3.0)
      elif key == KEY_H:
        heights = _CrawlPolicy.STEP_HEIGHTS
        i = (
          heights.index(min(heights, key=lambda h: abs(h - float(c[0, 3])))) + 1
        ) % len(heights)
        c[:, 3] = heights[i]
    print(switcher.status(), end="", flush=True)

  class _StatusPolicy:
    def __call__(self, obs):
      out = switcher(obs)
      if int(switcher.time[0] / switcher.dt) % 10 == 0:
        print(switcher.status(), end="", flush=True)
      return out

    def reset(self):
      switcher.reset()

  print(__doc__)
  NativeMujocoViewer(wenv, _StatusPolicy(), key_callback=on_key).run()
  env.close()


if __name__ == "__main__":
  main()
