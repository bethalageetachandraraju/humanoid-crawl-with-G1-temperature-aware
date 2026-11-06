"""Walk -> crawl -> walk: the full loop with the two-way stand <-> crawl policy.

The braced 29-DoF G1 under a state machine, three policies:

  WALK --C--> STOPPING --still--> DOWN --phi 1--> HOLD --gate--> CRAWL
  CRAWL --U--> CRAWL_STOP --still--> HOLD --gate--> UP --phi 0--> STAND_HOLD
  STAND_HOLD --steady--> WALK

* WALK: the walk policy. W/S forward/back, Q/E sideways, A/D turn, X stop.
* C: the walk command goes to zero; the descent starts once the robot stands
  still on both feet (a descent started mid-step can fall).
* DOWN / UP: the stand <-> crawl policy (``Mjlab-StandCrawl-Unitree-G1-29dof``)
  follows its schedule, phi 0 -> 1 or 1 -> 0 over ``--transition-s``; it sees
  ``[phi, dphi/dt]``.
* HOLD -> CRAWL: the crawl-pose gate (still, pelvis height and orientation,
  four limbs down) for ``--dwell-s``. CRAWL: the crawl policy, same keys, H step
  height.
* U (while crawling): the crawl command goes to zero; once the crawl policy is
  still on four limbs, the stand <-> crawl policy takes over at phi 1 (HOLD),
  settles into the crawl pose, and when the crawl-pose gate holds, stands up.
  STAND_HOLD -> WALK once upright and still.
* U in HOLD (or still going down) stands back up once the pose is reached;
  C in STAND_HOLD goes straight back down.

PD targets blend over ``--blend-s`` at every hand-over and every policy runs in
shadow mode. P push, F freeze, Enter reset.

Headless test of the whole loop (walk, down, crawl 4 s, up, walk), N robots::

  uv run python scripts/play_chain.py --eval 256 [--robust | --hard] [--push 0.3]

``--robust`` is level-1 randomization of the original curricula; ``--hard`` is
level 1 of the hardened ones (``TRANSITION_RAMP_HARD``) plus a 2 degree IMU tilt
and arm-end mass uncertainty.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_switcher import (  # noqa: E402
  CRAWL_ABOVE_LIMBS,
  CRAWL_CHECKPOINT,
  CRAWL_TASK,
  LIMBS,
  _CrawlPolicy,
  _Policy,
  _WalkPolicy,
  latest_checkpoint,
  load_actor,
  make_env_cfg,
)

import mjlab.tasks  # noqa: E402, F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import BRACED_LOADS_29, MOTORS_29  # noqa: E402
from mjlab.tasks.g1_walk.config.g1.env_cfgs import (  # noqa: E402
  ARM_END_BODIES,
  IMU_TILT_MAX_DEG,
  TRANSITION_ACTION_CLIP,
  TRANSITION_ACTION_SCALE,
  _add_robust_training,
)
from mjlab.tasks.g1_walk.mdp import (  # noqa: E402
  TRANSITION_RAMP,
  TRANSITION_RAMP_HARD,
  RobustnessRamp,
  TransitionSchedule,
)
from mjlab.tasks.g1_walk.robot import JOINTS_29, STAND_HEIGHT  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import NativeMujocoViewer  # noqa: E402
from mjlab.viewer.native.keys import (  # noqa: E402
  KEY_A,
  KEY_C,
  KEY_D,
  KEY_E,
  KEY_F,
  KEY_H,
  KEY_P,
  KEY_Q,
  KEY_S,
  KEY_U,
  KEY_V,
  KEY_W,
  KEY_X,
)

WALK_TASK = "Mjlab-Walk-Flat-Unitree-G1-29dof"
# The hardened walk (rough terrain, hard randomization, IMU tilt, yaw tracking).
WALK_CHECKPOINT = (
  "logs/rsl_rl/g1_walk_29dof/2026-10-23_06-41-09_walk_hard_track/model_55499.pt"
)
STAND_CRAWL_TASK = "Mjlab-StandCrawl-Unitree-G1-29dof"
(
  WALK,
  STOPPING,
  DOWN,
  HOLD,
  CRAWL,
  CRAWL_STOP,
  UP,
  STAND_HOLD,
  FREEZE,
) = range(9)
MODE_NAMES = (
  "WALK",
  "STOPPING",
  "DOWN",
  "HOLD",
  "CRAWL",
  "CRAWL_STOP",
  "UP",
  "STAND_HOLD",
  "FREEZE",
)
TRANSITION_MODES = (DOWN, HOLD, UP, STAND_HOLD)


class _StandCrawlPolicy(_Policy):
  """Actor obs: gravity, gyro, q - stand, qd, last action, [phi, dphi/dt],
  ref - q. Target = ref(phi) + TRANSITION_ACTION_SCALE * clip(residual)."""

  def __init__(self, n, device, actor):
    super().__init__(n, device)
    self.actor = actor
    self.schedule = TransitionSchedule(device)
    self.phase = torch.zeros(n, device=device)
    self.rate = torch.zeros(n, device=device)
    self.q_stand = self.schedule.q_stand

  def reset(self, env_ids):
    super().reset(env_ids)
    self.phase[env_ids] = 0.0
    self.rate[env_ids] = 0.0

  def targets(self, robot):
    ref = self.schedule.joint_pos(self.phase)
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
        self.rate[:, None],
        ref - q,
      ],
      dim=1,
    )
    self.last_action = self.actor(obs).clamp(
      -TRANSITION_ACTION_CLIP, TRANSITION_ACTION_CLIP
    )
    return ref + TRANSITION_ACTION_SCALE * self.last_action


class _Tilt:
  """A fixed per-robot IMU tilt (--hard): applied to every policy's gravity."""

  def __init__(self, n: int, device: str, max_deg: float) -> None:
    angle = math.radians(max_deg) * torch.rand(n, device=device)
    heading = 2 * math.pi * torch.rand(n, device=device)
    k = torch.stack([heading.cos(), heading.sin(), torch.zeros_like(heading)], 1)
    kx = torch.zeros(n, 3, 3, device=device)
    kx[:, 0, 2], kx[:, 1, 2] = k[:, 1], -k[:, 0]
    kx[:, 2, 0], kx[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = angle.sin()[:, None, None], angle.cos()[:, None, None]
    self.rot = torch.eye(3, device=device) + s * kx + (1 - c) * (kx @ kx)

  def __call__(self, g: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nij,nj->ni", self.rot, g)


class _CrawlVariants:
  """Several crawl policies (same observation and action layout) run every tick
  in shadow, each keeping its own last action; the active one drives. Switching
  mid-crawl is then a 0.5 s target blend, like any hand-over (key V)."""

  def __init__(self, n: int, device: str, actors: dict) -> None:
    self.names = list(actors)
    self.policies = [_CrawlPolicy(n, device, a) for a in actors.values()]
    self.command = self.policies[0].command
    for policy in self.policies[1:]:
      policy.command = self.command
    self.active = 0

  @property
  def name(self) -> str:
    return self.names[self.active]

  def reset(self, env_ids: torch.Tensor) -> None:
    for policy in self.policies:
      policy.reset(env_ids)

  def targets(self, robot) -> torch.Tensor:
    out = [policy.targets(robot) for policy in self.policies]
    return out[self.active]


class ChainSwitcher:
  def __init__(self, env: ManagerBasedRlEnv, args, device: str) -> None:
    self.env, self.args, self.device = env, args, device
    self.robot = env.scene["robot"]
    assert tuple(self.robot.joint_names) == JOINTS_29
    n = env.num_envs
    self.n, self.dt = n, env.step_dt
    self.walk = _WalkPolicy(
      n, device, load_actor(WALK_TASK, args.walk_checkpoint, device)[0]
    )
    variants = getattr(args, "crawl_variants", None)
    if variants:
      actors = {}
      for spec in variants:
        name, _, path = spec.partition("=")
        actors[name] = load_actor(CRAWL_TASK, path, device)[0]
      self.crawl = _CrawlVariants(n, device, actors)
      crawl_parts = self.crawl.policies
    else:
      self.crawl = _CrawlPolicy(
        n, device, load_actor(CRAWL_TASK, args.crawl_checkpoint, device)[0]
      )
      crawl_parts = [self.crawl]
    self.trans = _StandCrawlPolicy(
      n, device, load_actor(STAND_CRAWL_TASK, args.stand_crawl_checkpoint, device)[0]
    )
    if args.hard:
      tilt = _Tilt(n, device, IMU_TILT_MAX_DEG)

      def tilted_base(robot, t=tilt):
        return t(robot.data.projected_gravity_b), robot.data.root_link_ang_vel_b

      for p in (self.walk, *crawl_parts, self.trans):
        setattr(p, "base", tilted_base)  # noqa: B010  (an instance override)
    z = lambda: torch.zeros(n, device=device)  # noqa: E731
    self.mode = torch.full((n,), WALK, dtype=torch.long, device=device)
    self.queued = torch.full((n,), -1, dtype=torch.long, device=device)
    self.time, self.dwell, self.mode_t0 = z(), z(), z()
    self.blend_t0 = torch.full((n,), -1e9, device=device)
    self.blend_from = torch.zeros(n, len(JOINTS_29), device=device)
    self.applied = torch.zeros(n, len(JOINTS_29), device=device)
    self.frozen = torch.zeros(n, len(JOINTS_29), device=device)
    self.g_crawl = self.trans.schedule.g_crawl
    sensor = env.scene["limb_contacts"]
    lookup = {name: i for i, name in enumerate(sensor.primary_names)}
    self.limb_ids = [lookup[name] for name in LIMBS]
    self.limb_body_ids = self.robot.find_bodies(LIMBS, preserve_order=True)[0]
    self.forbidden_names = list(env.scene["forbidden_contact"].primary_names)
    kkp = torch.tensor([MOTORS_29[j] for j in JOINTS_29], device=device)
    self.kp, self.kd, self.peak = kkp[:, 0], kkp[:, 1], kkp[:, 2]
    self.braced = [
      (
        env.scene[f"robot/{j.removesuffix('_joint')}_torque"],
        torch.tensor(a, device=device),
      )
      for j, (_, a) in BRACED_LOADS_29.items()
    ]
    self.stats = self._new_stats()

  # -- bookkeeping -------------------------------------------------------------

  SEGMENTS = ("dn", "cr", "up", "wk")

  def _new_stats(self) -> dict[str, torch.Tensor]:
    n, dev = self.n, self.device
    f = lambda: torch.zeros(n, device=dev)  # noqa: E731
    b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    st = {
      f"{k}_{s}": f()
      for k in ("motor", "braced", "speed", "rise")
      for s in self.SEGMENTS
    }
    st |= {k: b() for k in ("reached_crawl", "began_up", "stood", "walking_again")}
    st |= {k: b() for k in ("failed",)}
    st |= {
      "fail_mode": torch.full((n,), -1, dtype=torch.long, device=dev),
      "fail_body": torch.full((n,), -1, dtype=torch.long, device=dev),
      "fail_phase": f(),
      "crawl_x0": torch.zeros(n, 2, device=dev),
      "crawled_m": f(),
      "walk_x0": torch.zeros(n, 2, device=dev),
      "walked_m": f(),
    }
    return st

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
    for p in (self.walk, self.crawl, self.trans):
      p.reset(ids)
    self.walk.command[ids] = 0.0
    self.crawl.command[ids] = 0.0
    self.crawl.command[ids, 3] = _CrawlPolicy.STEP_HEIGHTS[0]
    self.mode[ids] = WALK
    self.queued[ids] = -1
    for t in (self.time, self.dwell, self.mode_t0):
      t[ids] = 0.0
    self.blend_t0[ids] = -1e9
    self.applied[ids] = self.robot.data.joint_pos[ids]
    for k, v in self.stats.items():
      v[ids] = -1 if k in ("fail_mode", "fail_body") else 0

  def _set(self, mask: torch.Tensor, mode: int, blend: bool) -> None:
    ids = mask.nonzero().squeeze(1)
    if len(ids) == 0:
      return
    self.mode[ids] = mode
    self.mode_t0[ids] = self.time[ids]
    self.dwell[ids] = 0.0
    if blend:
      self.blend_t0[ids] = self.time[ids]
      self.blend_from[ids] = self.applied[ids]
    if mode == FREEZE:
      self.frozen[ids] = self.robot.data.joint_pos[ids]
    st, pos = self.stats, self.robot.data.root_link_pos_w[ids, :2]
    if mode == CRAWL:
      st["reached_crawl"][ids] = True
      st["crawl_x0"][ids] = pos
    if mode == UP:
      st["began_up"][ids] = True
    if mode == STAND_HOLD:
      st["stood"][ids] = True
    if mode == WALK and bool(st["stood"][ids].any()):
      back = ids[st["stood"][ids]]
      st["walking_again"][back] = True
      st["walk_x0"][back] = self.robot.data.root_link_pos_w[back, :2]

  # -- keys ----------------------------------------------------------------------

  def press_c(self) -> None:
    walk = self.mode == WALK
    self.walk.command[walk] = 0.0
    self._set(walk, STOPPING, blend=False)
    self._set(self.mode == STAND_HOLD, DOWN, blend=False)
    self.queued[self.mode == UP] = DOWN

  def switch_crawl(self, k: int | None = None) -> None:
    """Next crawl variant (or variant ``k``), blended in over ``blend_s``."""
    if not isinstance(self.crawl, _CrawlVariants):
      return
    nv = len(self.crawl.names)
    self.crawl.active = (self.crawl.active + 1) % nv if k is None else k % nv
    crawling = (self.mode == CRAWL) | (self.mode == CRAWL_STOP)
    self.blend_t0[crawling] = self.time[crawling]
    self.blend_from[crawling] = self.applied[crawling]

  def press_u(self) -> None:
    crawl = self.mode == CRAWL
    self.crawl.command[crawl, :3] = 0.0
    self._set(crawl, CRAWL_STOP, blend=False)
    self.queued[(self.mode == HOLD) | (self.mode == DOWN)] = UP

  def push(self, speed: float, mask: torch.Tensor | None = None) -> None:
    vel = self.robot.data.root_link_vel_w.clone()
    kick = torch.randn(self.n, 2, device=self.device)
    kick = kick / kick.norm(dim=1, keepdim=True).clamp(min=1e-6) * speed
    if mask is not None:
      kick = kick * mask[:, None].float()
    vel[:, :2] += kick
    self.robot.write_root_link_velocity_to_sim(vel)

  # -- gates -----------------------------------------------------------------------

  def contacts(self) -> torch.Tensor:
    found = self.env.scene["limb_contacts"].data.found
    assert found is not None
    return found[:, self.limb_ids] > 0  # left foot, right foot, left tip, right tip

  def crawl_gate(self) -> torch.Tensor:
    d = self.robot.data
    still = d.joint_vel.abs().amax(dim=1) < self.args.gate_joint_speed
    limbs = d.body_link_pos_w[:, self.limb_body_ids, 2].mean(dim=1)
    height = ((d.root_link_pos_w[:, 2] - limbs) - CRAWL_ABOVE_LIMBS).abs() < 0.03
    orient = (d.projected_gravity_b - self.g_crawl).norm(dim=1) < 0.15
    return still & height & orient & self.contacts().all(dim=1)

  def crawl_settled(self) -> torch.Tensor:
    """The crawl policy has stopped: still, on four limbs, roughly prone."""
    d = self.robot.data
    still = d.joint_vel.abs().amax(dim=1) < self.args.gate_joint_speed
    orient = (d.projected_gravity_b - self.g_crawl).norm(dim=1) < 0.25
    return still & orient & self.contacts().all(dim=1)

  def stand_gate(self) -> torch.Tensor:
    d = self.robot.data
    still = d.joint_vel.abs().amax(dim=1) < 0.6
    slow = (d.root_link_lin_vel_w[:, :2].norm(dim=1) < 0.1) & (
      d.root_link_ang_vel_w[:, 2].abs() < 0.2
    )
    feet = self.contacts()[:, :2].all(dim=1)
    return still & slow & feet

  # -- step ---------------------------------------------------------------------------

  def __call__(self, obs) -> torch.Tensor:
    del obs
    robot, d = self.robot, self.robot.data
    self.time += self.dt
    m = self.mode.clone()
    inv_t = 1.0 / self.args.transition_s
    self.trans.rate = torch.where(
      m == DOWN,
      torch.full_like(self.trans.rate, inv_t),
      torch.where(m == UP, torch.full_like(self.trans.rate, -inv_t), 0.0),
    )
    t_walk = self.walk.targets(robot)
    t_trans = self.trans.targets(robot)
    t_crawl = self.crawl.targets(robot)
    use_walk = (m == WALK) | (m == STOPPING)
    use_crawl = (m == CRAWL) | (m == CRAWL_STOP)
    target = torch.where(
      use_walk[:, None], t_walk, torch.where(use_crawl[:, None], t_crawl, t_trans)
    )
    target = torch.where((m == FREEZE)[:, None], self.frozen, target)
    s = ((self.time - self.blend_t0) / self.args.blend_s).clamp(0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    self.applied = self.blend_from + (target - self.blend_from) * s[:, None]

    # Phase and modes.
    self.trans.phase = (self.trans.phase + self.trans.rate * self.dt).clamp(0.0, 1.0)
    self.trans.phase = torch.where(m == STOPPING, 0.0, self.trans.phase)
    self.trans.phase = torch.where(m == CRAWL_STOP, 1.0, self.trans.phase)
    stand_ok = self.stand_gate()
    upright = d.projected_gravity_b[:, 2] < -0.97
    tall = d.root_link_pos_w[:, 2] > STAND_HEIGHT - 0.06
    crawl_ok = self.crawl_gate()
    # The crawl policy stands ~14 cm higher than the crawl pose: CRAWL_STOP only
    # waits for it to be still on four limbs, then HOLD (the stand <-> crawl
    # policy at phi 1) settles it into the crawl pose before UP.
    settled = self.crawl_settled()
    if self.args.relaxed_handover:
      # Hand the stand to the walk policy once upright and tall on both feet,
      # still or not: it balances better than the transition at phi 0.
      feet = self.contacts()[:, :2].all(dim=1)
      upright_ok = (d.projected_gravity_b[:, 2] < -0.94) & tall & feet
    else:
      upright_ok = stand_ok & upright & tall
    ok = torch.where(
      m == STOPPING,
      stand_ok,
      torch.where(
        m == STAND_HOLD,
        upright_ok,
        torch.where(m == HOLD, crawl_ok, torch.where(m == CRAWL_STOP, settled, False)),
      ),
    )
    self.dwell = torch.where(ok, self.dwell + self.dt, torch.zeros_like(self.dwell))
    dwell = torch.where(
      m == HOLD,
      torch.full_like(self.dwell, self.args.dwell_s),
      torch.full_like(self.dwell, 0.5),
    )
    ready = self.dwell >= dwell
    going_up = self.queued == UP
    self._set((m == STOPPING) & ready, DOWN, blend=True)
    self._set((m == DOWN) & (self.trans.phase >= 1.0), HOLD, blend=False)
    self._set((m == HOLD) & ready & ~going_up, CRAWL, blend=True)
    stop_done = (m == CRAWL_STOP) & ready
    self._set(stop_done, HOLD, blend=True)
    self.queued[stop_done] = UP
    up = (m == HOLD) & ready & going_up
    self._set(up, UP, blend=False)
    self.queued[up] = -1
    self._set((m == UP) & (self.trans.phase <= 0.0), STAND_HOLD, blend=False)
    self._set((m == STAND_HOLD) & ready & (self.queued != DOWN), WALK, blend=True)
    go = (self.mode == STAND_HOLD) & (self.queued == DOWN)
    self._set(go, DOWN, blend=False)
    self.queued[go] = -1

    # Safety bookkeeping by segment.
    st = self.stats
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
    seg = {
      "dn": (m == DOWN) | (m == HOLD),
      "cr": (m == CRAWL) | (m == CRAWL_STOP),
      "up": (m == UP) | (m == STAND_HOLD),
      "wk": (m == WALK) & st["walking_again"],
    }
    for tag, mask in seg.items():
      for key, value in (
        ("motor", ratio),
        ("braced", braced),
        ("speed", speed),
        ("rise", rise),
      ):
        k = f"{key}_{tag}"
        st[k] = torch.where(mask, torch.maximum(st[k], value), st[k])
    g = d.projected_gravity_b
    fell = (g[:, 2] > 0.5) | (g[:, 1].abs() > 0.8) | (g[:, 0] < -0.6)
    upright_walk = use_walk & (g[:, 2] > -0.5)  # Walking but tilted > 60 deg.
    found = self.env.scene["forbidden_contact"].data.found
    assert found is not None
    hit = found > 0
    bad = fell | upright_walk | hit.any(dim=1)
    first = bad & ~st["failed"]
    st["fail_mode"][first] = m[first]
    st["fail_phase"][first] = self.trans.phase[first]
    st["fail_body"][first] = torch.where(
      hit[first].any(dim=1), hit[first].float().argmax(dim=1), -2
    )
    st["failed"] |= bad
    pos = d.root_link_pos_w[:, :2]
    st["crawled_m"] = torch.where(
      seg["cr"], (pos - st["crawl_x0"]).norm(dim=1), st["crawled_m"]
    )
    st["walked_m"] = torch.where(
      seg["wk"], (pos - st["walk_x0"]).norm(dim=1), st["walked_m"]
    )
    return self.applied

  def status(self) -> str:
    i = 0
    mode = MODE_NAMES[int(self.mode[i])]
    txt = f"\r[{mode:10s}] t={float(self.time[i]):6.1f}s"
    if mode in ("WALK", "STOPPING"):
      c = self.walk.command[i].tolist()
      txt += f"  cmd fwd {c[0]:+.2f} lat {c[1]:+.2f} turn {c[2]:+.2f}"
    elif mode in ("CRAWL", "CRAWL_STOP"):
      c = self.crawl.command[i].tolist()
      txt += f"  cmd fwd {c[0]:+.2f} lat {c[1]:+.2f} turn {c[2]:+.2f} step {c[3]:.2f}"
      if isinstance(self.crawl, _CrawlVariants):
        txt += f"  [{self.crawl.name}]"
    else:
      txt += f"  phase {float(self.trans.phase[i]):.2f}"
    return txt + "   "


def add_randomization(cfg, ramp: RobustnessRamp, arm_mass: bool) -> None:
  """Level-1 randomization of ``ramp``, re-drawn every reset (no pushes)."""
  _add_robust_training(cfg, "none", play=False, ramp=ramp)
  cfg.curriculum = {}
  cfg.events.pop("push_robot")
  at = lambda knob: knob[1]  # noqa: E731
  r = ramp
  p = {k: cfg.events[k].params for k in cfg.events if k.startswith("dr_")}
  p["dr_friction"]["ranges"] = (at(r.friction_low), at(r.friction_high))
  p["dr_body_mass"]["ranges"] = (1 - at(r.mass_scale), 1 + at(r.mass_scale))
  p["dr_torso_payload"]["ranges"] = (at(r.payload_low_kg), at(r.payload_high_kg))
  p["dr_torso_com"]["ranges"] = (-at(r.com_offset_m), at(r.com_offset_m))
  p["dr_joint_friction"]["ranges"] = (0.0, at(r.joint_friction_nm))
  p["dr_armature"]["ranges"] = (1 - at(r.armature_scale), 1 + at(r.armature_scale))
  g = at(r.pd_gain_scale)
  p["dr_motor_gains"]["kp_range"] = (1 - g, 1 + g)
  p["dr_motor_gains"]["kd_range"] = (1 - g, 1 + g)
  p["dr_motor_strength"]["effort_limit_range"] = (at(r.effort_scale_low), 1.0)
  if arm_mass:
    from mjlab.envs.mdp.dr import body_mass

    cfg.events["dr_arm_end_mass"] = EventTermCfg(
      func=body_mass,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=ARM_END_BODIES),
        "ranges": (-0.05, 0.2),
        "operation": "add",
      },
    )


def evaluate(sw: ChainSwitcher, wenv, args) -> None:
  """Walk 3 s, C, down, hold, crawl 4 s, U, up, walk 3 s."""
  dt, n = sw.dt, sw.n
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  sw.reset()
  T = args.transition_s
  total = int((1 + 3 + 2 + T + 3 + args.eval_crawl_s + 3 + T + 2 + 3 + 6) / dt)
  pushed_dn = torch.zeros(n, dtype=torch.bool, device=sw.device)
  pushed_up = torch.zeros(n, dtype=torch.bool, device=sw.device)
  with torch.inference_mode():
    for step in range(total):
      t = step * dt
      m = sw.mode
      if 1.0 <= t < 4.0:
        sw.walk.command[m == WALK, 0] = 0.5
      if step == int(4.0 / dt):
        sw.press_c()
      crawling = m == CRAWL
      since = sw.time - sw.mode_t0
      speed = torch.where(since[crawling] > 1.0, args.eval_crawl_speed, 0.0)
      period = max(int(args.switch_every / dt), 1) if args.switch_every else 0
      if period and args.switch_at_rest:
        # Stop 2 s before each switch, resume 1 s after it.
        k = step % period
        resting = k >= period - int(2.0 / dt) or k < int(1.0 / dt)
        speed = speed * (0.0 if resting else 1.0)
      sw.crawl.command[crawling, 0] = speed
      if period and step % period == 0:
        sw.switch_crawl()
      # Up after --eval-crawl-s of crawling.
      up_now = crawling & (since > args.eval_crawl_s)
      if up_now.any():
        sw.crawl.command[up_now, :3] = 0.0
        sw._set(up_now, CRAWL_STOP, blend=False)
      again = (m == WALK) & sw.stats["walking_again"]
      sw.walk.command[again, 0] = 0.5
      if args.push:
        kick = ((m == DOWN) & (sw.trans.phase > 0.5) & ~pushed_dn) | (
          (m == UP) & (sw.trans.phase < 0.5) & ~pushed_up
        )
        if kick.any():
          sw.push(args.push, kick)
          pushed_dn |= kick & (m == DOWN)
          pushed_up |= kick & (m == UP)
      obs = wenv.step(sw(obs))[0]
  st = sw.stats
  pct = lambda x: 100.0 * float(x.float().mean())  # noqa: E731
  failed = st["fail_mode"] >= 0
  by_mode = ", ".join(
    f"{MODE_NAMES[k]} x{int((st['fail_mode'][failed] == k).sum())}"
    for k in range(len(MODE_NAMES))
    if bool((st["fail_mode"][failed] == k).any())
  )
  bodies: dict[str, int] = {}
  for b in st["fail_body"][failed].tolist():
    name = "tilt" if b < 0 else sw.forbidden_names[b]
    bodies[name] = bodies.get(name, 0) + 1

  def peaks(tag: str) -> str:
    q = lambda k, p: float(st[f"{k}_{tag}"].quantile(p))  # noqa: E731
    return (
      f"motor/limit {q('motor', 0.5):.2f} (p95 {q('motor', 0.95):.2f}), braced "
      f"{q('braced', 0.5):.1f} Nm (p95 {q('braced', 0.95):.1f}), joint speed "
      f"{q('speed', 0.5):.1f} (p95 {q('speed', 0.95):.1f}) rad/s, force rise "
      f"{q('rise', 0.5):.0f} (p95 {q('rise', 0.95):.0f}) N/step"
    )

  label = "hard" if args.hard else ("randomized" if args.robust else "nominal")
  print(
    f"\n=== full loop, {n} robots, {label}{', push %.1f m/s' % args.push if args.push else ''} ===\n"
    f"reached crawl (gate passed):   {pct(st['reached_crawl']):5.1f}%\n"
    f"crawled > 1 m:                 {pct(st['crawled_m'] > 1.0):5.1f}%\n"
    f"started standing up:           {pct(st['began_up']):5.1f}%\n"
    f"stood up:                      {pct(st['stood']):5.1f}%\n"
    f"walking again (> 0.8 m):       {pct(st['walking_again'] & (st['walked_m'] > 0.8)):5.1f}%\n"
    f"any failure (fall/forbidden):  {pct(failed):5.1f}%"
    f"{'   first: ' + by_mode if by_mode else ''}{'   bodies: ' + str(bodies) if bodies else ''}\n"
    f"  down:  {peaks('dn')}\n  crawl: {peaks('cr')}\n  up:    {peaks('up')}\n"
    f"  walk:  {peaks('wk')}"
  )


def add_chain_args(ap: argparse.ArgumentParser) -> None:
  """Checkpoints, state-machine timing and randomization (shared with
  ``chain_load_monitor.py``)."""
  ap.add_argument("--walk-checkpoint", default=None)
  ap.add_argument("--stand-crawl-checkpoint", default="latest")
  ap.add_argument("--crawl-checkpoint", default=CRAWL_CHECKPOINT)
  ap.add_argument(
    "--crawl-variants",
    nargs="*",
    default=None,
    help="name=checkpoint ... : several crawl styles, V switches while crawling "
    "(overrides --crawl-checkpoint)",
  )
  ap.add_argument("--eval-crawl-s", type=float, default=5.0)
  ap.add_argument("--eval-crawl-speed", type=float, default=0.5)
  ap.add_argument(
    "--switch-at-rest",
    action="store_true",
    help="eval: stop the crawl 2 s before each switch, resume 1 s after",
  )
  ap.add_argument(
    "--switch-every", type=float, default=0.0, help="eval: switch variant every s"
  )
  ap.add_argument("--transition-s", type=float, default=6.0)
  ap.add_argument("--blend-s", type=float, default=0.5)
  ap.add_argument("--dwell-s", type=float, default=1.0)
  ap.add_argument("--gate-joint-speed", type=float, default=0.5)
  ap.add_argument("--robust", action="store_true")
  ap.add_argument("--hard", action="store_true")
  ap.add_argument(
    "--relaxed-handover",
    action="store_true",
    help="STAND_HOLD -> WALK once upright and tall on both feet, without waiting "
    "for the robot to be still",
  )


def resolve_checkpoints(args) -> None:
  if args.walk_checkpoint is None:
    args.walk_checkpoint = WALK_CHECKPOINT
  if args.stand_crawl_checkpoint == "latest":
    runs = sorted(Path("logs/rsl_rl/g1_stand_to_crawl_29dof").glob("*stand_crawl*"))
    args.stand_crawl_checkpoint = (
      latest_checkpoint("g1_stand_to_crawl_29dof")
      if not runs
      else max(
        (p for r in runs for p in r.glob("model_*.pt")),
        key=lambda p: (p.stat().st_mtime, int(p.stem.split("_")[1])),
      )
    )
  print(
    f"walk: {args.walk_checkpoint}\nstand<->crawl: {args.stand_crawl_checkpoint}\n"
    f"crawl: {args.crawl_checkpoint}"
  )


def chain_key_handler(sw: ChainSwitcher):
  """Keys for the MuJoCo window (see the module docstring)."""
  n, device = sw.n, sw.device

  def on_key(key: int) -> None:
    w, c = sw.walk.command, sw.crawl.command
    mode = int(sw.mode[0])
    if key == KEY_C:
      sw.press_c()
    elif key == KEY_U:
      sw.press_u()
    elif key == KEY_F:
      sw._set(torch.ones(n, dtype=torch.bool, device=device), FREEZE, blend=True)
    elif key == KEY_P:
      sw.push(0.5)
    elif key == KEY_V:
      sw.switch_crawl()
    elif key == KEY_X:
      w[:] = 0.0
      c[:, :3] = 0.0
    elif mode == WALK:
      steps = {KEY_W: (0, 0.1), KEY_S: (0, -0.1), KEY_Q: (1, 0.1)}
      steps |= {KEY_E: (1, -0.1), KEY_A: (2, 0.2), KEY_D: (2, -0.2)}
      lim = ((-0.6, 1.0), (-0.5, 0.5), (-0.8, 0.8))
      if key in steps:
        i, dv = steps[key]
        w[:, i] = (w[:, i] + dv).clamp(*lim[i])
    elif mode == CRAWL:
      steps = {KEY_W: (0, 0.25), KEY_S: (0, -0.25), KEY_Q: (1, 0.1)}
      steps |= {KEY_E: (1, -0.1), KEY_A: (2, 0.25), KEY_D: (2, -0.25)}
      lim = ((-1.0, 2.5), (-0.5, 0.5), (-3.0, 3.0))
      if key in steps:
        i, dv = steps[key]
        c[:, i] = (c[:, i] + dv).clamp(*lim[i])
      elif key == KEY_H:
        hs = _CrawlPolicy.STEP_HEIGHTS
        j = (hs.index(min(hs, key=lambda h: abs(h - float(c[0, 3])))) + 1) % len(hs)
        c[:, 3] = hs[j]
    print(sw.status(), end="", flush=True)

  return on_key


def make_chain_cfg(n: int, args):
  cfg = make_env_cfg(n, robust=False, no_delay=False)
  if args.hard:
    add_randomization(cfg, TRANSITION_RAMP_HARD, arm_mass=True)
  elif args.robust:
    add_randomization(cfg, TRANSITION_RAMP, arm_mass=False)
  return cfg


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  add_chain_args(ap)
  ap.add_argument("--eval", type=int, default=0)
  ap.add_argument("--push", type=float, default=0.0)
  args = ap.parse_args()
  resolve_checkpoints(args)
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  n = args.eval or 1
  cfg = make_chain_cfg(n, args)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  sw = ChainSwitcher(env, args, device)
  if args.eval:
    evaluate(sw, wenv, args)
    env.close()
    return

  on_key = chain_key_handler(sw)

  class _StatusPolicy:
    def __call__(self, obs):
      out = sw(obs)
      if int(sw.time[0] / sw.dt) % 10 == 0:
        print(sw.status(), end="", flush=True)
      return out

    def reset(self):
      sw.reset()

  print(__doc__)
  NativeMujocoViewer(wenv, _StatusPolicy(), key_callback=on_key).run()
  env.close()


if __name__ == "__main__":
  main()
