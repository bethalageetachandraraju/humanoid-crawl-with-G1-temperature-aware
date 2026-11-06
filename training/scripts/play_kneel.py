"""Walk, take a knee, stand back up, walk on: two policies, one robot.

The braced 29-DoF G1 under a state machine:

  WALK --K--> STOPPING --still 0.5 s--> KNEEL_DOWN --phi 1--> KNEELING
  KNEELING --U--> STANDING_UP --phi 0--> STAND_HOLD --steady 0.5 s--> WALK

* WALK: the walk policy (``Mjlab-Walk-Flat-Unitree-G1-29dof``). W/S forward and
  back, Q/E sideways, A/D turn, X stop.
* K: the walk command goes to zero; the kneel starts only once the robot has
  stood still for 0.5 s (a descent started while stepping falls).
* KNEEL_DOWN / STANDING_UP: the kneel policy (``Mjlab-StandKneel-Unitree-G1-
  29dof``) follows its schedule; the phase moves over ``--kneel-s`` seconds.
  A key pressed mid-way is queued until the pose is reached.
* KNEELING: holds the kneel until U.
* STAND_HOLD: the kneel policy holds the stand until the robot is upright,
  still and on both feet, then the walk policy takes over.

PD targets are blended over ``--blend-s`` at every hand-over, and both policies
run in shadow mode so each one's last action is always its own. Other keys: P
push, Enter reset.

Headless test of the whole sequence, ``N`` robots::

  uv run python scripts/play_kneel.py --eval 256 [--robust] [--push 0.3]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_switcher import (  # noqa: E402
  _Policy,
  _WalkPolicy,
  latest_checkpoint,
  load_actor,
)

import mjlab.tasks  # noqa: E402, F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs import mdp as envs_mdp  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionActionCfg  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.observation_manager import (  # noqa: E402
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import BRACED_LOADS_29, MOTORS_29  # noqa: E402
from mjlab.tasks.g1_walk.config.g1.env_cfgs import (  # noqa: E402
  KNEEL_FORBIDDEN_BODIES,
  TRANSITION_ACTION_CLIP,
  TRANSITION_ACTION_SCALE,
  _add_robust_training,
  unitree_g1_walk_env_cfg,
)
from mjlab.tasks.g1_walk.mdp import KNEEL_RAMP, KneelSchedule  # noqa: E402
from mjlab.tasks.g1_walk.robot import (  # noqa: E402
  JOINTS_29,
  STAND_HEIGHT,
  get_g1_kneel_robot_cfg_29,
)
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import NativeMujocoViewer  # noqa: E402
from mjlab.viewer.native.keys import (  # noqa: E402
  KEY_A,
  KEY_D,
  KEY_E,
  KEY_K,
  KEY_P,
  KEY_Q,
  KEY_S,
  KEY_U,
  KEY_W,
  KEY_X,
)

WALK_TASK = "Mjlab-Walk-Flat-Unitree-G1-29dof"
KNEEL_TASK = "Mjlab-StandKneel-Unitree-G1-29dof"
WALK_CHECKPOINT = (
  "logs/rsl_rl/g1_walk_29dof/2026-10-18_07-45-18_walk_sym_spin2/model_25000.pt"
)
CONTACT_BODIES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "right_knee_link",
)
WALK, STOPPING, KNEEL_DOWN, KNEELING, STANDING_UP, STAND_HOLD = range(6)
MODE_NAMES = ("WALK", "STOPPING", "KNEEL_DOWN", "KNEELING", "STANDING_UP", "STAND_HOLD")


class _KneelPolicy(_Policy):
  """Actor obs: gravity, gyro, q - stand, qd, last action, [phi, rate], ref - q.

  Target = schedule(phi) + TRANSITION_ACTION_SCALE * clip(residual).
  """

  def __init__(self, n, device, actor):
    super().__init__(n, device)
    self.actor = actor  # None: play the schedule with zero residual.
    self.schedule = KneelSchedule(device)
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


class KneelSwitcher:
  def __init__(self, env: ManagerBasedRlEnv, args, device: str) -> None:
    self.env, self.args, self.device = env, args, device
    self.robot = env.scene["robot"]
    assert tuple(self.robot.joint_names) == JOINTS_29
    n = env.num_envs
    self.n, self.dt = n, env.step_dt
    walk_actor, _ = load_actor(WALK_TASK, args.walk_checkpoint, device)
    kneel_actor = (
      load_actor(KNEEL_TASK, args.kneel_checkpoint, device)[0]
      if args.kneel_checkpoint
      else None
    )
    self.walk = _WalkPolicy(n, device, walk_actor)
    self.kneel = _KneelPolicy(n, device, kneel_actor)
    z = lambda: torch.zeros(n, device=device)  # noqa: E731
    self.mode = torch.full((n,), WALK, dtype=torch.long, device=device)
    self.time, self.dwell, self.blend_t0 = z(), z(), z() - 1e9
    self.queued = torch.full((n,), -1, dtype=torch.long, device=device)
    self.blend_from = torch.zeros(n, len(JOINTS_29), device=device)
    self.applied = torch.zeros(n, len(JOINTS_29), device=device)
    sensor = env.scene["kneel_contacts"]
    lookup = {name: i for i, name in enumerate(sensor.primary_names)}
    self.contact_ids = [lookup[b] for b in CONTACT_BODIES]
    table = torch.tensor([MOTORS_29[j] for j in JOINTS_29], device=device)
    self.kp, self.kd, self.peak = table[:, 0], table[:, 1], table[:, 2]
    self.braced = [
      (
        env.scene[f"robot/{j.removesuffix('_joint')}_torque"],
        torch.tensor(a, device=device),
      )
      for j, (_, a) in BRACED_LOADS_29.items()
    ]
    self.stats = self._new_stats()

  def _new_stats(self) -> dict[str, torch.Tensor]:
    n, dev = self.n, self.device
    f = lambda: torch.zeros(n, device=dev)  # noqa: E731
    b = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
    return {
      "max_motor_ratio": f(),
      "max_braced_nm": f(),
      "max_joint_speed": f(),
      "max_force_rise": f(),
      "max_knee_force": f(),
      "fell": b(),
      "fell_mode": torch.full((n,), -1, dtype=torch.long, device=dev),
      "fell_phase": f(),
      "fell_time": f(),
      "fell_body": torch.full((n,), -1, dtype=torch.long, device=dev),
      "forbidden": b(),
      "kneeled": b(),
      "kneel_err": f(),
      "stood": b(),
      "walking_again": b(),
    }

  # -- state ----------------------------------------------------------------

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    ids = torch.arange(self.n, device=self.device) if env_ids is None else env_ids
    for p in (self.walk, self.kneel):
      p.reset(ids)
    self.walk.command[ids] = 0.0
    self.mode[ids] = WALK
    self.time[ids] = 0.0
    self.dwell[ids] = 0.0
    self.queued[ids] = -1
    self.blend_t0[ids] = -1e9
    self.applied[ids] = self.robot.data.joint_pos[ids]
    for k, v in self.stats.items():
      v[ids] = -1 if k in ("fell_mode", "fell_body") else 0

  def _set(self, mask: torch.Tensor, mode: int, blend: bool) -> None:
    ids = mask.nonzero().squeeze(1)
    if len(ids) == 0:
      return
    self.mode[ids] = mode
    self.dwell[ids] = 0.0
    if blend:
      self.blend_t0[ids] = self.time[ids]
      self.blend_from[ids] = self.applied[ids]

  def press_kneel(self) -> None:
    walk = self.mode == WALK
    self.walk.command[walk] = 0.0
    self._set(walk, STOPPING, blend=False)
    self._set(self.mode == STAND_HOLD, KNEEL_DOWN, blend=False)
    self.queued[self.mode == STANDING_UP] = KNEEL_DOWN

  def press_up(self) -> None:
    self._set(self.mode == KNEELING, STANDING_UP, blend=False)
    self.queued[self.mode == KNEEL_DOWN] = STANDING_UP

  def push(self, speed: float) -> None:
    vel = self.robot.data.root_link_vel_w.clone()
    kick = torch.randn(self.n, 2, device=self.device)
    vel[:, :2] += kick / kick.norm(dim=1, keepdim=True).clamp(min=1e-6) * speed
    self.robot.write_root_link_velocity_to_sim(vel)

  def contacts(self) -> torch.Tensor:
    found = self.env.scene["kneel_contacts"].data.found
    assert found is not None
    return found[:, self.contact_ids] > 0  # left foot, right foot, right knee

  # -- step ------------------------------------------------------------------

  def __call__(self, obs) -> torch.Tensor:
    del obs
    d = self.robot.data
    self.time += self.dt
    m = self.mode.clone()  # This step's mode; _set edits self.mode in place.
    # Phase and rate for the kneel policy, from the mode.
    rate = 1.0 / self.args.kneel_s
    self.kneel.rate = torch.where(
      m == KNEEL_DOWN,
      torch.full_like(self.kneel.rate, rate),
      torch.where(m == STANDING_UP, torch.full_like(self.kneel.rate, -rate), 0.0),
    )
    t_walk = self.walk.targets(self.robot)
    t_kneel = self.kneel.targets(self.robot)
    use_walk = (m == WALK) | (m == STOPPING)
    target = torch.where(use_walk[:, None], t_walk, t_kneel)
    s = ((self.time - self.blend_t0) / self.args.blend_s).clamp(0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)
    self.applied = self.blend_from + (target - self.blend_from) * s[:, None]

    # Advance phases and modes.
    self.kneel.phase = (self.kneel.phase + self.kneel.rate * self.dt).clamp(0.0, 1.0)
    still = d.joint_vel.abs().amax(dim=1) < self.args.still_joint_speed
    slow = (d.root_link_lin_vel_w[:, :2].norm(dim=1) < 0.1) & (
      d.root_link_ang_vel_w[:, 2].abs() < 0.2
    )
    contact = self.contacts()
    feet = contact[:, 0] & contact[:, 1]
    upright = d.projected_gravity_b[:, 2] < -0.97
    tall = d.root_link_pos_w[:, 2] > STAND_HEIGHT - 0.06
    ok = torch.where(
      m == STOPPING,
      still & slow & feet,
      torch.where(m == STAND_HOLD, still & slow & feet & upright & tall, False),
    )
    self.dwell = torch.where(ok, self.dwell + self.dt, torch.zeros_like(self.dwell))
    ready = self.dwell >= self.args.dwell_s
    # Walk -> kneel policy (blend); kneel policy holding the stand -> walk.
    self.kneel.phase = torch.where(m == STOPPING, 0.0, self.kneel.phase)
    self._set((m == STOPPING) & ready, KNEEL_DOWN, blend=True)
    self._set((m == STAND_HOLD) & ready, WALK, blend=True)
    down_done = (m == KNEEL_DOWN) & (self.kneel.phase >= 1.0)
    up_done = (m == STANDING_UP) & (self.kneel.phase <= 0.0)
    self.stats["kneeled"] |= down_done
    self.stats["stood"] |= up_done
    self._set(down_done, KNEELING, blend=False)
    self._set(up_done, STAND_HOLD, blend=False)
    # Queued keys take effect once the pose is reached.
    for src, dst in ((KNEELING, STANDING_UP), (STAND_HOLD, KNEEL_DOWN)):
      go = (self.mode == src) & (self.queued == dst)
      self._set(go, dst, blend=False)
      self.queued[go] = -1
    self.stats["walking_again"] |= (m == STAND_HOLD) & ready

    # Safety bookkeeping.
    demand = self.kp * (self.applied - d.joint_pos) - self.kd * d.joint_vel
    st = self.stats
    active = self.mode >= KNEEL_DOWN  # The kneel policy in charge.
    upd = lambda k, v: st.__setitem__(  # noqa: E731
      k, torch.where(active, torch.maximum(st[k], v), st[k])
    )
    upd("max_motor_ratio", (demand.abs() / self.peak).amax(dim=1))
    braced = torch.zeros(self.n, device=self.device)
    for sensor, axis in self.braced:
      braced = torch.maximum(braced, (sensor.data @ axis).abs())
    upd("max_braced_nm", braced)
    upd("max_joint_speed", d.joint_vel.abs().amax(dim=1))
    hist = self.env.scene["kneel_contacts"].data.force_history
    if hist is not None:
      force = hist.norm(dim=-1)  # [B, bodies, H]
      rise = torch.relu(force[:, :, 1:] - force[:, :, :-1]).amax(dim=2).amax(dim=1)
      upd("max_force_rise", rise)
      upd("max_knee_force", force[:, self.contact_ids[2]].amax(dim=1))
    g = d.projected_gravity_b
    forbidden = self.env.scene["forbidden_contact"].data.found
    assert forbidden is not None
    failed = (g[:, 2] > -0.5) | (forbidden > 0).any(dim=1)
    first = failed & ~st["fell"] & ~st["forbidden"]
    st["fell_mode"][first] = m[first]
    st["fell_phase"][first] = self.kneel.phase[first]
    st["fell_time"][first] = self.time[first]
    hit = forbidden > 0
    st["fell_body"][first] = torch.where(
      hit[first].any(dim=1), hit[first].float().argmax(dim=1), -2
    )
    st["fell"] |= g[:, 2] > -0.5
    st["forbidden"] |= (forbidden > 0).any(dim=1)
    kneel_now = self.mode == KNEELING
    err = (d.joint_pos - self.kneel.schedule.q_kneel).square().mean(dim=1).sqrt()
    st["kneel_err"] = torch.where(kneel_now, err, st["kneel_err"])
    return self.applied

  def status(self) -> str:
    i = 0
    mode = MODE_NAMES[int(self.mode[i])]
    txt = f"\r[{mode:11s}] t={float(self.time[i]):6.1f}s"
    if mode in ("WALK", "STOPPING"):
      c = self.walk.command[i].tolist()
      txt += f"  cmd fwd {c[0]:+.2f} lat {c[1]:+.2f} turn {c[2]:+.2f}"
    else:
      txt += f"  phase {float(self.kneel.phase[i]):.2f}"
    c = self.contacts()[i].tolist()
    txt += f"  contacts L{int(c[0])} R{int(c[1])} knee{int(c[2])}"
    return txt + "   "


def make_env_cfg(n: int, robust: bool):
  cfg = unitree_g1_walk_env_cfg(play=True)
  cfg.scene.num_envs = n
  cfg.scene.entities = {"robot": get_g1_kneel_robot_cfg_29()}
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
      name="kneel_contacts",
      primary=ContactMatch(
        mode="body",
        pattern=r"^(left|right)_(ankle_roll_link|knee_link)$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=3,
    ),
    ContactSensorCfg(
      name="forbidden_contact",
      primary=ContactMatch(mode="body", pattern=KNEEL_FORBIDDEN_BODIES, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=3,
    ),
  )
  if robust:
    _add_robust_training(cfg, "none", play=False, ramp=KNEEL_RAMP)
    cfg.curriculum = {}
    cfg.events.pop("push_robot")
    r = KNEEL_RAMP
    p = {k: cfg.events[k].params for k in cfg.events if k.startswith("dr_")}
    p["dr_friction"]["ranges"] = (r.friction_low[1], r.friction_high[1])
    p["dr_body_mass"]["ranges"] = (1 - r.mass_scale[1], 1 + r.mass_scale[1])
    p["dr_torso_payload"]["ranges"] = (r.payload_low_kg[1], r.payload_high_kg[1])
    p["dr_torso_com"]["ranges"] = (-r.com_offset_m[1], r.com_offset_m[1])
    p["dr_joint_friction"]["ranges"] = (0.0, r.joint_friction_nm[1])
    p["dr_armature"]["ranges"] = (1 - r.armature_scale[1], 1 + r.armature_scale[1])
    g = r.pd_gain_scale[1]
    p["dr_motor_gains"]["kp_range"] = (1 - g, 1 + g)
    p["dr_motor_gains"]["kd_range"] = (1 - g, 1 + g)
    p["dr_motor_strength"]["effort_limit_range"] = (r.effort_scale_low[1], 1.0)
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = replace(robot_cfg.articulation)
  return cfg


def evaluate(sw: KneelSwitcher, wenv, args) -> None:
  """Walk 3 s, K, kneel, hold 4 s (push), U, stand, walk 3 s; report."""
  dt, n = sw.dt, sw.n
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  sw.reset()
  total = int((1.0 + 3.0 + 2.0 + args.kneel_s + 4.0 + args.kneel_s + 2.0 + 3.0) / dt)
  t_k, t_u = None, None
  pushed = False
  with torch.inference_mode():
    for step in range(total):
      t = step * dt
      if 1.0 <= t < 4.0:
        sw.walk.command[sw.mode == WALK, 0] = 0.5
      if t >= 4.0 and t_k is None:
        sw.press_kneel()
        t_k = t
      kneeling = sw.mode == KNEELING
      if kneeling.all() and t_u is None:
        t_u = t + 4.0
      if args.push and not pushed and t_u is not None and t >= t_u - 2.0:
        sw.push(args.push)
        pushed = True
      if t_u is not None and t >= t_u:
        sw.press_up()
      walking = sw.mode == WALK
      if sw.stats["walking_again"].any():
        sw.walk.command[walking & sw.stats["walking_again"], 0] = 0.5
      obs = wenv.step(sw(obs))[0]
  st = sw.stats
  pct = lambda x: 100.0 * float(x.float().mean())  # noqa: E731
  q = lambda k, p: float(st[k].quantile(p))  # noqa: E731
  failed = st["fell_mode"] >= 0
  where = ", ".join(
    f"{MODE_NAMES[k]} x{int((st['fell_mode'][failed] == k).sum())}"
    f" (phase {float(st['fell_phase'][failed & (st['fell_mode'] == k)].median()):.2f})"
    for k in range(len(MODE_NAMES))
    if bool((st["fell_mode"][failed] == k).any())
  )
  print(f"\nfirst failure by mode: {where or 'none'}")
  if failed.any():
    names = list(sw.env.scene["forbidden_contact"].primary_names)
    bodies: dict[str, int] = {}
    for b in st["fell_body"][failed].tolist():
      key = "tilt" if b == -2 else names[b]
      bodies[key] = bodies.get(key, 0) + 1
    times = st["fell_time"][failed]
    print(
      f"first failure body: {bodies}; time median {float(times.median()):.1f} s "
      f"(min {float(times.min()):.1f}, max {float(times.max()):.1f})"
    )
  print(
    f"\n=== kneel chain, {n} robots, {'randomized' if args.robust else 'nominal'}"
    f"{f', {args.push} m/s push while kneeling' if args.push else ''} ===\n"
    f"reached the kneel:            {pct(st['kneeled']):5.1f}%   joint RMS err "
    f"{float(st['kneel_err'][st['kneeled']].median()) if st['kneeled'].any() else float('nan'):.3f} rad\n"
    f"stood back up:                {pct(st['stood']):5.1f}%\n"
    f"walking again:                {pct(st['walking_again']):5.1f}%\n"
    f"fell:                         {pct(st['fell']):5.1f}%\n"
    f"forbidden body on the ground: {pct(st['forbidden']):5.1f}%\n"
    f"peaks (median / p95 over robots), kneel policy in charge:\n"
    f"  motor demand / limit {q('max_motor_ratio', 0.5):.2f} / {q('max_motor_ratio', 0.95):.2f}\n"
    f"  braced wrist {q('max_braced_nm', 0.5):.1f} / {q('max_braced_nm', 0.95):.1f} Nm (limit 5)\n"
    f"  joint speed {q('max_joint_speed', 0.5):.1f} / {q('max_joint_speed', 0.95):.1f} rad/s\n"
    f"  knee ground force {q('max_knee_force', 0.5):.0f} / {q('max_knee_force', 0.95):.0f} N, "
    f"force rise {q('max_force_rise', 0.5):.0f} / {q('max_force_rise', 0.95):.0f} N/step\n"
  )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--walk-checkpoint", default=WALK_CHECKPOINT)
  ap.add_argument("--kneel-checkpoint", default="latest", help="'none': schedule only")
  ap.add_argument("--kneel-s", type=float, default=6.0, help="down / up duration")
  ap.add_argument("--blend-s", type=float, default=0.5)
  ap.add_argument("--dwell-s", type=float, default=0.5)
  ap.add_argument("--still-joint-speed", type=float, default=0.6)
  ap.add_argument("--eval", type=int, default=0, help="headless test with N robots")
  ap.add_argument("--robust", action="store_true", help="level-1 randomization")
  ap.add_argument("--push", type=float, default=0.0, help="eval: kick while kneeling")
  args = ap.parse_args()
  if args.kneel_checkpoint == "latest":
    args.kneel_checkpoint = latest_checkpoint("g1_kneel_29dof")
  elif args.kneel_checkpoint == "none":
    args.kneel_checkpoint = None
  print(
    f"walk: {args.walk_checkpoint}\nkneel: {args.kneel_checkpoint or 'schedule only'}"
  )
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  n = args.eval or 1
  env = ManagerBasedRlEnv(cfg=make_env_cfg(n, args.robust), device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  sw = KneelSwitcher(env, args, device)
  if args.eval:
    evaluate(sw, wenv, args)
    env.close()
    return

  def on_key(key: int) -> None:
    w = sw.walk.command
    walking = int(sw.mode[0]) == WALK
    if key == KEY_K:
      sw.press_kneel()
    elif key == KEY_U:
      sw.press_up()
    elif key == KEY_P:
      sw.push(0.4)
    elif key == KEY_X:
      w[:] = 0.0
    elif walking:
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
    print(sw.status(), end="", flush=True)

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
