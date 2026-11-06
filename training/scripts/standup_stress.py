"""Stress test of the chain's stand <-> crawl transitions, one condition per group.

The play_chain loop (stand, down, crawl, up, walk) with the deploy controller's
PD-target clamp and its 8 s transitions, on N robots split into groups. Each
group gets one fixed perturbation, most of them past the training ranges, so
the report shows which margin runs out first:

  uv run python scripts/standup_stress.py --stand-crawl-checkpoint <pt> \
    --crawl-checkpoint <pt> [--group 48] [--carpet]

Success is "walking again": the stand-up finished and the robot passed the
stand gate (upright, tall, still) and walked off. ``stuck`` counts robots that
finished the up schedule but never got there.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_chain import (  # noqa: E402
  CRAWL,
  CRAWL_STOP,
  DOWN,
  MODE_NAMES,
  STAND_HOLD,
  UP,
  WALK,
  ChainSwitcher,
  add_chain_args,
  make_chain_cfg,
  resolve_checkpoints,
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp.dr import (  # noqa: E402
  body_com_offset,
  body_mass,
  effort_limits,
  joint_friction,
  pd_gains,
)
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.g1_crawl.config.g1.env_cfgs import LIMB_MATERIALS_29  # noqa: E402
from mjlab.tasks.g1_crawl.mdp.thermal import floor_friction  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import (  # noqa: E402
  CRAWL_SELF_COLLISION_29_TORSION,
  deploy_target_clip,
)
from mjlab.tasks.g1_walk.robot import STAND_HEIGHT  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402


@dataclass
class Condition:
  name: str
  effort: float = 1.0
  leg_effort: float = 1.0  # Hip pitch and knee motors only, on top of ``effort``.
  payload_kg: float = 0.0
  mass_scale: float = 1.0
  kp_scale: float = 1.0
  joint_friction_nm: float = 0.0
  floor: float = 0.8
  delay_steps: int = 0  # Whole 20 ms control steps.
  lag_ms: int | None = None  # Fixed motor command lag (else 0-20 ms, as trained).
  twist_m: float | None = None  # Limb twist friction (needs --carpet).
  push: float = 0.0  # m/s, once going down and once standing up.
  com_x_m: float = 0.0  # Torso centre of mass shifted forward (+) or back.
  arm_end_kg: float = 0.0  # Added to each forearm extension (brace end).
  ankle_effort: float = 1.0  # Ankle motors only, on top of ``effort``.


CONDITIONS = (
  Condition("nominal"),
  Condition("effort 0.75", effort=0.75),
  Condition("effort 0.6", effort=0.6),
  Condition("effort 0.5", effort=0.5),
  Condition("legs 0.6", leg_effort=0.6),
  Condition("payload +5", payload_kg=5.0),
  Condition("payload +8", payload_kg=8.0),
  Condition("mass x1.2", mass_scale=1.2),
  Condition("kp x0.7", kp_scale=0.7),
  Condition("kp x0.8", kp_scale=0.8),
  Condition("kp x1.3", kp_scale=1.3),
  Condition("jfric 1.5", joint_friction_nm=1.5),
  Condition("floor 0.25", floor=0.25),
  Condition("floor 1.5", floor=1.5),
  Condition("delay 20ms", delay_steps=1),
  Condition("delay 40ms", delay_steps=2),
  Condition("carpet 1.0", floor=1.0, twist_m=0.01),
  Condition("carpet 1.3", floor=1.3, twist_m=0.02),
  Condition("carpet 1.5", floor=1.5, twist_m=0.03),
  Condition("carpet 1.3 heavy", floor=1.3, twist_m=0.02, effort=0.8, payload_kg=3.0),
  Condition("lag 30ms", lag_ms=30),
  Condition("lag 40ms", lag_ms=40),
  Condition("lag 50ms", lag_ms=50),
  Condition("lag 60ms", lag_ms=60),
  Condition("push 0.4", push=0.4),
  Condition("com +5cm", com_x_m=0.05),
  Condition("com +8cm", com_x_m=0.08),
  Condition("com -5cm", com_x_m=-0.05),
  Condition("arm end +0.5", arm_end_kg=0.5),
  Condition("arm end +1", arm_end_kg=1.0),
  Condition("ankles 0.5", ankle_effort=0.5),
  Condition(
    "front heavy",
    effort=0.85,
    payload_kg=2.0,
    com_x_m=0.04,
    arm_end_kg=0.5,
    delay_steps=1,
  ),
  Condition(
    "combo",
    effort=0.8,
    payload_kg=3.0,
    kp_scale=0.85,
    joint_friction_nm=0.8,
    floor=0.4,
    delay_steps=1,
  ),
  Condition(
    "combo heavy",
    effort=0.7,
    payload_kg=5.0,
    kp_scale=0.85,
    joint_friction_nm=1.0,
    floor=0.35,
    delay_steps=1,
    push=0.3,
  ),
)


def _sel(env, **kw) -> SceneEntityCfg:
  cfg = SceneEntityCfg("robot", **kw)
  cfg.resolve(env.scene)
  return cfg


def _declare_fields(cfg) -> None:
  """Startup events with neutral values, so each randomized model field gets
  per-robot copies; the groups' values are written once the env exists."""
  robot = SceneEntityCfg("robot")
  neutral = {
    "s_effort": (effort_limits, {"asset_cfg": robot, "effort_limit_range": (1.0, 1.0)}),
    "s_mass": (
      body_mass,
      {"asset_cfg": SceneEntityCfg("robot", body_names=(r".*",)), "ranges": (1.0, 1.0)},
    ),
    "s_com": (
      body_com_offset,
      {
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
        "ranges": (0.0, 0.0),
      },
    ),
    "s_gains": (
      pd_gains,
      {"asset_cfg": robot, "kp_range": (1.0, 1.0), "kd_range": (1.0, 1.0)},
    ),
    "s_jfric": (
      joint_friction,
      {
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
        "ranges": (0.0, 0.0),
      },
    ),
    "s_floor": (floor_friction, {"ranges": (0.8, 0.8), "materials": LIMB_MATERIALS_29}),
  }
  for name, (func, params) in neutral.items():
    cfg.events[name] = EventTermCfg(func=func, mode="startup", params=params)


def _apply(env: ManagerBasedRlEnv, ids: torch.Tensor, c: Condition) -> None:
  effort_limits(env, ids, (c.effort, c.effort), asset_cfg=_sel(env))
  mjm = env.sim.mj_model
  default = env.sim.get_default_field("actuator_forcerange")
  for keys, scale in (
    (("hip_pitch", "knee"), c.leg_effort),
    (("ankle",), c.ankle_effort),
  ):
    if scale == 1.0:
      continue
    found = [i for i in range(mjm.nu) if any(k in mjm.actuator(i).name for k in keys)]
    ctrl = torch.tensor(found, device=env.device)
    env.sim.model.actuator_forcerange[ids[:, None], ctrl] = (
      default[ctrl] * c.effort * scale
    )
  body_mass(env, ids, (c.mass_scale,) * 2, asset_cfg=_sel(env, body_names=(r".*",)))
  if c.payload_kg:
    torso = _sel(env, body_names=("torso_link",))
    body_mass(env, ids, (c.payload_kg,) * 2, asset_cfg=torso, operation="add")
  if c.com_x_m:
    torso = _sel(env, body_names=("torso_link",))
    body_com_offset(env, ids, (c.com_x_m,) * 2, asset_cfg=torso, axes=[0])
  if c.arm_end_kg:
    ends = _sel(env, body_names=(r".*_forearm_extension",))
    body_mass(env, ids, (c.arm_end_kg,) * 2, asset_cfg=ends, operation="add")
  pd_gains(env, ids, (c.kp_scale,) * 2, (c.kp_scale,) * 2, asset_cfg=_sel(env))
  joint_friction(
    env, ids, (c.joint_friction_nm,) * 2, asset_cfg=_sel(env, joint_names=(r".*",))
  )
  twist = None if c.twist_m is None else (c.twist_m, c.twist_m)
  floor_friction(
    env, ids, ranges=(c.floor, c.floor), materials=LIMB_MATERIALS_29, torsional=twist
  )


MAX_LAG = 12  # Physics steps (60 ms).


def _fix_lags(env: ManagerBasedRlEnv, conds, g: int) -> None:
  """One motor command lag per robot for the whole run: the condition's, or a
  draw in 0-20 ms (the trained range) for the others."""
  dt_ms = env.physics_dt * 1000.0
  lags = torch.randint(0, 5, (env.num_envs,), device=env.device)
  for k, c in enumerate(conds):
    if c.lag_ms is not None:
      lags[k * g : (k + 1) * g] = round(c.lag_ms / dt_ms)
  robot = env.scene["robot"]
  buffers = {id(a._delay_buffer): a._delay_buffer for a in robot.actuators}
  for buffer in buffers.values():
    if buffer is None:
      continue
    buffer.min_lag, buffer.max_lag = 0, MAX_LAG
    buffer.set_lags(lags)

    def hold(b=buffer) -> None:
      b._step_count += 1

    buffer._update_lags = hold


def run(sw: ChainSwitcher, wenv, args, conds, g: int) -> None:
  """Stand 2 s, C, down, hold, crawl, U, up, walk 0.3 m/s."""
  dt, n, dev = sw.dt, sw.n, sw.device
  group = torch.arange(n, device=dev) // g
  delay = torch.tensor([c.delay_steps for c in conds], device=dev)[group]
  push = torch.tensor([c.push for c in conds], device=dev)[group]
  hist: list[torch.Tensor | None] = [None, None, None]
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  sw.reset()
  T = args.transition_s
  total = int((2 + 2 + T + 3 + args.eval_crawl_s + 3 + T + 2 + 8) / dt)
  up_t0 = torch.full((n,), -1.0, device=dev)
  up_done = torch.full((n,), -1.0, device=dev)
  walk_t = torch.full((n,), -1.0, device=dev)
  pushed_dn = torch.zeros(n, dtype=torch.bool, device=dev)
  pushed_up = torch.zeros(n, dtype=torch.bool, device=dev)
  robot = sw.robot
  zeros = lambda: torch.zeros(n, device=dev)  # noqa: E731
  sh_n, sh_still, sh_up, sh_tall, sh_slow, sh_feet = (zeros() for _ in range(6))
  with torch.inference_mode():
    for step in range(total):
      m = sw.mode
      if step == int(2.0 / dt):
        sw.press_c()
      crawling = m == CRAWL
      since = sw.time - sw.mode_t0
      speed = torch.where(since[crawling] > 1.0, args.eval_crawl_speed, 0.0)
      sw.crawl.command[crawling, 0] = speed
      up_now = crawling & (since > args.eval_crawl_s)
      if up_now.any():
        sw.crawl.command[up_now, :3] = 0.0
        sw._set(up_now, CRAWL_STOP, blend=False)
      again = (m == WALK) & sw.stats["walking_again"]
      sw.walk.command[again, 0] = 0.3
      kick = (push > 0) & (
        ((m == DOWN) & (sw.trans.phase > 0.5) & ~pushed_dn)
        | ((m == UP) & (sw.trans.phase < 0.6) & ~pushed_up)
      )
      if kick.any():
        vel = robot.data.root_link_vel_w.clone()
        d = torch.randn(n, 2, device=dev)
        d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6) * push[:, None]
        vel[:, :2] += d * kick[:, None].float()
        robot.write_root_link_velocity_to_sim(vel)
        pushed_dn |= kick & (m == DOWN)
        pushed_up |= kick & (m == UP)
      target = sw(obs)
      hist = [target.clone(), *hist[:-1]]
      sent = target
      for k in (1, 2):
        old = hist[k]
        if old is not None:
          sent = torch.where((delay == k)[:, None], old, sent)
      obs = wenv.step(sent)[0]
      m = sw.mode
      up_t0 = torch.where((m == UP) & (up_t0 < 0), sw.time, up_t0)
      up_done = torch.where((m == STAND_HOLD) & (up_done < 0), sw.time, up_done)
      back = (m == WALK) & sw.stats["walking_again"] & (walk_t < 0)
      walk_t = torch.where(back, sw.time, walk_t)
      hold = m == STAND_HOLD
      if hold.any():
        d = robot.data
        sh_n += hold.float()
        sh_still += (hold & (d.joint_vel.abs().amax(dim=1) < 0.6)).float()
        sh_up += (hold & (d.projected_gravity_b[:, 2] < -0.97)).float()
        sh_tall += (hold & (d.root_link_pos_w[:, 2] > STAND_HEIGHT - 0.06)).float()
        sh_slow += (hold & (d.root_link_lin_vel_w[:, :2].norm(dim=1) < 0.1)).float()
        sh_feet += (hold & sw.contacts()[:, :2].all(dim=1)).float()
  st = sw.stats
  print(
    f"\n=== stand <-> crawl stress, {g} robots per condition, T {T:.0f} s, crawl "
    f"{args.eval_crawl_speed} m/s for {args.eval_crawl_s:.0f} s"
    f"{', carpet contacts' if args.carpet else ''} ===\n"
    f"{'condition':14s} {'crawl':>6s} {'up':>6s} {'walk':>6s} {'fail':>6s} "
    f"{'stuck':>6s} {'up->walk':>9s}  first failures"
  )
  for k, c in enumerate(conds):
    s = slice(k * g, (k + 1) * g)
    pct = lambda x: 100.0 * float(x.float().mean())  # noqa: E731
    failed = st["fail_mode"][s] >= 0
    stuck = (up_done[s] >= 0) & (walk_t[s] < 0) & ~failed
    dur = (walk_t[s] - up_t0[s])[walk_t[s] >= 0]
    modes: dict[str, int] = {}
    for fm, fb in zip(
      st["fail_mode"][s][failed].tolist(),
      st["fail_body"][s][failed].tolist(),
      strict=True,
    ):
      body = "tilt" if fb < 0 else sw.forbidden_names[fb].replace("_link", "")
      key = f"{MODE_NAMES[fm]}:{body}"
      modes[key] = modes.get(key, 0) + 1
    if bool(stuck.any()):

      def fr(x: torch.Tensor, s=s, k=stuck) -> float:
        return float((x[s][k] / sh_n[s][k].clamp(min=1)).mean())

      why = (
        f"   [stuck in STAND_HOLD: still {fr(sh_still):.0%}, upright {fr(sh_up):.0%}, "
        f"tall {fr(sh_tall):.0%}, slow {fr(sh_slow):.0%}, feet {fr(sh_feet):.0%}]"
      )
    else:
      why = ""
    print(
      f"{c.name:14s} {pct(st['reached_crawl'][s]):5.0f}% {pct(st['began_up'][s]):5.0f}% "
      f"{pct(st['walking_again'][s]):5.0f}% {pct(failed):5.0f}% {pct(stuck):5.0f}% "
      f"{float(dur.median()) if len(dur) else float('nan'):8.1f}s  "
      + ", ".join(f"{k} x{v}" for k, v in sorted(modes.items(), key=lambda x: -x[1]))
      + why
    )


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  add_chain_args(ap)
  ap.add_argument("--group", type=int, default=48)
  ap.add_argument("--carpet", action="store_true", help="condim-4 limb contacts")
  ap.add_argument("--only", nargs="*", default=None, help="condition names")
  ap.set_defaults(transition_s=8.0, eval_crawl_s=8.0, eval_crawl_speed=0.3)
  args = ap.parse_args()
  resolve_checkpoints(args)
  configure_torch_backends()
  device = "cuda:0"
  conds = [c for c in CONDITIONS if args.only is None or c.name in args.only]
  g = args.group
  cfg = make_chain_cfg(g * len(conds), args)
  cfg.actions["joint_pos"].clip = deploy_target_clip()
  if args.carpet:
    cfg.scene.entities["robot"].collisions = (CRAWL_SELF_COLLISION_29_TORSION,)
    cfg.sim.njmax = (cfg.sim.njmax or 300) + 100
  _declare_fields(cfg)
  robot_cfg = cfg.scene.entities["robot"]
  art = robot_cfg.articulation
  robot_cfg.articulation = replace(
    art, actuators=tuple(replace(a, delay_max_lag=MAX_LAG) for a in art.actuators)
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  for k, c in enumerate(conds):
    _apply(env, torch.arange(k * g, (k + 1) * g, device=device), c)
  _fix_lags(env, conds, g)
  sw = ChainSwitcher(env, args, device)
  run(sw, wenv, args, conds, g)
  env.close()


if __name__ == "__main__":
  main()
