"""Long mixed mission through the whole chain, with every motor's temperature.

Each robot runs walk -> stand-to-crawl -> crawl -> crawl-to-stand -> walk, over
and over, with fresh random commands every few seconds (walk up to 0.6 m/s and
turns; crawl 0.1-0.35 m/s, sideways, turns and rests), for ``--minutes``. The
PD targets are clamped to the joint ranges as the robot controller does. A
two-node winding/housing heat model per motor (``g1_motor_safety.motor_model``,
with fans) runs on the actuator torques and joint speeds; ``--pessimism``
multiplies the copper loss (the first robot test heated about 2x faster than
that model). Robots that fall are counted and leave the temperature statistics
from then on.

  uv run python scripts/mission_thermal.py --crawl-checkpoint <pt> \
      --stand-crawl-checkpoint <pt> --minutes 40 --num-envs 32 [--robust|--hard]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_chain import (  # noqa: E402
  CRAWL,
  CRAWL_STOP,
  DOWN,
  HOLD,
  MODE_NAMES,
  STAND_HOLD,
  STOPPING,
  UP,
  WALK,
  ChainSwitcher,
  add_chain_args,
  make_chain_cfg,
  resolve_checkpoints,
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionActionCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import deploy_target_clip  # noqa: E402
from mjlab.tasks.g1_motor_safety import motor_model as mm  # noqa: E402
from mjlab.tasks.g1_walk.robot import JOINTS_29  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

SHORT = [
  j.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
  for j in JOINTS_29
]
PHASE_OF = {
  WALK: "walk",
  STOPPING: "walk",
  DOWN: "transition",
  HOLD: "transition",
  UP: "transition",
  STAND_HOLD: "transition",
  CRAWL: "crawl",
  CRAWL_STOP: "crawl",
}
PHASES = ("walk", "transition", "crawl")


class HeatModel:
  """Two-node winding/housing model per motor, fans on the winding reading."""

  def __init__(self, n: int, device: str, ambient: float, start: float, k: float):
    motors = [mm.motor_for_joint(j) for j in JOINTS_29]
    fields = ("r25", "kt", "c_w", "r_wh", "c_h", "r_ha", "k_iron", "gear", "q_drv")
    self.p = {
      f: torch.tensor([getattr(m, f) for m in motors], device=device) for f in fields
    }
    self.p["n"] = torch.tensor([float(m.n_motors) for m in motors], device=device)
    self.tau_cont = torch.tensor([m.continuous_torque() for m in motors], device=device)
    self.t_w = torch.full((n, len(motors)), start, device=device)
    self.t_h = self.t_w.clone()
    self.t_amb = torch.tensor(ambient, device=device)
    self.k = k  # Copper-loss multiplier (model pessimism).
    self.k_fan = mm.COOLING["k_fan"]

  def step(self, tau: torch.Tensor, dq: torch.Tensor, dt: float) -> None:
    current = tau / (self.p["n"] * self.p["gear"] * self.p["kt"])
    copper = mm.copper_resistance(self.p["r25"], self.t_w) * current.square()
    power = self.k * copper + self.p["k_iron"] * dq.square()
    gain = 1.0 + self.k_fan * mm.fan_duty(self.t_w)
    self.t_w, self.t_h = mm.thermal_step(
      self.t_w, self.t_h, self.t_amb, power, self.p, dt, gain, self.p["q_drv"]
    )


def main() -> None:
  ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  add_chain_args(ap)
  ap.add_argument("--num-envs", type=int, default=32)
  ap.add_argument("--minutes", type=float, default=40.0)
  ap.add_argument("--walk-s", type=float, default=60.0, help="Walking per cycle.")
  ap.add_argument("--crawl-s", type=float, default=300.0, help="Crawling per cycle.")
  ap.add_argument("--pessimism", type=float, default=2.0)
  ap.add_argument("--ambient", type=float, default=30.0)
  ap.add_argument("--start-temp", type=float, default=40.0)
  ap.add_argument("--push", type=float, default=0.0, help="Random kicks (m/s).")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--json", type=Path, default=None)
  args = ap.parse_args()
  resolve_checkpoints(args)
  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0"
  cfg = make_chain_cfg(args.num_envs, args)
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.clip = deploy_target_clip()  # The robot controller's clamp.
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  sw = ChainSwitcher(env, args, device)
  n, dt = sw.n, sw.dt
  robot = env.scene["robot"]
  act_ids, _ = robot.find_actuators(JOINTS_29, preserve_order=True)
  act_ids = torch.tensor(act_ids, device=device)
  heat = HeatModel(n, device, args.ambient, args.start_temp, args.pessimism)

  def rand(lo: float, hi: float, k: int) -> torch.Tensor:
    return torch.empty(k, device=device).uniform_(lo, hi)

  next_walk = torch.zeros(n, device=device)
  next_crawl = torch.zeros(n, device=device)
  next_push = rand(8.0, 15.0, n)
  alive = torch.ones(n, dtype=torch.bool, device=device)
  peak_w = heat.t_w.clone()
  sq = {p: torch.zeros(n, 25, device=device) for p in PHASES}
  cnt = {p: torch.zeros(n, device=device) for p in PHASES}
  cycles = torch.zeros(n, device=device)
  fail_mode = torch.full((n,), -1, dtype=torch.long, device=device)
  fail_t = torch.full((n,), float("nan"), device=device)
  trace_t, trace_max, trace_r0 = [], [], []
  was_crawl = torch.zeros(n, dtype=torch.bool, device=device)
  phase_modes = {
    ph: torch.tensor([k for k, v in PHASE_OF.items() if v == ph], device=device)
    for ph in PHASES
  }

  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  sw.reset()
  steps = int(args.minutes * 60.0 / dt)
  with torch.inference_mode():
    for step in range(steps):
      t = step * dt
      m = sw.mode
      since = sw.time - sw.mode_t0
      # Walking: new command every 4-10 s; after --walk-s, go down.
      walking = m == WALK
      new = walking & (sw.time >= next_walk)
      if new.any():
        k = int(new.sum())
        vx = torch.where(rand(0, 1, k) < 0.2, 0.0, rand(0.0, 0.6, k))
        vy = torch.where(rand(0, 1, k) < 0.7, 0.0, rand(-0.2, 0.2, k))
        wz = torch.where(rand(0, 1, k) < 0.6, 0.0, rand(-0.5, 0.5, k))
        sw.walk.command[new] = torch.stack([vx, vy, wz], dim=1)
        next_walk[new] = sw.time[new] + rand(4.0, 10.0, k)
      go_down = walking & (since > args.walk_s) & alive
      if go_down.any():
        sw.walk.command[go_down] = 0.0
        sw._set(go_down, STOPPING, blend=False)
      # Crawling: new command every 5-15 s; after --crawl-s, stand up.
      crawling = m == CRAWL
      new = crawling & (sw.time >= next_crawl)
      if new.any():
        k = int(new.sum())
        u = rand(0, 1, k)
        fwd = torch.where(
          u < 0.25, 0.0, torch.where(u < 0.35, -0.1, rand(0.1, 0.35, k))
        )
        lat = torch.where(rand(0, 1, k) < 0.8, 0.0, rand(-0.2, 0.2, k))
        turn = torch.where(rand(0, 1, k) < 0.7, 0.0, rand(-0.4, 0.4, k))
        sw.crawl.command[new, :3] = torch.stack([fwd, lat, turn], dim=1)
        next_crawl[new] = sw.time[new] + rand(5.0, 15.0, k)
      go_up = crawling & (since > args.crawl_s)
      if go_up.any():
        sw.crawl.command[go_up, :3] = 0.0
        sw._set(go_up, CRAWL_STOP, blend=False)
      if args.push:
        kick = alive & (sw.time >= next_push)
        if kick.any():
          sw.push(args.push, kick)
          next_push[kick] = sw.time[kick] + rand(8.0, 15.0, int(kick.sum()))
      cycles += (was_crawl & (m == UP)).float()
      was_crawl = (m == CRAWL) | (m == CRAWL_STOP)

      obs = wenv.step(sw(obs))[0]

      tau = robot.data.actuator_force[:, act_ids]
      dq = robot.data.joint_vel  # Joint order == JOINTS_29 (the switcher checks).
      heat.step(tau, dq, dt)
      failed_now = sw.stats["failed"] & alive
      if failed_now.any():
        fail_mode[failed_now] = sw.mode[failed_now]
        fail_t[failed_now] = t
      alive &= ~sw.stats["failed"]
      peak_w = torch.where(alive[:, None], torch.maximum(peak_w, heat.t_w), peak_w)
      for phase in PHASES:
        mask = alive & torch.isin(m, phase_modes[phase])
        sq[phase][mask] += tau[mask].square()
        cnt[phase][mask] += 1
      if step % int(10.0 / dt) == 0:
        live_tw = heat.t_w[alive] if alive.any() else heat.t_w
        trace_t.append(t / 60.0)
        trace_max.append(float(live_tw.max()))
        trace_r0.append(float(heat.t_w[0].max()))
      if step % int(60.0 / dt) == 0:
        hot = heat.t_w[alive].max(dim=0).values if alive.any() else heat.t_w[0]
        i = int(hot.argmax())
        modes = {MODE_NAMES[k]: int((m == k).sum()) for k in range(9) if (m == k).any()}
        print(
          f"[{t / 60:5.1f} min] alive {int(alive.sum())}/{n}  cycles "
          f"{float(cycles.mean()):.1f}  hottest winding {float(hot.max()):5.1f} C "
          f"({SHORT[i]})  modes {modes}",
          flush=True,
        )

  # Results.
  tau_cont = heat.tau_cont.cpu().numpy()
  ok = alive.cpu().numpy()
  peak = peak_w.cpu().numpy()
  out: dict = {
    "checkpoints": {
      "walk": str(args.walk_checkpoint),
      "stand_crawl": str(args.stand_crawl_checkpoint),
      "crawl": str(args.crawl_checkpoint),
    },
    "minutes": args.minutes,
    "robots": n,
    "randomization": "hard" if args.hard else ("robust" if args.robust else "nominal"),
    "push_mps": args.push,
    "pessimism": args.pessimism,
    "ambient_c": args.ambient,
    "completed_frac": float(ok.mean()),
    "failures": {
      MODE_NAMES[k]: int((fail_mode == k).sum())
      for k in range(9)
      if (fail_mode == k).any()
    },
    "cycles_mean": float(cycles.mean()),
    "peak_winding_c": {
      SHORT[j]: {
        "median": round(float(np.median(peak[:, j])), 1),
        "max": round(float(peak[:, j].max()), 1),
      }
      for j in range(25)
    },
    "rms_x_cont": {},
    "trace": {"minutes": trace_t, "hottest_c": trace_max, "robot0_c": trace_r0},
  }
  for phase in PHASES:
    c = cnt[phase].clamp(min=1).unsqueeze(1)
    rms = (sq[phase] / c).sqrt().mean(dim=0).cpu().numpy()
    out["rms_x_cont"][phase] = {
      SHORT[j]: round(float(rms[j] / tau_cont[j]), 3) for j in range(25)
    }
  print(
    f"\n=== mission {args.minutes:.0f} min, {n} robots, {out['randomization']}, "
    f"push {args.push} m/s, heat x{args.pessimism}, ambient {args.ambient} C ===\n"
    f"completed without a fall: {100 * out['completed_frac']:.1f}%  failures "
    f"{out['failures']}  stand<->crawl cycles per robot {out['cycles_mean']:.1f}"
  )
  order = sorted(range(25), key=lambda j: -float(np.median(peak[:, j])))
  print(f"{'motor':18s} {'peak winding C':>15s}   {'RMS / continuous rating':>30s}")
  print(
    f"{'':18s} {'median':>7s} {'max':>7s}   {'walk':>8s} {'transition':>10s} {'crawl':>8s}"
  )
  for j in order[:12]:
    r = [out["rms_x_cont"][p][SHORT[j]] for p in PHASES]
    print(
      f"{SHORT[j]:18s} {np.median(peak[:, j]):7.1f} {peak[:, j].max():7.1f}   "
      f"{r[0]:8.2f} {r[1]:10.2f} {r[2]:8.2f}"
    )
  if args.json is not None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
