"""Record where a crawl policy hands back to the stand <-> crawl policy.

The play_chain loop (hard randomization unless --nominal, the deploy target
clamp): walk, down, crawl with random commands for a random time, U. When the
crawl has settled (CRAWL_STOP -> HOLD) each robot's pose is saved, heading
removed, as a ``gait_animation.v1`` file that ``reset_crawl_start`` can start
episodes from (``stop_frames``):

  uv run python scripts/record_crawl_stops.py --crawl-checkpoint <pt> \
    --fwd -0.2 0.5 --lat 0.25 --turn 0.7 --out <file.json> [--num 1024]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_chain import (  # noqa: E402
  CRAWL,
  CRAWL_STOP,
  HOLD,
  ChainSwitcher,
  add_chain_args,
  make_chain_cfg,
  resolve_checkpoints,
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import deploy_target_clip  # noqa: E402
from mjlab.tasks.g1_walk.robot import JOINTS_29  # noqa: E402
from mjlab.utils.lab_api.math import quat_inv, quat_mul, yaw_quat  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  add_chain_args(ap)
  ap.add_argument("--num", type=int, default=1024)
  ap.add_argument("--fwd", type=float, nargs=2, default=(-0.2, 0.5))
  ap.add_argument("--lat", type=float, default=0.25)
  ap.add_argument("--turn", type=float, default=0.7)
  ap.add_argument("--crawl-s", type=float, nargs=2, default=(3.0, 12.0))
  ap.add_argument("--nominal", action="store_true")
  ap.add_argument("--out", required=True)
  ap.set_defaults(transition_s=8.0)
  args = ap.parse_args()
  args.hard = not args.nominal
  resolve_checkpoints(args)
  configure_torch_backends()
  dev = "cuda:0"
  n = args.num
  cfg = make_chain_cfg(n, args)
  cfg.actions["joint_pos"].clip = deploy_target_clip()
  env = ManagerBasedRlEnv(cfg=cfg, device=dev)
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  sw = ChainSwitcher(env, args, dev)
  robot = sw.robot
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs
  sw.reset()
  u = lambda lo, hi: torch.empty(n, device=dev).uniform_(lo, hi)  # noqa: E731
  crawl_for = u(*args.crawl_s)
  next_cmd = torch.zeros(n, device=dev)
  saved = torch.zeros(n, dtype=torch.bool, device=dev)
  frames = torch.zeros(n, 7 + len(JOINTS_29), device=dev)
  steps = int((4 + args.transition_s + 3 + args.crawl_s[1] + 8) / sw.dt)
  with torch.inference_mode():
    for step in range(steps):
      if step == int(2.0 / sw.dt):
        sw.press_c()
      m = sw.mode.clone()
      since = sw.time - sw.mode_t0
      crawling = m == CRAWL
      new = crawling & (sw.time >= next_cmd)
      if new.any():
        c = sw.crawl.command
        c[new, 0] = u(*args.fwd)[new]
        c[new, 1] = u(-args.lat, args.lat)[new] * (torch.rand(n, device=dev) < 0.4)[new]
        c[new, 2] = (
          u(-args.turn, args.turn)[new] * (torch.rand(n, device=dev) < 0.4)[new]
        )
        next_cmd[new] = sw.time[new] + u(1.5, 4.0)[new]
      stop = crawling & (since > crawl_for)
      if stop.any():
        sw.crawl.command[stop, :3] = 0.0
        sw._set(stop, CRAWL_STOP, blend=False)
      obs = wenv.step(sw(obs))[0]
      handed = (m == CRAWL_STOP) & (sw.mode == HOLD) & ~saved & ~sw.stats["failed"]
      if handed.any():
        d = robot.data
        q = d.root_link_quat_w[handed]
        frames[handed, 2] = (
          d.root_link_pos_w[handed, 2] - env.scene.env_origins[handed, 2]
        )
        frames[handed, 3:7] = quat_mul(quat_inv(yaw_quat(q)), q)
        frames[handed, 7:] = d.joint_pos[handed]
        saved |= handed
  kept = frames[saved].cpu().tolist()
  meta = [{"name": "floating_base_joint", "type": "free", "qposadr": 0, "qposdim": 7}]
  meta += [
    {"name": j, "type": "hinge", "qposadr": 7 + i, "qposdim": 1}
    for i, j in enumerate(JOINTS_29)
  ]
  out = {
    "schema": "gait_animation.v1",
    "dt": sw.dt,
    "fps": 1.0 / sw.dt,
    "nq": 7 + len(JOINTS_29),
    "nv": 6 + len(JOINTS_29),
    "frames": kept,
    "metadata": {
      "joints": meta,
      "recorded_by": "scripts/record_crawl_stops.py",
      "crawl_checkpoint": str(args.crawl_checkpoint),
      "randomization": "nominal" if args.nominal else "hard",
      "note": "crawl -> stand <-> crawl hand-over poses (CRAWL_STOP settled), heading removed",
    },
  }
  Path(args.out).write_text(json.dumps(out))
  z = frames[saved, 2]
  print(
    f"saved {len(kept)} of {n} hand-over poses to {args.out}; pelvis z "
    f"{float(z.mean()):.3f} +- {float(z.std()):.3f} m; failed {int(sw.stats['failed'].sum())}"
  )


if __name__ == "__main__":
  main()
