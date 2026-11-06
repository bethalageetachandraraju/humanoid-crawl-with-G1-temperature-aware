"""Flip test for one terrain tile type with a crawl checkpoint (user-run check).

Spawns N envs on a single tile type at a fixed difficulty, runs the policy for a
few seconds with mild commands and no pushes, and reports how many envs rolled
over, the mean forward speed and (optionally) a frame sheet. Zero-ish flips on
the flat box tile is the reference; a tile that flips a large fraction of envs
within seconds is a contact problem, not a learning problem.

Example:
  uv run python scripts/crawl_tile_test.py --checkpoint <model.pt> --tile meadows
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.g1_crawl.config.g1.env_cfgs import CRAWL_ZONES_CFG
from mjlab.tasks.g1_crawl.mdp import CrawlVelocityCommandCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.terrains.config import ALL_TERRAIN_PRESETS

TILES = {name: cfg for name, cfg in CRAWL_ZONES_CFG.sub_terrains.items()}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--task", default="Mjlab-Crawl-Rough-Unitree-G1")
  ap.add_argument(
    "--tile",
    default="flat",
    help="zone name from the crawl layout (" + ", ".join(TILES) + ") or any "
    "mjlab preset name (" + ", ".join(sorted(ALL_TERRAIN_PRESETS)) + ")",
  )
  ap.add_argument("--difficulty", type=float, default=0.5)
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--seconds", type=float, default=6.0)
  ap.add_argument("--render", action="store_true", help="save a frame sheet PNG")
  args = ap.parse_args()

  if args.tile in TILES:
    tile = replace(TILES[args.tile], proportion=1.0)
  else:
    tile = ALL_TERRAIN_PRESETS[args.tile](proportion=1.0)
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  cmd = cfg.commands["base_velocity"]
  assert isinstance(cmd, CrawlVelocityCommandCfg)
  cmd.ang_vel_x = (-1.0, 1.0)
  cfg.events.pop("push_robot", None)
  cfg.curriculum = {}
  assert cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator
  cfg.scene.terrain.terrain_generator = replace(
    cfg.scene.terrain.terrain_generator,
    sub_terrains={"t": tile},
    num_cols=4,
    curriculum=False,
    difficulty_range=(args.difficulty, args.difficulty),
  )
  cfg.scene.terrain.max_init_terrain_level = None
  if args.render:
    cfg.viewer.height, cfg.viewer.width = 360, 480
    cfg.viewer.distance, cfg.viewer.elevation = 3.0, -25.0

  agent = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(
    cfg=cfg, device="cuda:0", render_mode="rgb_array" if args.render else None
  )
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = MjlabOnPolicyRunner(wenv, asdict(agent), device="cuda:0")
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0"
  )
  policy = runner.get_inference_policy(device="cuda:0")
  obs = wenv.get_observations()
  obs = obs[0] if isinstance(obs, tuple) else obs

  robot = env.scene["robot"]
  flipped = 0
  speeds: list[float] = []
  frames: list[np.ndarray] = []
  steps = int(args.seconds / env.step_dt)
  with torch.inference_mode():
    for t in range(steps):
      obs = wenv.step(policy(obs))[0]
      if "flipped" in cfg.terminations:
        flipped += int(env.termination_manager.get_term("flipped").sum())
      speeds.append(robot.data.root_link_lin_vel_b[:, 2].mean().item())
      if args.render and t % max(1, steps // 6) == 0:
        frame = env.render()
        if frame is not None:
          frames.append(np.asarray(frame))
  print(
    f"tile={args.tile} difficulty={args.difficulty}: flips {flipped}/{args.num_envs}"
    f" in {args.seconds:.0f}s ({100 * flipped / args.num_envs:.1f}%),"
    f" mean forward speed {np.mean(speeds):.2f} m/s"
  )
  if frames:
    from PIL import Image

    h, w = frames[0].shape[:2]
    sheet = Image.new("RGB", (w * 3, h * 2))
    for k, f in enumerate(frames[:6]):
      sheet.paste(Image.fromarray(f), ((k % 3) * w, (k // 3) * h))
    out = Path("logs") / f"tile_{args.tile}_{args.difficulty}.png"
    sheet.save(out)
    print(f"saved {out}")
  env.close()


if __name__ == "__main__":
  main()
