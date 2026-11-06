"""Evaluate a crawl checkpoint: limb-contact statistics, orientation, speed, frames."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.g1_crawl.mdp import CrawlVelocityCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

DEFAULT_TASK = "Mjlab-Crawl-Flat-Unitree-G1"

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--task", default=DEFAULT_TASK)
ap.add_argument(
  "--step-height", type=float, default=None, help="commanded swing height (cmd7 tasks)"
)
ap.add_argument("--steps", type=int, default=500)
ap.add_argument("--num-envs", type=int, default=32)
ap.add_argument(
  "--cmd",
  type=float,
  nargs=3,
  default=[1.0, 0.0, 0.0],
  help="[lin_z(fwd), lin_y, ang_x] or 'nan' to keep sampled",
)
ap.add_argument("--out", default=None)
ap.add_argument("--render", action="store_true")
args = ap.parse_args()
device = "cuda:0"
env_cfg = load_env_cfg(args.task, play=True)
agent_cfg = load_rl_cfg(args.task)
env_cfg.scene.num_envs = args.num_envs
env_cfg.viewer.height, env_cfg.viewer.width = 360, 480
env = ManagerBasedRlEnv(
  cfg=env_cfg, device=device, render_mode="rgb_array" if args.render else None
)
wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = MjlabOnPolicyRunner(wenv, asdict(agent_cfg), device=device)
runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.scene["robot"]
crawl = env.scene["crawl_contacts"]
bodyc = env.scene["body_ground_contacts"]
limb_names = crawl.primary_names
pelvis_id = robot.find_bodies(["pelvis"])[0][0]
excluded = {
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_forearm",
  "right_forearm",
}
und_ids = [i for i, n in enumerate(bodyc.primary_names) if n not in excluded]
cmd_term = env.command_manager.get_term("base_velocity")
assert isinstance(cmd_term, CrawlVelocityCommand)
fixed_cmd = None if any(np.isnan(args.cmd)) else torch.tensor(args.cmd, device=device)
has_step = cmd_term.vel_command_b.shape[1] >= 4
height_sensor = (
  env.scene["limb_height_scan"] if "limb_height_scan" in env.scene.sensors else None
)

res = wenv.get_observations()
obs = res[0] if isinstance(res, tuple) else res
stats = {
  k: []
  for k in (
    "contact",
    "force",
    "grav_x",
    "pelvis_z",
    "fwd_vel_b",
    "xy_speed_w",
    "undesired",
    "act_abs",
    "cmd_fwd",
  )
}
frames = []
with torch.inference_mode():
  for t in range(args.steps):
    if fixed_cmd is not None:
      cmd_term.vel_command_b[:, :3] = fixed_cmd
      cmd_term.is_standing_env[:] = False
    if has_step and args.step_height is not None:
      cmd_term.vel_command_b[:, 3] = args.step_height
      cmd_term.vel_command_b[:, 4:] = 0.0
    act = policy(obs)
    out = wenv.step(act)
    obs = out[0]
    found = crawl.data.found
    force = crawl.data.force
    body_found = bodyc.data.found
    assert found is not None and force is not None and body_found is not None
    stats["contact"].append((found > 0).float().mean(0).cpu().numpy())
    stats["force"].append(torch.linalg.vector_norm(force, dim=-1).mean(0).cpu().numpy())
    stats["grav_x"].append(robot.data.projected_gravity_b[:, 0].mean().item())
    stats["pelvis_z"].append(
      (robot.data.body_link_pos_w[:, pelvis_id, 2] - env.scene.env_origins[:, 2])
      .mean()
      .item()
    )
    stats["fwd_vel_b"].append(robot.data.root_link_lin_vel_b[:, 2].mean().item())
    stats["xy_speed_w"].append(
      torch.linalg.vector_norm(robot.data.root_link_lin_vel_w[:, :2], dim=-1)
      .mean()
      .item()
    )
    stats["undesired"].append((body_found[:, und_ids] > 0).float().sum(1).mean().item())
    stats["act_abs"].append(act.abs().mean().item())
    stats["cmd_fwd"].append(cmd_term.vel_command_b[:, 0].mean().item())
    if height_sensor is not None:
      h = height_sensor.data.heights
      swing = (found == 0).float()  # Limb off the ground.
      peak = torch.where(swing > 0, h, torch.zeros_like(h))
      stats.setdefault("swing_peak", []).append(peak.amax(0).cpu().numpy())
    if args.render and t % 2 == 0:
      fr = env.render()
      if fr is not None:
        frames.append(np.asarray(fr))
S = {k: np.array(v) for k, v in stats.items()}
half = args.steps // 2
print(
  f"\n=== {Path(args.checkpoint).name}  ({args.num_envs} envs, {args.steps} steps = {args.steps * env.step_dt:.0f}s, cmd={'sampled' if fixed_cmd is None else args.cmd}) ==="
)
print(
  "contact fraction per limb (2nd half):",
  {n: round(float(S["contact"][half:, i].mean()), 2) for i, n in enumerate(limb_names)},
)
print(
  "mean |ground force| N per limb (2nd half):",
  {n: round(float(S["force"][half:, i].mean()), 1) for i, n in enumerate(limb_names)},
)
hands = [i for i, n in enumerate(limb_names) if "forearm" in n]
feet = [i for i, n in enumerate(limb_names) if "ankle" in n]
tot = S["force"][half:].sum(1).mean()
print(
  f"hand share of total ground force: {S['force'][half:, hands].sum(1).mean() / max(tot, 1e-6):.2f}  (feet {S['force'][half:, feet].sum(1).mean() / max(tot, 1e-6):.2f})"
)
print(
  f"projected gravity x (1 = prone): mean {S['grav_x'].mean():.3f}  min {S['grav_x'].min():.3f}"
)
print(
  f"pelvis height: mean {S['pelvis_z'].mean():.3f} m   forward base-z vel: {S['fwd_vel_b'][half:].mean():.2f} m/s (cmd {S['cmd_fwd'][half:].mean():.2f})   world xy speed: {S['xy_speed_w'][half:].mean():.2f} m/s"
)
print(
  f"undesired body contacts (bodies/step): {S['undesired'].mean():.3f}    mean |action|: {S['act_abs'].mean():.2f}"
)
if args.render and frames:
  from PIL import Image

  out = Path(args.out or Path(args.checkpoint).parent / "eval")
  out.mkdir(parents=True, exist_ok=True)
  pick = np.linspace(0, len(frames) - 1, 12).astype(int)
  tiles = [Image.fromarray(frames[i]) for i in pick]
  w, h = tiles[0].size
  sheet = Image.new("RGB", (w * 4, h * 3))
  for k, im in enumerate(tiles):
    sheet.paste(im, ((k % 4) * w, (k // 4) * h))
  sheet.save(out / f"{Path(args.checkpoint).stem}_sheet.png")
  gif = [Image.fromarray(f) for f in frames[::2]]
  gif[0].save(
    out / f"{Path(args.checkpoint).stem}.gif",
    save_all=True,
    append_images=gif[1:],
    duration=80,
    loop=0,
  )
  print(
    "saved",
    out / f"{Path(args.checkpoint).stem}_sheet.png",
    "and .gif",
    f"({len(frames)} frames)",
  )
env.close()
