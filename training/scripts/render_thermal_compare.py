"""Side-by-side video: deployed policy vs thermal fine-tune, same start.

Both run in the thermal fine-tune task (the robot's PD-target clamp), nominal
robot, one robot each. The overlay shows the shoulder-roll torques and the
busiest motor (1 s RMS, in Nm and as a fraction of its estimated continuous
rating).

  uv run python scripts/render_thermal_compare.py --scenario crawl_rest \
      --old <crawl.pt> --new <crawl.pt> --out rest.mp4
  scenarios: crawl_rest, crawl_move (--speed), stand_crawl (down, hold, up)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg  # noqa: E402
import matplotlib  # noqa: E402
import mediapy  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_switcher import load_actor  # noqa: E402

import mjlab.tasks  # noqa: E402, F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.g1_crawl import mdp as crawl_mdp  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import CRAWL_JOINTS_29  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402
from mjlab.viewer.viewer_config import ViewerConfig  # noqa: E402

CRAWL_TASK = "Mjlab-Crawl-Flat-Cmd7-Cool-Unitree-G1-29dof"
STAND_CRAWL_TASK = "Mjlab-StandCrawl-Cool-Unitree-G1-29dof"
FONT = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())  # No system ffmpeg here.
SHORT = [
  j.replace("_joint", "").replace("left_", "L ").replace("right_", "R ")
  for j in CRAWL_JOINTS_29
]


def build(args) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper, str]:
  task = STAND_CRAWL_TASK if args.scenario == "stand_crawl" else CRAWL_TASK
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = 1
  cfg.viewer = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="torso_link",
    distance=args.distance,
    elevation=args.elevation,
    azimuth=args.azimuth,
    width=640,
    height=480,
    max_extra_envs=0,
  )
  cfg.episode_length_s = 1e6
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
  return env, RslRlVecEnvWrapper(env, clip_actions=None), task


def phase_plan(seconds: float) -> list[tuple[str, float, float]]:
  """(label, duration s, dphi/dt) for the stand <-> crawl scenario."""
  return [
    ("standing", 1.0, 0.0),
    ("going down (8 s)", 8.0, 1 / 8),
    (f"holding the crawl pose ({seconds:.0f} s)", seconds, 0.0),
    ("standing up (8 s)", 8.0, -1 / 8),
    ("standing", 2.0, 0.0),
  ]


def run(env, wenv, task, checkpoint, args, label) -> list[np.ndarray]:
  policy, _ = load_actor(task, checkpoint, args.device)
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  obs, _ = wenv.reset()
  robot = env.scene["robot"]
  ids, _ = robot.find_actuators(CRAWL_JOINTS_29, preserve_order=True)
  ratings = crawl_mdp.continuous_ratings(CRAWL_JOINTS_29)
  tau_cont = np.array([ratings[j] for j in CRAWL_JOINTS_29])
  roll = [CRAWL_JOINTS_29.index(f"{s}_shoulder_roll_joint") for s in ("left", "right")]
  crawl = task == CRAWL_TASK
  cmd = env.command_manager.get_term("base_velocity" if crawl else "phase")
  cmd.manual_control()
  if crawl:
    plan = [
      (
        "resting, zero command"
        if args.scenario == "crawl_rest"
        else f"crawling forward {args.speed:.1f} m/s",
        args.seconds,
        0.0,
      )
    ]
  else:
    plan = phase_plan(args.seconds)
  font = ImageFont.truetype(FONT, 17)
  small = ImageFont.truetype(FONT, 15)
  window: list[np.ndarray] = []
  frames: list[np.ndarray] = []
  phase = 0.0
  with torch.no_grad():
    for name, duration, rate in plan:
      for k in range(int(round(duration / env.step_dt))):
        if crawl:
          v = cmd.vel_command_b
          v[:, :3] = 0.0
          if args.scenario == "crawl_move":
            v[:, 0] = args.speed
          v[:, 3] = 0.04
        else:
          phase = min(max(phase + rate * env.step_dt, 0.0), 1.0)
          cmd.manual_phase[:] = phase
          cmd.manual_rate[:] = rate if 0.0 < phase < 1.0 or rate == 0 else 0.0
        obs, _, _, _ = wenv.step(policy(obs["actor"]))
        tau = robot.data.actuator_force[0, ids].cpu().numpy()
        window = (window + [tau**2])[-50:]
        if k % 2:
          continue
        rms = np.sqrt(np.mean(window, axis=0))
        ratio = rms / tau_cont
        top = int(np.argmax(ratio))
        img = Image.fromarray(env.render())
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, 640, 30), fill=(20, 20, 20))
        d.text((10, 5), label, font=font, fill=(255, 255, 255))
        d.rectangle((0, 420, 640, 480), fill=(20, 20, 20))
        d.text((10, 424), name, font=small, fill=(220, 220, 220))
        d.text(
          (10, 449),
          f"shoulder roll L {rms[roll[0]]:4.1f} Nm  R {rms[roll[1]]:4.1f} Nm   "
          f"busiest: {SHORT[top]} {ratio[top]:.2f}x rating",
          font=small,
          fill=(255, 210, 120) if ratio[top] > 1.0 else (160, 230, 160),
        )
        frames.append(np.asarray(img))
  return frames


def main() -> None:
  ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  ap.add_argument("--scenario", choices=("crawl_rest", "crawl_move", "stand_crawl"))
  ap.add_argument("--old", required=True)
  ap.add_argument("--new", required=True)
  ap.add_argument("--old-label", default="OLD: deployed policy")
  ap.add_argument("--new-label", default="NEW: thermal fine-tune")
  ap.add_argument("--seconds", type=float, default=12.0)
  ap.add_argument("--speed", type=float, default=0.3)
  ap.add_argument("--distance", type=float, default=2.2)
  ap.add_argument("--elevation", type=float, default=-15.0)
  ap.add_argument("--azimuth", type=float, default=135.0)
  ap.add_argument("--seed", type=int, default=3)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()

  env, wenv, task = build(args)
  old = run(env, wenv, task, args.old, args, args.old_label)
  new = run(env, wenv, task, args.new, args, args.new_label)
  n = min(len(old), len(new))
  gap = np.full((480, 6, 3), 255, dtype=np.uint8)
  video = [np.concatenate([old[i], gap, new[i]], axis=1) for i in range(n)]
  args.out.parent.mkdir(parents=True, exist_ok=True)
  mediapy.write_video(str(args.out), video, fps=25)
  print(f"{args.out}  {n / 25:.1f} s")


if __name__ == "__main__":
  main()
