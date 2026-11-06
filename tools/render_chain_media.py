"""Render the whole chain (walk -> stand-to-crawl -> crawl -> crawl-to-stand -> walk)
from the deploy bundle, with the same CPU MuJoCo + ONNX loop as
``deploy/sim2sim_chain_cpu.py``, to an MP4 with the controller state written on
every frame. Also saves the joint torques/speeds for ``tools/plot_motor_heat.py``.

  MUJOCO_GL=egl python tools/render_chain_media.py --out media/sim/full_chain
  ... --crawl-onnx crawl_scuttle.onnx --out media/sim/full_chain_scuttle

Needs mujoco, onnxruntime, numpy, pillow and imageio-ffmpeg.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))
import sim2sim_chain_cpu as s2s  # noqa: E402

LABEL = {
  s2s.WALK: "WALK",
  s2s.STOPPING: "STOPPING",
  s2s.DOWN: "STAND -> CRAWL",
  s2s.HOLD: "HOLD (crawl pose)",
  s2s.CRAWL: "CRAWL",
  s2s.CRAWL_STOP: "CRAWL STOP",
  s2s.UP: "CRAWL -> STAND",
  s2s.STAND_HOLD: "STAND",
}


def font(size: int) -> ImageFont.ImageFont:
  for f in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
  ):
    if Path(f).exists():
      return ImageFont.truetype(f, size)
  return ImageFont.load_default()


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--bundle", default=str(ROOT / "deploy" / "g1_chain_v3_1"))
  ap.add_argument("--crawl-onnx", default="crawl.onnx")
  ap.add_argument("--crawl-name", default="creepy")
  ap.add_argument("--out", required=True, help="path prefix, no extension")
  ap.add_argument("--width", type=int, default=960)
  ap.add_argument("--height", type=int, default=540)
  ap.add_argument("--fps", type=int, default=25)
  ap.add_argument("--transition-s", type=float, default=8.0)
  ap.add_argument("--hold-s", type=float, default=0.0, help="still crawl hold first")
  ap.add_argument("--no-video", action="store_true")
  args = ap.parse_args()

  bundle = Path(args.bundle)
  ch = s2s.Chain(bundle, args.crawl_onnx, args.transition_s)
  ch.relaxed, ch.hold_timeout = True, 8.0
  kp = np.array([ch.cfg["pd"][j]["kp"] for j in ch.joints])
  kd = np.array([ch.cfg["pd"][j]["kd"] for j in ch.joints])
  robot = s2s.Robot(
    bundle / "mujoco" / "scene.xml",
    ch.joints,
    kp,
    kd,
    SimpleNamespace(effort=1.0, pd_1khz=False, payload=0.0, delay_steps=0),
  )
  m, d = robot.model, robot.data
  m.vis.global_.offwidth = max(m.vis.global_.offwidth, args.width)
  m.vis.global_.offheight = max(m.vis.global_.offheight, args.height)
  renderer = None if args.no_video else mujoco.Renderer(m, args.height, args.width)
  cam = mujoco.MjvCamera()
  cam.elevation = -16.0
  big, small = font(args.height // 18), font(args.height // 32)
  writer = None
  if renderer is not None:
    writer = imageio_ffmpeg.write_frames(
      args.out + ".mp4",
      (args.width, args.height),
      fps=args.fps,
      codec="libx264",
      quality=None,
      output_params=["-crf", "23", "-preset", "slow", "-pix_fmt", "yuv420p"],
      macro_block_size=8,
    )
    writer.send(None)

  # Script: (seconds into the mode, command) per mode, applied every tick.
  t, t_mode_start, last_mode = 0.0, 0.0, -1
  stood_up = False
  t_walk2 = None
  every = 1.0 / args.fps
  next_frame = 0.0
  heading_lp = pz_lp = None
  modes, taus, dqs, ts, frames_mode = [], [], [], [], []
  while True:
    mo = ch.mode
    if mo != last_mode:
      t_mode_start, last_mode = t, mo
      print(f"t {t:6.2f}  {s2s.MODE_NAMES[mo]}", flush=True)
    tm = t - t_mode_start
    sub = ""
    if mo == s2s.WALK and not stood_up:
      if t < 1.0:
        ch.walk_cmd[:] = 0.0
      elif t < 4.0:
        ch.walk_cmd[:] = (0.5, 0.0, 0.0)
      elif t < 6.5:
        ch.walk_cmd[:] = (0.3, 0.0, 0.5)
      else:
        ch.press_c()
      v, wz = ch.walk_cmd[0], ch.walk_cmd[2]
      sub = f"vx {v:.1f} m/s  wz {wz:.1f} rad/s" if t >= 1.0 else "standing"
    elif mo == s2s.CRAWL:
      c = ch.crawl_cmd
      if tm < args.hold_s + 1.0:
        c[:3] = 0.0
      elif tm < args.hold_s + 7.0:
        c[:3] = (0.3, 0.0, 0.0)
      elif tm < args.hold_s + 10.0:
        c[:3] = (0.15, 0.0, 0.5)
      elif tm < args.hold_s + 13.0:
        c[:3] = (0.3, 0.0, 0.0)
      else:
        ch.press_u()
      sub = f"{args.crawl_name}  fwd {c[0]:.2f} m/s  turn {c[2]:.1f}"
    elif mo in (s2s.DOWN, s2s.UP):
      sub = f"phase {ch.phase:.2f}  ({args.transition_s:.0f} s)"
    elif mo == s2s.STAND_HOLD:
      stood_up = True
    if mo == s2s.WALK and stood_up:
      t_walk2 = t if t_walk2 is None else t_walk2
      ch.walk_cmd[:] = (0.4, 0.0, 0.0)
      sub = "vx 0.4 m/s"
      if t - t_walk2 > 4.0:
        break
    if t > 120.0:
      print("timeout")
      break

    tg = ch.step(
      robot.quat(),
      robot.gyro(),
      robot.q(),
      robot.qd(),
      robot.above_limbs(),
      robot.pelvis_over_feet(),
    )
    robot.step(tg)
    t += ch.dt
    ts.append(t)
    modes.append(ch.mode)
    taus.append(d.actuator_force[robot.act].copy())
    dqs.append(robot.qd())
    g = s2s.gravity_in_body(robot.quat())
    if g[2] > 0.5 or abs(g[1]) > 0.8:
      print(f"FELL at t {t:.2f} in {s2s.MODE_NAMES[ch.mode]}")
      break

    if renderer is not None and t >= next_frame:
      next_frame += every
      # Camera follows the pelvis from its left-front, heading low-passed.
      w, x, y, z = d.qpos[3:7]
      yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
      heading_lp = yaw if heading_lp is None else heading_lp + 0.03 * (
        (yaw - heading_lp + 180) % 360 - 180
      )
      cam.azimuth = heading_lp + 120.0
      # Closer and lower when crawling (pelvis 0.2-0.4 m) than standing (0.76 m).
      pz_lp = d.qpos[2] if pz_lp is None else pz_lp + 0.05 * (d.qpos[2] - pz_lp)
      cam.distance = 2.0 + 1.4 * pz_lp
      cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.55 * pz_lp + 0.2]
      renderer.update_scene(d, camera=cam)
      img = Image.fromarray(renderer.render())
      dr = ImageDraw.Draw(img)
      pad = args.height // 40
      dr.rectangle((0, 0, args.width, pad * 2 + big.size + small.size + 6), fill=(20, 20, 24))
      dr.text((pad, pad), LABEL[ch.mode], font=big, fill=(255, 255, 255))
      dr.text((pad, pad + big.size + 6), sub, font=small, fill=(190, 200, 210))
      note = f"MuJoCo sim-to-sim, ONNX policies, real time  t {t:5.1f} s"
      tw = dr.textlength(note, font=small)
      dr.text((args.width - tw - pad, pad), note, font=small, fill=(150, 160, 170))
      writer.send(np.ascontiguousarray(np.asarray(img)))
      frames_mode.append(ch.mode)
  if writer is not None:
    writer.close()
  np.savez(
    args.out + "_log.npz",
    t=np.array(ts),
    mode=np.array(modes),
    tau=np.array(taus),
    dq=np.array(dqs),
    joints=np.array(ch.joints),
    frame_mode=np.array(frames_mode),
    fps=args.fps,
  )
  print(f"done: {t:.1f} s, {len(frames_mode)} frames")


if __name__ == "__main__":
  main()
