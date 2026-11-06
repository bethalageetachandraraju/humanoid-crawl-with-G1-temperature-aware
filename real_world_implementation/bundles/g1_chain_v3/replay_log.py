"""Replay a real-robot log (g1_chain_real.py .npz) in MuJoCo: how the robot moved.

Kinematic playback, no physics: every 20 ms the model takes the logged joint
angles. The base is rebuilt from what the robot measured:
- roll and pitch from the IMU's gravity, yaw by integrating the gyro;
- height: the lowest point of the robot rests on the floor;
- position: leg odometry, the limbs on the floor stay where they are.
Position and heading drift a little over a run; the posture is exact.
Optional ghost (blue, see-through): the PD targets the controller sent, in the
same base pose, so tracking errors show as the gap between the two robots.

  python replay_log.py LOG.npz                      # viewer
  python replay_log.py LOG.npz --start 35 --end 101 --video out.mp4 --ghost

Viewer keys: Space pause, Left/Right -/+ 5 s, Up/Down speed x2 / /2, G ghost,
Enter restart. Needs mujoco and numpy (video: mediapy + ffmpeg, the mjlab venv).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sim2sim_mujoco import load_model  # noqa: E402  (also picks the NVIDIA GPU)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from chain_controller import LIMBS, MODE_NAMES, gravity_in_body  # noqa: E402

GHOST_RGBA = (0.25, 0.55, 1.0, 0.28)
DOWN_TOL = 0.015  # a limb within 1.5 cm of the floor counts as standing on it


def quat_mul(a, b):
  w1, x1, y1, z1 = a
  w2, x2, y2, z2 = b
  return np.array([
    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  ])  # fmt: skip


def quat_roll_pitch(g: np.ndarray) -> np.ndarray:
  """Orientation with zero yaw whose body-frame gravity is ``g``."""
  pitch = math.asin(max(-1.0, min(1.0, float(g[0]))))
  roll = math.atan2(-g[1], -g[2])
  qy = np.array([math.cos(pitch / 2), 0, math.sin(pitch / 2), 0])
  qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
  return quat_mul(qy, qx)


class Replay:
  def __init__(self, log: Path, bundle: Path) -> None:
    self.L = L = np.load(log)
    self.model, _ = load_model(bundle / "mujoco" / "scene.xml")
    m = self.model
    self.data = mujoco.MjData(m)
    self.ghost = mujoco.MjData(m)
    joints = [str(j) for j in L["joints"]]
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{j}") for j in joints]
    self.qadr = np.array([m.jnt_qposadr[i] for i in jid])
    self.t = L["t"] - L["t"][0]
    stamp = log.stem.replace("chain_real_", "").replace("chain_sim_", "")
    try:
      end = datetime.strptime(stamp, "%Y%m%d_%H%M%S")
      self.t0 = end - timedelta(seconds=float(self.t[-1]))
    except ValueError:
      self.t0 = None
    # Collision geoms: limbs (for odometry) and everything (for the floor).
    self.limb_body = [
      mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{n}") for n in LIMBS
    ]
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    self.coll = [
      g for g in range(m.ngeom)
      if g != floor and (m.geom_contype[g] or m.geom_conaffinity[g])
      and m.geom_bodyid[g] != 0
    ]  # fmt: skip
    self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
    self._poses()

  def _lowest(self, d: mujoco.MjData, g: int) -> np.ndarray:
    """Lowest world point of collision geom g (exact for spheres and capsules)."""
    m = self.model
    c = d.geom_xpos[g]
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
      ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
      e = c + ax if (c + ax)[2] < (c - ax)[2] else c - ax
      return e - np.array([0, 0, m.geom_size[g, 0]])
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
      return c - np.array([0, 0, m.geom_size[g, 0]])
    return c - np.array([0, 0, m.geom_rbound[g]])  # boxes, cylinders: conservative

  def _poses(self) -> None:
    """Base pose of every log frame: orientation, floor contact, odometry."""
    L, m, d = self.L, self.model, self.data
    n = len(self.t)
    self.qpos = np.zeros((n, m.nq))
    yaw, xy = 0.0, np.zeros(2)
    prev: dict[int, np.ndarray] = {}
    for k in range(n):
      g = L["gravity"][k]
      q_rp = quat_roll_pitch(g)
      if k:
        # World yaw rate from the gyro (body frame), integrated.
        w_world = _rot(q_rp) @ L["gyro"][k]
        yaw += float(w_world[2]) * float(self.t[k] - self.t[k - 1])
      quat = quat_mul(np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), q_rp)
      d.qpos[:] = 0
      d.qpos[3:7] = quat
      d.qpos[self.qadr] = L["q"][k]
      mujoco.mj_kinematics(m, d)
      low = {g_: self._lowest(d, g_) for g_ in self.coll}
      z = -min(p[2] for p in low.values())
      # Limbs on the floor: their lowest points should not slide.
      feet: dict[int, np.ndarray] = {}
      for b in self.limb_body:
        pts = [p for g_, p in low.items() if m.geom_bodyid[g_] == b]
        p = min(pts, key=lambda v: v[2])
        if p[2] + z < DOWN_TOL:
          feet[b] = p[:2]
      common = [b for b in feet if b in prev]
      if common:
        xy = np.mean([prev[b] - feet[b] for b in common], axis=0)
      prev = {b: feet[b] + xy for b in feet}
      self.qpos[k, :2] = xy
      self.qpos[k, 2] = z
      self.qpos[k, 3:7] = quat
      self.qpos[k, self.qadr] = L["q"][k]
    assert np.allclose(gravity_in_body(quat_roll_pitch(L["gravity"][0])),
                       L["gravity"][0] / np.linalg.norm(L["gravity"][0]), atol=1e-6)  # fmt: skip

  def set(self, k: int) -> None:
    self.data.qpos[:] = self.qpos[k]
    mujoco.mj_forward(self.model, self.data)
    self.ghost.qpos[:] = self.qpos[k]
    self.ghost.qpos[self.qadr] = self.L["target"][k]
    mujoco.mj_kinematics(self.model, self.ghost)

  def text(self, k: int) -> str:
    L = self.L
    mode = str(L["modes"][L["mode"][k]])
    cm = int(L["chain_mode"][k])
    chain = MODE_NAMES[cm] if cm >= 0 else ""
    clock = (self.t0 + timedelta(seconds=float(self.t[k]))).strftime("%H:%M:%S") if self.t0 else ""
    lines = [f"{clock}  t {self.t[k]:6.1f} s", f"{mode} {chain}"]
    if "style" in L.files and chain in ("CRAWL", "CRAWL_STOP"):
      lines[-1] += f" [{L['style'][k]}]"
    if chain in ("DOWN", "UP", "HOLD"):
      lines.append(f"phase {L['phase'][k]:.2f}")
    if chain == "WALK":
      c = L["cmd"][k]
      lines.append(f"walk  fwd {c[0]:+.2f} left {c[1]:+.2f} turn {c[2]:+.2f}")
    if chain == "CRAWL":
      c = L["ccmd"][k]
      lines.append(f"crawl fwd {c[0]:+.2f} left {c[1]:+.2f} turn {c[2]:+.2f}")
    key = "temp" if "temp" in L.files else "temps" if "temps" in L.files else None
    if key:
      t = L[key][k].max(axis=1)
      lines.append(f"hottest motor m{int(t.argmax())} {int(t.max())} C")
    err = np.abs(L["target"][k] - L["q"][k])
    j = int(err.argmax())
    lines.append(f"largest target error {str(L['joints'][j])[:-6]} {err[j]:.2f} rad")
    return "\n".join(lines)

  def frame(self, t: float) -> int:
    return int(min(len(self.t) - 1, np.searchsorted(self.t, t)))

  def is_driven(self, k: int) -> bool:
    return str(self.L["modes"][self.L["mode"][k]]) != "DAMP"


def _rot(q: np.ndarray) -> np.ndarray:
  m = np.zeros(9)
  mujoco.mju_quat2Mat(m, q)
  return m.reshape(3, 3)


def add_ghost(rp: Replay, scene: mujoco.MjvScene, opt: mujoco.MjvOption) -> None:
  n0 = scene.ngeom
  mujoco.mjv_addGeoms(
    rp.model, rp.ghost, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene
  )
  for i in range(n0, scene.ngeom):
    scene.geoms[i].rgba[:] = GHOST_RGBA


def video(rp: Replay, args) -> None:
  import mediapy
  from PIL import Image, ImageDraw

  try:  # no system ffmpeg: use the one bundled with imageio-ffmpeg
    import imageio_ffmpeg

    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
  except ImportError:
    pass

  w, h = 1280, 720
  rp.model.vis.global_.offwidth = max(rp.model.vis.global_.offwidth, w)
  rp.model.vis.global_.offheight = max(rp.model.vis.global_.offheight, h)
  r = mujoco.Renderer(rp.model, h, w)
  opt = mujoco.MjvOption()
  opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]  # visual meshes, no collision capsules
  cam = mujoco.MjvCamera()
  cam.distance, cam.azimuth, cam.elevation = 2.6, args.azimuth, -18.0
  fps = 25
  t_end = args.end if args.end else rp.t[-1]
  frames = []
  look = None
  for t in np.arange(args.start, t_end, 1.0 / fps * args.speed):
    k = rp.frame(t)
    rp.set(k)
    target = rp.data.xpos[rp.pelvis] * [1, 1, 0] + [0, 0, 0.45]
    look = target if look is None else 0.9 * look + 0.1 * target  # smooth follow
    cam.lookat[:] = look
    r.update_scene(rp.data, camera=cam, scene_option=opt)
    if args.ghost and rp.is_driven(k):
      add_ghost(rp, r.scene, opt)
    img = Image.fromarray(r.render())
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, 470, 150), fill=(255, 255, 255))
    draw.multiline_text((16, 14), rp.text(k), fill=(10, 10, 10), spacing=4)
    if args.ghost:
      draw.text((16, h - 28), "grey: measured robot   blue: PD targets sent", fill=(10, 10, 10))
    frames.append(np.asarray(img))
  mediapy.write_video(args.video, frames, fps=fps)
  print(f"wrote {args.video}: {len(frames)} frames, {args.start:.0f}-{t_end:.0f} s")


def viewer(rp: Replay, args) -> None:
  import mujoco.viewer

  state = {"t": args.start, "speed": args.speed, "pause": False, "ghost": args.ghost}

  def on_key(code: int) -> None:
    if code == 32:
      state["pause"] = not state["pause"]
    elif code == 262:
      state["t"] += 5
    elif code == 263:
      state["t"] = max(0.0, state["t"] - 5)
    elif code == 265:
      state["speed"] *= 2
    elif code == 264:
      state["speed"] /= 2
    elif code in (ord("G"), ord("g")):
      state["ghost"] = not state["ghost"]
    elif code == 257:
      state["t"] = args.start

  print(__doc__)
  with mujoco.viewer.launch_passive(
    rp.model, rp.data, key_callback=on_key, show_left_ui=False, show_right_ui=False
  ) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = rp.pelvis
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 2.8, args.azimuth, -18.0
    t_end = args.end if args.end else rp.t[-1]
    last = time.perf_counter()
    while v.is_running():
      now = time.perf_counter()
      if not state["pause"]:
        state["t"] += (now - last) * state["speed"]
      last = now
      if state["t"] > t_end:
        state["t"] = args.start
      k = rp.frame(state["t"])
      with v.lock():
        rp.set(k)
        v.user_scn.ngeom = 0
        if state["ghost"] and rp.is_driven(k):
          add_ghost(rp, v.user_scn, v.opt)
      v.set_texts(
        (
          None,
          mujoco.mjtGridPos.mjGRID_TOPLEFT,
          rp.text(k) + f"\nspeed x{state['speed']:g}{'  PAUSED' if state['pause'] else ''}",
          "Space pause  Left/Right 5 s\nUp/Down speed  G ghost  Enter restart",
        )
      )
      v.sync()
      time.sleep(1 / 60)


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("log")
  ap.add_argument("--bundle", default=str(HERE))
  ap.add_argument("--start", type=float, default=0.0, help="s from the log start")
  ap.add_argument("--end", type=float, default=0.0, help="s; 0 = the end")
  ap.add_argument("--speed", type=float, default=1.0)
  ap.add_argument("--ghost", action="store_true", help="show the PD targets too")
  ap.add_argument("--video", default="", help="render to this .mp4 instead")
  ap.add_argument("--azimuth", type=float, default=135.0, help="camera angle")
  ap.add_argument(
    "--igpu", action="store_true", help="render on the default (integrated) GPU"
  )
  args = ap.parse_args()
  rp = Replay(Path(args.log), Path(args.bundle))
  if args.video:
    video(rp, args)
  else:
    viewer(rp, args)


if __name__ == "__main__":
  main()
