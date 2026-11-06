"""Sim-to-sim: the deploy controller driving plain MuJoCo on the CPU (no mjlab).

``chain_controller.py`` sees only what the real robot gives it (pelvis IMU
quaternion and gyro, joint encoders) and returns PD targets at 50 Hz. Here
MuJoCo plays the robot: ``mujoco/scene.xml`` is the exact model the policies
trained on (braced 29-DoF G1, the same PD actuators and solver settings). Its
contact sensors are only monitors and are dropped on load, so any MuJoCo >= 3.3
works (also the asap_deploy env).

  python sim2sim_mujoco.py                     # viewer + keyboard
  python sim2sim_mujoco.py --start crawl       # start lying in the crawl pose
  python sim2sim_mujoco.py --auto              # headless: walk, C, crawl, U, walk
  python sim2sim_mujoco.py --auto --delay-steps 4 --push 0.3 --frames run.png
  python sim2sim_mujoco.py --test moving       # DEPLOY.md tests: moving, hold,
  python sim2sim_mujoco.py --test all          #   cycles, switch (or all)

Keys (viewer, click it first), as on the robot (g1_chain_real.py): each press
adds 0.1 to the command. W / S forward +0.1 / -0.1, A / D sideways left / right,
Q / E rotation left / right, X all to zero (walk and crawl; the walk stands
still below ~0.25 m/s). C go down to crawl, U stand up, V next crawl style
(creepy / scuttle; while crawling it switches once the command is 0 and the
robot is still), H crawl step height, P push, Enter reset.
The viewer's side panels are hidden so these letters don't also toggle MuJoCo
render flags.

Physics and policies run on the CPU (about 16x real time). The viewer renders
on the NVIDIA GPU: on a hybrid laptop (this one: AMD iGPU + RTX 3080, PRIME
on-demand) OpenGL otherwise lands on the iGPU and the loop runs at half speed.
``--igpu`` keeps the default GPU.

Needs only: ``pip install mujoco onnxruntime numpy`` (any OS).
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time
import zlib
from pathlib import Path


def prefer_nvidia_gl() -> None:
  """Ask glvnd for the NVIDIA OpenGL driver (PRIME render offload). Must run
  before the first OpenGL context exists; settings already in the environment
  win."""
  nv = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
  if not sys.platform.startswith("linux") or not nv.exists() or "--igpu" in sys.argv:
    return
  os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
  os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
  os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(nv))
  if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")  # XWayland, no EGL warnings


prefer_nvidia_gl()

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chain_controller import (  # noqa: E402
  CRAWL,
  CRAWL_STOP,
  DOWN,
  HOLD,
  LIMBS,
  MODE_NAMES,
  STAND_HOLD,
  STEP_HEIGHTS,
  STOPPING,
  UP,
  WALK,
  ChainController,
  gravity_in_body,
)

DECIMATION = 4
# Each press adds KEY_STEP along (forward, left, rotate left); X zeroes.
MOVES = {
  "W": (1, 0, 0),
  "S": (-1, 0, 0),
  "A": (0, 1, 0),
  "D": (0, -1, 0),
  "Q": (0, 0, 1),
  "E": (0, 0, -1),
  "X": (0, 0, 0),
}
KEY_STEP = 0.1
SEGMENTS = {
  "down": (DOWN, HOLD),
  "crawl": (CRAWL, CRAWL_STOP),
  "up": (UP, STAND_HOLD),
}


def load_model(xml: Path) -> tuple[mujoco.MjModel, list[str]]:
  """The scene without its contact sensors (sim monitors, MuJoCo >= 3.5), plus
  the bodies the training's forbidden-contact sensor watched."""
  text = xml.read_text()
  sensors = re.search(r"<sensor>.*?</sensor>", text, re.S)
  assert sensors is not None
  block = sensors.group(0)
  forbidden = sorted(set(re.findall(r'name="forbidden_contact_(\w+?)_found"', block)))
  text = text.replace(block, re.sub(r"\s*<contact [^>]*/>", "", block))
  text = text.replace('meshdir="assets"', f'meshdir="{xml.parent / "assets"}"')
  return mujoco.MjModel.from_xml_string(text), forbidden


class Robot:
  """MuJoCo standing in for the G1: state in, PD targets out, policy order."""

  def __init__(self, xml: Path, joints: list[str], delay_steps: int) -> None:
    self.model, forbidden = load_model(xml)
    self.data = mujoco.MjData(self.model)
    m = self.model
    name = lambda kind, n: mujoco.mj_name2id(m, kind, f"robot/{n}")  # noqa: E731
    jid = [name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
    assert min(jid) >= 0, "joint names do not match the bundle"
    self.qadr = np.array([m.jnt_qposadr[i] for i in jid])
    self.vadr = np.array([m.jnt_dofadr[i] for i in jid])
    act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    self.act = np.array([act_of_joint[i] for i in jid])
    self.peak = np.abs(m.actuator_forcerange[self.act, 1])
    gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
    self.gyro_adr = int(m.sensor_adr[gyro])
    self.braced = []
    for side in ("left", "right"):
      for j, axis in (("pitch", 1), ("yaw", 2)):
        s = mujoco.mj_name2id(
          m, mujoco.mjtObj.mjOBJ_SENSOR, f"robot/{side}_wrist_{j}_torque"
        )
        self.braced.append(int(m.sensor_adr[s]) + axis)
    self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    self.limb_body = [name(mujoco.mjtObj.mjOBJ_BODY, n) for n in LIMBS]
    self.forbidden = {name(mujoco.mjtObj.mjOBJ_BODY, n): n for n in forbidden}
    self.delay = delay_steps
    self.reset()

  def reset(self, key: str = "stand") -> None:
    k = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key)
    mujoco.mj_resetDataKeyframe(self.model, self.data, k)
    mujoco.mj_forward(self.model, self.data)
    self.buffer = [self.q().copy()] * (self.delay + 1)

  # Sensors (what the real robot has).
  def quat(self) -> np.ndarray:
    return self.data.qpos[3:7].copy()

  def gyro(self) -> np.ndarray:
    return self.data.sensordata[self.gyro_adr : self.gyro_adr + 3].copy()

  def q(self) -> np.ndarray:
    return self.data.qpos[self.qadr].copy()

  def qd(self) -> np.ndarray:
    return self.data.qvel[self.vadr].copy()

  def step(self, targets: np.ndarray) -> None:
    """Hold the targets for one 50 Hz tick (4 physics steps), with delay."""
    for _ in range(DECIMATION):
      self.buffer = self.buffer[1:] + [targets]
      self.data.ctrl[self.act] = self.buffer[0]
      mujoco.mj_step(self.model, self.data)

  def push(self, speed: float, rng: np.random.Generator) -> None:
    d = rng.normal(size=2)
    self.data.qvel[:2] += d / max(np.linalg.norm(d), 1e-6) * speed

  # Monitors (sim only: what the gates on the robot have to estimate).
  def ground_bodies(self) -> set[int]:
    n = self.data.ncon
    g1, g2 = self.data.contact.geom1[:n], self.data.contact.geom2[:n]
    out = set()
    for a, b in zip(g1, g2, strict=True):
      if self.floor in (a, b):
        out.add(int(self.model.geom_bodyid[b if a == self.floor else a]))
    return out

  def limbs_touching(self) -> np.ndarray:
    on = self.ground_bodies()
    return np.array([b in on for b in self.limb_body])

  def forbidden_touching(self) -> list[str]:
    return sorted(self.forbidden[b] for b in self.ground_bodies() & set(self.forbidden))

  def above_limbs(self) -> float:
    return float(self.data.qpos[2] - self.data.xpos[self.limb_body, 2].mean())

  def motor_ratio(self) -> float:
    return float((np.abs(self.data.actuator_force[self.act]) / self.peak).max())

  def braced_nm(self) -> float:
    return float(np.abs(self.data.sensordata[self.braced]).max())


def write_png(path: Path, img: np.ndarray) -> None:
  """RGB uint8 image to PNG with the standard library only."""
  h, w, _ = img.shape
  raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

  def chunk(tag: bytes, data: bytes) -> bytes:
    body = tag + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

  png = b"\x89PNG\r\n\x1a\n" + chunk(
    b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
  )
  png += chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
  path.write_bytes(png)


class Frames:
  def __init__(self, robot: Robot) -> None:
    self.robot = robot
    self.renderer = mujoco.Renderer(robot.model, 360, 300)
    self.cam = mujoco.MjvCamera()
    self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.4, 120.0, -15.0
    self.images: list[np.ndarray] = []

  def grab(self) -> None:
    self.cam.lookat[:] = self.robot.data.qpos[:3] * [1.0, 1.0, 0.0] + [0, 0, 0.45]
    self.renderer.update_scene(self.robot.data, camera=self.cam)
    self.images.append(self.renderer.render().copy())

  def save(self, path: Path) -> None:
    write_png(path, np.concatenate(self.images, axis=1))


def run_auto(args, ctrl: ChainController, robot: Robot) -> bool:
  """play_chain.py's eval: walk 0.5 m/s, C at 4 s, down, hold, crawl (0.5 after
  1 s), U after 5 s of crawling, up, walk 0.5 m/s for 3 s."""
  rng = np.random.default_rng(args.seed)
  frames = Frames(robot) if args.frames else None
  ctrl.reset(robot.q())
  dt = ctrl.dt
  T = ctrl.transition_s
  limit_s = 1 + 3 + 2 + T + 3 + 5 + 3 + T + 2 + 3 + 6
  t = 0.0
  pressed_c = False
  t_mode: dict[int, float] = {}
  crawl_x0 = walk_x0 = None
  crawled = walked = 0.0
  pushed = {DOWN: False, UP: False}
  fell = forbidden = None
  peak = {s: {"motor": 0.0, "braced": 0.0, "speed": 0.0} for s in SEGMENTS}
  tilt_dev = 0.0
  gate_log: list[str] = []
  fk_err = 0.0
  fk_miss = n_four = 0
  shots: list[str] = []
  last_mode = -1
  while t < limit_s:
    mode = ctrl.mode
    if mode != last_mode:
      extra = ""
      if mode in (CRAWL, UP, WALK, DOWN) and last_mode >= 0:
        lim = ctrl.limb
        true_down = robot.limbs_touching()
        extra = (
          f"   true: pelvis z {robot.data.qpos[2]:.3f}, over limbs "
          f"{robot.above_limbs():.3f} (fk {lim.above_limbs:.3f}), touching "
          f"{''.join('x' if b else '.' for b in true_down)} (fk "
          f"{''.join('x' if b else '.' for b in lim.down)})"
          if lim is not None
          else ""
        )
      print(f"  t {t:5.2f} s  {MODE_NAMES[mode]}{extra}")
      gate_log.append(MODE_NAMES[mode])
      t_mode.setdefault(mode, t)
      if mode == WALK and last_mode == STAND_HOLD:
        walk_x0 = robot.data.qpos[:2].copy()
        limit_s = t + 3.0
      if mode == CRAWL:
        crawl_x0 = robot.data.qpos[:2].copy()
      if frames and MODE_NAMES[mode] not in shots and mode in (HOLD, CRAWL, WALK):
        shots.append(MODE_NAMES[mode])
        frames.grab()
      last_mode = mode

    # Scripted operator (play_chain.evaluate).
    if 1.0 <= t < 4.0 and mode == WALK:
      ctrl.set_walk_command(0.5, 0.0, 0.0)
    if t >= 4.0 and not pressed_c:
      ctrl.press_down()
      pressed_c = True
    if mode == CRAWL:
      since = t - t_mode[CRAWL]
      ctrl.set_crawl_command(0.5 if since > 1.0 else 0.0, 0.0, 0.0)
      if since > 5.0:
        ctrl.press_up()
    if mode == WALK and walk_x0 is not None:
      ctrl.set_walk_command(0.5, 0.0, 0.0)
    if args.push:
      for m, half in ((DOWN, ctrl.phase > 0.5), (UP, ctrl.phase < 0.5)):
        if mode == m and half and not pushed[m]:
          robot.push(args.push, rng)
          pushed[m] = True
    if frames and mode in (DOWN, UP) and abs(ctrl.phase - 0.5) < 0.01:
      tag = f"{MODE_NAMES[mode]}50"
      if tag not in shots:
        shots.append(tag)
        frames.grab()

    targets = ctrl.step(robot.quat(), robot.gyro(), robot.q(), robot.qd())
    robot.step(targets)
    t += dt

    g = gravity_in_body(robot.quat())
    ge = ctrl.expected_gravity()
    tilt_dev = max(tilt_dev, float(np.degrees(np.arccos(np.clip(g @ ge, -1, 1)))))
    if (g[2] > 0.5 or abs(g[1]) > 0.8 or g[0] < -0.6) and fell is None:
      fell = (round(t, 2), MODE_NAMES[ctrl.mode], round(ctrl.phase, 2))
    if ctrl.mode in (WALK, STOPPING) and g[2] > -0.5 and fell is None:
      fell = (round(t, 2), MODE_NAMES[ctrl.mode], "tilted > 60 deg")
    bad = robot.forbidden_touching()
    if bad and forbidden is None:
      forbidden = (round(t, 2), MODE_NAMES[ctrl.mode], round(ctrl.phase, 2), bad)
    if robot.limbs_touching().all():
      lim = ctrl.fk(robot.quat(), robot.q(), ctrl.gates.limb_down_m)
      fk_err = max(fk_err, abs(lim.above_limbs - robot.above_limbs()))
      fk_miss += int(not lim.down.all())
      n_four += 1
    for seg, modes in SEGMENTS.items():
      if ctrl.mode in modes:
        p = peak[seg]
        p["motor"] = max(p["motor"], robot.motor_ratio())
        p["braced"] = max(p["braced"], robot.braced_nm())
        p["speed"] = max(p["speed"], float(np.abs(robot.qd()).max()))
    if ctrl.mode in (CRAWL, CRAWL_STOP) and crawl_x0 is not None:
      crawled = float(np.linalg.norm(robot.data.qpos[:2] - crawl_x0))
    if ctrl.mode == WALK and walk_x0 is not None:
      walked = float(np.linalg.norm(robot.data.qpos[:2] - walk_x0))

  if frames and walk_x0 is not None:
    frames.grab()
    shots.append("walk again")
  ok = (
    CRAWL in t_mode
    and walk_x0 is not None
    and fell is None
    and forbidden is None
    and crawled > 1.0
    and walked > 0.8
  )
  print(f"reached the crawl: {CRAWL in t_mode}   crawled {crawled:.2f} m")
  print(f"stood up:          {STAND_HOLD in t_mode}")
  print(f"walking again:     {walk_x0 is not None} (walked {walked:.2f} m in 3 s)")
  print(f"fell:              {fell or 'no'}")
  print(f"forbidden contact: {forbidden or 'no'}")
  for seg, p in peak.items():
    print(
      f"  {seg:5s} peaks: motor/limit {p['motor']:.2f}, braced wrist "
      f"{p['braced']:.1f} Nm, joint speed {p['speed']:.1f} rad/s"
    )
  print(
    f"max tilt from the expected gravity (e-stop margin): {tilt_dev:.0f} deg; "
    f"FK pelvis-over-limbs error on 4 limbs: {100 * fk_err:.1f} cm; FK missed a "
    f"touching limb in {fk_miss} of {n_four} ticks"
  )
  print("PASS" if ok else "FAIL")
  if frames:
    frames.save(Path(args.frames))
    print(f"frames ({', '.join(shots)}): {args.frames}")
  return ok


# -- DEPLOY.md tests (v3) -------------------------------------------------------


def continuous_nm(joint: str) -> float:
  """Estimated continuous torque of the joint's motor class (stand-ins from
  g1_motor_safety/motor_model.py, as in the 10-29 report and the v3 README)."""
  if any(k in joint for k in ("hip_roll", "knee")):
    return 36.0
  if any(k in joint for k in ("hip_pitch", "hip_yaw", "waist_yaw")):
    return 23.0
  if any(k in joint for k in ("ankle", "waist_roll", "waist_pitch")):
    return 13.5
  return 6.8


class Plan:
  """A scripted operator: steps robot and controller, measures segments."""

  def __init__(self, args, ctrl: ChainController, robot: Robot) -> None:
    self.args, self.ctrl, self.robot = args, ctrl, robot
    self.rng = np.random.default_rng(args.seed)
    self.t = 0.0
    self.cont = np.array([continuous_nm(j) for j in ctrl.joints])
    self.short = [j.removesuffix("_joint") for j in ctrl.joints]
    self.pelvis = mujoco.mj_name2id(
      robot.model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis"
    )
    self.fell = self.forbidden = None
    self.rows: list[str] = []
    self.fail: list[str] = []
    self.forces: list[np.ndarray] | None = None
    self.peak_ratio = 0.0

  def tick(self) -> None:
    c, r = self.ctrl, self.robot
    r.step(c.step(r.quat(), r.gyro(), r.q(), r.qd()))
    self.t += c.dt
    if self.forces is not None:
      self.forces.append(r.data.actuator_force[r.act].copy())
    self.peak_ratio = max(self.peak_ratio, r.motor_ratio())
    g = gravity_in_body(r.quat())
    if self.fell is None and (g[2] > 0.5 or abs(g[1]) > 0.8 or g[0] < -0.6):
      self.fell = (round(self.t, 1), c.mode_name)
    if self.fell is None and c.mode in (WALK, STOPPING) and g[2] > -0.5:
      self.fell = (round(self.t, 1), c.mode_name)
    bad = r.forbidden_touching()
    if bad and self.forbidden is None:
      self.forbidden = (round(self.t, 1), c.mode_name, bad)

  def run(self, seconds: float) -> None:
    for _ in range(int(round(seconds / self.ctrl.dt))):
      self.tick()

  def until(self, cond, timeout: float, what: str) -> float | None:
    t0 = self.t
    while not cond():
      if self.t - t0 > timeout:
        self.fail.append(f"{what}: not within {timeout:.0f} s ({self.ctrl.mode_name})")
        return None
      self.tick()
    return self.t - t0

  def heading(self) -> np.ndarray:
    """Forward direction on the floor: pelvis x standing, the head (pelvis z)
    while crawling."""
    rot = self.robot.data.xmat[self.pelvis].reshape(3, 3)
    v = rot[:2, 2] if self.ctrl.mode in (CRAWL, CRAWL_STOP, HOLD) else rot[:2, 0]
    return v / max(np.linalg.norm(v), 1e-9)

  def loads(self, label: str, forces: list[np.ndarray]) -> float:
    f = np.array(forces)
    ratio = np.sqrt((f * f).mean(axis=0)) / self.cont
    j = int(ratio.argmax())
    self.rows.append(
      f"    {label}: busiest motor {self.short[j]} {ratio[j]:.2f}x continuous"
    )
    return float(ratio[j])

  # Operator actions -------------------------------------------------------------

  def walk(self, vx: float, seconds: float) -> None:
    self.ctrl.set_walk_command(vx, 0.0, 0.0)
    self.run(seconds)
    self.ctrl.set_walk_command(0.0, 0.0, 0.0)

  def down(self) -> float | None:
    """C, then until the crawl policy drives; returns the HOLD time."""
    c = self.ctrl
    c.press_down()
    if self.until(lambda: c.mode == HOLD, 30, "reach HOLD") is None:
      return None
    return self.until(lambda: c.mode == CRAWL, 30, "crawl gate in HOLD")

  def up(self) -> bool:
    c = self.ctrl
    c.press_up()
    return self.until(lambda: c.mode == WALK, 40, "stand up and walk") is not None

  def crawl(
    self,
    fwd: float,
    lat: float,
    turn: float,
    seconds: float,
    min_track: float | None = 0.7,
  ) -> float:
    """Crawl with a command; measures speed (after 2 s) and motor load."""
    c = self.ctrl
    c.set_crawl_command(fwd, lat, turn)
    cmd = c.crawl_cmd[:3].copy()  # after clipping to the style's ranges
    self.run(2.0)
    p0 = self.robot.data.xpos[self.pelvis, :2].copy()
    h0 = self.heading()
    left0 = np.array([-h0[1], h0[0]])
    yaw, prev = 0.0, h0
    self.forces = []
    n = int(round((seconds - 2.0) / c.dt))
    for _ in range(n):
      self.tick()
      h = self.heading()
      yaw += float(np.arctan2(prev[0] * h[1] - prev[1] * h[0], prev @ h))
      prev = h
    dur = n * c.dt
    d = self.robot.data.xpos[self.pelvis, :2] - p0
    v_fwd, v_lat, w = d @ h0 / dur, d @ left0 / dur, yaw / dur
    label = f"[{c.style}] crawl {cmd[0]:+.2f} {cmd[1]:+.2f} {cmd[2]:+.2f}"
    ratio = self.loads(
      f"{label}: fwd {v_fwd:+.2f} lat {v_lat:+.2f} m/s, yaw {w:+.2f} rad/s", self.forces
    )
    self.forces = None
    if min_track is not None:
      # Crawl turn > 0 is clockwise: the yaw rate is negative.
      for name, want, got in (
        ("fwd", cmd[0], v_fwd),
        ("lat", cmd[1], v_lat),
        ("turn", -cmd[2], w),
      ):
        if abs(want) > 1e-6 and got / want < min_track:
          self.fail.append(
            f"{label}: {name} {got:+.2f} < {min_track:.0%} of {want:+.2f}"
          )
    c.set_crawl_command(0.0, 0.0, 0.0)
    return ratio

  def switch(self, style: str) -> bool:
    c = self.ctrl
    c.set_crawl_command(0.0, 0.0, 0.0)
    c.request_style(style)
    self.peak_ratio = 0.0
    took = self.until(lambda: c.style == style, 10, f"switch to {style}")
    if took is None:
      return False
    self.peak_ratio = 0.0
    self.run(0.6)  # the blend
    self.rows.append(
      f"    switched to {style} after {took:.1f} s, motor/limit peak in the blend "
      f"{self.peak_ratio:.2f}"
    )
    return True

  def finish(self, name: str, extra_ok: bool = True) -> bool:
    if self.fell:
      self.fail.append(f"fell at {self.fell}")
    if self.forbidden:
      self.fail.append(f"forbidden contact at {self.forbidden}")
    ok = extra_ok and not self.fail
    print(f"\n=== {name}: {'PASS' if ok else 'FAIL'} ({self.t:.0f} s simulated)")
    print("\n".join(self.rows))
    for f in self.fail:
      print(f"    FAIL: {f}")
    return ok


def test_moving(p: Plan) -> bool:
  """Walk 10 s, C, creepy 0.2 / 0.35 / 0.5 m/s, sideways 0.2, turn 0.5, U."""
  p.walk(0.0, 10.0)
  hold = p.down()
  if hold is None:
    return p.finish("moving")
  p.rows.append(f"    crawl gate passed {hold:.1f} s after reaching HOLD")
  for fwd, lat, turn, sec in (
    (0.2, 0, 0, 30),
    (0.35, 0, 0, 30),
    (0.5, 0, 0, 20),
    (0, 0.2, 0, 15),
    (0, 0, 0.5, 15),
  ):
    p.crawl(fwd, lat, turn, sec)
  p.run(3.0)
  if p.up():
    p.walk(0.0, 10.0)
  return p.finish("moving")


def test_hold(p: Plan) -> bool:
  """C; stand<->crawl holds the pose 20 s; crawl at 0 for 60 s; U."""
  c = p.ctrl
  p.walk(0.0, 5.0)
  c.press_down()
  if p.until(lambda: c.mode == HOLD, 30, "reach HOLD") is None:
    return p.finish("hold")
  gate = p.until(lambda: c.gate_ok, 10, "crawl gate in HOLD")
  if gate is not None:
    p.rows.append(f"    crawl gate true {gate:.1f} s after reaching HOLD (want < ~3 s)")
  dwell, c.dwell_s = c.dwell_s, 20.0  # hold the pose with the stand<->crawl policy
  p.forces = []
  p.until(lambda: c.mode == CRAWL, 30, "hand over to CRAWL")
  hold = p.loads("stand<->crawl holding the crawl pose", p.forces)
  c.dwell_s = dwell
  rest = p.crawl(0.0, 0.0, 0.0, 60.0, min_track=None)
  ok = (gate is not None and gate < 3.5) and hold <= 0.7 and rest <= 0.7
  if p.up():
    p.walk(0.0, 5.0)
  return p.finish("hold", ok)


def test_cycles(p: Plan) -> bool:
  """5 x (walk 0.3 m/s 30 s, down, crawl 0.3 m/s 60 s, up)."""
  done = 0
  for _ in range(5):
    p.walk(0.3, 30.0)
    if p.down() is None:
      break
    p.crawl(0.3, 0.0, 0.0, 60.0)
    if not p.up():
      break
    done += 1
  p.rows.append(f"    cycles completed: {done} of 5")
  return p.finish("cycles", done == 5)


def test_switch(p: Plan) -> bool:
  """Creepy 0.3, stop, scuttle 0.75 / 1.0, stop, creepy 0.3, U."""
  c = p.ctrl
  if "scuttle" not in c.crawl_styles:
    print("no scuttle in this bundle")
    return True
  p.walk(0.0, 5.0)
  if p.down() is None:
    return p.finish("switch")
  p.crawl(0.3, 0, 0, 10)
  if p.switch("scuttle"):
    p.crawl(0.75, 0, 0, 15)
    p.crawl(1.0, 0, 0, 10)
    if p.switch(c.crawl_styles[0]):
      p.crawl(0.3, 0, 0, 10)
  p.run(2.0)
  if p.up():
    p.walk(0.0, 5.0)
  return p.finish("switch")


TESTS = {
  "moving": test_moving,
  "hold": test_hold,
  "cycles": test_cycles,
  "switch": test_switch,
}


def run_viewer(args, ctrl: ChainController, robot: Robot) -> None:
  import mujoco.viewer

  rng = np.random.default_rng(args.seed)
  keys: list[int] = []
  key = "crawl" if args.start == "crawl" else "stand"

  def reset() -> None:
    robot.reset(key)
    ctrl.reset(robot.q(), start_crawl=args.start == "crawl")

  reset()
  print(__doc__)

  # Side panels hidden: with them shown, MuJoCo also binds W/S/A/D/... to render
  # toggles (wireframe, shadows, ...).
  with mujoco.viewer.launch_passive(
    robot.model,
    robot.data,
    key_callback=keys.append,
    show_left_ui=False,
    show_right_ui=False,
  ) as v:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = mujoco.mj_name2id(
      robot.model, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis"
    )
    v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 120.0, -15.0
    last = None
    while v.is_running():
      t0 = time.perf_counter()
      while keys:
        code = keys.pop(0)
        if code == 257:  # Enter: reset.
          with v.lock():
            reset()
          continue
        handle_key(chr(code) if code < 256 else "", ctrl, robot, rng)
      with v.lock():
        if robot.data.time < 1e-9:  # The viewer's Backspace reset.
          reset()
        targets = ctrl.step(robot.quat(), robot.gyro(), robot.q(), robot.qd())
        robot.step(targets)
      v.set_texts(
        (
          None,
          mujoco.mjtGridPos.mjGRID_TOPLEFT,
          f"{ctrl.mode_name}\n{command_text(ctrl)}\nphase {ctrl.phase:.2f}\n"
          f"{'waiting: ' + ctrl.gate_why if ctrl.gate_why else ''}"
          f"{style_text(ctrl)}",
          "C down to crawl  U stand up  V crawl style\n"
          "W/S forward +-0.1  A/D left/right 0.1\n"
          "Q/E rotate left/right 0.1  X zero\nH step height  P push  Enter reset",
        )
      )
      v.sync()
      status = (ctrl.mode_name, command_text(ctrl), round(ctrl.phase, 2))
      if status != last:
        print(f"\r{status[0]:<11} {status[1]:<40} phase {status[2]:.2f}   ", end="")
        last = status
      time.sleep(max(0.0, ctrl.dt - (time.perf_counter() - t0)))


def style_text(ctrl: ChainController) -> str:
  if not ctrl.style_pending:
    return ""
  return f"\nstyle -> {ctrl.style_pending}: {ctrl.style_why or 'switching'}"


def command_text(ctrl: ChainController) -> str:
  if ctrl.mode == WALK:
    c = ctrl.walk_cmd
    return f"vx {c[0]:+.1f}  vy {c[1]:+.1f}  wz {c[2]:+.1f}"
  if ctrl.mode == CRAWL:
    c = ctrl.crawl_cmd
    return (
      f"[{ctrl.style}] fwd {c[0]:+.2f} lat {c[1]:+.1f} turn {c[2]:+.2f} step {c[3]:.2f}"
    )
  return f"[{ctrl.style}]"


def handle_key(k: str, ctrl: ChainController, robot: Robot, rng) -> None:
  if k == "C":
    ctrl.press_down()
  elif k == "U":
    ctrl.press_up()
  elif k == "P":
    robot.push(0.4, rng)
  elif k == "V":
    print(f"\ncrawl style -> {ctrl.request_style()}")
  elif k == "H" and ctrl.mode == CRAWL:
    hs = STEP_HEIGHTS
    i = int(np.argmin([abs(h - ctrl.crawl_cmd[3]) for h in hs]))
    ctrl.set_step_height(hs[(i + 1) % len(hs)])
  elif k == "X":
    ctrl.stop()
  elif k in MOVES:
    step = np.array(MOVES[k], dtype=float) * KEY_STEP
    if ctrl.mode == WALK:
      ctrl.set_walk_command(*np.round(ctrl.walk_cmd + step, 3))
    elif ctrl.mode == CRAWL:
      # The crawl's turn input is clockwise-positive: rotate left lowers it.
      step[2] = -step[2]
      ctrl.set_crawl_command(*np.round(ctrl.crawl_cmd[:3] + step, 3))


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  ap.add_argument("--bundle", default=str(HERE))
  ap.add_argument("--auto", action="store_true", help="headless scripted test")
  ap.add_argument(
    "--test",
    choices=("moving", "hold", "cycles", "switch", "all"),
    help="headless DEPLOY.md tests (v3)",
  )
  ap.add_argument("--style", default=None, help="crawl style to start with")
  ap.add_argument("--start", choices=("stand", "crawl"), default="stand")
  ap.add_argument(
    "--transition-s", type=float, default=8.0, help="down / up time (train 4-8)"
  )
  ap.add_argument("--delay-steps", type=int, default=0, help="0-4 physics steps")
  ap.add_argument("--push", type=float, default=0.0, help="auto: kick mid-down/up")
  ap.add_argument("--frames", default=None, help="auto: save a PNG strip")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument(
    "--igpu", action="store_true", help="render on the default (integrated) GPU"
  )
  args = ap.parse_args()
  ctrl = ChainController(args.bundle, transition_s=args.transition_s)
  robot = Robot(
    Path(args.bundle) / "mujoco" / "scene.xml", ctrl.joints, args.delay_steps
  )
  if args.test:
    names = (
      ("moving", "hold", "cycles", "switch") if args.test == "all" else (args.test,)
    )
    results = []
    for name in names:
      robot.reset("stand")
      ctrl.reset(robot.q())
      results.append(TESTS[name](Plan(args, ctrl, robot)))
    print(
      "\n"
      + "\n".join(
        f"{n:8s} {'PASS' if ok else 'FAIL'}"
        for n, ok in zip(names, results, strict=True)
      )
    )
    sys.exit(0 if all(results) else 1)
  if args.style:
    ctrl.request_style(args.style)
  if args.auto:
    if args.start != "stand":
      sys.exit("--auto starts standing")
    sys.exit(0 if run_auto(args, ctrl, robot) else 1)
  run_viewer(args, ctrl, robot)


if __name__ == "__main__":
  main()
