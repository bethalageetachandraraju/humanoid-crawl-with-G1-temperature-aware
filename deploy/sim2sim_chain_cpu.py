"""CPU MuJoCo sim-to-sim of a chain bundle (walk <-> stand_crawl <-> crawl).

What the laptop runs, rebuilt from the bundle alone: the ONNX policies, the
``chain.json`` layouts, gates and joint-range clamp, and ``mujoco/scene.xml``
as the robot. No mjlab, no GPU. The gates use only IMU, encoders and forward
kinematics, as on the robot (no contact sensing).

  uv run python scripts/sim2sim_chain_cpu.py --bundle deploy/g1_chain_v3
  ... --pd-1khz            # 1 ms physics, explicit PD torques (DDS-sim style)
  ... --delay-steps 4      # PD targets applied 4 physics steps late
  ... --frames run.png     # PNG strip at the key moments

Sequence: stand 1 s, walk 0.5 m/s 3 s, C, down, hold, crawl ``--crawl-speed``
for ``--crawl-s``, U, crawl stop, hold, up, stand, walk 0.3 m/s ``--walk-s``
(passes if it walked at least half the commanded distance, without a fall).
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

WALK, STOPPING, DOWN, HOLD, CRAWL, CRAWL_STOP, UP, STAND_HOLD = range(8)
MODE_NAMES = (
  "WALK",
  "STOPPING",
  "DOWN",
  "HOLD",
  "CRAWL",
  "CRAWL_STOP",
  "UP",
  "STAND_HOLD",
)
LIMBS = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_forearm_extension",
  "right_forearm_extension",
)
FORBIDDEN = ("pelvis", "torso", "waist", "shoulder", "elbow", "wrist")


def smoothstep(x):
  x = np.clip(x, 0.0, 1.0)
  return x * x * (3.0 - 2.0 * x)


def gravity_in_body(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = quat_wxyz / np.linalg.norm(quat_wxyz)
  return -np.array(
    [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)]
  )


class Onnx:
  def __init__(self, path: Path) -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    self.sess = ort.InferenceSession(
      str(path), opts, providers=["CPUExecutionProvider"]
    )
    self.name = self.sess.get_inputs()[0].name

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    out = self.sess.run(None, {self.name: obs[None].astype(np.float32)})[0]
    return np.asarray(out, dtype=np.float64)[0]


class Chain:
  """The chain.json state machine (one robot), numpy only."""

  def __init__(
    self,
    bundle: Path,
    crawl_file: str,
    transition_s: float,
    trans_file: str | None = None,
  ) -> None:
    cfg = json.loads((bundle / "chain.json").read_text())
    self.cfg = cfg
    J = cfg["joints"]
    self.joints = J
    self.dt = 1.0 / cfg["control_hz"]
    vec = lambda d: np.array([d[j] for j in J])  # noqa: E731
    self.q_stand, self.q_crawl = vec(cfg["stand_pose"]), vec(cfg["crawl_pose"])
    self.q_lo = np.array([cfg["joint_limits"][j][0] for j in J])
    self.q_hi = np.array([cfg["joint_limits"][j][1] for j in J])
    pw, ps = cfg["policies"]["walk"], cfg["policies"]["stand_crawl"]
    self.walk_scale = vec(pw["scale"])
    self.walk = Onnx(bundle / pw["file"])
    self.trans = Onnx(Path(trans_file) if trans_file else bundle / ps["file"])
    self.crawl = Onnx(bundle / crawl_file)
    kf = ps["keyframes"]
    self.phis = np.array([k["phi"] for k in kf])
    self.kq = np.array([[k["joint_pos"][j] for j in J] for k in kf])
    pitch = np.radians(kf[-1]["pitch_deg"])
    self.g_crawl = np.array([np.sin(pitch), 0.0, -np.cos(pitch)])
    self.above = cfg["crawl_pelvis_above_limbs_m"]
    self.T = transition_s
    self.relaxed = False
    self.hold_timeout = 0.0
    self.reset()

  def reset(self) -> None:
    n = len(self.joints)
    self.mode, self.time, self.dwell, self.queued = WALK, 0.0, 0.0, -1
    self.phase = self.rate = 0.0
    self.walk_cmd, self.crawl_cmd = np.zeros(3), np.zeros(4)
    self.crawl_cmd[3] = 0.03
    self.last = {k: np.zeros(n) for k in ("walk", "trans", "crawl")}
    self.blend_t0 = -1e9
    self.applied = self.q_stand.copy()
    self.blend_from = self.applied.copy()
    self.mode_t0 = 0.0

  def ref(self, phi: float) -> np.ndarray:
    i = int(
      np.clip(np.searchsorted(self.phis, phi, side="right") - 1, 0, len(self.phis) - 2)
    )
    s = smoothstep((phi - self.phis[i]) / (self.phis[i + 1] - self.phis[i]))
    return self.kq[i] + (self.kq[i + 1] - self.kq[i]) * s

  def _set(self, mode: int, blend: bool) -> None:
    self.mode, self.dwell, self.mode_t0 = mode, 0.0, self.time
    if blend:
      self.blend_t0, self.blend_from = self.time, self.applied.copy()

  def press_c(self) -> None:
    if self.mode == WALK:
      self.walk_cmd[:] = 0.0
      self._set(STOPPING, False)

  def press_u(self) -> None:
    if self.mode == CRAWL:
      self.crawl_cmd[:3] = 0.0
      self._set(CRAWL_STOP, False)
    elif self.mode in (HOLD, DOWN):
      self.queued = UP

  def step(self, quat, gyro, q, qd, above_limbs, pelvis_over_feet) -> np.ndarray:
    g, w = gravity_in_body(quat), gyro
    self.time += self.dt
    m = self.mode
    self.rate = 1.0 / self.T if m == DOWN else (-1.0 / self.T if m == UP else 0.0)
    # Shadow mode: every policy runs every tick.
    cmd = self.walk_cmd if m == WALK else np.zeros(3)
    a = self.walk(np.concatenate([g, w, cmd, q - self.q_stand, qd, self.last["walk"]]))
    self.last["walk"] = np.clip(a, -5.0, 5.0)
    t_walk = self.q_stand + self.walk_scale * self.last["walk"]
    ref = self.ref(self.phase)
    a = self.trans(
      np.concatenate(
        [
          g,
          w,
          q - self.q_stand,
          qd,
          self.last["trans"],
          [self.phase, self.rate],
          ref - q,
        ]
      )
    )
    self.last["trans"] = np.clip(a, -1.6, 1.6)
    t_trans = ref + 0.5 * self.last["trans"]
    c7 = np.concatenate(
      [self.crawl_cmd if m == CRAWL else [0, 0, 0, self.crawl_cmd[3]], [0, 0, 0]]
    )
    a = self.crawl(np.concatenate([g, w, c7, q - self.q_crawl, qd, self.last["crawl"]]))
    self.last["crawl"] = np.clip(a, -5.0, 5.0)
    t_crawl = self.q_crawl + 0.5 * self.last["crawl"]
    target = (
      t_walk
      if m in (WALK, STOPPING)
      else (t_crawl if m in (CRAWL, CRAWL_STOP) else t_trans)
    )
    s = smoothstep((self.time - self.blend_t0) / 0.5)
    self.applied = self.blend_from + (target - self.blend_from) * s

    self.phase = float(np.clip(self.phase + self.rate * self.dt, 0.0, 1.0))
    if m == STOPPING:
      self.phase = 0.0
    if m == CRAWL_STOP:
      self.phase = 1.0
    still = np.abs(qd).max()
    if m == STOPPING:
      ok, need = still < 0.6 and np.linalg.norm(w) < 0.3, 0.5
    elif m == HOLD:
      ok = (
        still < 0.5
        and abs(above_limbs - self.above) < 0.03
        and np.linalg.norm(g - self.g_crawl) < 0.15
      )
      need = 1.0
    elif m == CRAWL_STOP:
      ok, need = still < 0.5 and np.linalg.norm(g - self.g_crawl) < 0.25, 0.5
    elif m == STAND_HOLD and self.relaxed:
      # Upright and tall: the walk policy takes over, still or not.
      ok, need = g[2] < -0.94 and pelvis_over_feet > 0.66, 0.3
    elif m == STAND_HOLD:
      ok = (
        g[2] < -0.97
        and pelvis_over_feet > 0.66
        and still < 0.6
        and np.linalg.norm(w) < 0.3
      )
      need = 0.5
    else:
      ok, need = False, 0.0
    self.dwell = self.dwell + self.dt if ok else 0.0
    ready = self.dwell >= need
    if m == STOPPING and ready:
      self._set(DOWN, True)
    elif m == DOWN and self.phase >= 1.0:
      self._set(HOLD, False)
    elif (
      m == HOLD and self.hold_timeout and self.time - self.mode_t0 > self.hold_timeout
    ):
      # The gate never passed (DEPLOY.md): stand up instead of waiting.
      self.queued = -1
      self._set(UP, False)
    elif m == HOLD and ready:
      if self.queued == UP:
        self.queued = -1
        self._set(UP, False)
      else:
        self._set(CRAWL, True)
    elif m == CRAWL_STOP and ready:
      self.queued = UP
      self._set(HOLD, True)
    elif m == UP and self.phase <= 0.0:
      self._set(STAND_HOLD, False)
    elif m == STAND_HOLD and ready:
      self._set(WALK, True)
    self.applied = np.clip(self.applied, self.q_lo, self.q_hi)
    return self.applied.copy()


class Robot:
  def __init__(self, xml: Path, joints, kp, kd, args) -> None:
    m = mujoco.MjModel.from_xml_path(str(xml))
    self.model, self.data = m, mujoco.MjData(m)
    jid = [
      mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{j}") for j in joints
    ]
    self.qadr = np.array([m.jnt_qposadr[i] for i in jid])
    self.vadr = np.array([m.jnt_dofadr[i] for i in jid])
    act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    self.act = np.array([act_of_joint[i] for i in jid])
    self.kp, self.kd = kp, kd
    self.peak = np.abs(m.actuator_forcerange[self.act, 1]) * args.effort
    m.actuator_forcerange[self.act, 0] = -self.peak
    m.actuator_forcerange[self.act, 1] = self.peak
    self.explicit = args.pd_1khz
    if self.explicit:
      # Torque actuators; the PD runs here every 1 ms physics step.
      m.opt.timestep = 0.001
      m.actuator_gainprm[:, 0] = 1.0
      m.actuator_biasprm[:, :3] = 0.0
    self.substeps = round(0.02 / m.opt.timestep)
    gyro = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "robot/imu_ang_vel")
    self.gyro_adr = int(m.sensor_adr[gyro])
    body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"robot/{n}")  # noqa: E731
    self.pelvis = body("pelvis")
    self.limbs = [body(n) for n in LIMBS]
    self.torso = body("torso_link")
    self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    if args.payload:
      m.body_mass[self.torso] += args.payload
    self.delay = args.delay_steps
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(m, self.data, key)
    mujoco.mj_forward(m, self.data)
    self.buffer = [self.q().copy()] * (self.delay + 1)

  def quat(self):
    return self.data.qpos[3:7].copy()

  def gyro(self):
    return self.data.sensordata[self.gyro_adr : self.gyro_adr + 3].copy()

  def q(self):
    return self.data.qpos[self.qadr].copy()

  def qd(self):
    return self.data.qvel[self.vadr].copy()

  def above_limbs(self) -> float:
    x = self.data.xpos
    return float(x[self.pelvis, 2] - x[self.limbs, 2].mean())

  def pelvis_over_feet(self) -> float:
    x = self.data.xpos
    return float(x[self.pelvis, 2] - x[self.limbs[:2], 2].mean() + 0.04)

  def step(self, targets: np.ndarray) -> None:
    for _ in range(self.substeps):
      self.buffer = self.buffer[1:] + [targets]
      tgt = self.buffer[0]
      if self.explicit:
        tau = self.kp * (tgt - self.q()) - self.kd * self.qd()
        self.data.ctrl[self.act] = np.clip(tau, -self.peak, self.peak)
      else:
        self.data.ctrl[self.act] = tgt
      mujoco.mj_step(self.model, self.data)

  def ground_bodies(self) -> set[str]:
    n, d = self.data.ncon, self.data
    out = set()
    for i in range(n):
      g1, g2 = d.contact.geom1[i], d.contact.geom2[i]
      if self.floor in (g1, g2):
        other = g2 if g1 == self.floor else g1
        out.add(
          self.model.body(self.model.geom_bodyid[other]).name.removeprefix("robot/")
        )
    return out


def write_png(path: Path, img: np.ndarray) -> None:
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


def run(args) -> bool:
  bundle = Path(args.bundle)
  ch = Chain(bundle, args.crawl_onnx, args.transition_s, args.stand_crawl_onnx)
  ch.relaxed = args.relaxed_handover
  ch.hold_timeout = args.hold_timeout
  kp = np.array([ch.cfg["pd"][j]["kp"] for j in ch.joints])
  kd = np.array([ch.cfg["pd"][j]["kd"] for j in ch.joints])
  robot = Robot(bundle / "mujoco" / "scene.xml", ch.joints, kp, kd, args)
  rng = np.random.default_rng(args.seed)
  renderer = cam = None
  frames: list[np.ndarray] = []
  if args.frames:
    renderer = mujoco.Renderer(robot.model, 300, 300)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 2.4, 110.0, -12.0
  t, last_mode = 0.0, -1
  log = []
  fell = forbidden = None
  t_crawl = t_up = t_stand = t_walk = None
  pushed_dn = pushed_up = False
  limit = 1 + 3 + 3 + 2 * args.transition_s + 3 + args.crawl_s + 3 + 3 + 6 + 20
  x_walk0 = None
  next_frame = 0.0
  rec: list = []
  while t < limit:
    m = ch.mode
    if m != last_mode:
      log.append(f"t {t:6.2f}  {MODE_NAMES[m]}")
      last_mode = m
    if 1.0 <= t < 4.0 and m == WALK:
      ch.walk_cmd[:] = (0.5, 0.0, 0.0)
    if 4.0 <= t and t - ch.dt < 4.0:
      ch.press_c()
    if m == CRAWL:
      t_crawl = t_crawl if t_crawl is not None else t
      ch.crawl_cmd[0] = args.crawl_speed if t - t_crawl > 1.0 else 0.0
      if t - t_crawl > args.crawl_s:
        ch.press_u()
    if m == UP and t_up is None:
      t_up = t
    if m == STAND_HOLD and t_stand is None:
      t_stand = t
    if m == WALK and t_stand is not None:
      if t_walk is None:
        t_walk, x_walk0 = t, robot.data.qpos[:2].copy()
        limit = t + args.walk_s
      ch.walk_cmd[:] = (0.3, 0.0, 0.0)
    if args.push and m == DOWN and ch.phase > 0.5 and not pushed_dn:
      d = rng.normal(size=2)
      robot.data.qvel[:2] += d / np.linalg.norm(d) * args.push
      pushed_dn = True
    if args.push and m == UP and ch.phase < 0.6 and not pushed_up:
      d = rng.normal(size=2)
      robot.data.qvel[:2] += d / np.linalg.norm(d) * args.push
      pushed_up = True
    q_now = robot.q()
    tg = ch.step(
      robot.quat(),
      robot.gyro(),
      q_now,
      robot.qd(),
      robot.above_limbs(),
      robot.pelvis_over_feet(),
    )
    robot.step(tg)
    if args.log:
      rec.append(
        (
          t,
          ch.mode,
          ch.phase,
          q_now,
          tg,
          robot.data.actuator_force[robot.act].copy(),
          gravity_in_body(robot.quat()),
        )
      )
    t += ch.dt
    if renderer is not None and t >= next_frame:
      cam.lookat[:] = robot.data.qpos[:3] * [1, 1, 0] + [0, 0, 0.4]
      renderer.update_scene(robot.data, camera=cam)
      frames.append(renderer.render().copy())
      next_frame = t + args.frame_every
    g = gravity_in_body(robot.quat())
    if fell is None and (g[2] > 0.5 or abs(g[1]) > 0.8 or g[0] < -0.6):
      fell = (round(t, 2), MODE_NAMES[ch.mode], round(ch.phase, 2))
    bad = {b for b in robot.ground_bodies() if any(k in b for k in FORBIDDEN)}
    if forbidden is None and bad:
      forbidden = (round(t, 2), MODE_NAMES[ch.mode], round(ch.phase, 2), sorted(bad))
    if ch.mode == UP and (t - ch.mode_t0) > args.transition_s + 0.1:
      pass
  walked = (
    float(np.linalg.norm(robot.data.qpos[:2] - x_walk0)) if x_walk0 is not None else 0.0
  )
  ok = t_walk is not None and fell is None and walked > 0.15 * args.walk_s
  print("\n".join(log))
  print(
    f"crawl reached: {t_crawl is not None}; up started: {t_up is not None}; "
    f"stand reached: {t_stand is not None}; walking again: {t_walk is not None} "
    f"(walked {walked:.2f} m)\nfell: {fell or 'no'}\nforbidden contact: {forbidden or 'no'}\n"
    f"final mode {MODE_NAMES[ch.mode]}, phase {ch.phase:.2f}, pelvis z {robot.data.qpos[2]:.2f} m"
  )
  print("PASS" if ok else "FAIL")
  if args.log and rec:
    np.savez(
      args.log,
      joints=np.array(ch.joints),
      t=np.array([r[0] for r in rec]),
      chain_mode=np.array([r[1] for r in rec]),
      phase=np.array([r[2] for r in rec]),
      q=np.array([r[3] for r in rec]),
      target=np.array([r[4] for r in rec]),
      tau=np.array([r[5] for r in rec]),
      gravity=np.array([r[6] for r in rec]),
    )
  if frames and args.frames:
    rows = [
      np.concatenate(frames[i : i + 12], axis=1) for i in range(0, len(frames), 12)
    ]
    w = rows[0].shape[1]
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    write_png(Path(args.frames), np.concatenate(rows, axis=0))
    print(f"frames: {args.frames}")
  return ok


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawTextHelpFormatter
  )
  ap.add_argument("--bundle", default="deploy/g1_chain_v3")
  ap.add_argument("--crawl-onnx", default="crawl.onnx", help="in the bundle, or a path")
  ap.add_argument("--stand-crawl-onnx", default=None, help="path; default the bundle's")
  ap.add_argument("--transition-s", type=float, default=8.0)
  ap.add_argument("--crawl-speed", type=float, default=0.3)
  ap.add_argument("--crawl-s", type=float, default=10.0)
  ap.add_argument("--walk-s", type=float, default=4.0, help="walk at 0.3 m/s after")
  ap.add_argument("--delay-steps", type=int, default=0)
  ap.add_argument("--pd-1khz", action="store_true")
  ap.add_argument(
    "--relaxed-handover",
    action="store_true",
    help="STAND_HOLD -> WALK when upright (g_z < -0.94) and tall for 0.3 s",
  )
  ap.add_argument(
    "--hold-timeout", type=float, default=0.0, help="HOLD -> UP after this many s"
  )
  ap.add_argument("--push", type=float, default=0.0)
  ap.add_argument("--payload", type=float, default=0.0)
  ap.add_argument("--effort", type=float, default=1.0)
  ap.add_argument("--frames", default=None)
  ap.add_argument("--frame-every", type=float, default=1.0)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--log", default=None, help="save an .npz like the robot's")
  args = ap.parse_args()
  sys.exit(0 if run(args) else 1)


if __name__ == "__main__":
  main()
