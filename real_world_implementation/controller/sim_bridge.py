"""MuJoCo stand-in for the real G1, speaking Unitree SDK2 DDS (unitree_hg).

Publishes rt/lowstate (pelvis IMU + 29 motors, motor order) and applies
rt/lowcmd the way the motor drivers do: tau = kp (q* - q) + kd (dq* - dq) + tau,
clipped to the motor peak torque, at 1 kHz. The braced wrist motors have no
joint in the model (welded by the brace) and report zeros.

Runs on loopback with DDS domain 1 only, so it can never talk to the robot.

  python sim_bridge.py                 # viewer; robot starts hanging on a band
  python sim_bridge.py --headless --lower-at 6 --release-at 13
  python sim_bridge.py ... --heat 23:2:40:15   # motor 23 reads +2 C/s from 40 s, 15 s
  python sim_bridge.py ... --remote-script "5:Start,12:A,20:ly=0.6"   # fake remote

Viewer keys (click the window first): 7 lower the band 5 cm, 8 raise it,
9 release / re-attach it, Backspace reset. Same order as on the real gantry:
stand up while hanging, lower until the feet touch, start the walk policy,
then release.
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import threading
import time
from pathlib import Path


def prefer_nvidia_gl() -> None:
  """Render the viewer on the NVIDIA GPU (PRIME render offload): on this hybrid
  laptop OpenGL otherwise lands on the AMD iGPU, which can't keep up. Must run
  before the first OpenGL context; settings already in the environment win."""
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
from g1_config import (  # noqa: E402
  JOINTS,
  MOTOR_OF_JOINT,
  NUM_MOTORS,
  PEAK_TORQUE,
  STAND_POSE,
)

HERE = Path(__file__).resolve().parent
MODEL_XML = HERE / "assets" / "g1_29dof_braced_chain_sim.xml"
BAND_K, BAND_D, BAND_ROT_D = 3000.0, 400.0, 20.0
BAND_POINT = np.array([0.0, 0.0, 0.40])  # harness point in the torso frame
PASSIVE_KD = 1.0  # joint drag before the first LowCmd arrives


def _id(m: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
  """Look a name up with or without mjlab's 'robot/' prefix."""
  for n in (name, f"robot/{name}"):
    i = mujoco.mj_name2id(m, kind, n)
    if i >= 0:
      return i
  raise KeyError(f"{name} not in the bridge model")


class Bridge:
  def __init__(self, args: argparse.Namespace) -> None:
    self.m = mujoco.MjModel.from_xml_path(str(args.model))
    self.d = mujoco.MjData(self.m)
    m = self.m
    jids = [_id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
    self.qadr = np.array([m.jnt_qposadr[j] for j in jids])
    self.vadr = np.array([m.jnt_dofadr[j] for j in jids])
    # Actuator driving each policy joint (the model's actuator order may differ).
    act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    self.act = np.array([act_of_joint[j] for j in jids])
    self.pelvis = _id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    self.torso = _id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    self._jacp = np.zeros((3, m.nv))
    sens = lambda n: m.sensor_adr[_id(m, mujoco.mjtObj.mjOBJ_SENSOR, n)]  # noqa: E731
    self.gyro, self.acc, self.quat = (
      sens("imu_ang_vel"),
      sens("imu_lin_acc"),
      sens("imu_quat"),
    )
    self.tau = np.zeros(len(JOINTS))
    self.lock = threading.Lock()
    self.cmd = {k: np.zeros(NUM_MOTORS) for k in ("q", "dq", "tau", "kp", "kd", "mode")}
    self.n_cmd = 0
    self.v_sum, self.v_n = np.zeros(3), 0
    self.keys: list[int] = []
    self.args = args
    self.heat = (
      tuple(float(v) for v in args.heat.split(":")) if args.heat else None
    )
    # Fake Unitree remote: "t:Button" presses a button for 0.3 s; "t:ly=0.6"
    # sets a stick (lx, rx, ry, ly) from time t on.
    self.remote_events = []
    for item in filter(None, args.remote_script.split(",")):
      t, what = item.split(":", 1)
      self.remote_events.append((float(t), what.strip()))
    self.sticks = {"lx": 0.0, "rx": 0.0, "ry": 0.0, "ly": 0.0}
    self.reset()

  def reset(self) -> None:
    mujoco.mj_resetData(self.m, self.d)
    self.d.qpos[2] = 0.76 + self.args.hang
    self.d.qpos[3:7] = (1, 0, 0, 0)
    self.d.qpos[self.qadr] = STAND_POSE
    mujoco.mj_forward(self.m, self.d)
    self.band_on = True
    self.anchor = self._anchor_above(self._band_point())
    self.lowering = False

  def _band_point(self) -> np.ndarray:
    """Harness point: upper torso, above the robot's centre of mass."""
    d = self.d
    return d.xpos[self.torso] + d.xmat[self.torso].reshape(3, 3) @ BAND_POINT

  def _anchor_above(self, p: np.ndarray) -> np.ndarray:
    """Anchor point that holds the robot static at p (band sag = m g / k)."""
    sag = 9.81 * self.m.body_subtreemass[self.pelvis] / BAND_K
    return p + np.array([0.0, 0.0, sag])

  # -- DDS ---------------------------------------------------------------------
  def on_cmd(self, msg) -> None:
    mc = msg.motor_cmd
    with self.lock:
      for i in range(NUM_MOTORS):
        c = mc[i]
        self.cmd["q"][i], self.cmd["dq"][i], self.cmd["tau"][i] = c.q, c.dq, c.tau
        self.cmd["kp"][i], self.cmd["kd"][i], self.cmd["mode"][i] = c.kp, c.kd, c.mode
      self.n_cmd += 1

  def fill_state(self, st, tick: int) -> None:
    d = self.d
    st.tick = tick
    st.mode_pr = 0
    st.mode_machine = 5
    quat = d.sensordata[self.quat : self.quat + 4]
    w, x, y, z = quat
    st.imu_state.quaternion = [float(v) for v in quat]
    st.imu_state.gyroscope = [float(v) for v in d.sensordata[self.gyro : self.gyro + 3]]
    st.imu_state.accelerometer = [
      float(v) for v in d.sensordata[self.acc : self.acc + 3]
    ]
    st.imu_state.rpy = [
      math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
      math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
      math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    ]
    q = d.qpos[self.qadr]
    dq = d.qvel[self.vadr]
    for i in range(NUM_MOTORS):
      ms = st.motor_state[i]
      ms.q = ms.dq = ms.tau_est = 0.0
      ms.mode = 1
      ms.temperature = [40, 40]
    if self.remote_events:
      st.wireless_remote = self._remote_bytes(self.d.time)
    if self.heat is not None:  # a fake hot motor, to test the thermal guard
      m, rate, start, dur = self.heat
      t = self.d.time
      up = rate * min(max(t - start, 0.0), dur)
      extra = max(0.0, up - 0.5 * max(t - start - dur, 0.0))  # cools 0.5 C/s
      st.motor_state[int(m)].temperature = [40, int(round(40 + extra))]
    for i, mi in enumerate(MOTOR_OF_JOINT):
      ms = st.motor_state[mi]
      ms.q, ms.dq, ms.tau_est = float(q[i]), float(dq[i]), float(self.tau[i])

  def _remote_bytes(self, t: float) -> list[int]:
    from inputs import Remote

    raw = bytearray(40)
    for te, what in self.remote_events:
      if "=" in what and t >= te:
        k, v = what.split("=")
        self.sticks[k] = float(v)
    for k, off in (("lx", 4), ("rx", 8), ("ry", 12), ("ly", 20)):
      raw[off : off + 4] = struct.pack("<f", self.sticks[k])
    for name, byte, bit in Remote.BUTTONS:
      if any(
        name in w.split("+") and te <= t < te + 0.3 for te, w in self.remote_events
      ):
        raw[byte] |= 1 << bit
    return list(raw)

  # -- physics -----------------------------------------------------------------
  def step(self) -> None:
    d = self.d
    q, dq = d.qpos[self.qadr], d.qvel[self.vadr]
    with self.lock:
      c = {k: v[MOTOR_OF_JOINT] for k, v in self.cmd.items()}
    tau = c["tau"] + c["kp"] * (c["q"] - q) + c["kd"] * (c["dq"] - dq)
    if self.n_cmd == 0:  # no controller yet: gearbox drag only, not fully limp
      tau = -PASSIVE_KD * dq
      c["mode"][:] = 1
    self.tau = np.clip(tau * (c["mode"] > 0), -PEAK_TORQUE, PEAK_TORQUE)
    d.ctrl[self.act] = self.tau
    mat = d.xmat[self.pelvis].reshape(3, 3)
    self.v_sum += np.array([*(mat.T @ d.qvel[0:3])[:2], d.qvel[5]])
    self.v_n += 1

    d.qfrc_applied[:] = 0.0
    if self.band_on:
      if self.lowering:
        self.anchor[2] -= 0.1 * self.m.opt.timestep
      p = self._band_point()
      mujoco.mj_jac(self.m, d, self._jacp, None, p, self.torso)
      v = self._jacp @ d.qvel
      f = BAND_K * (self.anchor - p) - BAND_D * v
      f[2] = max(f[2], 0.0)  # a rope only pulls up
      w_world = d.xmat[self.pelvis].reshape(3, 3) @ d.qvel[3:6]
      mujoco.mj_applyFT(
        self.m, d, f, -BAND_ROT_D * w_world, p, self.torso, d.qfrc_applied
      )
      if self.lowering and f[2] < 0.5 * 9.81 * self.m.body_subtreemass[self.pelvis]:
        self.lowering = False
        print(
          "[bridge] feet on the ground, band still carries ~half the weight",
          flush=True,
        )
    mujoco.mj_step(self.m, d)

  def on_key(self, key: int) -> None:
    """Viewer thread: queue it; the physics loop applies it between steps."""
    self.keys.append(key)

  def handle_key(self, key: int) -> None:
    if key == ord("7"):
      self.anchor[2] -= 0.05
    elif key == ord("8"):
      self.anchor[2] += 0.05
    elif key == ord("9"):
      self.band_on = not self.band_on
      self.anchor = self._anchor_above(self._band_point())
      self.lowering = False
      print(f"[bridge] band {'on' if self.band_on else 'off'}", flush=True)
    elif key == 259:  # Backspace
      with self.lock:
        for v in self.cmd.values():
          v[:] = 0.0
      self.reset()


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("--net", default="lo")
  ap.add_argument(
    "--model",
    default=str(MODEL_XML),
    help="bridge MJCF (tools/make_sim_model.py): the braced G1 of the bundle",
  )
  ap.add_argument("--domain", type=int, default=1)
  ap.add_argument("--headless", action="store_true")
  ap.add_argument(
    "--igpu", action="store_true", help="render on the default (integrated) GPU"
  )
  ap.add_argument("--hang", type=float, default=0.08, help="feet this far above ground")
  ap.add_argument(
    "--lower-at",
    type=float,
    default=-1.0,
    help="lower the band at this time (s) until the feet take half the weight",
  )
  ap.add_argument(
    "--release-at", type=float, default=-1.0, help="release the band at this time (s)"
  )
  ap.add_argument(
    "--duration", type=float, default=0.0, help="headless: stop after N s"
  )
  ap.add_argument(
    "--remote-script",
    default="",
    help="fake remote: 't:Button' presses (Start, A, B, X, Y, Select, Up...; "
    "'R1+A' together), 't:ly=0.6' sets a stick, e.g. '5:Start,12:R1+A,20:ly=0.6'",
  )
  ap.add_argument(
    "--heat",
    default="",
    help="MOTOR:RATE:START:SECONDS fake heating of one motor's temperature reading "
    "(C/s from sim time START for SECONDS, then cooling 0.5 C/s), for the guard",
  )
  args = ap.parse_args()
  if args.domain == 0 or args.net != "lo":
    sys.exit("sim_bridge only runs on loopback (--net lo) with a domain other than 0")

  from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
  )
  from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

  ChannelFactoryInitialize(args.domain, args.net)
  br = Bridge(args)
  pub = ChannelPublisher("rt/lowstate", LowState_)
  pub.Init()
  sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
  sub.Init(br.on_cmd, 10)
  state = unitree_hg_msg_dds__LowState_()
  print(
    f"[bridge] {Path(args.model).name}, {br.m.body_subtreemass[br.pelvis]:.2f} kg, "
    f"{1 / br.m.opt.timestep:.0f} Hz physics, lowstate 500 Hz on {args.net} "
    f"domain {args.domain}. Robot hangs on a band {args.hang:.2f} m up.",
    flush=True,
  )

  viewer = None
  if not args.headless:
    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(
      br.m, br.d, key_callback=br.on_key, show_left_ui=False, show_right_ui=False
    )
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = br.pelvis
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.0, 135.0, -15.0

  dt = br.m.opt.timestep
  t0 = time.perf_counter()
  n = 0
  last_print, last_wall, next_sync = 0.0, t0, t0
  try:
    while viewer is None or viewer.is_running():
      sim_t = n * dt
      if 0 <= args.lower_at <= sim_t < args.lower_at + dt and br.band_on:
        br.lowering = True
        print("[bridge] lowering band", flush=True)
      if 0 <= args.release_at <= sim_t < args.release_at + dt and br.band_on:
        br.band_on = br.lowering = False
        print("[bridge] band released", flush=True)
      while br.keys:
        br.handle_key(br.keys.pop(0))
      br.step()
      n += 1
      if n % 2 == 0:
        br.fill_state(state, n)
        pub.Write(state)
      if viewer is not None and time.perf_counter() >= next_sync:
        next_sync = time.perf_counter() + 1 / 30
        viewer.sync()
      if sim_t - last_print >= 2.0:
        rt = (sim_t - last_print) / max(time.perf_counter() - last_wall, 1e-6)
        last_print, last_wall = sim_t, time.perf_counter()
        p = br.d.xpos[br.pelvis]
        mat = br.d.xmat[br.pelvis].reshape(3, 3)
        heading = math.degrees(math.atan2(mat[1, 0], mat[0, 0]))
        v = br.v_sum / max(br.v_n, 1)
        br.v_sum[:], br.v_n = 0.0, 0
        print(
          f"[bridge] t {sim_t:5.1f}s pelvis z {p[2]:.3f} xy {p[0]:+.2f} {p[1]:+.2f} "
          f"heading {heading:+6.1f} | body vel fwd {v[0]:+.2f} lat {v[1]:+.2f} "
          f"yaw {v[2]:+.2f} | band {'on' if br.band_on else 'off'} "
          f"lowcmds {br.n_cmd} | {rt:.2f}x real time",
          flush=True,
        )
      if args.duration and sim_t >= args.duration:
        break
      ahead = t0 + n * dt - time.perf_counter()
      if ahead > 0:
        time.sleep(ahead)
  except KeyboardInterrupt:
    pass
  finally:
    if viewer is not None:
      viewer.close()


if __name__ == "__main__":
  main()
