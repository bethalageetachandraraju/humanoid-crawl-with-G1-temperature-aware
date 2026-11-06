"""Walk controller for the braced G1 29-DoF over Unitree SDK2 (unitree_hg LowCmd).

Same code for the real robot and for DDS sim-to-sim against sim_bridge.py.

  # sim-to-sim (loopback, DDS domain 1 - can never reach the robot)
  python g1_walk_real.py --sim
  # real robot, read-only: prints IMU / joints, publishes nothing
  python g1_walk_real.py --real --net enp3s0 --monitor
  # real robot, policy computed but NOT applied (robot holds the stand pose)
  python g1_walk_real.py --real --net enp3s0 --wrists braced --dry-run
  # real robot, walking
  python g1_walk_real.py --real --net enp3s0 --wrists braced

Modes: DAMP (start, e-stop) -> 1: STANDUP (slow move to stand pose) -> STAND
-> 2: WALK (policy). Space / 0 = damping at any time.
Loop: LowCmd at 500 Hz, policy at 50 Hz (each target held for 10 writes).
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from pathlib import Path

import numpy as np
from g1_config import (
  CONTROL_DT,
  JOINTS,
  KD,
  KP,
  LOWCMD_DT,
  MAX_JOINT_SPEED,
  MAX_TILT_DEG,
  MOTOR_OF_JOINT,
  NUM_JOINTS,
  NUM_MOTORS,
  PEAK_TORQUE,
  STAND_POSE,
  STATE_TIMEOUT_S,
  TEMP_STOP_C,
  TEMP_WARN_C,
  WRIST_DAMP_KD,
  WRIST_HOLD_KD,
  WRIST_HOLD_KP,
  WRIST_MOTORS,
  check_against_chain,
  soft_limits,
)
from inputs import Remote, TerminalKeys
from walk_policy import WalkPolicy, gravity_from_quat

HERE = Path(__file__).resolve().parent
# Walk-only mode uses the bundle's walk policy (same observation and scale).
DEFAULT_POLICY = HERE.parent / "bundles" / "g1_chain_v3" / "walk.onnx"
DECIMATION = round(CONTROL_DT / LOWCMD_DT)  # 10
WRIST_IDX = np.array(sorted(WRIST_MOTORS))

HELP = """keys (this terminal):
  1      stand up slowly to the stand pose (from DAMP or WALK)
  2      start the walk policy (from STAND)
  Space  DAMPING (e-stop)      0  damping too
  w/s    forward / back 0.1 m/s     q/e  left / right 0.1 m/s
  a/d    turn left / right 0.1 rad/s     x  zero command
  Ctrl+C quit (robot is left in damping)
remote (--remote): Start = stand up, A = walk, B = DAMPING,
  left stick = forward / sideways, right stick = turn"""


def smoothstep(s: float) -> float:
  s = min(max(s, 0.0), 1.0)
  return s * s * (3.0 - 2.0 * s)


class Controller:
  MODES = ("DAMP", "STANDUP", "STAND", "WALK")
  LOG_PREFIX = "walk"

  def __init__(self, args: argparse.Namespace, policy: WalkPolicy | None) -> None:
    self.args = args
    self.policy = policy
    self.low_state = None
    self.state_stamp = 0.0
    self.n_states = 0
    self.remote = Remote()
    self.remote_events: list[str] = []
    self.mode = "DAMP"
    self.mode_t0 = time.monotonic()
    self.q0 = STAND_POSE.copy()
    self.wrist_q0 = np.zeros(len(WRIST_IDX))
    self.target = STAND_POSE.copy()
    self.policy_target = STAND_POSE.copy()
    self.action = np.zeros(NUM_JOINTS)
    self.obs = np.zeros(84)
    self.cmd = np.zeros(3)
    self.cmd_goal = np.zeros(3)
    self.cmd_lo = np.array([args.min_vx, -args.max_vy, -args.max_wz])
    self.cmd_hi = np.array([args.max_vx, args.max_vy, args.max_wz])
    self.soft_lo, self.soft_hi = soft_limits(args.soft_limit)
    self.tick_n = 0
    self.estop_reason = ""
    self.clamped = 0
    self.late = 0
    self.loop_ms = 0.0
    self.policy_ms = 0.0
    self.dry_stats = (0.0, 0.0, "")
    self.log: dict[str, list] = {
      k: []
      for k in (
        "t", "mode", "q", "dq", "gravity", "gyro", "cmd", "target", "action",
        "tau_est", "temp_max",
      )
    }  # fmt: skip
    self.publisher = None
    self.crc = None
    self.lowcmd = None
    self.quit_requested = False  # e.g. the remote's Select in g1_chain_real
    # Loop ticks per 50 Hz policy step (500 Hz loop: 10; 100 Hz: 2).
    self.decimation = max(1, round(CONTROL_DT * args.loop_hz))

  # -- DDS --------------------------------------------------------------------
  def on_state(self, msg) -> None:
    self.low_state = msg
    self.state_stamp = time.monotonic()
    self.n_states += 1
    if self.args.remote:
      self.remote_events += self.remote.update(msg.wireless_remote)

  def init_publisher(self) -> None:
    from unitree_sdk2py.core.channel import ChannelPublisher
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.utils.crc import CRC

    self.lowcmd = unitree_hg_msg_dds__LowCmd_()
    self.crc = CRC()
    self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    self.publisher.Init()

  # -- state ------------------------------------------------------------------
  def read(self, st):
    ms = st.motor_state
    q = np.array([ms[m].q for m in MOTOR_OF_JOINT])
    dq = np.array([ms[m].dq for m in MOTOR_OF_JOINT])
    quat = np.array(st.imu_state.quaternion, dtype=np.float64)
    gyro = np.array(st.imu_state.gyroscope, dtype=np.float64)
    return q, dq, gravity_from_quat(quat), gyro

  @staticmethod
  def tilt_deg(gravity: np.ndarray) -> float:
    return math.degrees(math.acos(min(1.0, max(-1.0, -gravity[2]))))

  # -- modes ------------------------------------------------------------------
  def set_mode(self, mode: str, q: np.ndarray | None = None, st=None) -> None:
    if mode == self.mode:
      return
    print(f"\n[{time.strftime('%H:%M:%S')}] {self.mode} -> {mode}", flush=True)
    if mode == "STANDUP":
      # From DAMP start at the measured pose; from WALK at the last target.
      self.q0 = self.target.copy() if self.mode == "WALK" else q.copy()
      if st is not None:
        self.wrist_q0 = np.array([st.motor_state[m].q for m in WRIST_IDX])
    if mode == "WALK":
      assert self.policy is not None
      self.policy.reset()
      self.cmd[:] = 0.0
      self.cmd_goal[:] = 0.0
    self.mode = mode
    self.mode_t0 = time.monotonic()

  def estop(self, reason: str) -> None:
    if self.mode != "DAMP":
      self.estop_reason = reason
      print(f"\n!!! DAMPING: {reason}", flush=True)
      self.set_mode("DAMP")

  def handle_key(self, k: str, q: np.ndarray, dq: np.ndarray, grav, st) -> None:
    if k in (" ", "0"):
      self.estop("operator")
    elif k == "1":
      if self.mode in ("DAMP", "WALK"):
        self.set_mode("STANDUP", q, st)
    elif k == "2":
      self.try_walk(dq, grav)
    elif k in "wsqeadx" and self.mode == "WALK" and not self.args.remote:
      step = {"w": (0, 0.1), "s": (0, -0.1), "q": (1, 0.1), "e": (1, -0.1)}
      step.update({"a": (2, 0.1), "d": (2, -0.1)})
      if k == "x":
        self.cmd_goal[:] = 0.0
      else:
        i, dv = step[k]
        self.cmd_goal[i] = round(self.cmd_goal[i] + dv, 3)
      np.clip(self.cmd_goal, self.cmd_lo, self.cmd_hi, out=self.cmd_goal)
      print(
        f"\ncmd goal vx {self.cmd_goal[0]:+.2f} vy {self.cmd_goal[1]:+.2f} "
        f"wz {self.cmd_goal[2]:+.2f}",
        flush=True,
      )

  def try_walk(self, dq: np.ndarray, grav: np.ndarray) -> None:
    if self.args.dry_run:
      print("\n--dry-run: walking is disabled", flush=True)
    elif self.mode != "STAND":
      print(f"\nwalk only from STAND (now {self.mode})", flush=True)
    elif time.monotonic() - self.mode_t0 < 1.0:
      print("\nwait 1 s in STAND first", flush=True)
    elif np.abs(dq).max() > 2.0 or self.tilt_deg(grav) > 20.0:
      print(
        f"\nnot still enough: |dq|max {np.abs(dq).max():.1f} rad/s, "
        f"tilt {self.tilt_deg(grav):.0f} deg",
        flush=True,
      )
    else:
      self.set_mode("WALK")

  # -- one 2 ms tick ----------------------------------------------------------
  def tick(self, keys: list[str]) -> None:
    now = time.monotonic()
    st = self.low_state
    if st is None or now - self.state_stamp > STATE_TIMEOUT_S:
      self.estop("lowstate timeout")
      self.publish(None)
      return
    q, dq, grav, gyro = self.read(st)

    for k in keys:
      self.handle_key(k, q, dq, grav, st)
    for b in self.remote_events:
      if b == "B":
        self.estop("remote B")
      elif b == "Start" and self.mode in ("DAMP", "WALK"):
        self.set_mode("STANDUP", q, st)
      elif b == "A":
        self.try_walk(dq, grav)
    self.remote_events.clear()

    self.check_safety(st, dq, grav)

    policy_tick = self.tick_n % self.decimation == 0
    if self.mode == "STANDUP":
      s = smoothstep((now - self.mode_t0) / self.args.standup_s)
      self.target = self.q0 + (STAND_POSE - self.q0) * s
      if s >= 1.0:
        self.set_mode("STAND")
    elif self.mode == "STAND":
      self.target = STAND_POSE.copy()
      if self.args.dry_run and policy_tick and self.policy is not None:
        self.run_policy(grav, gyro, q, dq)
        dev = np.abs(self.policy_target - STAND_POSE)
        self.dry_stats = (
          float(np.abs(self.action).max()),
          float(dev.max()),
          JOINTS[int(dev.argmax())],
        )
    elif self.mode == "WALK" and policy_tick:
      lim = np.array([0.5, 0.5, 1.0]) * CONTROL_DT  # accel limits per step
      self.cmd += np.clip(self.cmd_goal - self.cmd, -lim, lim)
      self.run_policy(grav, gyro, q, dq)
      w = smoothstep((now - self.mode_t0) / self.args.blend_s)
      tgt = (1.0 - w) * STAND_POSE + w * self.policy_target
      clipped = np.clip(tgt, self.soft_lo, self.soft_hi)
      self.clamped += int(np.any(clipped != tgt))
      self.target = clipped

    if self.args.remote and self.mode == "WALK":
      dz = lambda v: 0.0 if abs(v) < 0.1 else v  # noqa: E731
      self.cmd_goal = np.clip(
        np.array([
          dz(self.remote.ly) * (self.args.max_vx if self.remote.ly > 0 else -self.args.min_vx),
          -dz(self.remote.lx) * self.args.max_vy,
          -dz(self.remote.rx) * self.args.max_wz,
        ]),
        self.cmd_lo,
        self.cmd_hi,
      )  # fmt: skip

    if policy_tick:
      self.record(now, q, dq, grav, gyro, st)
    self.publish(st)
    self.tick_n += 1

  def check_safety(self, st, dq: np.ndarray, grav: np.ndarray) -> None:
    """Damping on tilt, joint speed or motor temperature while driven."""
    if self.mode == "DAMP":
      return
    tilt = self.tilt_deg(grav)
    if tilt > self.args.max_tilt:
      self.estop(f"tilt {tilt:.0f} deg > {self.args.max_tilt:.0f}")
    elif np.abs(dq).max() > MAX_JOINT_SPEED:
      j = int(np.abs(dq).argmax())
      self.estop(f"{JOINTS[j]} speed {dq[j]:.1f} rad/s")
    elif self.tick_n % 50 == 0:
      temps = [max(st.motor_state[m].temperature) for m in range(NUM_MOTORS)]
      if max(temps) >= TEMP_STOP_C:
        self.estop(f"motor {int(np.argmax(temps))} at {max(temps)} C")

  def run_policy(self, grav, gyro, q, dq) -> None:
    assert self.policy is not None
    t0 = time.perf_counter()
    try:
      self.policy_target, self.action, self.obs = self.policy.step(
        grav, gyro, self.cmd, q, dq
      )
    except FloatingPointError as e:
      self.estop(str(e))
    self.policy_ms = 1e3 * (time.perf_counter() - t0)

  def publish(self, st) -> None:
    if self.publisher is None:
      return
    c = self.lowcmd
    c.mode_pr = 0  # PR: ankles / waist as serial pitch-roll joints
    c.mode_machine = st.mode_machine if st is not None else c.mode_machine
    active = self.mode != "DAMP"
    for i, m in enumerate(MOTOR_OF_JOINT):
      mc = c.motor_cmd[m]
      mc.mode = 1
      mc.q = float(self.target[i]) if active else 0.0
      mc.dq = 0.0
      mc.tau = 0.0
      mc.kp = float(KP[i]) if active else 0.0
      mc.kd = float(KD[i])
    for j, m in enumerate(WRIST_IDX):
      mc = c.motor_cmd[m]
      mc.mode = 1
      mc.dq = 0.0
      mc.tau = 0.0
      if active and self.args.wrists == "hold":
        s = smoothstep((time.monotonic() - self.mode_t0) / self.args.standup_s)
        mc.q = float(self.wrist_q0[j] * (1 - s)) if self.mode == "STANDUP" else 0.0
        mc.kp, mc.kd = float(WRIST_HOLD_KP), float(WRIST_HOLD_KD)
      else:
        mc.q, mc.kp, mc.kd = 0.0, 0.0, float(WRIST_DAMP_KD)
    c.crc = self.crc.Crc(c)
    self.publisher.Write(c)

  def record(self, now, q, dq, grav, gyro, st) -> None:
    L = self.log
    L["t"].append(now)
    L["mode"].append(self.MODES.index(self.mode))
    L["q"].append(q)
    L["dq"].append(dq)
    L["gravity"].append(grav)
    L["gyro"].append(gyro)
    L["cmd"].append(self.cmd.copy())
    L["target"].append(self.target.copy())
    L["action"].append(self.action.copy())
    L["tau_est"].append(np.array([st.motor_state[m].tau_est for m in MOTOR_OF_JOINT]))
    L["temp_max"].append(
      max(max(st.motor_state[m].temperature) for m in range(NUM_MOTORS))
    )

  def status(self) -> str:
    st = self.low_state
    if st is None:
      return "waiting for rt/lowstate ..."
    q, dq, grav, gyro = self.read(st)
    kp = KP if self.mode != "DAMP" else 0.0
    demand = np.abs(kp * (self.target - q) - KD * dq) / PEAK_TORQUE
    j = int(demand.argmax())
    temps = [max(st.motor_state[m].temperature) for m in range(NUM_MOTORS)]
    hot = " HOT" if max(temps) >= TEMP_WARN_C else ""
    s = (
      f"[{self.mode:7s}] tilt {self.tilt_deg(grav):4.1f}deg |dq|max {np.abs(dq).max():5.2f} "
      f"torque/peak {demand[j]:.2f} ({JOINTS[j][:-6]}) "
      f"Tmax {max(temps)}C{hot} loop {self.loop_ms:.2f}ms pol {self.policy_ms:.2f}ms "
      f"late {self.late}"
    )
    if self.mode == "WALK":
      s += f" | cmd {self.cmd[0]:+.2f} {self.cmd[1]:+.2f} {self.cmd[2]:+.2f} clamp {self.clamped}"
    if self.mode == "STAND" and self.args.dry_run:
      a, d, jn = self.dry_stats
      s += f" | dry: |a|max {a:.2f}, max |target-stand| {d:.2f} rad ({jn})"
    if self.mode == "DAMP" and self.estop_reason:
      s += f" | last stop: {self.estop_reason}"
    return s

  def save_log(self) -> None:
    if not self.log["t"]:
      return
    out = (
      Path(self.args.log_dir)
      / f"{self.LOG_PREFIX}_{'sim' if self.args.sim else 'real'}_"
      f"{time.strftime('%Y%m%d_%H%M%S')}.npz"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
      out,
      joints=np.array(JOINTS),
      modes=np.array(self.MODES),
      policy=str(self.policy.path) if self.policy else "",
      **{k: np.array(v) for k, v in self.log.items()},
    )
    print(f"log -> {out}")


# ---------------------------------------------------------------------------


def confirm(prompt: str, answer: str) -> None:
  if input(prompt).strip().lower() != answer:
    sys.exit("aborted")


def release_builtin_controller(auto: bool = False) -> None:
  from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
    MotionSwitcherClient,
  )

  msc = MotionSwitcherClient()
  msc.SetTimeout(5.0)
  msc.Init()
  status, result = msc.CheckMode()
  name = (result or {}).get("name", "")
  if not name:
    print("built-in motion control: none active")
    return
  print(f"built-in motion control active: '{name}'")
  if not auto:
    confirm(
      "Releasing it makes the robot go LIMP. Robot hanging on the gantry? "
      "Type 'release': ",
      "release",
    )
  for _ in range(10):
    msc.ReleaseMode()
    time.sleep(1.0)
    status, result = msc.CheckMode()
    if not (result or {}).get("name", ""):
      print("released")
      return
  sys.exit("could not release the built-in controller")


def monitor(ctl: Controller) -> None:
  """Read-only: never creates a publisher."""
  print(
    "READ-ONLY monitor. Tilt the robot forward: gravity x should go positive;\n"
    "tilt it to its left: gravity y should go negative. Ctrl+C to stop.\n"
  )
  n_last, t_last = 0, time.monotonic()
  while True:
    time.sleep(0.5)
    st = ctl.low_state
    if st is None:
      print("waiting for rt/lowstate ...")
      continue
    q, dq, grav, gyro = ctl.read(st)
    now = time.monotonic()
    rate = (ctl.n_states - n_last) / (now - t_last)
    n_last, t_last = ctl.n_states, now
    dev = q - STAND_POSE
    j = int(np.abs(dev).argmax())
    temps = [max(st.motor_state[m].temperature) for m in range(NUM_MOTORS)]
    wr = " ".join(f"{st.motor_state[m].q:+.2f}" for m in WRIST_IDX)
    held = [b for b, p in ctl.remote.pressed.items() if p] if ctl.args.remote else []
    print(
      f"{rate:5.0f} Hz mode_machine {st.mode_machine} | gravity {grav[0]:+.3f} "
      f"{grav[1]:+.3f} {grav[2]:+.3f} (tilt {ctl.tilt_deg(grav):4.1f}) gyro "
      f"{gyro[0]:+.2f} {gyro[1]:+.2f} {gyro[2]:+.2f} | max|q-stand| {abs(dev[j]):.2f} "
      f"({JOINTS[j][:-6]}) | wrist p/y L,R {wr} | Tmax {max(temps)}C"
      + (f" | remote {held} ly {ctl.remote.ly:+.2f}" if ctl.args.remote else ""),
      flush=True,
    )


def make_parser(doc: str) -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(
    description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  where = ap.add_mutually_exclusive_group(required=True)
  where.add_argument(
    "--sim", action="store_true", help="talk to sim_bridge.py (lo, domain 1)"
  )
  where.add_argument("--real", action="store_true", help="talk to the robot")
  ap.add_argument("--net", default=None, help="network interface (real: e.g. enp3s0)")
  ap.add_argument(
    "--wrists",
    choices=("braced", "hold"),
    default=None,
    help="braced: wrist pitch/yaw motors in damping (braces fitted); "
    "hold: PD them to 0 (no braces). Required with --real.",
  )
  ap.add_argument("--monitor", action="store_true", help="read-only, publish nothing")
  ap.add_argument(
    "--dry-run", action="store_true", help="run the policy, never apply it"
  )
  ap.add_argument(
    "--remote", action="store_true", help="use the Unitree wireless remote"
  )
  ap.add_argument("--standup-s", type=float, default=3.0)
  ap.add_argument("--blend-s", type=float, default=0.5)
  ap.add_argument("--max-vx", type=float, default=0.5)
  ap.add_argument("--min-vx", type=float, default=-0.3)
  ap.add_argument("--max-vy", type=float, default=0.3)
  ap.add_argument("--max-wz", type=float, default=0.3)
  ap.add_argument("--max-tilt", type=float, default=MAX_TILT_DEG)
  ap.add_argument(
    "--soft-limit",
    type=float,
    default=0.95,
    help="clamp targets to this fraction of each joint range",
  )
  ap.add_argument("--log-dir", default=str(HERE / "logs"))
  ap.add_argument(
    "--script", default="", help="sim only: 't:key,...' e.g. '1:1,6:2,8:w'"
  )
  ap.add_argument(
    "--duration", type=float, default=0.0, help="sim only: quit after N s"
  )
  ap.add_argument(
    "--loop-hz",
    type=int,
    default=500,
    choices=(50, 100, 250, 500),
    help="motor-command rate; the policy always runs at 50 Hz (robot's own "
    "computer: 100)",
  )
  ap.add_argument(
    "--poll-state",
    action="store_true",
    help="take the newest robot state once per tick instead of decoding every "
    "message (slow CPUs, e.g. the robot's own computer)",
  )
  ap.add_argument(
    "--yes",
    action="store_true",
    help="skip the typed confirmations (checklist, releasing the built-in "
    "controller): you have checked them",
  )
  return ap


def check_args(args: argparse.Namespace) -> int:
  """Validate --sim / --real options; returns the DDS domain."""
  if args.sim:
    args.net = args.net or "lo"
    args.wrists = args.wrists or "braced"
    return 1
  if not args.net:
    sys.exit("--real needs --net <interface on 192.168.123.x>, e.g. --net enp3s0")
  if args.script or args.duration:
    sys.exit("--script / --duration are for --sim only")
  if args.wrists is None and not args.monitor:
    sys.exit(
      "--real needs --wrists braced (braces fitted) or --wrists hold (no braces)"
    )
  args.wrists = args.wrists or "braced"
  return 0


def real_robot_checklist(args: argparse.Namespace, extra: str = "") -> None:
  print(
    "\nREAL ROBOT. Before continuing check that:\n"
    "  - the robot hangs on the gantry / harness, feet off the ground\n"
    "  - the waist is UNLOCKED (the policy drives waist roll and pitch)\n"
    f"  - wrists: '{args.wrists}' matches the hardware (braces fitted -> braced)\n"
    "  - someone holds the remote, and this terminal has focus (Space = damping)\n"
    f"  - command limits: vx {args.min_vx}..{args.max_vx}, vy +-{args.max_vy}, "
    f"wz +-{args.max_wz}\n" + extra
  )
  if args.yes:
    print("--yes: checklist and release confirmed on the command line")
  else:
    confirm("Type 'yes' to continue: ", "yes")


class _SafeStream:
  """stdout / stderr that survive a dead terminal: when the ssh / WiFi session
  that started the controller goes away, writing to it fails; from then on the
  text goes to ``logs/console_<time>.txt`` and the control loop keeps running."""

  def __init__(self, stream, fallback: Path) -> None:
    self.stream, self.fallback, self.file = stream, fallback, None

  def _switch(self):
    if self.file is None:
      self.fallback.parent.mkdir(parents=True, exist_ok=True)
      self.file = open(self.fallback, "a", buffering=1)  # noqa: SIM115
    return self.file

  def write(self, text: str) -> int:
    if self.file is None:
      try:
        return self.stream.write(text)
      except (OSError, ValueError):
        pass
    return self._switch().write(text)

  def flush(self) -> None:
    if self.file is None:
      try:
        self.stream.flush()
        return
      except (OSError, ValueError):
        self._switch()
    self.file.flush()

  def __getattr__(self, name):
    return getattr(self.stream, name)


class _StatePoller:
  """The newest rt/lowstate on demand (--poll-state). The SDK's subscriber
  decodes every message (~1000/s) in Python through two threads: cheap on the
  laptop, but on the robot's own computer it slowed the 50 Hz policy to ~38 Hz
  and the walk went unstable. This reader keeps only the newest sample, and the
  loop takes it once per tick."""

  def __init__(self, low_state_type) -> None:
    from cyclonedds.internal import InvalidSample
    from cyclonedds.qos import Policy, Qos
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.core.channel import ChannelFactory

    participant = ChannelFactory()._ChannelFactory__participant  # the SDK's own
    topic = Topic(participant, "rt/lowstate", low_state_type)
    self.reader = DataReader(participant, topic, Qos(Policy.History.KeepLast(1)))
    self.invalid = InvalidSample

  def poll(self, ctl: Controller) -> None:
    for sample in self.reader.take(N=1):
      if not isinstance(sample, self.invalid):
        ctl.on_state(sample)


def run(args: argparse.Namespace, ctl: Controller, domain: int, help_text: str) -> None:
  """DDS setup, read-only monitor or the control loop (--loop-hz), clean
  shutdown."""
  from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
  from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

  ChannelFactoryInitialize(domain, args.net)
  poller = None
  if args.poll_state and not args.monitor:
    poller = _StatePoller(LowState_)
  else:
    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init(ctl.on_state, 10)

  t_wait = time.monotonic()
  while ctl.low_state is None:
    if time.monotonic() - t_wait > 5.0:
      sys.exit(f"no rt/lowstate on {args.net} (domain {domain}) after 5 s")
    time.sleep(0.05)
    if poller is not None:
      poller.poll(ctl)
  print(
    f"lowstate OK on {args.net}, domain {domain}, mode_machine {ctl.low_state.mode_machine}"
  )

  if args.monitor:
    try:
      monitor(ctl)
    except KeyboardInterrupt:
      return

  if args.real:
    release_builtin_controller(auto=args.yes)
  # Keep running if the SSH / WiFi session that started us drops (the remote
  # still drives the robot; printing moves to a file); a plain kill stops
  # cleanly: damping, then the log.
  console = Path(args.log_dir) / f"console_{time.strftime('%Y%m%d_%H%M%S')}.txt"
  sys.stdout = _SafeStream(sys.stdout, console)
  sys.stderr = _SafeStream(sys.stderr, console)
  signal.signal(signal.SIGHUP, signal.SIG_IGN)

  def _stop(signum, frame):
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, _stop)
  ctl.init_publisher()
  print("\n" + help_text + "\n\nstarting in DAMP. Press 1 to stand up.\n", flush=True)

  script = []
  for item in filter(None, args.script.split(",")):
    t, k = item.split(":", 1)
    script.append((float(t), " " if k == "space" else k))
  script.sort()

  period = 1.0 / args.loop_hz
  print(
    f"loop {args.loop_hz} Hz (policy every {ctl.decimation} ticks = 50 Hz), robot "
    f"state {'polled once per tick' if poller else 'every message'}",
    flush=True,
  )
  t_start = time.monotonic()
  next_t = time.perf_counter()
  next_status = 0.0
  try:
    with TerminalKeys() as term:
      while True:
        t_loop = time.perf_counter()
        el = time.monotonic() - t_start
        keys = term.get_all()
        while script and script[0][0] <= el:
          keys.append(script.pop(0)[1])
        if poller is not None:
          poller.poll(ctl)
        ctl.tick(keys)
        ctl.loop_ms = 1e3 * (time.perf_counter() - t_loop)
        if el >= next_status:
          next_status = el + 0.5
          print(ctl.status(), flush=True)
        if (args.duration and el > args.duration) or ctl.quit_requested:
          break
        next_t += period
        slack = next_t - time.perf_counter()
        if slack > 0:
          time.sleep(slack)
        elif slack < -0.02:
          ctl.late += 1
          next_t = time.perf_counter()
  except KeyboardInterrupt:
    print("\nCtrl+C")
  finally:
    ctl.estop("exit")
    for _ in range(int(1.0 / period)):  # keep sending damping for 1 s
      ctl.publish(ctl.low_state)
      time.sleep(period)
    ctl.save_log()
    if args.real:
      print(
        "Robot left in DAMPING (low-level). Restart Unitree's controller "
        "from the remote / power cycle to get the built-in modes back."
      )


def main() -> None:
  ap = make_parser(__doc__)
  ap.add_argument("--policy", default=str(DEFAULT_POLICY))
  args = ap.parse_args()
  domain = check_args(args)
  for p in check_against_chain():
    print("config:", p)
  policy = None if args.monitor else WalkPolicy(args.policy)
  if policy is not None:
    print(f"policy: {policy.path}")
  if args.real and not args.monitor:
    real_robot_checklist(args)
  run(args, Controller(args, policy), domain, HELP)


if __name__ == "__main__":
  main()
