"""Walk -> crawl -> stand -> walk on the braced G1 (Unitree SDK2), keyboard driven.

Runs the chain bundle's controller (default ../bundles/g1_chain_v3/
chain_controller.py, 2026-10-30: walk.onnx = walk_hard_track/55499,
stand_crawl.onnx = push_v1/30200, crawl styles creepy (default) and scuttle)
inside the robot I/O, safety stops and damping of g1_walk_real.py. Another
bundle: --bundle <dir> (see ../README.md, "Bundle format"). Motor temperatures
go through thermal_guard.py (the training team's guard): stand up from the
crawl side at 72 C or > 1 C/s heating, damping at 85 C, no going down again
until every motor is < 60 C.

  # gantry, feet off the ground: move through the 8 stand<->crawl keyframes,
  # NO policy (checks joint directions and clearances)
  python g1_chain_real.py --real --net enp3s0 --wrists braced --schedule-only
  # real robot: walk, c down to crawl, crawl, u stand up, walk
  python g1_chain_real.py --real --net enp3s0 --wrists braced
  # same code against sim_bridge.py (loopback, DDS domain 1)
  python g1_chain_real.py --sim

Modes: DAMP -> 1: STANDUP -> STAND -> 2: CHAIN (all three policies, shadow mode).
Inside CHAIN the bundle's state machine runs: WALK -c-> STOPPING -> DOWN -> HOLD
-> CRAWL -u-> CRAWL_STOP -> HOLD -> UP -> STAND_HOLD -> WALK. Space / 0 = damping
at any time. The gates use the IMU, joint speeds and forward kinematics; the
status line says what a gate is waiting for.
"""

from __future__ import annotations

import os

# One math thread: the policies are small matrix-vector products, and a
# multi-threaded BLAS stalls the 500 Hz loop when other processes are busy
# (measured: 5.7 ms policy steps and hundreds of late ticks). Before numpy loads.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
  os.environ.setdefault(_var, "1")

import argparse
import inspect
import math
import sys
import time
from collections import deque
from pathlib import Path

# Robot's own computer (g1_onboard/): Unitree's SDK and the CycloneDDS Python
# bindings live in ./vendor (nothing installed system-wide); the CycloneDDS C
# library is the one Unitree built in ~/cyclonedds_ws.
_VENDOR = Path(__file__).resolve().parent / "vendor"
if _VENDOR.is_dir():
  sys.path.insert(0, str(_VENDOR))
  _DDS = Path.home() / "cyclonedds_ws" / "install" / "cyclonedds"
  if _DDS.is_dir():
    os.environ.setdefault("CYCLONEDDS_HOME", str(_DDS))
    try:  # make libddsc findable for the bindings' C extension
      import ctypes

      ctypes.CDLL(str(_DDS / "lib" / "libddsc.so.0"), mode=ctypes.RTLD_GLOBAL)
    except OSError:
      pass

import numpy as np
from g1_config import (
  CONTROL_DT,
  JOINT_LOWER,
  JOINT_UPPER,
  JOINTS,
  KD,
  KP,
  MAX_JOINT_SPEED,
  MOTOR_OF_JOINT,
  NUM_MOTORS,
  STAND_POSE,
  STATE_TIMEOUT_S,
  TEMP_STOP_C,
  WALK_ACTION_SCALE,
  WRIST_MOTORS,
)
from g1_walk_real import (
  WRIST_IDX,
  Controller,
  check_args,
  make_parser,
  real_robot_checklist,
  run,
  smoothstep,
)
from thermal_guard import DAMP as GUARD_DAMP
from thermal_guard import STAND_UP as GUARD_STAND_UP
from thermal_guard import ThermalGuard

HERE = Path(__file__).resolve().parent
# On the robot's own computer the bundle sits next to this file (g1_onboard/);
# in real_world_implementation/ it is ../bundles/g1_chain_v3.
DEFAULT_BUNDLE = (
  HERE / "bundle"
  if (HERE / "bundle" / "chain.json").exists()
  else HERE.parent / "bundles" / "g1_chain_v3"
)

HELP = """keys (this terminal):
  1      stand up slowly to the stand pose (from DAMP, or CHAIN while walking)
  2      start the policies (from STAND); the walk policy balances the robot
  c      go down to crawl (stops walking first)     u  stand up again
  w / s  forward speed +{step} / -{step}
  a / d  sideways +{step} (left) / -{step} (right)
  q / e  rotation +{step} (left) / -{step} (right)     x  all to zero
         each press adds to the current command (m/s, rad/s). Limits:
         walk  {walk}
         crawl {crawl}
         the walk stands still below ~0.25 m/s forward: w w w
  v      next crawl style ({styles}); while crawling it switches once
         the crawl command is 0 (x) and the robot is still
  h      crawl step height (0.03 / 0.08 / 0.12 m)
  g      pass the gate the controller waits on (only when the robot is
         visibly settled and the status shows a wrong estimate)
  Space  DAMPING (e-stop)      0  damping too      Ctrl+C quit
thermal guard: on the crawl side the robot stands up by itself (like u) at
  {hot:.0f} C or when a motor heats > {rate:g} C/s; damping at 85 C; after a
  stand-up, c waits until every motor is < {cool:.0f} C. HOLD stands up by itself
  if the crawl gate has not passed after {hold:g} s.
remote (--remote): B or F1 = DAMPING, Start = stand pose, R1+X = start the
  policies (stand / walk), R1+Down = stand -> crawl, R1+Up = crawl -> stand,
  left stick = forward / back / left / right, right stick = rotate, tap R1 /
  R2 = crawl step height up / down, L1 / L2 = speed cap up / down,
  Select = quit (damping, log saved)"""

HELP_SCHEDULE = """--schedule-only (no policy, feet off the ground):
  1      stand up slowly to the stand pose
  2      enter the keyframe schedule at phi = 0
  c      move along the schedule to the crawl pose (phi -> 1)
  u      move back to the stand pose (phi -> 0)     p  pause the motion
  Space  DAMPING (e-stop)      Ctrl+C quit"""

WAITING = ("STOPPING", "HOLD", "CRAWL_STOP", "STAND_HOLD")
CRAWL_SIDE = ("HOLD", "CRAWL", "CRAWL_STOP")  # where the guard stands the robot up
# Unitree remote (--remote), the operator's mapping (2026-10-30). R1 is the
# modifier for the pose changes; a short tap of R1 on its own (pressed and
# released with nothing else) raises the crawl step height.
REMOTE_SINGLE = {
  "B": " ",  # damping (e-stop)
  "F1": " ",  # damping (e-stop)
  "Start": "1",  # stand up slowly to the stand pose
  "R2": "step-",  # crawl step height down
  "L1": "speed+",  # speed cap up (walk or crawl, whichever is running)
  "L2": "speed-",  # speed cap down
  "Select": "quit",  # damping, save the log, exit
}
REMOTE_COMBO = {
  ("R1", "X"): "2",  # start the policies (stand / walk)
  ("R1", "Down"): "c",  # stand -> crawl
  ("R1", "Up"): "u",  # crawl -> stand
}
REMOTE_TAP = {"R1": "step+"}  # crawl step height up
SPEED_STEP = 0.1  # m/s per L1 / L2 press (forward cap; sideways and turn follow)
STEP_HEIGHT_STEP = 0.02  # m per R1 tap / R2 press
# Each press adds key_step along (forward, left, rotate left); x zeroes.
MOVES = {
  "w": (1, 0, 0),
  "s": (-1, 0, 0),
  "a": (0, 1, 0),
  "d": (0, -1, 0),
  "q": (0, 0, 1),
  "e": (0, 0, -1),
  "x": (0, 0, 0),
}
MOTOR_NAMES = {int(m): j[: -len("_joint")] for j, m in zip(JOINTS, MOTOR_OF_JOINT)}
MOTOR_NAMES.update(WRIST_MOTORS)


def hottest(st) -> tuple[int, int]:
  """(motor index, its higher temperature reading in C) of the hottest motor."""
  temps = [max(st.motor_state[m].temperature) for m in range(NUM_MOTORS)]
  m = int(np.argmax(temps))
  return m, int(temps[m])


def load_bundle(bundle: Path, args: argparse.Namespace):
  sys.path.insert(0, str(bundle))
  import chain_controller as cc

  kw = dict(transition_s=args.transition_s, blend_s=args.blend_s, dwell_s=args.dwell_s)
  if "hold_timeout_s" in inspect.signature(cc.ChainController).parameters:
    kw["hold_timeout_s"] = args.hold_timeout_s  # v3 controller
  chain = cc.ChainController(bundle, **kw)
  # The bundle must describe exactly the robot this controller drives.
  assert chain.joints == list(JOINTS), "bundle joint order differs"
  motors = [chain.cfg["motor_index"][j] for j in JOINTS]
  assert motors == list(MOTOR_OF_JOINT), "bundle motor mapping differs"
  assert np.allclose(chain.kp, KP) and np.allclose(chain.kd, KD), "bundle gains differ"
  assert np.allclose(chain.q_stand, STAND_POSE), "bundle stand pose differs"
  assert np.allclose(chain.walk_scale, WALK_ACTION_SCALE), "walk scale differs"
  assert np.allclose(chain.q_lo, JOINT_LOWER, atol=1e-3)
  assert np.allclose(chain.q_hi, JOINT_UPPER, atol=1e-3)
  braced = sorted(v["motor_index"] for v in chain.cfg["braced_joints"].values())
  assert braced == list(WRIST_IDX), "bundle braced motors differ"
  return cc, chain


class ChainRealController(Controller):
  MODES = ("DAMP", "STANDUP", "STAND", "CHAIN", "SCHEDULE")
  LOG_PREFIX = "chain"

  def __init__(self, args: argparse.Namespace, cc, chain) -> None:
    super().__init__(args, policy=None)
    self.cc = cc
    self.chain = chain
    self.chain_mode = ""
    self.ccmd = np.zeros(3)  # crawl command [fwd, lat, turn]
    self.ccmd_goal = np.zeros(3)
    # Speed caps for the keys and the sticks; L1 / L2 change them.
    self.walk_cap = np.array([args.max_vx, args.max_vy, args.max_wz])
    self.walk_back = abs(args.min_vx)
    self.crawl_cap = np.array(
      [args.max_crawl_fwd, args.max_crawl_lat, args.max_crawl_turn]
    )
    self.apply_caps()
    self.tap_held: dict[str, bool] = {}
    self.tap_used: set[str] = set()
    self.phi = 0.0  # --schedule-only
    self.phi_rate = 0.0
    self.chain_mode_t0 = time.monotonic()
    self.wait_warned = False
    self.guard = ThermalGuard(
      warn_c=args.guard_warn,
      stand_up_c=args.guard_stand_up,
      stop_c=TEMP_STOP_C,
      resume_c=args.guard_resume,
      rate_c_per_s=args.guard_rate,
    )
    self.policy_dts: deque = deque(maxlen=50)  # real policy intervals in CHAIN
    self.last_policy_t = None
    self.guard_action = "ok"
    self.guard_up_sent = False
    self.timeout_told = False
    keys = ("chain_mode", "phase", "above_limbs", "pelvis_height", "gate_ok", "ccmd")
    keys += ("temp", "tau_motor", "q_motor", "dq_motor", "guard", "style")
    for k in keys:
      self.log[k] = []

  # -- modes ------------------------------------------------------------------
  def set_mode(self, mode: str, q: np.ndarray | None = None, st=None) -> None:
    if mode == self.mode:
      return
    print(f"\n[{time.strftime('%H:%M:%S')}] {self.mode} -> {mode}", flush=True)
    if mode == "STANDUP" and self.mode == "DAMP" and self.low_state is not None:
      # A new start: clear a latched guard stop (the 85 C check in check_safety
      # still applies), but keep c locked while any motor is still warm.
      self.guard.reset()
      self.guard.allow_down = hottest(self.low_state)[1] < self.guard.resume_c
    if mode == "STANDUP":
      from_policy = self.mode in ("CHAIN", "SCHEDULE")
      self.q0 = self.target.copy() if from_policy else q.copy()
      if st is not None:
        self.wrist_q0 = np.array([st.motor_state[m].q for m in WRIST_IDX])
    if mode == "CHAIN":
      self.policy_dts.clear()
      self.last_policy_t = None
      # All policies start fresh; PD targets blend from the held stand pose.
      self.chain.reset()
      self.chain._set(self.cc.WALK, blend=True)
      self.chain_mode = "WALK"
      self.cmd[:] = 0.0
      self.cmd_goal[:] = 0.0
      self.ccmd[:] = 0.0
      self.ccmd_goal[:] = 0.0
    if mode == "SCHEDULE":
      self.phi, self.phi_rate = 0.0, 0.0
    self.mode = mode
    self.mode_t0 = time.monotonic()

  def can_stand_up(self) -> bool:
    """Key 1 only where a joint-space move to the stand pose is safe."""
    if self.mode == "DAMP":
      return True
    if self.mode == "CHAIN":
      return self.chain.mode_name in ("WALK", "STAND_HOLD")
    return self.mode == "SCHEDULE" and self.phi == 0.0

  def handle_key(self, k: str, q, dq, grav, st) -> None:
    k = k.lower()
    if k in (" ", "0"):
      self.estop("operator")
    elif k == "1":
      if self.can_stand_up():
        self.set_mode("STANDUP", q, st)
      else:
        print("\n1 is only allowed standing (press u and wait first)", flush=True)
    elif k == "2":
      self.try_start(dq, grav)
    elif self.mode == "SCHEDULE":
      if k in "cu":
        self.phi_rate = (1.0 if k == "c" else -1.0) / self.args.schedule_s
      elif k == "p":
        self.phi_rate = 0.0
    elif self.mode == "CHAIN":
      self.chain_key(k)

  def chain_key(self, k: str) -> None:
    ch = self.chain
    if k == "c":
      if self.args.wrists != "braced":
        print(
          "\nthe crawl needs the wrist braces and the 100 mm extensions "
          "(--wrists braced)",
          flush=True,
        )
        return
      m, temp = hottest(self.low_state)
      if ch.mode_name == "WALK" and not self.guard.allow_down:
        print(
          f"\nthermal guard: no crawl until every motor is < "
          f"{self.guard.resume_c:.0f} C (motor {m} {MOTOR_NAMES.get(m, '?')} at "
          f"{temp} C). Walk or stand to cool.",
          flush=True,
        )
        return
      ch.press_down()
      self.cmd[:] = 0.0
      self.cmd_goal[:] = 0.0
    elif k == "u":
      ch.press_up()
      self.ccmd[:] = 0.0
      self.ccmd_goal[:] = 0.0
    elif k == "g":
      if ch.mode_name in WAITING:
        print(f"\noperator passes the {ch.mode_name} gate", flush=True)
        ch.force_gate()
    elif k == "v":
      if len(getattr(ch, "crawl_styles", ())) < 2:
        print("\nthis bundle has one crawl style", flush=True)
        return
      name = ch.request_style()
      when = (
        " (once the crawl command is 0 and it is still)"
        if ch.mode_name in ("CRAWL", "CRAWL_STOP")
        else ""
      )
      note = "  scuttle: bursts of 2-3 min" if name == "scuttle" else ""
      print(f"\ncrawl style -> {name}{when}{note}", flush=True)
    elif k in ("step+", "step-"):
      dv = STEP_HEIGHT_STEP if k == "step+" else -STEP_HEIGHT_STEP
      ch.set_step_height(round(ch.crawl_cmd[3] + dv, 3))
      print(f"\ncrawl step height {ch.crawl_cmd[3]:.2f} m", flush=True)
    elif k in ("speed+", "speed-"):
      self.change_cap(1 if k == "speed+" else -1)
    elif k == "h" and ch.mode_name == "CRAWL":
      hs = self.cc.STEP_HEIGHTS
      i = int(np.argmin([abs(h - ch.crawl_cmd[3]) for h in hs]))
      ch.set_step_height(hs[(i + 1) % len(hs)])
      print(f"\nstep height {ch.crawl_cmd[3]:.2f} m", flush=True)
    elif k in MOVES and not self.args.remote:
      step = np.array(MOVES[k], dtype=float) * self.args.key_step
      if ch.mode_name == "WALK":
        goal, lo, hi = self.cmd_goal, self.cmd_lo, self.cmd_hi
      elif ch.mode_name == "CRAWL":
        # The crawl's turn input is clockwise-positive: rotate left lowers it.
        step[2] = -step[2]
        goal, lo, hi = self.ccmd_goal, self.ccmd_lo, self.ccmd_hi
      else:
        print(f"\nmove keys only while walking or crawling (now {ch.mode_name})")
        return
      if k == "x":
        goal[:] = 0.0
      else:
        wanted = np.round(goal + step, 3)
        goal[:] = np.clip(wanted, lo, hi)
        if np.any(goal != wanted):
          print(f"\nlimit reached ({lo} .. {hi})", end="")
      print(
        f"\n{ch.mode_name.lower()} command: forward {goal[0]:+.2f}  left "
        f"{goal[1]:+.2f}  turn {goal[2]:+.2f}",
        flush=True,
      )

  def try_start(self, dq: np.ndarray, grav: np.ndarray) -> None:
    target = "SCHEDULE" if self.args.schedule_only else "CHAIN"
    if self.args.dry_run:
      print("\n--dry-run: the policies are never applied", flush=True)
    elif self.mode != "STAND":
      print(f"\n2 only from STAND (now {self.mode})", flush=True)
    elif time.monotonic() - self.mode_t0 < 1.0:
      print("\nwait 1 s in STAND first", flush=True)
    elif target == "CHAIN" and (np.abs(dq).max() > 2.0 or self.tilt_deg(grav) > 20.0):
      print(
        f"\nnot still enough: |dq|max {np.abs(dq).max():.1f} rad/s, "
        f"tilt {self.tilt_deg(grav):.0f} deg",
        flush=True,
      )
    else:
      self.set_mode(target)

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
    for k in self.remote_keys():
      if k == "quit":
        print("\nremote Select: quit", flush=True)
        self.quit_requested = True
      else:
        self.handle_key(k, q, dq, grav, st)

    self.check_safety(st, dq, grav)

    policy_tick = self.tick_n % self.decimation == 0
    if self.mode == "STANDUP":
      s = smoothstep((now - self.mode_t0) / self.args.standup_s)
      self.target = self.q0 + (STAND_POSE - self.q0) * s
      if s >= 1.0:
        self.set_mode("STAND")
    elif self.mode == "STAND":
      self.target = STAND_POSE.copy()
      if self.args.dry_run and policy_tick:
        quat = np.array(st.imu_state.quaternion, dtype=np.float64)
        t0 = time.perf_counter()
        tgt = self.chain.step(quat, gyro, q, dq)
        self.policy_ms = 1e3 * (time.perf_counter() - t0)
        dev = np.abs(tgt - STAND_POSE)
        self.dry_stats = (
          float(np.abs(self.chain.walk_last).max()),
          float(dev.max()),
          JOINTS[int(dev.argmax())],
        )
    elif self.mode == "SCHEDULE" and policy_tick:
      self.phi = float(np.clip(self.phi + self.phi_rate * CONTROL_DT, 0.0, 1.0))
      if self.phi in (0.0, 1.0):
        self.phi_rate = 0.0
      self.target = np.clip(self.chain.schedule(self.phi), self.soft_lo, self.soft_hi)
    elif self.mode == "CHAIN" and policy_tick:
      self.chain_step(st, q, dq, gyro)

    if self.args.remote and self.mode == "CHAIN":
      self.remote_command()
    if policy_tick and self.mode != "DAMP":
      self.thermal_guard(now, st)

    if policy_tick:
      self.record(now, q, dq, grav, gyro, st)
    self.publish(st)
    self.tick_n += 1

  def remote_keys(self) -> list[str]:
    """Remote buttons since the last tick, as terminal keys: REMOTE_SINGLE on
    press, REMOTE_COMBO when both are held (whichever came second), REMOTE_TAP
    when the button is released without having been used in a combo."""
    held = self.remote.pressed
    out: list[str] = []
    for b in self.remote_events:  # a fresh press starts a new tap
      self.tap_used.discard(b)
    for b in self.remote_events:
      if b in REMOTE_SINGLE:
        out.append(REMOTE_SINGLE[b])
      for (mod, btn), key in REMOTE_COMBO.items():
        if b in (mod, btn) and held.get(mod) and held.get(btn):
          self.tap_used.add(mod)
          if key not in out:
            out.append(key)
    for b, key in REMOTE_TAP.items():
      was = self.tap_held.get(b, False) or b in self.remote_events
      if was and not held.get(b) and b not in self.tap_used:
        out.append(key)
      self.tap_held[b] = bool(held.get(b))
    self.remote_events.clear()
    return out

  def apply_caps(self) -> None:
    w, c = self.walk_cap, self.crawl_cap
    self.cmd_lo = np.array([-self.walk_back, -w[1], -w[2]])
    self.cmd_hi = w.copy()
    self.ccmd_lo, self.ccmd_hi = -c, c.copy()

  def change_cap(self, sign: int) -> None:
    """L1 / L2: forward cap +-SPEED_STEP for whatever runs (crawl or walk);
    sideways, turn (and the walk's backward cap) scale with it. Never past the
    policy's trained ranges."""
    ch = self.chain
    crawl = ch.mode_name in ("CRAWL", "CRAWL_STOP", "HOLD", "DOWN")
    if crawl:
      r = ch.crawl_ranges
      top = np.array([r["fwd"][1], r["lat"][1], r["turn"][1]])
      cap = self.crawl_cap
    else:
      r = ch.walk_ranges
      top = np.array([r["vx"][1], r["vy"][1], r["wz"][1]])
      cap = self.walk_cap
    new_fwd = float(np.clip(round(cap[0] + sign * SPEED_STEP, 3), SPEED_STEP, top[0]))
    k = new_fwd / cap[0]
    cap[:] = np.minimum(np.round(cap * k, 3), top)
    if not crawl:
      self.walk_back = float(min(round(self.walk_back * k, 3), -r["vx"][0]))
    self.apply_caps()
    what = "crawl" if crawl else "walk"
    print(
      f"\n{what} speed cap: forward {cap[0]:.2f}, sideways {cap[1]:.2f}, turn "
      f"{cap[2]:.2f}" + ("" if crawl else f", back {self.walk_back:.2f}"),
      flush=True,
    )

  def thermal_guard(self, now: float, st) -> None:
    """thermal_guard.py every policy tick: damping at 85 C; on the crawl side
    (HOLD, CRAWL, CRAWL_STOP) stand up as if u was pressed at 72 C or > 1 C/s
    heating; elsewhere just warn."""
    temps = np.array([list(st.motor_state[m].temperature) for m in range(NUM_MOTORS)])
    crawl_side = self.mode == "CHAIN" and self.chain.mode_name in CRAWL_SIDE
    self.guard_action = self.guard.update(now, temps, in_crawl_side=crawl_side)
    if self.guard_action == GUARD_DAMP:
      self.estop(f"thermal guard: {self.guard.reason}")
    elif self.guard_action == GUARD_STAND_UP and not self.guard_up_sent:
      self.guard_up_sent = True
      self.chain.press_up()
      self.ccmd[:] = 0.0
      self.ccmd_goal[:] = 0.0
      print(
        f"\n\a!!! thermal guard: {self.guard.reason}, motor "
        f"{MOTOR_NAMES.get(self.guard.hottest_motor, '?')} hottest: standing up (u)",
        flush=True,
      )

  def remote_command(self) -> None:
    """Sticks: left = forward / back and sideways, right = turn (stick right
    = clockwise), scaled by the current speed caps."""
    dz = lambda v: 0.0 if abs(v) < 0.1 else v  # noqa: E731
    r = self.remote
    if self.chain.mode_name == "WALK":
      w = self.walk_cap
      fwd = w[0] if r.ly > 0 else self.walk_back
      goal = [dz(r.ly) * fwd, -dz(r.lx) * w[1], -dz(r.rx) * w[2]]
      self.cmd_goal = np.clip(np.array(goal), self.cmd_lo, self.cmd_hi)
    elif self.chain.mode_name == "CRAWL":
      c = self.crawl_cap
      # The crawl's turn input is clockwise-positive: stick right is positive.
      goal = [dz(r.ly) * c[0], -dz(r.lx) * c[1], dz(r.rx) * c[2]]
      self.ccmd_goal = np.clip(np.array(goal), self.ccmd_lo, self.ccmd_hi)

  def chain_step(self, st, q, dq, gyro) -> None:
    ch = self.chain
    # The policies were trained at exactly 50 Hz: if steps come late (a slow
    # computer), the walk goes unstable within a second. Stop first.
    now_p = time.monotonic()
    if self.last_policy_t is not None:
      self.policy_dts.append(now_p - self.last_policy_t)
    self.last_policy_t = now_p
    slow = sum(d > 1.3 * CONTROL_DT for d in self.policy_dts)
    if slow >= 5:
      hz = 1.0 / float(np.median(self.policy_dts))
      self.estop(
        f"policy steps late: {slow} of the last {len(self.policy_dts)} over "
        f"{1.3e3 * CONTROL_DT:.0f} ms (policy at {hz:.0f} Hz, needs 50); "
        "use --loop-hz 100 --poll-state"
      )
      return
    if ch.mode_name == "WALK":
      lim = np.array([0.5, 0.5, 1.0]) * CONTROL_DT  # command accel limits
      self.cmd += np.clip(self.cmd_goal - self.cmd, -lim, lim)
      ch.set_walk_command(*self.cmd)
    else:
      self.cmd[:] = 0.0
      self.cmd_goal[:] = 0.0
    if ch.mode_name == "CRAWL":
      lim = np.array([0.5, 0.25, 1.0]) * CONTROL_DT
      self.ccmd += np.clip(self.ccmd_goal - self.ccmd, -lim, lim)
      ch.set_crawl_command(*self.ccmd)
    else:
      self.ccmd[:] = 0.0
      self.ccmd_goal[:] = 0.0
    quat = np.array(st.imu_state.quaternion, dtype=np.float64)
    t0 = time.perf_counter()
    tgt = ch.step(quat, gyro, q, dq)
    self.policy_ms = 1e3 * (time.perf_counter() - t0)
    if not np.all(np.isfinite(tgt)):
      self.estop("policy output is not finite")
      return
    if ch.mode_name != self.chain_mode:
      self.chain_mode_t0 = time.monotonic()
      self.wait_warned = False
      if ch.mode_name == "WALK":
        self.guard_up_sent = False
      if getattr(ch, "hold_timed_out", False) and not self.timeout_told:
        self.timeout_told = True
        print(
          f"\n\a!!! HOLD: no crawl gate after {ch.hold_timeout_s:g} s "
          f"({ch.gate_why}): standing up",
          flush=True,
        )
      if ch.mode_name == "DOWN":
        self.timeout_told = False
        ch.hold_timed_out = False
      lim = ch.limb
      print(
        f"\n[{time.strftime('%H:%M:%S')}]   chain: {self.chain_mode} -> "
        f"{ch.mode_name}  (pelvis over limbs {lim.above_limbs:.3f} m, over soles "
        f"{lim.pelvis_height:.3f} m, limbs down "
        f"{''.join('x' if d else '.' for d in lim.down)})",
        flush=True,
      )
      self.chain_mode = ch.mode_name
    waited = time.monotonic() - self.chain_mode_t0
    if ch.mode_name in WAITING and not ch.gate_ok and waited > 5.0:
      if not self.wait_warned:
        self.wait_warned = True
        fix = (
          f"it stands up by itself at {self.args.hold_timeout_s:g} s, or u then g"
          if ch.mode_name == "HOLD"
          else "g passes it"
        )
        print(
          f"\n\a!!! {ch.mode_name} has waited 5 s: {ch.gate_why}. If the robot is "
          f"settled, {fix}; Space lies it down.",
          flush=True,
        )
    self.action = ch.active_action
    clipped = np.clip(tgt, self.soft_lo, self.soft_hi)
    self.clamped += int(np.any(clipped != tgt))
    self.target = clipped

  def check_safety(self, st, dq: np.ndarray, grav: np.ndarray) -> None:
    """Damping on orientation, joint speed or motor temperature while driven.
    In CHAIN the orientation is checked against where the current mode should
    have the pelvis (upright, the schedule's pitch, or prone)."""
    if self.mode == "DAMP":
      return
    if self.mode == "CHAIN":
      ge = self.chain.expected_gravity()
      off = math.degrees(math.acos(float(np.clip(grav @ ge, -1.0, 1.0))))
      what = f"{off:.0f} deg off the {self.chain.mode_name} orientation"
    else:
      off = self.tilt_deg(grav)
      what = f"tilt {off:.0f} deg"
    if off > self.args.max_tilt:
      self.estop(f"{what} > {self.args.max_tilt:.0f}")
    elif np.abs(dq).max() > MAX_JOINT_SPEED:
      j = int(np.abs(dq).argmax())
      self.estop(f"{JOINTS[j]} speed {dq[j]:.1f} rad/s")
    elif self.tick_n % 50 == 0:
      temps = [max(st.motor_state[m].temperature) for m in range(NUM_MOTORS)]
      if max(temps) >= TEMP_STOP_C:
        self.estop(f"motor {int(np.argmax(temps))} at {max(temps)} C")

  def record(self, now, q, dq, grav, gyro, st) -> None:
    super().record(now, q, dq, grav, gyro, st)
    ch = self.chain
    chain = self.mode == "CHAIN"
    self.log["chain_mode"].append(ch.mode if chain else -1)
    self.log["phase"].append(self.phi if self.mode == "SCHEDULE" else ch.phase)
    lim = ch.limb
    self.log["above_limbs"].append(lim.above_limbs if chain and lim else np.nan)
    self.log["pelvis_height"].append(lim.pelvis_height if chain and lim else np.nan)
    self.log["gate_ok"].append(bool(ch.gate_ok) if chain else False)
    self.log["ccmd"].append(self.ccmd.copy())
    ms = st.motor_state
    self.log["temp"].append(
      np.array([list(ms[m].temperature) for m in range(NUM_MOTORS)])
    )
    self.log["tau_motor"].append(np.array([ms[m].tau_est for m in range(NUM_MOTORS)]))
    self.log["q_motor"].append(np.array([ms[m].q for m in range(NUM_MOTORS)]))
    self.log["dq_motor"].append(np.array([ms[m].dq for m in range(NUM_MOTORS)]))
    self.log["guard"].append(self.guard_action)
    self.log["style"].append(getattr(ch, "style", "crawl"))

  def status(self) -> str:
    s = super().status()
    if self.mode == "CHAIN":
      ch = self.chain
      s += f" | {ch.mode_name} phase {ch.phase:.2f}"
      if self.policy_dts:
        s += f" pol {1.0 / float(np.median(self.policy_dts)):.0f}Hz"
      if ch.mode_name == "WALK":
        s += f" cmd {self.cmd[0]:+.2f} {self.cmd[1]:+.2f} {self.cmd[2]:+.2f}"
        s += f" cap {self.walk_cap[0]:.1f}"
      elif ch.mode_name == "CRAWL":
        s += (
          f" crawl {self.ccmd[0]:+.2f} {self.ccmd[1]:+.2f} {self.ccmd[2]:+.2f} "
          f"step {ch.crawl_cmd[3]:.2f} cap {self.crawl_cap[0]:.2f}"
        )
      elif ch.mode_name in WAITING and ch.gate_why:
        s += f" waiting: {ch.gate_why}"
      if len(getattr(ch, "crawl_styles", ())) > 1:
        s += f" [{ch.style}"
        if ch.style_pending:
          s += f" -> {ch.style_pending}: {ch.style_why or 'switching'}"
        s += "]"
      if self.clamped:
        s += f" clamp {self.clamped}"
    if self.mode == "SCHEDULE":
      s += f" | schedule phi {self.phi:.2f}"
    if self.low_state is not None:
      m, temp = hottest(self.low_state)
      s += f" | hottest m{m} {MOTOR_NAMES.get(m, '?')} {temp}C"
    if self.guard_action != "ok":
      s += f" | guard {self.guard_action}: {self.guard.reason}"
    if not self.guard.allow_down:
      s += " (no crawl until < 60 C)"
    return s


def main() -> None:
  ap = make_parser(__doc__)
  ap.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
  ap.add_argument(
    "--transition-s",
    type=float,
    default=8.0,
    help="down / up time (trained 4-8 s; 8 s on hardware first)",
  )
  ap.add_argument(
    "--dwell-s",
    type=float,
    default=1.0,
    help="crawl gate hold time before handing over",
  )
  ap.add_argument(
    "--max-crawl-fwd", type=float, default=0.5, help="cap (v3 creepy: -0.2..0.5)"
  )
  ap.add_argument("--max-crawl-lat", type=float, default=0.25)
  ap.add_argument("--max-crawl-turn", type=float, default=0.7)
  ap.add_argument(
    "--hold-timeout-s",
    type=float,
    default=8.0,
    help="HOLD without the crawl gate this long: stand up (v3 controller)",
  )
  ap.add_argument(
    "--key-step",
    type=float,
    default=0.1,
    help="w/s a/d q/e change the command by this much per press",
  )
  t = ap.add_argument_group("thermal guard (thermal_guard.py)")
  t.add_argument("--guard-warn", type=float, default=65.0, help="C: warn")
  t.add_argument(
    "--guard-stand-up", type=float, default=72.0, help="C: stand up from the crawl"
  )
  t.add_argument(
    "--guard-rate", type=float, default=1.0, help="C/s: stand up when heating faster"
  )
  t.add_argument(
    "--guard-resume", type=float, default=60.0, help="C: crawl again below this"
  )
  ap.add_argument(
    "--schedule-only",
    action="store_true",
    help="gantry check: c/u move the PD targets along the keyframes, no policy",
  )
  ap.add_argument(
    "--schedule-s", type=float, default=16.0, help="--schedule-only speed"
  )
  ap.set_defaults(soft_limit=1.0)  # clamp at the joint ranges, as in the bundle
  args = ap.parse_args()
  domain = check_args(args)

  cc, chain = load_bundle(Path(args.bundle), args)
  styles = getattr(chain, "crawl_styles", ["crawl"])
  print(
    f"bundle: {args.bundle} (down / up {args.transition_s:.0f} s, crawl styles "
    f"{', '.join(styles)})"
  )
  if args.real and not args.monitor:
    if args.schedule_only:
      extra = "  - --schedule-only: feet OFF the ground the whole time\n"
    else:
      extra = (
        "  - wrist braces AND the 100 mm forearm extensions on (the crawl stands "
        "on them)\n"
        "  - a clear, flat, non-slip floor 3 m long; the harness slack enough "
        "for the\n    robot to lie prone (pelvis 0.22 m), someone paying it out\n"
      )
    real_robot_checklist(args, extra)
  ctl = ChainRealController(args, cc, chain)
  a = args
  help_text = HELP.format(
    step=f"{a.key_step:g}",
    walk=f"forward {a.min_vx:g}..{a.max_vx:g}, sideways +-{a.max_vy:g}, "
    f"turn +-{a.max_wz:g}",
    crawl=f"forward +-{a.max_crawl_fwd:g}, sideways +-{a.max_crawl_lat:g}, "
    f"turn +-{a.max_crawl_turn:g} (and each style's own ranges)",
    cool=a.guard_resume,
    hot=a.guard_stand_up,
    rate=a.guard_rate,
    hold=a.hold_timeout_s,
    styles=" / ".join(getattr(chain, "crawl_styles", ["crawl"])),
  )
  run(args, ctl, domain, HELP_SCHEDULE if args.schedule_only else help_text)


if __name__ == "__main__":
  main()
