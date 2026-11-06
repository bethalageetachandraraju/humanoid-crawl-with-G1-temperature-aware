"""Watch the modified 29-DoF crawl G1 in the sim, with a live chart of every
motor's torque against its limit.

Runs the real environment (play settings, one robot) in MJLab's native viewer,
and next to it a matplotlib window with one bar per joint:

* 25 motors: PD torque demand ``kp (q_target - q) - kd qdot`` as % of the motor's
  peak torque -- exactly what the ``motor_torque_margin`` reward sees. Over 100%
  means the motor is saturated: it delivers its peak and the load wins.
* 4 braced wrist joints: moment across the joint about its axis as % of
  ``BRACED_TORQUE_LIMIT`` -- what ``braced_wrist_torque`` sees.

Dashed orange line = 80% (penalties start), red line = 100% (limit). Bar colour
follows the same thresholds; the black tick is the peak over the last 3 s.

The task follows the checkpoint's input size: 88 inputs = 7 commands + gyro
(``Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof``), 81 = the first 29-DoF run
(``Mjlab-Crawl-Flat-Unitree-G1-29dof``), 75 = the 23-DoF policy mapped onto this
robot (waist roll/pitch held at 0; never trained on this body, so a pessimistic
preview that stops below a 0.5 m/s command).

Heading hold (on by default, G toggles): while the turn command is 0 the robot
is steered back to the heading it had when turning stopped, turn = -1.0 x
heading error (at most 0.5 rad/s) -- the loop to run on the real robot.

``--mode hold`` sends action 0 instead: every motor holds the crawl pose.

On exit the run is saved to ``--out``: a CSV of every control step and a
summary PNG (time series plus peak / 95th percentile per joint).

Keys (MuJoCo window): W/S forward +/-0.1 m/s, Q/E sideways left/right +/-0.1
m/s, A/D turn +/-0.25 rad/s, H step height (7-command policies), G heading hold,
X stop, M hold <-> policy, Space pause, - / = playback speed, Enter reset.

Usage::

  uv run python scripts/crawl_torque_monitor.py --checkpoint logs/rsl_rl/<run>/model_N.pt
  uv run python scripts/crawl_torque_monitor.py --mode hold
  uv run python scripts/crawl_torque_monitor.py --fwd 0.3 --playback 0.25
"""

from __future__ import annotations

import argparse
import collections
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.g1_crawl import mdp
from mjlab.tasks.g1_crawl.config.g1.env_cfgs import BRACED_AXES, BRACED_TORQUE_SENSORS
from mjlab.tasks.g1_crawl.mdp import CrawlVelocityCommand
from mjlab.tasks.g1_crawl.robot import CRAWL_JOINTS
from mjlab.tasks.g1_crawl.robot29 import (
  BRACED_LOADS_29,
  BRACED_TORQUE_LIMIT,
  CRAWL_JOINTS_29,
  MOTORS_29,
)
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import (
  KEY_A,
  KEY_D,
  KEY_E,
  KEY_G,
  KEY_H,
  KEY_M,
  KEY_Q,
  KEY_S,
  KEY_W,
  KEY_X,
)

# Actor input size -> (task, description).
TASKS_BY_INPUTS = {
  88: ("Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof", "29-DoF, 7 commands + gyro"),
  81: ("Mjlab-Crawl-Flat-Unitree-G1-29dof", "29-DoF, 3 commands"),
  75: (
    "Mjlab-Crawl-Flat-Unitree-G1-29dof",
    "23-DoF policy mapped onto this robot (waist roll/pitch held at 0)",
  ),
}
STEP_HEIGHTS = (0.03, 0.06, 0.10, 0.14)
HEADING_GAIN, HEADING_MAX_TURN = 1.0, 0.5
# Row layout: t, policy?, cmd fwd/lat/turn/step, actual fwd/lat, then joints.
HEAD = 8
DEFAULT_CHECKPOINT = (
  "logs/rsl_rl/g1_crawl_29dof/2026-10-10_18-26-14_braced_flat_elegant/model_7999.pt"
)
SOFT, LIMIT = 0.8, 1.0
GREEN, ORANGE, RED = "#2e9e4f", "#f0a020", "#d33a2c"
PLOT_HZ = 8.0
PEAK_WINDOW_S = 3.0


def _label(joint: str) -> str:
  side = {"left": "L ", "right": "R "}.get(joint.split("_")[0], "")
  body = joint.removeprefix("left_").removeprefix("right_").removesuffix("_joint")
  return side + body.replace("_", " ")


class _Actor(torch.nn.Module):
  """RSL-RL MLP actor rebuilt from a checkpoint: Linear/ELU stack, no obs norm."""

  def __init__(self, actor_state_dict: dict[str, torch.Tensor]) -> None:
    super().__init__()
    mlp = {k.removeprefix("mlp."): v for k, v in actor_state_dict.items()}
    linear = sorted(int(k.split(".")[0]) for k in mlp if k.endswith(".weight"))
    layers: list[torch.nn.Module] = []
    for index in linear:  # RSL-RL numbers Linear layers 0, 2, 4, ... with ELUs between.
      out_dim, in_dim = mlp[f"{index}.weight"].shape
      layers += [torch.nn.Linear(in_dim, out_dim), torch.nn.ELU()]
    self.mlp = torch.nn.Sequential(*layers[:-1])
    self.mlp.load_state_dict({k: v for k, v in mlp.items() if k[0].isdigit()})
    self.in_dim = mlp[f"{linear[0]}.weight"].shape[1]

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.mlp(obs)


class Driver:
  """Policy callable for the viewer: drives the robot and records the loads."""

  def __init__(
    self, env: ManagerBasedRlEnv, actor: _Actor, about: str, mode: str
  ) -> None:
    self.env = env
    self.robot = env.scene["robot"]
    self.device = env.device
    self.mode = mode
    self.cmd = {"fwd": 0.0, "lat": 0.0, "turn": 0.0, "step": STEP_HEIGHTS[1]}
    self.heading_hold = True
    self.turn_applied = 0.0
    self.num_actions = env.action_manager.total_action_dim

    # Keyboard owns the command (same takeover as play_teleop.py).
    term = env.command_manager.get_term("base_velocity")
    assert isinstance(term, CrawlVelocityCommand)
    term.manual_control()
    self.command_term = term
    self.num_cmd = term.vel_command_b.shape[1]
    # Only policies trained with a lateral command get one.
    self.lateral = tuple(term.cfg.lin_vel_y) != (0.0, 0.0)
    self.heading_target: float | None = None

    self.actor = actor.to(self.device)
    env_obs_dim = env.observation_manager.group_obs_dim["actor"][0]
    self.legacy = self.actor.in_dim != env_obs_dim
    if self.legacy:
      ids, _ = self.robot.find_joints(CRAWL_JOINTS, preserve_order=True)
      self.ids23 = torch.tensor(ids, device=self.device)
      self.cols23 = torch.tensor(
        [list(CRAWL_JOINTS_29).index(n) for n in CRAWL_JOINTS], device=self.device
      )
      self.last23 = torch.zeros(1, len(CRAWL_JOINTS), device=self.device)
    print(f"[monitor] policy: {about}")

    crawl = env.scene["crawl_contacts"]
    self.limb_names = list(crawl.primary_names)
    bodies = env.scene["body_ground_contacts"]
    self.torso_idx = list(bodies.primary_names).index("torso_link")
    self.log: list[np.ndarray] = []
    self.on_step = None  # Set by the panel.

  def _turn(self) -> float:
    """The operator's turn rate, or heading hold while it is zero."""
    heading = float(self.command_term.heading()[0])
    if not self.heading_hold or self.cmd["turn"] != 0.0:
      self.heading_target = heading
      return self.cmd["turn"]
    if self.heading_target is None:
      self.heading_target = heading
    error = (self.heading_target - heading + np.pi) % (2 * np.pi) - np.pi
    # Positive turn commands are clockwise seen from above.
    return float(np.clip(-HEADING_GAIN * error, -HEADING_MAX_TURN, HEADING_MAX_TURN))

  def __call__(self, obs) -> torch.Tensor:
    self.turn_applied = self._turn()
    command = self.command_term.vel_command_b
    command[:, 0] = self.cmd["fwd"]
    command[:, 1] = self.cmd["lat"] if self.lateral else 0.0
    command[:, 2] = self.turn_applied
    if self.num_cmd > 3:
      command[:, 3] = self.cmd["step"]
    if self.mode == "hold":
      action = torch.zeros(1, self.num_actions, device=self.device)
    elif self.legacy:
      action = self._legacy_action()
    else:
      action = self.actor(obs["actor"]).clamp(-5.0, 5.0)
    self._record()
    return action

  def _legacy_action(self) -> torch.Tensor:
    data = self.robot.data
    command = torch.tensor(
      [[self.cmd["fwd"], 0.0, self.turn_applied]], device=self.device
    )
    obs = torch.cat(
      [
        data.projected_gravity_b,
        command,
        data.joint_pos[:, self.ids23] - data.default_joint_pos[:, self.ids23],
        data.joint_vel[:, self.ids23],
        self.last23,
      ],
      dim=1,
    )
    a23 = self.actor(obs).clamp(-5.0, 5.0)
    self.last23 = a23
    action = torch.zeros(1, self.num_actions, device=self.device)
    action[:, self.cols23] = a23
    return action

  def _record(self) -> None:
    env = self.env
    ratio = mdp.motor_torque_ratio(env, MOTORS_29)[0].cpu().numpy()
    braced = mdp.braced_joint_moment(env, BRACED_TORQUE_SENSORS, BRACED_AXES)
    braced = braced[0].cpu().numpy()
    limbs = env.scene["crawl_contacts"].data.force[0].norm(dim=-1).cpu().numpy() > 1.0
    torso = (
      env.scene["body_ground_contacts"].data.force[0, self.torso_idx].norm().item()
      > 1.0
    )
    vel = self.robot.data.root_link_lin_vel_b[0]
    row = np.concatenate(
      [
        [env.episode_length_buf[0].item() * env.step_dt],
        [1.0 if self.mode == "policy" else 0.0],
        [self.cmd["fwd"], self.cmd["lat"], self.turn_applied, self.cmd["step"]],
        [vel[2].item(), vel[1].item()],
        ratio,
        braced / BRACED_TORQUE_LIMIT,
        limbs.astype(float),
        [float(torso)],
      ]
    )
    self.log.append(row)
    if self.on_step is not None:
      self.on_step(row)


class TorquePanel:
  """Live horizontal bar chart: % of limit per joint, with a peak-hold tick."""

  def __init__(self, driver: Driver, speed_getter) -> None:
    self.driver = driver
    self.speed_getter = speed_getter
    self.names = list(MOTORS_29) + list(BRACED_LOADS_29)
    self.limits = [MOTORS_29[n][2] for n in MOTORS_29] + [BRACED_TORQUE_LIMIT] * len(
      BRACED_LOADS_29
    )
    labels = [_label(n) for n in MOTORS_29] + [
      _label(n) + " (braced)" for n in BRACED_LOADS_29
    ]
    n = len(self.names)
    self.peaks: collections.deque[np.ndarray] = collections.deque(
      maxlen=int(PEAK_WINDOW_S * 50)
    )
    self.last_draw = 0.0
    self.alive = True

    plt.ion()
    self.fig, self.ax = plt.subplots(figsize=(9.5, 10.5))
    self.fig.canvas.manager.set_window_title("Crawl torque monitor")  # type: ignore[union-attr]
    y = np.arange(n)[::-1]
    self.y = y
    self.bars = self.ax.barh(y, np.zeros(n), color=GREEN, height=0.72)
    (self.peak_marks,) = self.ax.plot(
      np.zeros(n), y, "|", color="black", markersize=13, mew=2
    )
    self.texts = [
      self.ax.text(151, yi, "", va="center", ha="left", fontsize=8, family="monospace")
      for yi in y
    ]
    self.ax.set_yticks(y, labels, fontsize=8.5)
    self.ax.set_xlim(0, 150)
    self.ax.axvline(SOFT * 100, color=ORANGE, ls="--", lw=1.2)
    self.ax.axvline(LIMIT * 100, color=RED, lw=1.5)
    self.ax.set_xlabel(
      "% of limit  (motors: PD torque demand / peak;  braced: moment / limit)"
    )
    # Group separators: legs | waist | arms | braced.
    for boundary in (12, 15, 25):
      self.ax.axhline(n - boundary - 0.5, color="0.75", lw=0.8)
    self.title = self.ax.set_title("", fontsize=9, loc="left", family="monospace")
    self.fig.subplots_adjust(left=0.22, right=0.80, top=0.93, bottom=0.06)
    self.fig.canvas.mpl_connect("close_event", self._on_close)
    self.fig.show()
    driver.on_step = self.update

  def _on_close(self, _event) -> None:
    self.alive = False

  def update(self, row: np.ndarray) -> None:
    values = row[HEAD : HEAD + len(self.names)]
    self.peaks.append(values)
    now = time.perf_counter()
    if not self.alive or now - self.last_draw < 1.0 / PLOT_HZ:
      return
    self.last_draw = now
    peak = np.max(np.stack(self.peaks), axis=0)
    for bar, text, v, pk, lim in zip(
      self.bars, self.texts, values, peak, self.limits, strict=True
    ):
      bar.set_width(min(v, 1.5) * 100)
      bar.set_color(GREEN if v < SOFT else ORANGE if v < LIMIT else RED)
      text.set_text(f"{v * lim:5.1f}/{lim:3.0f} Nm  pk {pk * 100:3.0f}%")
    self.peak_marks.set_xdata(np.minimum(peak, 1.5) * 100)
    t, policy, fwd, lat, turn, step, vel, vel_lat = row[:HEAD]
    k = HEAD + len(self.names)
    limbs = row[k : k + 4]
    torso = row[k + 4] > 0.5
    ground = "  ".join(
      f"{_short(n)}{'+' if c else '-'}"
      for n, c in zip(self.driver.limb_names, limbs, strict=True)
    )
    worst = int(np.argmax(values))
    d = self.driver
    gait = f"  step {step * 100:.0f} cm" if d.num_cmd > 3 else ""
    hold = "  [heading hold]" if d.heading_hold and d.cmd["turn"] == 0.0 else ""
    self.title.set_text(
      f"t {t:5.1f} s   x{self.speed_getter():g} speed   mode "
      f"{'policy' if policy else 'hold'}   cmd fwd {fwd:+.1f}  lat {lat:+.1f} m/s"
      f"  turn {turn:+.2f}{hold}{gait}\n"
      f"actual fwd {vel:+.2f}  lat {vel_lat:+.2f} m/s   "
      f"on ground: {ground}   torso {'TOUCHING' if torso else 'clear'}   "
      f"highest: {_label(self.names[worst])} {values[worst] * 100:.0f}%"
    )
    self.fig.canvas.draw_idle()
    self.fig.canvas.flush_events()


def _short(body: str) -> str:
  side = "L" if body.startswith("left") else "R"
  return side + ("foot" if "ankle" in body else "hand")


def save_run(driver: Driver, out: Path) -> None:
  if not driver.log:
    return
  out.mkdir(parents=True, exist_ok=True)
  data = np.stack(driver.log)
  names = list(MOTORS_29) + list(BRACED_LOADS_29)
  limits = np.array(
    [MOTORS_29[n][2] for n in MOTORS_29] + [BRACED_TORQUE_LIMIT] * len(BRACED_LOADS_29)
  )
  k = HEAD + len(names)
  header = ["t", "policy_mode", "cmd_fwd", "cmd_lat", "cmd_turn", "cmd_step_height"]
  header += ["vel_fwd", "vel_lat"]
  header += [f"{n}_nm" for n in names] + [f"{n}_ground" for n in driver.limb_names]
  header += ["torso_ground"]
  table = data.copy()
  table[:, HEAD:k] *= limits  # Ratios -> Nm.
  np.savetxt(
    out / "torques.csv", table, delimiter=",", header=",".join(header), comments=""
  )

  step = np.arange(len(data)) * 0.02
  ratios_all = data[:, HEAD:k] * 100
  # Statistics skip the first 0.2 s after every reset: the robot is placed in an
  # animation frame and snaps to the PD targets, which is not a gait load.
  settled = data[:, 0] >= 0.2
  ratios = ratios_all[settled] if settled.any() else ratios_all
  fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 10), height_ratios=(1, 1.2))
  groups = [(0, 12, "tab:blue", "legs"), (12, 15, "tab:purple", "waist"),
            (15, 25, "tab:green", "arms"), (25, 29, "tab:red", "braced wrist")]  # fmt: skip
  for a, b, colour, name in groups:
    for j in range(a, b):
      top.plot(step, ratios_all[:, j], color=colour, lw=0.7, alpha=0.8,
               label=name if j == a else None)  # fmt: skip
  top.axhline(SOFT * 100, color=ORANGE, ls="--")
  top.axhline(LIMIT * 100, color=RED)
  top.set_ylabel("% of limit")
  top.set_xlabel("sim time (s)")
  top.legend(loc="upper right")
  top.set_title("Torque demand vs limit, every joint")
  x = np.arange(len(names))
  p95 = np.percentile(ratios, 95, axis=0)
  peak = ratios.max(axis=0)
  bottom.bar(x - 0.2, p95, 0.4, label="95th percentile", color="tab:blue")
  bottom.bar(x + 0.2, peak, 0.4, label="peak", color="tab:orange")
  bottom.axhline(SOFT * 100, color=ORANGE, ls="--")
  bottom.axhline(LIMIT * 100, color=RED)
  bottom.set_xticks(x, [_label(n) for n in names], rotation=70, ha="right", fontsize=8)
  bottom.set_ylabel("% of limit")
  bottom.legend()
  fig.tight_layout()
  fig.savefig(out / "summary.png", dpi=120)
  plt.close(fig)

  print(
    f"\n[monitor] {len(data)} control steps ({len(data) * 0.02:.0f} s) saved to {out}"
  )
  print(f"  {'joint':<30}{'limit':>7}{'p95 %':>8}{'peak %':>8}{'>80%':>8}{'>100%':>8}")
  for j in np.argsort(-peak)[:12]:
    r = ratios[:, j]
    print(
      f"  {_label(names[j]):<30}{limits[j]:6.0f}N{p95[j]:8.0f}{peak[j]:8.0f}"
      f"{(r > 80).mean() * 100:7.1f}%{(r > 100).mean() * 100:7.1f}%"
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mode", choices=("policy", "hold"), default="policy")
  parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
  parser.add_argument("--fwd", type=float, default=0.5, help="Initial forward m/s.")
  parser.add_argument("--turn", type=float, default=0.0, help="Initial turn rad/s.")
  parser.add_argument(
    "--playback", type=float, default=0.5, help="Sim speed vs real time (1/32..8)."
  )
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("logs/torque_monitor") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
  )
  args = parser.parse_args()

  ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  actor = _Actor(ckpt["actor_state_dict"])
  if actor.in_dim not in TASKS_BY_INPUTS:
    raise SystemExit(f"{args.checkpoint}: no crawl task has {actor.in_dim} inputs")
  task, about = TASKS_BY_INPUTS[actor.in_dim]
  print(f"[monitor] {args.checkpoint.name}: {actor.in_dim} inputs -> {task}")
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=5.0)
  driver = Driver(env, actor, about, args.mode)
  driver.cmd.update(fwd=args.fwd, turn=args.turn)

  def on_key(key: int) -> None:
    if key == KEY_W:
      driver.cmd["fwd"] = round(min(2.0, driver.cmd["fwd"] + 0.1), 2)
    elif key == KEY_S:
      driver.cmd["fwd"] = round(max(-1.0, driver.cmd["fwd"] - 0.1), 2)
    elif key == KEY_A:
      driver.cmd["turn"] = round(min(2.0, driver.cmd["turn"] + 0.25), 2)
    elif key == KEY_D:
      driver.cmd["turn"] = round(max(-2.0, driver.cmd["turn"] - 0.25), 2)
    elif key == KEY_Q:
      driver.cmd["lat"] = round(min(0.5, driver.cmd["lat"] + 0.1), 2)
    elif key == KEY_E:
      driver.cmd["lat"] = round(max(-0.5, driver.cmd["lat"] - 0.1), 2)
    elif key == KEY_H:
      i = STEP_HEIGHTS.index(driver.cmd["step"])
      driver.cmd["step"] = STEP_HEIGHTS[(i + 1) % len(STEP_HEIGHTS)]
    elif key == KEY_G:
      driver.heading_hold = not driver.heading_hold
      driver.heading_target = None
    elif key == KEY_X:
      driver.cmd.update(fwd=0.0, lat=0.0, turn=0.0)
    elif key == KEY_M:
      driver.mode = "hold" if driver.mode == "policy" else "policy"
    print(
      f"\r[monitor] mode {driver.mode:<6} fwd {driver.cmd['fwd']:+.1f} "
      f"lat {driver.cmd['lat']:+.1f} m/s  turn {driver.cmd['turn']:+.2f} rad/s  "
      f"step {driver.cmd['step'] * 100:.0f} cm  heading hold "
      f"{'on' if driver.heading_hold else 'off'}   ",
      end="",
      flush=True,
    )

  viewer = NativeMujocoViewer(wrapped, driver, key_callback=on_key)
  multipliers = viewer.SPEED_MULTIPLIERS
  target = min(multipliers, key=lambda m: abs(m - args.playback))
  while viewer._time_multiplier > target:
    viewer.decrease_speed()
  while viewer._time_multiplier < target:
    viewer.increase_speed()
  TorquePanel(driver, lambda: viewer._time_multiplier)
  print(
    "[monitor] keys in the MuJoCo window: W/S fwd +/-0.1, Q/E sideways +/-0.1, "
    "A/D turn +/-0.25, H step height, G heading hold, X stop, M hold<->policy, "
    "Space pause, -/= speed, Enter reset"
  )
  try:
    viewer.run()
  finally:
    save_run(driver, args.out)
    env.close()


if __name__ == "__main__":
  main()
