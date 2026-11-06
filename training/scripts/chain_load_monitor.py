"""Brace and wrist loads through the full loop, live and summarized.

Runs ``play_chain.py``'s state machine and policies unchanged: walk ->
stand-to-crawl -> crawl -> crawl-to-stand -> walk. Next to the MuJoCo window, a
chart shows the last 10 s for each arm:

1. nose tip on the ground (N): total force, and the part across the nose (it
   bends the nose; the rest pushes along it);
2. force through the wrist (N), at the welded wrist-pitch joint;
3. braced wrist moments (Nm) at the welded pitch and yaw joints. The brace has
   to carry these; the 4010 motors behind them are rated 5 Nm;
4. wrist-roll motor torque (Nm); its peak is 25 Nm.

Nose forces are the peak of the four physics steps in each 20 ms control step.
On exit (close the MuJoCo window, or the end of ``--auto --headless``) the
terminal gets a table per phase, and ``--out`` gets a CSV of the run, the table
and a PNG of the whole run.

Keys as in play_chain.py: C down to crawl, U up to stand; W/S, Q/E, A/D speed,
sideways and turn in both walk and crawl; H crawl step height; X stop; P push;
F freeze; Space pause; Enter reset.

  uv run python scripts/chain_load_monitor.py                 # you drive
  uv run python scripts/chain_load_monitor.py --auto          # scripted loop
  uv run python scripts/chain_load_monitor.py --auto --headless --num-envs 64
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_chain import (  # noqa: E402
  CRAWL,
  CRAWL_STOP,
  DOWN,
  HOLD,
  STAND_HOLD,
  UP,
  WALK,
  ChainSwitcher,
  add_chain_args,
  chain_key_handler,
  make_chain_cfg,
  resolve_checkpoints,
)

import mjlab.tasks  # noqa: E402, F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.sensor import BuiltinSensorCfg, ObjRef  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import (  # noqa: E402
  BRACED_LOADS_29,
  BRACED_TORQUE_LIMIT,
  MOTORS_29,
)
from mjlab.utils.lab_api.math import quat_apply_inverse  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import NativeMujocoViewer  # noqa: E402

SIDES = ("left", "right")
ROLL_PEAK = MOTORS_29["left_wrist_roll_joint"][2]
PHASES = ("walking", "stand->crawl", "crawling", "crawl->stand")
PHASE_COLORS = ("#d9e8f5", "#fde2c4", "#d8efd3", "#ead9f2")
# Columns of one record, per robot.
QUANTITIES = (
  ("nose_L", "nose tip force, left", "N"),
  ("nose_R", "nose tip force, right", "N"),
  ("side_L", "  across the nose, left", "N"),
  ("side_R", "  across the nose, right", "N"),
  ("wrist_L", "force through the wrist, left", "N"),
  ("wrist_R", "force through the wrist, right", "N"),
  ("pitch_L", "braced wrist pitch moment, left", "Nm"),
  ("yaw_L", "braced wrist yaw moment, left", "Nm"),
  ("pitch_R", "braced wrist pitch moment, right", "Nm"),
  ("yaw_R", "braced wrist yaw moment, right", "Nm"),
  ("roll_L", "wrist-roll motor torque, left", "Nm"),
  ("roll_R", "wrist-roll motor torque, right", "Nm"),
)
COL = {key: i for i, (key, _, _) in enumerate(QUANTITIES)}
WINDOW_S = 10.0
PLOT_HZ = 10.0


def make_cfg(n: int, args):
  cfg = make_chain_cfg(n, args)
  sensors = []
  for s in cfg.scene.sensors:
    if s.name == "limb_contacts":
      s.history_length = 4  # = decimation: every physics step.
    sensors.append(s)
  for side in SIDES:
    site, _ = BRACED_LOADS_29[f"{side}_wrist_pitch_joint"]
    sensors.append(
      BuiltinSensorCfg(
        name=f"{side}_wrist_force",
        sensor_type="force",
        obj=ObjRef(type="site", name=site, entity="robot"),
      )
    )
  cfg.scene.sensors = tuple(sensors)
  return cfg


class Recorder:
  """Loads of every robot at every control step, plus each robot's phase."""

  def __init__(self, sw: ChainSwitcher) -> None:
    self.sw, env = sw, sw.env
    robot = sw.robot
    self.contacts = env.scene["limb_contacts"]
    names = list(self.contacts.primary_names)
    self.hand_ids = [names.index(f"{s}_forearm_extension") for s in SIDES]
    self.roll_body = [robot.find_bodies([f"{s}_wrist_roll_link"])[0][0] for s in SIDES]
    self.wrist_force = [env.scene[f"robot/{s}_wrist_force"] for s in SIDES]
    self.moments = [
      (
        env.scene[f"robot/{s}_wrist_{a}_torque"],
        torch.tensor(BRACED_LOADS_29[f"{s}_wrist_{a}_joint"][1], device=sw.device),
      )
      for s in SIDES
      for a in ("pitch", "yaw")
    ]
    acts = list(robot.actuator_names)
    self.roll_act = [acts.index(f"{s}_wrist_roll_joint") for s in SIDES]
    self.rows: list[torch.Tensor] = []
    self.phases: list[torch.Tensor] = []
    self.time: list[float] = []

  def phase(self) -> torch.Tensor:
    m, sw = self.sw.mode, self.sw
    up = sw.queued == UP
    p = torch.zeros_like(m)
    p[(m == DOWN) | ((m == HOLD) & ~up)] = 1
    p[(m == CRAWL) | (m == CRAWL_STOP)] = 2
    p[(m == UP) | (m == STAND_HOLD) | ((m == HOLD) & up)] = 3
    return p

  def record(self) -> torch.Tensor:
    d = self.sw.robot.data
    hist = self.contacts.data.force_history  # [n, limbs, substeps, 3], world.
    assert hist is not None
    nose, side, wrist = [], [], []
    for k in range(2):
      q = d.body_link_quat_w[:, self.roll_body[k]]  # Brace frame: +x along it.
      f = hist[:, self.hand_ids[k]]
      h = f.shape[1]
      fb = quat_apply_inverse(q.repeat_interleave(h, 0), f.reshape(-1, 3))
      fb = fb.reshape(-1, h, 3)
      nose.append(fb.norm(dim=-1).amax(dim=1))
      side.append(fb[..., 1:].norm(dim=-1).amax(dim=1))
      wrist.append(self.wrist_force[k].data.norm(dim=-1))
    moments = [(s.data @ a).abs() for s, a in self.moments]
    roll = d.actuator_force[:, self.roll_act].abs()
    row = torch.stack(
      [*nose, *side, *wrist, *moments, roll[:, 0], roll[:, 1]], dim=1
    )  # Same order as QUANTITIES.
    self.rows.append(row)
    self.phases.append(self.phase())
    self.time.append(float(self.sw.time[0]))
    return row

  def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = torch.stack(self.rows).cpu().numpy()  # [steps, n, quantities]
    phases = torch.stack(self.phases).cpu().numpy()  # [steps, n]
    return rows, phases, np.array(self.time)


class LivePlot:
  """The last WINDOW_S seconds of robot 0, both arms."""

  def __init__(self, dt: float) -> None:
    import matplotlib.pyplot as plt

    self.plt = plt
    n = int(WINDOW_S / dt)
    self.t: collections.deque[float] = collections.deque(maxlen=n)
    self.rows: collections.deque[np.ndarray] = collections.deque(maxlen=n)
    self.phase: collections.deque[int] = collections.deque(maxlen=n)
    self.last = 0.0
    self.alive = True
    plt.ion()
    self.fig, axs = plt.subplots(
      5,
      1,
      figsize=(9, 10.5),
      sharex=True,
      gridspec_kw={"height_ratios": [3, 3, 3, 3, 0.5]},
    )
    self.fig.canvas.manager.set_window_title("Brace and wrist loads")  # type: ignore[union-attr]
    self.axs = axs
    spec = (
      ("nose tip force (N)", ("nose_L", "nose_R", "side_L", "side_R"), None),
      ("force through the wrist (N)", ("wrist_L", "wrist_R"), None),
      (
        "braced wrist moments (Nm)",
        ("pitch_L", "yaw_L", "pitch_R", "yaw_R"),
        BRACED_TORQUE_LIMIT,
      ),
      ("wrist-roll motor torque (Nm)", ("roll_L", "roll_R"), ROLL_PEAK),
    )
    style = {"L": "#1f5fa8", "R": "#c8781e"}
    self.lines = []
    for ax, (title, keys, limit) in zip(axs[:4], spec, strict=True):
      for key in keys:
        dashed = key.startswith(("side", "yaw"))
        label = {
          "nose": "total",
          "side": "across",
          "wrist": "",
          "pitch": "pitch",
          "yaw": "yaw",
          "roll": "",
        }[key.split("_")[0]]
        (line,) = ax.plot(
          [],
          [],
          color=style[key[-1]],
          ls="--" if dashed else "-",
          lw=1.2,
          label=f"{'left' if key[-1] == 'L' else 'right'} {label}".strip(),
        )
        self.lines.append((line, COL[key]))
      if limit is not None:
        ax.axhline(limit, color="red", lw=1, ls=":")
      ax.set_ylabel(title, fontsize=9)
      ax.legend(loc="upper left", fontsize=7, ncol=4)
      ax.grid(alpha=0.25)
    self.texts = [
      ax.text(0.99, 0.95, "", transform=ax.transAxes, ha="right", va="top", fontsize=8)
      for ax in axs[:4]
    ]
    axs[4].set_yticks([])
    axs[4].set_xlabel("time (s)")
    self.band = None
    self.title = axs[0].set_title("", fontsize=10, loc="left")
    self.fig.tight_layout()
    self.fig.canvas.mpl_connect("close_event", self._closed)
    self.fig.show()

  def _closed(self, _event) -> None:
    self.alive = False

  def update(self, t: float, row: np.ndarray, phase: int) -> None:
    self.t.append(t)
    self.rows.append(row)
    self.phase.append(phase)
    now = time.perf_counter()
    if not self.alive or now - self.last < 1.0 / PLOT_HZ or len(self.t) < 2:
      return
    self.last = now
    t_arr = np.array(self.t)
    data = np.stack(self.rows)
    for line, col in self.lines:
      line.set_data(t_arr, data[:, col])
    groups = (
      (0, 1, 2, 3),
      (4, 5),
      (6, 7, 8, 9),
      (10, 11),
    )
    for ax, text, cols in zip(self.axs[:4], self.texts, groups, strict=True):
      top = max(float(data[:, list(cols)].max()) * 1.15, 1.0)
      ax.set_ylim(0, top)
      peaks = "  ".join(f"{QUANTITIES[c][0]} {data[:, c].max():.0f}" for c in cols)
      text.set_text(f"10 s peaks: {peaks}")
    self.axs[0].set_xlim(t_arr[0], max(t_arr[-1], t_arr[0] + WINDOW_S))
    if self.band is not None:
      self.band.remove()
    from matplotlib.colors import ListedColormap

    self.band = self.axs[4].imshow(
      np.array(self.phase)[None],
      aspect="auto",
      cmap=ListedColormap(PHASE_COLORS),
      vmin=0,
      vmax=3,
      extent=(t_arr[0], t_arr[-1], 0, 1),
    )
    self.title.set_text(
      f"t {t:5.1f} s   phase: {PHASES[phase]}   "
      f"nose L {row[0]:4.0f} N  R {row[1]:4.0f} N"
    )
    self.fig.canvas.draw_idle()
    self.fig.canvas.flush_events()


class AutoScript:
  """Walk 0.5 m/s 3 s, C, down, crawl 0.5 m/s for --crawl-s, U, up, walk 3 s."""

  def __init__(self, sw: ChainSwitcher, crawl_s: float) -> None:
    self.sw, self.crawl_s = sw, crawl_s
    self.walked_again = torch.zeros(sw.n, device=sw.device)

  def __call__(self) -> None:
    sw = self.sw
    t, m = float(sw.time[0]), sw.mode
    if 1.0 <= t < 4.0:
      sw.walk.command[m == WALK, 0] = 0.5
    if abs(t - 4.0) < 0.5 * sw.dt:
      sw.press_c()
    crawling = m == CRAWL
    since = sw.time - sw.mode_t0
    sw.crawl.command[crawling, 0] = torch.where(since[crawling] > 1.0, 0.5, 0.0)
    up_now = crawling & (since > self.crawl_s)
    if up_now.any():
      sw.crawl.command[up_now, :3] = 0.0
      sw._set(up_now, CRAWL_STOP, blend=False)
    again = (m == WALK) & sw.stats["walking_again"]
    sw.walk.command[again, 0] = 0.5
    self.walked_again += again.float() * sw.dt

  def done(self) -> bool:
    return bool((self.walked_again >= 3.0).all())


def summarize(rec: Recorder, out: Path, label: str) -> str:
  rows, phases, t = rec.arrays()
  n = rows.shape[1]
  lines = [
    f"Brace and wrist loads, {label}: {n} robot(s), {rows.shape[0]} control steps "
    f"({rows.shape[0] * rec.sw.dt:.0f} s each)",
    "Nose forces count only steps with that tip on the ground. Braced moments: the"
    f" 4010 motors behind the brace are rated {BRACED_TORQUE_LIMIT:.0f} Nm; "
    f"wrist-roll peak {ROLL_PEAK:.0f} Nm.",
  ]
  both = (
    ("nose tip force", ("nose_L", "nose_R"), "N"),
    ("  across the nose", ("side_L", "side_R"), "N"),
    ("force through the wrist", ("wrist_L", "wrist_R"), "N"),
    ("braced wrist pitch moment", ("pitch_L", "pitch_R"), "Nm"),
    ("braced wrist yaw moment", ("yaw_L", "yaw_R"), "Nm"),
    ("wrist-roll motor torque", ("roll_L", "roll_R"), "Nm"),
  )
  head = f"{'':28s}{'median':>9s}{'p95':>9s}{'p99':>9s}{'max':>9s}"
  for p, name in enumerate(PHASES):
    mask = phases == p
    if not mask.any():
      continue
    lines.append(f"\n{name} ({mask.sum() * rec.sw.dt / n:.1f} s per robot)")
    lines.append(head)
    for title, keys, unit in both:
      vals = []
      for key in keys:
        v = rows[..., COL[key]][mask]
        if key.startswith(("nose", "side")):
          v = v[rows[..., COL["nose_" + key[-1]]][mask] > 1.0]
        vals.append(v)
      v = np.concatenate(vals)
      if v.size == 0:
        lines.append(f"{title:28s}  (never on the ground)")
        continue
      q = np.percentile(v, (50, 95, 99))
      cells = "".join(f"{x:9.1f}" for x in (*q, v.max()))
      extra = ""
      if "braced" in title:
        extra = f"   over {BRACED_TORQUE_LIMIT:.0f} Nm {100 * (v > BRACED_TORQUE_LIMIT).mean():.1f}%"
      lines.append(f"{title:28s}{cells} {unit}{extra}")
  text = "\n".join(lines)
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.txt").write_text(text + "\n")
  header = "t," + ",".join(k for k, _, _ in QUANTITIES) + ",phase"
  np.savetxt(
    out / "robot0.csv",
    np.column_stack([t, rows[:, 0], phases[:, 0]]),
    delimiter=",",
    header=header,
    comments="",
    fmt="%.3f",
  )
  plot_run(rows[:, 0], phases[:, 0], t, out / "loads.png", label)
  return text


def plot_run(rows: np.ndarray, phases: np.ndarray, t: np.ndarray, path, label) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
  spec = (
    ("nose tip force (N)", ("nose_L", "nose_R", "side_L", "side_R"), None),
    ("force through the wrist (N)", ("wrist_L", "wrist_R"), None),
    ("braced wrist moments (Nm)", ("pitch_L", "yaw_L", "pitch_R", "yaw_R"), 5.0),
    ("wrist-roll motor torque (Nm)", ("roll_L", "roll_R"), ROLL_PEAK),
  )
  for ax, (title, keys, limit) in zip(axs, spec, strict=True):
    starts = np.flatnonzero(np.diff(phases, prepend=-1))
    for i, s in enumerate(starts):
      e = starts[i + 1] if i + 1 < len(starts) else len(t) - 1
      ax.axvspan(t[s], t[e], color=PHASE_COLORS[phases[s]], lw=0)
    for key in keys:
      ax.plot(
        t,
        rows[:, COL[key]],
        color="#1f5fa8" if key[-1] == "L" else "#c8781e",
        ls="--" if key.startswith(("side", "yaw")) else "-",
        lw=0.9,
        label=QUANTITIES[COL[key]][1].strip(),
      )
    if limit is not None:
      ax.axhline(limit, color="red", ls=":", lw=1)
    ax.set_ylabel(title)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
  handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in PHASE_COLORS]
  axs[0].legend(handles, PHASES, loc="upper right", fontsize=8, ncol=4, title="phase")
  axs[0].set_title(f"Robot 0, {label}")
  axs[-1].set_xlabel("time (s)")
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  add_chain_args(ap)
  ap.add_argument("--auto", action="store_true", help="run the scripted loop")
  ap.add_argument("--headless", action="store_true", help="no windows (--auto)")
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--crawl-s", type=float, default=8.0)
  ap.add_argument(
    "--out",
    type=Path,
    default=Path("logs/chain_loads") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
  )
  args = ap.parse_args()
  if args.headless and not args.auto:
    ap.error("--headless needs --auto")
  resolve_checkpoints(args)
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=make_cfg(args.num_envs, args), device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  sw = ChainSwitcher(env, args, device)
  rec = Recorder(sw)
  script = AutoScript(sw, args.crawl_s) if args.auto else None
  label = "hard" if args.hard else ("randomized" if args.robust else "nominal")

  if args.headless:
    obs = wenv.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    sw.reset()
    limit = int(80.0 / sw.dt)
    with torch.inference_mode():
      for _ in range(limit):
        assert script is not None
        script()
        obs = wenv.step(sw(obs))[0]
        rec.record()
        if script.done():
          break
      else:
        print(f"stopped after {limit * sw.dt:.0f} s; not every robot walked again")
  else:
    plot = LivePlot(sw.dt)

    class _Policy:
      started = False

      def __call__(self, obs):
        if self.started:  # Loads from the step that just ran.
          row = rec.record()
          plot.update(float(sw.time[0]), row[0].cpu().numpy(), int(rec.phases[-1][0]))
        self.started = True
        if script is not None:
          script()
        out = sw(obs)
        if int(sw.time[0] / sw.dt) % 10 == 0:
          print(sw.status(), end="", flush=True)
        return out

      def reset(self):
        sw.reset()

    print(__doc__)
    try:
      NativeMujocoViewer(wenv, _Policy(), key_callback=chain_key_handler(sw)).run()
    except KeyboardInterrupt:
      pass
  env.close()
  if not rec.rows:
    print("\nnothing recorded")
    return
  print("\n\n" + summarize(rec, args.out, label))
  print(f"\nsaved {args.out}/summary.txt, robot0.csv and loads.png")


if __name__ == "__main__":
  main()
