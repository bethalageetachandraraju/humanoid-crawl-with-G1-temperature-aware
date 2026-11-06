"""Session overview of one or more g1_chain_real.py logs: hottest motor, and
temperature, torque and estimated current of all 29 motors.

  python analysis/plot_session.py example_logs/chain_real_20261030_093943.npz \
    controller/logs/chain_real_<stamp>.npz --out session_<date>
  # needs numpy + matplotlib; writes analysis/<out>/

Needs logs from 2026-10-30 on (``temp`` and ``tau_motor`` for all 29 motors).

Current is NOT measured by the robot (LowState has torque, temperature and
voltage, no current). It is estimated from the torque estimate:
I = tau / (n * gear * kt), with the gear ratios of mjlab's G1 and the torque
constants of g1_motor_safety/motor_model.py (stand-ins, not measured on a G1).
Ankle and waist roll/pitch are two motors in a linkage (Unitree's A/B motors):
per motor, half the joint torque is assumed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from plot_motors import (  # noqa: E402
  CLASSES,
  MOTOR_OF_POLICY,
  PHASES,
  T_STAND_UP,
  T_STOP,
  WRISTS,
  motor_class,
  phases,
  rolling_rms,
  runs,
)

DT = 0.02
N = 29
# Reference palette (dataviz skill): categorical slots, sequential ramps, status.
SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE = LinearSegmentedColormap.from_list(
  "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
ORANGE = LinearSegmentedColormap.from_list(
  "orange", ["#fdf3ee", "#fbd3c0", "#f4a27e", "#eb6834", "#c94e1f", "#9a3913", "#6b260a"]
)
WARN, CRIT, INK, INK2, GRID = "#fab219", "#d03b3b", "#0b0b0b", "#52514e", "#e6e5e1"
# (gear, kt Nm/A, motors per joint) per class: mjlab G1 gears, motor_model.py kt.
ELEC = {
  "7520-14": (4.5 * (1 + 48 / 22), 0.10, 1),
  "7520-22": (4.5 * 5.0, 0.10, 1),
  "5020x2": ((1 + 46 / 18) * (1 + 56 / 16), 0.055, 2),
  "5020": ((1 + 46 / 18) * (1 + 56 / 16), 0.055, 1),
  "4010": (25.0, 0.025, 1),
}
plt.rcParams.update(
  {
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "font.size": 9,
  }
)


def amps_per_nm(cls: str) -> float:
  gear, kt, n = ELEC[cls]
  return 1.0 / (n * gear * kt)


class Run:
  def __init__(self, path: Path) -> None:
    L = np.load(path)
    assert "temp" in L.files and "tau_motor" in L.files, f"{path}: log before 10-30"
    self.name = path.stem.replace("chain_real_", "")
    self.t = L["t"] - L["t"][0]
    stamp = datetime.strptime(self.name, "%Y%m%d_%H%M%S")
    self.start = stamp - timedelta(seconds=float(self.t[-1]))
    self.label = f"{self.start:%H:%M:%S}-{stamp:%H:%M:%S}"
    self.ph = phases([str(m) for m in L["modes"]], L["mode"], L["chain_mode"])
    self.temp = L["temp"].max(axis=2).astype(float)  # (T, 29), hotter reading
    self.tau = L["tau_motor"].astype(float)  # (T, 29), motor order
    joints = [str(j) for j in L["joints"]]
    names = {MOTOR_OF_POLICY[i]: j.removesuffix("_joint") for i, j in enumerate(joints)}
    self.names = [(names | WRISTS)[m] for m in range(N)]
    self.cls = [motor_class(n) for n in self.names]
    self.peak = np.array([CLASSES[c][0] for c in self.cls])
    self.cont = np.array([CLASSES[c][1] for c in self.cls])
    k = np.array([amps_per_nm(c) for c in self.cls])
    self.amps = self.tau * k
    self.amps_cont = self.cont * k
    self.amps_peak = self.peak * k
    self.short = [short(n) for n in self.names]


def short(name: str) -> str:
  return (
    name.replace("left_", "L ").replace("right_", "R ").replace("_", " ")
    .replace("shoulder", "shldr")
  )  # fmt: skip


def shade(ax, run: Run) -> None:
  for name, colour in PHASES.items():
    for a, b in runs(np.array([p == name for p in run.ph]), merge=1):
      ax.axvspan(run.t[a], run.t[min(b, len(run.t) - 1)], color=colour, lw=0, zorder=0)


def phase_legend(fig, y=0.005) -> None:
  fig.legend(
    handles=[Patch(color=c, label=n) for n, c in PHASES.items()],
    loc="lower center",
    ncol=5,
    frameon=False,
    bbox_to_anchor=(0.5, y),
  )


def fig_hottest(run: Run):
  hot = run.temp.max(axis=1)
  who = run.temp.argmax(axis=1)
  fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
  for ax in (a1, a2):
    shade(ax, run)
    ax.axhline(T_STOP, color=CRIT, ls="--", lw=1)
    ax.axhline(T_STAND_UP, color=WARN, ls="--", lw=1)
    ax.set_ylabel("temperature (C)")
    ax.set_ylim(min(30, hot.min() - 3), 90)
  a1.plot(run.t, hot, color=SLOT[0], lw=2)
  a1.set_title("hottest motor (hotter of its two readings)", loc="left")
  a1.text(1, T_STOP + 0.8, "85 C damping stop", fontsize=8, color=INK2)
  a1.text(1, T_STAND_UP + 0.8, "72 C guard stands the robot up", fontsize=8, color=INK2)
  # Which motor is hottest: label every stretch of 5 s or more.
  edges = np.r_[0, np.where(np.diff(who) != 0)[0] + 1, len(who)]
  for a, b in zip(edges[:-1], edges[1:], strict=True):
    if (b - a) * DT >= 5:
      mid = (a + b) // 2
      a1.annotate(
        run.short[who[a]],
        xy=(run.t[mid], hot[mid]),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=INK,
      )
  top = np.argsort(-run.temp.max(axis=0))[:3]
  for s, m in enumerate(top):
    a2.plot(run.t, run.temp[:, m], color=SLOT[s], lw=2, label=f"m{m} {run.short[m]}")
    a2.annotate(
      f"m{m} {run.short[m]}",
      xy=(run.t[-1], run.temp[-1, m]),
      xytext=(4, 0),
      textcoords="offset points",
      fontsize=8,
      color=INK,
      va="center",
    )
  a2.set_title("the three motors that got hottest", loc="left")
  a2.legend(loc="upper left", frameon=False)
  a2.set_xlabel("time since start (s)")
  a2.set_xlim(0, run.t[-1] * 1.08)
  fig.suptitle(f"Run {run.label}: hottest motor", x=0.01, ha="left", fontsize=12)
  phase_legend(fig)
  fig.tight_layout(rect=(0, 0.04, 1, 0.97))
  return fig


def binned(x: np.ndarray, n: int) -> np.ndarray:
  """Mean over consecutive bins of n samples: (T, M) -> (T // n, M)."""
  k = (len(x) // n) * n
  return x[:k].reshape(-1, n, x.shape[1]).mean(axis=1)


def fig_heatmaps(run: Run):
  n = int(0.5 / DT)
  temp = binned(run.temp, n)
  load = binned(np.abs(run.tau), n) / run.peak
  amps = binned(np.abs(run.amps), n)
  t_edges = np.arange(len(temp) + 1) * n * DT
  order = list(range(N))
  fig = plt.figure(figsize=(15, 16))
  gs = fig.add_gridspec(
    4,
    2,
    height_ratios=[0.25, 6, 6, 6],
    width_ratios=[60, 1],
    hspace=0.28,
    wspace=0.02,
    top=0.96,
    bottom=0.05,
    left=0.11,
    right=0.96,
  )
  strip = fig.add_subplot(gs[0, 0])
  shade(strip, run)
  strip.set_xlim(0, t_edges[-1])
  strip.set_yticks([])
  strip.tick_params(bottom=False, labelbottom=False)
  strip.grid(False)
  strip.set_title(
    f"Run {run.label}: all 29 motors (0.5 s bins), phases on top", loc="left", fontsize=12
  )
  panels = (
    (temp, ORANGE, 30, 75, "temperature (C), hotter reading (scale ends at 75)"),
    (load, BLUE, 0, 0.8, "|torque| / peak torque"),
    (amps, BLUE, 0, max(10.0, float(np.percentile(amps, 99.5))), "estimated |current| (A)"),
  )
  for row, (data, cmap, lo, hi, title) in enumerate(panels, start=1):
    ax = fig.add_subplot(gs[row, 0], sharex=strip)
    im = ax.pcolormesh(
      t_edges, np.arange(N + 1), data.T[order], cmap=cmap, vmin=lo, vmax=hi
    )
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_yticklabels([f"m{m} {run.short[m]}" for m in order], fontsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title(title, loc="left")
    ax.tick_params(labelbottom=True)
    if row == 3:
      ax.set_xlabel("time since start (s)")
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[row, 1]))
    cb.outline.set_visible(False)
  fig.text(
    0.01,
    0.003,
    "Current is estimated from the torque estimate (I = tau / (n gear kt), motor_model.py "
    "constants, not measured). Linkage joints (ankles, waist roll/pitch): half the joint "
    "torque per motor.",
    fontsize=8,
    color=INK2,
  )
  return fig


def fig_multiples(run: Run, what: str):
  fig, axs = plt.subplots(6, 5, figsize=(18, 15), sharex=True)
  win = int(2.0 / DT)
  for m, ax in zip(range(30), axs.flat, strict=True):
    if m >= N:
      ax.axis("off")
      continue
    shade(ax, run)
    if what == "temperature":
      ax.plot(run.t, run.temp[:, m], color=SLOT[0], lw=1.6)
      ax.axhline(T_STOP, color=CRIT, ls="--", lw=0.8)
      ax.axhline(T_STAND_UP, color=WARN, ls="--", lw=0.8)
      ax.set_ylim(30, 90)
      hit = run.temp[:, m].max() >= 65
    else:
      y = run.tau[:, m] if what == "torque" else run.amps[:, m]
      cont = run.cont[m] if what == "torque" else run.amps_cont[m]
      rms = rolling_rms(y, win)
      ax.plot(run.t, np.abs(y), color="#b7d3f6", lw=0.6)
      ax.plot(run.t, rms, color=SLOT[0], lw=1.6)
      ax.axhline(cont, color=INK2, ls="--", lw=0.9)
      ax.set_ylim(0, max(cont * 1.6, float(np.abs(y).max()) * 1.05))
      hit = rms.max() >= cont
    ax.set_title(
      f"m{m} {run.short[m]}", fontsize=9, color=CRIT if hit else INK, loc="left"
    )
    ax.tick_params(labelsize=7)
  unit = {"temperature": "C", "torque": "Nm", "current": "A (estimated)"}[what]
  head = {
    "temperature": "temperature (hotter reading); dashed: 72 C stand-up, 85 C stop; "
    "red title = reached 65 C",
    "torque": "|torque| (light) and its 2-s RMS (blue); dashed: estimated continuous "
    "rating; red title = RMS above the rating",
    "current": "estimated |current| (light) and its 2-s RMS (blue); dashed: current at "
    "the estimated continuous torque; red title = above it",
  }[what]
  fig.suptitle(f"Run {run.label}: {what} of all 29 motors ({unit})\n{head}", fontsize=12)
  phase_legend(fig)
  fig.tight_layout(rect=(0, 0.03, 1, 0.96))
  return fig


def fig_compare(all_runs: list[Run]):
  fig, ax = plt.subplots(figsize=(14, 5))
  for s, run in enumerate(all_runs):
    hot = run.temp.max(axis=1)
    ax.plot(run.t, hot, color=SLOT[s], lw=2, label=f"run {run.label}")
    crawl = np.array([p == "crawl" for p in run.ph])
    for a, b in runs(crawl, merge=1):
      ax.plot(run.t[a:b], hot[a:b], color=SLOT[s], lw=5, solid_capstyle="round")
    ax.annotate(
      f"run {run.label}",
      xy=(run.t[-1], hot[-1]),
      xytext=(4, 0),
      textcoords="offset points",
      fontsize=8,
      va="center",
    )
  ax.axhline(T_STAND_UP, color=WARN, ls="--", lw=1)
  ax.axhline(T_STOP, color=CRIT, ls="--", lw=1)
  ax.set_ylim(30, 90)
  ax.set_xlabel("time since start (s)")
  ax.set_ylabel("hottest motor (C)")
  ax.legend(loc="upper left", frameon=False)
  ax.set_title(
    "Hottest motor in each run; thick line = crawling. Dashed: 72 C stand-up, 85 C stop",
    loc="left",
  )
  fig.tight_layout()
  return fig


def summary_text(run: Run) -> str:
  hot = run.temp.max(axis=0)
  m = int(hot.argmax())
  rms = np.array([rolling_rms(run.tau[:, k], int(2 / DT)).max() for k in range(N)])
  over = [f"m{k} {run.short[k]} x{rms[k] / run.cont[k]:.2f}" for k in range(N)
          if rms[k] >= run.cont[k]]  # fmt: skip
  amp = int(np.argmax(np.abs(run.amps).max(axis=0)))
  return (
    f"run {run.label}: {run.t[-1]:.0f} s; hottest m{m} {run.short[m]} {hot[m]:.0f} C; "
    f"highest estimated current m{amp} {run.short[amp]} "
    f"{np.abs(run.amps[:, amp]).max():.1f} A; 2-s RMS over the rating: "
    + (", ".join(over) or "none")
  )


def main() -> None:
  ap = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  ap.add_argument("logs", nargs="+")
  ap.add_argument("--out", default="session")
  args = ap.parse_args()
  out = Path(__file__).resolve().parent / args.out
  out.mkdir(parents=True, exist_ok=True)
  all_runs = [Run(Path(p)) for p in args.logs]
  with PdfPages(out / f"{args.out}.pdf") as pdf:
    pages = [("compare_hottest", fig_compare(all_runs))]
    for run in all_runs:
      pages += [
        (f"{run.name}_1_hottest", fig_hottest(run)),
        (f"{run.name}_2_heatmaps", fig_heatmaps(run)),
        (f"{run.name}_3_temperature", fig_multiples(run, "temperature")),
        (f"{run.name}_4_torque", fig_multiples(run, "torque")),
        (f"{run.name}_5_current", fig_multiples(run, "current")),
      ]
    for name, fig in pages:
      fig.savefig(out / f"{name}.png", dpi=100)
      pdf.savefig(fig)
      plt.close(fig)
  print(f"wrote {out}/{args.out}.pdf and {len(pages)} PNGs")
  for run in all_runs:
    print(" ", summary_text(run))


if __name__ == "__main__":
  main()
