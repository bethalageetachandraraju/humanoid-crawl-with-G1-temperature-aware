"""Torque and temperature of every G1 motor in a g1_chain_real.py log.

One figure per motor (29): torque on top, temperature below; the whole run on
the left, a zoom on stand->crawl->stand on the right. Phase-coloured
background. Limits drawn as lines; markers where a value rises suddenly,
gets near its limit or crosses it. Also a summary table and an overview page.

  python analysis/plot_motors.py example_logs/chain_real_20261030_093943.npz
  # needs numpy + matplotlib; writes analysis/plots_<log stamp>/

Limits (the same as docs/REPORT_2026-10-29_first_robot_test.md): peak torque per motor class (mjlab G1), the
estimated continuous torque from g1_motor_safety/README.md (stand-ins, not
measured on a G1), and the controller's temperature thresholds (72 C stand up
from the floor, 85 C damping stop).
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DT = 0.02  # log rate (50 Hz)
NUM_MOTORS = 29
MOTOR_OF_POLICY = list(range(20)) + [22, 23, 24, 25, 26]
WRISTS = {
  20: "left_wrist_pitch",
  21: "left_wrist_yaw",
  27: "right_wrist_pitch",
  28: "right_wrist_yaw",
}
CHAIN = ["WALK", "STOPPING", "DOWN", "HOLD", "CRAWL", "CRAWL_STOP", "UP", "STAND_HOLD"]
# Motor classes: (label, peak Nm, estimated continuous Nm).
CLASSES = {
  "7520-14": (88.0, 23.0),
  "7520-22": (139.0, 36.0),
  "5020x2": (50.0, 13.5),
  "5020": (25.0, 6.8),
  "4010": (5.0, 1.8),
}
T_STAND_UP, T_STOP, T_CRAWL_OK = 72.0, 85.0, 60.0
T_NEAR = 65.0  # within 7 C of the stand-up limit
PHASES = {  # name: colour
  "stand / walk": "#cfe8fc",
  "stand → crawl": "#ffd9a8",
  "crawl": "#ffb3b3",
  "crawl → stand": "#dcc8f0",
  "damping": "#e0e0e0",
}
NEAR_TORQUE = 0.8  # of peak
RMS_WINDOW_S = 5.0
RISE_TORQUE_WINDOW_S = 2.0
RISE_TORQUE = 0.5  # rolling-RMS increase, in units of the continuous rating
RISE_TEMP_WINDOW_S = 3.0
RISE_TEMP = 1.0  # C/s (readings are whole degrees: 3 C in 3 s)
MERGE_S = 3.0  # markers closer than this count as one event


def motor_class(name: str) -> str:
  if "wrist_pitch" in name or "wrist_yaw" in name:
    return "4010"
  if any(k in name for k in ("hip_pitch", "hip_yaw", "waist_yaw")):
    return "7520-14"
  if any(k in name for k in ("hip_roll", "knee")):
    return "7520-22"
  if any(k in name for k in ("ankle", "waist_roll", "waist_pitch")):
    return "5020x2"
  return "5020"


def phases(mode_names: list[str], mode: np.ndarray, chain: np.ndarray) -> list[str]:
  """Per sample: one of PHASES."""
  out, after_crawl = [], False
  for m, c in zip(mode, chain, strict=True):
    name = mode_names[m]
    if name == "DAMP":
      out.append("damping")
      continue
    if name != "CHAIN" or c < 0:
      out.append("stand / walk")
      continue
    cn = CHAIN[c]
    if cn in ("CRAWL", "CRAWL_STOP"):
      after_crawl = True
      out.append("crawl")
    elif cn in ("STOPPING", "DOWN"):
      after_crawl = False
      out.append("stand → crawl")
    elif cn == "HOLD":
      out.append("crawl → stand" if after_crawl else "stand → crawl")
    elif cn in ("UP", "STAND_HOLD"):
      out.append("crawl → stand")
    else:
      after_crawl = False
      out.append("stand / walk")
  return out


def runs(mask: np.ndarray, merge: int = int(MERGE_S / DT)) -> list[tuple[int, int]]:
  """[start, end) index pairs of the True runs; runs less than ``merge``
  samples apart are joined (whole-degree readings flicker)."""
  d = np.diff(np.r_[0, mask.astype(int), 0])
  out: list[tuple[int, int]] = []
  for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0], strict=True):
    if out and a - out[-1][1] < merge:
      out[-1] = (out[-1][0], b)
    else:
      out.append((a, b))
  return out


def rolling_rms(x: np.ndarray, n: int) -> np.ndarray:
  """Trailing RMS over n samples (shorter at the start)."""
  c = np.cumsum(np.r_[0.0, x * x])
  i = np.arange(1, len(x) + 1)
  lo = np.maximum(0, i - n)
  return np.sqrt((c[i] - c[lo]) / (i - lo))


def shade(ax, t, ph) -> None:
  for name, colour in PHASES.items():
    for a, b in runs(np.array([p == name for p in ph])):
      ax.axvspan(t[a], t[min(b, len(t) - 1)], color=colour, lw=0, zorder=0)


def analyse(t, tau, temp, peak, cont):
  """Markers and numbers for one motor. tau / temp may be None."""
  r = {"flags": []}
  if tau is not None:
    a = np.abs(tau)
    rms = rolling_rms(tau, int(RMS_WINDOW_S / DT))
    r["rms"] = rms
    r["max_abs"] = float(a.max())
    r["max_rms"] = float(rms.max())
    r["near"] = [s for s, _ in runs((a >= NEAR_TORQUE * peak) & (a < peak))]
    r["over"] = [s for s, _ in runs(a >= peak)]
    r["over_cont"] = runs(rms >= cont)
    n = int(RISE_TORQUE_WINDOW_S / DT)
    rms2 = rolling_rms(tau, n)
    jump = np.r_[np.zeros(n), rms2[n:] - rms2[:-n]]
    r["tau_rise"] = [s for s, _ in runs(jump >= RISE_TORQUE * cont)]
    if r["over"]:
      r["flags"].append("torque over peak")
    elif r["near"]:
      r["flags"].append(f"torque ≥ {NEAR_TORQUE:.0%} peak")
    if r["over_cont"]:
      r["flags"].append(f"RMS ×{r['max_rms'] / cont:.2f} continuous")
    if r["tau_rise"]:
      r["flags"].append("sudden load rise")
  if temp is not None:
    hot = temp.max(axis=1).astype(float)
    n = int(RISE_TEMP_WINDOW_S / DT)
    slope = np.r_[np.zeros(n), (hot[n:] - hot[:-n]) / RISE_TEMP_WINDOW_S]
    r["hot"] = hot
    r["slope"] = slope
    r["temp_rise"] = runs(slope >= RISE_TEMP)
    r["temp_near"] = [s for s, _ in runs((hot >= T_NEAR) & (hot < T_STAND_UP))]
    r["temp_up"] = [s for s, _ in runs((hot >= T_STAND_UP) & (hot < T_STOP))]
    r["temp_over"] = [s for s, _ in runs(hot >= T_STOP)]
    r["t0"], r["tmax"] = float(hot[0]), float(hot.max())
    r["max_slope"] = float(slope.max())
    if r["temp_over"]:
      r["flags"].append(f"crossed {T_STOP:.0f} C stop")
    elif r["temp_up"]:
      r["flags"].append(f"crossed {T_STAND_UP:.0f} C stand-up limit")
    elif r["temp_near"]:
      r["flags"].append(f"near limit: {r['tmax']:.0f} C (≥ {T_NEAR:.0f})")
    if r["temp_rise"]:
      r["flags"].append(f"sudden temp rise {r['max_slope']:.1f} C/s")
  return r


def draw_torque(ax, t, tau, r, peak, cont, zoom: bool) -> None:
  if tau is None:
    ax.text(
      0.5,
      0.5,
      "torque not logged\n(braced wrist motor: damping, kp 0, kd 0.5)",
      ha="center",
      va="center",
      transform=ax.transAxes,
      color="0.4",
    )
    ax.set_ylim(-peak * 1.1, peak * 1.1)
  else:
    ax.plot(t, tau, lw=0.4 if not zoom else 0.7, color="0.35", label="torque (tau_est)")
    rms = r["rms"]
    ax.plot(t, rms, lw=1.6, color="#1f5fbf", label=f"{RMS_WINDOW_S:.0f}-s RMS")
    over = np.where(rms >= cont, rms, np.nan)
    ax.plot(t, over, lw=2.4, color="#d62728", label="RMS over continuous")
    for s in r["near"]:
      ax.plot(t[s], tau[s], "^", color="#ff7f0e", ms=7, mec="k", mew=0.5)
    for s in r["over"]:
      ax.plot(t[s], tau[s], "X", color="#d62728", ms=9, mec="k", mew=0.5)
    for s in r["tau_rise"]:
      ax.annotate(
        "",
        xy=(t[s], rms[s]),
        xytext=(t[s], rms[s] - 0.35 * cont),
        arrowprops=dict(arrowstyle="-|>", color="#9467bd", lw=1.5),
      )
    if r["over_cont"] and not zoom:
      a, _ = r["over_cont"][0]
      ax.annotate(
        f"RMS > continuous\n(max ×{r['max_rms'] / cont:.2f})",
        xy=(t[a], rms[a]),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=8,
        color="#d62728",
        arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.8),
      )
    lim = max(peak * 1.08, r["max_abs"] * 1.1)  # the peak lines stay in view
    ax.set_ylim(-lim, lim)
  for sgn in (1, -1):
    ax.axhline(sgn * peak, color="#d62728", ls="--", lw=1.0)
    ax.axhline(sgn * NEAR_TORQUE * peak, color="#ff7f0e", ls=":", lw=0.9)
    ax.axhline(sgn * cont, color="#2ca02c", ls="-.", lw=1.0)
  ax.axhline(0, color="0.6", lw=0.5)
  ax.set_ylabel("torque (Nm)")


def draw_temp(ax, t, temp, r, zoom: bool) -> None:
  if temp is None:
    ax.text(0.5, 0.5, "per-motor temperature not logged", ha="center",
            va="center", transform=ax.transAxes, color="0.4")
    ax.set_ylim(20, 95)
  else:
    ax.plot(t, temp[:, 0], color="#8c564b", lw=1.0, label="reading 1 (slow)")
    ax.plot(t, temp[:, 1], color="#e377c2", lw=1.4, label="reading 2 (fast)")
    hot = r["hot"]
    for a, b in r["temp_rise"]:
      k = a + int(np.argmax(r["slope"][a:b]))
      ax.plot(t[a], hot[a], "o", color="#9467bd", ms=6, mec="k", mew=0.5)
      ax.annotate(
        f"+{r['slope'][k]:.1f} C/s",
        xy=(t[k], hot[k]),
        xytext=(-58, 4),
        textcoords="offset points",
        fontsize=8,
        color="#9467bd",
      )
    for s in r["temp_near"]:
      ax.plot(t[s], hot[s], "^", color="#ff7f0e", ms=8, mec="k", mew=0.5)
    for s in r["temp_up"]:
      ax.plot(t[s], hot[s], "X", color="#ff7f0e", ms=10, mec="k", mew=0.5)
    for s in r["temp_over"]:
      ax.plot(t[s], hot[s], "X", color="#d62728", ms=10, mec="k", mew=0.5)
    lo = min(35.0, float(temp.min()) - 3)
    ax.set_ylim(lo, max(90.0, float(temp.max()) + 3))
  ax.axhline(T_STOP, color="#d62728", ls="--", lw=1.0)
  ax.axhline(T_STAND_UP, color="#ff7f0e", ls=":", lw=1.0)
  ax.axhline(T_NEAR, color="#ff7f0e", ls=(0, (1, 4)), lw=0.8)
  ax.axhline(T_CRAWL_OK, color="#2ca02c", ls="-.", lw=0.8)
  if not zoom:
    for y, txt in ((T_STOP, "85 C stop"), (T_STAND_UP, "72 C stand up"),
                   (T_NEAR, "65 C near"), (T_CRAWL_OK, "60 C crawl allowed below")):
      ax.text(0.003, y + 0.6, txt, transform=ax.get_yaxis_transform(), fontsize=7,
              color="0.25")
  ax.set_ylabel("temperature (C)")
  ax.set_xlabel("time (s)")


def legend_handles():
  from matplotlib.lines import Line2D

  h = [Patch(color=c, label=n) for n, c in PHASES.items()]
  h += [
    Line2D([], [], color="0.35", lw=0.8, label="torque"),
    Line2D([], [], color="#1f5fbf", lw=1.6, label=f"{RMS_WINDOW_S:.0f}-s RMS torque"),
    Line2D([], [], color="#d62728", lw=2.4, label="RMS above continuous"),
    Line2D([], [], color="#d62728", ls="--", label="peak torque / 85 C stop"),
    Line2D([], [], color="#ff7f0e", ls=":", label="80% peak / 72 C stand-up"),
    Line2D([], [], color="#2ca02c", ls="-.", label="est. continuous / 60 C"),
    Line2D([], [], color="#8c564b", label="temp reading 1"),
    Line2D([], [], color="#e377c2", label="temp reading 2"),
    Line2D([], [], marker="^", ls="", color="#ff7f0e", mec="k",
           label="near limit (80% peak, 65 C)"),
    Line2D([], [], marker="X", ls="", color="#ff7f0e", mec="k", label="crossed 72 C"),
    Line2D([], [], marker="X", ls="", color="#d62728", mec="k",
           label="crossed peak / 85 C"),
    Line2D([], [], marker="o", ls="", color="#9467bd", mec="k",
           label=f"sudden temp rise (≥{RISE_TEMP:g} C/s)"),
    Line2D([], [], marker=r"$\uparrow$", ls="", color="#9467bd", ms=9,
           label="sudden load rise"),
  ]
  return h


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("log")
  ap.add_argument("--out", default=None, help="default: plots_<log stamp>/ here")
  args = ap.parse_args()
  log = Path(args.log)
  L = np.load(log)
  stamp = re.search(r"(\d{8}_\d{6})", log.name)
  out = Path(args.out) if args.out else Path(__file__).resolve().parent / (
    f"plots_{stamp.group(1) if stamp else log.stem}"
  )
  out.mkdir(parents=True, exist_ok=True)

  t = L["t"] - L["t"][0]
  joints = [str(j) for j in L["joints"]]
  ph = phases([str(m) for m in L["modes"]], L["mode"], L["chain_mode"])
  # (N, 29, 2): "temp" from 2026-10-30 on, "temps" before.
  key = "temp" if "temp" in L.files else "temps" if "temps" in L.files else None
  temps = L[key] if key else None
  end = datetime.strptime(stamp.group(1), "%Y%m%d_%H%M%S") if stamp else None
  start_txt = (end - timedelta(seconds=float(t[-1]))).strftime("%H:%M:%S") if end else ""
  names = {MOTOR_OF_POLICY[i]: j.removesuffix("_joint") for i, j in enumerate(joints)}
  names |= WRISTS
  policy_of_motor = {m: i for i, m in enumerate(MOTOR_OF_POLICY)}

  # Zoom: stand->crawl->stand, with some walking either side.
  down = np.array([p != "stand / walk" and p != "damping" for p in ph])
  if down.any():
    z0 = max(0.0, t[np.argmax(down)] - 6)
    z1 = min(t[-1], t[len(down) - 1 - np.argmax(down[::-1])] + 15)
  else:
    z0, z1 = 0.0, t[-1]

  results = {}
  pdf = PdfPages(out / "all_motors.pdf")
  pages = []
  for m in range(NUM_MOTORS):
    name = names[m]
    cls = motor_class(name)
    peak, cont = CLASSES[cls]
    if "tau_motor" in L.files:  # all 29 motors, motor order (2026-10-30 on)
      tau = L["tau_motor"][:, m]
    else:
      tau = L["tau_est"][:, policy_of_motor[m]] if m in policy_of_motor else None
    temp = temps[:, m, :].astype(float) if temps is not None else None
    r = analyse(t, tau, temp, peak, cont)
    results[m] = (name, cls, peak, cont, r)

    fig, axs = plt.subplots(
      2, 2, figsize=(16, 8.2), sharex="col",
      gridspec_kw=dict(width_ratios=[2.2, 1], hspace=0.08, wspace=0.12),
    )
    for col, zoom in ((0, False), (1, True)):
      for row in (0, 1):
        shade(axs[row, col], t, ph)
      draw_torque(axs[0, col], t, tau, r, peak, cont, zoom)
      draw_temp(axs[1, col], t, temp, r, zoom)
      if zoom:
        axs[0, col].set_xlim(z0, z1)
        axs[0, col].set_title("zoom: stand → crawl → stand", fontsize=10)
      else:
        axs[0, col].set_xlim(0, t[-1])
        axs[0, col].set_title(f"whole run (start {start_txt})", fontsize=10)
    head = f"motor {m}  {name}   [{cls}: peak {peak:g} Nm, est. continuous {cont:g} Nm]"
    stats = []
    if tau is not None:
      stats.append(
        f"max |τ| {r['max_abs']:.1f} Nm ({r['max_abs'] / peak:.0%} of peak), "
        f"max {RMS_WINDOW_S:.0f}-s RMS {r['max_rms']:.1f} Nm (×{r['max_rms'] / cont:.2f} cont)"
      )
    if temp is not None:
      stats.append(
        f"temp {r['t0']:.0f} → max {r['tmax']:.0f} C, fastest rise {r['max_slope']:.1f} C/s"
      )
    flag = "  ⚑ " + "; ".join(r["flags"]) if r["flags"] else "  no flags"
    fig.suptitle(head + "\n" + "   ·   ".join(stats) + "\n" + flag, fontsize=11,
                 x=0.06, ha="left", y=0.995,
                 color="#b00000" if r["flags"] else "black")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=8, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.05, right=0.99)
    png = out / f"motor_{m:02d}_{name}.png"
    fig.savefig(png, dpi=110)
    pages.append(fig)
    plt.close(fig)

  # Summary table + overview first, then the 29 motor pages.
  def severity(m):
    r = results[m][4]
    return (-r.get("tmax", 0.0), -r.get("max_rms", 0.0) / results[m][3])

  order = sorted(results, key=severity)
  fig = plt.figure(figsize=(16, 11))
  fig.suptitle(f"{log.name}: {t[-1]:.0f} s, start {start_txt} — sorted by peak "
               "temperature (red rows: flagged)",
               fontsize=13, x=0.03, ha="left")
  rows = []
  for m in order:
    name, cls, peak, cont, r = results[m]
    rows.append([
      f"m{m}", name, cls,
      f"{r['max_abs']:.1f}" if "max_abs" in r else "–",
      f"{r['max_abs'] / peak:.0%}" if "max_abs" in r else "–",
      f"×{r['max_rms'] / cont:.2f}" if "max_rms" in r else "–",
      f"{r['t0']:.0f}→{r['tmax']:.0f}" if "tmax" in r else "–",
      f"{r['max_slope']:.1f}" if "max_slope" in r else "–",
      "; ".join(r["flags"]) or "–",
    ])
  ax = fig.add_axes([0.02, 0.02, 0.96, 0.9])
  ax.axis("off")
  tbl = ax.table(
    cellText=rows,
    colLabels=["motor", "joint", "class", "max |τ| Nm", "% peak",
               f"{RMS_WINDOW_S:.0f}-s RMS / cont", "temp C", "max C/s", "flags"],
    colWidths=[0.04, 0.13, 0.06, 0.06, 0.05, 0.08, 0.06, 0.05, 0.47],
    loc="upper left", cellLoc="left",
  )
  tbl.auto_set_font_size(False)
  tbl.set_fontsize(8)
  tbl.scale(1, 1.32)
  for (i, _), cell in tbl.get_celld().items():
    if i > 0 and rows[i - 1][-1] != "–":
      cell.set_facecolor("#fde8e8")
  pdf.savefig(fig)
  fig.savefig(out / "summary.png", dpi=100)
  plt.close(fig)

  # Overview: hot temperature reading of every motor, phase background.
  fig, axs = plt.subplots(6, 5, figsize=(18, 15), sharex=True)
  for m, ax in zip(range(30), axs.flat, strict=True):
    if m >= NUM_MOTORS:
      ax.axis("off")
      continue
    name, cls, peak, cont, r = results[m]
    shade(ax, t, ph)
    if "hot" in r:
      ax.plot(t, r["hot"], color="#e377c2", lw=1.2)
      ax.set_ylim(min(35, r["hot"].min() - 3), 90)
    ax.axhline(T_STOP, color="#d62728", ls="--", lw=0.8)
    ax.axhline(T_STAND_UP, color="#ff7f0e", ls=":", lw=0.8)
    ax.set_title(f"m{m} {name}", fontsize=9,
                 color="#b00000" if r["flags"] else "black")
    ax.tick_params(labelsize=7)
  fig.suptitle("temperature (hotter reading) of all 29 motors; red title = flagged",
               fontsize=12)
  fig.legend(handles=[Patch(color=c, label=n) for n, c in PHASES.items()],
             loc="lower center", ncol=5, frameon=False)
  fig.tight_layout(rect=(0, 0.03, 1, 0.97))
  pdf.savefig(fig)
  fig.savefig(out / "overview_temperatures.png", dpi=90)
  plt.close(fig)
  for f in pages:
    pdf.savefig(f)
  pdf.close()
  print(f"wrote {out}: all_motors.pdf, summary.png, overview_temperatures.png, "
        f"29 motor_*.png")
  for m in order:
    name, cls, peak, cont, r = results[m]
    if r["flags"]:
      print(f"  m{m:2d} {name:22s} " + "; ".join(r["flags"]))


if __name__ == "__main__":
  main()
