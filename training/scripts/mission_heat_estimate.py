"""Motor temperatures over a long session, from measured per-activity loads.

Each activity's per-motor RMS torque comes from a ``thermal_eval.py`` JSON
(``joints``, or ``joints_transition`` for the stand <-> crawl descent and
stand-up). A schedule of (activity, seconds) segments, repeated, is integrated
with the two-node winding/housing model of ``g1_motor_safety.motor_model``
(fans, driver heat) at 0.1 s, each motor's copper loss at that activity's RMS
torque times ``--pessimism`` (the first robot test heated about 2x faster than
the model). The gait's torque ripple averages out: the winding time constants
are 10-25 s. Iron loss (speed-dependent, small) is left out.

  uv run python scripts/mission_heat_estimate.py \\
      --activity crawl=crawl_move0.2.json --activity rest=crawl_rest.json \\
      --activity down=hold.json:joints_transition --activity hold=hold.json \\
      --schedule down:8,hold:3,crawl:240,rest:60,down:8 --repeat 6 --plot temps.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mjlab.tasks.g1_motor_safety import motor_model as mm
from mjlab.tasks.g1_walk.robot import JOINTS_29

SHORT = [
  j.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
  for j in JOINTS_29
]


def load_activity(spec: str) -> tuple[str, np.ndarray]:
  name, rest = spec.split("=", 1)
  path, _, key = rest.partition(":")
  data = json.loads(Path(path).read_text())[key or "joints"]
  return name, np.array([data[s]["rms_nm"] for s in SHORT])


def main() -> None:
  ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  ap.add_argument("--activity", action="append", required=True)
  ap.add_argument("--schedule", required=True, help="name:seconds,name:seconds,...")
  ap.add_argument("--repeat", type=int, default=1)
  ap.add_argument("--pessimism", type=float, default=2.0)
  ap.add_argument("--ambient", type=float, default=30.0)
  ap.add_argument("--start-temp", type=float, default=40.0)
  ap.add_argument("--limit", type=float, default=70.0, help="Winding target, C.")
  ap.add_argument("--title", default="")
  ap.add_argument("--plot", type=Path, default=None)
  ap.add_argument("--json", type=Path, default=None)
  args = ap.parse_args()

  acts = dict(load_activity(a) for a in args.activity)
  segs = [(n, float(t)) for n, t in (x.split(":") for x in args.schedule.split(","))]
  segs = segs * args.repeat
  motors = [mm.motor_for_joint(j) for j in JOINTS_29]
  g = lambda f: np.array([getattr(m, f) for m in motors], dtype=float)  # noqa: E731
  r25, kt, gear, n_mot = g("r25"), g("kt"), g("gear"), g("n_motors")
  c_w, r_wh, c_h, r_ha, q_drv = g("c_w"), g("r_wh"), g("c_h"), g("r_ha"), g("q_drv")
  tau_cont = np.array([m.continuous_torque() for m in motors])
  k_fan = mm.COOLING["k_fan"]
  dt = 0.1
  t_w = np.full(25, args.start_temp)
  t_h = t_w.copy()
  peak = t_w.copy()
  ts, tw_hist, act_hist = [], [], []
  t = 0.0
  for name, dur in segs:
    tau = acts[name]
    current = tau / (n_mot * gear * kt)
    for _ in range(int(round(dur / dt))):
      r = r25 * (1.0 + mm.ALPHA_CU * (t_w - mm.T_REF))
      power = args.pessimism * r * current**2
      fan = np.clip((t_w - mm.FAN_ON) / (mm.FAN_FULL - mm.FAN_ON), 0.0, 1.0)
      q_wh = (t_w - t_h) / r_wh
      q_ha = (1.0 + k_fan * fan) * (t_h - args.ambient) / r_ha
      t_w = t_w + dt * (power - q_wh) / c_w
      t_h = t_h + dt * (q_wh - q_ha + q_drv) / c_h
      peak = np.maximum(peak, t_w)
      t += dt
      if int(round(t / dt)) % 10 == 0:
        ts.append(t / 60.0)
        tw_hist.append(t_w.copy())
        act_hist.append(name)
  hist = np.array(tw_hist)
  order = np.argsort(-peak)
  total_min = t / 60.0
  crawl_min = sum(d for n_, d in segs if n_ not in ("walk", "down", "hold")) / 60.0
  print(
    f"session {total_min:.1f} min ({crawl_min:.1f} min crawling/resting), "
    f"heat x{args.pessimism}, ambient {args.ambient} C, start {args.start_temp} C"
  )
  print(
    f"{'motor':18s} {'peak C':>7s} {'end C':>7s}  "
    + "  ".join(f"{a:>10s}" for a in acts)
  )
  for j in order[:10]:
    loads = "  ".join(f"{acts[a][j] / tau_cont[j]:9.2f}x" for a in acts)
    print(f"{SHORT[j]:18s} {peak[j]:7.1f} {t_w[j]:7.1f}  {loads}")
  worst = float(peak.max())
  verdict = "PASS" if worst <= args.limit else "FAIL"
  print(
    f"hottest winding {worst:.1f} C ({SHORT[int(order[0])]}): {verdict} (target {args.limit} C)"
  )
  if args.json is not None:
    args.json.write_text(
      json.dumps(
        {
          "peak_c": {SHORT[j]: round(float(peak[j]), 1) for j in range(25)},
          "hottest_c": round(worst, 1),
          "verdict": verdict,
          "minutes": total_min,
          "pessimism": args.pessimism,
          "trace_minutes": [round(x, 3) for x in ts],
          "trace_hottest_c": [round(float(x), 2) for x in hist.max(axis=1)],
        },
        indent=1,
      )
    )
  if args.plot is not None:
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    colors = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
    for c, j in zip(colors, order[:5], strict=False):
      ax.plot(ts, hist[:, j], color=c, linewidth=2, label=SHORT[j])
    ax.axhline(args.limit, color="#52514e", linestyle="--", linewidth=1)
    ax.text(
      0, args.limit + 0.8, f"target {args.limit:.0f} C", color="#52514e", fontsize=9
    )
    ax.axhline(85, color="#e34948", linestyle=":", linewidth=1)
    ax.text(0, 85.8, "robot controller stop 85 C", color="#e34948", fontsize=9)
    ax.set_xlabel("minutes", color="#52514e")
    ax.set_ylabel("winding temperature, C", color="#52514e")
    ax.set_title(
      args.title or "Motor windings over the session", loc="left", fontweight="bold"
    )
    ax.grid(color="#e6e5e0")
    for s in ("top", "right"):
      ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(args.plot, dpi=130, facecolor="#fcfcfb")


if __name__ == "__main__":
  main()
