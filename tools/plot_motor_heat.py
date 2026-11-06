"""Estimated motor winding temperature over a chain run, from the joint torques
and speeds that ``tools/render_chain_media.py`` saves (``*_log.npz``).

Uses the two-node (winding + housing) G1 motor model of the training code
(``mjlab.tasks.g1_motor_safety.motor_model``), passive cooling, 25 C ambient.
The model is an estimate: on 2026-10-29 the real motors heated about 2x faster.

  python tools/plot_motor_heat.py --run v3.1=a_log.npz --run v2=b_log.npz \
    --out media/images/motor_heat.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mjlab.tasks.g1_motor_safety.motor_model import (
  motor_for_joint,
  motor_power,
  thermal_step,
)

COLORS = ("#c2410c", "#1d4ed8")  # first run warm, second cool
MODE_BANDS = {2: "stand->crawl", 3: "hold", 4: "crawl", 6: "crawl->stand"}


def winding_temps(log: dict, t_amb: float = 25.0) -> np.ndarray:
  joints = [str(j) for j in log["joints"]]
  ps = [motor_for_joint(j) for j in joints]
  p = {
    k: torch.tensor([getattr(x, a) for x in ps], dtype=torch.float64)
    for k, a in (
      ("n", "n_motors"),
      ("gear", "gear"),
      ("kt", "kt"),
      ("r25", "r25"),
      ("k_iron", "k_iron"),
      ("r_wh", "r_wh"),
      ("r_ha", "r_ha"),
      ("c_w", "c_w"),
      ("c_h", "c_h"),
    )
  }
  dt = float(np.diff(log["t"]).mean())
  t_w = torch.full((len(joints),), t_amb, dtype=torch.float64)
  t_h = t_w.clone()
  amb = torch.full_like(t_w, t_amb)
  out = []
  for tau, dq in zip(log["tau"], log["dq"]):
    pw = motor_power(torch.from_numpy(tau), torch.from_numpy(dq), t_w, p)
    t_w, t_h = thermal_step(t_w, t_h, amb, pw, p, dt)
    out.append(t_w.numpy().copy())
  return np.array(out)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--run", action="append", required=True, help="label=path.npz")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  runs = []
  for spec in args.run:
    label, path = spec.split("=", 1)
    log = dict(np.load(path))
    runs.append((label, log, winding_temps(log)))

  fig, ax = plt.subplots(figsize=(9, 4.6), dpi=130)
  t0, m0 = runs[0][1]["t"], runs[0][1]["mode"]
  ymax = max(r[2].max() for r in runs) + 3
  for mode, name in MODE_BANDS.items():
    idx = np.nonzero(m0 == mode)[0]
    if len(idx):
      a, b = t0[idx[0]], t0[idx[-1]]
      ax.axvspan(a, b, color="#94a3b8", alpha=0.12, lw=0)
      ax.text((a + b) / 2, ymax - 0.5, name, ha="center", va="top", fontsize=8, color="#475569")
  for (label, log, tw), c in zip(runs, COLORS):
    hot = tw.max(axis=1)
    j = int(np.argmax(tw.max(axis=0)))
    joint = str(log["joints"][j]).removesuffix("_joint")
    ax.plot(log["t"], hot, color=c, lw=2, label=f"{label}: hottest motor (peak {hot.max():.0f} C, {joint})")
  ax.set_xlabel("time (s)")
  ax.set_ylabel("winding temperature, model (C)")
  ax.set_ylim(24, ymax)
  ax.grid(alpha=0.25, lw=0.6)
  for s in ("top", "right"):
    ax.spines[s].set_visible(False)
  ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9)
  ax.set_title(
    "Walk, stand->crawl, 60 s still in the crawl, crawl, crawl->stand, walk",
    fontsize=10,
    loc="left",
  )
  fig.tight_layout()
  fig.savefig(args.out)
  for label, log, tw in runs:
    peak = tw.max(axis=0)
    top = np.argsort(peak)[::-1][:4]
    print(label, ", ".join(f"{log['joints'][i]} {peak[i]:.1f}" for i in top))


if __name__ == "__main__":
  main()
