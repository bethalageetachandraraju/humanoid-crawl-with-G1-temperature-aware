"""One-page training dashboard for the 29-DoF crawl run, from its TensorBoard log.

Writes two PNGs next to the run (or to ``--out``):

* ``dashboard.png`` -- grouped panels with plain-language titles and the limits
  drawn in: learning progress and the robustness curriculum, command tracking,
  gait (is it crawling on four limbs?), gait elegance, torque safety, effort and
  PPO health, and the disturbances the curriculum is currently applying.
* ``reward_terms.png`` -- every reward term on its own panel.

Curves are lightly smoothed (raw values faint behind). ``--watch 120`` redraws
every two minutes while training runs.

Usage::

  uv run python scripts/plot_crawl_dashboard.py                  # latest run
  uv run python scripts/plot_crawl_dashboard.py logs/rsl_rl/g1_crawl_29dof/<run>
  uv run python scripts/plot_crawl_dashboard.py --watch 120
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
  EventAccumulator,
)

EXPERIMENT = Path("logs/rsl_rl/g1_crawl_29dof")
M = "Episode_Metrics/"
C = "Curriculum/robustness/"
R = "Episode_Reward/"
GOOD, WARN, BAD = "#2e9e4f", "#f0a020", "#d33a2c"

# (title, [(tag, label)], y label, reference lines [(y, colour, label)], band)
Panel = tuple[str, list[tuple[str, str]], str, list[tuple[float, str, str]], tuple]
SECTIONS: list[tuple[str, list[Panel]]] = [
  (
    "Learning progress",
    [
      ("Episode reward", [("Train/mean_reward", "mean")], "reward", [], ()),
      (
        "Episode length (1000 = full 20 s)",
        [("Train/mean_episode_length", "mean")],
        "steps",
        [(1000, GOOD, "full")],
        (),
      ),
      (
        "Robustness level (0 = mild, 1 = full)",
        [(C + "level", "level")],
        "level",
        [],
        (),
      ),
      (
        "Curriculum gate: crawled / commanded",
        [(C + "gate_progress_ratio", "progress")],
        "ratio",
        [(0.75, GOOD, "level up"), (0.45, BAD, "level down")],
        (),
      ),
      (
        "Episodes ending in a roll-over",
        [("Episode_Termination/flipped", "flipped")],
        "fraction",
        [(0.05, GOOD, "level-up max"), (0.25, BAD, "level down")],
        (),
      ),
    ],
  ),
  (
    "Command tracking",
    [
      (
        "Forward speed actually achieved",
        [(M + "track/fwd_speed_mps", "actual")],
        "m/s",
        [],
        (),
      ),
      (
        "Forward speed error",
        [(M + "track/fwd_speed_error_mps", "|cmd - actual|")],
        "m/s",
        [],
        (),
      ),
      (
        "Turn-rate error",
        [(M + "track/turn_rate_error_radps", "|cmd - actual|")],
        "rad/s",
        [],
        (),
      ),
      (
        "Sideways speed error (7-command runs)",
        [(M + "track/lateral_speed_error_mps", "|cmd - actual|")],
        "m/s",
        [],
        (),
      ),
      (
        "Heading drift while holding straight",
        [("Metrics/base_velocity/error_heading", "|heading error|")],
        "rad (episode mean)",
        [],
        (),
      ),
    ],
  ),
  (
    "Gait: is it crawling on four limbs?",
    [
      (
        "Time each limb pair is on the ground",
        [
          (M + "gait/hands_on_ground_frac", "hands"),
          (M + "gait/feet_on_ground_frac", "feet"),
        ],
        "fraction",
        [],
        (),
      ),
      (
        "Share of body weight on the hands",
        [(M + "gait/hand_load_share", "hands")],
        "fraction",
        [],
        (0.40, 0.55, "healthy crawl"),
      ),
      (
        "Pelvis height",
        [(M + "gait/body_height_m", "height")],
        "m",
        [(0.22, GOOD, "crawl pose")],
        (),
      ),
      (
        "Prone alignment (1 = belly down)",
        [(M + "gait/prone_alignment", "alignment")],
        "gravity along base +X",
        [],
        (),
      ),
      (
        "Unwanted ground contact",
        [
          (M + "gait/torso_on_ground_frac", "torso/pelvis (time)"),
          (M + "gait/other_bodies_touching", "other bodies (count)"),
        ],
        "",
        [],
        (),
      ),
    ],
  ),
  (
    "Gait elegance",
    [
      (
        "Diagonal limbs in step (1 = trot)",
        [(M + "gait/diagonal_sync", "sync")],
        "score",
        [(0.48, WARN, "reference crawl")],
        (),
      ),
      (
        "Height of moving hands/feet",
        [
          (M + "gait/swing_height_m", "actual"),
          (M + "gait/step_height_cmd_m", "commanded"),
        ],
        "m",
        [(0.05, GOOD, "3-command target 5 cm")],
        (),
      ),
      (
        "Swing duration of limbs in the air",
        [("Metrics/air_time_mean", "mean")],
        "s",
        [],
        (0.10, 0.40, "rewarded range"),
      ),
      (
        "Touchdown impact force",
        [("Metrics/landing_force_mean", "mean")],
        "N",
        [],
        (),
      ),
      (
        "Action jerk (lower = smoother)",
        [(M + "effort/action_jerk", "mean")],
        "|second difference|",
        [],
        (),
      ),
    ],
  ),
  (
    "Torque safety",
    [
      (
        "Braced wrist: peak moment per episode",
        [(M + "safety/braced_wrist_peak_nm", "peak")],
        "Nm",
        [(5.0, BAD, "5 Nm limit"), (4.0, WARN, "penalty starts")],
        (),
      ),
      (
        "Braced wrist: time over 5 Nm",
        [(M + "safety/braced_wrist_over_limit_frac", "over limit")],
        "fraction of time",
        [],
        (),
      ),
      (
        "Motors: peak demand / limit",
        [
          (M + "safety/leg_motor_peak_ratio", "legs"),
          (M + "safety/waist_motor_peak_ratio", "waist"),
          (M + "safety/arm_motor_peak_ratio", "arms"),
        ],
        "x motor peak",
        [(1.0, BAD, "limit"), (0.8, WARN, "penalty starts")],
        (),
      ),
      (
        "Motors: time any motor saturated",
        [(M + "safety/motor_saturated_frac", "saturated")],
        "fraction of time",
        [],
        (),
      ),
      (
        "Wrist roll peak torque",
        [(M + "safety/wrist_roll_peak_nm", "peak")],
        "Nm",
        [(25.0, BAD, "25 Nm limit"), (20.0, WARN, "80%")],
        (),
      ),
    ],
  ),
  (
    "Effort and learning health",
    [
      (
        "Mechanical power",
        [(M + "effort/mech_power_w", "sum |tau x qdot|")],
        "W",
        [],
        (),
      ),
      (
        "Joints near a limit",
        [(M + "safety/joint_near_limit_frac", "any joint")],
        "fraction of time",
        [],
        (),
      ),
      (
        "Learning rate (1e-5 = frozen)",
        [("Loss/learning_rate", "lr")],
        "lr (log)",
        [(1e-5, BAD, "floor")],
        (1e-4, 3e-3, "healthy"),
      ),
      (
        "Action noise std",
        [("Policy/mean_std", "std")],
        "std",
        [],
        (),
      ),
      (
        "Losses",
        [
          ("Loss/value", "value"),
          ("Loss/surrogate", "surrogate"),
          ("Loss/symmetry", "mirror"),
        ],
        "loss",
        [],
        (),
      ),
    ],
  ),
  (
    "Disturbances the curriculum is applying",
    [
      ("Pushes", [(C + "push_mps", "kick")], "m/s", [], ()),
      (
        "Friction and payload",
        [(C + "friction_min", "friction min"), (C + "payload_max_kg", "payload kg")],
        "",
        [],
        (),
      ),
      (
        "Motors: delay and strength",
        [(C + "delay_max_ms", "delay ms"), (C + "motor_strength_min", "strength min")],
        "",
        [],
        (),
      ),
      (
        "Sensor noise scale",
        [(C + "obs_noise_scale", "x base noise")],
        "scale",
        [],
        (),
      ),
      (
        "Torque-limit penalty weights (0 = off)",
        [
          (C + "motor_penalty_weight", "motors"),
          (C + "braced_penalty_weight", "braced wrists"),
        ],
        "weight",
        [],
        (),
      ),
    ],
  ),
]


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
  if len(values) < 3 or window <= 1:
    return values
  window = min(window, len(values))
  kernel = np.ones(window) / window
  padded = np.concatenate([np.full(window - 1, values[0]), values])
  return np.convolve(padded, kernel, mode="valid")


def _series(acc: EventAccumulator, tag: str) -> tuple[np.ndarray, np.ndarray]:
  if tag not in acc.Tags().get("scalars", []):
    return np.array([]), np.array([])
  events = acc.Scalars(tag)
  return np.array([e.step for e in events]), np.array([e.value for e in events])


def _panel(ax, acc: EventAccumulator, panel: Panel, smooth: int) -> None:
  title, curves, ylabel, lines, band = panel
  drawn = False
  for i, (tag, label) in enumerate(curves):
    x, y = _series(acc, tag)
    if len(y) == 0:
      continue
    colour = f"C{i}"
    ax.plot(x, y, color=colour, alpha=0.2, lw=0.8)
    ax.plot(x, _smooth(y, smooth), color=colour, lw=1.6, label=f"{label}: {y[-1]:.3g}")
    drawn = True
  for y, colour, label in lines:
    ax.axhline(y, color=colour, ls="--", lw=1.0, label=label)
  if band:
    ax.axhspan(band[0], band[1], color=GOOD, alpha=0.12, label=band[2])
  if "log" in ylabel:
    ax.set_yscale("log")
  if not drawn:
    ax.text(0.5, 0.5, "no data yet", ha="center", va="center", transform=ax.transAxes)
  ax.set_title(title, fontsize=9.5)
  ax.set_ylabel(ylabel, fontsize=8)
  ax.tick_params(labelsize=7)
  ax.grid(alpha=0.25)
  if drawn or lines:
    ax.legend(fontsize=6.5, loc="best")


def draw(run_dir: Path, out: Path, smooth: int) -> None:
  acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  acc.Reload()
  tags = acc.Tags().get("scalars", [])
  if not tags:
    print(f"[dashboard] no scalars in {run_dir} yet")
    return
  last = max(acc.Scalars("Train/mean_reward")[-1].step, 0)

  rows, cols = len(SECTIONS), 5
  fig, axes = plt.subplots(rows, cols, figsize=(26, 4.0 * rows), squeeze=False)
  fig.suptitle(f"{run_dir.name} -- iteration {last}", fontsize=15, y=0.995)
  for r, (section, panels) in enumerate(SECTIONS):
    for c in range(cols):
      ax = axes[r, c]
      if c < len(panels):
        _panel(ax, acc, panels[c], smooth)
      else:
        ax.set_visible(False)
    axes[r, 0].annotate(
      section,
      xy=(-0.28, 0.5),
      xycoords="axes fraction",
      rotation=90,
      va="center",
      ha="center",
      fontsize=12,
      fontweight="bold",
    )
  for ax in axes[-1]:
    ax.set_xlabel("training iteration", fontsize=8)
  fig.tight_layout(rect=(0.02, 0, 1, 0.985))
  out.mkdir(parents=True, exist_ok=True)
  fig.savefig(out / "dashboard.png", dpi=90)
  plt.close(fig)

  rewards = sorted(t for t in tags if t.startswith(R))
  cols = 6
  rows = math.ceil(len(rewards) / cols)
  fig, axes = plt.subplots(rows, cols, figsize=(26, 3.2 * rows), squeeze=False)
  fig.suptitle(
    f"{run_dir.name} -- reward terms (per second of episode), iteration {last}",
    fontsize=14,
  )
  for i, ax in enumerate(axes.flat):
    if i >= len(rewards):
      ax.set_visible(False)
      continue
    _panel(
      ax, acc, (rewards[i].removeprefix(R), [(rewards[i], "")], "", [], ()), smooth
    )
  fig.tight_layout(rect=(0, 0, 1, 0.97))
  fig.savefig(out / "reward_terms.png", dpi=80)
  plt.close(fig)
  print(
    f"[dashboard] iteration {last}: wrote {out / 'dashboard.png'} and reward_terms.png"
  )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_dir", type=Path, nargs="?", help="Default: latest run.")
  parser.add_argument("--out", type=Path, default=None, help="Default: the run dir.")
  parser.add_argument("--smooth", type=int, default=25, help="Moving average window.")
  parser.add_argument("--watch", type=float, default=0.0, help="Redraw every N s.")
  args = parser.parse_args()

  run_dir = args.run_dir
  if run_dir is None:
    runs = sorted(p for p in EXPERIMENT.iterdir() if p.is_dir())
    if not runs:
      parser.error(f"no runs in {EXPERIMENT}")
    run_dir = runs[-1]
  out = args.out or run_dir
  while True:
    draw(run_dir, out, args.smooth)
    if args.watch <= 0:
      break
    time.sleep(args.watch)


if __name__ == "__main__":
  main()
