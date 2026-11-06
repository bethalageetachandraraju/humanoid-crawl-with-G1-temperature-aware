#!/usr/bin/env python3
"""Plot training scalars from an mjlab/RSL-RL TensorBoard run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _series(accumulator: EventAccumulator, tag: str) -> tuple[list[int], list[float]]:
  values = accumulator.Scalars(tag)
  return [item.step for item in values], [item.value for item in values]


def _plot_tags(
  ax: plt.Axes,
  accumulator: EventAccumulator,
  tags: list[str],
  *,
  label_prefix: str = "",
  legend: bool = True,
) -> None:
  plotted = False
  for tag in tags:
    steps, values = _series(accumulator, tag)
    if not values:
      continue
    label = tag.removeprefix(label_prefix) if label_prefix else tag
    ax.plot(steps, values, linewidth=1.5, label=label)
    plotted = True
  if not plotted:
    ax.text(
      0.5, 0.5, "No scalar data", ha="center", va="center", transform=ax.transAxes
    )
  if plotted and legend:
    ax.legend(fontsize=8, loc="best")
  ax.grid(True, alpha=0.25)
  ax.set_xlabel("Training iteration")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("run_dir", type=Path, help="TensorBoard run directory")
  parser.add_argument(
    "--output",
    type=Path,
    help="PNG output path (default: <run_dir>/training_curves.png)",
  )
  args = parser.parse_args()

  run_dir = args.run_dir.expanduser().resolve()
  if not run_dir.is_dir():
    parser.error(f"run directory does not exist: {run_dir}")
  accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
  accumulator.Reload()
  scalar_tags = accumulator.Tags().get("scalars", [])
  if not scalar_tags:
    parser.error(f"no TensorBoard scalar data found in {run_dir}")

  reward_tags = sorted(tag for tag in scalar_tags if tag.startswith("Episode_Reward/"))
  max_step = max(
    (event.step for tag in scalar_tags for event in accumulator.Scalars(tag)), default=0
  )
  fig = plt.figure(figsize=(17, 19), layout="constrained")
  outer = fig.add_gridspec(4, 1, height_ratios=(1, 2.2, 1.2, 1.2))
  fig.suptitle(
    f"Training curves — {run_dir.name} ({max_step + 1} logged iterations)", fontsize=16
  )

  summary = outer[0].subgridspec(1, 2, wspace=0.15)
  ax = fig.add_subplot(summary[0, 0])
  _plot_tags(ax, accumulator, ["Train/mean_reward"])
  ax.set_title("Mean episode reward")
  ax.set_ylabel("Reward")
  ax = fig.add_subplot(summary[0, 1])
  _plot_tags(ax, accumulator, ["Train/mean_episode_length"])
  ax.set_title("Mean episode length")
  ax.set_ylabel("Steps")

  component_rows = max(3, -(-len(reward_tags) // 5))
  components = outer[1].subgridspec(component_rows, 5, hspace=0.4, wspace=0.28)
  for index, tag in enumerate(reward_tags):
    ax = fig.add_subplot(components[index // 5, index % 5])
    _plot_tags(ax, accumulator, [tag], legend=False)
    ax.set_title(tag.removeprefix("Episode_Reward/"), fontsize=9)
    ax.set_ylabel("Reward", fontsize=8)
    ax.tick_params(labelsize=7)
  for index in range(len(reward_tags), component_rows * 5):
    fig.add_subplot(components[index // 5, index % 5]).set_visible(False)

  diagnostics = outer[2].subgridspec(1, 2, wspace=0.15)
  ax = fig.add_subplot(diagnostics[0, 0])
  _plot_tags(ax, accumulator, [tag for tag in scalar_tags if tag.startswith("Loss/")])
  ax.set_title("Optimization losses")
  ax.set_ylabel("Loss")
  ax = fig.add_subplot(diagnostics[0, 1])
  _plot_tags(
    ax, accumulator, [tag for tag in scalar_tags if tag.startswith("Metrics/")]
  )
  ax.set_title("Tracked metrics")
  ax.set_ylabel("Error")

  performance = outer[3].subgridspec(1, 2, wspace=0.15)
  ax = fig.add_subplot(performance[0, 0])
  _plot_tags(ax, accumulator, [tag for tag in scalar_tags if tag.startswith("Perf/")])
  ax.set_title("Training performance")
  ax.set_ylabel("Value")
  ax = fig.add_subplot(performance[0, 1])
  _plot_tags(
    ax,
    accumulator,
    [tag for tag in scalar_tags if tag.startswith("Episode_Termination/")],
  )
  ax.set_title("Episode terminations")
  ax.set_ylabel("Fraction")

  output = (
    args.output.expanduser().resolve()
    if args.output
    else run_dir / "training_curves.png"
  )
  output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(output, dpi=150)
  print(f"Saved {output}")
  print(
    f"Read {len(scalar_tags)} scalar tags through iteration {max_step} from {run_dir}"
  )


if __name__ == "__main__":
  main()
