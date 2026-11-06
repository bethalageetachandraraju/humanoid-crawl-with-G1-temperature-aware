"""Export a checkpoint's actor to ONNX with MJLab's deployment metadata.

The velocity-runner tasks (walk, stand-to-crawl) already export every saved
checkpoint next to it; the crawl tasks use the plain runner and do not. This
does it for any task::

  uv run python scripts/export_policy_onnx.py --task Mjlab-Crawl-Flat-Cmd7-Unitree-G1-29dof \
      --checkpoint logs/rsl_rl/g1_crawl_29dof/2026-10-12_17-55-24_gen2_flat_16k/model_49999.pt

writes ``<checkpoint dir>/exported/<checkpoint stem>.onnx``: input ``obs``
(1, D) raw observation (normalizer inside if the task trained with one), output
``actions`` (1, 25); metadata lists the joint order, PD gains, default joint
positions, observation term names and the action scale.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--task", required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--out", default=None, help="output .onnx path")
  args = ap.parse_args()
  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, asdict(agent_cfg), device=device)
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
  )
  checkpoint = Path(args.checkpoint)
  out = (
    Path(args.out)
    if args.out
    else checkpoint.parent / "exported" / f"{checkpoint.stem}.onnx"
  )
  out.parent.mkdir(parents=True, exist_ok=True)
  runner.export_policy_to_onnx(str(out.parent), out.name)
  attach_metadata_to_onnx(str(out), get_base_metadata(env, "local"))
  print(f"wrote {out}")
  env.close()


if __name__ == "__main__":
  main()
