"""Train while watching every environment live in the Viser viewer.

mjlab's ``train`` command has no live viewer (see the docs FAQ). This script
runs the normal RSL-RL training loop in a background thread and streams the
simulation state of all environments to a Viser web page at a fixed rate. It is
meant for small environment counts (around 100) to eyeball what the policy is
doing while it learns; use ``uv run train`` for the real run.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import viser

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.sim import Simulation
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.viser.scene import MjlabViserScene


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--task", default="Mjlab-Crawl-Flat-Unitree-G1")
  ap.add_argument("--num-envs", type=int, default=100)
  ap.add_argument("--max-iterations", type=int, default=3000)
  ap.add_argument("--save-interval", type=int, default=100)
  ap.add_argument("--port", type=int, default=8080)
  ap.add_argument("--fps", type=float, default=30.0, help="viewer refresh rate")
  ap.add_argument("--log-root", default="logs/rsl_rl")
  ap.add_argument(
    "--checkpoint", default=None, help="model_*.pt to resume from (optional)"
  )
  ap.add_argument(
    "--spread-terrain",
    action="store_true",
    help="terrain tasks: spawn envs over all difficulty levels and skip the flat "
    "warm-up, so every terrain type is visible at once",
  )
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0"
  env_cfg = load_env_cfg(args.task)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg.max_iterations = args.max_iterations
  agent_cfg.save_interval = args.save_interval
  agent_cfg.logger = "tensorboard"
  if args.spread_terrain:
    terrain = env_cfg.scene.terrain
    assert terrain is not None and terrain.terrain_generator is not None, (
      "--spread-terrain needs a task with a terrain generator"
    )
    terrain.max_init_terrain_level = None
    # Keep the random level assignment fixed: with the curriculum active, envs
    # that do not cover their commanded distance get demoted to flat quickly.
    env_cfg.curriculum = {}

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_viewer"
  log_dir = (Path(args.log_root) / agent_cfg.experiment_name).resolve() / stamp

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  agent_dict = asdict(agent_cfg)
  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", agent_dict)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, agent_dict, str(log_dir), device)
  if args.checkpoint is not None:
    print(f"[INFO] Resuming from {args.checkpoint}", flush=True)
    runner.load(args.checkpoint)

  sim = env.sim
  assert isinstance(sim, Simulation)
  server = viser.ViserServer(label="mjlab-train", port=args.port)
  scene = MjlabViserScene(
    server=server,
    mj_model=sim.mj_model,
    num_envs=env.num_envs,
    sim_model=sim.model,
    expanded_fields=sim.expanded_fields,
  )
  scene.debug_visualization_enabled = False
  status = server.gui.add_html("")

  worker = threading.Thread(
    target=runner.learn,
    kwargs={
      "num_learning_iterations": args.max_iterations,
      "init_at_random_ep_len": True,
    },
    daemon=True,
  )
  worker.start()
  print(f"[INFO] Live viewer: http://localhost:{args.port}", flush=True)
  print(f"[INFO] Logging experiment in directory: {log_dir}", flush=True)

  period = 1.0 / args.fps
  try:
    while worker.is_alive():
      t0 = time.perf_counter()
      try:
        with server.atomic():
          scene.update(sim.data)
          server.flush()
        status.content = (
          f"<b>iteration</b> {runner.current_learning_iteration}"
          f" / {args.max_iterations}<br/><b>envs</b> {env.num_envs}"
        )
      except Exception as exc:  # The viewer must never take training down.
        print(f"[WARN] viewer update failed: {exc!r}", flush=True)
      time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    worker.join()
    env.close()
    server.stop()
  except KeyboardInterrupt:
    # The training thread is a daemon; leave immediately rather than racing it.
    print("\n[INFO] Ctrl+C received, stopping.", flush=True)
    server.stop()
    os._exit(0)


if __name__ == "__main__":
  main()
