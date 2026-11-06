"""Drive a trained G1 crawl policy with the keyboard in the native viewer.

Keys (window must have focus):
  W / S   forward speed +0.25 / -0.25 m/s      (command lin-z, base frame)
  A / D   turn left / right by 0.25 rad/s, up to 3 (command ang-x, base frame)
  X       stop (zero command)
  H       cycle commanded step height 3 / 8 / 12 cm (7-value command tasks)
  Enter   reset episode, Space pause, +/- playback speed (viewer defaults)

The crawl task's turn reward compares the ang-x command against world yaw rate,
so if A/D feel reversed for a checkpoint, pass ``--invert-turn``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

import mjlab.tasks  # noqa: F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.g1_crawl.mdp import CrawlVelocityCommand
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import KEY_A, KEY_D, KEY_H, KEY_S, KEY_W, KEY_X

DEFAULT_TASK = "Mjlab-Crawl-Flat-Unitree-G1"


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", required=True, help="path to model_*.pt")
  ap.add_argument("--task", default=DEFAULT_TASK, help="registered crawl task id")
  ap.add_argument("--num-envs", type=int, default=1)
  ap.add_argument("--speed-step", type=float, default=0.25)
  ap.add_argument("--turn-step", type=float, default=0.25)
  ap.add_argument("--max-speed", type=float, default=2.5)
  ap.add_argument("--max-turn", type=float, default=3.0)
  ap.add_argument("--invert-turn", action="store_true")
  ap.add_argument(
    "--headless-steps",
    type=int,
    default=0,
    help="debug: step this many times without a window and exit",
  )
  args = ap.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env_cfg = load_env_cfg(args.task, play=True)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(wenv, asdict(agent_cfg), device=device)
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=device
  )
  policy = runner.get_inference_policy(device=device)

  # Take the command away from the random sampler: keyboard owns it from now on.
  cmd = env.command_manager.get_term("base_velocity")
  assert isinstance(cmd, CrawlVelocityCommand)
  cmd.manual_control()
  state = {"speed": 0.0, "turn": 0.0}
  step_state = {"idx": 0}
  step_heights = (0.03, 0.08, 0.12)
  has_step = cmd.vel_command_b.shape[1] >= 4
  sign = -1.0 if args.invert_turn else 1.0

  def apply() -> None:
    cmd.vel_command_b[:, 0] = state["speed"]
    cmd.vel_command_b[:, 1] = 0.0
    cmd.vel_command_b[:, 2] = sign * state["turn"]
    step_txt = ""
    if has_step:
      cmd.vel_command_b[:, 3] = step_heights[step_state["idx"]]
      cmd.vel_command_b[:, 4:] = 0.0
      step_txt = f"   step height {step_heights[step_state['idx']]:.2f} m"
    print(
      f"\rcommand: forward {state['speed']:+.2f} m/s   turn {state['turn']:+.2f} rad/s"
      f"{step_txt}   ",
      end="",
      flush=True,
    )

  def on_key(key: int) -> None:
    if key == KEY_W:
      state["speed"] = min(args.max_speed, state["speed"] + args.speed_step)
    elif key == KEY_S:
      state["speed"] = max(-1.0, state["speed"] - args.speed_step)
    elif key == KEY_A:
      state["turn"] = min(args.max_turn, state["turn"] + args.turn_step)
    elif key == KEY_D:
      state["turn"] = max(-args.max_turn, state["turn"] - args.turn_step)
    elif key == KEY_X:
      state["speed"] = 0.0
      state["turn"] = 0.0
    elif key == KEY_H and has_step:
      step_state["idx"] = (step_state["idx"] + 1) % len(step_heights)
    else:
      return
    apply()

  apply()
  print("\n" + (__doc__ or ""))

  if args.headless_steps > 0:
    obs = wenv.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    on_key(KEY_W)
    with torch.inference_mode():
      for _ in range(args.headless_steps):
        obs = wenv.step(policy(obs))[0]
    print("\nheadless ok; command now", cmd.vel_command_b[0].tolist())
    env.close()
    return

  NativeMujocoViewer(wenv, policy, key_callback=on_key).run()
  env.close()


if __name__ == "__main__":
  main()
