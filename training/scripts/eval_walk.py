"""Score a walk checkpoint: standing, forward/back, sideways, turning.

Runs ``--num-envs`` robots through a fixed command sequence (deterministic
actor, no observation noise), with or without level-1 randomization, and
reports per command phase the velocity tracking error, the fall rate and the
peak motor and braced-wrist loads. Uses the switcher's environment and policy
wrapper, so what is scored is exactly what the deployment code computes.

  uv run python scripts/eval_walk.py                 # latest checkpoint, nominal
  uv run python scripts/eval_walk.py --robust        # level-1 randomization
  uv run python scripts/eval_walk.py --hard          # WALK_RAMP_HARD level 1, IMU tilt
  uv run python scripts/eval_walk.py --checkpoint logs/rsl_rl/g1_walk_29dof/<run>/model_N.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_chain import _Tilt, add_randomization  # noqa: E402
from play_switcher import (  # noqa: E402
  WALK_TASK,
  _WalkPolicy,
  latest_checkpoint,
  load_actor,
  make_env_cfg,
)

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import BRACED_LOADS_29, MOTORS_29  # noqa: E402
from mjlab.tasks.g1_walk.config.g1.env_cfgs import IMU_TILT_MAX_DEG  # noqa: E402
from mjlab.tasks.g1_walk.mdp import WALK_RAMP_HARD  # noqa: E402
from mjlab.tasks.g1_walk.robot import JOINTS_29  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402

# (name, vx, vy, wz, seconds)
PHASES = (
  ("stand", 0.0, 0.0, 0.0, 4.0),
  ("forward 0.5", 0.5, 0.0, 0.0, 5.0),
  ("forward 1.0", 1.0, 0.0, 0.0, 5.0),
  ("backward 0.4", -0.4, 0.0, 0.0, 5.0),
  ("left 0.4", 0.0, 0.4, 0.0, 5.0),
  ("right 0.4", 0.0, -0.4, 0.0, 5.0),
  ("turn left 0.6", 0.0, 0.0, 0.6, 5.0),
  ("turn right 0.6", 0.0, 0.0, -0.6, 5.0),
  ("fwd + turn", 0.5, 0.0, 0.5, 5.0),
  ("stand again", 0.0, 0.0, 0.0, 4.0),
)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--checkpoint", default=None)
  ap.add_argument("--num-envs", type=int, default=256)
  ap.add_argument("--robust", action="store_true")
  ap.add_argument(
    "--hard", action="store_true", help="hard-ramp level 1 + IMU tilt + arm mass"
  )
  ap.add_argument("--push", type=float, default=0.0, help="kick (m/s) once per phase")
  ap.add_argument("--seed", type=int, default=0, help="same robots across runs")
  args = ap.parse_args()
  checkpoint = args.checkpoint or latest_checkpoint("g1_walk_29dof")
  assert checkpoint, "no walk checkpoint"
  print(f"checkpoint: {checkpoint}")

  configure_torch_backends()
  torch.manual_seed(args.seed)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  n = args.num_envs
  cfg = make_env_cfg(n, args.robust and not args.hard, no_delay=False)
  if args.hard:
    add_randomization(cfg, WALK_RAMP_HARD, arm_mass=True)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  robot = env.scene["robot"]
  assert tuple(robot.joint_names) == JOINTS_29
  actor, _ = load_actor(WALK_TASK, checkpoint, device)
  policy = _WalkPolicy(n, device, actor)
  if args.hard:
    tilt = _Tilt(n, device, IMU_TILT_MAX_DEG)

    def tilted_base(robot, t=tilt):
      return t(robot.data.projected_gravity_b), robot.data.root_link_ang_vel_b

    setattr(policy, "base", tilted_base)  # noqa: B010
  env.reset()
  dt = env.step_dt
  kp_kd_peak = torch.tensor([MOTORS_29[j] for j in JOINTS_29], device=device)
  kp, kd, peak = kp_kd_peak[:, 0], kp_kd_peak[:, 1], kp_kd_peak[:, 2]
  braced = [
    (
      env.scene[f"robot/{j.removesuffix('_joint')}_torque"],
      torch.tensor(a, device=device),
    )
    for j, (_, a) in BRACED_LOADS_29.items()
  ]
  fallen = torch.zeros(n, dtype=torch.bool, device=device)
  rows = []
  print(
    f"\n{'phase':16s} {'|v err| m/s':>12s} {'|w err| rad/s':>14s} {'speed m/s':>10s} {'falls':>6s} {'motor pk':>9s} {'braced pk Nm':>12s} {'power W':>8s}"
  )
  with torch.inference_mode():
    for name, vx, vy, wz, seconds in PHASES:
      policy.command[:, 0], policy.command[:, 1], policy.command[:, 2] = vx, vy, wz
      steps = int(seconds / dt)
      err_v = torch.zeros(n, device=device)
      err_w = torch.zeros(n, device=device)
      speed = torch.zeros(n, device=device)
      power = torch.zeros(n, device=device)
      pk_motor = torch.zeros(n, device=device)
      pk_braced = torch.zeros(n, device=device)
      counted = 0
      for step in range(steps):
        if args.push and step == steps // 2:
          vel = robot.data.root_link_vel_w.clone()
          kick = torch.randn(n, 2, device=device)
          vel[:, :2] += (
            kick / kick.norm(dim=1, keepdim=True).clamp(min=1e-6) * args.push
          )
          robot.write_root_link_velocity_to_sim(vel)
        target = policy.targets(robot)
        env.step(target)
        d = robot.data
        g = d.projected_gravity_b
        fallen |= (g[:, 2] > -0.3) | (d.root_link_pos_w[:, 2] < 0.3)
        demand = kp * (target - d.joint_pos) - kd * d.joint_vel
        pk_motor = torch.maximum(pk_motor, (demand.abs() / peak).amax(dim=1))
        for sensor, axis in braced:
          pk_braced = torch.maximum(pk_braced, (sensor.data @ axis).abs())
        if step >= int(0.4 * steps):  # Settled part of the phase.
          v = d.root_link_lin_vel_b
          err_v += (v[:, :2] - policy.command[:, :2]).norm(dim=1)
          err_w += (d.root_link_ang_vel_b[:, 2] - policy.command[:, 2]).abs()
          speed += v[:, :2].norm(dim=1)
          power += (d.actuator_force * d.joint_vel).abs().sum(dim=1)
          counted += 1
      ok = ~fallen

      def m(x: torch.Tensor, ok=ok, counted=counted) -> float:
        return float(x[ok].mean() / counted) if ok.any() else float("nan")

      rows.append(
        (
          name,
          m(err_v),
          m(err_w),
          m(speed),
          float(fallen.float().mean()),
          float(pk_motor[ok].median()) if ok.any() else float("nan"),
          float(pk_braced[ok].quantile(0.95)) if ok.any() else float("nan"),
          m(power),
        )
      )
      r = rows[-1]
      print(
        f"{r[0]:16s} {r[1]:12.3f} {r[2]:14.3f} {r[3]:10.2f} {100 * r[4]:5.1f}% {r[5]:9.2f} {r[6]:12.1f} {r[7]:8.0f}"
      )
  print(
    f"\n{n} robots, {'hard randomization + IMU tilt' if args.hard else ('level-1 randomization' if args.robust else 'nominal robot')}"
    f"{f', {args.push} m/s kick each phase' if args.push else ''}; falls are cumulative; "
    "errors over the settled 60% of each phase, fallen robots excluded; motor peak = median over robots "
    "of the busiest motor's demand/limit; braced = p95 of the peak wrist pitch/yaw moment (limit 5 Nm)."
  )
  env.close()


if __name__ == "__main__":
  main()
