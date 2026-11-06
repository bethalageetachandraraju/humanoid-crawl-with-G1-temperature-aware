"""Motor load of a crawl or stand <-> crawl policy: the thermal fine-tune's checks.

Runs a checkpoint in the thermal fine-tune task (the robot's PD-target clamp,
floor friction, the heat terms at full weight) and reports, per scenario, each
motor's RMS torque in Nm and as a fraction of its estimated continuous rating,
the share of robots whose busiest motor stays inside the 0.6 budget, falls, and
the braced-wrist moment. Old and new checkpoints run in the same task, so the
comparison is the robot's conditions for both.

Scenarios:
  crawl rest      zero command (robots start from the crawl reset), --seconds
  crawl move      straight ahead at --speed (heading hold), --seconds
  crawl train     the training distribution: per-term reward per 20 s episode
  hold            stand <-> crawl: descend, hold phi = 1 for --seconds, stand up;
                  also the deploy gate (|g - g_crawl| < 0.15, max |qd| < 0.5,
                  pelvis over limbs 0.151 +- 0.03, for 1 s) and knee peak.
  hold_stuck      as hold, but every robot starts in the crawl pose, 40 % of
                  them nose-down with the hips high (the posture the real robot
                  got stuck in)
  hold_train      the stand <-> crawl training distribution, per-term reward

``--robust`` uses the training randomization at level 1 (pushes, mass, motor
strength, floor friction); otherwise the nominal robot on a nominal floor.
``--friction LO HI`` fixes the floor friction range (e.g. 0.35 0.45 for tiles).

  uv run python scripts/thermal_eval.py --checkpoint <crawl.pt> --scenario rest
  uv run python scripts/thermal_eval.py --checkpoint <stand_crawl.pt> \
      --scenario hold --robust
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_switcher import load_actor  # noqa: E402

import mjlab.tasks  # noqa: E402, F401  (registers the tasks)
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionActionCfg  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.g1_crawl import mdp as crawl_mdp  # noqa: E402
from mjlab.tasks.g1_crawl.config.g1.env_cfgs import (  # noqa: E402
  BRACED_AXES,
  BRACED_TORQUE_SENSORS,
  HEAT_BUDGET,
)
from mjlab.tasks.g1_crawl.robot29 import (  # noqa: E402
  CRAWL_JOINTS_29,
  deploy_target_clip,
)
from mjlab.tasks.g1_walk.mdp import TransitionPhaseCommandCfg  # noqa: E402
from mjlab.tasks.registry import load_env_cfg  # noqa: E402

CRAWL_TASK = "Mjlab-Crawl-Flat-Cmd7-Cool-Unitree-G1-29dof"
WALK_TASK = "Mjlab-Walk-Flat-Unitree-G1-29dof"
STAND_CRAWL_TASK = "Mjlab-StandCrawl-Cool-Unitree-G1-29dof"
SHORT = [
  j.replace("_joint", "").replace("left_", "L_").replace("right_", "R_")
  for j in CRAWL_JOINTS_29
]
CRAWL_ABOVE_LIMBS = 0.151
LIMB_BODIES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
  "left_forearm_extension",
  "right_forearm_extension",
)


def build_env(args) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper, str]:
  crawl = args.scenario in ("rest", "move", "train")
  walk = args.scenario == "walk"
  task = CRAWL_TASK if crawl else WALK_TASK if walk else STAND_CRAWL_TASK
  task = args.task or task
  train_dist = args.scenario in ("train", "hold_train")
  cfg = load_env_cfg(task, play=not (args.robust or train_dist))
  cfg.scene.num_envs = args.num_envs
  cfg.curriculum.pop("thermal_ramp", None)  # Heat terms at full weight.
  if crawl:
    cmd = cfg.commands["base_velocity"]
    assert isinstance(cmd, crawl_mdp.CrawlVelocityCommandCfg)
    if args.scenario == "rest":
      cmd.rel_standing_envs = 1.0
      cmd.standing_time_range = None
      cmd.resampling_time_range = (1e4, 1e4)
    elif args.scenario == "move":
      cmd.lin_vel_z = (args.speed, args.speed)
      cmd.straight_fwd_range = None  # Exactly --speed, not a task's cruise band.
      cmd.rel_standing_envs = 0.0
      cmd.rel_straight_envs = 1.0
      cmd.resampling_time_range = (1e4, 1e4)
    if args.scenario != "train":
      cfg.episode_length_s = args.seconds + 1.0
  elif walk:
    twist = cast(Any, cfg.commands["twist"])
    twist.ranges.lin_vel_x = (args.speed, args.speed)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_standing_envs = 1.0 if args.speed == 0.0 else 0.0
    for name in ("rel_heading_envs", "rel_forward_envs", "rel_spin_envs"):
      setattr(twist, name, 0.0)
    twist.resampling_time_range = (1e4, 1e4)
    cfg.episode_length_s = args.seconds + 1.0
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.clip = deploy_target_clip()  # As the robot controller does.
  else:
    cmd = cfg.commands["phase"]
    assert isinstance(cmd, TransitionPhaseCommandCfg)
    if args.scenario in ("hold", "hold_stuck"):
      cmd.hold_s = (args.seconds, args.seconds)
      cmd.duration_s = (8.0, 8.0)
      cmd.up_s = (8.0, 8.0)
      cmd.stand_up_prob = 1.0
      start = cfg.events.get("reset_crawl_start")
      if start is not None:
        stuck = args.scenario == "hold_stuck"
        start.params["prob"] = 1.0 if stuck else 0.0
        start.params["gait_frac"] = 0.0
        start.params["nose_down_frac"] = 0.4 if stuck else 0.0
      cfg.episode_length_s = 1.5 + 8.0 + args.seconds + 8.0 + 2.0
  if args.friction is not None and "dr_friction" in cfg.events:
    cfg.events["dr_friction"].params["ranges"] = tuple(args.friction)
    robust = cfg.curriculum.get("robustness")
    if robust is not None:
      lo, hi = args.friction
      from dataclasses import replace

      robust.params["ramp"] = replace(
        robust.params["ramp"], friction_low=(lo, lo), friction_high=(hi, hi)
      )
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wenv = RslRlVecEnvWrapper(env, clip_actions=None)
  return env, wenv, task


def deploy_gate(env: ManagerBasedRlEnv, cmd) -> torch.Tensor:
  """The robot controller's HOLD -> CRAWL gate, from what the robot measures."""
  robot = env.scene["robot"]
  g = robot.data.projected_gravity_b
  tilt_ok = (g - cmd.schedule.g_crawl).norm(dim=1) < 0.15
  still = robot.data.joint_vel.abs().max(dim=1).values < 0.5
  ids, _ = robot.find_bodies(LIMB_BODIES, preserve_order=True)
  limbs_z = robot.data.body_link_pos_w[:, ids, 2].mean(dim=1)
  above = robot.data.root_link_pos_w[:, 2] - limbs_z
  height_ok = (above - CRAWL_ABOVE_LIMBS).abs() < 0.03
  return tilt_ok & still & height_ok


def main() -> None:
  ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument(
    "--scenario",
    choices=("rest", "move", "train", "hold", "hold_stuck", "hold_train", "walk"),
    default="rest",
  )
  ap.add_argument("--seconds", type=float, default=60.0)
  ap.add_argument("--speed", type=float, default=0.3)
  ap.add_argument("--num-envs", type=int, default=512)
  ap.add_argument("--robust", action="store_true")
  ap.add_argument("--friction", type=float, nargs=2, default=None)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--json", type=Path, default=None, help="Also write the results.")
  ap.add_argument("--task", default=None, help="Override the scenario's task.")
  args = ap.parse_args()

  env, wenv, task = build_env(args)
  policy, _ = load_actor(task, args.checkpoint, args.device)
  robot = env.scene["robot"]
  act_ids, _ = robot.find_actuators(CRAWL_JOINTS_29, preserve_order=True)
  act_ids = torch.tensor(act_ids, device=env.device)
  joint_ids, _ = robot.find_joints(CRAWL_JOINTS_29, preserve_order=True)
  knee_ids = [CRAWL_JOINTS_29.index(j) for j in CRAWL_JOINTS_29 if "knee" in j]
  ratings = crawl_mdp.continuous_ratings(CRAWL_JOINTS_29)
  tau_cont = torch.tensor([ratings[j] for j in CRAWL_JOINTS_29], device=env.device)
  n = env.num_envs
  crawl = task.startswith("Mjlab-Crawl")
  walk = task.startswith("Mjlab-Walk")
  cmd_name = "base_velocity" if crawl else "twist" if walk else "phase"
  cmd = cast(Any, env.command_manager.get_term(cmd_name))
  action_term = cast(Any, env.action_manager.get_term("joint_pos"))
  clip = action_term._clip  # (n, 25, 2): the robot's target clamp.

  steps = int(round(env.max_episode_length_s / env.step_dt)) - 1
  if args.scenario in ("train", "hold_train"):
    steps = int(round(env.max_episode_length_s / env.step_dt)) * 2
  sq = torch.zeros(n, 25, device=env.device)
  count = torch.zeros(n, device=env.device)
  sq_tr = torch.zeros(n, 25, device=env.device)
  count_tr = torch.zeros(n, device=env.device)
  beyond = torch.zeros(n, 25, device=env.device)
  braced_moments: list[torch.Tensor] = []
  speed_sum = torch.zeros(n, device=env.device)
  fell = torch.zeros(n, dtype=torch.bool, device=env.device)
  done_once = torch.zeros(n, dtype=torch.bool, device=env.device)
  knee_peak = torch.zeros(n, device=env.device)
  gate_time = torch.full((n,), float("nan"), device=env.device)
  gate_run = torch.zeros(n, device=env.device)
  hold_t = torch.zeros(n, device=env.device)
  term_names = env.reward_manager.active_terms
  term_sum = torch.zeros(len(term_names), device=env.device)
  episodes = 0.0

  obs = wenv.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      action = policy(obs["actor"])
      obs, _, dones, _ = wenv.step(action)
      tau = robot.data.actuator_force[:, act_ids]
      if crawl or walk:
        live = ~done_once
      else:
        live = cmd.hold_crawl & ~done_once
      if args.scenario in ("train", "hold_train"):
        live = torch.ones_like(live)
        term_sum += env.reward_manager._step_reward.sum(dim=0) * env.step_dt
        episodes += float(dones.sum())
      sq[live] += tau[live].square()
      count[live] += 1
      if not (crawl or walk):  # Going down or standing up (phase moving).
        moving = (cmd.rate != 0.0) & ~done_once
        sq_tr[moving] += tau[moving].square()
        count_tr[moving] += 1
      want = action_term.raw_action * action_term._scale + action_term._offset
      out = (want < clip[..., 0] - 1e-4) | (want > clip[..., 1] + 1e-4)
      beyond[live] += out[live].float()
      if live.any():
        braced_moments.append(
          crawl_mdp.braced_joint_moment(env, BRACED_TORQUE_SENSORS, BRACED_AXES)[live]
          .max(dim=1)
          .values
        )
      if crawl:
        speed_sum += robot.data.root_link_lin_vel_b[:, 2] * live
      elif walk:
        speed_sum += robot.data.root_link_lin_vel_b[:, 0] * live
      else:
        q = robot.data.joint_pos[:, joint_ids]
        knee_peak = torch.maximum(knee_peak, q[:, knee_ids].max(dim=1).values)
        hold_t += live.float() * env.step_dt
        ok = deploy_gate(env, cmd) & live
        gate_run = torch.where(ok, gate_run + env.step_dt, torch.zeros_like(gate_run))
        first = (gate_run >= 1.0) & torch.isnan(gate_time)
        gate_time[first] = hold_t[first]
      term = env.termination_manager.terminated & ~env.termination_manager.time_outs
      fell |= term & ~done_once
      done_once |= dones.bool()

  out: dict = {"checkpoint": str(args.checkpoint), "scenario": args.scenario}
  out["robust"] = args.robust
  out["friction"] = args.friction
  if args.scenario in ("train", "hold_train"):
    per_ep = term_sum / max(episodes, 1.0)
    rows = sorted(zip(term_names, per_ep.tolist(), strict=True), key=lambda r: r[1])
    out["reward_per_episode"] = dict(rows)
    print(f"\n{args.scenario}: reward per episode ({episodes:.0f} episodes)")
    for name, value in rows:
      print(f"  {name:28s} {value:+9.2f}")
    print(f"  {'TOTAL':28s} {sum(v for _, v in rows):+9.2f}")
  rms = (sq / count.clamp(min=1).unsqueeze(1)).sqrt()
  valid = count > 10
  ratio = rms / tau_cont
  worst = ratio[valid].max(dim=1).values
  mean_rms = rms[valid].mean(dim=0)
  mean_ratio = ratio[valid].mean(dim=0)
  p95_ratio = ratio[valid].quantile(0.95, dim=0) if valid.sum() > 1 else mean_ratio
  bm = torch.cat(braced_moments) if braced_moments else torch.zeros(1)
  out["joints"] = {
    SHORT[i]: {
      "rms_nm": round(float(mean_rms[i]), 2),
      "x_cont": round(float(mean_ratio[i]), 3),
      "x_cont_p95": round(float(p95_ratio[i]), 3),
      "target_beyond_limit_frac": round(
        float((beyond / count.clamp(min=1).unsqueeze(1))[valid].mean(dim=0)[i]), 4
      ),
    }
    for i in range(25)
  }
  if bool((count_tr > 10).any()):
    ok = count_tr > 10
    rms_tr = (sq_tr[ok] / count_tr[ok].unsqueeze(1)).sqrt().mean(dim=0)
    out["joints_transition"] = {
      SHORT[i]: {
        "rms_nm": round(float(rms_tr[i]), 2),
        "x_cont": round(float(rms_tr[i] / tau_cont[i]), 3),
      }
      for i in range(25)
    }
    out["transition_s_per_robot"] = round(float(count_tr[ok].mean()) * env.step_dt, 1)
  out["worst_motor_x_cont_mean"] = round(float(worst.mean()), 3)
  out["worst_motor_x_cont_p95"] = round(float(worst.quantile(0.95)), 3)
  out["share_within_budget"] = round(float((worst <= HEAT_BUDGET).float().mean()), 3)
  out["fell_frac"] = round(float(fell.float().mean()), 4)
  out["braced_moment_p95_nm"] = round(
    float(bm.quantile(0.95)) if bm.numel() > 1 else 0.0, 2
  )
  if crawl or walk:
    out["speed_mps"] = round(float((speed_sum / count.clamp(min=1))[valid].mean()), 3)
  else:
    out["knee_peak_rad_mean"] = round(float(knee_peak.mean()), 3)
    out["knee_peak_rad_max"] = round(float(knee_peak.max()), 3)
    reached = ~torch.isnan(gate_time)
    out["gate_reached_frac"] = round(float(reached.float().mean()), 3)
    out["gate_within_3s_frac"] = round(float((gate_time <= 3.0).float().mean()), 3)
    out["gate_time_median_s"] = (
      round(float(gate_time[reached].median()), 2) if reached.any() else None
    )

  order = sorted(range(25), key=lambda i: -float(mean_ratio[i]))
  print(f"\n{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).name}")
  print(
    f"scenario {args.scenario}  robust {args.robust}  friction {args.friction}  "
    f"robots {n}  measured {float(count[valid].mean()) * env.step_dt:.1f} s each"
  )
  print(f"{'joint':18s} {'RMS Nm':>7s} {'x cont':>7s} {'p95':>6s} {'tgt>lim':>8s}")
  for i in order[:10]:
    j = out["joints"][SHORT[i]]
    print(
      f"{SHORT[i]:18s} {j['rms_nm']:7.2f} {j['x_cont']:7.2f} {j['x_cont_p95']:6.2f} "
      f"{j['target_beyond_limit_frac'] * 100:7.1f}%"
    )
  keys = [k for k in out if k not in ("joints", "reward_per_episode", "checkpoint")]
  print("  ".join(f"{k} {out[k]}" for k in keys))
  if args.json is not None:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
  main()
