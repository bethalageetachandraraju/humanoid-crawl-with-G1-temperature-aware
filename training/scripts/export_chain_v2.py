"""Bundle walk + two-way stand <-> crawl + crawl for deployment (chain v2).

Writes, into ``--out`` (default ``deploy/g1_chain_v2``):

* ``walk.onnx``, ``stand_crawl.onnx``, ``crawl.onnx``: the actors, exported from
  the given checkpoints and checked against the torch actors;
* ``chain.json``: joint order, motor indices, PD gains, limits, every policy's
  observation layout and action mapping, the stand <-> crawl keyframes and
  phase semantics, the state machine and its gates, sources and checks;
* ``mujoco/scene.xml`` + ``assets/``: the braced 29-DoF G1 as these tasks
  simulate it, with ``stand`` and ``crawl`` keyframes, loadable by plain MuJoCo.

::

  uv run python scripts/export_chain_v2.py --walk-checkpoint <pt> \\
      --stand-crawl-checkpoint <pt> --out deploy/g1_chain_v2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_kneel_bundle import (  # noqa: E402
  BRACED_MOTORS,
  MOTOR_INDEX,
  SOFT_LIMIT_FACTOR,
  check_onnx,
  export_onnx,
  joint_limits,
)
from play_chain import STAND_CRAWL_TASK, WALK_TASK  # noqa: E402
from play_switcher import CRAWL_ABOVE_LIMBS, CRAWL_CHECKPOINT, CRAWL_TASK  # noqa: E402

from mjlab.scene import Scene  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import (  # noqa: E402
  BRACED_LOADS_29,
  BRACED_TORQUE_LIMIT,
  G1_XML,
  MOTORS_29,
)
from mjlab.tasks.g1_walk.config.g1.env_cfgs import (  # noqa: E402
  IMU_TILT_MAX_DEG,
  TRANSITION_ACTION_CLIP,
  TRANSITION_ACTION_SCALE,
  _action_scale_29,
  unitree_g1_stand_to_crawl_env_cfg,
)
from mjlab.tasks.g1_walk.mdp.keyframes import KEYFRAME_TABLE  # noqa: E402
from mjlab.tasks.g1_walk.robot import (  # noqa: E402
  CRAWL_HEIGHT,
  CRAWL_POSE_29,
  CRAWL_ROT,
  JOINTS_29,
  STAND_HEIGHT,
  STAND_POSE_29,
)
from mjlab.tasks.registry import load_rl_cfg  # noqa: E402

# The hardened walk every chain test runs (play_chain's WALK_CHECKPOINT) and
# chain v2 / v3 ship; the old default (walk_sym_spin2/25000) predates it.
DEFAULT_WALK = (
  "logs/rsl_rl/g1_walk_29dof/2026-10-23_06-41-09_walk_hard_track/model_55499.pt"
)


def export_mjcf(out: Path) -> dict:
  cfg = unitree_g1_stand_to_crawl_env_cfg(play=True, bidirectional=True)
  cfg.scene.num_envs = 1
  scene = Scene(cfg.scene, "cpu")
  spec = scene.spec
  o, mj = spec.option, cfg.sim.mujoco
  o.timestep = mj.timestep
  o.integrator = {
    "euler": mujoco.mjtIntegrator.mjINT_EULER,
    "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
  }[mj.integrator]
  o.cone = {
    "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL,
    "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC,
  }[mj.cone]
  o.impratio, o.iterations, o.tolerance = mj.impratio, mj.iterations, mj.tolerance
  o.ls_iterations, o.ls_tolerance = mj.ls_iterations, mj.ls_tolerance
  o.ccd_iterations = mj.ccd_iterations

  ref = spec.compile()
  names = [ref.joint(i).name.removeprefix("robot/") for i in range(ref.njnt)]
  qadr = {n: int(ref.jnt_qposadr[i]) for i, n in enumerate(names)}
  act_joint = [
    ref.joint(int(ref.actuator_trnid[a, 0])).name.removeprefix("robot/")
    for a in range(ref.nu)
  ]

  def key(name: str, height: float, quat: tuple, pose: dict) -> None:
    qpos = np.zeros(ref.nq)
    qpos[:3] = [0.0, 0.0, height]
    qpos[3:7] = quat
    for j, v in pose.items():
      qpos[qadr[j]] = v
    spec.add_key(name=name, qpos=qpos.tolist(), ctrl=[pose[j] for j in act_joint])

  for k in list(spec.keys):
    spec.delete(k)
  key("stand", STAND_HEIGHT, (1.0, 0.0, 0.0, 0.0), dict(STAND_POSE_29))
  key("crawl", CRAWL_HEIGHT + 0.01, CRAWL_ROT, dict(CRAWL_POSE_29))
  scene.write(out)
  (out / "assets").mkdir(exist_ok=True)
  for m in spec.meshes:
    if m.file:
      shutil.copy(G1_XML.parent / "assets" / m.file, out / "assets" / Path(m.file).name)
  a = spec.compile()
  b = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
  fields = (
    "body_mass",
    "body_inertia",
    "dof_armature",
    "jnt_range",
    "geom_friction",
    "geom_condim",
    "geom_priority",
    "geom_size",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
  )
  diff = max(
    float(
      (np.abs(getattr(a, f) - getattr(b, f)) / (1e-6 + np.abs(getattr(a, f)))).max()
    )
    for f in fields
  )
  assert (a.nq, a.nu, a.ngeom) == (b.nq, b.nu, b.ngeom) and diff < 1e-5, diff
  return {
    "nq": b.nq,
    "nu": b.nu,
    "mass_kg": float(b.body_subtreemass[0]),
    "max_relative_field_diff": diff,
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--walk-checkpoint", default=DEFAULT_WALK)
  ap.add_argument("--stand-crawl-checkpoint", required=True)
  ap.add_argument("--crawl-checkpoint", default=CRAWL_CHECKPOINT)
  ap.add_argument("--out", default="deploy/g1_chain_v2")
  ap.add_argument("--crawl-fwd", type=float, nargs=2, default=(-1.0, 2.5))
  ap.add_argument("--crawl-lat", type=float, default=0.5, help="+- m/s")
  ap.add_argument("--crawl-turn", type=float, default=3.0, help="+- rad/s")
  ap.add_argument("--note", default="", help="Stored as chain.json's 'note'.")
  ap.add_argument(
    "--relaxed-handover",
    action="store_true",
    help="STAND_HOLD -> WALK once upright and tall (play_chain --relaxed-handover)",
  )
  ap.add_argument(
    "--stand-crawl-training",
    default=None,
    help="Replaces chain.json's training.stand_crawl text.",
  )
  ap.add_argument(
    "--crawl-variants",
    nargs="*",
    default=[],
    help="name=checkpoint[:fwd_lo,fwd_hi,lat,turn] ... : extra crawl styles, "
    "exported as crawl_<name>.onnx (same inputs and outputs as crawl.onnx)",
  )
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  ckpts = {
    "walk": (WALK_TASK, Path(args.walk_checkpoint)),
    "stand_crawl": (STAND_CRAWL_TASK, Path(args.stand_crawl_checkpoint)),
    "crawl": (CRAWL_TASK, Path(args.crawl_checkpoint)),
  }
  checks = {}
  clips = {}
  for name, (task, ck) in ckpts.items():
    export_onnx(task, ck, out / f"{name}.onnx")
    clip = load_rl_cfg(task).clip_actions
    clips[name] = float(clip) if clip else None
    checks[name] = check_onnx(task, ck, out / f"{name}.onnx", clip or 1e9)
    assert checks[name] < 1e-4, (name, checks[name])
  variants = {}
  for spec in args.crawl_variants:
    name, _, rest = spec.partition("=")
    path, _, ranges = rest.partition(":")
    onnx = out / f"crawl_{name}.onnx"
    export_onnx(CRAWL_TASK, Path(path), onnx)
    err = check_onnx(CRAWL_TASK, Path(path), onnx, clips["crawl"] or 1e9)
    assert err < 1e-4, (name, err)
    checks[f"crawl_{name}"] = err
    lo, hi, lat, turn = (
      [float(x) for x in ranges.split(",")] if ranges else (-1.0, 2.5, 0.5, 3.0)
    )
    variants[name] = {
      "file": onnx.name,
      "checkpoint": path,
      "command_ranges": {
        "fwd": [lo, hi],
        "lat": [-lat, lat],
        "turn": [-turn, turn],
        "step_height": [0.02, 0.15],
      },
    }
  print(f"onnx vs torch: {checks}")
  mj = export_mjcf(out / "mujoco")
  print(f"mujoco: {mj}")
  model = mujoco.MjModel.from_xml_path(str(out / "mujoco" / "scene.xml"))

  phase_doc = (
    "phi 0 = standing, 1 = crawl pose. Down: phi += dt/T, dphi/dt = +1/T. Up: "
    "phi -= dt/T, dphi/dt = -1/T. Holding either end: dphi/dt = 0. T trained "
    "in [4, 8] s (6 s default; 8 s on hardware first)."
  )
  chain = {
    "version": 2,
    "control_hz": 50,
    "physics_dt_s": 0.005,
    "decimation": 4,
    "joints": list(JOINTS_29),
    "motor_index": dict(zip(JOINTS_29, MOTOR_INDEX, strict=True)),
    "pd": {
      j: {"kp": kp, "kd": kd, "peak_nm": peak}
      for j, (kp, kd, peak) in MOTORS_29.items()
    },
    "joint_limits": joint_limits(model),
    "soft_joint_limits": joint_limits(model, SOFT_LIMIT_FACTOR),
    "braced_joints": {
      j: {
        "motor_index": BRACED_MOTORS[j],
        "site": site,
        "axis": axis,
        "limit_nm": BRACED_TORQUE_LIMIT,
        "mode": "damping (kp 0, small kd); the brace holds it at 0",
      }
      for j, (site, axis) in BRACED_LOADS_29.items()
    },
    "stand_pose": dict(STAND_POSE_29),
    "stand_pelvis_height_m": STAND_HEIGHT,
    "crawl_pose": dict(CRAWL_POSE_29),
    "crawl_pelvis_height_m": CRAWL_HEIGHT,
    "crawl_pelvis_above_limbs_m": CRAWL_ABOVE_LIMBS,
    "policies": {
      "walk": {
        "file": "walk.onnx",
        "input_dim": 84,
        "obs": "gravity_b(3), gyro_b(3), [vx, vy, wz](3), q - stand_pose(25), qd(25), last_action(25)",
        "action_clip": clips["walk"],
        "target": "stand_pose + scale * clip(action)",
        "scale": _action_scale_29(),
        "command_ranges": {"vx": [-0.6, 1.0], "vy": [-0.5, 0.5], "wz": [-0.8, 0.8]},
      },
      "stand_crawl": {
        "file": "stand_crawl.onnx",
        "input_dim": 108,
        "obs": "gravity_b(3), gyro_b(3), q - stand_pose(25), qd(25), last_action(25), [phi, dphi/dt](2), ref(phi) - q(25)",
        "action_clip": TRANSITION_ACTION_CLIP,
        "action_scale": TRANSITION_ACTION_SCALE,
        "target": f"ref(phi) + {TRANSITION_ACTION_SCALE} * clip(action, +-{TRANSITION_ACTION_CLIP})",
        "phase": phase_doc,
        "schedule": "per joint, smoothstep between consecutive keyframes: s = x*x*(3-2x) of the position inside the interval",
        "keyframes": KEYFRAME_TABLE,
      },
      "crawl": {
        "file": "crawl.onnx",
        "input_dim": 88,
        "obs": "gravity_b(3), gyro_b(3), [fwd, lat, turn, step_h, 0, 0, 0](7), q - crawl_pose(25), qd(25), last_action(25)",
        "action_clip": clips["crawl"],
        "target": "crawl_pose + 0.5 * clip(action)",
        "command_ranges": {
          "fwd": list(args.crawl_fwd),
          "lat": [-args.crawl_lat, args.crawl_lat],
          "turn": [-args.crawl_turn, args.crawl_turn],
          "step_height": [0.02, 0.15],
        },
        "turn_sign": "positive = clockwise from above (opposite of walk wz)",
      },
    },
    "switcher": {
      "states": (
        "WALK -C-> STOPPING -still-> DOWN -phi 1-> HOLD -crawl gate-> CRAWL; "
        "CRAWL -U-> CRAWL_STOP -still on 4 limbs-> HOLD (stand_crawl at phi 1) "
        "-crawl gate-> UP -phi 0-> STAND_HOLD -upright, still-> WALK"
      ),
      "stopping": "walk command 0; still (max |qd| < 0.6 rad/s), base slow, both feet down, 0.5 s. A descent started mid-step can fall.",
      "crawl_gate": "max |qd| < 0.5 rad/s, pelvis above mean limb height within 0.03 m of crawl_pelvis_above_limbs_m, |gravity_b - crawl gravity| < 0.15, four limbs down; 1.0 s",
      "crawl_stop": "crawl command 0; max |qd| < 0.5 rad/s, |gravity_b - crawl gravity| < 0.25, four limbs down, 0.5 s. The crawl policy rests ~14 cm above the crawl pose; stand_crawl at phi 1 then settles it into the pose before the gate.",
      "stand_hold": (
        "gravity_z < -0.94, pelvis > 0.70 m, both feet down; 0.3 s (relaxed: the "
        "walk policy settles the stand, still or not)"
        if args.relaxed_handover
        else "gravity_z < -0.97, pelvis > 0.70 m, max |qd| < 0.6 rad/s, base slow, "
        "both feet down; 0.5 s"
      ),
      "blend_s": 0.5,
      "shadow": "run every policy every tick on live observations, so its last-action input is its own",
      "real_robot": "the sim gates use contacts and base velocity; on the robot use IMU and encoders (limb heights by forward kinematics with mujoco/scene.xml) as in deploy/g1_kneel/kneel_controller.py",
    },
    "training": {
      "walk": "see sources; hard runs: WALK_RAMP_HARD level 1 + 2 deg IMU tilt + arm-end mass",
      "stand_crawl": args.stand_crawl_training
      or (
        "TRANSITION_RAMP_HARD to level 1: pushes 0.6 m/s, gains/armature "
        "+-25/30 percent, motors 75-100 percent, joint friction <= 1 Nm, "
        "payload -3..+5 kg, friction 0.2-1.4, noise x2, 5-20 ms delay, IMU tilt "
        f"<= {IMU_TILT_MAX_DEG:.0f} deg, arm ends -0.05..+0.2 kg"
      ),
    },
    "sources": {k: str(v[1]) for k, v in ckpts.items()},
    "crawl_variants": {
      **variants,
      "_doc": (
        "Same observation, action and target formula as policies.crawl; switch "
        "while the crawl command is 0 and the robot is still, blending targets "
        "over blend_s (a mid-stride switch jolts the robot)."
      ),
    }
    if variants
    else {},
    "checks": {"onnx_vs_torch_max_abs": checks, "mujoco": mj},
    "note": args.note,
  }
  (out / "chain.json").write_text(json.dumps(chain, indent=2))
  print(f"wrote {out}/: walk.onnx, stand_crawl.onnx, crawl.onnx, chain.json, mujoco/")


if __name__ == "__main__":
  main()
