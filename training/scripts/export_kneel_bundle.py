"""Bundle the walk and kneel policies for deployment and sim-to-sim.

Writes, into ``--out`` (default ``deploy/g1_kneel``):

* ``walk.onnx``, ``kneel.onnx``: the actors (normalizer inside), exported from
  the given checkpoints and checked against the torch actors.
* ``kneel_chain.json``: everything a controller needs: joint order, real motor
  indices, PD gains, soft limits, both policies' observation layouts and action
  mappings, the kneel keyframe table, and the switcher settings.
* ``mujoco/scene.xml`` + ``mujoco/assets/``: the braced 29-DoF G1 exactly as
  the kneel task simulates it (collisions, actuators, solver options), with
  ``stand`` and ``kneel`` keyframes, loadable by plain MuJoCo. Checked against
  the model mjlab compiles.

::

  uv run python scripts/export_kneel_bundle.py --out deploy/g1_kneel
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_kneel import KNEEL_TASK, WALK_CHECKPOINT, WALK_TASK  # noqa: E402
from play_switcher import latest_checkpoint, load_actor  # noqa: E402

from mjlab.scene import Scene  # noqa: E402
from mjlab.tasks.g1_crawl.robot29 import (  # noqa: E402
  BRACED_LOADS_29,
  BRACED_TORQUE_LIMIT,
  G1_XML,
  MOTORS_29,
)
from mjlab.tasks.g1_walk.config.g1.env_cfgs import (  # noqa: E402
  TRANSITION_ACTION_CLIP,
  TRANSITION_ACTION_SCALE,
  _action_scale_29,
  unitree_g1_kneel_env_cfg,
)
from mjlab.tasks.g1_walk.mdp.kneel_keyframes import (  # noqa: E402
  KNEEL_KEYFRAME_TABLE,
  KNEEL_SWING_WINDOW,
)
from mjlab.tasks.g1_walk.robot import (  # noqa: E402
  JOINTS_29,
  STAND_HEIGHT,
  STAND_POSE_29,
)

# Policy index -> Unitree G1 29-DoF motor index (G1JointIndex in unitree_sdk2).
MOTOR_INDEX = list(range(20)) + [22, 23, 24, 25, 26]
BRACED_MOTORS = {
  "left_wrist_pitch_joint": 20,
  "left_wrist_yaw_joint": 21,
  "right_wrist_pitch_joint": 27,
  "right_wrist_yaw_joint": 28,
}
SOFT_LIMIT_FACTOR = 0.9


def export_onnx(task: str, checkpoint: Path, out: Path) -> None:
  subprocess.run(
    [
      sys.executable,
      "scripts/export_policy_onnx.py",
      "--task",
      task,
      "--checkpoint",
      str(checkpoint),
      "--out",
      str(out),
    ],
    check=True,
  )


def check_onnx(task: str, checkpoint: Path, path: Path, clip: float) -> float:
  """Max |clip(onnx) - torch| over random observations (mean action)."""
  actor, dim = load_actor(task, checkpoint, "cpu")
  sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
  rng = np.random.default_rng(0)
  worst = 0.0
  for _ in range(64):
    obs = rng.normal(0.0, 0.5, size=(1, dim)).astype(np.float32)
    a_onnx = np.clip(sess.run(None, {"obs": obs})[0], -clip, clip)
    with torch.inference_mode():
      a_torch = actor(torch.from_numpy(obs)).numpy()
    worst = max(worst, float(np.abs(a_onnx - a_torch).max()))
  return worst


def export_mjcf(out: Path) -> dict:
  """The kneel task's scene (plane + robot) as plain MJCF, plus checks."""
  cfg = unitree_g1_kneel_env_cfg(play=True)
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
  o.solver = {
    "newton": mujoco.mjtSolver.mjSOL_NEWTON,
    "cg": mujoco.mjtSolver.mjSOL_CG,
    "pgs": mujoco.mjtSolver.mjSOL_PGS,
  }[mj.solver]
  o.impratio = mj.impratio
  o.iterations = mj.iterations
  o.tolerance = mj.tolerance
  o.ls_iterations = mj.ls_iterations
  o.ls_tolerance = mj.ls_tolerance
  o.ccd_iterations = mj.ccd_iterations

  # Keyframes: stand (the reset pose) and kneel (the end of the schedule).
  ref = spec.compile()
  names = [ref.joint(i).name.removeprefix("robot/") for i in range(ref.njnt)]
  qadr = {n: int(ref.jnt_qposadr[i]) for i, n in enumerate(names)}
  act_joint = [
    ref.joint(int(ref.actuator_trnid[a, 0])).name.removeprefix("robot/")
    for a in range(ref.nu)
  ]

  def key(name: str, height: float, pitch_deg: float, pose: dict) -> None:
    qpos = np.zeros(ref.nq)
    p = math.radians(pitch_deg) / 2.0
    qpos[:7] = [0.0, 0.0, height, math.cos(p), 0.0, math.sin(p), 0.0]
    for j, v in pose.items():
      qpos[qadr[j]] = v
    ctrl = [pose[j] for j in act_joint]
    spec.add_key(name=name, qpos=qpos.tolist(), ctrl=ctrl)

  for k in list(spec.keys):
    spec.delete(k)
  key("stand", STAND_HEIGHT, 0.0, dict(STAND_POSE_29))
  last = KNEEL_KEYFRAME_TABLE[-1]
  key("kneel", last["height"] + 0.005, last["pitch_deg"], last["joint_pos"])
  scene.write(out)
  # The G1 meshes are loaded from files (not spec assets): copy the ones used.
  mesh_src = G1_XML.parent / "assets"
  (out / "assets").mkdir(exist_ok=True)
  for m in spec.meshes:
    if m.file:
      shutil.copy(mesh_src / m.file, out / "assets" / Path(m.file).name)

  # Plain MuJoCo must load it and agree with mjlab's compiled model.
  a = spec.compile()
  b = mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
  fields = (
    "body_mass",
    "body_inertia",
    "dof_armature",
    "dof_damping",
    "jnt_range",
    "geom_friction",
    "geom_condim",
    "geom_priority",
    "geom_contype",
    "geom_conaffinity",
    "geom_size",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
  )
  # Relative: the XML writer rounds to ~6 significant digits.
  diff = {
    f: float(
      (np.abs(getattr(a, f) - getattr(b, f)) / (1e-6 + np.abs(getattr(a, f)))).max()
    )
    for f in fields
  }
  assert (a.nq, a.nv, a.nu, a.ngeom) == (b.nq, b.nv, b.nu, b.ngeom)
  assert max(diff.values()) < 1e-5, diff
  return {
    "nq": b.nq,
    "nu": b.nu,
    "mass_kg": float(b.body_subtreemass[0]),
    "timestep": b.opt.timestep,
    "max_relative_field_diff": max(diff.values()),
  }


def joint_limits(model: mujoco.MjModel, factor: float = 1.0) -> dict:
  out = {}
  for i in range(model.njnt):
    name = model.joint(i).name.removeprefix("robot/")
    if name in JOINTS_29:
      lo, hi = model.jnt_range[i]
      mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0 * factor
      out[name] = [round(float(mid - half), 4), round(float(mid + half), 4)]
  return {j: out[j] for j in JOINTS_29}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--walk-checkpoint", default=WALK_CHECKPOINT)
  ap.add_argument("--kneel-checkpoint", default="latest")
  ap.add_argument("--out", default="deploy/g1_kneel")
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  walk_ckpt = Path(args.walk_checkpoint)
  kneel_ckpt = (
    latest_checkpoint("g1_kneel_29dof")
    if args.kneel_checkpoint == "latest"
    else Path(args.kneel_checkpoint)
  )
  assert kneel_ckpt is not None

  export_onnx(WALK_TASK, walk_ckpt, out / "walk.onnx")
  export_onnx(KNEEL_TASK, kneel_ckpt, out / "kneel.onnx")
  err_walk = check_onnx(WALK_TASK, walk_ckpt, out / "walk.onnx", 5.0)
  err_kneel = check_onnx(
    KNEEL_TASK, kneel_ckpt, out / "kneel.onnx", TRANSITION_ACTION_CLIP
  )
  print(f"onnx vs torch: walk {err_walk:.2e}, kneel {err_kneel:.2e}")
  assert err_walk < 1e-4 and err_kneel < 1e-4

  mj = export_mjcf(out / "mujoco")
  print(f"mujoco/scene.xml: {mj}")
  model = mujoco.MjModel.from_xml_path(str(out / "mujoco" / "scene.xml"))

  walk_scale = _action_scale_29()
  chain = {
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
    "kneel_pose": KNEEL_KEYFRAME_TABLE[-1]["joint_pos"],
    "policies": {
      "walk": {
        "file": "walk.onnx",
        "input_dim": 84,
        "observation_names": [
          "projected_gravity(3)",
          "base_ang_vel(3)",
          "velocity_commands(3)=[vx, vy, wz]",
          "joint_pos(25)=q - stand_pose",
          "joint_vel(25)=qd",
          "actions(25)=last clipped action",
        ],
        "action_clip": 5.0,
        "target": "stand_pose + scale * clip(action, -5, 5)",
        "scale": walk_scale,
        "command_ranges": {"vx": [-0.6, 1.0], "vy": [-0.5, 0.5], "wz": [-0.8, 0.8]},
      },
      "kneel": {
        "file": "kneel.onnx",
        "input_dim": 108,
        "observation_names": [
          "projected_gravity(3)",
          "base_ang_vel(3)",
          "joint_pos(25)=q - stand_pose",
          "joint_vel(25)=qd",
          "actions(25)=last clipped action",
          "phase(2)=[phi, dphi/dt]",
          "ref_joint_err(25)=ref(phi) - q",
        ],
        "action_clip": TRANSITION_ACTION_CLIP,
        "action_scale": TRANSITION_ACTION_SCALE,
        "target": (
          f"ref(phi) + {TRANSITION_ACTION_SCALE} * "
          f"clip(action, -{TRANSITION_ACTION_CLIP}, {TRANSITION_ACTION_CLIP})"
        ),
        "phase": (
          "phi 0 = standing, 1 = kneeling. Going down: phi += dt / T and "
          "dphi/dt = +1/T. Standing up: phi -= dt / T and dphi/dt = -1/T. "
          "Holding either end: dphi/dt = 0. T in [4, 8] s (trained), 6 s default."
        ),
        "schedule": (
          "ref(phi) per joint: find the keyframe interval [phi_i, phi_i+1] "
          "containing phi, s = smoothstep((phi - phi_i) / (phi_i+1 - phi_i)) "
          "with smoothstep(x) = x*x*(3 - 2x), ref = q_i + (q_i+1 - q_i) * s"
        ),
        "swing_window_phi": list(KNEEL_SWING_WINDOW),
        "keyframes": [
          {"name": f"K{i}", **k} for i, k in enumerate(KNEEL_KEYFRAME_TABLE)
        ],
      },
    },
    "switcher": {
      "states": (
        "WALK -K-> STOPPING -still dwell-> KNEEL_DOWN -phi 1-> KNEELING "
        "-U-> STANDING_UP -phi 0-> STAND_HOLD -upright+still dwell-> WALK"
      ),
      "stop_gate": "walk command 0; max |qd| < 0.6 rad/s and |gyro| < 0.3 rad/s "
      "for 0.5 s (sim also checked both feet down). Never start the kneel while "
      "stepping: it falls.",
      "stand_gate": "gravity_z < -0.97, max |qd| < 0.6 rad/s, |gyro| < 0.3 rad/s "
      "for 0.5 s (sim also checked pelvis > 0.70 m and both feet down)",
      "blend_s": 0.5,
      "blend": "smoothstep from the last applied PD target to the new policy's "
      "target, at WALK->KNEEL_DOWN and STAND_HOLD->WALK",
      "shadow": "run both policies every step on live observations, so each "
      "one's last-action input is always its own",
      "queue": "K or U pressed mid-transition waits until the pose is reached",
    },
    "training_sim": {
      "simulator": "mjlab 1.6.0 on MuJoCo Warp",
      "action_delay_physics_steps": "0-4 (0-20 ms), held per robot for ~1 s",
      "encoder_bias": "randomized per episode",
      "observation_noise": "gravity +-0.05, gyro +-0.2, q +-0.01, qd +-1.5 (uniform)",
      "kneel_robustness_level_at_end": 0.7,
    },
    "sources": {"walk": str(walk_ckpt), "kneel": str(kneel_ckpt)},
    "checks": {
      "onnx_vs_torch_max_abs": {"walk": err_walk, "kneel": err_kneel},
      "mujoco": mj,
    },
  }
  (out / "kneel_chain.json").write_text(json.dumps(chain, indent=2))
  print(f"wrote {out}/: walk.onnx, kneel.onnx, kneel_chain.json, mujoco/")


if __name__ == "__main__":
  main()
