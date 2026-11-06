"""Solve the modified 29-DoF G1's crawl pose and retarget the crawl animation.

The rc4 crawl motion was captured on the source project's 23-DoF G1, whose
forearm is one rigid link ending in a 2 cm ball. The 29-DoF robot on the bench is
modified for crawling (``mjlab/tasks/g1_crawl/robot29.py``): hands removed, wrist
pitch/yaw braced, and a printed forearm extension with a domed tip. Its forearm
is therefore also one rigid link with a round tip, just a different length, so
the source arm angles do not put the tip on the ground.

This script solves, per frame and per side, for the five arm joints (shoulder
pitch/roll/yaw, elbow, wrist roll) that put the extension where the 23-DoF ball
was:

* position: the dome tip (the ``*_hand_contact`` site) matches the ``*_palm``
  site of the 23-DoF model, which sits on the tip of its ball hand;
* direction: the extension's long axis matches the 23-DoF forearm's +X axis, so
  the forearm stays near-vertical when planted, as in the source gait.

Both are expressed in the torso frame, so the result is independent of where the
root is. Legs, ankles and waist yaw are copied unchanged; waist roll and pitch
are held at zero (the source robot has neither).

Outputs (both read by ``robot29.py`` and the crawl task):

* ``assets/crawl_pose_29dof.json`` -- the crawl default pose, forced left-right
  symmetric (the action offset, ``joint_pos_rel`` and the mirror loss all
  measure against it);
* ``assets/animation_rc4_29dof.json`` -- the 80 reset frames.

Re-run after changing ``EXTENSION_LENGTH`` in ``robot29.py``::

  uv run python scripts/retarget_crawl_animation.py
  uv run python scripts/retarget_crawl_animation.py --render /tmp/check.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.tasks.g1_crawl import robot29
from mjlab.tasks.g1_crawl.robot import CRAWL_INIT_STATE, get_crawl_spec

SOURCE = Path(__file__).resolve().parents[1] / "src/mjlab/tasks/g1_crawl/assets"
SOURCE = SOURCE / "animation_rc4.json"

# 23-DoF hand: sphere centre (0.18,0,0) radius 0.02 in ``*_forearm``, so the tip
# is (0.20,0,0) -- the ``*_palm`` site. Its axis is the forearm +X.
TIP_23 = np.array([0.20, 0.0, 0.0])
AXIS = np.array([1.0, 0.0, 0.0])  # Forearm +X on both robots.

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")
# Sign each arm joint picks up under a left-right reflection: pitch axes (Y) keep
# their sign, roll (X) and yaw (Z) flip.
MIRROR_SIGN = {
  "shoulder_pitch": 1.0,
  "shoulder_roll": -1.0,
  "shoulder_yaw": -1.0,
  "elbow": 1.0,
  "wrist_roll": -1.0,
}

# Residual weights: ``AXIS_WEIGHT`` trades forearm direction against tip
# position; ``REG_WEIGHT`` only breaks ties between equivalent arm poses.
AXIS_WEIGHT = 0.30
REG_WEIGHT = 0.004
SOFT_LIMIT_FACTOR = 0.9


class _Chain:
  """Forward kinematics for one model, with torso-frame readout."""

  def __init__(self, model: mujoco.MjModel) -> None:
    self.model = model
    self.data = mujoco.MjData(model)
    self.torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

  def set_joints(self, joint_pos: dict[str, float]) -> None:
    self.data.qpos[:] = 0.0
    self.data.qpos[3] = 1.0  # Identity root quaternion; targets are torso-frame.
    for name, value in joint_pos.items():
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
      if jid >= 0:
        self.data.qpos[self.model.jnt_qposadr[jid]] = value
    mujoco.mj_kinematics(self.model, self.data)

  def in_torso(
    self, body: str, local_point: np.ndarray, local_axis: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
    torso_rot = self.data.xmat[self.torso].reshape(3, 3)
    body_rot = self.data.xmat[bid].reshape(3, 3)
    offset = self.data.xpos[bid] + body_rot @ local_point - self.data.xpos[self.torso]
    return torso_rot.T @ offset, torso_rot.T @ (body_rot @ local_axis)

  def soft_limits(self, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    low, high = [], []
    for name in names:
      jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
      lo, hi = self.model.jnt_range[jid]
      mid, half = (lo + hi) / 2, (hi - lo) / 2 * SOFT_LIMIT_FACTOR
      low.append(mid - half)
      high.append(mid + half)
    return np.array(low), np.array(high)


def _non_arm(joint_pos: dict[str, float]) -> dict[str, float]:
  """Legs and waist yaw, plus waist roll/pitch pinned to zero."""
  out = {
    name: value
    for name, value in joint_pos.items()
    if not any(part in name for part in ("shoulder", "elbow", "wrist"))
  }
  out["waist_roll_joint"] = 0.0
  out["waist_pitch_joint"] = 0.0
  return out


def retarget_pose(
  chain23: _Chain,
  chain29: _Chain,
  pose23: dict[str, float],
  regularizer: dict[str, float],
  seed: dict[str, float] | None = None,
  mirror: bool = False,
) -> tuple[dict[str, float], float, float]:
  """Retarget one 23-DoF pose. ``mirror`` solves the left arm and reflects it."""
  tip_29 = np.array([robot29.EXTENSION_LENGTH, 0.0, 0.0])
  base = _non_arm(pose23)
  out = dict(base)
  worst_pos = worst_axis = 0.0
  chain23.set_joints(pose23)
  for side in ("left",) if mirror else ("left", "right"):
    target_point, target_axis = chain23.in_torso(f"{side}_forearm", TIP_23, AXIS)
    names = [f"{side}_{joint}_joint" for joint in ARM_JOINTS]
    low, high = chain29.soft_limits(names)
    reg = np.array([regularizer[name] for name in names])
    start = np.array([(seed or regularizer)[name] for name in names])

    def residual(x: np.ndarray, names=names, reg=reg, side=side, tp=target_point,
                 ta=target_axis) -> np.ndarray:  # fmt: skip
      chain29.set_joints({**base, **dict(zip(names, x, strict=True))})
      point, axis = chain29.in_torso(f"{side}_forearm_extension", tip_29, AXIS)
      return np.concatenate(
        [point - tp, AXIS_WEIGHT * (axis - ta), REG_WEIGHT * (x - reg)]
      )

    solution = least_squares(
      residual,
      np.clip(start, low + 1e-6, high - 1e-6),
      bounds=(low, high),
      xtol=1e-12,
      ftol=1e-12,
      gtol=1e-12,
      max_nfev=4000,
    )
    res = residual(solution.x)
    out.update(dict(zip(names, solution.x, strict=True)))
    worst_pos = max(worst_pos, float(np.linalg.norm(res[:3])))
    worst_axis = max(worst_axis, float(np.linalg.norm(res[3:6]) / AXIS_WEIGHT))
  chain23.set_joints(pose23)
  if mirror:
    for joint, sign in MIRROR_SIGN.items():
      out[f"right_{joint}_joint"] = sign * out[f"left_{joint}_joint"]
  return out, worst_pos, worst_axis


def _resolve(patterns: dict[str, float], names: list[str]) -> dict[str, float]:
  out = {}
  for name in names:
    for pattern, value in patterns.items():
      if re.fullmatch(pattern, name):
        out[name] = float(value)
  missing = sorted(set(names) - set(out))
  assert not missing, missing
  return out


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--render", type=Path, default=None, help="Write a 23 vs 29 comparison sheet."
  )
  args = parser.parse_args()

  source = json.loads(SOURCE.read_text())
  frames = np.asarray(source["frames"], dtype=np.float64)
  qposadr23 = {
    joint["name"]: int(joint["qposadr"])
    for joint in source["metadata"]["joints"]
    if joint["type"] != "free"
  }

  model23 = get_crawl_spec().compile()
  model29 = robot29.get_crawl_spec_29(check_pose=False).compile()
  chain23, chain29 = _Chain(model23), _Chain(model29)
  names29 = [
    mujoco.mj_id2name(model29, mujoco.mjtObj.mjOBJ_JOINT, j)
    for j in range(1, model29.njnt)
  ]
  assert tuple(names29) == robot29.CRAWL_JOINTS_29, names29

  # Default pose, symmetric. The 23-DoF arm angles seed and regularize the solve.
  assert CRAWL_INIT_STATE.joint_pos is not None
  pose23_default = _resolve(CRAWL_INIT_STATE.joint_pos, list(qposadr23))
  default29, err_pos, err_axis = retarget_pose(
    chain23, chain29, pose23_default, regularizer=pose23_default, mirror=True
  )
  print(
    f"extension {robot29.EXTENSION_LENGTH * 1000:.1f} mm: default pose tip error "
    f"{err_pos * 1000:.3f} mm, axis error {err_axis:.5f}"
  )
  for joint in ARM_JOINTS:
    name = f"left_{joint}_joint"
    print(f"  {name:<28} {default29[name]:+.4f}  (23-DoF {pose23_default[name]:+.4f})")
  robot29.CRAWL_POSE_29.write_text(
    json.dumps(
      {
        "extension_length": robot29.EXTENSION_LENGTH,
        "generated_by": Path(__file__).name,
        "joint_pos": {name: default29[name] for name in robot29.CRAWL_JOINTS_29},
      },
      indent=1,
    )
  )
  print(f"wrote {robot29.CRAWL_POSE_29}")

  # Animation frames, each seeded by the previous solution for continuity.
  out_frames, errors = [], []
  previous: dict[str, float] | None = None
  for frame in frames:
    pose23 = {name: float(frame[adr]) for name, adr in qposadr23.items()}
    pose29, err_pos, err_axis = retarget_pose(
      chain23, chain29, pose23, regularizer=default29, seed=previous
    )
    errors.append((err_pos, err_axis))
    previous = pose29
    row = np.zeros(model29.nq)
    row[:7] = frame[:7]
    for name, value in pose29.items():
      jid = mujoco.mj_name2id(model29, mujoco.mjtObj.mjOBJ_JOINT, name)
      row[model29.jnt_qposadr[jid]] = value
    out_frames.append(row)
  errors = np.asarray(errors)
  arr = np.asarray(out_frames)
  print(
    f"{len(frames)} frames: tip error mean {errors[:, 0].mean() * 1000:.3f} mm, "
    f"max {errors[:, 0].max() * 1000:.3f} mm; axis error max {errors[:, 1].max():.5f}; "
    f"max frame step {np.abs(np.diff(arr[:, 7:], axis=0)).max():.4f} rad"
  )

  joints_meta = [
    {"name": "floating_base_joint", "type": "free", "qposadr": 0, "qposdim": 7}
  ]
  for name in robot29.CRAWL_JOINTS_29:
    jid = mujoco.mj_name2id(model29, mujoco.mjtObj.mjOBJ_JOINT, name)
    joints_meta.append(
      {
        "name": name,
        "type": "hinge",
        "qposadr": int(model29.jnt_qposadr[jid]),
        "qposdim": 1,
      }
    )
  robot29.CRAWL_ANIMATION_29.write_text(
    json.dumps(
      {
        "schema": source.get("schema"),
        "dt": source.get("dt"),
        "fps": source.get("fps"),
        "nq": int(model29.nq),
        "nv": int(model29.nv),
        "frames": arr.tolist(),
        "metadata": {
          "joints": joints_meta,
          "base_forward_velocity_mps": source["metadata"].get(
            "base_forward_velocity_mps"
          ),
          "retargeted_from": SOURCE.name,
          "retargeted_by": Path(__file__).name,
          "extension_length": robot29.EXTENSION_LENGTH,
          "tip_error_max_mm": float(errors[:, 0].max() * 1000),
        },
      }
    )
  )
  print(f"wrote {robot29.CRAWL_ANIMATION_29}")

  if args.render is not None:
    _render_sheet(model23, model29, frames, qposadr23, arr, args.render)


def _render_sheet(model23, model29, frames, qposadr23, arr29, path: Path) -> None:
  from PIL import Image

  def shot(model, qpos):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, 480, 640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = qpos[:3]
    camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -20
    renderer.update_scene(data, camera)
    image = Image.fromarray(renderer.render())
    renderer.close()
    return image

  tiles = []
  for index in range(0, len(frames), max(1, len(frames) // 4))[:4]:
    q23 = np.zeros(model23.nq)
    q23[:7] = frames[index][:7]
    for name, adr in qposadr23.items():
      jid = mujoco.mj_name2id(model23, mujoco.mjtObj.mjOBJ_JOINT, name)
      q23[model23.jnt_qposadr[jid]] = frames[index][adr]
    tiles.append((shot(model23, q23), shot(model29, arr29[index])))
  width, height = tiles[0][0].size
  sheet = Image.new("RGB", (width * 2, height * len(tiles)), "white")
  for row, (left, right) in enumerate(tiles):
    sheet.paste(left, (0, row * height))
    sheet.paste(right, (width, row * height))
  sheet.save(path)
  print(f"wrote {path} (left column 23-DoF, right column 29-DoF)")


if __name__ == "__main__":
  main()
