"""Braced G1 29-DoF: joint order, motor mapping, PD gains, limits, safety numbers.

Everything here must match what the policies were trained with; the
controller asserts it against the bundle's chain.json at start-up
(g1_chain_real.load_bundle), and ``python g1_config.py`` prints the check.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CHAIN_JSON = next(
  (
    p
    for p in (
      HERE / "bundle" / "chain.json",  # g1_onboard/ on the robot
      HERE.parent / "bundles" / "g1_chain_v3" / "chain.json",
    )
    if p.exists()
  ),
  HERE.parent / "bundles" / "g1_chain_v3" / "chain.json",
)

# --- Joints: policy order (= MuJoCo order of the training model) -------------
JOINTS = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
)
NUM_JOINTS = len(JOINTS)

# --- Unitree G1 29-DoF motor indices (G1JointIndex in the SDK example) --------
NUM_MOTORS = 29  # LowCmd_ has 35 slots; 29..34 unused on the G1.
# policy 0-19 -> motor 0-19, policy 20-24 -> motor 22-26.
MOTOR_OF_JOINT = np.array(list(range(20)) + [22, 23, 24, 25, 26])
# Wrist pitch / yaw, locked by the 3D-printed brace, not driven by any policy.
WRIST_MOTORS = {
  20: "left_wrist_pitch",
  21: "left_wrist_yaw",
  27: "right_wrist_pitch",
  28: "right_wrist_yaw",
}

# --- PD gains: armature * (2 pi 10 Hz)^2, damping ratio 2 (mjlab G1) ---------
_OMEGA = 10 * 2.0 * 3.1415926535


def _reflected(rotor: tuple[float, ...], gear: tuple[float, ...]) -> float:
  return rotor[0] * (gear[1] * gear[2]) ** 2 + rotor[1] * gear[2] ** 2 + rotor[2]


_ARM = {
  "7520_14": _reflected((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + 48 / 22)),
  "7520_22": _reflected((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5)),
  "linkage": 2 * _reflected((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 4.5)),
  "5020": _reflected((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + 46 / 18, 4.5)),
  "4010": _reflected((0.068e-4, 0.0, 0.0), (1, 5, 5)),
}
_PEAK = {"7520_14": 88.0, "7520_22": 139.0, "linkage": 50.0, "5020": 25.0, "4010": 5.0}


def _motor_class(joint: str) -> str:
  if any(k in joint for k in ("hip_pitch", "hip_yaw", "waist_yaw")):
    return "7520_14"
  if any(k in joint for k in ("hip_roll", "knee")):
    return "7520_22"
  if any(k in joint for k in ("ankle", "waist_roll", "waist_pitch")):
    return "linkage"
  return "5020"


KP = np.array([_ARM[_motor_class(j)] * _OMEGA**2 for j in JOINTS])
KD = np.array([4.0 * _ARM[_motor_class(j)] * _OMEGA for j in JOINTS])
PEAK_TORQUE = np.array([_PEAK[_motor_class(j)] for j in JOINTS])
WALK_ACTION_SCALE = 0.25 * PEAK_TORQUE / KP

# Wrist pitch / yaw (4010 motors).
WRIST_DAMP_KD = 0.5  # braced: kp 0, small kd (context §5.3)
WRIST_HOLD_KP = _ARM["4010"] * _OMEGA**2  # no brace fitted: hold at 0 like mjlab G1
WRIST_HOLD_KD = 4.0 * _ARM["4010"] * _OMEGA

# --- Poses --------------------------------------------------------------------
STAND_POSE = np.zeros(NUM_JOINTS)
for _n, _v in {
  "left_hip_pitch_joint": -0.312,
  "right_hip_pitch_joint": -0.312,
  "left_knee_joint": 0.669,
  "right_knee_joint": 0.669,
  "left_ankle_pitch_joint": -0.363,
  "right_ankle_pitch_joint": -0.363,
  "left_shoulder_pitch_joint": 0.2,
  "right_shoulder_pitch_joint": 0.2,
  "left_shoulder_roll_joint": 0.2,
  "right_shoulder_roll_joint": -0.2,
  "left_elbow_joint": 0.6,
  "right_elbow_joint": 0.6,
}.items():
  STAND_POSE[JOINTS.index(_n)] = _v

# --- Joint ranges (MJCF g1_29dof_rev_1_0) ---------------------------------------
_RANGE = {
  "hip_pitch": (-2.5307, 2.8798),
  "left_hip_roll": (-0.5236, 2.9671),
  "right_hip_roll": (-2.9671, 0.5236),
  "hip_yaw": (-2.7576, 2.7576),
  "knee": (-0.087267, 2.8798),
  "ankle_pitch": (-0.87267, 0.5236),
  "ankle_roll": (-0.2618, 0.2618),
  "waist_yaw": (-2.618, 2.618),
  "waist_roll": (-0.52, 0.52),
  "waist_pitch": (-0.52, 0.52),
  "shoulder_pitch": (-3.0892, 2.6704),
  "left_shoulder_roll": (-1.5882, 2.2515),
  "right_shoulder_roll": (-2.2515, 1.5882),
  "shoulder_yaw": (-2.618, 2.618),
  "elbow": (-1.0472, 2.0944),
  "wrist_roll": (-1.97222, 1.97222),
}


def _range(joint: str) -> tuple[float, float]:
  name = joint[: -len("_joint")] if joint.endswith("_joint") else joint  # py3.8
  if name in _RANGE:
    return _RANGE[name]
  return _RANGE[name.split("_", 1)[1]]


JOINT_LOWER = np.array([_range(j)[0] for j in JOINTS])
JOINT_UPPER = np.array([_range(j)[1] for j in JOINTS])


def soft_limits(factor: float) -> tuple[np.ndarray, np.ndarray]:
  mid = 0.5 * (JOINT_LOWER + JOINT_UPPER)
  half = 0.5 * (JOINT_UPPER - JOINT_LOWER) * factor
  return mid - half, mid + half


# --- Walk policy interface (context §6.2) ---------------------------------------
WALK_OBS_DIM = 84
WALK_CLIP = 5.0
WALK_CMD_LO = np.array([-0.6, -0.5, -0.8])
WALK_CMD_HI = np.array([1.0, 0.5, 0.8])
CONTROL_DT = 0.02  # 50 Hz policy
LOWCMD_DT = 0.002  # 500 Hz LowCmd writes (each policy target held 10 writes)

# --- Safety -------------------------------------------------------------------------
MAX_TILT_DEG = 45.0  # walk policy terminated at 70 deg in training
MAX_JOINT_SPEED = 30.0  # rad/s
STATE_TIMEOUT_S = 0.1  # no lowstate for this long -> damping
TEMP_WARN_C = 70
TEMP_STOP_C = 85


def check_against_chain(path: Path = CHAIN_JSON) -> list[str]:
  """Compare the numbers above with chain.json, if present. Returns problems."""
  if not path.exists():
    return [f"{path} not found; using built-in values (identical when last checked)"]
  cfg = json.loads(path.read_text())
  bad = []
  if list(cfg.get("joints", JOINTS)) != list(JOINTS):
    bad.append("joint order differs from chain.json")
  for i, j in enumerate(JOINTS):
    pd = cfg.get("pd", {}).get(j)
    if pd is None:
      continue
    for key, ours in (("kp", KP[i]), ("kd", KD[i]), ("peak_nm", PEAK_TORQUE[i])):
      if not math.isclose(pd[key], ours, rel_tol=1e-4):
        bad.append(f"{j} {key}: chain {pd[key]:.4f} vs ours {ours:.4f}")
    sp = cfg.get("stand_pose", {}).get(j)
    if sp is not None and abs(sp - STAND_POSE[i]) > 1e-6:
      bad.append(f"{j} stand pose: chain {sp} vs ours {STAND_POSE[i]}")
    sc = cfg.get("policies", {}).get("walk", {}).get("scale", {}).get(j)
    if sc is not None and not math.isclose(sc, WALK_ACTION_SCALE[i], rel_tol=1e-4):
      bad.append(f"{j} walk scale: chain {sc} vs ours {WALK_ACTION_SCALE[i]}")
  return bad


if __name__ == "__main__":
  print(f"{'policy joint':28s} motor   kp      kd     peak  scale   range")
  for i, j in enumerate(JOINTS):
    print(
      f"{j:28s} {MOTOR_OF_JOINT[i]:3d}  {KP[i]:7.3f} {KD[i]:6.3f} {PEAK_TORQUE[i]:6.1f}"
      f"  {WALK_ACTION_SCALE[i]:.4f}  [{JOINT_LOWER[i]:+.3f}, {JOINT_UPPER[i]:+.3f}]"
    )
  print(
    f"wrist motors {sorted(WRIST_MOTORS)}: braced kp 0 kd {WRIST_DAMP_KD}; "
    f"unbraced hold kp {WRIST_HOLD_KP:.3f} kd {WRIST_HOLD_KD:.3f}"
  )
  problems = check_against_chain()
  print("chain.json check:", "OK" if not problems else problems)
