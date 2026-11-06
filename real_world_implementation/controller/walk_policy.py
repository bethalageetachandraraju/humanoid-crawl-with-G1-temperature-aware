"""Walk policy (ONNX) with the exact training observation (context §6.2).

obs    = [gravity_b(3), gyro_b(3), cmd(3) = vx vy wz, q - STAND(25), dq(25),
          last_action(25)]                                   -> 84 values
action = clip(onnx(obs), -5, 5)        (the clipped action is fed back)
target = STAND + scale * action
"""

from __future__ import annotations

import numpy as np
from g1_config import (
  JOINTS,
  NUM_JOINTS,
  STAND_POSE,
  WALK_ACTION_SCALE,
  WALK_CLIP,
  WALK_CMD_HI,
  WALK_CMD_LO,
  WALK_OBS_DIM,
)


def gravity_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
  """Gravity unit vector in the IMU (= pelvis) frame: R^T [0, 0, -1]."""
  w, x, y, z = q_wxyz / np.linalg.norm(q_wxyz)
  return np.array(
    [-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))]
  )


class WalkPolicy:
  def __init__(self, path: str) -> None:
    import onnxruntime as ort  # only the walk-only controller needs it

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    inp, out = self.sess.get_inputs()[0], self.sess.get_outputs()[0]
    if inp.shape[-1] != WALK_OBS_DIM or out.shape[-1] != NUM_JOINTS:
      raise ValueError(f"{path}: {inp.shape} -> {out.shape}, expected 84 -> 25")
    self.input_name = inp.name
    self._check_metadata(path)
    self.path = path
    self.last_action = np.zeros(NUM_JOINTS)

  def _check_metadata(self, path: str) -> None:
    meta = self.sess.get_modelmeta().custom_metadata_map

    def floats(key: str) -> np.ndarray | None:
      return np.array([float(v) for v in meta[key].split(",")]) if key in meta else None

    if "joint_names" in meta and meta["joint_names"].split(",") != list(JOINTS):
      raise ValueError(f"{path}: joint order in metadata differs from JOINTS")
    for key, ours in (
      ("default_joint_pos", STAND_POSE),
      ("action_scale", WALK_ACTION_SCALE),
    ):
      v = floats(key)
      if v is not None and np.abs(v - ours).max() > 2e-3:
        raise ValueError(f"{path}: metadata {key} differs from g1_config")

  def reset(self) -> None:
    self.last_action[:] = 0.0

  def step(
    self,
    gravity: np.ndarray,
    gyro: np.ndarray,
    cmd: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (target joint positions, clipped action, observation)."""
    cmd = np.clip(cmd, WALK_CMD_LO, WALK_CMD_HI)
    obs = np.concatenate([gravity, gyro, cmd, q - STAND_POSE, dq, self.last_action])
    raw = self.sess.run(None, {self.input_name: obs.astype(np.float32)[None]})[0][0]
    if not np.all(np.isfinite(raw)):
      raise FloatingPointError("policy output is not finite")
    action = np.clip(raw.astype(np.float64), -WALK_CLIP, WALK_CLIP)
    self.last_action = action
    return STAND_POSE + WALK_ACTION_SCALE * action, action, obs
