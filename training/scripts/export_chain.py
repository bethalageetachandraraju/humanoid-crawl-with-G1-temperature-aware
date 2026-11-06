"""Bundle the walk, stand-to-crawl and crawl policies for deployment.

Collects the three ONNX actors into one folder with ``chain.json``: joint order,
PD gains and limits, each policy's observation layout and action mapping, the
transition keyframe table (so the robot computes the same reference schedule),
the stand and crawl poses, and the switcher's gate and blend settings::

  uv run python scripts/export_chain.py --walk logs/rsl_rl/g1_walk_29dof/<run> \
      --transition logs/rsl_rl/g1_stand_to_crawl_29dof/<run> --out deploy/g1_chain

The crawl policy is exported from its checkpoint if its run has no ONNX yet.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import onnx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play_switcher import CRAWL_CHECKPOINT, CRAWL_TASK  # noqa: E402

from mjlab.tasks.g1_crawl.robot29 import (  # noqa: E402
  BRACED_LOADS_29,
  BRACED_TORQUE_LIMIT,
  MOTORS_29,
)
from mjlab.tasks.g1_walk.config.g1.env_cfgs import (  # noqa: E402
  TRANSITION_ACTION_CLIP,
  TRANSITION_ACTION_SCALE,
  _action_scale_29,
)
from mjlab.tasks.g1_walk.mdp.keyframes import KEYFRAME_TABLE  # noqa: E402
from mjlab.tasks.g1_walk.robot import (  # noqa: E402
  CRAWL_HEIGHT,
  CRAWL_POSE_29,
  JOINTS_29,
  STAND_POSE_29,
)


def latest_onnx(run: Path) -> Path:
  files = sorted(run.glob("*.onnx")) + sorted(run.glob("exported/*.onnx"))
  if not files:
    raise FileNotFoundError(f"no .onnx in {run}")
  return max(files, key=lambda p: p.stat().st_mtime)


def layout(path: Path) -> dict:
  meta = {e.key: e.value for e in onnx.load(str(path)).metadata_props}
  model = onnx.load(str(path))
  dim = model.graph.input[0].type.tensor_type.shape.dim[1].dim_value
  return {
    "file": path.name,
    "input_dim": dim,
    "observation_names": meta["observation_names"].split(","),
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--walk", required=True, help="walk run directory")
  ap.add_argument("--transition", required=True, help="stand-to-crawl run directory")
  ap.add_argument("--crawl-checkpoint", default=CRAWL_CHECKPOINT)
  ap.add_argument("--out", default="deploy/g1_chain")
  args = ap.parse_args()
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  crawl_run = Path(args.crawl_checkpoint).parent
  try:
    crawl_onnx = latest_onnx(crawl_run)
  except FileNotFoundError:
    subprocess.run(
      [
        sys.executable,
        "scripts/export_policy_onnx.py",
        "--task",
        CRAWL_TASK,
        "--checkpoint",
        args.crawl_checkpoint,
      ],
      check=True,
    )
    crawl_onnx = latest_onnx(crawl_run)
  files = {
    "walk": latest_onnx(Path(args.walk)),
    "transition": latest_onnx(Path(args.transition)),
    "crawl": crawl_onnx,
  }
  for name, src in files.items():
    shutil.copy(src, out / f"{name}.onnx")

  walk_scale = _action_scale_29()
  chain = {
    "control_hz": 50,
    "joints": list(JOINTS_29),
    "pd": {
      j: {"kp": kp, "kd": kd, "peak_nm": peak}
      for j, (kp, kd, peak) in MOTORS_29.items()
    },
    "braced_joints": {
      j: {"site": site, "axis": axis, "limit_nm": BRACED_TORQUE_LIMIT}
      for j, (site, axis) in BRACED_LOADS_29.items()
    },
    "stand_pose": STAND_POSE_29,
    "crawl_pose": CRAWL_POSE_29,
    "crawl_pelvis_height_m": CRAWL_HEIGHT,
    "policies": {
      "walk": {
        **layout(out / "walk.onnx"),
        "obs": "gravity_b(3), gyro_b(3), [vx, vy, wz](3), q - stand_pose(25), qd(25), last_action(25)",
        "target": "stand_pose + scale * clip(action, +-5)",
        "scale": walk_scale,
        "command_ranges": {"vx": [-0.6, 1.0], "vy": [-0.5, 0.5], "wz": [-0.8, 0.8]},
      },
      "transition": {
        **layout(out / "transition.onnx"),
        "obs": "gravity_b(3), gyro_b(3), q - stand_pose(25), qd(25), last_action(25), [phi, 1/T](2), ref(phi) - q(25)",
        "target": f"ref(phi) + {TRANSITION_ACTION_SCALE} * clip(action, +-{TRANSITION_ACTION_CLIP})",
        "schedule": "smoothstep between consecutive keyframes in joint space (see keyframes); phi runs 0 -> 1 over T seconds, T in [4, 8], 6-8 recommended; hold at phi = 1",
        "keyframes": KEYFRAME_TABLE,
      },
      "crawl": {
        **layout(out / "crawl.onnx"),
        "obs": "gravity_b(3), gyro_b(3), [fwd, lat, turn, step_height, 0, 0, 0](7), q - crawl_pose(25), qd(25), last_action(25)",
        "target": "crawl_pose + 0.5 * clip(action, +-5)",
        "command_ranges": {
          "fwd": [-1.0, 2.5],
          "lat": [-0.5, 0.5],
          "turn": [-3.0, 3.0],
          "step_height": [0.02, 0.15],
        },
      },
    },
    "switcher": {
      "states": "WALK -> TRANSITION (phi advances at 1/T per s) -> HOLD -> CRAWL",
      "gate": {
        "dwell_s": 1.0,
        "max_joint_speed_radps": 0.5,
        "pelvis_height_tol_m": 0.03,
        "gravity_dir_tol": 0.15,
        "contacts": "both feet and both extension tips",
      },
      "blend_s": 0.5,
      "shadow": "run every policy on live observations before it is in charge, so its last-action input is its own",
      "braced_wrist_motors": "damping mode (kp 0, small kd) in every state",
    },
    "sources": {k: str(v) for k, v in files.items()},
  }
  (out / "chain.json").write_text(json.dumps(chain, indent=2))
  print(f"wrote {out}/: walk.onnx, transition.onnx, crawl.onnx, chain.json")


if __name__ == "__main__":
  main()
