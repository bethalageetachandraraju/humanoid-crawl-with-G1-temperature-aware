"""Build g1_onboard/: everything the robot's own computer needs, nothing else.

  python3 controller/tools/make_onboard_package.py [--bundle DIR] [--out DIR]
  # defaults: --bundle bundles/g1_chain_v3  --out build/g1_onboard

Copies the controller (g1_chain_real.py and the modules it imports), the bundle
with its numpy weights (.npz, no ONNX, so no onnxruntime), scene.xml without
meshes (the controller only reads joint geometry from it), check_robot.py, a
start script and a README. Needs only Python >= 3.8, numpy and unitree_sdk2py
on the robot. Re-run after changing the controller or the bundle.
"""

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CODE = (
  "g1_chain_real.py",
  "g1_walk_real.py",
  "g1_config.py",
  "inputs.py",
  "walk_policy.py",
  "thermal_guard.py",
  "check_robot.py",
)
BUNDLE_FILES = ("chain.json", "chain_controller.py", "selftest.npz", "README.md")

RUN = """#!/bin/bash
# Start the chain controller on this computer, driven by the Unitree remote.
#   ./run_robot.sh                      # asks you to confirm the checklist
#   ./run_robot.sh --max-crawl-fwd 0.35 # extra options go to g1_chain_real.py
# NET: the interface on 192.168.123.x (check_robot.py prints it).
cd "$(dirname "$0")"
# The robot's computer is slower than a laptop: 100 Hz motor commands and the
# newest state once per tick keep the 50 Hz policy on time (500 Hz + every
# message made it run at ~38 Hz and the walk went unstable, 2026-10-31).
exec python3 g1_chain_real.py --real --net "${NET:-eth0}" --wrists braced --remote \
  --loop-hz 100 --poll-state "$@"
"""

BUILD = """#!/bin/bash
# One-time, on the robot: compile the CycloneDDS Python bindings (0.10.2) into
# ./vendor against the CycloneDDS 0.10.2 that Unitree built in ~/cyclonedds_ws.
# Nothing outside this folder changes: no sudo, no system pip, no cache, no
# network. Undo: rm -rf vendor/cyclonedds* vendor/bin (or the whole folder).
set -e
cd "$(dirname "$0")"
export CYCLONEDDS_HOME="${CYCLONEDDS_HOME:-$HOME/cyclonedds_ws/install/cyclonedds}"
for need in include bin lib/libddsc.so; do
  if [ ! -e "$CYCLONEDDS_HOME/$need" ]; then
    echo "missing $CYCLONEDDS_HOME/$need: set CYCLONEDDS_HOME to the CycloneDDS 0.10.2 install"
    exit 1
  fi
done
echo "building against $CYCLONEDDS_HOME"
python3 -m pip install --no-deps --no-build-isolation --no-index --no-cache-dir \
  --target vendor vendor_src/cyclonedds-0.10.2.tar.gz
python3 check_robot.py
"""

README = """# g1_onboard: the walk <-> crawl controller on the G1's own computer

Built by `real_world_implementation/controller/tools/make_onboard_package.py`
on the laptop.
Needs on the robot: Python >= 3.8, numpy, unitree_sdk2py (+ cyclonedds).
No onnxruntime, MuJoCo or torch: the policies run in numpy (`bundle/*.npz`,
checked against onnxruntime to 3e-5).

## Once: copy, build the DDS bindings into this folder, check

From the laptop (robot on the cable or the same WiFi; `unitree@<robot ip>`):

    scp -r build/g1_onboard unitree@192.168.123.164:~/   # from real_world_implementation/

On the robot (ssh unitree@192.168.123.164):

    cd ~/g1_onboard && ./build_vendor.sh

That compiles the CycloneDDS Python bindings into `vendor/` (about a minute,
nothing installed system-wide) and runs `check_robot.py`. It must end with READY and prints the network interface to use (`--net`).
Then, read-only (publishes nothing):

    python3 g1_chain_real.py --real --net eth0 --monitor --remote

Press remote buttons: they show in the status. Tilt the robot: gravity moves.

## Run

    cd ~/g1_onboard && NET=eth0 ./run_robot.sh --max-crawl-fwd 0.35

Type `yes` and `release` in the ssh session as on the laptop, then drive with
the remote. If ssh or WiFi drops, the controller keeps running (the remote
still works; B = damping). `kill <pid>` stops it cleanly (damping, log saved).

| Remote | Action |
|---|---|
| **B or F1** | **DAMPING (e-stop)** |
| Start | stand up slowly to the stand pose (hanging) |
| R1 + X | start the policies (stand / walk) |
| left stick | forward / back / left / right (walk and crawl) |
| right stick | rotate (right = clockwise) |
| R1 + Down | stand -> crawl |
| R1 + Up | crawl -> stand |
| tap R1 / R2 | crawl step height up / down (0.02 m) |
| L1 / L2 | speed cap up / down (0.1 m/s; walk or crawl, whichever runs) |
| Select | quit: damping, log saved |

Hold R1, then press X / Up / Down; a tap of R1 alone changes the step height.
The ssh terminal keys still work too (Space = damping, c, u, v, h, g).
Thermal guard, HOLD timeout and logging are the same as on the laptop. Logs:
`~/g1_onboard/logs/chain_real_*.npz`; copy them back with
`scp unitree@192.168.123.164:~/g1_onboard/logs/*.npz controller/logs/` (from
real_world_implementation/ on the laptop).
"""


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--bundle", default=str(HERE.parent / "bundles" / "g1_chain_v3"))
  ap.add_argument("--out", default=str(HERE.parent / "build" / "g1_onboard"))
  args = ap.parse_args()
  src, out = Path(args.bundle), Path(args.out)
  if out.exists():
    shutil.rmtree(out / "bundle", ignore_errors=True)
  out.mkdir(parents=True, exist_ok=True)
  (out / "bundle" / "mujoco").mkdir(parents=True, exist_ok=True)
  for f in CODE:
    shutil.copy2(HERE / f, out / f)
  for f in BUNDLE_FILES:
    shutil.copy2(src / f, out / "bundle" / f)
  npz = sorted(p for p in src.glob("*.npz") if p.name != "selftest.npz")
  assert npz, f"no policy .npz in {src}: run tools/onnx_to_npz.py first"
  for p in npz:
    shutil.copy2(p, out / "bundle" / p.name)
  shutil.copy2(src / "mujoco" / "scene.xml", out / "bundle" / "mujoco" / "scene.xml")
  # Unitree's SDK (pure Python + its ARM crc library), no install needed: the
  # copy in third_party/ (unitree_sdk2_python 1.0.1, BSD-3).
  sdk = HERE / "third_party" / "unitree_sdk2_python" / "unitree_sdk2py"
  assert sdk.is_dir(), f"missing {sdk}"
  shutil.rmtree(out / "vendor" / "unitree_sdk2py", ignore_errors=True)
  shutil.copytree(
    sdk,
    out / "vendor" / "unitree_sdk2py",
    ignore=shutil.ignore_patterns("__pycache__", "test", "*.pyc"),
  )
  # Pure-Python extras for Python 3.8 (typing_extensions 4.12.2) and the
  # CycloneDDS Python bindings' source, compiled on the robot by build_vendor.sh.
  for f in (HERE / "vendor_extra").glob("*.py"):
    shutil.copy2(f, out / "vendor" / f.name)
  (out / "vendor_src").mkdir(exist_ok=True)
  for f in (HERE / "vendor_src").glob("*.tar.gz"):
    shutil.copy2(f, out / "vendor_src" / f.name)
  (out / "build_vendor.sh").write_text(BUILD)
  (out / "build_vendor.sh").chmod(0o755)
  (out / "run_robot.sh").write_text(RUN)
  (out / "run_robot.sh").chmod(0o755)
  (out / "README.md").write_text(README)
  size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
  print(f"wrote {out} ({size / 1e6:.1f} MB): {len(CODE)} scripts, bundle with "
        f"{', '.join(p.name for p in npz)}")  # fmt: skip


if __name__ == "__main__":
  main()
