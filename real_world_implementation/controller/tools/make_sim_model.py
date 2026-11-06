"""Write the bundle's braced G1 as a standalone MJCF for the DDS sim bridge.

Source: the training model shipped in the bundle (bundles/g1_chain_v3/mujoco/
scene.xml by default). Its contact sensors (sim-only monitors, MuJoCo >= 3.5)
are dropped so older MuJoCo builds load it, and a pelvis-IMU orientation
sensor is added for the bridge's rt/lowstate. The PD position
actuators become plain torque motors: the bridge runs the motor-side PD
(tau = kp (q* - q) + kd (dq* - dq) + tau_ff) itself, like the G1 drivers, at
1 kHz.

  python controller/tools/make_sim_model.py [--bundle bundles/<name>]
  # writes controller/assets/g1_29dof_braced_chain_sim.xml

Re-run when a new bundle changes the robot model (mass, geoms, joints).
"""

import argparse
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # real_world_implementation/
OUT = HERE.parent / "assets" / "g1_29dof_braced_chain_sim.xml"


def _to_motors(spec: mujoco.MjSpec) -> None:
  for act in spec.actuators:
    tau = float(act.forcerange[1])
    act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    act.gainprm[:] = 0
    act.gainprm[0] = 1.0
    act.biastype = mujoco.mjtBias.mjBIAS_NONE
    act.biasprm[:] = 0
    act.ctrllimited = True
    act.ctrlrange[:] = (-tau, tau)
    act.forcelimited = False


def from_bundle(xml: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml))
  spec.meshdir = str(xml.parent / "assets")
  for s in list(spec.sensors):
    if s.type == mujoco.mjtSensor.mjSENS_CONTACT:
      spec.delete(s)
  spec.add_sensor(
    name="robot/imu_quat",
    type=mujoco.mjtSensor.mjSENS_FRAMEQUAT,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="robot/imu_in_pelvis",
  )
  return spec


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--bundle", default=str(ROOT / "bundles" / "g1_chain_v3"))
  ap.add_argument("--out", default=str(OUT))
  args = ap.parse_args()
  src = Path(args.bundle) / "mujoco" / "scene.xml"
  spec = from_bundle(src)
  spec.option.timestep = 0.001
  _to_motors(spec)
  spec.compile()
  out = Path(args.out)
  out.write_text(spec.to_xml())
  m = mujoco.MjModel.from_xml_path(str(out))
  pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/pelvis")
  if pelvis < 0:
    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  print(f"wrote {out} from {src}: {m.nu} motors, {m.body_subtreemass[pelvis]:.2f} kg")


if __name__ == "__main__":
  main()
