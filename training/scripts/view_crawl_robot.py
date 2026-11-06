"""Open the crawl robot in the MuJoCo viewer, posed in its crawl initial state.

The robot is built through MJLab's ``Entity`` exactly as the training task builds
it (collision settings, position actuators, ``init_state`` keyframe), then a floor
and a light are added.

Two modes:

* default (static): gravity and contacts off, actuators holding the crawl pose,
  so the robot stays put for inspection. Toggle geom groups with the number keys
  in the viewer: 2 = visual meshes, 3 = collision geoms (green).
* ``--physics``: gravity and contacts on, actuators holding the crawl pose, i.e.
  what the policy sees with action = 0.

Usage::

  uv run python scripts/view_crawl_robot.py              # 29-DoF, static
  uv run python scripts/view_crawl_robot.py --physics    # 29-DoF, gravity on
  uv run python scripts/view_crawl_robot.py --dof23      # source 23-DoF robot
"""

from __future__ import annotations

import argparse

import mujoco
import mujoco.viewer

from mjlab.entity import Entity
from mjlab.tasks.g1_crawl.robot import get_g1_crawl_robot_cfg
from mjlab.tasks.g1_crawl.robot29 import get_g1_crawl_robot_cfg_29


def build(dof23: bool) -> mujoco.MjModel:
  robot = Entity(get_g1_crawl_robot_cfg() if dof23 else get_g1_crawl_robot_cfg_29())
  spec = robot.spec
  spec.visual.headlight.ambient[:] = [0.4, 0.4, 0.4]
  spec.add_texture(
    name="floor",
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
    rgb1=[0.82, 0.84, 0.86],
    rgb2=[0.70, 0.72, 0.75],
    width=300,
    height=300,
  )
  floor = spec.add_material(name="floor")
  floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "floor"
  floor.texrepeat[:] = [8, 8]
  spec.worldbody.add_geom(
    name="floor",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=[5, 5, 0.1],
    material="floor",
  )
  spec.worldbody.add_light(pos=[1, -1, 3], dir=[-0.3, 0.3, -1], diffuse=[0.7] * 3)
  return spec.compile()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--physics", action="store_true", help="Gravity + contacts on.")
  parser.add_argument("--dof23", action="store_true", help="Show the 23-DoF robot.")
  args = parser.parse_args()

  model = build(args.dof23)
  if not args.physics:
    model.opt.disableflags |= (
      mujoco.mjtDisableBit.mjDSBL_GRAVITY | mujoco.mjtDisableBit.mjDSBL_CONTACT
    )
  data = mujoco.MjData(model)
  key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  mujoco.mj_resetDataKeyframe(model, data, key)
  # Position actuators: target = the crawl pose (action = 0 in the task).
  for act in range(model.nu):
    joint = model.actuator_trnid[act, 0]
    data.ctrl[act] = data.qpos[model.jnt_qposadr[joint]]
  mujoco.mj_forward(model, data)

  print(
    f"{'23' if args.dof23 else '29'}-DoF crawl robot: {model.njnt - 1} joints, "
    f"{model.nu} actuators, mass {model.body_mass.sum():.2f} kg, "
    f"{'physics on' if args.physics else 'static (gravity/contacts off)'}"
  )
  mujoco.viewer.launch(model, data)


if __name__ == "__main__":
  main()
