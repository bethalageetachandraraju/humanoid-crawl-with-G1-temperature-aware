"""Generate the 3D-printed wrist brace and forearm extension for the crawl G1.

Design 3 (split clamp, loose fit: v2 jammed on the real wrist). One part per arm, printed as two halves split on the
plane y = 0 of the wrist-roll frame (x along the forearm, y along the wrist-pitch
axis):

- cap A (+y), the light half, sleeve only;
- carrier B (-y), sleeve plus the forearm extension with its hemispherical tip,
  in one piece.

The brace locks wrist pitch and yaw at zero, the pose the simulation's braced
model assumes. It fits by clamping, not by interference:

- PETG stays ``RELIEF`` mm off the wrist everywhere, drafted so each half drops
  straight on and lifts straight off. Each half touches the wrist only on three
  V-shaped lands (``LANDS``), through conformal TPU pads.
- Six M4 screws pull the halves together across a seam gap that never closes
  (1.5 mm nominal), so size errors in the print change only the gap, not the
  fit, and the screw torque sets the clamp force.
- The relief also covers the wrist turned by up to ``CAPTURE_PITCH`` /
  ``CAPTURE_YAW`` degrees, so the halves still go on when the wrist is a little
  off zero, and the lands push it to zero as the screws close.

The outside follows ``mjlab/tasks/g1_crawl/robot29.py``: a sleeve of radius
``BRACE_RADIUS`` from ``BRACE_START`` to the wrist-yaw flange, then the extension
(``EXTENSION_RADIUS``, ``EXTENSION_LENGTH``) with its dome, the ground contact.
Two print flats at |y| = 37 and the seam gap sit inside that envelope. The 15 mm
taper between sleeve and extension is not in the simulation's collision model,
but it lies inside the convex hull of it, so flat-ground contact is unchanged.

Design 3 over design 2: the wrist did not go in. It now has room:

- ``RELIEF`` 2.5 mm (was 1.5) and the pad lands 2.0 mm off the wrist (was 1.0), so
  the bare plastic is a full millimetre further away and the seam can close on a
  wrist that is bigger than Unitree's CAD;
- an end pocket: the cavity continues ``END_GAP`` mm past the wrist-yaw flange (and
  ``RELIEF`` more all round), at the end disc's radius, so nothing at the end of the
  wrist (hand-mount screws, cable, a thicker flange) can bottom out on the wall;
- the sleeve end, taper, distal screws, key and distal fit ring move out by
  ``TAIL`` to make room for that pocket. The tip stays where it was, so the
  extension is ``TAIL`` shorter; the simulation's sleeve is ``TAIL`` shorter than
  the printed one, which is only ground contact of a 40 mm sleeve 9 mm further on;
- pads come thicker (``PAD_THICKNESSES``). The thinnest leaves a gap you can fill
  with tape or shims for a firm fit.

Method: signed distance fields on a voxel grid. Each link's distance comes from
Euclidean distance transforms of its occupancy, and the turned poses from
rotating those fields. The outer body is analytic, booleans are min/max, and
meshes come from marching cubes repaired and simplified by manifold3d. All
lengths are in mm.

Outputs in ``hardware/wrist_brace/``, in print orientation (flat face down):

- ``{side}_A_cap.stl``, ``{side}_B_carrier.stl``
- ``{side}_pads_{t}mm.stl``: the six TPU pads for that arm (L1 and L2 flat,
  to be bent on; L4 shaped), with A/B tags
- ``{side}_fitring_{proximal,distal}_{A,B}.stl``
- ``gap_gauge.stl``
- section pictures and ``report.txt``

Usage::

  uv run --with scikit-image python scripts/make_wrist_brace.py
  ... --sides left --cache /tmp/brace_cache   # one arm, reuse wrist fields
"""

from __future__ import annotations

import argparse
import functools
from pathlib import Path

import manifold3d
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from scipy import ndimage  # noqa: E402
from skimage import measure  # noqa: E402

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML  # noqa: E402
from mjlab.tasks.g1_crawl import robot29  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "hardware" / "wrist_brace"
LINKS = ("wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link")

# Outer geometry, from the simulation model.
BRACE_START = robot29.BRACE_START * 1000  # Past the wrist-roll joint.
BRACE_RADIUS = robot29.BRACE_RADIUS * 1000
EXT_RADIUS = robot29.EXTENSION_RADIUS * 1000
EXT_LENGTH = robot29.EXTENSION_LENGTH * 1000
TAPER = 15.0  # Sleeve-to-extension taper after the flange.
FLAT = 37.0  # Print flats at |y| = FLAT on the sleeve.

# Fit.
RELIEF = 2.5  # Bare PETG never comes closer to the wrist than this.
RELIEF_DRAFT = 2.0  # deg, opening toward the seam.
CAPTURE_PITCH = 3.0  # deg: halves still drop on with the wrist turned this far,
CAPTURE_YAW = 3.0  # keeping CAPTURE_MARGIN of bare clearance.
CAPTURE_MARGIN = 0.5
END_GAP = 6.0  # Cavity past the wrist-yaw flange (RELIEF comes on top of it).
TAIL = 9.0  # Sleeve end, taper and distal hardware sit this far past the flange.
PAD = 2.0  # Land offset from the wrist; a pad this thick just touches it.
PAD_NOMINAL = 1.8  # Start here: 0.2 mm of free play, the seam takes up the rest.
PAD_THICKNESSES = (1.4, 1.8, 2.2)  # Thinner if tight; thicker (or tape) if loose.
PAD_INSET = 0.5  # Pads stop this short of the land edges.
LAND_NMIN = 0.5  # Lands only where the wrist faces the closing direction.
# Land zones, per half: x range, z range, s*y range (s = +1 for A, -1 for B).
LANDS = {
  "L1": (12.0, 48.0, -17.0, 17.0, 20.0, 40.0),  # Roll-link arm arc.
  "L2": (72.0, 96.0, -17.0, 17.0, 20.0, 40.0),  # Yaw-motor housing.
  "L4": (111.0, 124.0, -22.0, 22.0, 6.0, 40.0),  # Distal yaw ring + chamfer.
}
# Pads print with their curvature axis vertical: this axis goes up.
PAD_UP = {"L1": 0, "L2": 2, "L4": 0}

# Split.
SEAM = 0.75  # A at y >= SEAM, B at y <= -SEAM.
A_END = 16.5 + TAIL  # A ends this far past the flange; B is full round beyond B_FULL.
B_FULL = 17.0 + TAIL
SHOULDER_FILLET = 3.0  # Where B's half section meets its full-round extension.
LEAD_IN = 1.0  # 45 deg chamfer on the cavity edges at the seam.
EDGE_CHAMFER = 0.5  # Outer seam edges.

# Fasteners: M4 x 20 socket head, DIN 125 washer, M4 x 8.1 heat-set insert.
# Axes along y at (x, z).
SCREWS = (
  (19.0, 33.5),
  (19.0, -33.5),
  (60.0, 33.2),
  (60.0, -33.2),
  (134.5 + TAIL - 2.0, 18.0),
  (134.5 + TAIL - 2.0, -18.0),
)
COUNTERBORE_R = 4.8  # In A, washer seat floor at y = COUNTERBORE_FLOOR.
COUNTERBORE_FLOOR = 8.0
HOLE_R = 2.25  # M4 clearance through A.
ORING_R = 2.7  # Recess under the washer for a 3 x 1 mm O-ring (captive screw).
ORING_DEPTH = 1.1
INSERT_R = 2.8  # In B: heat-set insert bore from the seam face,
INSERT_DEPTH = 10.0
MOUTH = 0.8  # with a 45 deg mouth chamfer,
TIP_R = 2.2  # and a relief for the screw tip below it.
TIP_DEPTH = 13.25
# Left/right coding key on B (z range per side), slot in A.
KEY_X = (139.0 + TAIL - 2.0, 143.0 + TAIL - 2.0)
KEY_TOP = 20.0
KEY_Z = {"left": (3.0, 9.0), "right": (-9.0, -3.0)}
SLOT_CLEARANCE = 0.5

LABEL_DEPTH = 0.6
LABEL_HEIGHT = 9.0
LABEL_X = 62.0
FIT_RINGS = {"proximal": (10.0, 32.0), "distal": (106.0, 146.0 + TAIL - 2.0)}
GAUGE = (0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4)  # Whole 0.2 mm layers.
DENSITY_G_CM3 = 1.27  # PETG.
# Printed-mass estimate: this much solid wall, then this fill fraction inside.
WALL_MM = 2.0
INFILL = 0.4
HARDWARE_G = 24.0  # Six screws, washers, inserts and O-rings.
# Single-curvature lands get flat pads (thickness = whole layers), bent on.
FLAT_PADS = ("L1", "L2")


def vmax(*a):
  return functools.reduce(np.maximum, a)


def vmin(*a):
  return functools.reduce(np.minimum, a)


def wrist_meshes(side: str) -> tuple[list[trimesh.Trimesh], float, dict[str, float]]:
  """Wrist link meshes in the wrist-roll frame (mm), flange x, joint x positions."""
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  model = spec.compile()
  data = mujoco.MjData(model)
  data.qpos[3] = 1.0  # All joints zero: wrist pitch/yaw in the braced pose.
  mujoco.mj_kinematics(model, data)
  roll = model.body(f"{side}_{LINKS[0]}").id
  r_rot = data.xmat[roll].reshape(3, 3)
  r_pos = data.xpos[roll]
  meshes = []
  for link in LINKS:
    body = model.body(f"{side}_{link}").id
    geom = next(
      g
      for g in range(model.ngeom)
      if model.geom_bodyid[g] == body
      and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
      and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])
      == f"{side}_{link}"
    )
    mesh_id = model.geom_dataid[geom]
    a, n = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    verts = model.mesh_vert[a : a + n]
    world = verts @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
    local = (world - r_pos) @ r_rot  # Into the wrist-roll frame.
    faces = model.mesh_face[fa : fa + fn]
    mesh = trimesh.Trimesh(local * 1000.0, faces, process=True)
    if not mesh.is_watertight:
      trimesh.repair.fill_holes(mesh)
    meshes.append(mesh)
  flange = float(meshes[2].bounds[1, 0])  # Distal face of the wrist-yaw link.
  joints = {}
  for axis in ("pitch", "yaw"):
    jid = model.joint(f"{side}_wrist_{axis}_joint").id
    joints[axis] = float(((data.xanchor[jid] - r_pos) @ r_rot)[0] * 1000.0)
  return meshes, flange, joints


class Grid:
  """Voxel grid with centres on multiples of the voxel size (trimesh aligns too)."""

  def __init__(self, lo: np.ndarray, hi: np.ndarray, voxel: float) -> None:
    self.v = voxel
    self.i0 = np.floor(lo / voxel).astype(int)
    self.shape = tuple((np.ceil(hi / voxel).astype(int) - self.i0 + 1).tolist())
    self.axes = [(self.i0[k] + np.arange(self.shape[k])) * voxel for k in range(3)]

  def coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
      self.axes[0][:, None, None].astype(np.float32),
      self.axes[1][None, :, None].astype(np.float32),
      self.axes[2][None, None, :].astype(np.float32),
    )

  def x_slice(self, sl: slice) -> Grid:
    """The same grid restricted to x indices ``sl``."""
    sub = Grid.__new__(Grid)
    sub.v = self.v
    sub.i0 = self.i0 + np.array([sl.start, 0, 0])
    sub.axes = [self.axes[0][sl], self.axes[1], self.axes[2]]
    sub.shape = (len(sub.axes[0]), self.shape[1], self.shape[2])
    return sub

  def occupancy(self, meshes: list[trimesh.Trimesh]) -> np.ndarray:
    occ = np.zeros(self.shape, dtype=bool)
    for mesh in meshes:
      filled = mesh.voxelized(pitch=self.v).fill()
      idx = np.round(filled.points / self.v).astype(int) - self.i0
      ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
      occ[tuple(idx[ok].T)] = True
    return occ

  def signed_distance(self, occ: np.ndarray) -> np.ndarray:
    """Approximate signed distance (mm) to the occupied set, negative inside."""
    half = self.v / 2
    outside = ndimage.distance_transform_edt(~occ).astype(np.float32) * self.v
    inside = ndimage.distance_transform_edt(occ).astype(np.float32) * self.v
    return np.where(occ, -(inside - half), outside - half)

  def lookup(self, field: np.ndarray, pts: np.ndarray, order: int = 1) -> np.ndarray:
    """Interpolate a field at points (mm)."""
    idx = (pts / self.v - self.i0).T
    return ndimage.map_coordinates(field, idx, order=order, mode="nearest")

  def moved(self, f: np.ndarray, rot: np.ndarray, shift: np.ndarray, nx: int):
    """Field of a body moved by p -> rot p + shift, on the first nx x-slices."""
    inv = rot.T
    offset = inv @ (self.i0 * self.v - shift) / self.v - self.i0
    return ndimage.affine_transform(
      f, inv, offset=offset, output_shape=(nx, *f.shape[1:]), order=1, mode="nearest"
    )


def rotation(axis: int, deg: float) -> np.ndarray:
  c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
  if axis == 1:
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrist_pose(joints: dict[str, float], pitch: float, yaw: float):
  """Rigid motions (rot, shift) of the pitch and yaw links for a wrist pose."""
  cp = np.array([joints["pitch"], 0.0, 0.0])
  cy = np.array([joints["yaw"], 0.0, 0.0])
  rp, ry = rotation(1, pitch), rotation(2, yaw)
  pitch_link = (rp, cp - rp @ cp)
  # Yaw about its own axis, then the pitch turn carries it too.
  yaw_link = (rp @ ry, rp @ (cy - ry @ cy - cp) + cp)
  return pitch_link, yaw_link


def drafted_sweep(c: np.ndarray, s: int, deg: float, v: float) -> np.ndarray:
  """Open a cavity field toward the seam: the half lifting off along s*y clears.

  acc_j = min(c_j, acc_(j+s) - v tan(deg)): every voxel on the seam side of the
  cavity is cavity too, widening by tan(deg) per unit of travel (draft).
  """
  out = c.copy()
  step = v * np.tan(np.radians(deg))
  n = c.shape[1]
  for j in range(n - 2, -1, -1) if s > 0 else range(1, n):
    np.minimum(out[:, j, :], out[:, j + s, :] - step, out=out[:, j, :])
  return out


def chamfer_and(f1: np.ndarray, f2: np.ndarray, c: float) -> np.ndarray:
  """Intersection of two SDFs with a 45 deg chamfer of size c on the edge."""
  return np.maximum(np.maximum(f1, f2), (f1 + f2 + c) / np.sqrt(2))


def round_union(a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
  """Union of two SDFs with a fillet of radius r in the concave corner."""
  u = np.maximum(r - a, 0)
  w = np.maximum(r - b, 0)
  return np.maximum(r, np.minimum(a, b)) - np.sqrt(u**2 + w**2)


def box(x, y, z, x0, x1, y0, y1, z0, z1):
  return vmax(x0 - x, x - x1, y0 - y, y - y1, z0 - z, z - z1)


def text_sdf(text: str, u: np.ndarray, w: np.ndarray, height: float) -> np.ndarray:
  """2-D signed distance (mm) to text centred at u = w = 0, ``height`` tall.

  ``u``, ``w`` are a regular grid (indexing ij) of the text's right and up axes.
  """
  path = TextPath(
    (0, 0), text, size=10, prop=FontProperties(family="DejaVu Sans", weight="bold")
  )
  ext = path.get_extents()
  scale = height / ext.height
  centre = np.array([(ext.x0 + ext.x1) / 2, (ext.y0 + ext.y1) / 2])
  pts = np.c_[u.ravel(), w.ravel()] / scale + centre
  inside = np.zeros(len(pts), bool)
  for poly in path.to_polygons():  # Even-odd rule: glyph holes come out.
    inside ^= MplPath(poly).contains_points(pts)
  inside = inside.reshape(u.shape)
  du = float(abs(u[1, 0] - u[0, 0]))
  out = ndimage.distance_transform_edt(~inside) * du
  inn = ndimage.distance_transform_edt(inside) * du
  return np.where(inside, -inn, out).astype(np.float32)


def to_mesh(sdf: np.ndarray, g: Grid, tolerance: float = 0.02) -> trimesh.Trimesh:
  """Marching cubes, then manifold3d to make it watertight and simplify it.

  The small bias avoids exactly-zero voxels, which make non-manifold output.
  """
  padded = np.pad(sdf + 1e-3, 1, constant_values=1.0)
  verts, faces, _, _ = measure.marching_cubes(padded, level=0.0, spacing=(g.v,) * 3)
  verts = verts + (g.i0 - 1) * g.v
  raw = trimesh.Trimesh(verts, faces, process=True)
  solid = manifold3d.Manifold(
    manifold3d.Mesh(
      vert_properties=np.asarray(raw.vertices, np.float32),
      tri_verts=np.asarray(raw.faces, np.uint32),
    )
  ).simplify(tolerance)
  out = solid.to_mesh()
  mesh = trimesh.Trimesh(out.vert_properties[:, :3], out.tri_verts, process=True)
  mesh.fix_normals()
  return mesh


def print_pose(mesh: trimesh.Trimesh, up: np.ndarray) -> trimesh.Trimesh:
  """Rotate so ``up`` (wrist frame) points +Z, then rest it on Z = 0."""
  m = mesh.copy()
  m.apply_transform(trimesh.geometry.align_vectors(up, [0, 0, 1]))
  m.apply_translation([-m.centroid[0], -m.centroid[1], -m.bounds[0, 2]])
  return m


# ---------------------------------------------------------------------------
# Geometry.


def link_fields(side, g, meshes, cache: Path | None) -> list[np.ndarray]:
  """Signed distance to each wrist link at zero pose (roll, pitch, yaw)."""
  path = cache / f"{side}_links_{g.v}.npz" if cache else None
  if path and path.exists():
    data = np.load(path)
    if tuple(data["shape"]) == g.shape:
      return [data[k] for k in LINKS]
  fields = [g.signed_distance(g.occupancy([m])) for m in meshes]
  if path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, shape=np.array(g.shape), **dict(zip(LINKS, fields, strict=True)))
  return fields


def capture_envelope(g, fields, joints, flange, pitch_max, yaw_max) -> np.ndarray:
  """Distance to the wrist swept over the capture range of pitch and yaw."""
  roll, pitch, yaw = fields
  env = vmin(roll, pitch, yaw)
  nx = int(np.searchsorted(g.axes[0], flange + 12.0))
  head = env[:nx]
  pitches = np.arange(-pitch_max, pitch_max + 1e-6, 0.5) if pitch_max > 0 else [0.0]
  yaws = np.linspace(-yaw_max, yaw_max, 5) if yaw_max > 0 else [0.0]
  for p in pitches:
    for k, yw in enumerate(yaws):
      if p == 0 and yw == 0:
        continue
      pitch_link, yaw_link = wrist_pose(joints, p, yw)
      if k == 0 and p != 0:
        np.minimum(head, g.moved(pitch, *pitch_link, nx), out=head)
      np.minimum(head, g.moved(yaw, *yaw_link, nx), out=head)
  return env


def land_zones(g: Grid, ny: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
  """Per half: each land's zone box SDF (mm) and "facing" (<= 0 where the wrist
  faces the closing direction; roughly mm on these ~30 mm radii)."""
  x, y, z = g.coords()
  out: dict[str, dict[str, np.ndarray]] = {}
  for half, s in (("A", 1), ("B", -1)):
    out[half] = {"facing": (30.0 * (LAND_NMIN - s * ny)).astype(np.float32)}
    for name, (x0, x1, z0, z1, y0, y1) in LANDS.items():
      ylo, yhi = (y0, y1) if s > 0 else (-y1, -y0)
      out[half][name] = box(x, y, z, x0, x1, ylo, yhi, z0, z1).astype(np.float32)
  return out


def build_side(side: str, args) -> dict:
  meshes, flange, joints = wrist_meshes(side)
  tip = flange + EXT_LENGTH
  margin = 3.0
  r_box = BRACE_RADIUS + margin
  lo = np.array([BRACE_START - margin, -r_box, -r_box])
  hi = np.array([tip + margin, r_box, r_box])
  g = Grid(lo, hi, args.voxel)
  print(f"[{side}] grid {g.shape}, flange at x = {flange:.1f} mm", flush=True)
  fields = link_fields(side, g, meshes, args.cache)
  dw = vmin(*fields)
  env = capture_envelope(
    g, fields, joints, flange, args.capture_pitch, args.capture_yaw
  )
  del fields
  print(f"[{side}] wrist fields and capture envelope done", flush=True)

  # Normals from the smoothed field; lands where the wrist faces each half.
  smooth = ndimage.gaussian_filter(dw, 1.0)
  grad = np.gradient(smooth)
  norm = np.sqrt(grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2)
  ny = (grad[1] / np.maximum(norm, 1e-6)).astype(np.float32)
  del smooth, grad, norm
  zones = land_zones(g, ny)
  del ny
  # Footprint of each land: <= 0 on it (pads, checks).
  masks = {
    h: {n: np.maximum(zones[h][n], zones[h]["facing"]) for n in LANDS} for h in zones
  }

  x, y, z = g.coords()
  r = np.sqrt(y**2 + z**2)
  # Outer body: sleeve, taper, extension, dome, print flats.
  dome_x = tip - EXT_RADIUS
  sleeve_end = flange + TAIL
  taper_r = BRACE_RADIUS + (EXT_RADIUS - BRACE_RADIUS) * np.clip(
    (x - sleeve_end) / TAPER, 0, 1
  )
  body = vmin(
    vmax(r - BRACE_RADIUS, BRACE_START - x, x - sleeve_end),
    vmax(r - taper_r, sleeve_end - x, x - (sleeve_end + TAPER)),
    vmax(r - EXT_RADIUS, sleeve_end - x, x - dome_x),
    np.sqrt((x - dome_x) ** 2 + r**2) - EXT_RADIUS,
  )
  body = np.maximum(body, np.where(x < flange + A_END, np.abs(y) - FLAT, -1e3))
  # B: the -y half plus everything round past the shoulder, with a fillet in
  # the corner. A: the +y half, 0.5 mm off B at the shoulder.
  region = {"B": round_union(y + SEAM, flange + B_FULL - x, SHOULDER_FILLET)}
  region["A"] = vmax(SEAM - y, x - (flange + A_END), 0.5 - region["B"])

  # End pocket: the wrist-yaw end disc, continued END_GAP past the flange.
  yaw_end = meshes[2].vertices[meshes[2].vertices[:, 0] > flange - 6.0]
  end_r = float(np.sqrt(yaw_end[:, 1] ** 2 + yaw_end[:, 2] ** 2).max())
  pocket = vmax(r - end_r, (flange - 1.0) - x, x - (flange + END_GAP))
  print(
    f"[{side}] end pocket: radius {end_r:.1f} + {RELIEF}, to x = "
    f"{flange + END_GAP + RELIEF:.1f} mm",
    flush=True,
  )
  relief_target = np.minimum(
    np.minimum(dw, pocket) - RELIEF, np.minimum(env, pocket) - CAPTURE_MARGIN
  )
  del env, pocket
  # Extra local relief for bumps on the real robot that the model lacks.
  no_land = np.zeros(g.shape, bool)
  for x0, x1, y0, y1, z0, z1, depth in args.relief_box or ():
    inside = box(x, y, z, x0, x1, y0, y1, z0, z1) <= 0
    deeper = np.minimum(relief_target, dw - RELIEF - depth)
    relief_target = np.where(inside, deeper, relief_target)
    no_land |= inside
  halves = {}
  for half, s in (("A", 1), ("B", -1)):
    relief = drafted_sweep(relief_target, s, RELIEF_DRAFT, g.v)
    # On a land the cavity stops PAD off the wrist; past the zone box edges the
    # land surface recedes at 45 deg (a ramp, so a turned wrist cams over it).
    land = np.full(g.shape, -1e3, np.float32)
    for name in LANDS:
      zb = zones[half][name]
      ok = (dw <= 5.0) & (zb <= 5.0) & (zones[half]["facing"] <= 0) & ~no_land
      land = np.where(ok, np.maximum(land, dw - PAD - np.maximum(zb, 0)), land)
    cav = np.maximum(relief, land)
    del relief, land, ok
    cav = drafted_sweep(cav, s, 0.0, g.v)  # Undo undercuts the lands made.
    solid = chamfer_and(body, region[half], EDGE_CHAMFER)
    solid = np.maximum(solid, -cav)
    seam = (SEAM - y) if s > 0 else (y + SEAM)
    lead = (-cav + seam + LEAD_IN) / np.sqrt(2)
    halves[half] = np.maximum(solid, np.where(x < flange + END_GAP + 3, lead, -1e3))
    del cav, solid, lead
  del relief_target

  # Screws: counterbore + O-ring recess + clearance in A, insert bore in B.
  cuts: dict[str, list[np.ndarray]] = {"A": [], "B": []}
  for sx, sz in SCREWS:
    rad = np.sqrt((x - sx) ** 2 + (z - sz) ** 2)
    hole = vmin(
      np.broadcast_to(rad - HOLE_R, halves["A"].shape),
      np.maximum(rad - COUNTERBORE_R, COUNTERBORE_FLOOR - y),
      vmax(rad - ORING_R, COUNTERBORE_FLOOR - ORING_DEPTH - y),
    )
    insert = vmin(
      np.maximum(rad - INSERT_R, -(SEAM + INSERT_DEPTH) - y),
      np.maximum(rad - (INSERT_R + MOUTH) - (y + SEAM), -(SEAM + 2.0) - y),
      np.maximum(rad - TIP_R, -(SEAM + TIP_DEPTH) - y),
    )
    insert = np.maximum(insert, y)  # Stops at the seam: nothing cut past it.
    cuts["A"].append(hole.astype(np.float32))
    cuts["B"].append(insert.astype(np.float32))
    halves["A"] = np.maximum(halves["A"], -hole)
    halves["B"] = np.maximum(halves["B"], -insert)

  # Left/right coding key on B and its slot in A.
  kz0, kz1 = KEY_Z[side]
  key = np.maximum(
    box(x, y, z, KEY_X[0], KEY_X[1], -SEAM - 1.0, KEY_TOP, kz0, kz1), r - taper_r
  )
  halves["B"] = np.minimum(halves["B"], key)
  c = SLOT_CLEARANCE
  slot = box(x, y, z, KEY_X[0] - 0.6, KEY_X[1], -5.0, 60.0, kz0 - c, kz1 + c)
  halves["A"] = np.maximum(halves["A"], -slot)

  # Labels, debossed in the flats: read from outside, arrow toward the tip.
  # Seen from +y, right = +x when up = -z; seen from -y, up = +z.
  tag = side[0].upper()
  plane = (g.shape[0], g.shape[2])
  for half, s in (("A", 1), ("B", -1)):
    u = np.broadcast_to(g.axes[0][:, None] - LABEL_X, plane)
    w = np.broadcast_to(-s * g.axes[2][None, :], plane)
    t2 = text_sdf(f"{tag}-{half}  >", u, w, LABEL_HEIGHT)[:, None, :]
    deboss = np.maximum(t2, (FLAT - LABEL_DEPTH) - s * y)
    halves[half] = np.maximum(halves[half], -deboss)

  # Conformal pads: between the land and the wrist, per half, zone, thickness.
  pads = {}
  for half in ("A", "B"):
    for name, m in masks[half].items():
      sel = np.where((m <= 1.0).any(axis=(1, 2)))[0]
      if not len(sel):
        continue
      sl = slice(max(int(sel[0]) - 3, 0), int(sel[-1]) + 4)
      # Only where the land really is behind the pad (the half's surface sits
      # PAD off the wrist, not the deeper relief).
      on_land = halves[half][sl] + dw[sl] - PAD - 0.3
      for t in PAD_THICKNESSES:
        f = vmax(PAD - t - dw[sl], dw[sl] - PAD, m[sl] + PAD_INSET, on_land)
        pads[(half, name, t)] = (sl, f.astype(np.float32))

  return {
    "side": side,
    "g": g,
    "meshes": meshes,
    "flange": flange,
    "joints": joints,
    "dw": dw,
    "halves": halves,
    "cuts": cuts,
    "body": body,
    "pads": pads,
    "land": {h: vmin(*masks[h].values()) for h in ("A", "B")},
  }


def flat_pad(
  arc: float, straight: float, t: float, label: str, v: float = 0.1
) -> trimesh.Trimesh:
  """Flat pad ``arc`` x ``straight`` x ``t`` (Z), for a single-curvature land.

  A 0.2 mm groove along the bend axis (Y) and the land name are in the glue
  face (top), so the pad is bent around the groove, label outward.
  """
  lo = np.array([-1.0, -1.0, -1.0])
  g = Grid(lo, np.array([arc + 1.0, straight + 1.0, t + 1.0]), v)
  x, y, z = g.coords()
  sdf = box(x, y, z, 0.0, arc, 0.0, straight, 0.0, t)
  groove = box(x, y, z, arc / 2 - 0.6, arc / 2 + 0.6, -2.0, straight + 2, t - 0.2, 9)
  sdf = np.maximum(sdf, -groove)
  plane = (g.shape[0], g.shape[1])
  u = np.broadcast_to(g.axes[0][:, None] - arc / 4, plane)
  w = np.broadcast_to(g.axes[1][None, :] - straight / 2, plane)
  t2 = text_sdf(label, u, w, 4.0)[:, :, None]
  sdf = np.maximum(sdf, -np.maximum(t2, (t - 0.2) - z))
  return to_mesh(sdf, g, tolerance=0.01)


def tag(label: str, v: float = 0.1) -> trimesh.Trimesh:
  """8 x 8 x 1 mm tile with a letter: marks which row of pads is A's or B's."""
  g = Grid(np.array([-1.0, -1.0, -1.0]), np.array([9.0, 9.0, 2.0]), v)
  x, y, z = g.coords()
  sdf = box(x, y, z, 0.0, 8.0, 0.0, 8.0, 0.0, 1.0)
  plane = (g.shape[0], g.shape[1])
  u = np.broadcast_to(g.axes[0][:, None] - 4.0, plane)
  w = np.broadcast_to(g.axes[1][None, :] - 4.0, plane)
  t2 = text_sdf(label, u, w, 5.0)[:, :, None]
  sdf = np.maximum(sdf, -np.maximum(t2, 0.6 - z))
  return to_mesh(sdf, g, tolerance=0.01)


def gap_gauge(v: float = 0.1) -> trimesh.Trimesh:
  """Comb of tongues 0.6-3.4 mm thick on a 4 mm handle, numbers debossed."""
  pitch, width, length, handle = 9.0, 6.5, 14.0, 12.0
  span = len(GAUGE) * pitch + 2.0
  g = Grid(np.array([-1.0, -length - 1.0, -1.0]), np.array([span + 1, 13.0, 5.0]), v)
  x, y, z = g.coords()
  sdf = box(x, y, z, 0.0, span, 0.0, handle, 0.0, 4.0)
  plane = (g.shape[0], g.shape[1])
  for k, t in enumerate(GAUGE):
    x0 = 1.0 + k * pitch + (pitch - width) / 2
    sdf = np.minimum(sdf, box(x, y, z, x0, x0 + width, -length, 0.5, 0.0, t))
    u = np.broadcast_to(g.axes[0][:, None] - (x0 + width / 2), plane)
    w = np.broadcast_to(g.axes[1][None, :] - handle / 2, plane)
    t2 = text_sdf(f"{t:g}", u, w, 4.0)[:, :, None]
    sdf = np.maximum(sdf, -np.maximum(t2, 3.4 - z))
  return to_mesh(sdf, g, tolerance=0.01)


# ---------------------------------------------------------------------------
# Checks.


def surface_points(meshes, n: int = 300_000) -> list[np.ndarray]:
  return [
    np.asarray(trimesh.sample.sample_surface_even(m, n, seed=k)[0])
    for k, m in enumerate(meshes)
  ]


def posed(points, joints, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
  """Wrist surface points (roll, pitch, yaw links) in a turned pose."""
  (rp, sp), (ry, sy) = wrist_pose(joints, pitch, yaw)
  roll, pitch_pts, yaw_pts = points
  return np.vstack([roll, pitch_pts @ rp.T + sp, yaw_pts @ ry.T + sy])


def checks(res: dict) -> list[str]:
  side, g = res["side"], res["g"]
  halves, dw, joints = res["halves"], res["dw"], res["joints"]
  flange = res["flange"]
  lines = [f"== {side} =="]
  # Same threshold as the meshes (to_mesh biases the field by 1e-3).
  solid = {h: f < -1e-3 for h, f in halves.items()}
  land = res["land"]
  x = g.axes[0]
  vox = (g.v / 10) ** 3

  # Volumes and the split at the flange (for robot29 BRACE/EXTENSION_MASS).
  for h in ("A", "B"):
    s = solid[h]
    sleeve = s[x < flange].sum() * vox
    ext = s[x >= flange].sum() * vox
    lines.append(
      f"half {h}: solid {sleeve + ext:.1f} cm3 (sleeve {sleeve:.1f}, past the "
      f"flange {ext:.1f}); 100% PETG {(sleeve + ext) * DENSITY_G_CM3:.0f} g"
    )

  # Printed mass: WALL_MM of solid skin, INFILL inside; hardware at the mean
  # screw position. The sim carries the mass as two points (sleeve centre and
  # capsule centre), so report the split that puts the total COM where it is.
  per_x = np.zeros(len(x))
  for s in solid.values():
    depth = ndimage.distance_transform_edt(s) * g.v
    per_x += np.where(s, np.where(depth < WALL_MM, 1.0, INFILL), 0.0).sum(axis=(1, 2))
  per_x *= vox * DENSITY_G_CM3
  plastic = per_x.sum()
  screw_x = float(np.mean([sx for sx, _ in SCREWS]))
  com = (float(per_x @ x) + HARDWARE_G * screw_x) / (plastic + HARDWARE_G)
  sim_sleeve = (BRACE_START + flange) / 2
  sim_ext = flange + EXT_LENGTH / 2
  f_ext = (com - sim_sleeve) / (sim_ext - sim_sleeve)
  lines.append(
    f"printed mass estimate ({WALL_MM:.0f} mm walls, {INFILL:.0%} infill): plastic "
    f"{plastic:.0f} g ({per_x[x < flange].sum():.0f} g up to the flange), plus "
    f"{HARDWARE_G:.0f} g hardware; COM at x = {com:.1f} mm"
  )
  lines.append(
    f"split for robot29 that keeps this COM: BRACE_MASS = {1 - f_ext:.2f} x and "
    f"EXTENSION_MASS = {f_ext:.2f} x the weighed total"
  )

  both = solid["A"] & solid["B"]
  gap = (ndimage.distance_transform_edt(~solid["A"]) * g.v)[solid["B"]].min()
  lines.append(f"A/B overlap voxels {int(both.sum())}, minimum A-B gap {gap:.2f} mm")

  for (h, name, t), (_, f) in sorted(res["pads"].items()):
    if t == PAD_NOMINAL:
      area = (f < 0).sum() * g.v**3 / t
      lines.append(f"half {h} land {name}: pad area {area:.0f} mm2")

  # Wrist-to-PETG distance at zero pose (voxels), on the lands and elsewhere.
  for h in ("A", "B"):
    d = dw[solid[h]]
    on = land[h][solid[h]] <= 0.6  # Lands and their edge ramps.
    lines.append(
      f"half {h}: wrist to PETG min {d[on].min():.2f} mm on the lands (pad {PAD}), "
      f"{d[~on].min():.2f} mm elsewhere (relief {RELIEF})"
    )

  # Wrist surface samples. A contact counts as on a land (padded) when the
  # nearest PETG surface point lies on the land footprint, else as bare PETG.
  pts = surface_points(res["meshes"])
  zero = posed(pts, joints)

  def clearance(p: np.ndarray, h: str) -> tuple[float, float, float]:
    """Min clearance to bare PETG on steep walls (|n_y| < 0.3: these would jam),
    to other bare PETG (inclined: cams the wrist), and to the pad faces."""
    val = g.lookup(halves[h], p)
    grad = np.stack(
      [
        g.lookup(halves[h], p + e * g.v) - g.lookup(halves[h], p - e * g.v)
        for e in np.eye(3)
      ],
      axis=1,
    )
    grad /= np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-6)
    surf = p - val[:, None] * grad
    on = g.lookup(land[h], surf) <= 1.0
    steep = np.abs(grad[:, 1]) < 0.3

    def lowest(sel, v=val):
      return float(v[sel].min()) if sel.any() else 9.0

    return lowest(~on & steep), lowest(~on & ~steep), lowest(on, val - PAD)

  lines.append(
    "clearance in mm (negative = interference), per half: steep bare PETG / "
    "inclined bare PETG / pad face"
  )
  lines.append("insertion, half moved off along its axis by d:")
  for d in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    row = f"  d = {d:4.1f}:"
    for h, s in (("A", 1), ("B", -1)):
      c = clearance(zero - np.array([0.0, s * d, 0.0]), h)
      row += f"  {h} {c[0]:+.2f} / {c[1]:+.2f} / {c[2]:+.2f}"
    lines.append(row)
  lines.append("capture, halves seated on a turned wrist:")
  for p_deg, y_deg in (
    (1, 0),
    (-1, 0),
    (2, 0),
    (-2, 0),
    (2.5, 0),
    (-2.5, 0),
    (0, 2),
    (0, -2),
    (0, 3),
    (0, -3),
    (2, 2),
    (-2, -2),
    (2, -2),
    (-2, 2),
  ):
    p = posed(pts, joints, p_deg, y_deg)
    row = f"  pitch {p_deg:+4.1f} yaw {y_deg:+4.1f}:"
    for h in ("A", "B"):
      c = clearance(p, h)
      row += f"  {h} {c[0]:+.2f} / {c[1]:+.2f} / {c[2]:+.2f}"
    lines.append(row)
  for label, p_deg, y_deg in (
    ("pitch", 0.5, 0),
    ("pitch", -0.5, 0),
    ("yaw", 0, 0.5),
    ("yaw", 0, -0.5),
  ):
    p = posed(pts, joints, p_deg, y_deg)
    squeeze = -min(clearance(p, "A")[2], clearance(p, "B")[2])
    verdict = "locked" if squeeze > 0.1 else "NOT locked"
    lines.append(
      f"lock: {label} {p_deg or y_deg:+.1f} deg squeezes the pads {squeeze:.2f} mm "
      f"-> {verdict}"
    )

  # Walls: per x slice and 2-degree bin, outer radius minus cavity radius.
  union = solid["A"] | solid["B"]
  yy, zz = np.meshgrid(g.axes[1], g.axes[2], indexing="ij")
  rr = np.sqrt(yy**2 + zz**2)
  ang = (((np.degrees(np.arctan2(zz, yy)) + 180) % 360) // 2).astype(int) % 180
  closing = int(np.ceil(1.0 / g.v))
  walls = []
  for i in np.where((x >= BRACE_START + 1) & (x <= flange))[0][:: int(1 / g.v)]:
    sl = union[i]
    closed = ndimage.binary_closing(sl, iterations=closing) | sl
    lab, _ = ndimage.label(~closed)
    cavity = (lab > 0) & (lab != lab[0, 0])
    if not cavity.any():
      continue
    inner = np.full(180, -1.0)
    outer = np.full(180, -1.0)
    np.maximum.at(inner, ang[cavity], rr[cavity])
    np.maximum.at(outer, ang[sl], rr[sl])
    ok = inner > 0
    w = outer[ok] - inner[ok]
    k = int(np.argmin(w))
    walls.append((x[i], w[k], np.arange(180)[ok][k] * 2 - 180))
  walls = np.array(walls)
  away = np.ones(len(walls), bool)
  for sx, _ in SCREWS:
    away &= np.abs(walls[:, 0] - sx) > COUNTERBORE_R + 1.0
  j = int(np.argmin(walls[:, 1]))
  k = int(np.argmin(np.where(away, walls[:, 1], 1e9)))
  lines.append(
    f"thinnest sleeve wall {walls[j, 1]:.2f} mm at x = {walls[j, 0]:.0f} "
    f"(angle {walls[j, 2]:.0f} deg from +y); away from the screws "
    f"{walls[k, 1]:.2f} mm at x = {walls[k, 0]:.0f} (angle {walls[k, 2]:.0f})"
  )

  # Screw walls, from each feature's surface: to the wrist cavity, and for the
  # blind insert bores also to the outside (the counterbores open outside).
  ycoord = g.axes[1][None, :, None]
  for k, (sx, sz) in enumerate(SCREWS):
    row = f"screw ({sx:.0f}, {sz:+.0f}) wall to cavity / outside:"
    ix = np.where((x > sx - 14) & (x < sx + 14))[0]
    iz = np.where((g.axes[2] > sz - 14) & (g.axes[2] < sz + 14))[0]
    sel = (slice(ix[0], ix[-1] + 1), slice(None), slice(iz[0], iz[-1] + 1))
    body = res["body"][sel]
    for h in ("A", "B"):
      s_loc = solid[h][sel]
      cut = res["cuts"][h][k][sel] <= 1e-3
      sy = ycoord if h == "A" else -ycoord
      rim = cut & (body < -0.5) & (sy > SEAM + 3.0) & ndimage.binary_dilation(s_loc)
      void = ~s_loc & ~cut & (np.abs(ycoord) > SEAM + 0.2)
      cavity = void & (dw[sel] < 5.0) & (body < -0.5)
      to_cav = ndimage.distance_transform_edt(~cavity) * g.v
      if h == "A":
        upper = ycoord >= COUNTERBORE_FLOOR - ORING_DEPTH - g.v
        for label, part in (("counterbore", rim & upper), ("hole", rim & ~upper)):
          row += f"  A {label} {to_cav[part].min():.1f}" if part.any() else ""
      elif rim.any():
        to_out = ndimage.distance_transform_edt(body < 0) * g.v
        row += f"  B insert {to_cav[rim].min():.1f} / {to_out[rim].min():.1f}"
    lines.append(row)
  return lines


def section_plot(res: dict, out: Path) -> None:
  side, g, halves = res["side"], res["g"], res["halves"]
  occ = res["dw"] < 0
  flange = res["flange"]
  stations = (
    20.0,
    38.0,
    60.0,
    84.0,
    118.0,
    flange + 4.0,
    flange + TAIL + 7.5,
    flange + 50.0,
  )
  fig, axes = plt.subplots(2, 4, figsize=(18, 9.4))
  yy, zz = np.meshgrid(g.axes[1], g.axes[2], indexing="ij")
  lim = BRACE_RADIUS + 3
  for ax, xs in zip(axes.ravel(), stations, strict=True):
    i = int(np.argmin(np.abs(g.axes[0] - xs)))
    for name, colour in (("A", "#f39c12"), ("B", "#2e86c1")):
      ax.contourf(yy, zz, halves[name][i], levels=[-1e9, 0], colors=[colour], alpha=0.8)
    for (_, _, t), (sl, f) in res["pads"].items():
      if t == PAD_NOMINAL and sl.start <= i < sl.stop:
        ax.contourf(yy, zz, f[i - sl.start], levels=[-1e9, 0], colors=["#27ae60"])
    ax.contour(yy, zz, occ[i].astype(float), levels=[0.5], colors="k", linewidths=1)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"x = {xs:.0f} mm", fontsize=10)
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("z (mm)")
  fig.suptitle(
    f"{side} brace sections: orange = cap A (+y), blue = carrier B (-y), green = "
    "TPU pads, black = wrist",
    fontsize=12,
  )
  fig.tight_layout()
  fig.savefig(out / f"{side}_sections.png", dpi=100)
  plt.close(fig)

  fig, axes = plt.subplots(3, 1, figsize=(16, 13))
  xx, yy2 = np.meshgrid(g.axes[0], g.axes[1], indexing="ij")
  for ax, zc in zip(axes, (0.0, 32.0, 18.0), strict=True):
    k = int(np.argmin(np.abs(g.axes[2] - zc)))
    for name, colour in (("A", "#f39c12"), ("B", "#2e86c1")):
      ax.contourf(
        xx, yy2, halves[name][:, :, k], levels=[-1e9, 0], colors=[colour], alpha=0.8
      )
    ax.contour(
      xx, yy2, occ[:, :, k].astype(float), levels=[0.5], colors="k", linewidths=1
    )
    ax.set_aspect("equal")
    ax.set_title(f"long section at z = {zc:.0f} mm", fontsize=10)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
  fig.tight_layout()
  fig.savefig(out / f"{side}_long.png", dpi=90)
  plt.close(fig)


# ---------------------------------------------------------------------------


def export_side(res: dict, out: Path, report: list[str]) -> None:
  """Print-ready STLs, plus wrist-frame copies in ``assembled/`` for viewing."""
  side, g = res["side"], res["g"]
  x = g.axes[0]
  assembled = out / "assembled"
  assembled.mkdir(exist_ok=True)
  trimesh.util.concatenate(res["meshes"]).export(assembled / f"{side}_wrist.stl")
  down = {"A": np.array([0.0, -1.0, 0.0]), "B": np.array([0.0, 1.0, 0.0])}
  for h, name in (("A", "cap"), ("B", "carrier")):
    mesh = to_mesh(res["halves"][h], g)
    mesh.export(assembled / f"{side}_{h}_{name}.stl")
    print_pose(mesh, down[h]).export(out / f"{side}_{h}_{name}.stl")
    report.append(
      f"{side}_{h}_{name}.stl: watertight {mesh.is_watertight}, "
      f"{len(mesh.faces)} faces, {mesh.volume / 1000:.1f} cm3"
    )
    for ring, (x0, x1) in FIT_RINGS.items():
      slab = np.maximum(x0 - x, x - x1)[:, None, None].astype(np.float32)
      rm = to_mesh(np.maximum(res["halves"][h], slab), g)
      print_pose(rm, down[h]).export(out / f"{side}_fitring_{ring}_{h}.stl")
  # Pads: one plate per thickness. Row A then row B, each: tag, L4 standing
  # with its curvature axis up, then L1 and L2 printed flat (identical for A
  # and B) to be bent on.
  for t in PAD_THICKNESSES:
    plate, placed, row_y = [], [], 0.0
    for h in ("A", "B"):
      items = [tag(h)]
      for name in LANDS:
        sl, f = res["pads"][(h, name, t)]
        pm = to_mesh(f, g.x_slice(sl), tolerance=0.01)
        parts = [p for p in pm.split(only_watertight=False) if p.volume > 1.0]
        if not parts:
          report.append(f"pad {h} {name} {t} mm: EMPTY")
          continue
        pm = trimesh.util.concatenate(parts)
        placed.append(pm)
        if name in FLAT_PADS:
          straight = float(pm.extents[PAD_UP[name]])
          arc = pm.volume / t / straight
          items.append(flat_pad(arc, straight, t, name))
          size = f"flat {arc:.1f} x {straight:.1f} x {t} mm"
        else:
          items.append(print_pose(pm, np.eye(3)[PAD_UP[name]]))
          size = f"shaped, {pm.extents.round(1).tolist()} mm"
        if t == PAD_NOMINAL:
          report.append(
            f"pad {h} {name}: {size}, {len(parts)} piece(s), watertight "
            f"{all(p.is_watertight for p in parts)}"
          )
      cursor, depth = 0.0, 0.0
      for item in items:
        item.apply_translation([cursor, row_y, 0.0] - item.bounds[0] * [1, 1, 0])
        cursor = item.bounds[1, 0] + 5.0
        depth = max(depth, float(item.extents[1]))
        plate.append(item)
      row_y += depth + 6.0
    trimesh.util.concatenate(plate).export(out / f"{side}_pads_{t:.1f}mm.stl")
    if t == PAD_NOMINAL:
      trimesh.util.concatenate(placed).export(assembled / f"{side}_pads.stl")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--voxel", type=float, default=0.3, help="Grid size, mm.")
  parser.add_argument("--sides", nargs="+", default=["left", "right"])
  parser.add_argument("--capture-pitch", type=float, default=CAPTURE_PITCH)
  parser.add_argument("--capture-yaw", type=float, default=CAPTURE_YAW)
  parser.add_argument("--cache", type=Path, default=None, help="Cache wrist fields.")
  parser.add_argument(
    "--relief-box",
    nargs=7,
    type=float,
    action="append",
    metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1", "DEPTH"),
    help="Cut DEPTH mm extra relief inside this box (wrist frame, mm); repeatable.",
  )
  parser.add_argument("--out", type=Path, default=OUT)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  report = [
    f"design 3 loose split clamp: end gap {END_GAP} mm, tail {TAIL} mm,  relief {RELIEF} mm + {RELIEF_DRAFT} deg draft, capture "
    f"pitch +-{args.capture_pitch} / yaw +-{args.capture_yaw} deg keeping "
    f"{CAPTURE_MARGIN} mm, lands {PAD} mm off the wrist, seam gap {2 * SEAM} mm, "
    f"voxel {args.voxel} mm",
    "",
  ]
  for side in args.sides:
    res = build_side(side, args)
    export_side(res, args.out, report)
    report.extend(checks(res))
    section_plot(res, args.out)
    report.append("")
    print("\n".join(report), flush=True)
    (args.out / "report.txt").write_text("\n".join(report) + "\n")
    del res
  gap_gauge().export(args.out / "gap_gauge.stl")
  (args.out / "report.txt").write_text("\n".join(report) + "\n")
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
