"""Add the forearm extension (50 mm cylinder + R25 dome) to a user's brace halves.

The input halves are left exactly as they are: the extension is unioned onto the
flat hand-end face, split at the same seam planes, with two through-bolts that
clamp the two extension halves together (socket head in A, hex nut in B).

With ``--body-nuts`` the six body screws become through-bolts too: B's blind
heat-set-insert bores get a clearance hole on through to a hex nut pocket opened
from B's outside. For M4 (``--screw m4``) A's 3.8 mm neck under each counterbore
also opens to 4.6 mm (an M4 screw would thread into it and could not clamp); M3
passes A's holes as they are.

The tip lands where the simulation has it (``robot29``): 225.5 mm past the
wrist-roll joint, i.e. EXTENSION_LENGTH past the wrist-yaw flange.

  uv run python scripts/add_brace_extension.py \
    --a hardware/wrist_brace/extension_addon/input/left_A_cap_v2.stl \
    --b hardware/wrist_brace/extension_addon/input/left_B_carrier_v2.stl \
    --roll-x -93.2 --out hardware/wrist_brace/extension_addon
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import manifold3d as mf
import numpy as np
import shapely
import trimesh

FLANGE_FROM_ROLL = 125.5  # Wrist-yaw flange, mm past the wrist-roll joint.
EXT_LENGTH = 100.0  # robot29.EXTENSION_LENGTH, flange to tip.
EXT_RADIUS = 25.0  # robot29.EXTENSION_RADIUS.
BLEND = 5.0  # From the end-face outline to the 50 mm circle.
OVERLAP = 0.5  # The add-on starts this far inside the end wall, for the union.
INSET = 0.1  # Base outline sits this far inside the user's end-face outline.
N_SEG = 256

# Nose bolts: axes along y at z = 0, counterbore floor in A and nut seat in B at
# |y| = HEAD_FLOOR. The counterbore matches the body's 9.6 mm ones (9 mm washer).
BOLT_X_FROM_END = (14.5, 44.5)
HEAD_R = 4.8
HEAD_FLOOR = 10.0
# A's neck opens (M4 only) up to this |y|, 0.3 mm into its own 4.5 mm hole (a
# radius equal to that hole's would leave coincident walls).
BODY_A_NECK = 7.6


@dataclass(frozen=True)
class Screw:
  hole_r: float  # Clearance.
  nut_af: float  # Hex pocket across flats.
  body_seat: float  # |y| of the nut seat in B's body pockets.
  a_neck_r: float | None  # Open A's 3.8 mm neck to this radius, or leave it.
  nose: str  # Hardware, for the report.
  body: str


SCREWS = {
  "m3": Screw(1.75, 5.8, 20.5, None, "M3 x 30 socket head", "M3 x 40 socket head"),
  "m4": Screw(2.25, 7.3, 19.5, 2.3, "M4 x 30 socket head", "M4 x 40 socket head"),
}


def to_manifold(m: trimesh.Trimesh) -> mf.Manifold:
  mesh = mf.Mesh(
    vert_properties=np.asarray(m.vertices, np.float32),
    tri_verts=np.asarray(m.faces, np.uint32),
  )
  out = mf.Manifold(mesh)
  assert out.status() == mf.Error.NoError, out.status()
  return out


def to_trimesh(m: mf.Manifold) -> trimesh.Trimesh:
  mesh = m.to_mesh()
  return trimesh.Trimesh(mesh.vert_properties[:, :3], mesh.tri_verts, process=True)


def main_body(path: Path) -> tuple[trimesh.Trimesh, int, int]:
  """Largest body, number of stray bodies dropped, faces whose winding was fixed.

  Fixing the winding only flips the vertex order of inverted triangles; no vertex
  moves.
  """
  parts = trimesh.load(path).split(only_watertight=False)
  body = max(parts, key=lambda p: len(p.faces))
  assert body.is_watertight, f"{path}: main body is not watertight"
  faces = body.faces.copy()
  trimesh.repair.fix_winding(body)
  return body, len(parts) - 1, int(np.any(body.faces != faces, axis=1).sum())


def end_outline(a: trimesh.Trimesh, b: trimesh.Trimesh, x_end: float, yc: float):
  """Radius of the joint end-face outline around (yc, 0), as a function of angle."""
  phis, rhos = [], []
  for m in (a, b):
    sec = m.section(plane_origin=[x_end - 0.01, 0, 0], plane_normal=[1, 0, 0])
    for loop in sec.discrete:
      line = shapely.segmentize(shapely.LineString(loop[:, 1:]), 0.2)
      p = shapely.get_coordinates(line)
      y, z = p[:, 0] - yc, p[:, 1]
      keep = (np.abs(y) > 0.2) | (np.abs(z) > 24.0)  # Drop the seam edge.
      phis.append(np.arctan2(z[keep], y[keep]))
      rhos.append(np.hypot(y[keep], z[keep]))
  phi, rho = np.concatenate(phis), np.concatenate(rhos)
  order = np.argsort(phi)
  phi, rho = phi[order], rho[order]
  grid = np.linspace(-np.pi, np.pi, N_SEG, endpoint=False)
  return grid, np.interp(grid, phi, rho, period=2 * np.pi)


def extension(x_end: float, tip_x: float, yc: float, phi, rho_end) -> trimesh.Trimesh:
  """Lofted base -> 50 mm cylinder -> hemispherical tip, closed."""
  dome_x = tip_x - EXT_RADIUS
  base = rho_end - INSET
  # No ring on the end-face plane itself: coincident edges make slivers there.
  rings = [(x_end - OVERLAP, base)]
  for s in np.linspace(0, 1, 21)[1:]:
    rings.append((x_end + 0.1 + s * BLEND, (1 - s) * base + s * EXT_RADIUS))
  rings.append((dome_x, np.full(N_SEG, EXT_RADIUS)))
  for a in np.linspace(0, np.pi / 2, 33)[1:-1]:
    rings.append(
      (dome_x + EXT_RADIUS * np.sin(a), np.full(N_SEG, EXT_RADIUS * np.cos(a)))
    )
  c, s = np.cos(phi), np.sin(phi)
  verts = [np.c_[np.full(N_SEG, x), yc + r * c, r * s] for x, r in rings]
  v = np.vstack(verts + [[[x_end - OVERLAP, yc, 0]], [[tip_x, yc, 0]]])
  n = len(rings)
  faces = []
  for i in range(n - 1):
    for j in range(N_SEG):
      a0, a1 = i * N_SEG + j, i * N_SEG + (j + 1) % N_SEG
      b0, b1 = a0 + N_SEG, a1 + N_SEG
      faces += [[a0, b0, b1], [a0, b1, a1]]
  base_c, tip_c = n * N_SEG, n * N_SEG + 1
  for j in range(N_SEG):
    faces.append([base_c, j, (j + 1) % N_SEG])
    k = (n - 1) * N_SEG
    faces.append([tip_c, k + (j + 1) % N_SEG, k + j])
  m = trimesh.Trimesh(v, np.array(faces), process=True)
  trimesh.repair.fix_normals(m)
  assert m.is_watertight and m.volume > 0
  return m


def bolt_tools(xs, yc: float, sc: Screw) -> tuple[mf.Manifold, mf.Manifold]:
  """(cut for A, cut for B): through holes, A counterbores, B hex nut pockets."""

  def cyl(r, y0, y1, x, n=64):
    """Cylinder along y from yc + y0 to yc + y1 (rotate -90 about x: +z -> +y)."""
    c = mf.Manifold.cylinder(y1 - y0, r, r, n)
    return c.rotate([-90, 0, 0]).translate([x, yc + y0, 0])

  cut_a, cut_b = [], []
  for x in xs:
    hole = cyl(sc.hole_r, -40, 40, x)
    head = cyl(HEAD_R, HEAD_FLOOR, 40, x)
    nut = cyl(sc.nut_af / np.sqrt(3), -40, -HEAD_FLOOR, x, n=6)  # Circumradius.
    cut_a += [hole, head]
    cut_b += [hole, nut]
  return mf.Manifold.batch_boolean(cut_a, mf.OpType.Add), mf.Manifold.batch_boolean(
    cut_b, mf.OpType.Add
  )


def body_holes(a: trimesh.Trimesh, seam_a: float) -> list[tuple[float, float]]:
  """(x, z) of the body screw axes, from the small loops on A's seam face."""
  sec = a.section(plane_origin=[0, seam_a + 0.5, 0], plane_normal=[0, 1, 0])
  out = []
  for loop in sec.discrete:
    if np.ptp(loop[:, 0]) < 12 and np.ptp(loop[:, 2]) < 12:
      lo, hi = loop.min(0), loop.max(0)  # Circle centre (vertices are uneven).
      out.append((float(lo[0] + hi[0]) / 2, float(lo[2] + hi[2]) / 2))
  return sorted(out)


def body_tools(holes, yc: float, sc: Screw) -> tuple[mf.Manifold, mf.Manifold]:
  """(cut for A, cut for B) turning the body screws into through-bolts with nuts."""

  def cyl(r, y0, y1, x, z, n=64):
    c = mf.Manifold.cylinder(y1 - y0, r, r, n)
    return c.rotate([-90, 0, 0]).translate([x, yc + y0, z])

  cut_a, cut_b = [], []
  for x, z in holes:
    if sc.a_neck_r is not None:
      cut_a.append(cyl(sc.a_neck_r, -0.5, BODY_A_NECK, x, z))
    cut_b.append(cyl(sc.hole_r, -sc.body_seat, 0.5, x, z))
    cut_b.append(cyl(sc.nut_af / np.sqrt(3), -80, -sc.body_seat, x, z, n=6))
  add = mf.OpType.Add
  empty = mf.Manifold()
  return (
    mf.Manifold.batch_boolean(cut_a, add) if cut_a else empty,
    mf.Manifold.batch_boolean(cut_b, add),
  )


def half_space(y_lo: float, y_hi: float, x0: float, x1: float) -> mf.Manifold:
  return mf.Manifold.cube([x1 - x0, y_hi - y_lo, 200]).translate([x0, y_lo, -100])


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--a", type=Path, required=True)
  ap.add_argument("--b", type=Path, required=True)
  ap.add_argument("--roll-x", type=float, required=True, help="wrist-roll joint x")
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--body-nuts", action="store_true", help="body screws -> nuts")
  ap.add_argument("--screw", choices=sorted(SCREWS), default="m3")
  args = ap.parse_args()
  sc = SCREWS[args.screw]

  a, a_junk, a_flip = main_body(args.a)
  b, b_junk, b_flip = main_body(args.b)
  x_end = float(a.bounds[1, 0])
  assert abs(b.bounds[1, 0] - x_end) < 0.05, "halves end at different x"
  seam_a, seam_b = float(a.bounds[0, 1]), float(b.bounds[1, 1])
  yc = (seam_a + seam_b) / 2
  tip_x = args.roll_x + FLANGE_FROM_ROLL + EXT_LENGTH
  phi, rho_end = end_outline(a, b, x_end, yc)
  ext = extension(x_end, tip_x, yc, phi, rho_end)
  xs = [x_end + d for d in BOLT_X_FROM_END]
  cut_a, cut_b = bolt_tools(xs, yc, sc)

  ext_m = to_manifold(ext)
  big = 1000.0
  ext_a = (ext_m ^ half_space(seam_a, big, -big, big)) - cut_a
  ext_b = (ext_m ^ half_space(-big, seam_b, -big, big)) - cut_b
  body_a, body_b = to_manifold(a), to_manifold(b)
  holes = body_holes(a, seam_a) if args.body_nuts else []
  if holes:
    tool_a, tool_b = body_tools(holes, yc, sc)
    body_a, body_b = body_a - tool_a, body_b - tool_b
  out_a = to_trimesh(body_a + ext_a)
  out_b = to_trimesh(body_b + ext_b)

  report = [
    f"inputs: {args.a.name}, {args.b.name}; dropped {a_junk} / {b_junk} zero-volume "
    f"stray triangles, fixed the winding of {a_flip} / {b_flip} inverted triangles",
    f"wrist-roll joint x = {args.roll_x:.2f}, flange x = "
    f"{args.roll_x + FLANGE_FROM_ROLL:.2f}, end face x = {x_end:.2f}, tip x = "
    f"{tip_x:.2f}",
    f"add-on length past your end face {tip_x - x_end:.2f} mm: blend {BLEND}, "
    f"cylinder {tip_x - EXT_RADIUS - x_end - BLEND:.2f}, dome R{EXT_RADIUS}",
    f"wrist-roll joint to tip {tip_x - args.roll_x:.2f} mm (simulation 225.5)",
    f"end-face outline radius {rho_end.min():.2f}..{rho_end.max():.2f} mm",
    f"nose bolts ({sc.nose} + 9 mm washer + nyloc) at x = "
    f"{', '.join(f'{x:.1f}' for x in xs)}, z = 0",
  ]
  if holes:
    report.append(
      f"body bolts ({sc.body} + 9 mm washer + nyloc), nut seat at |y| = "
      f"{sc.body_seat}: (x, z) = " + ", ".join(f"({x:.1f}, {z:.1f})" for x, z in holes)
    )
  for name, orig, new in (("A", a, out_a), ("B", b, out_b)):
    stem = (args.a if name == "A" else args.b).stem
    path = args.out / f"{stem}_ext.stl"
    new.export(path)
    # The original surface must survive except the end face under the add-on.
    pts, fid = trimesh.sample.sample_surface(orig, 60000, seed=0)
    keep = ~((pts[:, 0] > x_end - 0.02) & (orig.face_normals[fid, 0] > 0.99))
    for x, z in holes:  # ... and the screw holes reworked for nuts.
      keep &= np.hypot(pts[:, 0] - x, pts[:, 2] - z) > sc.nut_af / np.sqrt(3) + 0.3
    _, d, _ = trimesh.proximity.closest_point(new, pts[keep])
    lo_o, hi_o = orig.bounds
    lo_n, hi_n = new.bounds
    report.append(
      f"{name}: {path.name}  watertight {new.is_watertight}  volume "
      f"{orig.volume / 1000:.2f} -> {new.volume / 1000:.2f} cm3  original surface "
      f"moved max {d.max():.4f} mm  bounds y {lo_n[1]:.2f}..{hi_n[1]:.2f} "
      f"(was {lo_o[1]:.2f}..{hi_o[1]:.2f}) z {lo_n[2]:.2f}..{hi_n[2]:.2f} (was "
      f"{lo_o[2]:.2f}..{hi_o[2]:.2f}) x {lo_n[0]:.2f}..{hi_n[0]:.2f}"
    )
  gap = out_a.bounds[0, 1] - out_b.bounds[1, 1]
  report.append(
    f"seam gap between the halves {gap:.2f} mm (yours: {seam_a - seam_b:.2f})"
  )
  (args.out / "report.txt").write_text("\n".join(report) + "\n")
  print("\n".join(report))


if __name__ == "__main__":
  main()
