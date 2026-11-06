"""Minimal fit-test shells for the wrist brace: two thin halves per arm, loose.

Prints in a fraction of the material of the real brace (``make_wrist_brace.py``,
design 3) and answers one question before any full print: does each half drop
onto the real wrist, and do the two halves close around it without touching it?

What it keeps from design 3:

- the cavity: Unitree's wrist meshes in the braced pose (wrist pitch/yaw 0),
  dilated by ``RELIEF``, opened toward the seam by ``DRAFT`` so each half drops
  straight on, widened to admit a wrist turned up to ``CAPTURE_*`` degrees, and
  the end pocket past the wrist-yaw flange;
- the split: cap A on +y, B on -y (y = the wrist-pitch axis), the same frame and
  the same closing direction.

What it changes:

- more room: ``RELIEF`` 3.5 mm everywhere (design 3: 2.5 mm bare, 2.0 mm at the
  lands), ``END_GAP`` 8 mm (6 mm), capture margin 1 mm (0.5 mm). No lands, no
  pads: nothing is meant to touch the wrist;
- the halves meet on y = 0 (no seam gap), so a closed shell is exactly the wrist
  plus ``RELIEF`` all round;
- a ``WALL`` mm conformal skin instead of the 40 mm round sleeve, and no
  extension, screws, inserts or key;
- three small tabs per side on the seam, with M3 / zip-tie holes, to hold the
  halves shut with binder clips, M3 screws or zip ties;
- each half prints cavity up (seam face on top), the closing direction vertical,
  so the drafted cavity needs no support and the tabs carry 45 deg gussets. Only
  the rounded outside needs the slicer's supports (build plate only). With
  ``--skirt`` a support-free solid skirt is built instead (every layer within
  ``OVERHANG_DEG`` of the one below), at about three times the plastic.

Outputs in ``hardware/wrist_brace/fit_test/``: ``{side}_fit_{A,B}.stl`` in print
orientation, wrist-frame copies in ``assembled/``, section pictures and
``report.txt``.

Usage::

  uv run --with scikit-image python scripts/make_wrist_brace_fittest.py
  ... --sides left --cache /tmp/brace_cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_wrist_brace as wb  # noqa: E402

OUT = wb.OUT / "fit_test"

RELIEF = 3.5  # Wrist to plastic, everywhere (mm).
DRAFT = 2.0  # deg, opening toward the seam.
END_GAP = 8.0  # Cavity past the wrist-yaw flange (RELIEF comes on top).
CAPTURE_PITCH = 3.0  # deg: halves drop on a wrist turned this far,
CAPTURE_YAW = 3.0
CAPTURE_MARGIN = 1.0  # keeping this much clearance.
WALL = 1.6  # Skin thickness: four 0.4 mm lines.
OVERHANG_DEG = 45.0  # Steepest unsupported overhang of the skirt.
TAB_LEN = 12.0  # Along the forearm.
TAB_W = 7.0  # Out from the skin.
TAB_T = 2.4  # Thickness from the seam face.
TAB_HOLE_R = 1.7  # M3 or a 2.5 mm zip tie.
BASE_SHAVE = 0.6  # Flattens the outermost skin into strips that sit on the bed.
LABEL_H = 8.0
LABEL_DEPTH = 0.5
DENSITY_G_CM3 = 1.27  # PETG.


def sdf_from_mask(mask: np.ndarray, v: float) -> np.ndarray:
  """Signed distance (mm) to a boolean voxel set, negative inside."""
  out = ndimage.distance_transform_edt(~mask).astype(np.float32) * v
  inn = ndimage.distance_transform_edt(mask).astype(np.float32) * v
  return np.where(mask, -(inn - v / 2), out - v / 2)


def skirt(solid: np.ndarray, s: int, j_seam: int, angle_deg: float) -> np.ndarray:
  """Support-free closure of a half printed with -s*y up (seam on top).

  Walking down from the seam layer to the base (the half's outermost layer),
  each layer keeps the layer above it, shrunk by the overhang allowance: so
  every layer lies within ``angle_deg`` of the one below. Returns the added fill.
  """
  allow = np.tan(np.radians(angle_deg))  # Voxels of overhang per voxel layer.
  rows = np.where(solid.any(axis=(0, 2)))[0]
  j_base = int(rows.max()) if s > 0 else int(rows.min())
  fill = np.zeros_like(solid)
  above = solid[:, j_seam, :].copy()
  for j in range(j_seam + s, j_base + s, s):
    inside = ndimage.distance_transform_edt(above)
    carried = inside > allow
    layer = solid[:, j, :] | carried
    fill[:, j, :] = carried & ~solid[:, j, :]
    above = layer
  return fill


def build(side: str, args) -> dict:
  meshes, flange, joints = wb.wrist_meshes(side)
  x_end = flange + END_GAP + RELIEF + WALL
  reach = 54.0
  lo = np.array([wb.BRACE_START - 3.0, -reach, -reach])
  hi = np.array([x_end + 3.0, reach, reach])
  g = wb.Grid(lo, hi, args.voxel)
  print(f"[{side}] grid {g.shape}, flange at x = {flange:.1f} mm", flush=True)
  fields = wb.link_fields(side, g, meshes, args.cache)
  dw = wb.vmin(*fields)
  env = wb.capture_envelope(
    g, fields, joints, flange, args.capture_pitch, args.capture_yaw
  )
  del fields
  x, y, z = g.coords()
  r = np.sqrt(y**2 + z**2)
  yaw_end = meshes[2].vertices[meshes[2].vertices[:, 0] > flange - 6.0]
  end_r = float(np.sqrt(yaw_end[:, 1] ** 2 + yaw_end[:, 2] ** 2).max())
  pocket = wb.vmax(r - end_r, (flange - 1.0) - x, x - (flange + END_GAP))
  target = np.minimum(
    np.minimum(dw, pocket) - RELIEF, np.minimum(env, pocket) - CAPTURE_MARGIN
  )
  del env, pocket
  print(f"[{side}] cavity field done (end pocket radius {end_r:.1f} mm)", flush=True)

  j_seam = int(np.argmin(np.abs(g.axes[1])))
  ix = g.axes[0]
  tab_windows = (
    (wb.BRACE_START, wb.BRACE_START + TAB_LEN),
    ((wb.BRACE_START + flange - TAB_LEN) / 2, (wb.BRACE_START + flange + TAB_LEN) / 2),
    (x_end - TAB_LEN - 4.0, x_end - 4.0),
  )
  halves, info = {}, {}
  for half, s in (("A", 1), ("B", -1)):
    cav = wb.drafted_sweep(target, s, DRAFT, g.v)
    shell = wb.vmax(-cav, cav - WALL, wb.BRACE_START - x, -s * y)
    solid = shell < -1e-3

    # Tabs on the seam face, out from the skin's seam outline, each with a
    # 45 deg gusset under it down to the skin (the tabs print on top).
    seam_fp = (cav[:, j_seam, :] <= WALL) & (ix[:, None] >= wb.BRACE_START)
    out_d = ndimage.distance_transform_edt(~seam_fp) * g.v
    in_x = np.zeros(len(ix), bool)
    for x0, x1 in tab_windows:
      in_x |= (ix >= x0) & (ix <= x1)
    ty = (s * g.axes[1])[None, :, None]
    reach_out = TAB_W - np.maximum(ty - TAB_T, 0.0) * np.tan(np.radians(OVERHANG_DEG))
    tabs = (
      (out_d[:, None, :] <= reach_out)
      & in_x[:, None, None]
      & (ty >= 0)
      & (cav >= 0)
      & (ix[:, None, None] >= wb.BRACE_START)
    )
    solid |= tabs
    added = tabs & ~(shell < -1e-3)

    if args.skirt:
      fill = skirt(solid, s, j_seam, OVERHANG_DEG)
      added |= fill
    else:
      fill = np.zeros_like(solid)
    final = np.minimum(shell, sdf_from_mask(added | (shell < -1e-3), g.v))
    final = np.maximum(final, -cav)  # The cavity stays exactly as designed.
    final = np.maximum(final, wb.BRACE_START - x)
    final = np.maximum(final, -s * y)
    # Flat strips on the build plate: shave the outermost skin a little.
    outer = float((s * g.axes[1])[(final < -1e-3).any(axis=(0, 2))].max())
    final = np.maximum(final, s * y - (outer - BASE_SHAVE))

    # Holes through the tabs, centred TAB_W / 2 out from the skin.
    zs = g.axes[2]
    holes = []
    for x0, x1 in tab_windows:
      xc = (x0 + x1) / 2
      i = int(np.argmin(np.abs(ix - xc)))
      row = np.where(seam_fp[i])[0]
      for zc in (zs[row.max()] + TAB_W / 2, zs[row.min()] - TAB_W / 2):
        rad = np.sqrt((x - xc) ** 2 + (z - zc) ** 2) - TAB_HOLE_R
        final = np.maximum(final, -np.broadcast_to(rad, final.shape))
        holes.append((round(xc, 1), round(float(zc), 1)))

    # Label on the outside of the end cap, read from the hand end.
    tag = f"{side[0].upper()}{half}"
    plane = (g.shape[1], g.shape[2])
    yc = s * 16.0
    u = np.broadcast_to(-s * g.axes[2][None, :], plane)
    w = np.broadcast_to(s * (g.axes[1][:, None] - yc), plane)
    t2 = wb.text_sdf(tag, u, w, LABEL_H)[None, :, :]
    deboss = np.maximum(t2, (x_end - LABEL_DEPTH) - x)
    final = np.maximum(final, -deboss)

    halves[half] = final.astype(np.float32)
    info[half] = {
      "tab_holes": holes,
      "skirt_voxels": int(fill.sum()),
      "shell_voxels": int((shell < -1e-3).sum()),
    }
    del cav, shell, solid, tabs, added, fill, final
  return {
    "side": side,
    "g": g,
    "meshes": meshes,
    "flange": flange,
    "joints": joints,
    "dw": dw,
    "halves": halves,
    "info": info,
    "x_end": x_end,
  }


def checks(res: dict) -> list[str]:
  side, g, halves, dw = res["side"], res["g"], res["halves"], res["dw"]
  joints = res["joints"]
  vox = (g.v / 10) ** 3
  lines = [f"== {side} =="]
  for h in ("A", "B"):
    solid = halves[h] < -1e-3
    cm3 = solid.sum() * vox
    i = res["info"][h]
    lines.append(
      f"half {h}: {cm3:.1f} cm3 = {cm3 * DENSITY_G_CM3:.0f} g PETG solid "
      f"(skin {i['shell_voxels'] * vox:.1f} cm3, skirt {i['skirt_voxels'] * vox:.1f} "
      f"cm3); wrist to plastic min {dw[solid].min():.2f} mm; tab holes at (x, z) "
      f"{i['tab_holes']}"
    )
  both = (halves["A"] < -1e-3) & (halves["B"] < -1e-3)
  lines.append(f"A/B overlap voxels {int(both.sum())}")

  pts = wb.surface_points(res["meshes"])

  def clearance(p: np.ndarray, h: str) -> float:
    return float(g.lookup(halves[h], p).min())

  lines.append("clearance in mm (negative = touching), wrist surface to each half:")
  lines.append("insertion, half lifted off along its closing axis by d:")
  zero = wb.posed(pts, joints)
  for d in (0.0, 1.0, 3.0, 10.0, 20.0):
    row = f"  d = {d:4.1f}:"
    for h, s in (("A", 1), ("B", -1)):
      row += f"  {h} {clearance(zero - np.array([0.0, s * d, 0.0]), h):+.2f}"
    lines.append(row)
  lines.append("seated on a turned wrist:")
  for p_deg, y_deg in ((3, 0), (-3, 0), (0, 3), (0, -3), (3, 3), (-3, -3)):
    p = wb.posed(pts, joints, p_deg, y_deg)
    row = f"  pitch {p_deg:+.0f} yaw {y_deg:+.0f}:"
    for h in ("A", "B"):
      row += f"  {h} {clearance(p, h):+.2f}"
    lines.append(row)
  lines.append(
    f"hand end: cavity runs {END_GAP + RELIEF:.1f} mm past the wrist-yaw flange "
    f"(x = {res['flange']:.1f}), end cap outer face at x = {res['x_end']:.1f} mm"
  )
  return lines


def section_plot(res: dict, out: Path) -> None:
  side, g, halves = res["side"], res["g"], res["halves"]
  occ = res["dw"] < 0
  flange = res["flange"]
  stations = (14.0, 30.0, 48.0, 66.0, 84.0, 104.0, flange - 2.0, flange + 6.0)
  fig, axes = plt.subplots(2, 4, figsize=(18, 9.4))
  yy, zz = np.meshgrid(g.axes[1], g.axes[2], indexing="ij")
  lim = 54
  for ax, xs in zip(axes.ravel(), stations, strict=True):
    i = int(np.argmin(np.abs(g.axes[0] - xs)))
    for name, colour in (("A", "#f39c12"), ("B", "#2e86c1")):
      ax.contourf(yy, zz, halves[name][i], levels=[-1e9, 0], colors=[colour], alpha=0.8)
    ax.contour(yy, zz, occ[i].astype(float), levels=[0.5], colors="k", linewidths=1)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"x = {xs:.0f} mm", fontsize=10)
    ax.set_xlabel("y (mm, closing axis)")
    ax.set_ylabel("z (mm)")
  fig.suptitle(
    f"{side} fit test: orange = A (+y), blue = B (-y), black = wrist; "
    f"{RELIEF} mm relief, {WALL} mm skin",
    fontsize=12,
  )
  fig.tight_layout()
  fig.savefig(out / f"{side}_fit_sections.png", dpi=90)
  plt.close(fig)

  fig, axes = plt.subplots(2, 1, figsize=(16, 9))
  xx, yy2 = np.meshgrid(g.axes[0], g.axes[1], indexing="ij")
  for ax, zc in zip(axes, (0.0, 20.0), strict=True):
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
    ax.set_xlabel("x (mm, along the forearm)")
    ax.set_ylabel("y (mm)")
  fig.tight_layout()
  fig.savefig(out / f"{side}_fit_long.png", dpi=90)
  plt.close(fig)


def export(res: dict, out: Path, report: list[str]) -> None:
  side, g = res["side"], res["g"]
  assembled = out / "assembled"
  assembled.mkdir(parents=True, exist_ok=True)
  trimesh.util.concatenate(res["meshes"]).export(assembled / f"{side}_wrist.stl")
  # Cavity up: A prints with -y up, B with +y up.
  up = {"A": np.array([0.0, -1.0, 0.0]), "B": np.array([0.0, 1.0, 0.0])}
  for h in ("A", "B"):
    mesh = wb.to_mesh(res["halves"][h], g)
    mesh.export(assembled / f"{side}_fit_{h}.stl")
    printed = wb.print_pose(mesh, up[h])
    printed.export(out / f"{side}_fit_{h}.stl")
    ext = printed.extents.round(0).astype(int).tolist()
    report.append(
      f"{side}_fit_{h}.stl: watertight {mesh.is_watertight}, {mesh.volume / 1000:.1f}"
      f" cm3 ({mesh.volume / 1000 * DENSITY_G_CM3:.0f} g PETG), print size {ext} mm"
    )


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--voxel", type=float, default=0.3, help="Grid size, mm.")
  parser.add_argument("--sides", nargs="+", default=["left", "right"])
  parser.add_argument("--capture-pitch", type=float, default=CAPTURE_PITCH)
  parser.add_argument("--capture-yaw", type=float, default=CAPTURE_YAW)
  parser.add_argument(
    "--skirt",
    action="store_true",
    help="Support-free solid skirt under the outside (about 3x the plastic).",
  )
  parser.add_argument("--cache", type=Path, default=None, help="Cache wrist fields.")
  parser.add_argument("--out", type=Path, default=OUT)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)
  report = [
    f"fit test: relief {RELIEF} mm + {DRAFT} deg draft, end gap {END_GAP} mm, "
    f"capture pitch +-{args.capture_pitch} / yaw +-{args.capture_yaw} deg keeping "
    f"{CAPTURE_MARGIN} mm, skin {WALL} mm, halves meet at y = 0, "
    f"{f'{OVERHANG_DEG:.0f} deg skirt' if args.skirt else 'no skirt (slicer supports)'}"
    f", voxel {args.voxel} mm",
    "",
  ]
  for side in args.sides:
    res = build(side, args)
    export(res, args.out, report)
    report.extend(checks(res))
    section_plot(res, args.out)
    report.append("")
    print("\n".join(report), flush=True)
    (args.out / "report.txt").write_text("\n".join(report) + "\n")
    del res
  print(f"wrote {args.out}")


if __name__ == "__main__":
  main()
