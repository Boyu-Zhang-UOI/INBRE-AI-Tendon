"""Render the centerlines of two fascicles as tubes (manuscript Figure 3B).

Usage:
  python analysis/render_winding_pair.py <Point3DReconstruction.ply> <out_prefix> <idA> <idB> \
      [first_slice last_slice] [--um-per-slice 9] [--z-display 0.6]

idA, idB and the slice window are the contents of <prefix>_best.txt written by
winding.py. The axial axis is multiplied by --z-display before rendering, so
0.6 shows it compressed 1/0.6 = 1.7x relative to the true geometry, as in the
paper. Writes four views: <out_prefix>_t1.png (side, used in the paper), _t2,
_t3 and _t4_axial. Needs Open3D with offscreen rendering (EGL on a headless
Linux GPU machine).
"""
import argparse

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ply")
ap.add_argument("out_prefix")
ap.add_argument("idA", type=int)
ap.add_argument("idB", type=int)
ap.add_argument("window", nargs="*", type=float, help="first and last slice of the window to show")
ap.add_argument("--um-per-slice", type=float, default=9.0)
ap.add_argument("--z-display", type=float, default=0.6)
args = ap.parse_args()

pts = []
with open(args.ply) as f:
    for line in f:
        p = line.split()
        if len(p) == 4:
            try:
                pts.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
            except ValueError:
                pass
P = np.array(pts, dtype=np.float32)


def centerline(b):
    Q = P[P[:, 3] == b]
    C = []
    for z in np.unique(Q[:, 2]):
        xy = Q[Q[:, 2] == z][:, :2]
        if len(xy) >= 5:
            C.append([xy[:, 0].mean(), xy[:, 1].mean(), z])
    C = np.array(C)
    k = 9
    ker = np.ones(k) / k
    for d in range(2):
        C[:, d] = np.convolve(C[:, d], ker, mode="same")
        C[k // 2:-(k // 2), d] = np.convolve(C[:, d], ker, mode="valid")
    return C


def tube(C, radius, color):
    mesh = o3d.geometry.TriangleMesh()
    for i in range(len(C) - 1):
        a, b = C[i], C[i + 1]
        seg = b - a
        L = np.linalg.norm(seg)
        if L < 1e-6:
            continue
        cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=L, resolution=10, split=1)
        zax = np.array([0, 0, 1.0])
        v = seg / L
        axis = np.cross(zax, v)
        s = np.linalg.norm(axis)
        if s > 1e-8:
            ang = np.arccos(np.clip(np.dot(zax, v), -1, 1))
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / s * ang), center=(0, 0, 0))
        cyl.translate((a + b) / 2)
        mesh += cyl
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


CA, CB = centerline(args.idA), centerline(args.idB)
if len(args.window) == 2:
    z0, z1 = (w * args.um_per_slice for w in args.window)
    pad = (z1 - z0) * 0.35
    CA = CA[(CA[:, 2] >= z0 - pad) & (CA[:, 2] <= z1 + pad)].copy()
    CB = CB[(CB[:, 2] >= z0 - pad) & (CB[:, 2] <= z1 + pad)].copy()
CA[:, 2] *= args.z_display
CB[:, 2] *= args.z_display

radius = 39.6  # um
mA = tube(CA, radius, [0.85, 0.25, 0.15])
mB = tube(CB, radius, [0.15, 0.4, 0.85])
r = rendering.OffscreenRenderer(1400, 1400)
r.scene.set_background([1, 1, 1, 1])
mat = rendering.MaterialRecord()
mat.shader = "defaultLit"
r.scene.add_geometry("a", mA, mat)
r.scene.add_geometry("b", mB, mat)
allv = np.vstack([np.asarray(mA.vertices), np.asarray(mB.vertices)])
c = allv.mean(axis=0)
L = np.linalg.norm(allv.max(axis=0) - allv.min(axis=0))
eps = 1e-3 * L
views = [("t1", c + np.array([L * 0.8, 0, eps]), [0, 0, 1]),
         ("t2", c + np.array([0, L * 0.8, eps]), [0, 0, 1]),
         ("t3", c + np.array([L * 0.55, L * 0.55, L * 0.25]), [0, 0, 1]),
         ("t4_axial", c + np.array([eps, eps, L * 0.9]), [0, 1, 0])]
for name, eye, up in views:
    r.setup_camera(42.0, c.astype(np.float32), np.asarray(eye, dtype=np.float32), np.asarray(up, dtype=np.float32))
    o3d.io.write_image(f"{args.out_prefix}_{name}.png", r.render_to_image())
    print("rendered", name)
