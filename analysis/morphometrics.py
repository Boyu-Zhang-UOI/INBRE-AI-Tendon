"""Fascicle morphometrics from a TendonTrack point cloud (manuscript Figure 4).

For every fascicle (blob_id) and every slice, the cross-sectional area is the
convex-hull area of that fascicle's points in the slice; the equivalent
diameter is 2*sqrt(mean area / pi). Fascicles present on fewer than 20 slices
are skipped.

Usage:
  python analysis/morphometrics.py <Point3DReconstruction.ply> <out_dir> [--exclude 23,28,29]

The PLY is the pipeline output (coordinates in micrometres, 4th column blob_id).
--exclude lists blob ids an expert identified as non-fascicle structures; for
the example stack in the paper these are 23, 28 and 29.

Writes <out_dir>/Figure4_morphometrics.png/.pdf and
<out_dir>/fascicle_diameters.csv (blob_id, n_slices, mean_area_um2,
equivalent_diameter_um).
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def read_ply(path):
    pts = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 4:
                try:
                    pts.append([float(p[0]), float(p[1]), float(p[2]), float(p[3])])
                except ValueError:
                    pass
    return np.array(pts, dtype=np.float32)


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ply")
ap.add_argument("out_dir")
ap.add_argument("--exclude", default="", help="comma-separated blob ids to leave out")
args = ap.parse_args()
EXCLUDE = {int(x) for x in args.exclude.split(",") if x.strip()}
os.makedirs(args.out_dir, exist_ok=True)

P = read_ply(args.ply)
print("points:", P.shape)
blob_ids = np.unique(P[:, 3]).astype(int)
print("blobs in reconstruction:", len(blob_ids), "| excluded:", sorted(EXCLUDE))
blob_ids = np.array([b for b in blob_ids if b not in EXCLUDE])
P = P[np.isin(P[:, 3].astype(int), blob_ids)]
zs = np.unique(P[:, 2])
print("fascicles:", len(blob_ids), "| z range (um):", zs.min(), zs.max())

curves = {}
mean_diam = []
rows = []
for b in blob_ids:
    Q = P[P[:, 3] == b]
    areas = []
    for z in np.unique(Q[:, 2]):
        xy = Q[Q[:, 2] == z][:, :2]
        if len(xy) < 5:
            continue
        try:
            a = ConvexHull(xy).volume  # for 2D points, ConvexHull.volume is the area
        except Exception:
            continue
        areas.append((z, a))
    if len(areas) < 20:
        continue
    arr = np.array(areas)
    curves[b] = arr
    d = 2 * np.sqrt(arr[:, 1].mean() / np.pi)
    mean_diam.append(d)
    rows.append([int(b), len(arr), round(float(arr[:, 1].mean()), 1), round(float(d), 2)])

print("fascicles with >=20 usable slices:", len(curves))
print("mean equivalent diameter (um): %.1f +/- %.1f  (n=%d, range %.0f-%.0f)" % (
    np.mean(mean_diam), np.std(mean_diam), len(mean_diam), min(mean_diam), max(mean_diam)))

with open(os.path.join(args.out_dir, "fascicle_diameters.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["blob_id", "n_slices", "mean_area_um2", "equivalent_diameter_um"])
    w.writerows(rows)

counts = np.array([(z, len(np.unique(P[P[:, 2] == z][:, 3]))) for z in zs])

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
ax = axes[0]
for b, arr in curves.items():
    z, a = arr[:, 0], arr[:, 1]
    k = max(1, len(a) // 80)
    sm = np.convolve(a, np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    ax.plot(z[k:-k] if k else z, sm, lw=0.7, alpha=0.55)
ax.set_xlabel("Position along tendon (μm)")
ax.set_ylabel("Cross-sectional area (μm²)")
ax.set_title("A  Per-fascicle area profiles", fontsize=10, loc="left")

ax = axes[1]
ax.hist(mean_diam, bins=12, color="#4878a8", edgecolor="white")
ax.set_xlabel("Mean equivalent diameter (μm)")
ax.set_ylabel("Fascicles")
ax.set_title("B  Fascicle diameter distribution", fontsize=10, loc="left")

ax = axes[2]
ax.plot(counts[:, 0], counts[:, 1], color="#4878a8", lw=1.2)
ax.set_xlabel("Position along tendon (μm)")
ax.set_ylabel("Fascicles present")
ax.set_ylim(0, counts[:, 1].max() + 2)
ax.set_title("C  Fascicle count along the tendon", fontsize=10, loc="left")

plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, "Figure4_morphometrics.png"), dpi=600, bbox_inches="tight")
plt.savefig(os.path.join(args.out_dir, "Figure4_morphometrics.pdf"), bbox_inches="tight")
print("figure saved to", args.out_dir)
