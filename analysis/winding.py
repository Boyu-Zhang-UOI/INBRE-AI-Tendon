"""Find the fascicle pair that winds most around each other (manuscript Figure 3B).

Each fascicle's centerline is the per-slice centroid of its points. Tracks are
cut at jumps (> 72 um between consecutive slices, or a gap of more than 3
slices) and only the longest continuous run of more than 100 slices is kept;
tracks whose centroids stay within 54 um of another track are dropped as
duplicates. For every remaining pair whose mean separation is 54-810 um, the
relative angle of the vector joining them is unwrapped along the tendon, and
the pair with the largest rotation inside a 150-slice window is reported.

Usage:
  python analysis/winding.py <Point3DReconstruction.ply> <out_prefix> [--um-per-slice 9]

Writes <out_prefix>_angle.png (relative angle along the tendon) and
<out_prefix>_best.txt ("idA idB first_slice last_slice" of the fastest window,
the input for render_winding_pair.py).
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JUMP_UM = 72.0        # centroid jump that breaks a track
MAX_GAP_SLICES = 3.0
DUPLICATE_UM = 54.0   # tracks closer than this on average are the same structure
MIN_SEP_UM, MAX_SEP_UM = 54.0, 810.0
WINDOW = 150          # slices

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("ply")
ap.add_argument("out_prefix")
ap.add_argument("--um-per-slice", type=float, default=9.0, help="MICRONS_PER_ZSTEP used by the pipeline")
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
P[:, 2] /= args.um_per_slice   # z in slice units from here on; x, y stay in um

tracks = {}
for b in np.unique(P[:, 3]).astype(int):
    Q = P[P[:, 3] == b]
    t = {}
    for z in np.unique(Q[:, 2]):
        xy = Q[Q[:, 2] == z][:, :2]
        if len(xy) >= 5:
            t[round(float(z), 3)] = xy.mean(axis=0)
    if len(t) <= 100:
        continue
    zs_sorted = sorted(t)
    runs, cur = [], [zs_sorted[0]]
    for z0, z1 in zip(zs_sorted, zs_sorted[1:]):
        if np.linalg.norm(np.asarray(t[z1]) - np.asarray(t[z0])) > JUMP_UM or (z1 - z0) > MAX_GAP_SLICES:
            runs.append(cur)
            cur = []
        cur.append(z1)
    runs.append(cur)
    best = max(runs, key=len)
    if len(best) > 100:
        tracks[b] = {z: t[z] for z in best}

ids = sorted(tracks)
dup = set()
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = tracks[ids[i]], tracks[ids[j]]
        common = sorted(set(a) & set(b))
        if len(common) < 50:
            continue
        if np.linalg.norm(np.array([a[z] - b[z] for z in common]), axis=1).mean() < DUPLICATE_UM:
            dup.add(ids[j])
ids = [i for i in ids if i not in dup]
print(f"tracks: {len(tracks)}, after removing duplicates: {len(ids)}")

results = []
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        a, b = tracks[ids[i]], tracks[ids[j]]
        common = sorted(set(a) & set(b))
        if len(common) < WINDOW + 10:
            continue
        vec = np.array([a[z] - b[z] for z in common])
        dist = np.linalg.norm(vec, axis=1)
        if dist.mean() < MIN_SEP_UM or dist.mean() > MAX_SEP_UM:
            continue
        ang = np.degrees(np.unwrap(np.arctan2(vec[:, 1], vec[:, 0])))
        wturn, k0 = max((abs(ang[k + WINDOW] - ang[k]), k) for k in range(len(ang) - WINDOW))
        results.append((wturn, ids[i], ids[j], float(dist.mean()),
                        common[k0], common[k0 + WINDOW], float(ang[-1] - ang[0])))
results.sort(reverse=True)
print("top pairs (window turn deg, A, B, mean separation um, first slice, last slice, total turn deg):")
for r in results[:6]:
    print(" ", [round(x, 1) if isinstance(x, float) else x for x in r])

if results:
    wturn, A, B, d, z0, z1, tot = results[0]
    a, b = tracks[A], tracks[B]
    common = sorted(set(a) & set(b))
    vec = np.array([a[z] - b[z] for z in common])
    ang = np.degrees(np.unwrap(np.arctan2(vec[:, 1], vec[:, 0])))
    ang -= ang[0]
    mm = np.array(common) * args.um_per_slice / 1000
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(mm, ang, lw=1.4, color="#333")
    ax.axvspan(z0 * args.um_per_slice / 1000, z1 * args.um_per_slice / 1000,
               color="#f5c26b", alpha=0.4, label="fastest winding")
    ax.set_xlabel("Position along tendon (mm)")
    ax.set_ylabel(f"Relative angle of pair ({A},{B}) (deg)")
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_angle.png", dpi=600, bbox_inches="tight")
    print(f"BEST {A},{B}: {wturn:.0f} deg in window z[{z0},{z1}], total {tot:.0f} deg -> {args.out_prefix}_angle.png")
    with open(f"{args.out_prefix}_best.txt", "w") as fh:
        fh.write(f"{A} {B} {z0} {z1}\n")
