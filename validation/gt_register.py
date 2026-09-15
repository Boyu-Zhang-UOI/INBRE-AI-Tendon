"""Register hand-drawn GT segmentation masks (arbitrary cropped canvases)
back onto the original micro-CT frames via scale+translation search with NCC.

Outputs:
  out/gt_registered/<name>.png  — binary GT mask on the original canvas
  out/qc/<name>.jpg             — side-by-side overlay for visual QC
  out/register_log.csv          — per-image scale, offset, NCC score
"""
import os, csv, sys
import cv2
import numpy as np

ORIG = os.path.expanduser("~/tendon/data/ValidationDesignImagesOriginal30/ValidationDesignImagesOriginal30")
SEG = os.path.expanduser("~/tendon/data/DesignValidationDataHandSegmentation/DesignValidationDataHandSegmentation")
OUT = os.path.expanduser("~/tendon/out")
os.makedirs(f"{OUT}/gt_registered", exist_ok=True)
os.makedirs(f"{OUT}/qc", exist_ok=True)

BLUR = 5  # soften both signals so near-alignment scores well


def likelihood_map(gray):
    """Fascicle-likelihood proxy: bright small structures, background-suppressed."""
    g = gray.astype(np.float32) / 255.0
    # top-hat to emphasize bright blobs over larger-scale background
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
    m = cv2.GaussianBlur(tophat, (BLUR, BLUR), 0)
    return m


def register_one(orig_path, seg_path):
    img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    rgba = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
        return None
    gt = (rgba[:, :, 3] > 0).astype(np.float32)
    gh, gw = gt.shape

    like = likelihood_map(img)
    like_z = like - like.mean()  # zero-mean so empty padding scores 0
    pad = int(0.45 * max(H, W))
    like_p = cv2.copyMakeBorder(like_z, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    PH, PW = like_p.shape

    def try_scale(s):
        """Zero-mean matched filter, valid only where >=60% of template overlaps image."""
        tw, th = int(round(gw * s)), int(round(gh * s))
        if tw < 40 or th < 40 or tw >= PW or th >= PH:
            return None
        tmpl = cv2.resize(gt, (tw, th), interpolation=cv2.INTER_AREA)
        tmpl = cv2.GaussianBlur(tmpl, (BLUR, BLUR), 0)
        tz = tmpl - tmpl.mean()
        norm = np.sqrt((tz ** 2).sum())
        if norm < 1e-6:
            return None
        res = cv2.matchTemplate(like_p, tz, cv2.TM_CCORR) / norm
        # mask out placements with <60% horizontal/vertical overlap with the image
        ph, pw = res.shape
        x = np.arange(pw)[None, :] - pad
        y = np.arange(ph)[:, None] - pad
        ox = np.minimum(x + tw, W) - np.maximum(x, 0)
        oy = np.minimum(y + th, H) - np.maximum(y, 0)
        valid = (ox >= 0.6 * tw) & (oy >= 0.6 * th)
        if not valid.any():
            return None
        res[~valid] = -1e9
        _, mx, _, loc = cv2.minMaxLoc(res)
        return mx, s, loc[0] - pad, loc[1] - pad

    cands = [r for r in (try_scale(s) for s in np.geomspace(0.25, 1.8, 60)) if r]
    score, s, tx, ty = max(cands)
    for r in (try_scale(s2) for s2 in np.geomspace(s * 0.93, s * 1.07, 25)):
        if r and r[0] > score:
            score, s, tx, ty = r

    # render registered GT on original canvas
    tw, th = int(round(gw * s)), int(round(gh * s))
    tmpl = cv2.resize(gt, (tw, th), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((H, W), np.uint8)
    x0, y0 = tx, ty
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(W, x0 + tw), min(H, y0 + th)
    if xe > xs and ye > ys:
        canvas[ys:ye, xs:xe] = (tmpl[ys - y0:ye - y0, xs - x0:xe - x0] > 0.5) * 255
    return canvas, score, s, tx, ty


def qc_image(orig_path, mask, name, score):
    img = cv2.imread(orig_path)
    ov = img.copy()
    ov[mask > 0] = (0.35 * ov[mask > 0] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    both = np.hstack([img, ov])
    cv2.putText(both, f"{name}  NCC={score:.3f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return both


rows = []
names = sorted(f for f in os.listdir(ORIG) if f.lower().endswith((".jpeg", ".jpg")))
for f in names:
    base = f.rsplit(".", 1)[0]
    segp = os.path.join(SEG, base + ".png")
    if not os.path.exists(segp):
        print(f"{base}: GT missing"); continue
    r = register_one(os.path.join(ORIG, f), segp)
    if r is None:
        print(f"{base}: bad GT file"); continue
    mask, score, s, tx, ty = r
    cv2.imwrite(f"{OUT}/gt_registered/{base}.png", mask)
    cv2.imwrite(f"{OUT}/qc/{base}.jpg", qc_image(os.path.join(ORIG, f), mask, base, score))
    rows.append([base, f"{score:.4f}", f"{s:.4f}", tx, ty])
    print(f"{base}: NCC={score:.3f} scale={s:.3f} offset=({tx},{ty})")

with open(f"{OUT}/register_log.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image", "ncc", "scale", "tx", "ty"])
    w.writerows(rows)
scores = [float(r[1]) for r in rows]
print(f"\n{len(rows)} images registered. score mean={np.mean(scores):.3f} min={np.min(scores):.3f}")
order = np.argsort(scores)
print("Lowest-scoring 5 (check QC first):", [rows[i][0] for i in order[:5]])
