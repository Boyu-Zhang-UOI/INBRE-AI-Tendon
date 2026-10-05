"""Per-scan detection metrics on the 30 hand-annotated validation slices (manuscript Figure 5).

Scores any detector checkpoint the way validation/tendon_eval.py scores the
released one: confidence 0.25, seed de-duplication at IoU 0.7, a detection
counts as correct at box IoU >= 0.5, and AP/P/R/F1 are averaged over images.
Pooled values (all detections of a scan ranked together) are reported too.

Usage:
  python analysis/loso_eval.py <weights.pt> [tag]
  CONF=0.25 DEDUP=0.7 python analysis/loso_eval.py runs/loso_P50_1_k5/weights/best.pt loso_k5

Reads the validation images and box annotations under $TENDON_HOME/data
(default ~/tendon; the same layout as validation/tendon_eval.py) and writes
out_loso/<tag>.json. Figure 5 plots the P50_1 rows of the k = 0, 5, 10, 20, 40
models; the dotted line is the released fiberYOLO26Weights.pt.
"""
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(WEIGHTS)
CONF = float(os.environ.get("CONF", "0.25"))
DEDUP = float(os.environ.get("DEDUP", "0.7"))   # 0 turns de-duplication off
HOME = os.environ.get("TENDON_HOME", os.path.expanduser("~/tendon"))
ORIG = f"{HOME}/data/ValidationDesignImagesOriginal30/ValidationDesignImagesOriginal30"
LABELS = f"{HOME}/data/DesignValidationDataHandIdentification/DesignValidationDataHandIdentification/labels"
os.makedirs(f"{HOME}/out_loso", exist_ok=True)


def dedup_boxes(xyxy, conf, iou_thr=0.7):
    """YOLO26 is NMS-free: keep the higher-confidence box of any pair with IoU >= iou_thr (as the pipeline does)."""
    xyxy = np.asarray(xyxy, dtype=float)
    keep = []
    for i in np.argsort(-np.asarray(conf)):
        a = xyxy[i]
        dup = False
        for j in keep:
            b = xyxy[j]
            inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union >= iou_thr:
                dup = True
                break
        if not dup:
            keep.append(int(i))
    return keep


def scan_of(base):
    for pre, key in (("P21(1)", "P21_1"), ("P21(4)", "P21_4"), ("P50_1", "P50_1"), ("P50_3", "P50_3")):
        if base.startswith(pre):
            return key
    return "other"


def load_gt_boxes(base, W, H):
    p = os.path.join(LABELS, base.replace("(", "").replace(")", "") + ".txt")
    if not os.path.exists(p):
        return None
    boxes = []
    for line in open(p):
        f = line.split()
        if len(f) < 9:
            continue
        xs = [float(v) for v in f[1::2]]
        ys = [float(v) for v in f[2::2]]
        boxes.append([min(xs) * W, min(ys) * H, max(xs) * W, max(ys) * H])
    return np.array(boxes)


def box_iou(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def match(gt, preds, confs, thr=0.5):
    order = np.argsort(-confs)
    used, flags = set(), []
    for i in order:
        best, bi = 0, -1
        for j, g in enumerate(gt):
            if j in used:
                continue
            v = box_iou(preds[i], g)
            if v > best:
                best, bi = v, j
        if best >= thr:
            used.add(bi)
            flags.append(1)
        else:
            flags.append(0)
    return confs[order], np.array(flags), len(gt)


def ap_pr(flags, n_gt):
    if not len(flags) or not n_gt:
        return 0., 0., 0., 0.
    tp = np.cumsum(flags)
    fp = np.cumsum(1 - flags)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1)
    ap = sum((rec[i] - rec[i - 1]) * prec[i] for i in range(1, len(rec)))
    p, r = prec[-1], rec[-1]
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return float(ap), float(p), float(r), float(f1)


yolo = YOLO(WEIGHTS)
pools = {}
for f in sorted(os.listdir(ORIG)):
    if not f.lower().endswith((".jpeg", ".jpg")):
        continue
    base = f.rsplit(".", 1)[0]
    img = cv2.imread(os.path.join(ORIG, f))
    H, W = img.shape[:2]
    gtb = load_gt_boxes(base, W, H)
    if gtb is None or not len(gtb):
        continue
    r = yolo(os.path.join(ORIG, f), verbose=False, conf=CONF)[0]
    bx, cf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    if DEDUP > 0:
        keep = dedup_boxes(bx, cf, DEDUP)
        bx, cf = bx[keep], cf[keep]
    c, fl, n = match(gtb, bx, cf)
    per_img = ap_pr(fl, n)
    for key in (scan_of(base), "ALL"):
        d = pools.setdefault(key, {"c": [], "f": [], "n": 0, "imgs": 0, "per": []})
        d["c"] += list(c)
        d["f"] += list(fl)
        d["n"] += n
        d["imgs"] += 1
        d["per"].append(per_img)

out = {}
print(f"\n=== {TAG}  (conf={CONF}, dedup={DEDUP})")
print(f"{'scan':8}{'imgs':>5}{'GT':>6}   per-image avg: {'AP':>6}{'P':>7}{'R':>7}{'F1':>7}   pooled:{'AP':>7}{'F1':>7}")
for key in sorted(pools, key=lambda k: (k == "ALL", k)):
    d = pools[key]
    order = np.argsort(-np.array(d["c"]))
    pool = ap_pr(np.array(d["f"])[order], d["n"])
    per = np.array(d["per"], dtype=float).mean(axis=0)
    out[key] = dict(imgs=d["imgs"], n_gt=d["n"],
                    ap_img=round(per[0], 3), p_img=round(per[1], 3),
                    r_img=round(per[2], 3), f1_img=round(per[3], 3),
                    ap_pool=round(pool[0], 3), f1_pool=round(pool[3], 3))
    print(f"{key:8}{d['imgs']:5d}{d['n']:6d}{'':17}{per[0]:6.3f}{per[1]:7.3f}{per[2]:7.3f}{per[3]:7.3f}{'':10}{pool[0]:6.3f}{pool[3]:7.3f}")

json.dump(out, open(f"{HOME}/out_loso/{TAG}.json", "w"), indent=1)
