"""Sample training slices from the raw rec stacks (excluding validation slices),
preprocess like the pipeline (8-bit + Laplacian sharpen), and prelabel them for
human review (model-assisted labeling).

IMPORTANT - provenance: the labels released in this repository were produced by
the PREVIOUS detector, the one that existed before the documented retraining,
not by the shipped fiberYOLO26Weights.pt that train_yolo.py then produced from
them. Point PRELABEL_WEIGHTS at that prior checkpoint to reproduce the released
training set; using the shipped weights here would prelabel with a model trained
on this very set and yield different labels.

Outputs under ~/tendon/trainset/:
  images/<name>.jpg        preprocessed training images
  labels/<name>.txt        YOLO-format prelabels (class cx cy w h, normalized)
  manifest.csv             source path, scan, slice index, n_boxes per image
  qc/qc_sheet*.jpg         contact sheets of prelabeled samples
"""
import os, re, csv, random
import cv2
import numpy as np
from ultralytics import YOLO

random.seed(42)
HOME = os.path.expanduser("~/tendon")
RAW = f"{HOME}/rawdata"
OUT = f"{HOME}/trainset"
os.makedirs(f"{OUT}/images", exist_ok=True)
os.makedirs(f"{OUT}/labels", exist_ok=True)
os.makedirs(f"{OUT}/qc", exist_ok=True)

# scan key -> (directory, filename regex for rec slices, validation indices to exclude)
SCANS = {
    "P21_1": (f"{RAW}/P21_1M/P21_1M", r"^P21\(1\)_rec0*(\d+)\.tif$", {25, 27, 63, 83, 99}, 45),
    "P21_4": (f"{RAW}/P21_4M/P21_4M", r"^P21\(4\)_rec0*(\d+)\.tif$", {31, 43, 52, 67, 96, 422, 440, 506}, 45),
    "P50_1": (f"{RAW}/P50_1/P50_1", r"^P50_1_HRScan_rec0*(\d+)\.tif$", {249, 285, 297, 364, 394, 427, 520, 571, 641, 675}, 40),
    "P50_3sr": (f"{RAW}/OneDrive_2026-08-14_(2)/P50 3 Prox Tail SR Reconstructed", r"^P50_3_Tail_Proximal-Real_rec_Tra0*(\d+)\.png$", set(), 45),
    "P21_3": (f"{RAW}/OneDrive_2026-08-14_(1)/P21 3 Prox Tail HR", r"^P21 3 Prox Tail HR_rec0*(\d+)\.tif$", set(), 25),
}


def to_8bit(img):
    if img.dtype == np.uint16:
        lo, hi = np.percentile(img, (0.5, 99.5))
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def sharpen(gray):
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    sharp = gray.astype(np.int16) - lap
    return np.clip(sharp, 0, 255).astype(np.uint8)


# The detector used to generate the prelabels (see the provenance note above).
PRELABEL_WEIGHTS = os.environ.get("PRELABEL_WEIGHTS", f"{HOME}/fiberYOLO26Weights.pt")
yolo = YOLO(PRELABEL_WEIGHTS)
rows = []
picked = []
for key, (d, pat, excl, n) in SCANS.items():
    files = []
    for f in os.listdir(d):
        m = re.match(pat, f)
        if m and int(m.group(1)) not in excl:
            files.append((int(m.group(1)), f))
    files.sort()
    if len(files) < n:
        n = len(files)
    # even spacing through the stack, jittered, then dedup
    idxs = sorted({min(len(files) - 1, int(i * len(files) / n) + random.randint(0, max(1, len(files) // n) - 1))
                   for i in range(n)})
    for i in idxs:
        sl, f = files[i]
        picked.append((key, d, f, sl))

print(f"sampled {len(picked)} slices")
for key, d, f, sl in picked:
    img = cv2.imread(os.path.join(d, f), cv2.IMREAD_UNCHANGED)
    if img is None:
        print("skip unreadable:", f)
        continue
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = sharpen(to_8bit(img))
    rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    name = f"{key}_rec{sl:06d}"
    cv2.imwrite(f"{OUT}/images/{name}.jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])

    r = yolo(rgb, verbose=False)[0]
    H, W = rgb.shape[:2]
    lines = []
    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        if c < 0.25:
            continue
        x1, y1, x2, y2 = b
        cx, cy, w, h = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    open(f"{OUT}/labels/{name}.txt", "w").write("\n".join(lines) + "\n")
    rows.append([name, key, sl, os.path.join(d, f), len(lines)])

with open(f"{OUT}/manifest.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "scan", "slice", "source", "n_prelabels"])
    w.writerows(rows)

# QC sheets: draw boxes on a sample of images
sample = random.sample(rows, min(12, len(rows)))
tiles = []
for name, key, sl, src, nb in sample:
    im = cv2.imread(f"{OUT}/images/{name}.jpg")
    for line in open(f"{OUT}/labels/{name}.txt"):
        p = line.split()
        if len(p) != 5:
            continue
        _, cx, cy, w, h = map(float, p)
        H, W = im.shape[:2]
        x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
        x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
    im = cv2.resize(im, (420, int(im.shape[0] * 420 / im.shape[1])))
    cv2.putText(im, f"{name} ({nb} boxes)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(im)
hh = max(t.shape[0] for t in tiles)
tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for t in tiles]
rows_img = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
ww = max(r.shape[1] for r in rows_img)
rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, ww - r.shape[1], cv2.BORDER_CONSTANT, value=0) for r in rows_img]
cv2.imwrite(f"{OUT}/qc/qc_sheet.jpg", np.vstack(rows_img))

counts = [r[4] for r in rows]
print(f"done: {len(rows)} images | prelabels/image mean {np.mean(counts):.1f} min {min(counts)} max {max(counts)}")
per = {}
for r in rows:
    per[r[1]] = per.get(r[1], 0) + 1
print("per scan:", per)
