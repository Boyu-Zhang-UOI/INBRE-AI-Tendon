"""Fixed validation for the tendon fascicle pipeline.

Fixes vs the original DesignValidation.ipynb:
  - uses the repository weights (fiberYOLO26Weights.pt), not fiberYOLOBB2.pt
  - SAM2 multimask_output=False (single best mask per box, as the pipeline uses)
  - detections de-duplicated (IoU >= 0.7, keep higher confidence) exactly as the pipeline seeds SAM2
  - ground truth = hand-drawn masks REGISTERED onto the original canvases
  - adds per-fascicle instance-matched IoU/Dice alongside the global-mask metrics

Outputs: ~/tendon/out/eval_results.csv, eval_summary.txt, overlays in out/eval_vis/
"""
import os, csv, glob
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

HOME = os.path.expanduser("~/tendon")
ORIG = f"{HOME}/data/ValidationDesignImagesOriginal30/ValidationDesignImagesOriginal30"
LABELS = f"{HOME}/data/DesignValidationDataHandIdentification/DesignValidationDataHandIdentification/labels"
GTREG = f"{HOME}/out/gt_registered"
OUT = f"{HOME}/out"
os.makedirs(f"{OUT}/eval_vis", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

yolo = YOLO(f"{HOME}/fiberYOLO26Weights.pt")

ckpt = f"{HOME}/checkpoints/sam2_hiera_large.pt"
cfg_candidates = ["sam2_hiera_l.yaml", "configs/sam2/sam2_hiera_l.yaml"]
sam = None
for cfg in cfg_candidates:
    try:
        sam = build_sam2(cfg, ckpt, device=DEVICE)
        print("sam2 config:", cfg)
        break
    except Exception as e:
        last = e
if sam is None:
    raise SystemExit(f"SAM2 config not found: {last}")
predictor = SAM2ImagePredictor(sam)



def dedup_boxes(xyxy, conf, iou_thr=0.7):
    """YOLO26 is NMS-free: keep the higher-confidence box of any pair with IoU >= iou_thr (as the pipeline does)."""
    xyxy = np.asarray(xyxy, dtype=float); keep = []
    for i in np.argsort(-np.asarray(conf)):
        a = xyxy[i]; dup = False
        for j in keep:
            b = xyxy[j]
            inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union >= iou_thr:
                dup = True; break
        if not dup: keep.append(int(i))
    return keep


def norm_name(s):
    return s.replace("(", "").replace(")", "")


def load_gt_boxes(base, W, H):
    """Label Studio polygon export: class x1 y1 x2 y1 x2 y2 x1 y2 (normalized)."""
    p = os.path.join(LABELS, norm_name(base) + ".txt")
    boxes = []
    if not os.path.exists(p):
        return None
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


def det_metrics(gt, preds, confs, thr=0.5):
    """Greedy conf-descending matching; returns (tp_flags sorted by conf, n_gt)."""
    order = np.argsort(-confs)
    used = set()
    flags = []
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
    return np.array(flags), len(gt)


def ap_from_flags(flags, n_gt):
    tp = np.cumsum(flags)
    fp = np.cumsum(1 - flags)
    rec = tp / n_gt if n_gt else tp * 0
    prec = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for i in range(1, len(rec)):
        ap += (rec[i] - rec[i - 1]) * prec[i]
    p = prec[-1] if len(prec) else 0.0
    r = rec[-1] if len(rec) else 0.0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return ap, p, r, f1


def seg_metrics(pred_masks, gt_mask):
    """Global combined-mask metrics + per-instance matched IoU/Dice."""
    combined = np.zeros_like(gt_mask)
    for m in pred_masks:
        combined |= m
    inter = np.logical_and(combined, gt_mask).sum()
    union = np.logical_or(combined, gt_mask).sum()
    g_iou = inter / union if union else 0.0
    s = combined.sum() + gt_mask.sum()
    g_dice = 2 * inter / s if s else 0.0
    g_pa = (combined == gt_mask).mean()

    # GT instances from connected components
    n, cc = cv2.connectedComponents(gt_mask.astype(np.uint8))
    gt_inst = [(cc == i) for i in range(1, n) if (cc == i).sum() > 30]
    pairs = []
    for pi, pm in enumerate(pred_masks):
        for gi, gm in enumerate(gt_inst):
            i2 = np.logical_and(pm, gm).sum()
            if i2 == 0:
                continue
            u2 = np.logical_or(pm, gm).sum()
            pairs.append((i2 / u2, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, matched = set(), set(), []
    for v, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); matched.append(v)
    inst_iou = float(np.mean(matched)) if matched else 0.0
    inst_dice = float(np.mean([2 * v / (1 + v) for v in matched])) if matched else 0.0
    return g_iou, g_dice, g_pa, inst_iou, inst_dice, len(matched), len(gt_inst), len(pred_masks)


rows = []
names = sorted(f for f in os.listdir(ORIG) if f.lower().endswith((".jpeg", ".jpg")))
all_flags, total_gt = [], 0
per_img_ap = []

for f in names:
    base = f.rsplit(".", 1)[0]
    img_bgr = cv2.imread(os.path.join(ORIG, f))
    H, W = img_bgr.shape[:2]

    r = yolo(os.path.join(ORIG, f), verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    keep = dedup_boxes(boxes, confs)          # same seed de-duplication as the pipeline
    boxes, confs = boxes[keep], confs[keep]

    gtb = load_gt_boxes(base, W, H)
    det_line = ""
    if gtb is not None and len(gtb):
        flags, n_gt = det_metrics(gtb, boxes, confs)
        ap, p, rec, f1 = ap_from_flags(flags, n_gt)
        per_img_ap.append((ap, p, rec, f1))
        all_flags.append((confs[np.argsort(-confs)], flags))
        total_gt += n_gt
        det_line = f"AP={ap:.3f} P={p:.3f} R={rec:.3f}"

    gt_path = os.path.join(GTREG, base + ".png")
    seg_line = ""
    if os.path.exists(gt_path) and len(boxes):
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) > 0
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(box=boxes, multimask_output=False)
        masks = np.asarray(masks)
        if masks.ndim == 4:  # (B,1,H,W)
            masks = masks[:, 0]
        elif masks.ndim == 3 and masks.shape[0] == 1 and len(boxes) == 1:
            pass
        pred_masks = [m.astype(bool) for m in masks]
        gi, gd, gpa, ii, idice, nm, ng, npd = seg_metrics(pred_masks, gt_mask)
        rows.append([base, gi, gd, gpa, ii, idice, nm, ng, npd])
        seg_line = f"gIoU={gi:.3f} gDice={gd:.3f} instIoU={ii:.3f} matched={nm}/{ng}"

        vis = img_bgr.copy()
        comb = np.zeros(gt_mask.shape, bool)
        for m in pred_masks:
            comb |= m
        vis[gt_mask & ~comb] = (0.4 * vis[gt_mask & ~comb] + 0.6 * np.array([0, 200, 255])).astype(np.uint8)   # GT only: orange
        vis[comb & ~gt_mask] = (0.4 * vis[comb & ~gt_mask] + 0.6 * np.array([255, 80, 0])).astype(np.uint8)    # pred only: blue
        vis[comb & gt_mask] = (0.4 * vis[comb & gt_mask] + 0.6 * np.array([0, 220, 0])).astype(np.uint8)       # agreement: green
        cv2.imwrite(f"{OUT}/eval_vis/{base}.jpg", vis)

    print(f"{base}: {det_line} {seg_line}", flush=True)

# dataset-level detection AP (all detections pooled)
pool = sorted(
    [(c, fl) for confs, flags in all_flags for c, fl in zip(confs, flags)],
    key=lambda t: -t[0],
)
flags = np.array([fl for _, fl in pool])
ds_ap, ds_p, ds_r, ds_f1 = ap_from_flags(flags, total_gt)

a = np.array([r[1:6] for r in rows], dtype=float)
mean_ap = np.mean([x[0] for x in per_img_ap]); mean_p = np.mean([x[1] for x in per_img_ap])
mean_r = np.mean([x[2] for x in per_img_ap]); mean_f1 = np.mean([x[3] for x in per_img_ap])

summary = f"""FIXED VALIDATION RESULTS ({len(rows)} images, device={DEVICE})

Detection (YOLO26, repo weights, seed de-duplication IoU 0.7, IoU threshold 0.5):
  per-image averaged:  AP={mean_ap:.3f}  P={mean_p:.3f}  R={mean_r:.3f}  F1={mean_f1:.3f}
  dataset-pooled:      AP={ds_ap:.3f}  P={ds_p:.3f}  R={ds_r:.3f}  F1={ds_f1:.3f}

Segmentation (SAM2 hiera-large, box prompts, single-mask; registered hand-drawn GT):
  global mask IoU:   {a[:,0].mean():.3f} +/- {a[:,0].std():.3f}
  global Dice:       {a[:,1].mean():.3f} +/- {a[:,1].std():.3f}
  pixel accuracy:    {a[:,2].mean():.3f}
  instance IoU:      {a[:,3].mean():.3f} +/- {a[:,3].std():.3f}   (matched fascicle pairs)
  instance Dice:     {a[:,4].mean():.3f} +/- {a[:,4].std():.3f}
  matched fascicles: {int(sum(r[6] for r in rows))} of {int(sum(r[7] for r in rows))} GT instances
"""
print("\n" + summary)
with open(f"{OUT}/eval_summary.txt", "w") as fh:
    fh.write(summary)
with open(f"{OUT}/eval_results.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["image", "global_iou", "global_dice", "pixel_acc",
                "inst_iou", "inst_dice", "n_matched", "n_gt_inst", "n_pred"])
    w.writerows(rows)
