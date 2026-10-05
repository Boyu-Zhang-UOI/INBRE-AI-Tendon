"""Leave-one-scan-out retraining for the adaptation experiment (manuscript Figure 5).

The scan named on the command line is removed from the training set and then k
of its slices are added back (k = 0 withholds it entirely). Everything else is
the recipe of training/train_yolo.py: per-scan 80/20 split with seed 42,
YOLO26-nano from the COCO checkpoint, 50 epochs, batch 20, imgsz 640.

Usage:
  python analysis/loso_train.py P50_1 [k] [--prepare-only]

Expects the training set written by training/sample_prelabel.py under
$TENDON_HOME/trainset (default ~/tendon). Writes the split to
trainset/yolo_ds_loso_<scan>[_k<k>], weights to runs/loso_<scan>[_k<k>]/weights/best.pt
and the training time to out_loso/traintime_<scan>_k<k>.json. The paper used
k = 0, 5, 10, 20 and 40 for P50_1; score each model with loso_eval.py.
"""
import argparse
import csv
import json
import os
import random
import shutil
import time

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("scan")
ap.add_argument("k", type=int, nargs="?", default=0)
ap.add_argument("--prepare-only", action="store_true", help="write the split but do not train")
args = ap.parse_args()

random.seed(42)
HOME = os.environ.get("TENDON_HOME", os.path.expanduser("~/tendon"))
TS = f"{HOME}/trainset"
NAME = f"loso_{args.scan}" + (f"_k{args.k}" if args.k else "")
DS = f"{TS}/yolo_ds_{NAME}"

by_scan = {}
with open(f"{TS}/manifest.csv") as fh:
    for row in csv.DictReader(fh):
        by_scan.setdefault(row["scan"], []).append(row["name"])
assert args.scan in by_scan, f"unknown scan {args.scan}; have {sorted(by_scan)}"
held = sorted(by_scan.pop(args.scan))
if args.k:
    random.shuffle(held)
addback = held[:args.k]
print(f"held out {args.scan} ({len(held)} slices); adding back {len(addback)}")

splits = {"train": [], "val": []}
for scan, names in sorted(by_scan.items()):
    names = sorted(names)
    random.shuffle(names)
    n = max(1, int(round(0.2 * len(names))))
    splits["val"] += names[:n]
    splits["train"] += names[n:]
splits["train"] += addback
print({s: len(v) for s, v in splits.items()})

if os.path.isdir(DS):
    shutil.rmtree(DS)
for s, names in splits.items():
    for sub in ("images", "labels"):
        os.makedirs(f"{DS}/{sub}/{s}", exist_ok=True)
    for n in names:
        shutil.copy(f"{TS}/images/{n}.jpg", f"{DS}/images/{s}/{n}.jpg")
        shutil.copy(f"{TS}/labels/{n}.txt", f"{DS}/labels/{s}/{n}.txt")
with open(f"{DS}/data.yaml", "w") as fh:
    fh.write(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: Fiber\n")
if args.prepare_only:
    raise SystemExit(f"split written to {DS}")

from ultralytics import YOLO

t0 = time.time()
model = YOLO("yolo26n.pt")
model.train(data=f"{DS}/data.yaml", epochs=50, batch=20, imgsz=640, seed=42,
            project=f"{HOME}/runs", name=NAME, exist_ok=True, device=0, verbose=False)
dt = time.time() - t0
os.makedirs(f"{HOME}/out_loso", exist_ok=True)
json.dump({"heldout": args.scan, "k": args.k, "n_train": len(splits["train"]), "train_seconds": round(dt, 1)},
          open(f"{HOME}/out_loso/traintime_{args.scan}_k{args.k}.json", "w"), indent=1)
print(f"TRAINTIME {args.scan} k={args.k}: {dt:.1f}s")
print("weights:", f"{HOME}/runs/{NAME}/weights/best.pt")
