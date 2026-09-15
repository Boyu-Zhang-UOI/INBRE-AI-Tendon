"""Retrain YOLO26n on the prelabeled training set (model-assisted labels).

Split: per-scan 80/20, seed 42. Config matches the original run:
yolo26n.pt init, 50 epochs, batch 20, imgsz 640, default augmentation.
"""
import os, csv, random, shutil
from ultralytics import YOLO

random.seed(42)
HOME = os.path.expanduser("~/tendon")
TS = f"{HOME}/trainset"
DS = f"{HOME}/trainset/yolo_ds"

by_scan = {}
with open(f"{TS}/manifest.csv") as fh:
    for row in csv.DictReader(fh):
        by_scan.setdefault(row["scan"], []).append(row["name"])

splits = {"train": [], "val": []}
for scan, names in sorted(by_scan.items()):
    names = sorted(names)
    random.shuffle(names)
    k = max(1, int(round(0.2 * len(names))))
    splits["val"] += names[:k]
    splits["train"] += names[k:]
print({s: len(v) for s, v in splits.items()})

for s, names in splits.items():
    for sub in ("images", "labels"):
        os.makedirs(f"{DS}/{sub}/{s}", exist_ok=True)
    for n in names:
        shutil.copy(f"{TS}/images/{n}.jpg", f"{DS}/images/{s}/{n}.jpg")
        shutil.copy(f"{TS}/labels/{n}.txt", f"{DS}/labels/{s}/{n}.txt")

with open(f"{DS}/data.yaml", "w") as fh:
    fh.write(f"path: {DS}\ntrain: images/train\nval: images/val\nnames:\n  0: Fiber\n")

model = YOLO("yolo26n.pt")
model.train(data=f"{DS}/data.yaml", epochs=50, batch=20, imgsz=640, seed=42,
            project=f"{HOME}/runs", name="tendontrack_v2", exist_ok=True,
            device=0, verbose=False)
metrics = model.val(data=f"{DS}/data.yaml")
print("val mAP50:", round(metrics.box.map50, 3), "| mAP50-95:", round(metrics.box.map, 3),
      "| P:", round(metrics.box.mp, 3), "| R:", round(metrics.box.mr, 3))
print("best weights:", f"{HOME}/runs/tendontrack_v2/weights/best.pt")
