# Training

Fully reproducible training recipe for the shipped detector weights
(`fiberYOLO26Weights.pt`).

## Dataset

200 slices sampled at even, jittered intervals (seed 42) along five micro-CT
scans: P21(1), P21(4) and the P50_3 SR reconstruction (45 each), P50_1 (40) and
P21(3) (25), with all validation slices excluded by index. `manifest.csv`
records every training image's source scan, slice index, and prelabel count.
Slices were converted to 8 bits with a 0.5–99.5 percentile contrast stretch and
sharpened by subtracting their 3×3 Laplacian (`sample_prelabel.py`). This is
close to, but not the same as, the pipeline's preprocessing (16-bit values
divided by 256 and an 8-neighbour sharpening kernel); the detector is evaluated
on pipeline-preprocessed slices (`../validation/`).

Bounding-box labels were produced by the previous detector (trained on manual
annotations) and used after visual spot checks — model-assisted labeling
(`sample_prelabel.py`, confidence ≥ 0.25; mean 44.3 boxes/slice).

## Recipe

`train_yolo.py`: YOLO26-nano from the COCO-pretrained checkpoint, 50 epochs,
batch 20, imgsz 640, default Ultralytics augmentation, seed 42, per-scan 80/20
train/val split (160/40). Ultralytics 8.4.120, PyTorch 2.13.0+cu126, single
NVIDIA RTX 4000 Ada.

## Results

Held-out prelabel val split: mAP50 0.821, mAP50-95 0.726, P 0.801, R 0.755.
On the 30 independently hand-annotated validation slices (see `../validation/`,
detections de-duplicated as in the pipeline): detection P 0.914, R 0.850,
F1 0.879, AP 0.790; segmentation via SAM2 global IoU 0.637 ± 0.082,
Dice 0.775 ± 0.061; per-fascicle matched IoU 0.488 ± 0.135, 506/528 fascicles
recovered (95.8%).
