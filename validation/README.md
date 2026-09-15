# Validation

Reproducible validation of the detection + segmentation stages against manually
annotated ground truth (30 randomly selected micro-CT slices).

## Contents

| File | Description |
|---|---|
| `gt_register.py` | Registers the hand-drawn segmentation masks (drawn on cropped/zoomed canvases) back onto the original image frames via scale+translation search (zero-mean matched filter on a top-hat fascicle-likelihood map). Writes registered masks and per-image QC overlays. |
| `tendon_eval.py` | Runs YOLO26 (repository weights, detections de-duplicated as in the pipeline) + SAM2 (hiera-large, box prompts, single-mask output) on the 30 validation images and scores against the registered ground truth. Reports detection precision/recall/F1/AP@0.5 and segmentation global IoU/Dice/pixel accuracy plus per-fascicle instance-matched IoU/Dice. |
| `results/` | Output of the current run: summary, per-image CSV, and sample agreement overlays (green = model & annotator agree, orange = annotator only, blue = model only). |

## Current results

```
Detection (YOLO26, IoU threshold 0.5):  AP=0.790  P=0.914  R=0.850  F1=0.879
Segmentation (SAM2, registered GT):     global IoU 0.637 ± 0.082, Dice 0.775 ± 0.061
                                        instance IoU 0.488 ± 0.135, Dice 0.625 ± 0.125
                                        506 of 528 annotated fascicles matched (95.8%)
```

These numbers are for the shipped, fully documented weights retrained with the
recipe in `../training/`. Detections are de-duplicated before scoring exactly as
the pipeline de-duplicates them before seeding SAM2 (YOLO26 is NMS-free and can
emit two near-identical boxes on one fascicle; of any pair with IoU ≥ 0.7 the
higher-confidence box is kept). Without that step, duplicate boxes count as
false positives and precision reads 0.767 (F1 0.804) for the same detector.

Note: the hand-drawn segmentation ground truth covers a subset of the fascicles
visible in each slice (~18 per image), painted coarsely with a tablet; the
global metrics therefore penalize correct model predictions of unannotated
fascicles, and the instance metrics reflect annotator/model boundary style
differences as well as model error.

## Running

Both scripts read paths from constants at the top of the file — edit them to
match your layout, then:

```
python gt_register.py    # writes gt_registered/ + qc/ + register_log.csv
python tendon_eval.py    # writes eval_summary.txt, eval_results.csv, eval_vis/
```

Requirements: `ultralytics`, `sam2` (SAM 2 repository), `opencv-python`, plus
the `sam2_hiera_large.pt` checkpoint.
