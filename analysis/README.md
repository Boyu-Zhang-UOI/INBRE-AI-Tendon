# Analysis

Scripts behind the downstream analyses in the paper: fascicle morphometrics (Figure 4), fascicle winding (Figure 3B) and the cost of adapting the detector to an unseen scan (Figure 5). The first three take the pipeline's `OUTPUT/Point3DReconstruction.ply`, in which every point carries its fascicle id (`blob_id`) and coordinates are in micrometres.

| Script | Paper | Input | Output |
|---|---|---|---|
| `morphometrics.py` | Figure 4, Section 3 | point cloud | per-fascicle area profiles, equivalent diameters (`fascicle_diameters.csv`), fascicle count along the tendon |
| `winding.py` | Figure 3B, Section 3 | point cloud | the fascicle pair that rotates most around each other, its angle profile, `<prefix>_best.txt` |
| `render_winding_pair.py` | Figure 3B | point cloud + `_best.txt` | tube renders of the two centerlines |
| `loso_train.py` | Figure 5, Section 4 | training set from `../training/sample_prelabel.py` | detector retrained with one scan withheld, plus k of its slices |
| `loso_eval.py` | Figure 5, Section 4 | validation slices (see `../validation/`) | per-scan detection metrics |

## Reproducing the paper's numbers

For the 727-slice example stack (`scripts/run_pipeline.py`, default settings), an expert marked blob ids 23, 28 and 29 as non-tendon structures.

```bash
python analysis/morphometrics.py RUN/OUTPUT/Point3DReconstruction.ply out --exclude 23,28,29
# mean equivalent diameter 202.0 +/- 48.7 um (n = 27), segment 6516 um long
python analysis/winding.py RUN/OUTPUT/Point3DReconstruction.ply out/twist
# BEST 16,17: window slices 117-267, total rotation -139 deg
python analysis/render_winding_pair.py RUN/OUTPUT/Point3DReconstruction.ply out/tube 16 17 117 267
```

`render_winding_pair.py` shows the axial axis at 0.6 of its true length (`--z-display`), as in Figure 3B. It needs Open3D with offscreen rendering.

Figure 5 retrains the detector with scan P50_1 withheld and then adds back k of its slices:

```bash
for k in 0 5 10 20 40; do python analysis/loso_train.py P50_1 $k; done
python analysis/loso_eval.py ~/tendon/runs/loso_P50_1/weights/best.pt loso_k0
python analysis/loso_eval.py ~/tendon/runs/loso_P50_1_k5/weights/best.pt loso_k5   # and so on
python analysis/loso_eval.py fiberYOLO26Weights.pt shipped                          # dotted line
```

The P50_1 rows give AP 0.10, 0.62, 0.69, 0.77 and 0.75 for k = 0, 5, 10, 20, 40, and 0.75 for the released detector. Each retraining takes about 90 s on one RTX 4000 Ada. The training images and labels are not in this repository; they are available from the corresponding author on request, as are the validation slices. Both LOSO scripts look for them under `$TENDON_HOME` (default `~/tendon`), in the layout used by `training/` and `validation/`.
