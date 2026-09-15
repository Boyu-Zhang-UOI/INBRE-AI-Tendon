# TendonTrack

**TendonTrack** is an automated machine-learning pipeline for identifying, tracking, and 3D reconstruction of tendon fascicles from micro-CT images of rat tails. (Repository: INBRE-AI-Tendon.)

The pipeline takes a sequence of micro-CT scans as input and produces (1) a video of the scans with the fascicles annotated in color and (2) a 3D reconstruction of the tendon fascicles. Object detection is performed with a custom-trained Ultralytics YOLO26 model, and segmentation/tracking with Meta's SAM2.

## Repository contents

| File | Description |
|---|---|
| `TrackingTendonFibers.ipynb` | The main pipeline notebook (run in Google Colab): preprocessing, detection, segmentation, tracking, and 3D reconstruction |
| `DesignValidation.ipynb` | Validation notebook: compares pipeline detections and segmentations against manually annotated ground truth (precision, recall, AP, IoU, Dice) |
| `fiberYOLO26Weights.pt` | Trained YOLO26 weights for tendon fascicle detection |
| `scripts/run_pipeline.py` | Headless port of the pipeline notebook with per-stage timing, for running on a local GPU workstation instead of Colab |
| `training/` | Reproducible training recipe: sampling manifest, prelabeling and training scripts |
| `validation/` | Validation suite that produces the reported metrics, plus its results |
| `LICENSE.txt` | MIT License |

## Prerequisites

- The input data should be a **zip file** containing a sequence of micro-CT slices. **TIFF, BMP, PNG, and JPEG** are accepted (8- or 16-bit); scanner preview files (e.g. `*_spr.tif`) and any frame whose size differs from the stack are skipped automatically. The images should be named/ordered numerically, with the first image as the starting point and the last image as the ending point.
- The program runs in **Google Colab** and requires a GPU runtime. The free tier of Google Colab provides a T4 GPU, which is sufficient; large datasets may require more runtime than the free tier permits.

To get started, download [`TrackingTendonFibers.ipynb`](https://github.com/Boyu-Zhang-UOI/INBRE-AI-Tendon/blob/main/TrackingTendonFibers.ipynb) and upload it to your Google Drive to run in Google Colab.

## Usage

1. Open the notebook in Google Colab.
2. Connect to a GPU: click the drop-down menu next to the "Reconnect" button, choose "Change runtime type", and select a GPU.
3. Run each cell individually (which can help isolate potential errors), or use "Run all". If you use "Run all", check the customizable settings first — they are under the "User Customization" cell near the top of the notebook:
   1. `frameSkip`: process 1 out of every `frameSkip` frames to reduce input size. Set `frameSkip = 1` to process every frame, `2` for every other frame, etc.
   2. `fps`: frames per second of the output video; adjust based on how many frames you are processing.
   3. `SCALE_FACTOR`: dimensions of the output video relative to the input. `SCALE_FACTOR = 1` keeps the input size; `2` doubles it.
   4. `scale_unit`: the unit and "stretch" factor of the z-axis in the final reconstruction. Must be `"millimeters"`, `"microns"`, or `"pixels"`.
4. When the "Upload Input" cell runs, click the "Choose File" button below the cell and select the zip file of your input data.
5. Everything beyond this point runs without user interaction as long as all cells are prompted to run.
6. To retrieve the output, click the folder icon in the left toolbar and open the `OUTPUT` folder, which contains:
   - every processed frame,
   - the annotated video at your specified fps,
   - `Point3DReconstruction.ply` — the 3D point cloud of the fascicles,
   - `Mesh3DReconstruction.ply` — a mesh surrounding the fascicles,
   - `points3D.npz` — an archive of all reconstruction points, useful if you want to create your own mesh from the point cloud.

   The `.ply` files can be opened in any application that supports the PLY format (e.g., Blender, MeshLab).

## Validation

The reported metrics come from the `validation/` suite, which is the authoritative one: `validation/gt_register.py` registers the hand-drawn masks back onto the original image frames and `validation/tendon_eval.py` scores detection and segmentation against them on 30 randomly selected slices. Results and a description of the method are in [`validation/README.md`](validation/README.md).

`DesignValidation.ipynb` is the interactive Colab version of the same check, kept for exploration and visual overlays. It aligns ground truth and prediction by cropping rather than registration, so its numbers differ from the reported ones.

## Running without Colab

`scripts/run_pipeline.py` runs the same pipeline headlessly on a local CUDA workstation:

```
python scripts/run_pipeline.py <input_dir_with_slices> <output_root>
```

It loads `fiberYOLO26Weights.pt` from this repository and expects the SAM 2 checkpoint at `checkpoints/sam2_hiera_large.pt`; override either path with the `TENDONTRACK_WEIGHTS` and `SAM2_CHECKPOINT` environment variables.

## Troubleshooting

If you encounter an error when importing libraries, it is most likely a runtime issue. To resolve it, click the drop-down to the right of the "RAM and Disk" button, choose "Change runtime type", select **CPU** as the hardware accelerator, and click "Save". Then return to the "Change runtime type" menu, select your desired GPU again, and click "Save". This effectively resets the runtime; you will need to rerun previously run cells, but the error should not persist.

## Declaration of generative AI use

Portions of the code in this repository were developed with the assistance of generative AI tools: GitHub Copilot (code completion in the original pipeline notebook) and Claude by Anthropic (the headless script in `scripts/`, the `validation/` and `training/` scripts, and the fixes for duplicate seed detections and the point-cloud z step). All AI-assisted code was reviewed, tested, and edited by the authors, who take full responsibility for the content of this repository.

## License

This project is released under the [MIT License](LICENSE.txt).

## Acknowledgements

This work was supported by the INBRE program, NIH Grant P20GM103408 (National Institute of General Medical Sciences).
