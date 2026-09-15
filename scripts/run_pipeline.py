"""Headless port of TrackingTendonFibers.ipynb with per-stage timing.

Faithful to the notebook's processing logic (conversion, sharpening, video,
YOLO seed detection, SAM2 video propagation, point cloud, Poisson mesh);
only Colab-specific I/O and interactive plotting are removed, plus
offload_video_to_cpu=True so 729 full-res frames fit in VRAM.

Usage: python run_pipeline.py <input_dir_with_slices> <output_root>

Model paths default to this repository (fiberYOLO26Weights.pt) and to
checkpoints/sam2_hiera_large.pt; override with TENDONTRACK_WEIGHTS and
SAM2_CHECKPOINT.
"""
import os, sys, json, time
import cv2
import numpy as np
import torch
import supervision as sv
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor
import plotly.express as px
import open3d as o3d

INPUT = sys.argv[1]
ROOT = sys.argv[2]
INTERMEDIATE = f"{ROOT}/INTERMEDIATE"
OUTPUT = f"{ROOT}/OUTPUT"
os.makedirs(INTERMEDIATE, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
INTERMEDIATE_VIDEO = OUTPUT + "/Intermediate_Video.mp4"
OUTPUT_VIDEO = OUTPUT + "/Output_Video.mp4"

frameSkip = 1
fps = 8
SCALE_FACTOR = 1
# Physical calibration: set to your scan's voxel size (um/pixel, um/slice).
# The P21 high-resolution scans of Ditton et al. (2024) use 9 um isotropic voxels.
MICRONS_PER_PIXEL = 9.0
MICRONS_PER_ZSTEP = 9.0

# Model locations. Defaults resolve next to this repository so the script runs
# straight after a clone; override either with an environment variable.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_WEIGHTS = os.environ.get("TENDONTRACK_WEIGHTS", os.path.join(REPO, "fiberYOLO26Weights.pt"))
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", os.path.join(REPO, "checkpoints", "sam2_hiera_large.pt"))
T = {}

SEED_DEDUP_IOU = 0.7  # YOLO26 is NMS-free: suppress near-duplicate boxes on the same fascicle


def dedup_boxes(xyxy, conf, iou_thr=SEED_DEDUP_IOU):
    """Greedy suppression: of any two boxes with IoU >= iou_thr keep the higher-confidence one.

    Returns the indices to keep, in descending confidence order.
    """
    xyxy = np.asarray(xyxy, dtype=float)
    keep = []
    for i in np.argsort(-np.asarray(conf)):
        a = xyxy[i]
        duplicate = False
        for j in keep:
            b = xyxy[j]
            inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            if union > 0 and inter / union >= iou_thr:
                duplicate = True
                break
        if not duplicate:
            keep.append(int(i))
    return keep


def tick(stage):
    T[stage] = time.perf_counter()


def tock(stage):
    T[stage] = time.perf_counter() - T[stage]
    print(f"[timing] {stage}: {T[stage]:.1f}s", flush=True)


# ---------------- Stage 1: TIFF -> JPEG ----------------
tick("1_convert")
inputFiles = sorted(f for f in os.listdir(INPUT)
                    if f.lower().endswith((".tif", ".tiff", ".bmp", ".png", ".jpg", ".jpeg")))
# skip scanner previews and odd-sized frames (they corrupt SAM2's video dims)
ref = None
clean = []
for f in inputFiles:
    if "_spr" in f.lower():
        print("skipping preview file:", f); continue
    try:
        sz = Image.open(os.path.join(INPUT, f)).size
    except Exception:
        print("skipping unreadable:", f); continue
    if ref is None:
        ref = sz
    if sz != ref:
        print(f"skipping {f}: size {sz} != stack {ref}"); continue
    clean.append(f)
inputFiles = clean
for i, filename in enumerate(inputFiles):
    if i % frameSkip != 0:
        continue
    image = Image.open(os.path.join(INPUT, filename))
    if image.mode in ["I;16", "I"]:
        imageMod = (np.array(image, dtype=np.uint16) / 256).astype(np.uint8)
        image = Image.fromarray(imageMod)
    image.convert("RGB").save(os.path.join(INTERMEDIATE, f"{i:05}.jpeg"), "JPEG", quality=95)
tock("1_convert")

# ---------------- Stage 2: sharpen ----------------
tick("2_sharpen")
kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
for filename in os.listdir(INTERMEDIATE):
    p = os.path.join(INTERMEDIATE, filename)
    img = cv2.imread(p)
    if img is None:
        continue
    cv2.imwrite(p, cv2.filter2D(img, -1, kernel_sharpening))
tock("2_sharpen")

# ---------------- Stage 3: intermediate video ----------------
tick("3_video")
files = sorted(f for f in os.listdir(INTERMEDIATE) if f.endswith(".jpeg"))
frame0 = cv2.imread(os.path.join(INTERMEDIATE, files[0]))
h, w = frame0.shape[:2]
vw = cv2.VideoWriter(INTERMEDIATE_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
for f in files:
    vw.write(cv2.imread(os.path.join(INTERMEDIATE, f)))
vw.release()
tock("3_video")
print(f"frames: {len(files)}  size: {w}x{h}", flush=True)

# ---------------- Model setup ----------------
tick("4_model_setup")
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
if not os.path.exists(YOLO_WEIGHTS):
    raise SystemExit(f"detector weights not found: {YOLO_WEIGHTS}\n"
                     "Set TENDONTRACK_WEIGHTS to the path of fiberYOLO26Weights.pt.")
if not os.path.exists(SAM2_CHECKPOINT):
    raise SystemExit(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}\n"
                     "Download sam2_hiera_large.pt from the SAM 2 repository and set SAM2_CHECKPOINT.")
yolo_model = YOLO(YOLO_WEIGHTS)
sam2_model = None
for _cfg in ("configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_l.yaml"):
    try:
        sam2_model = build_sam2_video_predictor(_cfg, SAM2_CHECKPOINT)
        break
    except Exception as _e:
        _last = _e
if sam2_model is None:
    raise SystemExit(f"SAM2 config not found: {_last}")
tock("4_model_setup")

# ---------------- Stage 5: init + seed detection ----------------
tick("5_detect_seed")
inference_state = sam2_model.init_state(video_path=INTERMEDIATE, offload_video_to_cpu=True)
sam2_model.reset_state(inference_state)
INTERMEDIATE_PATHS = sorted(
    sv.list_files_with_extensions(INTERMEDIATE, extensions=["jpg", "jpeg"]))
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        FRAME_IDX = 0
        results = yolo_model(str(Path(INTERMEDIATE) / f"{FRAME_IDX:05d}.jpeg"))
        det = results[0].boxes
        xyxy_all = det.xyxy.cpu().numpy()
        conf_all = det.conf.cpu().numpy()
        keep = dedup_boxes(xyxy_all, conf_all)
        n_dup = len(xyxy_all) - len(keep)
        n_seed = 0
        for obj_id, k in enumerate(keep):
            sam2_model.add_new_points_or_box(
                inference_state=inference_state, frame_idx=FRAME_IDX, obj_id=obj_id,
                box=xyxy_all[k].tolist(), clear_old_points=True, normalize_coords=True)
            n_seed += 1
print(f"seeded {n_seed} fascicles ({n_dup} duplicate detections suppressed)", flush=True)
if n_seed == 0:
    raise SystemExit(
        "No fascicles were detected in the first slice, so there is nothing to track.\n"
        "Tracking is seeded from slice 0 only: start the stack at a slice where fascicles\n"
        "are visible, or re-train the detector for this scan (see training/README.md).")
tock("5_detect_seed")

# ---------------- Stage 6: propagate + annotate ----------------
tick("6_propagate")
video_info = sv.VideoInfo.from_video_path(INTERMEDIATE_VIDEO)
video_info.width = int(video_info.width * SCALE_FACTOR)
video_info.height = int(video_info.height * SCALE_FACTOR)
mask_annotator = sv.MaskAnnotator(
    color=sv.ColorPalette.from_hex(["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]),
    color_lookup=sv.ColorLookup.CLASS)
mask_list, targets = [], {}
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with sv.VideoSink(OUTPUT_VIDEO, video_info=video_info) as sink:
            for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(inference_state):
                frame = cv2.imread(str(INTERMEDIATE_PATHS[frame_idx]))
                th, tw = video_info.height, video_info.width
                if frame.shape[0] != th or frame.shape[1] != tw:
                    frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
                raw = np.squeeze((mask_logits > 0.0).cpu().numpy())
                if raw.ndim == 2:
                    resized = [cv2.resize(raw.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST)]
                    combined = resized[0]
                else:
                    resized = [cv2.resize(m.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST) for m in raw]
                    combined = np.any(np.array(resized), axis=0)
                masks_det = np.array(resized).astype(bool)
                if masks_det.ndim == 2:
                    masks_det = masks_det[np.newaxis]
                mask_list.append(combined.astype(np.uint8) * 255)
                detections = sv.Detections(
                    xyxy=sv.mask_to_xyxy(masks=masks_det), mask=masks_det,
                    class_id=np.array(object_ids))
                for i, obj_id in enumerate(object_ids):
                    targets.setdefault(obj_id, []).append(detections.xyxy[i])
                annotated = mask_annotator.annotate(scene=frame.copy(), detections=detections)
                cv2.imwrite(f"{OUTPUT}/Annotated{frame_idx:05d}.jpeg", annotated)
                sink.write_frame(annotated)
targets = {k: np.array(v) for k, v in targets.items()}
reconstructionPath = OUTPUT + "/points3D.npz"
np.savez_compressed(reconstructionPath, np.array(mask_list))
tock("6_propagate")

# ---------------- Stage 7: point cloud ----------------
tick("7_pointcloud")


def addPoints(mask, points_list, depth, blob_id):
    y_indices, x_indices = np.where(mask == 255)
    for x, y in zip(x_indices, y_indices):
        points_list.append([x, y, depth, blob_id])


masks = np.load(reconstructionPath)["arr_0"]
z_step_slices = frameSkip   # each processed frame is frameSkip slices deep; scaled to um by MICRONS_PER_ZSTEP below
depth = 0
points = []
prev_uint8, curr_uint8, after_uint8 = masks[0], masks[1], masks[2]
for index in range(1, len(masks) - 1):
    for obj_idx, (obj_id, bbox_list) in enumerate(targets.items()):
        if index >= len(bbox_list):
            continue
        x1, y1, x2, y2 = bbox_list[index].astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, curr_uint8.shape[1]), min(y2, curr_uint8.shape[0])
        blob_mask = np.zeros_like(curr_uint8)
        blob_mask[y1:y2, x1:x2] = curr_uint8[y1:y2, x1:x2]
        prev_mask = np.zeros_like(curr_uint8)
        prev_mask[(prev_uint8 == 0) & (blob_mask == 255)] = 255
        addPoints(prev_mask, points, depth, obj_idx)
        next_mask = np.zeros_like(curr_uint8)
        next_mask[(after_uint8 == 0) & (blob_mask == 255)] = 255
        addPoints(next_mask, points, depth, obj_idx)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for con in contours:
            for point in con:
                p = point[0]
                points.append([p[0], p[1], depth, obj_idx])
    depth += z_step_slices
    prev_uint8, curr_uint8 = curr_uint8, after_uint8
    after_uint8 = masks[index + 1]

points = np.array(points, dtype=np.float32)
points[:, 0] *= MICRONS_PER_PIXEL
points[:, 1] *= MICRONS_PER_PIXEL
points[:, 2] *= MICRONS_PER_ZSTEP
points = np.unique(points, axis=0)
with open(OUTPUT + "/Point3DReconstruction.ply", "w") as file:
    file.write(f"ply\nformat ascii 1.0\nelement vertex {points.shape[0]}\n"
               "property float32 x\nproperty float32 y\nproperty float32 z\n"
               "property int32 blob_id\nend_header\n")
    for p in points:
        file.write(f"{p[0]} {p[1]} {p[2]} {int(p[3])}\n")
print(f"point cloud: {points.shape[0]} pts, {int(points[:,3].max())+1} fibers", flush=True)
tock("7_pointcloud")

# ---------------- Stage 8: Poisson mesh ----------------
tick("8_mesh")
_palette = px.colors.qualitative.Plotly


def blob_color(idx):
    hx = _palette[idx % len(_palette)].lstrip("#")
    return [int(hx[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


all_v, all_t, all_c, off = [], [], [], 0
for blob_id in np.unique(points[:, 3]).astype(int):
    bp = points[points[:, 3] == blob_id][:, :3]
    if len(bp) < 4:
        continue
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bp)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=MICRONS_PER_ZSTEP * 3, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        dens = np.asarray(dens)
        mesh.remove_vertices_by_mask(dens < np.percentile(dens, 10))
    except Exception as e:
        print(f"blob {blob_id} skipped: {type(e).__name__}"); continue
    if len(mesh.vertices) == 0:
        continue
    v = np.asarray(mesh.vertices)
    all_v.append(v)
    all_t.append(np.asarray(mesh.triangles) + off)
    all_c.append(np.tile(blob_color(blob_id), (len(v), 1)))
    off += len(v)

all_v, all_t, all_c = np.vstack(all_v), np.vstack(all_t), np.vstack(all_c)
with open(OUTPUT + "/Mesh3DReconstruction.ply", "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(all_v)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(all_t)}\nproperty list uchar int vertex_indices\nend_header\n")
    for v, c in zip(all_v, all_c):
        f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0]*255)} {int(c[1]*255)} {int(c[2]*255)}\n")
    for t in all_t:
        f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
print(f"mesh: {len(all_v)} vertices, {len(all_t)} faces", flush=True)
tock("8_mesh")

T["total"] = sum(v for k, v in T.items() if k != "total")
T["n_frames"] = len(files)
T["n_fascicles_seeded"] = n_seed
T["n_duplicates_suppressed"] = n_dup
T["gpu"] = torch.cuda.get_device_name(0)
json.dump(T, open(f"{ROOT}/timing.json", "w"), indent=1)
print("\n=== TIMING SUMMARY ===")
for k in sorted(T):
    v = T[k]
    print(f"  {k}: {v:.1f}s" if isinstance(v, float) else f"  {k}: {v}")
