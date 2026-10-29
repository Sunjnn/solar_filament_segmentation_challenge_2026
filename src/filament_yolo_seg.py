# %% md
# # Solar Filament Segmentation — YOLO11s-seg Single-Stage
# 
# Single-stage instance segmentation with YOLO11-seg.
# Replaces the two-stage YOLO detect + SeedRefiner U-Net pipeline.
# YOLO11-seg predicts instance masks directly via its built-in segmentation head.
# 
# Key differences from the two-stage approach:
# - No separate refiner — mask quality comes from YOLO's seg head
# - `retina_masks=True` for higher-resolution mask outputs
# - Multi-scale inference (1280 + 640) with mask-level NMS fusion
# - Lower conf threshold (0.08) to catch more dim filaments

# %% md
# ## 1. Setup & Config

# %%
import sys, subprocess, os, json, math, random, shutil
from pathlib import Path
from collections import defaultdict

for pkg, imp in [("ultralytics", "ultralytics"), ("pycocotools", "pycocotools")]:
    try: __import__(imp)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import cv2, numpy as np, pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO
from pycocotools import mask as mask_utils

# %% md
# ## 2. Configuration

# %%
def find_data():
    for p in [
        Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    ]:
        if (p / "train/train_images").exists():
            return p
    raise FileNotFoundError("MAGFiLO not found")

DATA = find_data()
TRAIN_IMG = DATA / "train/train_images"
TEST_IMG = DATA / "test/test_images"
TRAIN_JSON = DATA / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUT = Path("/kaggle/working")

# ---- Model config ----
MODEL = "yolo11s-seg.pt"
IMGSZ = 1280
EPOCHS = 60
BATCH = 4
CONF = 0.08
IOU = 0.65
MIN_AREA = 150

# ---- General ----
VAL_FRAC = 0.15
SEED = 42
WORKERS = 2
DEVICE = 0 if torch.cuda.is_available() else "cpu"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
print(f"Data: {DATA}")

# %% md
# ## 3. Load COCO & Split

# %%
print("Loading COCO...")
with open(TRAIN_JSON) as f:
    coco = json.load(f)

images_by_id = {im["id"]: im for im in coco["images"]}
anns_by_image = defaultdict(list)
for ann in coco["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

samples = []
missing = 0
for image_id, im in images_by_id.items():
    path = TRAIN_IMG / im["file_name"]
    if not path.exists():
        alt = path.with_suffix(".jpg") if path.suffix == ".jpeg" else path.with_suffix(".jpeg")
        path = alt if alt.exists() else path
    if not path.exists():
        missing += 1; continue
    samples.append({
        "image_id": image_id, "file_name": im["file_name"],
        "path": path, "stem": path.stem,
        "height": im.get("height", 2048), "width": im.get("width", 2048),
        "annotations": anns_by_image[image_id],
    })

stems = sorted({s["stem"] for s in samples})
random.shuffle(stems)
n_val = max(1, int(len(stems) * VAL_FRAC))
val_stems, train_stems = set(stems[:n_val]), set(stems[n_val:])
train_s = [s for s in samples if s["stem"] in train_stems]
val_s = [s for s in samples if s["stem"] in val_stems]

print(f"Samples: {len(samples)} (missing {missing}) | train {len(train_stems)} stems, val {len(val_stems)} stems")

# %% md
# ## 4. Convert COCO Polygons → YOLO Seg Labels

# %%
SEG_DATA = OUT / "yolo_seg_data"

def polygon_to_yolo_line(segmentation, img_w, img_h):
    """Convert one COCO polygon to a YOLO seg label line."""
    poly = segmentation[0]
    parts = []
    for k in range(0, len(poly), 2):
        x = max(0.0, min(1.0, poly[k] / img_w))
        y = max(0.0, min(1.0, poly[k+1] / img_h))
        parts.extend([x, y])
    return "0 " + " ".join(f"{v:.6f}" for v in parts)

def build_seg_split(slist, split_name):
    img_dir = SEG_DATA / "images" / split_name
    lbl_dir = SEG_DATA / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    seen, n_inst = set(), 0
    for s in slist:
        fn = s["file_name"]
        if fn in seen: continue
        seen.add(fn)
        shutil.copy2(s["path"], img_dir / Path(fn).name)
        h, w = s["height"], s["width"]
        lines = []
        for ann in s["annotations"]:
            seg = ann.get("segmentation")
            if not seg or not seg[0]: continue
            try:
                lines.append(polygon_to_yolo_line(seg, w, h))
                n_inst += 1
            except Exception:
                continue
        (lbl_dir / f"{s['stem']}.txt").write_text("\n".join(lines) if lines else "")
    return len(seen), n_inst

if SEG_DATA.exists(): shutil.rmtree(SEG_DATA)
nt, bi = build_seg_split(train_s, "train")
nv, bv = build_seg_split(val_s, "val")

yaml_path = SEG_DATA / "data.yaml"
yaml_path.write_text(
    f"path: {SEG_DATA}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
print(f"Seg data: train={nt} imgs/{bi} instances, val={nv}/{bv}")

# %% md
# ## 5. Train YOLO11-seg

# %%
model = YOLO(MODEL)
model.train(
    data=str(yaml_path), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
    workers=WORKERS, seed=SEED, patience=15,
    project=str(OUT / "ultralytics_runs"), name="filament_yolo_seg",
    exist_ok=True, pretrained=True,
    # Disable augmentations that can corrupt polygon labels
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    # Grayscale-friendly augs
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
    degrees=180.0, translate=0.1, scale=0.5,
    shear=0.0, perspective=0.0,
    flipud=0.5, fliplr=0.5,
    device=DEVICE,
)

best_pt = OUT / "yolo_seg_best.pt"
runs = sorted((OUT / "ultralytics_runs" / "filament_yolo_seg" / "weights").glob("best.pt"))
if runs:
    shutil.copy2(runs[0], best_pt)
elif model.trainer.best and Path(model.trainer.best).exists():
    shutil.copy2(model.trainer.best, best_pt)
print(f"Seg weights: {best_pt} ({best_pt.stat().st_size/1e6:.1f}MB)")

# %% md
# ## 6. Inference Pipeline

# %%
def mask_to_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-6)

def mask_nms(scored_masks, iou_thr=0.65):
    if len(scored_masks) <= 1: return scored_masks
    scored = sorted(scored_masks, key=lambda x: -x[0])
    keep = []
    for i, (conf_i, mask_i) in enumerate(scored):
        suppressed = False
        for j in keep:
            if mask_iou(mask_i, scored[j][1]) > iou_thr:
                suppressed = True; break
        if not suppressed: keep.append(i)
    return [scored[i] for i in keep]

def resolve_overlapping(scored):
    if not scored: return []
    scored = sorted(scored, key=lambda t: -t[0])
    occupied = np.zeros_like(scored[0][1], dtype=bool)
    out = []
    for score, m in scored:
        m = m.copy(); m[occupied] = 0
        if m.sum() < MIN_AREA: continue
        occupied |= m.astype(bool)
        out.append(m.astype(np.uint8))
    return out

def extract_instances(result, h, w):
    """Extract binary masks at original resolution from one YOLO-seg result."""
    if result.masks is None: return []
    masks_data = result.masks.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    instances = []
    for i, c in enumerate(confs):
        m = cv2.resize(masks_data[i].astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        binary = (m >= 0.5).astype(np.uint8)
        if binary.sum() >= MIN_AREA:
            instances.append((float(c), binary))
    return instances

def predict_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return []
    h, w = img.shape
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    all_instances = []
    for sz in [1280, 640]:
        res = model.predict(source=bgr, conf=CONF, iou=IOU, imgsz=sz,
                            verbose=False, device=DEVICE, retina_masks=True)[0]
        all_instances.extend(extract_instances(res, h, w))
    if not all_instances: return []
    return resolve_overlapping(mask_nms(all_instances, iou_thr=0.65))

# Quick test
demo = val_s[0]
demo_masks = predict_image(demo["path"])
print(f"Demo {demo['stem']}: {len(demo_masks)} instances")

# %% md
# ## 7. Generate Submission

# %%
test_paths = sorted(TEST_IMG.glob("*.jpeg")) + sorted(TEST_IMG.glob("*.jpg"))
seen, unique = set(), []
for p in test_paths:
    if p.name not in seen: seen.add(p.name); unique.append(p)

sub_rows = []
for path in tqdm(unique, desc="test inference"):
    masks = predict_image(path)
    for i, m in enumerate(masks, start=1):
        sub_rows.append({
            "filament_id": f"{path.stem}_{i}",
            "segmentation_rle": mask_to_rle(m)
        })

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv("submission.csv", index=False)
print(f"Submission: {len(sub_df)} rows / {sub_df['filament_id'].str.rsplit('_',n=1).str[0].nunique()} images")
sub_df.head()