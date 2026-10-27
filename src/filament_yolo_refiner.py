# %% md
# # Solar Filament Segmentation — YOLO11s + Seed-Refiner U-Net
#
# Two-stage pipeline inspired by Tim Gluhkikh's approach (33 votes, LB ~0.409):
# 1. **YOLO11s** detects filament bounding boxes
# 2. **Seed-Refiner U-Net** refines masks within each box
#    - 3-channel input: [image_crop, CLAHE(image), seed_box]
#    - Seed = filled padded box (jittered during train, from YOLO during infer)
#
# Key insight: the "seed" tells the refiner WHERE to look, making it focused.

# %% md
# ## 1. Setup & Config

# %%
import sys, subprocess, os, json, math, random, shutil
from pathlib import Path
from collections import defaultdict

# Install missing packages
for pkg, imp in [("segmentation-models-pytorch", "segmentation_models_pytorch"), ("ultralytics", "ultralytics"), ("pycocotools", "pycocotools")]:
    try: __import__(imp)
    except ImportError: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import cv2, numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from pycocotools import mask as mask_utils
from tqdm.auto import tqdm
from ultralytics import YOLO

# %% md
# ## 2. Configuration

# %%
# ---- Data paths ----
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

# ---- YOLO config ----
YOLO_MODEL = "yolo11s.pt"
YOLO_IMGSZ = 1280
YOLO_EPOCHS = 40
YOLO_BATCH = 6
YOLO_CONF = 0.28
YOLO_IOU = 0.50

# ---- Refiner config ----
REFINE_CROP = 512
REFINE_EPOCHS = 15
REFINE_BATCH = 6
REFINE_LR = 3e-4
REFINE_THR = 0.50
PAD_RATIO = 0.40
CONTEXT = 1.5
MIN_CROP = 96
MIN_AREA = 200

# ---- General ----
VAL_FRAC = 0.15
SEED = 42
WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TTA knobs
USE_BOX_JITTER_TTA = True
BOX_JITTER_PX = 8

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

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

# Build sample list
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

# Split by stem to avoid data leak
stems = sorted({s["stem"] for s in samples})
random.shuffle(stems)
n_val = max(1, int(len(stems) * VAL_FRAC))
val_stems, train_stems = set(stems[:n_val]), set(stems[n_val:])
train_s = [s for s in samples if s["stem"] in train_stems]
val_s = [s for s in samples if s["stem"] in val_stems]

# Per-annotation list for refiner training
def build_ann_list(slist):
    rows = []
    for s in slist:
        for ann in s["annotations"]:
            if ann.get("iscrowd", 0): continue
            if not ann.get("segmentation"): continue
            rows.append({"sample": s, "ann": ann, "weight": 1.0})
    return rows

train_anns = build_ann_list(train_s)
val_anns = build_ann_list(val_s)

print(f"Samples: {len(samples)} (missing {missing}) | train stems {len(train_stems)} val {len(val_stems)}")
print(f"Refiner crops: train {len(train_anns)} val {len(val_anns)}")

# %% md
# ## 4. YOLO Detection Training

# %%
YOLO_DATA = OUT / "yolo_data"

def build_yolo_split(slist, split_name):
    img_d = YOLO_DATA / "images" / split_name
    lbl_d = YOLO_DATA / "labels" / split_name
    img_d.mkdir(parents=True, exist_ok=True); lbl_d.mkdir(parents=True, exist_ok=True)
    seen, n_boxes = set(), 0
    for s in slist:
        fn = s["file_name"]
        if fn in seen: continue
        seen.add(fn)
        shutil.copy2(s["path"], img_d / Path(fn).name)
        w, h = s["width"], s["height"]
        lines = []
        for ann in s["annotations"]:
            x, y, bw, bh = ann["bbox"]
            lines.append(f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} {bw/w:.6f} {bh/h:.6f}")
            n_boxes += 1
        (lbl_d / f"{s['stem']}.txt").write_text("\n".join(lines))
    return len(seen), n_boxes

if YOLO_DATA.exists(): shutil.rmtree(YOLO_DATA)
nt, bt = build_yolo_split(train_s, "train")
nv, bv = build_yolo_split(val_s, "val")
yaml_p = YOLO_DATA / "data.yaml"
yaml_p.write_text(f"path: {YOLO_DATA}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
print(f"YOLO data: train={nt}/{bt}, val={nv}/{bv}")

yolo = YOLO(YOLO_MODEL)
yolo.train(
    data=str(yaml_p), epochs=YOLO_EPOCHS, imgsz=YOLO_IMGSZ, batch=YOLO_BATCH,
    workers=WORKERS, seed=SEED, project=str(OUT/"ultralytics_runs"),
    name="filament_yolo", exist_ok=True, pretrained=True,
    patience=max(5, YOLO_EPOCHS//4),
    device=0 if DEVICE.type == "cuda" else "cpu",
)

# Find best weights
best_pt = OUT / "yolo_best.pt"
runs = sorted((OUT/"ultralytics_runs"/"filament_yolo"/"weights").glob("best.pt"))
if runs:
    shutil.copy2(runs[0], best_pt)
elif yolo.trainer.best and Path(yolo.trainer.best).exists():
    shutil.copy2(yolo.trainer.best, best_pt)
print(f"YOLO weights: {best_pt} ({best_pt.stat().st_size/1e6:.1f}MB)")

# %% md
# ## 5. Seed-Refiner Dataset

# %%
def ann_to_mask(ann, h, w):
    seg = ann.get("segmentation")
    if seg:
        rles = mask_utils.frPyObjects(seg, h, w)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
        m = mask_utils.decode(rle)
        return m.astype(np.uint8) if m.ndim == 2 else np.any(m, axis=2).astype(np.uint8)
    x, y, bw, bh = map(int, ann["bbox"])
    m = np.zeros((h, w), np.uint8); m[y:y+bh, x:x+bw] = 1
    return m

def square_bounds(bbox, h, w, context=CONTEXT, minimum=MIN_CROP):
    x, y, bw, bh = bbox
    cx, cy = x+bw/2, y+bh/2
    side = int(math.ceil(max(bw, bh) * context))
    side = max(side, minimum)
    x0, y0 = int(round(cx-side/2)), int(round(cy-side/2))
    x1, y1 = x0+side, y0+side
    if x0 < 0: x1 -= x0; x0 = 0
    if y0 < 0: y1 -= y0; y0 = 0
    if x1 > w: x0 -= x1-w; x1 = w
    if y1 > h: y0 -= y1-h; y1 = h
    return max(0,x0), max(0,y0), x1, y1

def seed_from_box(h, w, bbox, pad_ratio=PAD_RATIO):
    x, y, bw, bh = bbox
    px, py = bw*pad_ratio, bh*pad_ratio
    x0 = max(0, int(x-px)); y0 = max(0, int(y-py))
    x1 = min(w, int(x+bw+px)); y1 = min(h, int(y+bh+py))
    seed = np.zeros((h,w), np.uint8); seed[y0:y1, x0:x1] = 1
    return seed

def jitter_bbox(bbox, h, w, rng):
    x, y, bw, bh = map(float, bbox)
    x += rng.uniform(-0.15, 0.15) * bw
    y += rng.uniform(-0.15, 0.15) * bh
    sx, sy = rng.uniform(0.85, 1.25), rng.uniform(0.85, 1.25)
    cx, cy = x+bw/2, y+bh/2
    bw, bh = max(8.0, bw*sx), max(8.0, bh*sy)
    x, y = cx-bw/2, cy-bh/2
    x, y = np.clip(x, 0, w-2), np.clip(y, 0, h-2)
    bw, bh = min(bw, w-x), min(bh, h-y)
    pad = rng.uniform(PAD_RATIO*0.5, PAD_RATIO*1.5)
    pad = max(0.1, min(pad, 0.8))
    return [x, y, bw, bh], pad

class SeedRefineDataset(Dataset):
    def __init__(self, ann_list, crop_size, augment=True):
        self.ann_list = ann_list; self.crop_size = crop_size; self.augment = augment
        self._cache = {}
    def _load(self, path):
        k = str(path)
        if k not in self._cache:
            self._cache[k] = cv2.imread(k, cv2.IMREAD_GRAYSCALE)
        return self._cache[k]
    def __len__(self): return len(self.ann_list)
    def __getitem__(self, idx):
        item = self.ann_list[idx]
        s, ann = item["sample"], item["ann"]
        img = self._load(s["path"]); h, w = img.shape
        target = ann_to_mask(ann, h, w)
        gt_bbox = ann["bbox"]
        if self.augment:
            rng = np.random.default_rng(SEED + idx + random.randint(0, 10000))
            box_j, pad_j = jitter_bbox(gt_bbox, h, w, rng)
            seed_full = seed_from_box(h, w, box_j, pad_ratio=pad_j)
        else:
            seed_full = seed_from_box(h, w, gt_bbox)
        x0, y0, x1, y1 = square_bounds(gt_bbox, h, w)
        img_c = cv2.resize(img[y0:y1, x0:x1], (self.crop_size,)*2, interpolation=cv2.INTER_LINEAR)
        tgt_c = cv2.resize(target[y0:y1, x0:x1], (self.crop_size,)*2, interpolation=cv2.INTER_NEAREST)
        # Edge softening: Gaussian blur on binary mask (sigma 0.5-1.5)
        if self.augment:
            blur_sigma = 0.5 + random.random() * 1.0
            ksize = max(3, int(blur_sigma * 4 + 1) | 1)  # odd kernel
            tgt_c = cv2.GaussianBlur(tgt_c.astype(np.float32), (ksize, ksize), blur_sigma)
            tgt_c = np.clip(tgt_c, 0, 1)  # keep in [0,1]
        seed_c = cv2.resize(seed_full[y0:y1, x0:x1], (self.crop_size,)*2, interpolation=cv2.INTER_NEAREST)
        clahe = cv2.createCLAHE(2.0, (8,8)).apply(img_c)
        if self.augment and random.random() < 0.5:
            img_c, clahe = np.fliplr(img_c).copy(), np.fliplr(clahe).copy()
            tgt_c, seed_c = np.fliplr(tgt_c).copy(), np.fliplr(seed_c).copy()
        if self.augment and random.random() < 0.5:
            img_c, clahe = np.flipud(img_c).copy(), np.flipud(clahe).copy()
            tgt_c, seed_c = np.flipud(tgt_c).copy(), np.flipud(seed_c).copy()

        
        x = np.stack([img_c, clahe, seed_c*255], axis=0).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        x = (x - mean) / std
        return torch.from_numpy(x), torch.from_numpy(tgt_c[None].astype(np.float32))

# %% md
# ## 6. Train Seed-Refiner U-Net

# %%
def soft_dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(2,3))
    den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    return 1.0 - ((2*inter + eps) / (den + eps)).mean()

@torch.no_grad()
def batch_dice(logits, targets, thr=0.5):
    preds = (torch.sigmoid(logits) > thr).float()
    inter = (preds * targets).sum(dim=(2,3))
    den = preds.sum(dim=(2,3)) + targets.sum(dim=(2,3))
    return ((2*inter + 1e-6) / (den + 1e-6)).mean().item()

train_ds = SeedRefineDataset(train_anns, REFINE_CROP, True)
val_ds = SeedRefineDataset(val_anns, REFINE_CROP, False)
train_loader = DataLoader(train_ds, REFINE_BATCH, shuffle=True, num_workers=WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, REFINE_BATCH*2, shuffle=False, num_workers=WORKERS, pin_memory=True)

xb, yb, sb = next(iter(train_loader))
print(f"Refiner batch: {xb.shape} {yb.shape}, mask_mean={float(yb.mean()):.4f}, ")

refiner = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
opt = torch.optim.AdamW(refiner.parameters(), lr=REFINE_LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=REFINE_EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type=="cuda")

best_dice = -1.0
refiner_ckpt = OUT / "seed_refiner_best.pt"

for epoch in range(1, REFINE_EPOCHS+1):
    refiner.train()
    tr_losses, tr_dices = [], []
    for batch in tqdm(train_loader, leave=False, desc=f"refine train {epoch}"):
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=DEVICE.type=="cuda"):
            logits = refiner(x)
            loss = 0.5*F.binary_cross_entropy_with_logits(logits, y) + 0.5*soft_dice_loss(logits, y)
        tr_losses.append(loss.item()); tr_dices.append(batch_dice(logits.detach(), y))
    refiner.eval()
    va_losses, va_dices = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, leave=False, desc=f"refine val {epoch}"):
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = refiner(x)
            loss = 0.5*F.binary_cross_entropy_with_logits(logits, y) + 0.5*soft_dice_loss(logits, y)
            va_losses.append(loss.item()); va_dices.append(batch_dice(logits, y))
    sched.step()
    tr_l, tr_d = np.mean(tr_losses), np.mean(tr_dices)
    va_l, va_d = np.mean(va_losses), np.mean(va_dices)
    print(f"E{epoch:02d}/{REFINE_EPOCHS}  train loss={tr_l:.4f} dice={tr_d:.4f} | val loss={va_l:.4f} dice={va_d:.4f}")
    if va_d > best_dice:
        best_dice = va_d
        torch.save({"model": refiner.state_dict(), "epoch": epoch, "val_dice": va_d}, refiner_ckpt)
        print(f"  saved {refiner_ckpt} val_dice={va_d:.4f}")

print(f"Best refiner val dice: {best_dice:.4f}")
state = torch.load(refiner_ckpt, map_location=DEVICE, weights_only=False)
refiner.load_state_dict(state["model"])
refiner.eval()

# %% md
# ## 7. Inference Pipeline

# %%
def mask_to_rle(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")

def resolve_overlapping(scored):
    if not scored: return []
    scored = sorted(scored, key=lambda t: -t[0])
    occupied = np.zeros_like(scored[0][1], dtype=bool)
    out = []
    for score, m in scored:
        m = m.copy(); m[occupied] = 0
        if m.sum() == 0: continue
        occupied |= m.astype(bool)
        out.append(m.astype(np.uint8))
    return out

@torch.no_grad()
def refine_box(gray, bbox_xyxy, conf):
    h, w = gray.shape
    x1, y1, x2, y2 = bbox_xyxy
    bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
    bbox = [x1, y1, bw, bh]
    x0, y0, x1b, y1b = square_bounds(bbox, h, w)
    img_c = cv2.resize(gray[y0:y1b, x0:x1b], (REFINE_CROP,)*2, interpolation=cv2.INTER_LINEAR)
    clahe = cv2.createCLAHE(2.0, (8,8)).apply(img_c)
    mean = np.array([0.485,0.456,0.406], np.float32)[:,None,None]
    std  = np.array([0.229,0.224,0.225], np.float32)[:,None,None]

    shifts = [(0,0,False,False), (BOX_JITTER_PX,0,False,False), (-BOX_JITTER_PX,0,False,False), (0,BOX_JITTER_PX,False,False), (0,-BOX_JITTER_PX,False,False), (0,0,True,False), (0,0,False,True)] if USE_BOX_JITTER_TTA else [(0,0,False,False)]
    batch = []
    for tta_param in shifts:
        dx, dy, hflip, vflip = tta_param
        bbox_j = [x1+dx, y1+dy, bw, bh]
        sfull = seed_from_box(h, w, bbox_j)
        sc = cv2.resize(sfull[y0:y1b, x0:x1b], (REFINE_CROP,)*2, interpolation=cv2.INTER_NEAREST)
        ic = img_c.copy(); cc = clahe.copy()
        if hflip: ic, cc = np.fliplr(ic).copy(), np.fliplr(cc).copy(); sc = np.fliplr(sc).copy()
        if vflip: ic, cc = np.flipud(ic).copy(), np.flipud(cc).copy(); sc = np.flipud(sc).copy()
        x = np.stack([ic, cc, sc*255], 0).astype(np.float32)/255.0
        batch.append((x - mean) / std)
    x = torch.from_numpy(np.stack(batch)).to(DEVICE)
    logits = refiner(x)
    # Unflip predictions
    for i, (dx, dy, hflip, vflip) in enumerate(shifts):
        if vflip: logits[i] = logits[i].flip(-2)
        if hflip: logits[i] = logits[i].flip(-1)
    prob = torch.sigmoid(logits).mean(dim=0)[0].cpu().numpy()
    crop_m = (prob >= REFINE_THR).astype(np.uint8)
    full = np.zeros((h,w), np.uint8)
    full[y0:y1b, x0:x1b] = cv2.resize(crop_m, (x1b-x0, y1b-y0), interpolation=cv2.INTER_NEAREST)
    return full, float(conf)

def predict_image(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return []
    if img.shape != (2048, 2048):
        img = cv2.resize(img, (2048, 2048), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    res = yolo.predict(source=bgr, conf=YOLO_CONF, iou=YOLO_IOU, imgsz=YOLO_IMGSZ,
                       verbose=False, device=0 if DEVICE.type=="cuda" else "cpu")[0]
    scored = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for i in np.argsort(-confs):
            m, sc = refine_box(img, xyxy[i], float(confs[i]))
            if m.sum() >= MIN_AREA:
                scored.append((sc, m))
    return resolve_overlapping(scored)

# Quick test on one val image
demo = val_s[0]
demo_masks = predict_image(demo["path"])
print(f"Demo {demo['stem']}: {len(demo_masks)} instances detected")

# %% md
# ## 8. Generate Submission

# %%
test_paths = sorted(TEST_IMG.glob("*.jpeg")) + sorted(TEST_IMG.glob("*.jpg"))
seen, unique = set(), []
for p in test_paths:
    if p.name not in seen: seen.add(p.name); unique.append(p)

sub_rows = []
for path in tqdm(unique, desc="test inference"):
    masks = predict_image(path)
    for i, m in enumerate(masks, start=1):
        sub_rows.append({"filament_id": f"{path.stem}_{i}", "segmentation_rle": mask_to_rle(m)})

sub_df = pd.DataFrame(sub_rows)
sub_df.to_csv("submission.csv", index=False)
print(f"Submission: {len(sub_df)} rows / {sub_df['filament_id'].str.rsplit('_',n=1).str[0].nunique()} images")
sub_df.head()
