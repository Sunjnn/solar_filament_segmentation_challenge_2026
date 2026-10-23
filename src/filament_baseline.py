# %% md
# # Solar Filament Segmentation — U-Net Baseline
#
# Competition: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
#
# This notebook implements a U-Net baseline following the approach of
# Pavlo Ivanin's high-voted baseline (52 votes).
#
# ## Strategy
# - Parse COCO-format JSON annotations
# - Convert polygon annotations to binary masks via pycocotools
# - Train U-Net (ResNet34 encoder) with BCE + Dice loss
# - Instance separation via connectedComponentsWithStats
# - Submit COCO RLE format predictions

# %% md
# ## 1. Setup & Imports

# %%
import json
import os
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

try:
    import segmentation_models_pytorch as smp
except ImportError:
    os.system("pip install -q segmentation-models-pytorch")
    import segmentation_models_pytorch as smp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as mask_utils

# %% md
# ## 2. Configuration

# %%
class CFG:
    # Paths — auto-detect between competition data and attached dataset
    def _find_base():
        for p in [
            Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
            Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
        ]:
            if (p / "train/train_images").exists():
                return p
        raise FileNotFoundError("Cannot find MAGFiLO dataset")

    BASE = _find_base()
    TRAIN_IMAGES = BASE / "train/train_images"
    TEST_IMAGES = BASE / "test/test_images"
    ANNOTATIONS = BASE / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    # Model
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 3  # RGB (grayscale → 3 identical channels)
    NUM_CLASSES = 1   # binary: filament vs background

    # Image
    IMG_SIZE = 768    # larger for better detail

    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 35   # matching LB 1st config
    LEARNING_RATE = 3e-3   # matching LB 1st config
    NUM_WORKERS = 2
    SEED = 42
    VAL_SPLIT = 0.2

    # Inference
    THRESHOLD = 0.5
    MIN_AREA = 100   # stricter to reduce FP

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed(CFG.SEED)
print(f"Device: {CFG.DEVICE}")

# %% md
# ## 3. Load & Parse COCO Annotations

# %%
# Load COCO-format JSON
with open(CFG.ANNOTATIONS, "r") as f:
    coco_data = json.load(f)

print(f"Images: {len(coco_data['images'])}")
print(f"Annotations: {len(coco_data['annotations'])}")
print(f"Categories: {coco_data.get('categories', 'N/A')}")

# Build lookups
images_by_id = {img["id"]: img for img in coco_data["images"]}
anns_by_image = defaultdict(list)
for ann in coco_data["annotations"]:
    anns_by_image[ann["image_id"]].append(ann)

# %% md
# ## 4. Mask Generation from COCO Annotations

# %%
def coco_ann_to_mask(annotations, height, width):
    """Convert COCO annotations (polygons) to a single binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        seg = ann.get("segmentation")
        if not seg:
            continue
        # pycocotools handles polygon → RLE → mask
        try:
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
            m = mask_utils.decode(rle)
            mask = np.maximum(mask, m)
        except Exception:
            # Fallback: use bbox
            x, y, w, h = map(int, ann["bbox"])
            mask[y:y+h, x:x+w] = 1
    return mask.astype(np.uint8)


# %% md
# ## 5. Dataset & DataLoader

# %%
class FilamentDataset(Dataset):
    def __init__(self, samples, annotations_by_image, img_dir, transform=None, is_test=False):
        """
        samples: list of {"image_id", "file_name", "height", "width", ...}
        """
        self.samples = samples
        self.annotations_by_image = annotations_by_image
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = self.img_dir / s["file_name"]

        # Read as grayscale, then convert to 3-channel (matching ImageNet pretrained)
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((2048, 2048), dtype=np.uint8)
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # (H, W, 3)

        if self.is_test:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image, s["file_name"]

        h, w = gray.shape
        anns = self.annotations_by_image.get(s["image_id"], [])
        mask = coco_ann_to_mask(anns, h, w)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        mask = mask.unsqueeze(0).float()  # (1, H, W)
        return image, mask


# Augmentations
train_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# %% md
# ## 6. Loss Function

# %%
class DiceBCELoss(nn.Module):
    """0.5 * BCEWithLogits + 0.5 * DiceLoss"""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)

    def forward(self, pred, target):
        return 0.5 * self.bce(pred, target) + 0.5 * self.dice(pred, target)


# %% md
# ## 7. Training Loop

# %%
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(loader)


# %% md
# ## 8. Main Training

# %%
def main():
    # Prepare samples
    all_samples = []
    missing = 0
    for img_id, img_info in images_by_id.items():
        path = CFG.TRAIN_IMAGES / img_info["file_name"]
        if not path.exists():
            missing += 1
            continue
        all_samples.append({
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "height": img_info.get("height", 2048),
            "width": img_info.get("width", 2048),
        })
    print(f"Valid samples: {len(all_samples)} (missing: {missing})")

    # Train/val split
    train_samples, val_samples = train_test_split(
        all_samples, test_size=CFG.VAL_SPLIT, random_state=CFG.SEED, shuffle=True
    )
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = FilamentDataset(train_samples, anns_by_image, CFG.TRAIN_IMAGES, train_transform)
    val_ds = FilamentDataset(val_samples, anns_by_image, CFG.TRAIN_IMAGES, val_transform)
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=CFG.NUM_WORKERS)

    # Model
    model = smp.Unet(
        encoder_name=CFG.ENCODER,
        encoder_weights=CFG.ENCODER_WEIGHTS,
        in_channels=CFG.IN_CHANNELS,
        classes=CFG.NUM_CLASSES,
        activation=None,  # output logits
    ).to(CFG.DEVICE)

    print(f"Model: U-Net ({CFG.ENCODER}), params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(CFG.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CFG.DEVICE)
        val_loss = validate_epoch(model, val_loader, criterion, CFG.DEVICE)
        print(f"Epoch {epoch+1:02d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("  -> Best model saved")

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")

# Run training:
main()

# %% md
# ## 9. Inference & Submission

# %%
def mask_to_coco_rle(binary_mask):
    """Encode binary mask to COCO compressed RLE string (Fortran order)."""
    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle["counts"]


def predict_and_submit(model, test_dir, output_path="submission.csv"):
    """Run inference on test set and generate submission CSV."""
    model.eval()

    test_paths = sorted(Path(test_dir).glob("*.jpeg")) + sorted(Path(test_dir).glob("*.jpg"))
    # De-duplicate stems
    seen, unique = set(), []
    for p in test_paths:
        if p.stem not in seen:
            seen.add(p.stem)
            unique.append(p)

    submissions = []

    with torch.no_grad():
        for img_path in tqdm(unique, desc="Test inference"):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            # Preprocess
            image = cv2.resize(image, (CFG.IMG_SIZE, CFG.IMG_SIZE))
            image = val_transform(image=image)["image"]
            image = image.unsqueeze(0).to(CFG.DEVICE)

            logits = model(image)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]

            # Threshold + upscale to original size
            binary = (probs > CFG.THRESHOLD).astype(np.uint8)
            binary_full = cv2.resize(
                binary, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
            )

            # Instance separation
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_full)

            filament_count = 0
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] < CFG.MIN_AREA:
                    continue
                single_mask = (labels == label).astype(np.uint8)
                rle_str = mask_to_coco_rle(single_mask)
                if rle_str:
                    filament_count += 1
                    submissions.append({
                        "filament_id": f"{img_path.stem}_{filament_count}",
                        "segmentation_rle": rle_str,
                    })

            # Fallback: empty mask if nothing detected
            if filament_count == 0:
                empty_rle = mask_to_coco_rle(np.zeros((gray.shape[0], gray.shape[1]), dtype=np.uint8))
                submissions.append({
                    "filament_id": f"{img_path.stem}_1",
                    "segmentation_rle": empty_rle,
                })

    sub_df = pd.DataFrame(submissions)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved: {output_path} ({len(sub_df)} rows)")
    return sub_df

# Run inference on test set:
model = smp.Unet(encoder_name=CFG.ENCODER, encoder_weights=None,
                 in_channels=CFG.IN_CHANNELS, classes=CFG.NUM_CLASSES).to(CFG.DEVICE)
model.load_state_dict(torch.load("best_unet_model.pth"))
sub_df = predict_and_submit(model, CFG.TEST_IMAGES)
sub_df.head()