# %% md
# # Smoke Check — Local Sanity Test
#
# Quick local test: load 10 training images, parse annotations, create masks,
# train U-Net for 2 epochs on CPU, verify output format.
#
# Purpose: catch bugs before pushing to Kaggle.

# %% md
# ## 1. Setup

# %%
import json
import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# Add parent to path for local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# %% md
# ## 2. Load Small Subset

# %%
# TODO: Download a small subset of data if not already present
# TODO: or use kaggle api to download just 10 images

# For now, use Kaggle's download:
# kaggle competitions download -c filament-segmentation-2026 -p data
# Then unzip train_images and annotations

ANNOTATIONS_PATH = DATA_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

if not ANNOTATIONS_PATH.exists():
    print(f"❌ Annotations not found at {ANNOTATIONS_PATH}")
    print("   Run: kaggle competitions download -c filament-segmentation-2026 -p data")
    print("   Then unzip the downloaded files.")
    sys.exit(1)

with open(ANNOTATIONS_PATH, "r") as f:
    ann_data = json.load(f)

print(f"✅ Loaded annotations: {type(ann_data)}")

# %% md
# ## 3. Inspect Annotation Format

# %%
# TODO: Print structure of annotation JSON
# TODO: Identify key names, image_id format, polygon format

# %% md
# ## 4. Visualize Sample

# %%
# TODO: Load 3 images with their masks
# TODO: Overlay masks on images
# TODO: Check that polygon → mask conversion is correct

# %% md
# ## 5. Test Model Forward Pass

# %%
# TODO: Create a tiny U-Net
# TODO: Forward pass with random input
# TODO: Verify output shape is (B, 1, H, W)

# %% md
# ## 6. Test Submission Format

# %%
# TODO: Generate dummy predictions
# TODO: Convert to submission JSON format
# TODO: Verify the JSON schema matches what Kaggle expects

# %% md
# ## 7. Summary

# %%
# Print summary of what passed/failed