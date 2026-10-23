# %% md
# # Solar Filament Segmentation — Exploratory Data Analysis
#
# Competition: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
#
# This notebook explores:
# - Image size distribution across observatories
# - Annotation density (number of filaments per image)
# - Pixel intensity distributions
# - Site-specific differences (KSO vs Meudon vs BBSO vs Uccle)

# %% md
# ## 1. Setup

# %%
import json
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# %% md
# ## 2. Load Data

# %%
# Kaggle paths
COMPETITION = "/kaggle/input/filament-segmentation-2026"
TRAIN_IMAGES = Path(COMPETITION) / "train_images"
ANNOTATIONS_PATH = Path(COMPETITION) / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

# Load annotations
with open(ANNOTATIONS_PATH, "r") as f:
    ann_data = json.load(f)

print(f"Annotation type: {type(ann_data)}")

# List training images
train_images = sorted(TRAIN_IMAGES.glob("*.jpeg"))
print(f"Training images: {len(train_images)}")

# %% md
# ## 3. Parse Image Metadata

# %%
# Image naming: YYYYMMDDHHMMSSXX.jpeg
# XX = observatory code (Ch, Mh, Bh, Uh, Lh, Th)

def parse_filename(fname):
    """Extract timestamp and observatory from filename."""
    stem = Path(fname).stem  # e.g. "20230101120000Ch"
    timestamp_str = stem[:14]  # YYYYMMDDHHMMSS
    obs_code = stem[14:]  # Ch, Mh, Bh, Uh, Lh, Th
    return timestamp_str, obs_code

# TODO: Build DataFrame with image metadata
# TODO: Plot observatory distribution
# TODO: Plot temporal distribution

# %% md
# ## 4. Image Properties

# %%
# TODO: Check image sizes across observatories
# TODO: Check pixel value ranges (min, max, mean, std)
# TODO: Check if all images are grayscale

# %% md
# ## 5. Annotation Analysis

# %%
# TODO: Count filaments per image
# TODO: Distribution of polygon vertices
# TODO: Polygon area distribution
# TODO: Spatial distribution (where on the solar disk are filaments?)

# %% md
# ## 6. Site Comparison

# %%
# TODO: Compare image counts per observatory
# TODO: Compare filament counts per observatory
# TODO: Compare image resolutions per observatory
# TODO: Identify any systematic differences

# %% md
# ## 7. Summary

# %%
# TODO: Key findings table
# TODO: Recommendations for modeling