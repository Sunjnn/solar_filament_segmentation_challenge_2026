#!/usr/bin/env python3
"""Inspect the downloaded competition data: unzip, examine structure, image stats, annotation format."""
import json
import zipfile
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def unzip_data():
    """Unzip downloaded competition files."""
    for zf in sorted(DATA_DIR.glob("*.zip")):
        print(f"Unzipping: {zf.name} ({zf.stat().st_size / 1e6:.1f} MB)")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(DATA_DIR)
        print(f"  Done. Files: {len(z.namelist())}")
    print()

def inspect_structure():
    """Show directory structure of extracted data."""
    print("=== Data Directory Structure ===")
    for root, dirs, files in Path.walk(DATA_DIR):
        rel = root.relative_to(DATA_DIR)
        depth = len(rel.parts) if str(rel) != "." else 0
        if depth > 3:
            continue
        prefix = "  " * depth
        name = rel.name if str(rel) != "." else "data"
        print(f"{prefix}{name}/")
        if depth < 3:
            for d in sorted(dirs)[:10]:
                print(f"{prefix}  {d}/")
            for f in sorted(files)[:10]:
                print(f"{prefix}  {f}")
            if len(dirs) > 10:
                print(f"{prefix}  ... ({len(dirs)} dirs total)")
            if len(files) > 10:
                print(f"{prefix}  ... ({len(files)} files total)")
    print()

def inspect_images(image_dir, max_images=10):
    """Inspect image properties: size, channels, value range."""
    image_dir = Path(image_dir)
    if not image_dir.exists():
        print(f"Image dir not found: {image_dir}")
        return

    images = sorted(image_dir.glob("*.jpeg")) + sorted(image_dir.glob("*.jpg"))
    print(f"=== Image Inspection: {image_dir} ===")
    print(f"Total images: {len(images)}")

    sizes = Counter()
    observatories = Counter()
    value_stats = []

    for i, img_path in enumerate(images):
        if i >= max_images and max_images > 0:
            break
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [WARN] Cannot read: {img_path.name}")
            continue
        h, w = img.shape
        sizes[(w, h)] += 1

        # Parse observatory from filename
        stem = img_path.stem  # YYYYMMDDHHMMSSXX
        obs = stem[14:] if len(stem) >= 16 else "unknown"
        observatories[obs] += 1

        value_stats.append({
            "name": img_path.name,
            "size": f"{w}x{h}",
            "obs": obs,
            "min": img.min(),
            "max": img.max(),
            "mean": img.mean(),
            "std": img.std(),
        })

        if i == 0:
            print(f"\nFirst image: {img_path.name}")
            print(f"  Shape: {img.shape}, dtype: {img.dtype}")
            print(f"  Range: [{img.min()}, {img.max()}], mean={img.mean():.1f}, std={img.std():.1f}")

    print(f"\nImage sizes (first {max_images}):")
    for (w, h), cnt in sizes.most_common(5):
        print(f"  {w}x{h}: {cnt}")
    print(f"\nObservatories (first {max_images}):")
    for obs, cnt in observatories.most_common():
        print(f"  {obs}: {cnt}")
    print()

    # Show full directory stats (all images, just sizes)
    all_sizes = Counter()
    all_obs = Counter()
    for img_path in images:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            all_sizes[img.shape[1], img.shape[0]] += 1
            stem = img_path.stem
            all_obs[stem[14:] if len(stem) >= 16 else "?"] += 1

    print(f"Full dataset image sizes:")
    for (w, h), cnt in all_sizes.most_common():
        print(f"  {w}x{h}: {cnt}")
    print(f"\nFull dataset observatories:")
    for obs, cnt in sorted(all_obs.items()):
        print(f"  {obs}: {cnt}")
    print()

def inspect_annotations(ann_path):
    """Inspect COCO annotation JSON."""
    ann_path = Path(ann_path)
    if not ann_path.exists():
        # Try to find it
        candidates = list(DATA_DIR.rglob("*Annotations*.json"))
        if candidates:
            ann_path = candidates[0]
            print(f"Found annotations: {ann_path}")
        else:
            print(f"Annotation file not found: {ann_path}")
            return

    print(f"\n=== Annotation Inspection: {ann_path.name} ===")
    with open(ann_path) as f:
        data = json.load(f)

    print(f"Top-level keys: {list(data.keys())}")

    if "images" in data:
        images = data["images"]
        print(f"Images: {len(images)}")
        if images:
            print(f"  First: {images[0]}")
            # Check height/width distribution
            hw = Counter((img.get("height"), img.get("width")) for img in images)
            print(f"  Sizes: {hw.most_common(5)}")

    if "annotations" in data:
        anns = data["annotations"]
        print(f"Annotations: {len(anns)}")
        if anns:
            print(f"  First: {json.dumps(anns[0], indent=2)[:500]}")

            # Annotations per image
            per_img = Counter(ann.get("image_id") for ann in anns)
            counts = list(per_img.values())
            print(f"  Per-image counts: min={min(counts)}, max={max(counts)}, mean={np.mean(counts):.1f}, median={np.median(counts):.0f}")

            # Check segmentation format
            seg_types = Counter()
            for ann in anns[:100]:
                seg = ann.get("segmentation")
                if isinstance(seg, list):
                    if len(seg) > 0 and isinstance(seg[0], list):
                        seg_types["polygon_list"] += 1
                    elif len(seg) > 0:
                        seg_types[f"flat_list({len(seg)}elems)"] += 1
                    else:
                        seg_types["empty_list"] += 1
                elif isinstance(seg, dict):
                    seg_types["rle_dict"] += 1
                else:
                    seg_types[f"other({type(seg).__name__})"] += 1
            print(f"  Segmentation types (first 100): {dict(seg_types)}")

            # Check bbox
            if "bbox" in anns[0]:
                areas = [ann["bbox"][2] * ann["bbox"][3] for ann in anns]
                print(f"  Bbox area: min={min(areas):.0f}, max={max(areas):.0f}, mean={np.mean(areas):.1f}")

            # Check for iscrowd
            iscrowd = sum(1 for ann in anns if ann.get("iscrowd", 0))
            print(f"  iscrowd=1: {iscrowd}/{len(anns)}")

            # Check for annotator info in filament_id
            print(f"  Sample annotation keys: {list(anns[0].keys())}")

    if "categories" in data:
        cats = data["categories"]
        print(f"Categories: {cats}")

    print()

def main():
    unzip_data()
    inspect_structure()
    inspect_images(DATA_DIR)
    inspect_annotations(DATA_DIR / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

if __name__ == "__main__":
    main()