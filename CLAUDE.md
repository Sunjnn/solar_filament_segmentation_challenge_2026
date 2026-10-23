# CLAUDE.md — Solar Filament Segmentation Challenge 2026

## Project Context

- **Competition**: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
- **Deadline**: 2026-11-15 (~24 days from start)
- **Task**: Solar H-alpha filament segmentation → polygon masks
- **Metric**: Panoptic Quality (PQ)
- **Prize**: $3,000, Community Competition (requires technical report PDF + reproducible code)
- **Participants**: 1124 teams

## Environment

- **OS**: macOS arm64 (Apple Silicon)
- **Python**: 3.14.7 (via homebrew)
- **Package manager**: uv (`uv pip install`)
- **Virtual env**: `.venv/` at project root
- **Kaggle CLI**: `~/.kaggle/access_token` must exist (kaggle.json, chmod 600)

## Key Conventions

1. **Kernel push = git commit**: Each `kaggle kernels push` must correspond to exactly one git commit.
   Flow: edit `src/*.py` → `python src/make_notebook.py` → `git add` → `git commit` → `kaggle kernels push`
2. **Commit messages**: English, one line, conventional commit prefix (feat/fix/doc/chore), no trailing period.
3. **Submission**: Each submission references its commit hash in the description.
4. **README**: Records every run: model config, CV score, LB score, ablation results.
5. **No large files in git**: All generated files (ipynb, data, outputs, .venv, .cache) are gitignored.
6. **Read public notebooks seriously**: `kaggle kernels pull` to read actual code, not just summaries.

## Known Pitfalls

- **Multi-annotator consensus**: Annotations come from multiple human annotators' consensus.
  "Looks right" masks can score very low because the metric penalizes predictions that annotators didn't agree on.
  Overly precise boundaries are punished.
- **LB reliability**: Top scores (~0.69 PQ) may involve data leak or LB overfitting.
  Discussion threads 740400 and 745880 cover this.
  "Honest methods" score 0.46–0.55.
- **Self-eval vs LB gap**: Public notebooks report large gaps between self-evaluated scores and LB.
- **External data rules**: GONG H-alpha data is public but has usage restrictions. Start without it.

## Project Structure

```
filament_segmentation/
  CLAUDE.md                 # This file
  requirements.txt           # Python deps
  README.md                  # Run log, model table, findings
  src/                       # Python sources (percent-format, generate ipynb)
    make_notebook.py         # Convert percent-format .py → .ipynb
    filament_baseline.py     # Baseline U-Net notebook
    filament_eda.py          # EDA notebook
    exp_domain.py            # Cross-site/temporal generalization experiments
    smoke_check.py           # Local smoke test (10 images)
  kaggle_baseline/           # Kaggle kernel directory
    kernel-metadata.json
  outputs/                   # Kaggle outputs (gitignored)
  data/                      # Local data (gitignored)
  kaggle_fe/                 # Feature engineering / inference kernels
  .venv/                     # Virtual env (gitignored)
  .cache/                    # Temp files (gitignored)
```

## Dependency Notes

- No `brew install libomp` needed (unlike lightgbm projects) — this is an image task with torch.
- Some packages may have native dependency issues on macOS arm64; record any in this file.

## Competition-Specific Notes

### Image naming format
`YYYYMMDDHHMMSSXX.jpeg` — timestamp + observatory code:
- Ch = KSO (Kanzelhöhe)
- Mh = Meudon
- Bh = BBSO (Big Bear)
- Uh = Uccle
- Lh, Th = other stations

### Data Format & Paths

**Annotation JSON** is in **COCO format** with multi-annotator structure:

```json
{
  "info": {...},
  "licenses": [...],
  "categories": [
    {"id": 1, "name": "Left", "supercategory": "filament"},
    {"id": 2, "name": "Right", "supercategory": "filament"},
    {"id": 3, "name": "Unidentifiable", "supercategory": "filament"},
    {"id": 4, "name": "Ambiguous", "supercategory": "filament"}
  ],
  "images": [
    {
      "id": "040301-20140609195854Bh",  // annotatorID-imageFilename
      "file_name": "20140609195854Bh.jpeg",
      "height": 2048, "width": 2048,
      "date_captured": "2014-06-09 19:58:54"
    }
  ],
  "annotations": [
    {
      "id": int,
      "image_id": "040301-20140609195854Bh",  // matches images[].id
      "category_id": 1,  // 1=Left, 2=Right, 3=Unidentifiable, 4=Ambiguous
      "segmentation": [[x1,y1, x2,y2, ...]],  // polygon (float coords)
      "bbox": [x, y, w, h],
      "area": float,
      "iscrowd": 0,  // always 0
      "spine": [x1,y1, x2,y2, ...]  // centerline of the filament!
    }
  ]
}
```

**Key stats**:
- **707 unique training images**, 180 test images
- **37 annotators**, 296 images (42%) annotated by 2-3 people
- **8199 annotations** total, 7.1 per image on average (range 1-26)
- All images are **2048×2048** grayscale JPEG, pixel range [0, ~255], mean ~80
- 6 observatories: Bh(118), Ch(125), Lh(127), Mh(125), Th(110), Uh(102)
- Category distribution: Unidentifiable(3074, 37.5%), Right(2590, 31.6%), Left(2535, 30.9%)
- **Spine field**: polygon centerline of each filament (4-244 points), useful for topology

**Kaggle paths**:
- Competition data: `/kaggle/input/filament-segmentation-2026/`
- With MAGFiLO: `/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/`
  - Train images: `train/train_images/`
  - Test images: `test/test_images/`
  - Annotations: `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`

**Submission format**:
```csv
filament_id,segmentation_rle
20110120105534Ch_1,<COCO compressed RLE string>
```
- `filament_id` = `{image_stem}_{counter}` (counter starts at 1)
- `segmentation_rle` = COCO compressed RLE (Fortran-order, pycocotools.mask.encode)
- Image name stem excludes extension (e.g., "20110120105534Ch")

### Public Notebook Analysis (pulled & read 2026-10-22)

### 1. Baseline U-Net — Pavlo Ivanin (52 votes, claims 0.9 self-eval)
- **File**: `.cache/notebooks/baseline-unet/`
- **Model**: `smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)`
- **Image**: 2048×2048 → resize to 512×512
- **Input**: RGB (cv2 reads grayscale as BGR, then converts to RGB → 3 identical channels)
- **Loss**: 0.5×BCEWithLogits + 0.5×DiceLoss (smp)
- **Optimizer**: AdamW, lr=3e-4, batch=8, epochs=5
- **Aug**: HorizontalFlip(p=0.5), VerticalFlip(p=0.5), ImageNet normalization
- **Post**: threshold=0.5, connectedComponentsWithStats, min area=30px
- **Submission**: COCO RLE (Fortran order, pycocotools.mask.encode)
- **filament_id format**: `{image_stem}_{counter}`

### 2. LB 1st — RS (60 votes) — SAME as baseline, just more training
- **Identical code** to Pavlo Ivanin's baseline
- **Differences**: epochs=35 (vs 5), lr=3e-3 (vs 3e-4)
- Training time: ~2.3 hours on Kaggle GPU

### 3. Self-Evaluation Notebook — Azim Ahmadzadeh (47 votes)
- **File**: `.cache/notebooks/self-eval/`
- **PQ formula**: `sum(IoU_TP) / (|TP| + 0.5×|FP| + 0.5×|FN|)`, IoU threshold=0.5
- **Multi-annotator**: GT filament_id format = `annotator_id-image_id_filament_number`
  - Prediction format: `image_id_filament_number`
  - Each annotator's labels matched separately → FP/FN counted per annotator
- **This explains the "honest methods" gap**: If 2 of 5 annotators marked a filament, predicting it counts as TP for 2 annotators but FP for 3 annotators.
- **Image size**: 2048×2048 (hardcoded in rles_to_layers)
- **RLE**: COCO compressed RLE strings

### 4. Mask R-CNN + U-Net Refiner — Nomannic (57 votes)
- **File**: `.cache/notebooks/mask-rcnn/`
- **Two-stage**: Mask R-CNN detects → U-Net refines each detection's mask
- Uses COCO annotation format directly (pycocotools.coco.COCO)
- Refiner works on cropped regions around each detection

### 5. YOLO11s Detect → Crop U-Net — Tim Gluhkikh (33 votes)
- **File**: `.cache/notebooks/yolo11s-unet/`
- **Pipeline**: YOLO11s detect bounding boxes → crop around box → "Seed-Refiner" U-Net
- **Key innovation — "Seed" channel**: 3-channel input to refiner:
  - Channel 1: grayscale image crop
  - Channel 2: CLAHE(image) (contrast enhancement)
  - Channel 3: filled padded box (the "seed" — tells refiner WHERE to look)
- **Seed training**: jitter GT box during training (±15% shift, 0.85–1.25× scale)
- **Seed inference**: YOLO detection box padded by PAD_RATIO=0.40
- **Context crop**: square around bbox, side = max(bw,bh) × 1.5, min 96px
- **Refiner**: smp.Unet(resnet34, in_channels=3) trained with 0.45×BCE + 0.55×Dice
- **TTA**: Box-Jitter TTA (shift box ±8px, batch forward, mean probs)
- **Params**: YOLO_IMGSZ=1280, CONF=0.28, IOU=0.50, epochs=12, lr=3e-4
- **Baseline PQ**: ~0.409
- **Post**: non-overlapping mask resolution (higher-score mask wins pixels)

### Key Architectural Insights
1. **Single-stage U-Net (512×512)**: simplest, 0.46–0.55 "honest" PQ
2. **Two-stage (detect + refine)**: YOLO/Mask R-CNN finds WHERE, U-Net refines WHAT
3. **Seed idea is powerful**: giving the refiner a "hint" about where to segment
4. **Multi-annotator problem is fundamental**: need to predict what the CONSENSUS sees
5. **ImageNet normalization works**: despite H-alpha being fundamentally different from natural images

## Commands Reference

```bash
# Setup
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate

# Generate notebook
python src/make_notebook.py

# Kaggle
kaggle competitions download -c filament-segmentation-2026 -p data
kaggle kernels push -p kaggle_baseline
kaggle competitions submissions -c filament-segmentation-2026
```