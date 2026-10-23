# Solar Filament Segmentation Challenge 2026 — Run Log

> Competition: [filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
> Deadline: 2026-11-15 | Teams: 1124 | Prize: $3,000
> Metric: Panoptic Quality (PQ)

## Quick Links

- [Kaggle Competition](https://www.kaggle.com/competitions/filament-segmentation-2026)
- [Discussion: LB reliability (740400)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/740400)
- [Discussion: scoring convention (745880)](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/745880)

## Leaderboard Context

| Tier | PQ Range | Notes |
|------|----------|-------|
| Top | ~0.69 | Potentially data leak / LB overfitting |
| Pack | ~0.62 | Many teams clustered here |
| Honest | 0.46–0.55 | Methods without data leak |

## Public Notebooks Read (2026-10-22)

| # | Notebook | Votes | Key Idea | Notes |
|---|-----------|-------|----------|-------|
| 1 | LB 1st Solar Filament Segmentation 2026 | 60 | U-Net ResNet34, 35 epochs, lr=3e-3 | **Same code as #2**, just longer training |
| 2 | Baseline Pipeline (U-Net) | 52 | U-Net ResNet34, 5 epochs, lr=3e-4 | Claims 0.9 self-eval (not PQ!) |
| 3 | Mask R-CNN + U-Net Refiner | 57 | Two-stage: detect + refine | COCO-format based |
| 4 | YOLO11s Detect → Crop U-Net | 33 | YOLO + Seed-Refiner U-Net | PQ ~0.409; seed = filled padded box |
| 5 | Self Evaluation Notebook | 47 | Official PQ metric code | Multi-annotator evaluation |

### Key Insights from Notebooks

1. **PQ = sum(IoU_TP) / (|TP| + 0.5|FP| + 0.5|FN|)**, IoU threshold = 0.5
2. **Multi-annotator**: GT has per-annotator filament IDs. Predicting a filament only 2/5 annotators marked → 2 TP + 3 FP. This penalizes "over-detection".
3. **Image size**: 2048×2048, resized to 512×512 for training
4. **Input**: 3-channel RGB (grayscale → 3 identical channels) with ImageNet normalization
5. **Seed idea**: The YOLO+refiner notebook uses a 3-channel input [image, CLAHE, seed_box] — telling the refiner WHERE to look
6. **Instance separation**: cv2.connectedComponentsWithStats, min area 30–200px
7. **Submission**: COCO RLE (Fortran order), filament_id = `{image_stem}_{counter}`
8. **Self-eval vs LB**: Public notebooks' self-evaluated scores are NOT comparable to LB PQ due to different evaluation

## EDA Findings

### Data Structure (from `inspect_data.py`, 2026-10-22)
- **707 training images**, 180 test images, all 2048×2048 grayscale JPEG
- **37 annotators**, 8199 total annotations (7.1/img avg, range 1-26)
- **296/707 images (42%)** annotated by 2-3 people → multi-annotator consensus critical
- **6 observatories**: Bh(118), Ch(125), Lh(127), Mh(125), Th(110), Uh(102) — well balanced
- **4 categories**: Unidentifiable(3074, 37.5%), Right(2590), Left(2535), Ambiguous(0)
- **Spine field**: Each annotation has a polygon centerline (4-244 points) — useful for topology-aware losses
- **Image IDs**: `annotatorID-imageFilename` format, e.g. `040301-20140609195854Bh`
- **Pixel range**: [0, ~255], mean ~80, std ~58 (dark solar disk with bright background)
- **Bbox area**: min=13, max=385481, mean=13110 — huge size variation
- **No iscrowd** annotations (all 0)

### Key Visualization Insight
From `outputs/sample_visualization.png`:
- Annotators have **significant disagreement** on what counts as a filament
- Some annotators mark 2-3× more filaments than others on the same image
- "Unidentifiable" category heavily used — these may be ambiguous structures
- This directly explains why PQ punishes "over-detection": predicting what only 1/3 annotators saw = 2 FP

## Run History

| # | Date | Model | CV PQ | LB PQ | Commit | Notes |
|---|------|-------|-------|-------|--------|-------|
| 1 | 2026-10-23 | U-Net ResNet34, 512x512, 5ep, lr=3e-4, BCE+Dice | — | 0.18 | 900b1aa | First baseline, 5ep lr=3e-4 512x512 too weak |

## Known Issues & Workarounds

- **Data download 403**: Kaggle API returns 403. Competition rules likely need to be accepted on the website first.
  - Token format: KGAT (new style)
  - Workaround: accept rules at https://www.kaggle.com/competitions/filament-segmentation-2026/rules
- **uv venv uses CPython 3.12.11** (not system Python 3.14.7) — this is fine

## Submission Checklist

- [ ] Kaggle submission (score on LB)
- [ ] Technical report PDF
- [ ] Reproducible code repository
- [ ] README with run log