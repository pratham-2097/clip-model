# Expanded Dataset Training - YOLOv11

**Date:** 2025-01-27  
**Goal:** Maximum object detection accuracy and confidence  
**Status:** Training in progress

---

## Dataset Consolidation Summary

### Original Stage 2 Dataset
- **Location:** `STEP 2- Conditional classes.v1-stage-2--1.yolov11/`
- **Classes:** 9 conditional classes
- **Images:** 290 images (218 train + 47 valid + 25 test)

### Class Mapping (9 → 4 classes)

| Original Class | Consolidated Class |
|----------------|-------------------|
| 'Toe drain' | → 'toe_drain' |
| 'Toe drain- Blocked' | → 'toe_drain' |
| 'Toe drain- Damaged' | → 'toe_drain' |
| 'rock toe' | → 'rock_toe' |
| 'rock toe damaged' | → 'rock_toe' |
| 'slope drain' | → 'slope_drain' |
| 'slope drain blocked' | → 'slope_drain' |
| 'slope drain damaged' | → 'slope_drain' |
| 'vegetation' | → 'vegetation' |

### Consolidated Dataset Statistics

**Original Distribution (9 classes):**
- Toe drain: 52 instances
- Toe drain- Blocked: 78 instances
- Toe drain- Damaged: 76 instances
- rock toe: 153 instances
- rock toe damaged: 366 instances
- slope drain: 257 instances
- slope drain blocked: 97 instances
- slope drain damaged: 148 instances
- vegetation: 238 instances
- **Total:** 1,465 instances

**Consolidated Distribution (4 classes):**
- rock_toe: 519 instances (35.4%)
- slope_drain: 502 instances (34.3%)
- toe_drain: 206 instances (14.1%)
- vegetation: 238 instances (16.2%)

---

## Merged Dataset Statistics

### Final Merged Dataset
- **Location:** `dataset_merged/`
- **Total Images:** 375 images
- **Total Instances:** 1,605 instances

### Split Distribution

**Training Set:**
- Images: 320
- Instances: 1,395
  - rock_toe: 487 (34.9%)
  - slope_drain: 505 (36.2%)
  - toe_drain: 179 (12.8%)
  - vegetation: 224 (16.1%)

**Validation Set:**
- Images: 30
- Instances: 82
  - rock_toe: 28 (34.1%)
  - slope_drain: 42 (51.2%)
  - toe_drain: 3 (3.7%)
  - vegetation: 9 (11.0%)

**Test Set:**
- Images: 25
- Instances: 128
  - rock_toe: 46 (35.9%)
  - slope_drain: 39 (30.5%)
  - toe_drain: 19 (14.8%)
  - vegetation: 24 (18.8%)

### Overall Class Distribution
- **rock_toe:** 561 instances (35.0%)
- **slope_drain:** 586 instances (36.5%)
- **toe_drain:** 201 instances (12.5%)
- **vegetation:** 257 instances (16.0%)

---

## Training Configuration

### Model
- **Architecture:** YOLOv11-S (Small variant)
- **Pretrained:** Yes (COCO dataset weights)
- **Device:** MPS (Apple M2 Max)

### Training Strategy: Two-Phase Fine-Tuning

#### Phase 1: Freeze Backbone
- **Epochs:** 20 (increased from 15)
- **Frozen Layers:** First 10 layers
- **Optimizer:** SGD
- **Learning Rate:** 0.002
- **Batch Size:** 8
- **Image Size:** 640×640
- **Focus:** Learning feature representations

#### Phase 2: Full Fine-Tuning
- **Epochs:** 150 (extended for maximum accuracy)
- **Frozen Layers:** None (full model)
- **Optimizer:** AdamW
- **Learning Rate:** 0.0005 (lower for fine-tuning)
- **Batch Size:** 8
- **Image Size:** 640×640
- **Patience:** 50 (early stopping)
- **Focus:** Maximum accuracy convergence

### Data Augmentation
- **HSV-Hue:** 0.015
- **HSV-Saturation:** 0.7
- **HSV-Value:** 0.4
- **Translation:** 0.1
- **Scale:** 0.5
- **Flip LR:** 0.5
- **Mosaic:** 1.0

### Loss Weights
- **Box Loss:** 7.5
- **Class Loss:** 0.5
- **DFL Loss:** 1.5

---

## Expected Improvements

### Compared to Previous Models

**Previous YOLOv11-S (120 images):**
- mAP@0.5: 75.93%
- mAP@[0.5:0.95]: 51.11%
- Precision: 70.87%
- Recall: 80.75%

**Expected with Expanded Dataset (375 images):**
- **mAP@0.5:** Target >80% (improved from 75.93%)
- **mAP@[0.5:0.95]:** Target >55% (improved from 51.11%)
- **Precision:** Target >75% (improved from 70.87%)
- **Recall:** Target >85% (maintain or improve from 80.75%)
- **Per-class accuracy:** Improved, especially for minority classes

### Key Improvements
1. **More Training Data:** 3x more images (120 → 375)
2. **Better Class Balance:** More examples per class
3. **Extended Training:** 150 epochs vs 50 epochs
4. **Optimized Hyperparameters:** Fine-tuned for accuracy
5. **Better Augmentation:** Improved generalization

---

## Training Progress

**Status:** Training in progress (background)

**Command:**
```bash
cd yolov8_project
python3 scripts/train_yolov11_expanded.py
```

**Output Location:**
- Phase 1: `runs/detect/yolov11_expanded_freeze/`
- Phase 2: `runs/detect/yolov11_expanded_finetune/`
- Best Model: `runs/detect/yolov11_expanded_finetune/weights/best.pt`

**Estimated Time:**
- Phase 1: ~20-30 minutes
- Phase 2: ~2-4 hours (depending on convergence)
- Total: ~3-5 hours

---

## Next Steps

1. **Monitor Training:** Check progress in `runs/detect/yolov11_expanded_finetune/`
2. **Evaluate Model:** Run evaluation script when training completes
3. **Compare Metrics:** Compare with previous models
4. **Update UI:** Update Streamlit UI to use new model if improved
5. **Documentation:** Update project summary with new metrics

---

## Files Created

1. **`scripts/consolidate_classes.py`** - Class consolidation script
2. **`scripts/analyze_merged_dataset.py`** - Dataset analysis script
3. **`scripts/train_yolov11_expanded.py`** - Enhanced training script
4. **`dataset_consolidated/`** - Consolidated Stage 2 dataset
5. **`dataset_merged/`** - Final merged dataset

---

**Last Updated:** 2025-01-27  
**Training Status:** In Progress

