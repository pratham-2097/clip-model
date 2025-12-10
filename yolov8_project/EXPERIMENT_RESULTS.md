# Model Improvement Experiments - Results Tracker

**Goal:** Surpass YOLOv8-S (76.17% mAP@0.5) and YOLOv11-S (75.93% mAP@0.5) baselines

**Current Best:** mAP@0.5=74.50%, mAP@[0.5:0.95]=52.40% (v52)

---

## Baseline Targets

| Model | mAP@0.5 | mAP@[0.5:0.95] |
|-------|---------|----------------|
| **YOLOv8-S** | **76.17%** | **51.53%** |
| **YOLOv11-S** | **75.93%** | **51.11%** |

---

## Experiment Results

### Step 1: Different Hyperparameters

#### 1.1 Lower Learning Rate (lr0=0.0002)
- **Status:** Running
- **Config:** lr0=0.0002, patience=50, epochs=100
- **Best Metrics:** TBD
- **Notes:** Slower learning may find better minima

#### 1.2 Reduced Augmentation
- **Status:** Running
- **Config:** mosaic=0.3, mixup=0.0, close_mosaic=20
- **Best Metrics:** TBD
- **Notes:** Less augmentation may improve precision

#### 1.3 Higher Resolution (768px)
- **Status:** Pending (if memory allows)
- **Config:** imgsz=768, batch=4
- **Best Metrics:** TBD

---

### Step 2: Different Approaches

#### 2.1 Start from YOLOv11-S Baseline
- **Status:** Running
- **Config:** Start from baseline weights instead of expanded
- **Best Metrics:** TBD
- **Notes:** Fresh start may avoid local minima

#### 2.2 SGD Optimizer
- **Status:** Running
- **Config:** optimizer=SGD, lr0=0.01, momentum=0.937
- **Best Metrics:** TBD
- **Notes:** Different optimizer may escape plateau

#### 2.3 Class-Weighted Loss
- **Status:** Pending (after analysis)
- **Config:** Weighted loss for minority classes
- **Best Metrics:** TBD

---

## Dataset Analysis (Step 3)

### Class Distribution
- **rock_toe:** 561 instances (35.0%)
- **slope_drain:** 586 instances (36.5%)
- **toe_drain:** 201 instances (12.5%) ⚠️ Minority
- **vegetation:** 257 instances (16.0%) ⚠️ Minority

### Validation Set Issues
- **Total:** 30 images, 82 instances
- **toe_drain:** Only 3 instances in validation ⚠️ Very small!
- **vegetation:** Only 9 instances in validation ⚠️ Small

### Recommendations
1. **Class imbalance:** toe_drain and vegetation are underrepresented
2. **Small validation set:** May cause unstable metrics
3. **Consider:** Class-weighted loss or data augmentation for minority classes

---

## Final Comparison Table

| Experiment | mAP@0.5 | mAP@[0.5:0.95] | Status | Notes |
|------------|---------|----------------|--------|-------|
| YOLOv8-S (Baseline) | 76.17% | 51.53% | Target | - |
| YOLOv11-S (Baseline) | 75.93% | 51.11% | Target | - |
| Previous Best (v52) | 74.50% | 52.40% | Below | Plateaued early |
| 1.1 Lower LR | TBD | TBD | Running | - |
| 1.2 Reduced Aug | TBD | TBD | Running | - |
| 2.1 Baseline Start | TBD | TBD | Running | - |
| 2.2 SGD | TBD | TBD | Running | - |

---

## Next Steps

1. Monitor all experiments until completion
2. Run validation on all best.pt models
3. Compare results and select best model
4. If none surpass baselines, try Step 1.3 (higher resolution) and Step 2.3 (class weights)

---

**Last Updated:** Auto-updated by tracking script

