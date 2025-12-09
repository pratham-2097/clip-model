# 📊 YOLOv8 Model Metrics Documentation

**Model:** YOLOv8-S (Small)  
**Best Model Path:** `runs/detect/finetune_phase/weights/best.pt`  
**Training Date:** 2025-01-27  
**Evaluation Date:** 2025-01-27  
**Dataset:** 103 training images (after oversampling), 30 validation images

---

## 🎯 Overall Performance Metrics (Validation Set)

| Metric | Value | Percentage | Assessment |
|--------|-------|------------|------------|
| **mAP@0.5** | 0.7617 | **76.17%** | ✅ Excellent |
| **mAP@[0.5:0.95]** | 0.5153 | **51.53%** | ✅ Good |
| **Precision** | 0.7500 | **75.00%** | ✅ High |
| **Recall** | 0.7222 | **72.22%** | ✅ Good |
| **F1-Score** | 0.7358 | **73.58%** | ✅ Good |

### Interpretation
- **mAP@0.5 (76.17%):** When lenient about box placement (50% overlap), the model correctly detects objects 76.17% of the time
- **mAP@[0.5:0.95] (51.53%):** When strict about box placement (50-95% overlap), accuracy is 51.53% - indicates good bounding box precision
- **Precision (75.00%):** 3 out of 4 detections are correct (1 in 4 are false positives)
- **Recall (72.22%):** Finds about 3 out of 4 objects present (1 in 4 are missed)

---

## 📈 Per-Class Performance Metrics

| Class | mAP@0.5 | mAP@[0.5:0.95] | Precision | Recall | Status |
|-------|---------|----------------|-----------|--------|--------|
| **slope_drain** | 0.9167 | 0.6964 | 0.8605 | 0.8810 | ✅ Excellent |
| **rock_toe** | 0.8668 | 0.6381 | 0.7586 | 0.7857 | ✅ Excellent |
| **toe_drain** | 0.6672 | 0.3341 | 0.6667 | 0.6667 | ✅ Good |
| **vegetation** | 0.5963 | 0.3925 | 0.7143 | 0.5556 | ⚠️ Moderate |

### Class-Specific Details

#### ✅ slope_drain (Best Performing)
- **mAP@0.5:** 91.67% - Excellent detection accuracy
- **mAP@[0.5:0.95]:** 69.64% - Very tight bounding boxes
- **Precision:** 86.05% - Very few false positives
- **Recall:** 88.10% - Finds almost all slope drains
- **Instances in Validation:** 42 objects across 25 images

#### ✅ rock_toe (Second Best)
- **mAP@0.5:** 86.68% - Excellent detection accuracy
- **mAP@[0.5:0.95]:** 63.81% - Good bounding box precision
- **Precision:** 75.86% - Low false positive rate
- **Recall:** 78.57% - Finds most rock toes
- **Instances in Validation:** 28 objects across 15 images

#### ✅ toe_drain (Good)
- **mAP@0.5:** 66.72% - Good detection accuracy
- **mAP@[0.5:0.95]:** 33.41% - Moderate bounding box precision
- **Precision:** 66.67% - Some false positives
- **Recall:** 66.67% - Finds 2 out of 3 objects
- **Instances in Validation:** 3 objects across 3 images (very limited)

#### ⚠️ vegetation (Moderate)
- **mAP@0.5:** 59.63% - Moderate detection accuracy
- **mAP@[0.5:0.95]:** 39.25% - Moderate bounding box precision
- **Precision:** 71.43% - Some false positives
- **Recall:** 55.56% - Misses about half of vegetation instances
- **Instances in Validation:** 9 objects across 5 images

---

## ⚡ Inference Performance

| Metric | Value |
|--------|-------|
| **Preprocessing Time** | 7.7 ms per image |
| **Inference Time** | 20.5 ms per image |
| **Postprocessing Time** | 14.2 ms per image |
| **Total Time** | ~42.4 ms per image |
| **Throughput** | ~23.6 FPS (frames per second) |

**Device:** Apple M2 Max (MPS acceleration)

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | YOLOv8-S (Small) |
| **Input Size** | 640×640 pixels |
| **Batch Size** | 8 |
| **Epochs (Phase A)** | 15 (frozen backbone) |
| **Epochs (Phase B)** | 50 (full fine-tuning) |
| **Optimizer (Phase A)** | SGD |
| **Optimizer (Phase B)** | AdamW |
| **Learning Rate (Phase A)** | 0.002 |
| **Learning Rate (Phase B)** | 0.0005 |
| **Training Time** | ~15 minutes (0.25 hours) |
| **Total Parameters** | 11,127,132 |
| **GFLOPs** | 28.4 |

---

## 📉 Training History (Final Epoch)

| Metric | Training | Validation |
|--------|----------|------------|
| **Box Loss** | 0.7366 | 1.2220 |
| **Class Loss** | 0.6452 | 0.9270 |
| **DFL Loss** | 1.0202 | 1.3005 |
| **Precision** | 0.7129 | 0.7500 |
| **Recall** | 0.7552 | 0.7222 |
| **mAP@0.5** | 0.7028 | 0.7617 |
| **mAP@[0.5:0.95]** | 0.4563 | 0.5153 |

---

## 🧪 Test Dataset Performance (Challenging Data)

**Test Dataset:** `testforyolo/` (8 images, different distribution)

| Metric | Value | Percentage |
|--------|-------|------------|
| **Precision** | 0.3333 | 33.33% |
| **Recall** | 0.2000 | 20.00% |
| **F1-Score** | 0.2500 | 25.00% |

### Per-Class on Test Set

| Class | Precision | Recall | F1-Score | Status |
|-------|-----------|--------|----------|--------|
| **slope_drain** | 42.86% | 42.86% | 42.86% | ⚠️ Moderate |
| **rock_toe** | 0.00% | 0.00% | 0.00% | ❌ Poor |
| **vegetation** | 0.00% | 0.00% | 0.00% | ❌ Poor |
| **toe_drain** | N/A | N/A | N/A | Not in test set |

**Key Finding:** Model performs well on validation (same distribution) but struggles on challenging test data, indicating overfitting to training distribution.

---

## 📋 Dataset Statistics

| Split | Images | Objects | Classes |
|-------|--------|---------|---------|
| **Training** | 103 | ~171 | 4 |
| **Validation** | 30 | 82 | 4 |
| **Test** | 8 | 15 | 4 |

### Class Distribution (Training Set - After Oversampling)

| Class | Training Instances | Validation Instances | Total |
|-------|-------------------|---------------------|-------|
| **slope_drain** | 77 | 42 | 119 |
| **rock_toe** | 47 | 28 | 75 |
| **toe_drain** | 20 | 3 | 23 |
| **vegetation** | 27 | 9 | 36 |

---

## 🎯 Performance Summary

### ✅ Strengths
1. **Excellent overall detection** (76.17% mAP@0.5)
2. **Good bounding box precision** (51.53% mAP@[0.5:0.95])
3. **High precision** (75.00%) - few false positives
4. **Good recall** (72.22%) - finds most objects
5. **Fast inference** (~23.6 FPS on M2 Max)
6. **Excellent performance on dominant classes** (slope_drain: 91.67%, rock_toe: 86.68%)

### ⚠️ Areas for Improvement
1. **Minority class performance** (toe_drain: 66.72%, vegetation: 59.63%)
2. **Limited generalization** (poor performance on challenging test data)
3. **Small dataset** (103 training images limits generalization)
4. **Class imbalance** (toe_drain has only 3 validation examples)

---

## 📝 Notes

- **Evaluation Configuration:**
  - Confidence Threshold: 0.25
  - IoU Threshold: 0.5
  - Device: MPS (Apple M2 Max)
  - Split: Validation (30 images)

- **Model Selection:**
  - Best model selected based on validation mAP@0.5
  - Saved at: `runs/detect/finetune_phase/weights/best.pt`

- **Training Strategy:**
  - Two-phase approach: Freeze backbone → Full fine-tuning
  - Oversampling applied to balance classes
  - Data augmentation enabled (mosaic, flip, etc.)

---

**Last Updated:** 2025-01-27  
**Next Step:** Compare with YOLOv11 model performance

