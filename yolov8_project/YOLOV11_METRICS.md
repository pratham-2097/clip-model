# 📊 YOLOv11 Model Metrics Documentation

**Model:** YOLOv11-S (Small)  
**Best Model Path:** `runs/detect/yolov11_finetune_phase/weights/best.pt`  
**Training Date:** 2025-01-27  
**Evaluation Date:** 2025-01-27  
**Dataset:** 103 training images (after oversampling), 30 validation images

---

## 🎯 Overall Performance Metrics (Validation Set)

| Metric | Value | Percentage | Assessment |
|--------|-------|------------|------------|
| **mAP@0.5** | 0.7593 | **75.93%** | ✅ Excellent |
| **mAP@[0.5:0.95]** | 0.5111 | **51.11%** | ✅ Good |
| **Precision** | 0.7087 | **70.87%** | ✅ Good |
| **Recall** | 0.8075 | **80.75%** | ✅ Excellent |
| **F1-Score** | 0.7558 | **75.58%** | ✅ Good |

### Interpretation
- **mAP@0.5 (75.93%):** When lenient about box placement (50% overlap), the model correctly detects objects 75.93% of the time
- **mAP@[0.5:0.95] (51.11%):** When strict about box placement (50-95% overlap), accuracy is 51.11% - indicates good bounding box precision
- **Precision (70.87%):** About 7 in 10 detections are correct (3 in 10 are false positives)
- **Recall (80.75%):** Finds about 4 out of 5 objects present (1 in 5 are missed)

---

## 📈 Per-Class Performance Metrics

| Class | mAP@0.5 | mAP@[0.5:0.95] | Precision | Recall | Status |
|-------|---------|----------------|-----------|--------|--------|
| **slope_drain** | 0.9423 | 0.7146 | 0.9512 | 0.9286 | ✅ Excellent |
| **rock_toe** | 0.8831 | 0.6229 | 0.7059 | 0.8571 | ✅ Excellent |
| **vegetation** | 0.7012 | 0.4644 | 0.7778 | 0.7778 | ✅ Good |
| **toe_drain** | 0.5107 | 0.2424 | 0.4000 | 0.6667 | ⚠️ Moderate |

### Class-Specific Details

#### ✅ slope_drain (Best Performing)
- **mAP@0.5:** 94.23% - Excellent detection accuracy
- **mAP@[0.5:0.95]:** 71.46% - Very tight bounding boxes
- **Precision:** 95.12% - Very few false positives
- **Recall:** 92.86% - Finds almost all slope drains
- **Instances in Validation:** 42 objects across 25 images

#### ✅ rock_toe (Second Best)
- **mAP@0.5:** 88.31% - Excellent detection accuracy
- **mAP@[0.5:0.95]:** 62.29% - Good bounding box precision
- **Precision:** 70.59% - Some false positives
- **Recall:** 85.71% - Finds most rock toes
- **Instances in Validation:** 28 objects across 15 images

#### ✅ vegetation (Good)
- **mAP@0.5:** 70.12% - Good detection accuracy
- **mAP@[0.5:0.95]:** 46.44% - Moderate bounding box precision
- **Precision:** 77.78% - Low false positive rate
- **Recall:** 77.78% - Finds most vegetation instances
- **Instances in Validation:** 9 objects across 5 images

#### ⚠️ toe_drain (Moderate)
- **mAP@0.5:** 51.07% - Moderate detection accuracy
- **mAP@[0.5:0.95]:** 24.24% - Lower bounding box precision
- **Precision:** 40.00% - Higher false positive rate
- **Recall:** 66.67% - Finds 2 out of 3 objects
- **Instances in Validation:** 3 objects across 3 images (very limited)

---

## ⚡ Inference Performance

| Metric | Value |
|--------|-------|
| **Preprocessing Time** | 0.7 ms per image |
| **Inference Time** | 20.6 ms per image |
| **Postprocessing Time** | 12.9 ms per image |
| **Total Time** | ~34.2 ms per image |
| **Throughput** | ~29.2 FPS (frames per second) |

**Device:** Apple M2 Max (MPS acceleration)

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | YOLOv11-S (Small) |
| **Input Size** | 640×640 pixels |
| **Batch Size** | 8 |
| **Epochs (Phase A)** | 15 (frozen backbone) |
| **Epochs (Phase B)** | 50 (full fine-tuning) |
| **Optimizer (Phase A)** | SGD |
| **Optimizer (Phase B)** | AdamW |
| **Learning Rate (Phase A)** | 0.002 |
| **Learning Rate (Phase B)** | 0.0005 |
| **Training Time (Phase 1)** | 3.86 minutes |
| **Training Time (Phase 2)** | 15.31 minutes |
| **Total Training Time** | 19.17 minutes (0.32 hours) |
| **Total Parameters** | 9,414,348 |
| **GFLOPs** | 21.3 |

---

## 📋 Performance Summary

### ✅ Strengths
1. **Excellent recall** (80.75%) - finds most objects
2. **Good overall detection** (75.93% mAP@0.5)
3. **Good bounding box precision** (51.11% mAP@[0.5:0.95])
4. **Excellent performance on dominant classes** (slope_drain: 94.23%, rock_toe: 88.31%)
5. **Faster inference** (~29.2 FPS vs YOLOv8's ~23.6 FPS)
6. **Smaller model** (9.4M parameters vs YOLOv8's 11.1M)

### ⚠️ Areas for Improvement
1. **Lower precision** (70.87%) - more false positives than YOLOv8
2. **Minority class performance** (toe_drain: 51.07%)
3. **Small dataset** (103 training images limits generalization)

---

## 📝 Notes

- **Evaluation Configuration:**
  - Confidence Threshold: 0.25
  - IoU Threshold: 0.5
  - Device: MPS (Apple M2 Max)
  - Split: Validation (30 images)

- **Model Selection:**
  - Best model selected based on validation mAP@0.5
  - Saved at: `runs/detect/yolov11_finetune_phase/weights/best.pt`

- **Training Strategy:**
  - Two-phase approach: Freeze backbone → Full fine-tuning
  - Same strategy as YOLOv8 for fair comparison
  - Oversampling applied to balance classes
  - Data augmentation enabled (mosaic, flip, etc.)

---

**Last Updated:** 2025-01-27  
**Comparison:** See `MODEL_COMPARISON.md` for YOLOv8 vs YOLOv11 comparison

