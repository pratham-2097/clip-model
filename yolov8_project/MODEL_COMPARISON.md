# 📊 YOLOv8 vs YOLOv11 Model Comparison

**Comparison Date:** 2025-01-27  
**Dataset:** Same dataset (103 training images, 30 validation images)  
**Training Strategy:** Identical (two-phase freeze/unfreeze approach)

---

## 🎯 Overall Performance Comparison

| Metric | YOLOv8-S | YOLOv11-S | Difference | Winner |
|--------|----------|-----------|-------------|--------|
| **mAP@0.5** | **76.17%** | 75.93% | -0.24% | 🏆 YOLOv8 |
| **mAP@[0.5:0.95]** | **51.53%** | 51.11% | -0.42% | 🏆 YOLOv8 |
| **Precision** | **75.00%** | 70.87% | -4.13% | 🏆 YOLOv8 |
| **Recall** | 72.22% | **80.75%** | +8.53% | 🏆 YOLOv11 |
| **F1-Score** | **73.58%** | 75.58% | +2.00% | 🏆 YOLOv11 |

### Key Insights
- **YOLOv8:** Better precision (fewer false positives) and slightly better overall mAP
- **YOLOv11:** Better recall (finds more objects) and better F1-score (balanced metric)
- **Overall:** Very close performance, with YOLOv8 having slight edge in accuracy, YOLOv11 in recall

---

## 📈 Per-Class Performance Comparison

### slope_drain (Best Class for Both)

| Metric | YOLOv8-S | YOLOv11-S | Difference | Winner |
|--------|----------|-----------|-------------|--------|
| **mAP@0.5** | 91.67% | **94.23%** | +2.56% | 🏆 YOLOv11 |
| **mAP@[0.5:0.95]** | 69.64% | **71.46%** | +1.82% | 🏆 YOLOv11 |
| **Precision** | 86.05% | **95.12%** | +9.07% | 🏆 YOLOv11 |
| **Recall** | 88.10% | **92.86%** | +4.76% | 🏆 YOLOv11 |

**Winner: 🏆 YOLOv11** - Better across all metrics for slope_drain

---

### rock_toe

| Metric | YOLOv8-S | YOLOv11-S | Difference | Winner |
|--------|----------|-----------|-------------|--------|
| **mAP@0.5** | 86.68% | **88.31%** | +1.63% | 🏆 YOLOv11 |
| **mAP@[0.5:0.95]** | **63.81%** | 62.29% | -1.52% | 🏆 YOLOv8 |
| **Precision** | **75.86%** | 70.59% | -5.27% | 🏆 YOLOv8 |
| **Recall** | 78.57% | **85.71%** | +7.14% | 🏆 YOLOv11 |

**Winner: 🏆 YOLOv11** - Better mAP@0.5 and recall (finds more rock toes)

---

### vegetation

| Metric | YOLOv8-S | YOLOv11-S | Difference | Winner |
|--------|----------|-----------|-------------|--------|
| **mAP@0.5** | 59.63% | **70.12%** | +10.49% | 🏆 YOLOv11 |
| **mAP@[0.5:0.95]** | 39.25% | **46.44%** | +7.19% | 🏆 YOLOv11 |
| **Precision** | 71.43% | **77.78%** | +6.35% | 🏆 YOLOv11 |
| **Recall** | 55.56% | **77.78%** | +22.22% | 🏆 YOLOv11 |

**Winner: 🏆 YOLOv11** - Significantly better across all metrics (especially recall +22%)

---

### toe_drain (Minority Class)

| Metric | YOLOv8-S | YOLOv11-S | Difference | Winner |
|--------|----------|-----------|-------------|--------|
| **mAP@0.5** | **66.72%** | 51.07% | -15.65% | 🏆 YOLOv8 |
| **mAP@[0.5:0.95]** | **33.41%** | 24.24% | -9.17% | 🏆 YOLOv8 |
| **Precision** | **66.67%** | 40.00% | -26.67% | 🏆 YOLOv8 |
| **Recall** | 66.67% | **66.67%** | 0.00% | 🤝 Tie |

**Winner: 🏆 YOLOv8** - Much better performance on this minority class

---

## ⚡ Model Efficiency Comparison

| Metric | YOLOv8-S | YOLOv11-S | Difference | Winner |
|--------|----------|-----------|-------------|--------|
| **Parameters** | 11,127,132 | **9,414,348** | -1,712,784 | 🏆 YOLOv11 |
| **GFLOPs** | 28.4 | **21.3** | -7.1 | 🏆 YOLOv11 |
| **Inference Speed** | ~23.6 FPS | **~29.2 FPS** | +5.6 FPS | 🏆 YOLOv11 |
| **Model Size** | ~28.4 MB | **~19.2 MB** | -9.2 MB | 🏆 YOLOv11 |
| **Training Time** | ~15 min | **~19 min** | +4 min | 🏆 YOLOv8 |

**Winner: 🏆 YOLOv11** - More efficient (smaller, faster, fewer parameters)

---

## 📊 Training Comparison

| Aspect | YOLOv8-S | YOLOv11-S | Notes |
|--------|----------|-----------|-------|
| **Phase 1 Time** | ~3 min | ~3.86 min | Similar |
| **Phase 2 Time** | ~12 min | ~15.31 min | YOLOv11 slightly longer |
| **Total Time** | ~15 min | ~19.17 min | YOLOv8 faster |
| **Training Strategy** | Freeze/Unfreeze | Freeze/Unfreeze | Identical |
| **Optimizer** | SGD → AdamW | SGD → AdamW | Identical |
| **Learning Rate** | 0.002 → 0.0005 | 0.002 → 0.0005 | Identical |
| **Batch Size** | 8 | 8 | Identical |
| **Epochs** | 15 + 50 | 15 + 50 | Identical |

---

## 🎯 Use Case Recommendations

### Choose YOLOv8-S When:
- ✅ **Precision is critical** - Need fewer false positives (75% vs 71%)
- ✅ **Minority class performance matters** - Better on toe_drain (66.7% vs 51.1%)
- ✅ **Faster training needed** - 15 min vs 19 min
- ✅ **Overall mAP is priority** - Slightly better mAP@0.5 (76.2% vs 75.9%)

### Choose YOLOv11-S When:
- ✅ **Recall is critical** - Need to find more objects (80.8% vs 72.2%)
- ✅ **Model efficiency matters** - Smaller model (9.4M vs 11.1M params)
- ✅ **Faster inference needed** - 29.2 FPS vs 23.6 FPS
- ✅ **Vegetation detection is important** - Much better (70.1% vs 59.6%)
- ✅ **slope_drain is priority** - Better performance (94.2% vs 91.7%)
- ✅ **Deployment on edge devices** - Smaller model size and faster inference

---

## 📋 Summary Table: All Metrics

| Metric | YOLOv8-S | YOLOv11-S | Winner |
|--------|----------|-----------|--------|
| **Overall mAP@0.5** | 76.17% | 75.93% | 🏆 YOLOv8 (+0.24%) |
| **Overall mAP@[0.5:0.95]** | 51.53% | 51.11% | 🏆 YOLOv8 (+0.42%) |
| **Overall Precision** | 75.00% | 70.87% | 🏆 YOLOv8 (+4.13%) |
| **Overall Recall** | 72.22% | 80.75% | 🏆 YOLOv11 (+8.53%) |
| **Overall F1-Score** | 73.58% | 75.58% | 🏆 YOLOv11 (+2.00%) |
| **slope_drain mAP@0.5** | 91.67% | 94.23% | 🏆 YOLOv11 (+2.56%) |
| **rock_toe mAP@0.5** | 86.68% | 88.31% | 🏆 YOLOv11 (+1.63%) |
| **vegetation mAP@0.5** | 59.63% | 70.12% | 🏆 YOLOv11 (+10.49%) |
| **toe_drain mAP@0.5** | 66.72% | 51.07% | 🏆 YOLOv8 (+15.65%) |
| **Model Parameters** | 11.1M | 9.4M | 🏆 YOLOv11 (-15.4%) |
| **GFLOPs** | 28.4 | 21.3 | 🏆 YOLOv11 (-25.0%) |
| **Inference Speed** | 23.6 FPS | 29.2 FPS | 🏆 YOLOv11 (+23.7%) |
| **Model Size** | 28.4 MB | 19.2 MB | 🏆 YOLOv11 (-32.4%) |
| **Training Time** | 15 min | 19 min | 🏆 YOLOv8 (-26.7%) |

---

## 🏆 Overall Winner Analysis

### By Category:

1. **Accuracy (mAP):** 🏆 **YOLOv8** - Slightly better overall mAP
2. **Precision:** 🏆 **YOLOv8** - Significantly better (75% vs 71%)
3. **Recall:** 🏆 **YOLOv11** - Significantly better (81% vs 72%)
4. **Efficiency:** 🏆 **YOLOv11** - Smaller, faster, fewer parameters
5. **Class Performance:** 🏆 **YOLOv11** - Better on 3 out of 4 classes
6. **Training Speed:** 🏆 **YOLOv8** - Faster training

### Final Verdict:

**🏆 YOLOv11-S is the better choice for this project** because:
- ✅ Better recall (finds more objects - critical for inspection)
- ✅ Better on 3 out of 4 classes (slope_drain, rock_toe, vegetation)
- ✅ More efficient (smaller model, faster inference)
- ✅ Better F1-score (balanced metric)
- ✅ Only slightly worse overall mAP (0.24% difference)

**However, YOLOv8-S is better if:**
- Precision is more important than recall
- toe_drain detection is critical
- Faster training is needed

---

## 📝 Notes

- Both models were trained with identical strategies for fair comparison
- Same dataset, same splits, same augmentation
- Performance differences are relatively small (<5% in most metrics)
- Both models are production-ready
- Choice depends on specific use case requirements

---

**Last Updated:** 2025-01-27  
**Recommendation:** YOLOv11-S for deployment (better recall + efficiency)

