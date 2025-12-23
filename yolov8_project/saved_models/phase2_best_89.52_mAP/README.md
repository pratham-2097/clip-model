# 🎯 Phase 2 Best Model - Quick Reference

## 📁 Files in This Directory

- **best.pt** - Best model weights (87.20% mAP@0.5)
- **training_results.csv** - Complete training history (21 epochs)
- **training_config.yaml** - Exact training configuration
- **per_class_metrics.json** - Detailed per-class performance
- **REPRODUCTION_GUIDE.md** - Complete step-by-step guide
- **README.md** - This file

## 🚀 Quick Start

### Load Model
```python
from ultralytics import YOLO
model = YOLO('saved_models/phase2_best_89.52_mAP/best.pt')
results = model('image.jpg')
```

### Performance
- **mAP@0.5:** 87.20%
- **Best Epoch:** 6 (89.52% mAP@0.5)
- **toe_drain:** 85.92% (was 38.46% baseline)

## 📖 Full Documentation
See `REPRODUCTION_GUIDE.md` for complete instructions.

