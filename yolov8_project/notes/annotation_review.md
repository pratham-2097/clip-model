## Annotation Review — YOLOv8 Fine-Tuning (Pre-Expansion)

Date: 2025-11-12

### Summary
- Parsed 120 label files (train + valid) exported from Roboflow.
- Labels are in YOLO polygon/segmentation format (class id + polygon vertices). We will keep polygons as-is for now; Ultralytics handles them, but any downstream tooling expecting 5-value bounding boxes should convert first.
- Computed class distribution:
  - Class 0 (`rock_toe`): 106 instances
  - Class 1 (`slope_drain`): 138 instances
  - Class 2 (`toe_drain`): 10 instances
  - Class 3 (`vegetation`): 59 instances
- Identified four files with polygon pairs whose bounding boxes overlap with IoU > 0.3 (potential multi-object regions to double-check manually).

### Files to Review Manually
| File Stem | Classes | Max IoU | Notes |
|-----------|---------|---------|-------|
| `0ff04a56518786cf_jpg.rf.1a2d320e9eec125b6cd221064872b6d8` | 0 vs 0 | 0.87 | Two near-identical polygons — verify they truly represent distinct objects. |
| `DJI_0003_W_JPG.rf.2c26bda41bf678442b600dbe93c19b8a` | 1 vs 1 | 0.53 | Overlapping slope drain annotations — ensure separation is intentional. |
| `DJI_0005_W_JPG.rf.b6a8ddafedb638ed445a69a438fe8e5b` | 1 vs 1 | 0.43 | Possible duplicate coverage of same structure. |
| `0b172076bc7fc52f_jpg.rf.9f1fc09a3285df1a037d3ab713ba570c` | 1 vs 1 | 0.30 | Check for partial overlap vs. separate adjacent objects. |

### Next Manual Actions
1. Open each image + label pair in Roboflow/LabelImg.
2. Confirm overlapping polygons correspond to distinct objects; adjust or split as needed.
3. Ensure no missing classes in scenes where minority classes appear.

### Follow-Up
Once manual review is complete, update the dataset export (or modify local labels) before proceeding to oversampling.



