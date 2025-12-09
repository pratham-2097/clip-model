# 🧪 Model Testing & Evaluation Guide

**Last Updated:** 2025-01-27  
**Best Model:** `runs/detect/finetune_phase/weights/best.pt`

---

## 📊 Current Model Performance Summary

### Overall Metrics (Validation Set)
- **mAP@0.5:** 72.1% - Good overall detection accuracy
- **mAP@[0.5:0.95]:** 47.6% - Moderate bounding box precision
- **Precision:** ~71% - About 7 in 10 detections are correct
- **Recall:** ~76% - Finds about 3 in 4 objects

### Per-Class Performance (on Validation)
| Class | mAP@0.5 | Status | Notes |
|-------|---------|--------|-------|
| **slope_drain** | 92.4% | ✅ Excellent | Best performing class |
| **rock_toe** | 84.0% | ✅ Good | Reliable detection |
| **toe_drain** | 55.9% | ⚠️ Moderate | Few training examples (only 3 in validation) |
| **vegetation** | 55.9% | ⚠️ Moderate | Improved from 47.7% after oversampling |

### Performance on Clear Images
When provided with **clear, well-lit images** similar to your training data:
- ✅ **Detects ~76% of all objects** (Recall)
- ✅ **~71% of detections are correct** (Precision)
- ✅ **Works best on `slope_drain` and `rock_toe`** (84-92% accuracy)
- ⚠️ **Struggles more with `toe_drain` and `vegetation`** (56% accuracy)

**Note:** Performance drops significantly on challenging test data (different lighting/angles), indicating the model needs more diverse training examples.

---

## 🧪 How to Test Your Model

### Option 1: Comprehensive Evaluation (Recommended)

Run full evaluation with detailed metrics:

```bash
cd yolov8_project
conda activate yolov8  # or your environment name
python scripts/evaluate_model.py
```

**Options:**
- `--split val` - Evaluate on validation set (default)
- `--split test` - Evaluate on test set (if available)
- `--conf 0.25` - Confidence threshold (default: 0.25)
- `--device mps` - Device (mps for Mac, cuda:0 for NVIDIA GPU, cpu for CPU)

**Example:**
```bash
# Evaluate on validation set with custom confidence
python scripts/evaluate_model.py --conf 0.3 --split val

# Evaluate on test set
python scripts/evaluate_model.py --split test
```

**Output:**
- Overall metrics (mAP, Precision, Recall)
- Per-class performance breakdown
- Performance assessment and recommendations
- Saved visualizations (confusion matrix, PR curves)
- Validation predictions with bounding boxes

---

### Option 2: Test on Single Image

Test the model on a single image:

```bash
python scripts/test_single_image.py --image path/to/your/image.jpg
```

**Options:**
- `--image PATH` - Path to image file
- `--conf 0.25` - Confidence threshold
- `--weights PATH` - Model weights (default: best.pt)

**Example:**
```bash
# Test a single image
python scripts/test_single_image.py --image test_images/my_image.jpg

# Test with lower confidence (catch more objects)
python scripts/test_single_image.py --image test_images/my_image.jpg --conf 0.15
```

**Output:**
- Annotated image with bounding boxes
- Detection details (class, confidence, coordinates)
- Summary statistics

---

### Option 3: Test on Folder of Images

Test multiple images at once:

```bash
python scripts/test_single_image.py --folder test_images
```

Or use the batch inference script:

```bash
python scripts/infer_on_folder.py --input_folder test_images --output_folder outputs/test_results
```

**Output:**
- All images annotated with detections
- Label files (YOLO format) with confidence scores
- Summary of detections per class

---

### Option 4: Quick Validation Check

Use YOLOv8's built-in validation:

```bash
yolo detect val \
  model=runs/detect/finetune_phase/weights/best.pt \
  data=data.yaml \
  device=mps \
  conf=0.25
```

---

## 📈 Understanding the Metrics

### Key Metrics Explained

1. **mAP@0.5 (Mean Average Precision at IoU=0.5)**
   - **What it means:** Overall detection quality when we're lenient about box placement
   - **Your score:** 72.1% - Good
   - **Target:** >75% for production

2. **mAP@[0.5:0.95] (Mean Average Precision averaged over IoU 0.5-0.95)**
   - **What it means:** How tight and accurate the bounding boxes are
   - **Your score:** 47.6% - Moderate
   - **Target:** >50% for production

3. **Precision**
   - **What it means:** Of all detections, how many are correct?
   - **Your score:** ~71% - Good
   - **Interpretation:** About 3 in 10 detections are false positives

4. **Recall**
   - **What it means:** Of all objects present, how many did we find?
   - **Your score:** ~76% - Good
   - **Interpretation:** About 1 in 4 objects are missed

### Per-Class Metrics

- **mAP@0.5 > 80%:** Excellent - Model is very reliable for this class
- **mAP@0.5 60-80%:** Good - Model works well but could improve
- **mAP@0.5 40-60%:** Moderate - Model struggles, needs more data
- **mAP@0.5 < 40%:** Poor - Model needs significant improvement

---

## 🎯 Testing Scenarios

### Scenario 1: Testing on Clear Images (Similar to Training)
**Expected Performance:** Good
- Use validation set or similar images
- Should see ~72% mAP@0.5
- `slope_drain` and `rock_toe` should perform best

### Scenario 2: Testing on Challenging Images (Different from Training)
**Expected Performance:** Lower
- Different lighting, angles, or conditions
- May see 20-40% recall (based on test dataset results)
- Model may miss objects or have false positives

### Scenario 3: Testing on Single Clear Image
**What to Look For:**
- ✅ All visible objects are detected
- ✅ Bounding boxes are tight around objects
- ✅ Confidence scores are >0.5 for clear detections
- ✅ Correct class labels

---

## 🔧 Adjusting Confidence Threshold

The confidence threshold controls how certain the model must be before making a detection:

- **High threshold (0.5-0.7):** Fewer detections, but more accurate
  - Use when: You want to minimize false positives
  - Trade-off: May miss some objects

- **Medium threshold (0.25-0.5):** Balanced (default: 0.25)
  - Use when: General use case
  - Trade-off: Some false positives, but catches most objects

- **Low threshold (0.1-0.25):** More detections, but more false positives
  - Use when: You want to catch all objects, even uncertain ones
  - Trade-off: More false positives to review

**Example:**
```bash
# Conservative (fewer false positives)
python scripts/test_single_image.py --image test.jpg --conf 0.5

# Balanced (default)
python scripts/test_single_image.py --image test.jpg --conf 0.25

# Aggressive (catch everything)
python scripts/test_single_image.py --image test.jpg --conf 0.15
```

---

## 📁 Where Results Are Saved

### Evaluation Results
- **Location:** `runs/detect/val/` (or custom path)
- **Files:**
  - `confusion_matrix.png` - Class confusion visualization
  - `PR_curve.png` - Precision-Recall curves
  - `val_batch*.jpg` - Validation images with predictions
  - `predictions.json` - Detailed JSON results

### Test Image Results
- **Location:** `outputs/test_results/`
- **Files:**
  - `*.jpg` - Annotated images with bounding boxes
  - `labels/*.txt` - YOLO format labels with confidence

---

## 🚀 Quick Start Examples

### 1. Quick Model Check
```bash
cd yolov8_project
python scripts/evaluate_model.py
```

### 2. Test Your Own Image
```bash
# Place your image in test_images/ folder first
python scripts/test_single_image.py --image test_images/my_photo.jpg
```

### 3. Batch Test Multiple Images
```bash
python scripts/test_single_image.py --folder test_images
```

### 4. Full Evaluation with Custom Settings
```bash
python scripts/evaluate_model.py --conf 0.3 --split val --device mps
```

---

## 💡 Tips for Best Results

1. **Use Clear Images:** Model works best on well-lit, clear images similar to training data
2. **Check Confidence Scores:** Detections with confidence >0.5 are usually reliable
3. **Review False Positives:** If you see many false positives, increase confidence threshold
4. **Review Missed Objects:** If objects are missed, decrease confidence threshold
5. **Compare with Ground Truth:** Use validation set to see expected performance
6. **Test on Diverse Images:** Test on various conditions to understand model limitations

---

## 📊 Expected Performance Summary

| Scenario | Expected mAP@0.5 | Expected Recall | Notes |
|----------|------------------|-----------------|-------|
| **Clear images (similar to training)** | 70-75% | 75-80% | Good performance |
| **Challenging images (different conditions)** | 20-40% | 20-40% | Significant drop |
| **Best class (slope_drain)** | 90-95% | 85-90% | Excellent |
| **Worst class (toe_drain)** | 50-60% | 50-60% | Needs improvement |

---

## 🔍 Troubleshooting

### Problem: No detections on test image
**Solutions:**
- Lower confidence threshold: `--conf 0.15`
- Check if image is similar to training data
- Verify model weights are loaded correctly

### Problem: Too many false positives
**Solutions:**
- Increase confidence threshold: `--conf 0.4`
- Check if image characteristics match training data
- Review confusion matrix for class confusion

### Problem: Missing obvious objects
**Solutions:**
- Lower confidence threshold: `--conf 0.15`
- Check if object class has enough training examples
- Verify image quality and lighting

---

## 📝 Next Steps

1. ✅ **Test on validation set** - Verify current performance
2. ✅ **Test on your own images** - See how it works in practice
3. ⏳ **Collect more training data** - Especially for `toe_drain` and `vegetation`
4. ⏳ **Retrain with expanded dataset** - Improve generalization
5. ⏳ **Test on diverse conditions** - Understand model limitations

---

**For questions or issues, refer to:**
- `progress_tracking.md` - Full experiment history
- `yolov8_full_project_guide.md` - Complete project guide




