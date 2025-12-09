# Test Images Folder

Place your new test images here to evaluate the model's performance.

## How to Test

### Option 1: Test a Single Image
```bash
cd yolov8_project
conda activate yolov8
python scripts/test_single_image.py --image path/to/your/image.jpg
```

### Option 2: Test All Images in This Folder
```bash
cd yolov8_project
conda activate yolov8
python scripts/test_single_image.py --folder test_images
```

### Option 3: Test on Validation Set (for comparison)
```bash
cd yolov8_project
conda activate yolov8
python scripts/infer_on_folder.py --input_folder dataset/images/val
```

## Results

Results will be saved to:
- **Annotated images:** `outputs/test_results/*.jpg` (images with bounding boxes drawn)
- **Label files:** `outputs/test_results/labels/*.txt` (YOLO format labels with confidence scores)

## What to Look For

1. **Detection Accuracy:** Are objects correctly identified?
2. **Confidence Scores:** Higher is better (typically >0.5 is good)
3. **Bounding Box Precision:** Are boxes tight around objects?
4. **False Positives:** Are there detections where there shouldn't be?
5. **False Negatives:** Are objects missed?

## Model Performance Expectations

Based on training:
- **slope_drain:** ~92% accuracy (excellent)
- **rock_toe:** ~84% accuracy (good)
- **toe_drain:** ~56% accuracy (needs improvement - few examples)
- **vegetation:** ~56% accuracy (improved from 48%)

## Tips

- Use images similar to training data (aerial/drone imagery of infrastructure)
- Model works best on clear, well-lit images
- Confidence threshold is set to 0.25 (adjust with `--conf` flag if needed)


