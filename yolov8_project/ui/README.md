# YOLO Object Detection Web UI

A Streamlit-based web interface for running YOLOv8 and YOLOv11 object detection models on uploaded images.

## Features

- 🖼️ **Image Upload**: Upload images in JPG, PNG, or BMP format
- 🔄 **Model Selection**: Choose between YOLOv11-Best (recommended), YOLOv8, and YOLOv11 models
- ⚙️ **Configurable Confidence**: Adjust confidence threshold with a slider
- 📊 **Detailed Results**: View annotated images, per-object detections, and per-class summaries
- 🎯 **Accuracy Metrics**: See confidence scores for each detection and class-level statistics

## Installation

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install streamlit ultralytics pillow numpy
```

Or install from requirements (if available):

```bash
pip install -r requirements_ui.txt
```

### Model Weights

Ensure the trained model weights are available at:
- **YOLOv11-Best** (Recommended): `runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt` (82.3% mAP@0.5)
- **YOLOv8**: `runs/detect/finetune_phase/weights/best.pt`
- **YOLOv11**: `runs/detect/yolov11_finetune_phase/weights/best.pt`

## Usage

### Running the Application

1. Navigate to the project directory:
   ```bash
   cd yolov8_project
   ```

2. Run the Streamlit app:
   ```bash
   python3 -m streamlit run ui/app.py
   ```
   
   **Note:** If `streamlit` command is not found, use `python3 -m streamlit` instead.

3. The application will open in your default web browser at `http://localhost:8501`

### Using the Interface

1. **Upload an Image**: Click "Browse files" or drag and drop an image
2. **Select Model**: Choose YOLOv11-Best (recommended), YOLOv8, or YOLOv11 from the sidebar
3. **Adjust Confidence**: Use the slider to set the confidence threshold (default: 0.25)
4. **Run Detection**: Click the "🚀 Run Detection" button
5. **View Results**: 
   - See the annotated image with bounding boxes
   - Review per-object detection details (class, confidence, coordinates)
   - Check per-class summary statistics

## Detected Classes

The models detect the following infrastructure components:

- **rock_toe**: Rock toe structures
- **slope_drain**: Slope drainage structures
- **toe_drain**: Toe drainage structures
- **vegetation**: Vegetation areas

## Results Display

### Per-Object Detections
For each detected object, you'll see:
- Object number
- Class name
- Confidence percentage
- Bounding box coordinates (x1, y1) → (x2, y2)

### Per-Class Summary
For each detected class, you'll see:
- Total count of detections
- Average confidence score
- Minimum confidence score
- Maximum confidence score

## Troubleshooting

### Model Not Loading
- Ensure model weights exist at the expected paths
- Check that you're running from the `yolov8_project` directory
- Verify model files are not corrupted

### No Detections Found
- Try lowering the confidence threshold
- Ensure the image contains objects similar to training data
- Check image quality and lighting conditions

### Import Errors
- Make sure all dependencies are installed
- Verify you're using Python 3.7+
- Check that the `ui` directory structure is correct

## File Structure

```
yolov8_project/
├── ui/
│   ├── app.py          # Main Streamlit application
│   ├── inference.py    # Inference utility functions
│   └── README.md       # This file
├── runs/
│   └── detect/
│       ├── finetune_phase/weights/best.pt      # YOLOv8 model
│       └── yolov11_finetune_phase/weights/best.pt  # YOLOv11 model
└── ...
```

## Technical Details

- **Framework**: Streamlit
- **ML Framework**: Ultralytics YOLO
- **Device Support**: Auto-detects MPS (Mac), CUDA (NVIDIA), or CPU
- **Image Formats**: JPG, JPEG, PNG, BMP
- **Default Confidence**: 0.25

## Notes

- Models are cached after first load for faster subsequent runs
- Results are stored in session state and persist until cleared
- The application automatically handles image format conversion (RGBA → RGB)

