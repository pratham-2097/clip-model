# Infrastructure Inspection AI - Multi-Stage Object Detection System

**A comprehensive computer vision system for infrastructure site inspection using YOLO object detection and Vision-Language Models for conditional classification.**

---

## 🎯 Project Overview

This project implements a **multi-stage AI system** for automated infrastructure inspection that:

1. **Stage 1: Object Detection** ✅ **COMPLETE**
   - Detects key infrastructure components: `rock_toe`, `slope_drain`, `toe_drain`, `vegetation`
   - Achieves **82.3% mAP@0.5** (surpasses YOLOv8-S and YOLOv11-S baselines)
   - Real-time inference with Streamlit web interface

2. **Stage 2: Conditional Classification** 🚧 **IN PROGRESS**
   - Classifies object conditions: `damaged`, `blocked`, `normal`
   - Uses Vision-Language Models (VLMs) for contextual reasoning
   - Leverages spatial relationships and image context

3. **Stage 3: Deployment** ⏳ **PLANNED**
   - Quantized models for Nvidia A30 GPU server
   - Production-ready inference pipeline
   - Optimized for real-world deployment

---

## 🏆 Key Achievements

### Stage 1 Results
- **Best Model:** YOLOv11 Expanded (Reduced Augmentation)
- **Performance:** 82.3% mAP@0.5, 53.7% mAP@[0.5:0.95]
- **Improvement:** +6.13% over YOLOv8-S baseline, +6.37% over YOLOv11-S baseline
- **Precision:** 85.1% (excellent false positive control)
- **Per-Class Performance:**
  - `slope_drain`: 94.9% mAP@0.5
  - `toe_drain`: 99.5% mAP@0.5
  - `rock_toe`: 82.3% mAP@0.5
  - `vegetation`: 52.4% mAP@0.5

---

## 📁 Project Structure

```
CLIP model/
├── yolov8_project/
│   ├── ui/                          # Streamlit web interface
│   │   ├── app.py                   # Main UI application
│   │   ├── inference.py             # Model loading and inference
│   │   └── start_server.sh          # Server startup script
│   ├── scripts/                      # Training and utility scripts
│   │   ├── consolidate_classes.py   # Dataset consolidation
│   │   ├── train_yolov11_expanded.py # Model training
│   │   └── ...                      # Additional utilities
│   ├── STAGE1_COMPLETE.md           # Stage 1 completion report
│   ├── FINAL_MODEL_COMPARISON.md    # Model comparison results
│   └── PROJECT_SUMMARY.md            # Detailed project documentation
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Apple Silicon Mac (MPS) or NVIDIA GPU (CUDA) or CPU
- 8GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd "CLIP model"
   ```

2. **Create virtual environment:**
   ```bash
   cd yolov8_project
   python3 -m venv .venv
   source .venv/bin/activate  # On Mac/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements_ui.txt
   pip install ultralytics torch torchvision
   ```

### Running the Web UI

1. **Start the Streamlit server:**
   ```bash
   cd yolov8_project
   ./ui/start_server.sh
   ```

2. **Access the UI:**
   - Open your browser to: `http://localhost:8501`
   - Upload an image
   - Select model (YOLOv11-Best recommended)
   - Run detection

### Using the Best Model

The best model (82.3% mAP@0.5) is automatically selected as default in the UI. Model path:
```
runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt
```

---

## 📊 Dataset

### Stage 1 Dataset
- **Total Images:** 375 (merged dataset)
- **Total Instances:** 1,605 objects
- **Classes:** 4 (rock_toe, slope_drain, toe_drain, vegetation)
- **Splits:** Train (320), Val (30), Test (25)
- **Format:** YOLO format annotations

### Class Distribution
- `slope_drain`: 586 instances (36.5%)
- `rock_toe`: 561 instances (35.0%)
- `vegetation`: 257 instances (16.0%)
- `toe_drain`: 201 instances (12.5%)

---

## 🔧 Model Training

### Best Model Configuration

**Model:** YOLOv11 Expanded (Reduced Augmentation)
- **Architecture:** YOLOv11-S
- **Image Size:** 640×640
- **Batch Size:** 8
- **Learning Rate:** 0.00035 (cosine schedule)
- **Optimizer:** AdamW
- **Augmentation:** Reduced (mosaic=0.3, mixup=0.0)
- **Epochs:** 100 (best at epoch 35)

### Training Scripts

- `scripts/train_yolov11_expanded.py` - Main training script
- `scripts/consolidate_classes.py` - Dataset consolidation
- `scripts/analyze_merged_dataset.py` - Dataset analysis

---

## 📈 Performance Metrics

### Comparison with Baselines

| Model | mAP@0.5 | mAP@[0.5:0.95] | Precision | Recall |
|-------|---------|----------------|-----------|--------|
| YOLOv8-S (Baseline) | 76.17% | 51.53% | 75.00% | 72.22% |
| YOLOv11-S (Baseline) | 75.93% | 51.11% | 70.87% | 80.75% |
| **🏆 Best Model** | **82.30%** | **53.70%** | **85.10%** | **75.90%** |

### Improvements
- **+6.13%** mAP@0.5 over YOLOv8-S
- **+6.37%** mAP@0.5 over YOLOv11-S
- **+2.17%** mAP@[0.5:0.95] over YOLOv8-S
- **+10.10%** Precision over YOLOv8-S

---

## 🛠️ Technologies Used

- **Object Detection:** Ultralytics YOLOv11
- **Web Framework:** Streamlit
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV, PIL
- **Device Support:** MPS (Apple Silicon), CUDA (NVIDIA), CPU

---

## 📝 Documentation

- **STAGE1_COMPLETE.md** - Complete Stage 1 results and analysis
- **FINAL_MODEL_COMPARISON.md** - Detailed model comparison
- **PROJECT_SUMMARY.md** - Comprehensive project documentation
- **MODEL_TESTING_GUIDE.md** - Model evaluation guide
- **UI_UPDATE_SUMMARY.md** - UI integration documentation

---

## 🔄 Project Status

### ✅ Stage 1: Object Detection - COMPLETE
- [x] Dataset preparation and consolidation
- [x] Model training and optimization
- [x] Baseline comparison (YOLOv8-S, YOLOv11-S)
- [x] Best model identification (82.3% mAP@0.5)
- [x] Web UI integration
- [x] Documentation and results

### 🚧 Stage 2: Conditional Classification - IN PROGRESS
- [ ] Vision-Language Model research and selection
- [ ] Conditional class dataset preparation
- [ ] VLM fine-tuning for conditional classification
- [ ] Integration with Stage 1 detection
- [ ] End-to-end testing

### ⏳ Stage 3: Deployment - PLANNED
- [ ] Model quantization (INT8/INT4)
- [ ] Nvidia A30 server optimization
- [ ] Production pipeline deployment
- [ ] Performance benchmarking

---

## 🤝 Contributing

This is a research project. For questions or contributions, please contact the project maintainer.

---

## 📄 License

[Specify your license here]

---

## 🙏 Acknowledgments

- Ultralytics for YOLOv11 framework
- Roboflow for dataset management tools
- Streamlit for web framework

---

## 📧 Contact

For questions or inquiries about this project, please contact the project maintainer.

---

**Last Updated:** 2025-01-27  
**Version:** 1.0.0 (Stage 1 Complete)

