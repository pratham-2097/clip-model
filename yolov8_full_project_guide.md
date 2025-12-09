
# 🚀 YOLOv8 Object Detection Project — Full Setup, Training & Deployment Guide (M2 Max)

This guide walks through everything required for setting up, training, and deploying your YOLOv8 object detection model on macOS with Apple Silicon (M2 Max). It also includes troubleshooting for Miniforge/conda setup errors and is formatted so you can paste it directly into Cursor to auto-run tasks.

---

## 🧠 Project Understanding & Goals

You are building a **multi-modal object detection pipeline**. The goal is to detect key site components (like slope drains, rock toes, vegetation, etc.) and later classify their condition (damaged, blocked, etc.) through a multimodal classifier (CLIP/FLORENCE/BLIP-2).

For now, we are in **Phase 1: Object Detection**, using a small 120-image dataset (Roboflow export) to benchmark detection model performance before scaling to 12–15k images later.

---

## ⚙️ Phase 1 Objective

Build and deploy a **baseline YOLOv8 object detector** locally on your M2 Max chip.

| Stage | Goal | Model |
|--------|------|-------|
| Stage 1 | Train YOLOv8-S detector (object-level bounding boxes) | YOLOv8-S |
| Stage 2 | Integrate VLM classifier for conditional reasoning | CLIP / Florence / BLIP-2 |
| Stage 3 | Quantize for A30/Jetson deployment | YOLOv11-N (INT8) |

---

## 🧩 Environment Setup (Troubleshooting Included)

### 1️⃣ Step 1 — Install Miniforge (Apple Silicon-friendly Python/Conda)

**Troubleshooting your issue:**
> You installed Miniforge but got `zsh: command not found: conda`

That happened because you selected **“no”** when asked if Miniforge should update your shell profile.

✅ **Fix:** Run the command below exactly as written:

```bash
eval "$(/Users/prathamprabhu/miniforge3/bin/conda shell.zsh hook)"
```

Then check if conda works:
```bash
conda --version
```

If it shows a version, you’re good. Otherwise, initialize Conda permanently:
```bash
/Users/prathamprabhu/miniforge3/bin/conda init zsh
exec zsh
```

Now `conda` will be recognized automatically in new terminals.

---

### 2️⃣ Step 2 — Create and activate environment

```bash
conda create -n yolov8 python=3.10 -y
conda activate yolov8
```

You should now see `(yolov8)` before your terminal prompt.

If not, manually activate:
```bash
source ~/miniforge3/bin/activate yolov8
```

---

### 3️⃣ Step 3 — Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install ultralytics onnxruntime opencv-python pillow pyyaml matplotlib
```

Verify PyTorch MPS availability:

```bash
python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
PY
```

If both print `True`, your M2 GPU is ready to train 🚀

---

## 🗂️ Project Folder Structure

```
yolov8_project/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
├── data.yaml
├── scripts/
│   ├── train.sh
│   ├── export.sh
│   └── infer_on_folder.py
└── runs/  # YOLO training logs here
```

### Example `data.yaml`
```yaml
train: ./dataset/images/train
val: ./dataset/images/val
test: ./dataset/images/test

nc: 4
names: ['slope_drain','rock_toe','vegetation','blocked']
```

---

## 🧠 Step-by-Step Commands (for Cursor Automation)

### 🏋️‍♂️ Train the YOLOv8-S Model

`scripts/train.sh`
```bash
#!/bin/bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=8 device=mps
```

Then run:
```bash
chmod +x scripts/train.sh
./scripts/train.sh
```

This will train YOLOv8-S using your M2 GPU.

---

### 🧪 Validate Model

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml device=mps
```

---

### 📦 Export to ONNX (for deployment)

`scripts/export.sh`
```bash
#!/bin/bash
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=12
```

Run it:
```bash
bash scripts/export.sh
```

Now you’ll have `best.onnx` (usable for ONNXRuntime or TensorRT).

---

### 🧾 Run Inference on Test Images

`scripts/infer_on_folder.py`
```python
from ultralytics import YOLO
import argparse, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="runs/detect/train/weights/best.pt")
parser.add_argument("--input_folder", default="dataset/images/test")
parser.add_argument("--output_folder", default="outputs")
args = parser.parse_args()

model = YOLO(args.weights)
Path(args.output_folder).mkdir(exist_ok=True)
model.predict(source=args.input_folder, save=True, project=args.output_folder, name="results", device="mps")
```

Run:
```bash
python scripts/infer_on_folder.py
```

Outputs with bounding boxes will be saved under `outputs/results/`.

---

## 🧰 Optional Tools

| Tool | Use |
|------|-----|
| **Roboflow** | Annotating and exporting datasets |
| **Ultralytics CLI** | Simplifies training & exporting |
| **TensorBoard** | Optional visualization (`pip install tensorboard`) |
| **Docker** | For deploying model inference in isolated container |

---

## 🛠️ Troubleshooting Common Mac Issues

| Problem | Fix |
|----------|-----|
| `zsh: command not found: conda` | Run `eval "$(/Users/prathamprabhu/miniforge3/bin/conda shell.zsh hook)"` |
| `pip: command not found` | Activate environment first (`source ~/miniforge3/bin/activate`) |
| `torch.backends.mps.is_available() == False` | Update macOS + install latest PyTorch (≥2.3.0) |
| `ultralytics command not found` | Run `pip install -U ultralytics` again |
| Training very slow | Lower `imgsz` or `epochs`, or use A30 GPU remotely |

---

## ✅ Final Deliverables for Cursor

When you paste this document into Cursor, it can:
- Set up your environment automatically.
- Install PyTorch and Ultralytics.
- Run YOLOv8-S training on your dataset.
- Export and visualize detection outputs.

---

### 💾 Project Summary

| Stage | Task | Model | Output |
|--------|------|--------|---------|
| 1 | Object Detection | YOLOv8-S | Bounding boxes |
| 2 | Conditional Classification | CLIP / Florence | Condition reasoning |
| 3 | Quantized Deployment | YOLOv11-N | INT8 optimized model |

---

**Author:** Pratham Prabhu  
**System:** MacBook Pro M2 Max  
**Tools:** Conda, PyTorch (MPS), Ultralytics, ONNXRuntime

---

## 🚀 YOLOv8 Fine-Tuning Plan (Pre-Expansion Stage)

### Goal
Improve the current YOLOv8-S object detection model’s stability and accuracy using the existing dataset (≈120 images), without augmentation, before adding the remaining annotated samples.

### 🧠 Project Context
- Current dataset size: 120 images
- Remaining unannotated: 15–20 images (to add later)
- Current metrics:
  - mAP@0.5 → 0.747 ✅
  - mAP@[0.5:0.95] → 0.434 ⚪
- Issue: class imbalance
  - slope_drain → many examples (dominant)
  - rock_toe → good
  - toe_drain → very few (~3)
  - vegetation → few (~9)

This stage focuses on data consistency and training refinement to stabilize metrics and prepare for large-scale dataset expansion.

### ⚙️ Step 1 — Clean & Validate Annotations
**Tasks**
- Open the 17 ambiguous images with overlapping boxes.
- Verify each object has its own bounding box and correct class.
- Keep overlaps; YOLO handles them natively.
- If a box belongs to two visible classes (e.g., vegetation over rock_toe), label both separately.
- Save corrected annotations.

### 🗂️ Step 2 — Dataset Structure Verification
Ensure directory structure follows:

```
yolov8_project/
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
└── data.yaml
```

Check that every image in `dataset/images/train` has a matching `.txt` in `dataset/labels/train`.

### ⚖️ Step 3 — Handle Class Imbalance (No Augmentation)
#### 🔁 3.1 Duplicate Minority-Class Images (Manual Oversampling)
Duplicate images belonging to under-represented classes (toe_drain, vegetation) 3–6× to balance class frequencies.

Create `scripts/duplicate_minority.py`:

```python
from pathlib import Path
import shutil

train_imgs = Path("dataset/images/train")
train_lbls = Path("dataset/labels/train")
target = 20  # minimum desired count per class

counts, files_by_class = {}, {}
for lbl in train_lbls.glob("*.txt"):
    classes = set()
    for line in lbl.read_text().strip().splitlines():
        if not line:
            continue
        c = int(line.split()[0])
        classes.add(c)
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
        files_by_class.setdefault(c, []).append(lbl.stem)

for cls, cnt in counts.items():
    need = max(0, target - cnt)
    if need > 0:
        srcs = files_by_class[cls]
        for i in range(need):
            src = srcs[i % len(srcs)]
            src_img = train_imgs / f"{src}.jpg"
            src_lbl = train_lbls / f"{src}.txt"
            dst_img = train_imgs / f"{src}_dup{i}.jpg"
            dst_lbl = train_lbls / f"{src}_dup{i}.txt"
            shutil.copy(src_img, dst_img)
            shutil.copy(src_lbl, dst_lbl)
```

Run it:

```bash
python scripts/duplicate_minority.py
```

### 🧩 Step 4 — Freeze → Unfreeze Training Strategy
#### 🔒 Phase A: Freeze Backbone (Learn detection heads first)

```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=15 imgsz=640 batch=8 device=mps freeze=10 lr0=0.002 optimizer=SGD
```

#### 🔓 Phase B: Unfreeze & Fine-Tune

```bash
yolo detect train data=data.yaml model=runs/detect/train/weights/last.pt epochs=50 imgsz=640 batch=8 device=mps lr0=0.0005 optimizer=AdamW
```

### 🔧 Step 5 — Adjust Learning & Loss Weights
Create `hyp.yaml` in project root:

```yaml
# Custom hyperparameters for small-data fine-tuning
lr0: 0.0005
optimizer: AdamW
box: 0.05
cls: 0.7        # emphasize classification
dfl: 1.0
fl_gamma: 2.0   # focal loss gamma for class imbalance
```

Train with the custom hyp:

```bash
yolo detect train data=data.yaml model=yolov8s.pt hyp=hyp.yaml epochs=60 imgsz=640 batch=8 device=mps
```

### 🧪 Step 6 — Re-Validate Model

```bash
yolo detect val model=runs/detect/train/weights/best.pt data=data.yaml device=mps
```

Record:
- mAP50
- mAP50-95
- Per-class AP (focus on toe_drain and vegetation)

### 📊 Step 7 — Analyze Results
Inspect:
```
runs/detect/train/
├── results.png          # Loss curves
├── confusion_matrix.png # Class confusion
└── results.csv          # Metric logs
```

Look for:
- Steady loss decrease without spikes
- Higher recall for minority classes
- Smaller gap between mAP50 and mAP50-95

### 🧠 Step 8 — (Optional) Quick Cross-Validation

```bash
for i in 0 1 2; do
  yolo detect train data=data_fold${i}.yaml model=yolov8s.pt epochs=40 imgsz=640 batch=8 device=mps
done
```

Average metrics across folds to confirm generalization.

### ✅ Step 9 — Save Artifacts
- Copy `best.pt` to `models/yolov8s_finetuned.pt`.
- Retain `results.png` and `results.csv` for documentation.

### 🔍 Step 10 — Interpret Outcomes

| Metric         | Target | Expected Effect          |
|----------------|--------|---------------------------|
| mAP@0.5        | ≥ 0.78 | More stable detection     |
| mAP@[0.5:0.95] | ≥ 0.47 | Tighter bounding boxes    |
| Class variance | ↓      | Balanced performance      |

If metrics plateau, wait until the new 15–20 images are added, then retrain with the expanded dataset.

### ⚡ Step 11 — After Adding New Annotated Images
- Merge new images into the train split.
- Run `yolo detect val ...` to verify label consistency.
- Optionally retrain using the same fine-tuning workflow to integrate the new distribution.

### ⚠️ Notes on Overlaps
- Keep overlapping boxes; they increase realism.
- Ensure annotations stay consistent—no duplicate or missing labels.
- Never merge distinct objects into a single bounding box.

### 🧾 Summary of Next Actions

| Priority | Action                  | Command/Method                 |
|----------|-------------------------|--------------------------------|
| 🔹 1     | Verify + fix annotations| Manual (Roboflow or LabelImg)  |
| 🔹 2     | Run duplication script  | `python scripts/duplicate_minority.py` |
| 🔹 3     | Freeze–unfreeze training| See Step 4                     |
| 🔹 4     | Re-validate              | `yolo detect val ...`          |
| 🔹 5     | Log metrics              | Inspect `results.csv`          |
| 🔹 6     | Add new images later     | Merge and retrain              |

### 🧩 End Goal
Following these steps yields:
- A clean, balanced, non-augmented dataset
- A stable fine-tuned model (`best.pt`) ready for integration
- A reproducible setup for scaling to 15,000 images

**Author:** Pratham Vivek Prabhu  
**System:** MacBook Pro M2 Max  
**Framework:** PyTorch (MPS) + Ultralytics YOLOv8

