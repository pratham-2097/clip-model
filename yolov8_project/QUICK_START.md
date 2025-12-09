# ⚡ Quick Start: Run Evaluation in 3 Steps

## 🎯 The Simplest Way to Run

Copy and paste these commands **one at a time** into your Cursor terminal:

### Step 1: Go to the project folder
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
```

### Step 2: Activate conda environment (if you have one)
```bash
conda activate yolov8
```

**OR if you don't have conda, install ultralytics:**
```bash
pip3 install ultralytics
```

### Step 3: Run the evaluation
```bash
python3 scripts/evaluate_model.py
```

---

## 🔍 If You Get Errors

### Error: "No module named 'ultralytics'"

**Fix:**
```bash
# Option 1: Install in current Python
pip3 install ultralytics

# Option 2: Or use conda
conda activate yolov8
pip install ultralytics
```

### Error: "Model weights not found"

**Fix:**
```bash
# Check if model exists
ls runs/detect/finetune_phase/weights/best.pt

# If it doesn't exist, find your model:
find runs -name "*.pt" | head -5
```

### Error: "Data config not found"

**Fix:**
```bash
# Make sure you're in the right directory
pwd
# Should show: .../yolov8_project

# Check if data.yaml exists
ls data.yaml
```

---

## ✅ Success Checklist

Before running, verify:
- [ ] You're in `yolov8_project` folder (`pwd` shows it)
- [ ] Conda environment is activated OR ultralytics is installed
- [ ] Model file exists: `runs/detect/finetune_phase/weights/best.pt`
- [ ] Data config exists: `data.yaml`

---

## 🚀 One-Line Test

Test if everything is ready:
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project" && python3 -c "from ultralytics import YOLO; print('✅ Ready!')"
```

If this works, you can run:
```bash
python3 scripts/evaluate_model.py
```




