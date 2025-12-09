# 🚀 Step-by-Step: How to Run Model Evaluation in Cursor Terminal

**This guide will help you run the evaluation script and fix common errors.**

---

## 📋 Prerequisites Check

First, let's verify your setup:

### Step 1: Open Terminal in Cursor
1. In Cursor, press `` Ctrl+` `` (backtick) or go to **Terminal → New Terminal**
2. You should see a terminal at the bottom of the screen

### Step 2: Check Your Current Directory
```bash
pwd
```

**Expected output:** `/Users/prathamprabhu/Desktop/CLIP model` or similar

If you're not in the right directory, navigate:
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
```

### Step 3: Check if Conda Environment Exists
```bash
conda env list
```

Look for an environment named `yolov8` or similar. If you see it, note the name.

---

## 🔧 Step-by-Step Instructions

### **STEP 1: Navigate to Project Directory**

```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
```

Verify you're in the right place:
```bash
ls -la
```

You should see:
- `data.yaml`
- `scripts/` folder
- `runs/` folder
- `dataset/` folder

---

### **STEP 2: Activate Your Conda Environment**

**Option A: If you have a `yolov8` environment:**
```bash
conda activate yolov8
```

**Option B: If your environment has a different name:**
```bash
conda activate <your-environment-name>
```

**Option C: If you don't have a conda environment, create one:**
```bash
# Create new environment
conda create -n yolov8 python=3.10 -y

# Activate it
conda activate yolov8

# Install ultralytics
pip install ultralytics
```

**Verify activation worked:**
```bash
which python
```

You should see a path with your environment name in it.

---

### **STEP 3: Verify Ultralytics is Installed**

```bash
python -c "import ultralytics; print('✅ Ultralytics installed successfully')"
```

**If you get an error:**
```bash
pip install ultralytics
```

Then try the verification again.

---

### **STEP 4: Check Model File Exists**

```bash
ls -la runs/detect/finetune_phase/weights/best.pt
```

**Expected output:** File details should appear. If you see "No such file", the model path is wrong.

---

### **STEP 5: Check Data Config Exists**

```bash
ls -la data.yaml
```

**Expected output:** File details should appear.

---

### **STEP 6: Run the Evaluation Script**

**Basic command:**
```bash
python scripts/evaluate_model.py
```

**If you get a "command not found" error, try:**
```bash
python3 scripts/evaluate_model.py
```

**If you get a permission error:**
```bash
chmod +x scripts/evaluate_model.py
python scripts/evaluate_model.py
```

---

## ❌ Common Errors and Solutions

### **Error 1: `ModuleNotFoundError: No module named 'ultralytics'`**

**Solution:**
```bash
# Make sure your conda environment is activated
conda activate yolov8

# Install ultralytics
pip install ultralytics

# Verify installation
python -c "import ultralytics; print('OK')"
```

---

### **Error 2: `❌ Error: Model weights not found`**

**Solution:**
```bash
# Check if the file exists
ls -la runs/detect/finetune_phase/weights/best.pt

# If it doesn't exist, check what model files you have
find runs -name "*.pt" -type f
```

If the model is in a different location, specify it:
```bash
python scripts/evaluate_model.py --weights path/to/your/model.pt
```

---

### **Error 3: `❌ Error: Data config not found`**

**Solution:**
```bash
# Make sure you're in the yolov8_project directory
pwd

# Check if data.yaml exists
ls -la data.yaml

# If it's in a different location, specify it:
python scripts/evaluate_model.py --data path/to/data.yaml
```

---

### **Error 4: `RuntimeError: MPS backend not available`**

**Solution:**
If you're on Mac and MPS (Metal) isn't working, use CPU instead:
```bash
python scripts/evaluate_model.py --device cpu
```

Or if you have CUDA:
```bash
python scripts/evaluate_model.py --device cuda:0
```

---

### **Error 5: `FileNotFoundError: dataset/images/val`**

**Solution:**
Check if your validation images exist:
```bash
ls -la dataset/images/val/
```

If the folder is empty or doesn't exist, you might need to use a different split or check your `data.yaml` file.

---

### **Error 6: `Permission denied`**

**Solution:**
```bash
chmod +x scripts/evaluate_model.py
python scripts/evaluate_model.py
```

---

## 🎯 Complete Working Example

Here's the complete sequence that should work:

```bash
# 1. Navigate to project
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"

# 2. Activate environment
conda activate yolov8

# 3. Verify ultralytics
python -c "import ultralytics; print('OK')"

# 4. Run evaluation
python scripts/evaluate_model.py
```

---

## 🔍 Alternative: Run with Full Paths

If you're still having issues, try running with explicit paths:

```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
conda activate yolov8
python scripts/evaluate_model.py \
  --weights runs/detect/finetune_phase/weights/best.pt \
  --data data.yaml \
  --device mps \
  --conf 0.25
```

---

## 📊 What Success Looks Like

When the script runs successfully, you should see:

```
================================================================================
MODEL EVALUATION
================================================================================
📦 Loading model from: /path/to/best.pt
📊 Data config: /path/to/data.yaml
🔍 Evaluating on: val split
⚙️  Confidence threshold: 0.25
⚙️  IoU threshold: 0.5
================================================================================

🔄 Running evaluation...

[Progress bars and validation output]

================================================================================
EVALUATION RESULTS
================================================================================

📊 OVERALL METRICS
--------------------------------------------------------------------------------
  mAP@0.5:        0.7210 (72.10%)
  mAP@[0.5:0.95]: 0.4760 (47.60%)
  Precision:      0.7129 (71.29%)
  Recall:         0.7552 (75.52%)

[... more output ...]
```

---

## 🆘 Still Having Issues?

### Debug Checklist:

1. ✅ Are you in the correct directory? (`pwd` should show `yolov8_project`)
2. ✅ Is your conda environment activated? (`which python` should show your env)
3. ✅ Is ultralytics installed? (`python -c "import ultralytics"`)
4. ✅ Does the model file exist? (`ls runs/detect/finetune_phase/weights/best.pt`)
5. ✅ Does data.yaml exist? (`ls data.yaml`)
6. ✅ Are validation images present? (`ls dataset/images/val/`)

### Get More Information:

```bash
# Check Python version
python --version

# Check what's installed
pip list | grep ultralytics

# Check file permissions
ls -la scripts/evaluate_model.py

# Try running with verbose output
python -v scripts/evaluate_model.py 2>&1 | head -20
```

---

## 🎓 Quick Test Commands

Test each component individually:

```bash
# Test 1: Can you import ultralytics?
python -c "from ultralytics import YOLO; print('✅ Import works')"

# Test 2: Can you load the model?
python -c "from ultralytics import YOLO; model = YOLO('runs/detect/finetune_phase/weights/best.pt'); print('✅ Model loads')"

# Test 3: Can you read data.yaml?
python -c "import yaml; yaml.safe_load(open('data.yaml')); print('✅ Data config valid')"
```

---

## 📝 Notes

- **Device options:** `mps` (Mac), `cpu` (any), `cuda:0` (NVIDIA GPU)
- **Confidence threshold:** Lower (0.15) = more detections, Higher (0.5) = fewer but more accurate
- **Results location:** Saved to `runs/detect/val/` by default

---

**If you encounter a specific error message, copy it and I can help you fix it!**




