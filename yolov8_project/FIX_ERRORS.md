# 🔧 Fix Errors: Step-by-Step Guide

## 🚨 Your Current Situation

Based on diagnostics:
- ❌ Conda is not in your PATH
- ❌ Ultralytics is not installed in system Python

---

## ✅ Solution 1: Install Ultralytics in System Python (Easiest)

### Step 1: Open Terminal in Cursor
Press `` Ctrl+` `` or go to **Terminal → New Terminal**

### Step 2: Navigate to Project
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
```

### Step 3: Install Ultralytics
```bash
pip3 install ultralytics
```

**If you get "permission denied":**
```bash
pip3 install --user ultralytics
```

### Step 4: Verify Installation
```bash
python3 -c "from ultralytics import YOLO; print('✅ Ultralytics installed!')"
```

### Step 5: Run Evaluation
```bash
python3 scripts/evaluate_model.py
```

---

## ✅ Solution 2: Use Conda (If You Have It Installed)

### Step 1: Find Conda
Conda might be installed but not in PATH. Try:

```bash
# Option A: Check common locations
ls ~/anaconda3/bin/conda
ls ~/miniconda3/bin/conda
ls /opt/homebrew/bin/conda

# Option B: Find conda
which conda || find ~ -name conda 2>/dev/null | head -1
```

### Step 2: Initialize Conda (if found)
If you found conda, initialize it:
```bash
# Replace with your conda path
~/anaconda3/bin/conda init zsh
# Then restart terminal or run:
source ~/.zshrc
```

### Step 3: Create/Activate Environment
```bash
# Create environment (first time only)
conda create -n yolov8 python=3.10 -y

# Activate it
conda activate yolov8

# Install ultralytics
pip install ultralytics
```

### Step 4: Run Evaluation
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
python scripts/evaluate_model.py
```

---

## ✅ Solution 3: Use Python Virtual Environment

### Step 1: Create Virtual Environment
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
python3 -m venv venv
```

### Step 2: Activate It
```bash
source venv/bin/activate
```

### Step 3: Install Ultralytics
```bash
pip install ultralytics
```

### Step 4: Run Evaluation
```bash
python scripts/evaluate_model.py
```

---

## 🎯 Recommended: Quick Fix (Copy-Paste This)

**Open terminal in Cursor and run these commands one by one:**

```bash
# 1. Go to project
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"

# 2. Install ultralytics
pip3 install ultralytics

# 3. Test import
python3 -c "from ultralytics import YOLO; print('✅ Ready!')"

# 4. Run evaluation
python3 scripts/evaluate_model.py
```

---

## 🔍 Detailed Error Diagnosis

### Error 1: `ModuleNotFoundError: No module named 'ultralytics'`

**What it means:** Ultralytics library is not installed

**Fix:**
```bash
pip3 install ultralytics
```

**Verify:**
```bash
python3 -c "import ultralytics; print('OK')"
```

---

### Error 2: `❌ Error: Model weights not found`

**What it means:** The model file path is wrong

**Fix:**
```bash
# Check if file exists
ls -la runs/detect/finetune_phase/weights/best.pt

# If it doesn't exist, find your model
find runs -name "*.pt" -type f

# Then run with correct path
python3 scripts/evaluate_model.py --weights path/to/your/model.pt
```

---

### Error 3: `❌ Error: Data config not found`

**What it means:** Can't find data.yaml

**Fix:**
```bash
# Make sure you're in the right directory
pwd
# Should show: .../yolov8_project

# Check if data.yaml exists
ls -la data.yaml

# If you're in wrong directory, navigate:
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
```

---

### Error 4: `RuntimeError: MPS backend not available`

**What it means:** Metal (Mac GPU) not working

**Fix:** Use CPU instead
```bash
python3 scripts/evaluate_model.py --device cpu
```

---

### Error 5: `Permission denied` or `command not found`

**What it means:** Script not executable or wrong Python

**Fix:**
```bash
# Make script executable
chmod +x scripts/evaluate_model.py

# Use python3 explicitly
python3 scripts/evaluate_model.py
```

---

## 📋 Pre-Flight Checklist

Before running, check these:

```bash
# 1. Are you in the right directory?
pwd
# Should show: .../yolov8_project

# 2. Does data.yaml exist?
ls data.yaml
# Should show file details

# 3. Does model exist?
ls runs/detect/finetune_phase/weights/best.pt
# Should show file details

# 4. Is ultralytics installed?
python3 -c "import ultralytics; print('OK')"
# Should print: OK

# 5. Can you load the model?
python3 -c "from ultralytics import YOLO; YOLO('runs/detect/finetune_phase/weights/best.pt'); print('Model loads OK')"
# Should print: Model loads OK
```

**If all 5 checks pass, you're ready to run!**

---

## 🚀 Complete Working Example

Here's exactly what to type (copy-paste each line):

```bash
# Line 1: Navigate
cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"

# Line 2: Install (if needed)
pip3 install ultralytics

# Line 3: Verify
python3 -c "from ultralytics import YOLO; print('✅ Ready')"

# Line 4: Run
python3 scripts/evaluate_model.py
```

---

## 🆘 Still Stuck?

### Get More Info:

```bash
# Check Python version
python3 --version

# Check what's installed
pip3 list | grep -i yolo

# Check file permissions
ls -la scripts/evaluate_model.py

# Try with verbose output
python3 -v scripts/evaluate_model.py 2>&1 | head -30
```

### Common Issues:

1. **"pip3: command not found"**
   - Install Python: `brew install python3` (on Mac)

2. **"Permission denied"**
   - Use `pip3 install --user ultralytics`

3. **"No such file or directory"**
   - Make sure you're in `yolov8_project` folder

4. **"Module not found" even after install**
   - Try: `python3 -m pip install ultralytics`

---

## 💡 Pro Tips

- **Always check your current directory** with `pwd`
- **Use `python3` explicitly** (not just `python`)
- **Install with `pip3`** to match Python version
- **Check file paths** if you get "not found" errors

---

**Copy the error message you're seeing and I can help you fix it specifically!**




