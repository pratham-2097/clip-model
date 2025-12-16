# 📍 Where Is My Model Being Trained?

**Quick guide to find everything during training**

---

## 🗂️ Folder Structure

```
stage2_conditional/
│
├── 📁 models/                          ← YOUR TRAINED MODEL GOES HERE
│   ├── qwen2vl_lora_enhanced_final/    ← Final trained model (after training completes)
│   │   ├── adapter_config.json         ← LoRA configuration
│   │   ├── adapter_model.bin           ← Trained weights (the actual model!)
│   │   └── ...
│   │
│   └── qwen2vl_lora_enhanced_checkpoints/ ← Checkpoints (backups during training)
│       ├── checkpoint-100/              ← Saved at step 100
│       ├── checkpoint-200/              ← Saved at step 200
│       └── ...
│
├── 📁 experiments/                      ← RESULTS AND LOGS
│   ├── zeroshot_results.json           ← Zero-shot test results
│   ├── training_info_enhanced.json     ← Training configuration
│   └── training_logs/                   ← Detailed training logs
│
└── 📁 scripts/                          ← TRAINING SCRIPTS
    ├── test_qwen2vl_zeroshot.py         ← Zero-shot test (running now)
    └── finetune_qwen2vl_lora_enhanced.py ← Training script (next step)
```

---

## 🔍 How to Check What's Happening

### **1. Check if Zero-Shot is Running:**
```bash
cd stage2_conditional
ps aux | grep test_qwen2vl
```

### **2. Check Zero-Shot Results (when done):**
```bash
cat experiments/zeroshot_results.json
```

### **3. Check if Training is Running:**
```bash
ps aux | grep finetune_qwen2vl
```

### **4. Check Training Progress:**
```bash
# See latest checkpoint
ls -lth models/qwen2vl_lora_enhanced_checkpoints/ | head -5

# See training logs
tail -f experiments/training_logs/*.log
```

### **5. Check Final Model (after training):**
```bash
ls -lh models/qwen2vl_lora_enhanced_final/
```

---

## 📊 What Files Mean

### **During Zero-Shot:**
- **No model files created yet** (just testing)
- Results saved to: `experiments/zeroshot_results.json`

### **During Training:**
- **Checkpoints:** `models/qwen2vl_lora_enhanced_checkpoints/checkpoint-XXX/`
  - Created every 100 steps
  - These are backups (in case training stops)
  
### **After Training:**
- **Final Model:** `models/qwen2vl_lora_enhanced_final/`
  - This is your trained model!
  - Use this for predictions
  - Size: ~100-500 MB (LoRA is small!)

---

## 🎯 Quick Status Check Commands

```bash
# Navigate to stage2_conditional
cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"

# Check zero-shot status
echo "=== Zero-Shot Status ==="
ls -lh experiments/zeroshot_results.json 2>/dev/null && echo "✅ Zero-shot complete!" || echo "⏳ Still running..."

# Check training status
echo "=== Training Status ==="
ls -lh models/qwen2vl_lora_enhanced_final/ 2>/dev/null && echo "✅ Training complete!" || echo "⏳ Not started yet..."

# Check if processes are running
echo "=== Running Processes ==="
ps aux | grep -E "test_qwen2vl|finetune_qwen2vl" | grep -v grep || echo "No training processes running"
```

---

## 📈 Progress Indicators

### **Zero-Shot Progress:**
- ✅ Model downloading (first time only)
- ✅ "Testing on 47 images..."
- ✅ Progress bar: `[████████░░] 80%`
- ✅ "Overall Accuracy: XX%"
- ✅ Results saved to `experiments/zeroshot_results.json`

### **Training Progress:**
- ✅ "Loading Qwen2-VL 7B model..."
- ✅ "Setting up LoRA..."
- ✅ "Creating datasets..."
- ✅ "Starting training..."
- ✅ Progress: `Epoch 1/3: 100%|██████| 283/283 [05:23<00:00, 1.23s/it]`
- ✅ Loss decreasing: `loss: 2.543 → 1.876 → 1.234`
- ✅ Checkpoints saved: `checkpoint-100`, `checkpoint-200`, etc.
- ✅ "Saving final model..." → `models/qwen2vl_lora_enhanced_final/`

---

## 🎓 What's Happening Right Now

### **Step 1: Zero-Shot Test** (Currently Running)
- **Location:** Script running in background
- **What it's doing:** Testing model on 47 images
- **Output:** Will save to `experiments/zeroshot_results.json`
- **Time:** 30-60 minutes

### **Step 2: Training** (Next, After Zero-Shot)
- **Location:** Will run `scripts/finetune_qwen2vl_lora_enhanced.py`
- **What it's doing:** Training model on 1,130 images
- **Output:** Will save to `models/qwen2vl_lora_enhanced_final/`
- **Time:** 2-4 hours

---

## 💡 Beginner Tip

**Think of it like this:**
- **Zero-Shot** = Taking a test without studying
- **Training** = Studying with flashcards
- **Final Model** = Your brain after studying (saved to disk!)

**The model files are like saved brain states:**
- Checkpoints = Snapshots during studying
- Final Model = Your brain after finishing all flashcards

---

**Check back in 30-60 minutes for zero-shot results!** ⏰


