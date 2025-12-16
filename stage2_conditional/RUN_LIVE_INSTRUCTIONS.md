# 🖥️ How to See Everything Running Live in Terminal

**Step-by-step guide to watch training in real-time**

---

## 🎯 Quick Start (Copy-Paste These Commands)

### **Step 1: Stop Any Background Processes**
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"
pkill -f test_qwen2vl
pkill -f finetune_qwen2vl
```

### **Step 2: Run Zero-Shot Test LIVE**
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"
python3 scripts/test_qwen2vl_zeroshot.py
```
**This will show everything live in your terminal!**

### **Step 3: After Zero-Shot Completes, Run Training LIVE**
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"
python3 scripts/finetune_qwen2vl_lora_enhanced.py
```
**This will show training progress live!**

---

## 📋 Detailed Step-by-Step Instructions

### **STEP 1: Open Terminal**

1. Open Terminal app (or iTerm2)
2. You should see a prompt like: `prathamprabhu@Prathams-MacBook-Pro ~ %`

---

### **STEP 2: Navigate to Project**

Copy and paste this command:
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"
```

**What you'll see:**
```
prathamprabhu@Prathams-MacBook-Pro stage2_conditional %
```

✅ You're now in the right directory!

---

### **STEP 3: Stop Any Background Processes**

If there's a test running in the background, stop it:
```bash
pkill -f test_qwen2vl
pkill -f finetune_qwen2vl
```

**What you'll see:**
- Either nothing (no processes to kill)
- Or: `[1]  + terminated  python3 scripts/test_qwen2vl_zeroshot.py`

---

### **STEP 4: Run Zero-Shot Test LIVE**

Run this command (it will run in foreground, so you see everything):
```bash
python3 scripts/test_qwen2vl_zeroshot.py
```

**What you'll see LIVE:**
```
Loading Qwen2-VL 7B model...
Using device: cpu
Loading model and processor...
Fetching 5 files: 100%|████████████| 5/5 [02:30<00:00, 30.00s/file]
✅ Model loaded successfully

Testing zero-shot performance on validation set...
Testing on 47 images from valid split...
Testing valid: 100%|████████████████| 47/47 [15:23<00:00, 19.64s/it]

================================================================================
QWEN2-VL 7B ZERO-SHOT RESULTS
================================================================================

Overall Accuracy: 78/146 = 53.42%

Per-Class Accuracy:
  Toe drain                     :   2/  4 = 50.00%
  Toe drain- Blocked            :   5/ 13 = 38.46%
  ...
```

**You'll see:**
- ✅ Progress bars (████████)
- ✅ Real-time processing
- ✅ Results as they happen
- ✅ Final summary

**Time:** 30-60 minutes (you'll see everything!)

---

### **STEP 5: After Zero-Shot, Run Training LIVE**

Once zero-shot finishes, run training:
```bash
python3 scripts/finetune_qwen2vl_lora_enhanced.py
```

**What you'll see LIVE:**
```
================================================================================
ENHANCED QWEN2-VL 7B LoRA FINE-TUNING
With Class Weighting, Spatial Reasoning, and Oversampling
================================================================================

Using device: cpu

1. Loading Qwen2-VL 7B model...
Fetching 5 files: 100%|████████████| 5/5 [02:30<00:00, 30.00s/file]
✅ Model loaded successfully

2. Setting up LoRA...
trainable params: 8,388,608 || all params: 8,388,608 || trainable%: 100.0
✅ LoRA configured

3. Creating enhanced datasets...
Loaded 1130 samples from train split (with oversampling)
Loaded 207 samples from valid split
✅ Train: 1130 samples (with oversampling)
✅ Val: 207 samples

4. Configuring training...

5. Initializing weighted trainer...

6. Starting training with enhancements...
   - Class-weighted loss (handles 7.04x imbalance)
   - Enhanced spatial reasoning prompts
   - Oversampling of rare classes
   - Position encoding in prompts
================================================================================

{'loss': 2.543, 'learning_rate': 0.0002, 'epoch': 0.0}
Epoch 1/3: 100%|████████████| 283/283 [05:23<00:00, 1.23s/it]
{'loss': 1.876, 'learning_rate': 0.0002, 'epoch': 1.0}
Epoch 2/3: 100%|████████████| 283/283 [05:20<00:00, 1.22s/it]
{'loss': 1.234, 'learning_rate': 0.0002, 'epoch': 2.0}
Epoch 3/3: 100%|████████████| 283/283 [05:18<00:00, 1.21s/it]

7. Saving final model...
✅ Model saved to models/qwen2vl_lora_enhanced_final/
✅ Training info saved to experiments/training_info_enhanced.json
================================================================================
✅ Enhanced fine-tuning complete!
```

**You'll see:**
- ✅ Progress bars for each epoch
- ✅ Loss decreasing: 2.543 → 1.876 → 1.234
- ✅ Time remaining estimates
- ✅ Checkpoint saves
- ✅ Final model save

**Time:** 2-4 hours (you'll see everything!)

---

## 🎯 What Each Output Means

### **During Zero-Shot:**

| Output | What It Means |
|--------|---------------|
| `Fetching 5 files: 100%` | Downloading model (first time only) |
| `Testing on 47 images` | Processing validation images |
| `████████░░ 80%` | Progress bar (80% done) |
| `Overall Accuracy: XX%` | Final score |

### **During Training:**

| Output | What It Means |
|--------|---------------|
| `Epoch 1/3: 100%` | First epoch complete (1 of 3) |
| `loss: 2.543 → 1.876` | Getting better (lower = better) |
| `[05:23<00:00, 1.23s/it]` | Time: 5min 23sec, 1.23sec per iteration |
| `Saving checkpoint-100` | Backup saved (every 100 steps) |
| `Saving final model...` | Training complete! |

---

## 💡 Pro Tips

### **Tip 1: Keep Terminal Open**
- Don't close terminal while training
- You can minimize it, but keep it running

### **Tip 2: Watch for Errors**
- If you see `ERROR` or `❌`, something went wrong
- Most errors are self-explanatory

### **Tip 3: Check Progress**
- Progress bars show completion
- Loss decreasing = good!
- Time estimates are approximate

### **Tip 4: If You Need to Stop**
- Press `Ctrl + C` to stop
- Training will stop safely
- Checkpoints are saved (can resume later)

---

## 🔍 Alternative: Watch Log File Live

If you want to run in background but still see output:

**Terminal 1 (Run in background):**
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"
python3 scripts/finetune_qwen2vl_lora_enhanced.py > experiments/training_live.log 2>&1 &
```

**Terminal 2 (Watch live):**
```bash
cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"
tail -f experiments/training_live.log
```

**This shows everything in real-time!**

---

## ✅ Quick Checklist

- [ ] Open Terminal
- [ ] Navigate: `cd "/Users/prathamprabhu/Desktop/CLIP model/stage2_conditional"`
- [ ] Stop background: `pkill -f test_qwen2vl`
- [ ] Run zero-shot: `python3 scripts/test_qwen2vl_zeroshot.py`
- [ ] Watch progress (30-60 min)
- [ ] After zero-shot: `python3 scripts/finetune_qwen2vl_lora_enhanced.py`
- [ ] Watch training (2-4 hours)
- [ ] Done! Model saved to `models/qwen2vl_lora_enhanced_final/`

---

**Ready? Copy-paste the commands above and watch everything live!** 🚀


