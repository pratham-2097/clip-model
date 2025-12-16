# 🎓 Beginner-Friendly Training Explanation

**What's happening when we train the Qwen2-VL model**

---

## 🎯 What We're Doing (Simple Version)

Think of training like teaching a student:

1. **Zero-Shot Test** = Giving the student a test WITHOUT any teaching first
   - We see how smart the model is "out of the box"
   - This is our baseline (starting point)

2. **Fine-Tuning** = Teaching the model with our specific examples
   - We show it 1,130 training images with correct answers
   - The model learns patterns and gets better
   - We check progress with 207 validation images

---

## 📍 Where Everything Happens

### **Model Files Location:**
```
stage2_conditional/
├── models/                          ← Trained models saved here
│   ├── qwen2vl_lora_enhanced_final/ ← Final trained model (after training)
│   └── qwen2vl_lora_enhanced_checkpoints/ ← Checkpoints (saves during training)
│
├── experiments/                      ← Results and logs saved here
│   ├── zeroshot_results.json       ← Zero-shot test results
│   ├── training_info_enhanced.json  ← Training information
│   └── training_logs/               ← Training progress logs
│
└── scripts/
    ├── test_qwen2vl_zeroshot.py      ← Zero-shot test script
    └── finetune_qwen2vl_lora_enhanced.py ← Training script
```

---

## 🔄 Step-by-Step: What's Happening

### **Step 1: Zero-Shot Test** (Running Now)

**What it does:**
- Loads the pre-trained Qwen2-VL 7B model (already trained on general images)
- Tests it on 47 validation images WITHOUT any training
- Measures accuracy: "How many did it get right?"

**What you'll see:**
```
Loading Qwen2-VL 7B model...        ← Downloading model (first time only, ~14GB)
Testing on 47 images...              ← Processing each image
Overall Accuracy: XX%                ← Final score
```

**Where results go:**
- `experiments/zeroshot_results.json` - Detailed results
- Console output - Summary

**Time:** 30-60 minutes (first time includes download)

---

### **Step 2: Fine-Tuning** (After Zero-Shot)

**What it does:**
- Takes the same model
- Shows it 1,130 training examples with correct answers
- Model learns: "Oh, THIS is what a blocked toe drain looks like!"
- Saves improved model

**What you'll see:**
```
Loading model...                     ← Loads pre-trained model
Setting up LoRA...                   ← Prepares efficient training method
Creating datasets...                 ← Loads 1,130 training images
Starting training...                  ← Training begins!
Epoch 1/3: 100%|██████| 283/283     ← Progress bar
  Loss: 2.5 → 1.8 → 1.2             ← Getting better!
Epoch 2/3: ...
Epoch 3/3: ...
Saving final model...                ← Saves to models/qwen2vl_lora_enhanced_final/
```

**Where model is saved:**
- `models/qwen2vl_lora_enhanced_final/` - Final trained model
- `models/qwen2vl_lora_enhanced_checkpoints/` - Checkpoints (backups during training)

**Time:** 2-4 hours (depends on your computer)

---

## 🧠 What "Training" Actually Means (Simple)

**Imagine teaching someone to recognize cats:**

1. **Show examples:** "This is a cat, this is a dog, this is a cat..."
2. **They learn patterns:** "Cats have pointy ears, dogs have floppy ears"
3. **Test them:** "What's this?" → "Cat!" ✅
4. **If wrong:** Show more examples, they learn more

**That's exactly what we're doing:**
- Show model: "This is a blocked toe drain"
- Model learns: "Blocked drains have obstructions, are at bottom of image"
- Test: "What's this?" → Model predicts
- If wrong: Adjust and learn more

---

## 📊 What "LoRA" Means (Simple)

**Normal training:** Change ALL parts of the model (slow, needs lots of memory)

**LoRA training:** Only change SMALL parts of the model (fast, efficient)

**Analogy:**
- Normal = Rewriting entire textbook
- LoRA = Adding sticky notes to specific pages

**Why we use it:**
- ✅ Faster (hours instead of days)
- ✅ Uses less memory
- ✅ Still gets great results!

---

## 🎯 What "Enhanced" Means

Our enhanced script does 5 smart things:

1. **Class Weighting** = Gives more attention to rare classes
   - Like: "Pay extra attention to 'Toe drain' (only 52 examples)"

2. **Spatial Reasoning** = Uses position information
   - Like: "Toe drains are usually at the bottom of images"

3. **Oversampling** = Shows rare examples more often
   - Like: Showing "Toe drain" examples 7x more than "rock toe damaged"

4. **Position Encoding** = Tells model WHERE objects are
   - Like: "This object is at Y-position 0.75 (bottom of image)"

5. **Condition Weighting** = Emphasizes rare conditions
   - Like: "Blocked condition is rare, pay 4x more attention"

---

## 📈 How to Monitor Progress

### **During Zero-Shot:**
- Watch console for: "Testing on 47 images..."
- Progress bar shows: `[████████░░] 80%`
- Final: "Overall Accuracy: XX%"

### **During Training:**
- Watch for: `Epoch 1/3: 100%|██████| 283/283`
- Loss decreases: `Loss: 2.5 → 1.8 → 1.2` (lower = better!)
- Checkpoints saved: Every 100 steps
- Final model: Saved at end

### **Check Results:**
```bash
# See zero-shot results
cat experiments/zeroshot_results.json

# See training info
cat experiments/training_info_enhanced.json

# See where model is saved
ls -lh models/qwen2vl_lora_enhanced_final/
```

---

## ⏱️ Timeline

| Step | Time | What Happens |
|------|------|--------------|
| Zero-Shot | 30-60 min | Testing without training |
| Fine-Tuning | 2-4 hours | Training the model |
| **Total** | **3-5 hours** | Complete process |

---

## 🎓 Key Terms (Simple Definitions)

- **Zero-Shot:** Testing without training (baseline)
- **Fine-Tuning:** Teaching the model with our examples
- **LoRA:** Efficient training method (only changes small parts)
- **Epoch:** One full pass through all training data
- **Loss:** How wrong the model is (lower = better)
- **Checkpoint:** Saved progress (backup during training)
- **Validation:** Testing on images model hasn't seen

---

## ✅ Success Indicators

**Good Zero-Shot:**
- Accuracy >75% = Model is already pretty good!

**Good Training:**
- Loss decreases steadily: 2.5 → 1.8 → 1.2
- Final accuracy >90% = Success! 🎉

**Model Saved:**
- Files appear in `models/qwen2vl_lora_enhanced_final/`
- Can use this model for predictions!

---

**Now let's start! First zero-shot, then training!** 🚀


