# 🚀 Stage 2 Quick Start Guide

**Get started with Stage 2 implementation in 5 minutes!**

---

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- Dataset: `quen2-vl.yolov11/` (290 images, 9 classes)
- Stage 1 model: `yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt`

---

## ⚡ Quick Setup

```bash
# 1. Navigate to Stage 2 directory
cd stage2_conditional

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify dataset location
ls ../quen2-vl.yolov11/
```

---

## 🎯 Implementation Order

### 1️⃣ Dataset Analysis (5-10 min)
```bash
python scripts/analyze_stage2_dataset.py
```
**What it does:**
- Analyzes class distribution
- Calculates spatial relationships
- Identifies dataset challenges
- Generates comprehensive report

**Output:** `metadata/dataset_analysis.json`

---

### 2️⃣ Zero-Shot Testing (30-60 min)
```bash
python scripts/test_qwen2vl_zeroshot.py
```
**What it does:**
- Tests Qwen2-VL 7B without fine-tuning
- Measures baseline performance
- Tests spatial reasoning prompts

**Output:** `experiments/zeroshot_results.json`

**Target:** >80% accuracy

---

### 3️⃣ Fine-Tuning (2-4 hours)
```bash
python scripts/finetune_qwen2vl_lora.py
```
**What it does:**
- Fine-tunes Qwen2-VL 7B with LoRA
- Trains for 3-5 epochs
- Saves fine-tuned model

**Output:** `models/qwen2vl_lora_final/`

**Target:** >90% accuracy

**Note:** Skip if zero-shot >90%

---

### 4️⃣ Integration (15-30 min)
```bash
python scripts/integrate_stage1_stage2.py --split valid --num_images 10
```
**What it does:**
- Integrates Stage 1 detector with Stage 2 classifier
- Tests end-to-end pipeline
- Validates spatial reasoning

**Output:** `experiments/integration_results.json`

---

### 5️⃣ Final Evaluation (30-60 min)
```bash
python scripts/evaluate_qwen2vl.py --split test --model_path qwen2vl_lora_final
```
**What it does:**
- Comprehensive evaluation on test set
- Generates confusion matrix
- Calculates all metrics

**Output:** `experiments/final_evaluation_test.json` + confusion matrix

---

## 📊 Expected Timeline

| Phase | Time | Priority |
|-------|------|----------|
| Dataset Analysis | 10 min | ⭐⭐⭐ |
| Zero-Shot Testing | 60 min | ⭐⭐⭐ |
| Fine-Tuning | 2-4 hours | ⭐⭐ |
| Integration | 30 min | ⭐⭐ |
| Evaluation | 60 min | ⭐⭐⭐ |

**Total:** ~4-6 hours (excluding fine-tuning)

---

## 🎯 Success Criteria

- ✅ Dataset analyzed
- ✅ Zero-shot >80% OR Fine-tuned >90%
- ✅ Integration working
- ✅ Evaluation complete

---

## 🐛 Troubleshooting

### Issue: Model loading fails
**Solution:** Check GPU memory, use CPU if needed

### Issue: Dataset not found
**Solution:** Verify `quen2-vl.yolov11/` exists in project root

### Issue: Out of memory
**Solution:** Reduce batch size, use gradient accumulation

### Issue: Low accuracy
**Solution:** 
- Check prompts (spatial reasoning)
- Verify dataset quality
- Try fine-tuning

---

## 📚 Documentation

- **Full Plan:** `STAGE2_IMPLEMENTATION_PLAN.md`
- **Cursor Prompt:** `CURSOR_IMPLEMENTATION_PROMPT.md`
- **Summary:** `IMPLEMENTATION_SUMMARY.md`

---

**Ready to start? Run dataset analysis first!**


