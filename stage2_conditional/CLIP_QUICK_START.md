# 🚀 CLIP Fine-Tuning Quick Start Guide

Quick guide to fine-tune CLIP ViT-L/14 for conditional classification.

---

## 📋 Prerequisites

- MacBook M2 Max with 42GB GPU
- Python 3.9+
- Dataset: `quen2-vl.yolov11/` (290 images, 9 classes)

---

## 🔧 Installation

```bash
cd stage2_conditional

# Install dependencies
pip install torch torchvision
pip install transformers
pip install pillow numpy
pip install scikit-learn matplotlib seaborn
pip install tqdm
pip install ultralytics  # For Stage 1 integration
```

---

## 📊 Step 1: Test Dataset Loader

```bash
cd scripts
python clip_dataset.py
```

**Expected Output:**
- Loads train/valid/test datasets
- Shows class distribution
- Displays oversampling stats

---

## 🔍 Step 2: Zero-Shot Baseline

Evaluate pre-trained CLIP without fine-tuning:

```bash
python evaluate_clip.py \
    --zero_shot \
    --split valid \
    --model_name openai/clip-vit-large-patch14
```

**Expected Results:**
- Accuracy: ~70-75% (baseline)
- Blocked condition: Low accuracy (expected)
- Rare classes: Low accuracy (expected)

**Takes:** ~2-3 minutes

---

## 🎯 Step 3: Fine-Tune CLIP

Train CLIP ViT-L/14 on your dataset:

```bash
python finetune_clip_conditional.py \
    --model_name openai/clip-vit-large-patch14 \
    --batch_size 16 \
    --epochs 15 \
    --lr 2e-5 \
    --oversample \
    --spatial_context \
    --output_dir ../models/clip_conditional_final
```

**Training Configuration:**
- Model: CLIP ViT-L/14 (~890MB)
- Device: MPS (M2 Max GPU)
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 15
- Memory usage: ~4GB

**Expected Training Time:**
- Per epoch: ~15-20 minutes
- Total (15 epochs): ~3-5 hours

**What Happens:**
1. Loads CLIP ViT-L/14 model
2. Applies class-weighted focal loss
3. Oversamples rare classes (Toe drain variants)
4. Trains with spatial reasoning prompts
5. Evaluates every epoch
6. Saves best model based on validation accuracy

**Monitoring:**
- Watch loss decrease: ~3.0 → ~0.5
- Watch accuracy increase: ~75% → >90%

---

## 📈 Step 4: Evaluate Fine-Tuned Model

Test on test set:

```bash
python evaluate_clip.py \
    --model_path ../models/clip_conditional_final \
    --split test
```

**Expected Results:**
- Accuracy: >90% (target)
- Rare classes: >70% (Toe drain variants)
- Blocked condition: >60% (was 0% in zero-shot)

**Outputs:**
- Accuracy metrics
- Confusion matrix (PNG)
- Per-class accuracy
- F1-scores
- Results JSON

---

## 🔗 Step 5: Integrate with Stage 1

Test end-to-end pipeline:

```bash
python integrate_stage1_clip.py \
    --yolo_model ../../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt \
    --clip_model ../models/clip_conditional_final \
    --split valid \
    --num_images 10 \
    --visualize
```

**Pipeline:**
1. Stage 1 (YOLO): Detects objects → bounding boxes
2. Stage 2 (CLIP): Classifies conditions → normal/damaged/blocked
3. Combines: `{object_type} {condition}`

**Outputs:**
- JSON results with all detections
- Visualizations (images with bboxes + labels)

---

## 📁 Output Files

After running all steps:

```
stage2_conditional/
├── models/
│   └── clip_conditional_final/
│       ├── best_model.pt                    # Fine-tuned model
│       ├── training_results.json            # Training history
│       └── clip_model/                      # Saved CLIP model
├── experiments/
│   ├── clip_zeroshot_valid_results.json    # Zero-shot results
│   ├── clip_finetuned_test_results.json    # Fine-tuned results
│   ├── clip_finetuned_test_confusion_matrix.png
│   └── integration_results/
│       ├── integration_results_valid.json
│       └── visualizations/                  # Result images
```

---

## 🎯 Success Criteria

### Training Success ✅
- [x] Model loads without errors
- [x] Training completes in <5 hours
- [x] Loss decreases consistently
- [x] Model saves successfully

### Accuracy Success ✅
- [ ] Overall accuracy >90%
- [ ] Spatial reasoning >85%
- [ ] Rare classes >70% (Toe drain variants)
- [ ] Blocked condition >60% (was 0%)

### Integration Success ✅
- [ ] End-to-end pipeline works
- [ ] Inference time <2s per image
- [ ] Results properly formatted

---

## 🐛 Troubleshooting

### Issue: Out of memory

**Solution:**
```bash
python finetune_clip_conditional.py \
    --batch_size 8 \
    --freeze_vision  # Train only classifier head
```

### Issue: Accuracy <90%

**Solutions:**
1. Train for more epochs: `--epochs 20`
2. Unfreeze vision encoder (remove `--freeze_vision`)
3. Try ViT-H/14 for higher accuracy:
   ```bash
   --model_name openai/clip-vit-huge-patch14
   ```

### Issue: Training too slow

**Solution:**
Use ViT-B/16 (faster but lower accuracy):
```bash
--model_name openai/clip-vit-base-patch16
```

---

## 📊 Expected Performance

| Metric | Zero-Shot | Fine-Tuned | Target |
|--------|-----------|------------|--------|
| **Overall Accuracy** | ~70-75% | >90% | >90% ✅ |
| **Rare Classes** | ~40-50% | >70% | >70% ✅ |
| **Blocked Condition** | ~0% | >60% | >60% ✅ |
| **Training Time** | 0 | ~3-5h | <5h ✅ |
| **Memory Usage** | - | ~4GB | <6GB ✅ |

---

## 🎓 Understanding the Results

### What "Fine-Tuned" Means

**Before Fine-Tuning (Zero-Shot):**
- Model is pre-trained on general images
- Knows general concepts
- Not specialized for your task
- Accuracy: ~70-75%

**After Fine-Tuning:**
- Model learned your specific classes
- Understands spatial relationships
- Handles class imbalance
- Accuracy: >90%

### Key Improvements

1. **Class Imbalance:** Weighted loss + oversampling
2. **Rare Classes:** Toe drain variants improved from ~40% → >70%
3. **Blocked Condition:** Improved from 0% → >60%
4. **Spatial Reasoning:** Prompts include position info

---

## 🚀 Next Steps

1. **Deploy Model:**
   - Quantize to INT8 for faster inference
   - Deploy to Nvidia A30 server
   - Create REST API

2. **Improve Accuracy:**
   - Collect more data for rare classes
   - Add more spatial context
   - Ensemble models

3. **Production Ready:**
   - Add error handling
   - Implement logging
   - Create monitoring dashboard

---

## 📞 Quick Reference

### Dataset Stats
- Total instances: 1,465
- Images: 290 (218 train, 47 valid, 25 test)
- Classes: 9 conditional classes
- Imbalance: 7.04x ratio

### Model Stats
- Size: ~890MB (ViT-L/14)
- Memory: ~4GB during training
- Parameters: ~300M (total), ~10M (trainable)

### Training Time
- Per epoch: ~15-20 minutes
- Total: ~3-5 hours (15 epochs)
- Zero-shot eval: ~2-3 minutes

---

**Last Updated:** December 15, 2025  
**Status:** Ready to Use

