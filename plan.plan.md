<!-- d618c5bb-f664-49e1-9b65-3169a8cdbcb9 6f9b13c2-15e5-4775-824f-4f5a8b5a308c -->
# CLIP Fine-Tuning Plan for Conditional Classification

## Context

- **Previous Attempt:** Qwen2-VL 7B failed due to memory constraints on M2 Max (42GB limit)
- **New Approach:** Switch to CLIP ViT-L/14 (~890MB) which fits on M2 Max GPU
- **Task:** Conditional classification of 9 classes (normal/damaged/blocked for 4 object types)
- **Dataset:** 1,465 instances across 290 images, severe class imbalance (7.04x ratio)
- **Target Accuracy:** >90% overall, >85% spatial reasoning

## 🎯 Stage 2 Primary Objectives

### Accuracy Goals
1. **Overall Accuracy:** >90% for conditional classification
2. **Spatial Reasoning:** >85% accuracy on spatial relationships
3. **Conditional Classification:** All 9 classes classified accurately
4. **Class Imbalance:** Address 7.04x ratio (Toe drain: 52 vs rock toe damaged: 366)
5. **Rare Condition:** Improve "blocked" condition (11.9% of data, currently 0% in zero-shot)

### Integration Goals
1. **Stage 1 Integration:** Seamless integration with YOLOv11 detector (82.3% mAP@0.5)
2. **Performance:** Inference time <2s per image
3. **Deployment:** Must work on M2 Max GPU (42GB limit)

### Dataset Requirements
- **9 Conditional Classes:**
  - Toe drain (normal/blocked/damaged)
  - Rock toe (normal/damaged)
  - Slope drain (normal/blocked/damaged)
  - Vegetation
- **Dataset Size:** 1,465 instances across 290 images (640×640)
- **Spatial Relationships:** 4,612 relationships identified
  - Toe drain typically at bottom (Y-position: 0.66-0.79)
  - Slope drain typically in middle (Y-position: 0.44-0.52)
  - Toe drain ABOVE slope drain 57.7% of the time
- **Multi-Object Context:** Average 5.09 objects per image

---

## 🤖 CLIP Model Selection: ViT-L/14

### Recommended Model: **CLIP ViT-L/14**

**HuggingFace:** `openai/clip-vit-large-patch14`

| Metric | Value | Justification |
|--------|-------|---------------|
| **Size** | ~890MB | 93x smaller than Qwen2-VL 7B |
| **Base Accuracy** | 85-90% | Meets >90% target with fine-tuning |
| **Memory** | ~4GB | Fits on M2 Max with room to spare |
| **Training Time** | ~3-5 hours | Reasonable vs 24-48 hours for Qwen2-VL |
| **Zero-Shot Baseline** | ~70-75% | Strong starting point |

### Why Not Other Variants?

| Model | Size | Accuracy | Why Not? |
|-------|------|----------|----------|
| **ViT-H/14** | ~1.5GB, ~6GB mem | ~88-92% | Only 2-3% better, 2x slower, diminishing returns |
| **ViT-B/32** | ~150MB, ~2GB mem | ~80-85% | Too low for >90% target |
| **ViT-B/16** | ~150MB, ~2GB mem | ~82-87% | Still below target |

### Expected Performance with ViT-L/14

- **Zero-Shot:** ~70-75% (baseline)
- **After Fine-Tuning:** >90% (with proper strategies)
- **Training Time:** ~3-5 hours on M2 Max
- **Memory Usage:** ~4GB (comfortable on 42GB GPU)

---

## 📋 Implementation Plan

### Phase 1: Setup CLIP Fine-Tuning Infrastructure

#### 1.1 Create CLIP Fine-Tuning Script

**File:** `stage2_conditional/scripts/finetune_clip_conditional.py`

**Key Components:**

1. **Model Loading**
   - Load OpenAI CLIP ViT-L/14: `openai/clip-vit-large-patch14`
   - Load processor for image + text
   - Handle MPS device (M2 Max GPU)
   - Use float32 for MPS compatibility

2. **Class Imbalance Handling** ⭐ CRITICAL
   - **Weighted Loss:** Class weights inversely proportional to frequency
   - **Oversampling:** Oversample rare classes (Toe drain variants)
   - **Focal Loss:** For hard examples (blocked condition)
   
   ```python
   class_weights = {
       'Toe drain': 7.04,              # 52 instances → Rare
       'Toe drain- Blocked': 4.69,    # 78 instances → Rare
       'Toe drain- Damaged': 4.82,    # 76 instances → Rare
       'rock toe': 2.39,               # 153 instances
       'rock toe damaged': 1.0,        # 366 instances → Base
       'slope drain': 1.42,            # 257 instances
       'slope drain blocked': 3.77,   # 97 instances → Rare
       'slope drain damaged': 2.47,    # 148 instances
       'vegetation': 1.54              # 238 instances
   }
   ```

3. **Spatial Reasoning Prompts** ⭐ CRITICAL
   - **Format:** `"{object_type} {condition}"` (e.g., "toe drain normal")
   - **Enhanced Format:** `"{object_type} {condition} at {position}"` (e.g., "toe drain normal at bottom")
   - **Multi-Object Context:** Include all objects in image for context
   
   **Spatial Patterns to Include:**
   - Toe drain at bottom (Y: 0.66-0.79)
   - Slope drain in middle (Y: 0.44-0.52)
   - Toe drain above slope drain (57.7%)
   - Rock toe above toe drain

4. **Training Approach**
   - **Option A (Recommended):** Fine-tune vision encoder only (faster, less overfitting)
   - **Option B:** Fine-tune both vision + text encoders (better accuracy, more parameters)
   - Start with Option A, try Option B if accuracy not met

#### 1.2 Dataset Adapter

**File:** `stage2_conditional/scripts/clip_dataset.py`

**Features:**

1. **Data Loading**
   - Load YOLO format annotations from `quen2-vl.yolov11/`
   - Support train/valid/test splits (218/47/25 images)
   - Parse polygon annotations → bounding boxes
   - Handle normalized coordinates (0.0-1.0)

2. **Image Processing**
   - **Key Decision:** Use full images with bbox context (preserves spatial relationships)
   - Alternative: Extract crops (can experiment later)
   - Resize to CLIP input size (224×224 for ViT-L/14)
   - Apply CLIP preprocessing (normalize, etc.)

3. **Text Prompt Generation**
   - Generate prompts for all 9 classes
   - Include spatial context when available
   - Multi-object prompts for images with multiple objects
   
   **9 Classes:**
   - Toe drain, Toe drain- Blocked, Toe drain- Damaged
   - rock toe, rock toe damaged
   - slope drain, slope drain blocked, slope drain damaged
   - vegetation

4. **Oversampling Strategy**
   - Oversample rare classes (Toe drain: 52 → ~200+ samples)
   - Oversample blocked condition (11.9% → ~25%)
   - Use data augmentation for rare classes
   - Balance dataset while preserving validation set

#### 1.3 Training Configuration

**File:** `stage2_conditional/scripts/finetune_clip_conditional.py`

**Training Arguments:**

```python
training_config = {
    'model': 'openai/clip-vit-large-patch14',  # ViT-L/14
    'batch_size': 16,                           # CLIP is smaller
    'learning_rate': 2e-5,                      # CLIP fine-tuning LR
    'epochs': 15,                               # CLIP trains faster
    'optimizer': 'AdamW',                       # Standard
    'scheduler': 'cosine',                      # Cosine annealing
    'warmup_steps': 100,                        # Warm-up
    'weight_decay': 0.01,                       # Regularization
    'device': 'mps',                            # M2 Max GPU
    'dtype': 'float32',                         # MPS requires float32
}
```

**Training Strategy:**

- Start with learning rate 2e-5 (lower for large model)
- Use gradient accumulation if needed (accumulation_steps=2)
- Monitor validation accuracy every 50 steps
- Save best model based on validation accuracy
- Early stopping if no improvement for 3 evaluations

---

### Phase 2: Evaluation and Testing

#### 2.1 Evaluation Script

**File:** `stage2_conditional/scripts/evaluate_clip.py`

**Metrics to Track:**

1. **Overall Metrics**
   - Overall accuracy (target: >90%)
   - Weighted F1-score
   - Confusion matrix

2. **Per-Class Metrics** (Critical for imbalance)
   - Accuracy per class (especially rare classes)
   - Precision/Recall/F1 per class
   - Support (instance count) per class

3. **Per-Condition Metrics**
   - Normal condition accuracy (target: >90%)
   - Damaged condition accuracy (target: >85%)
   - Blocked condition accuracy (target: >60%, currently 0%)

4. **Per-Object-Type Metrics**
   - Toe drain variants (target: >85%)
   - Slope drain variants (target: >90%)
   - Rock toe variants (target: >85%)
   - Vegetation (target: >90%)

5. **Spatial Reasoning Metrics** (target: >85%)
   - Accuracy when spatial context provided
   - Improvement vs non-spatial prompts
   - Relationship detection accuracy

**Zero-Shot Baseline:**
- Test pre-trained CLIP without fine-tuning
- Establish baseline (~70-75% expected)
- Compare with fine-tuned performance
- Measure improvement from fine-tuning

#### 2.2 Integration with Stage 1

**File:** `stage2_conditional/scripts/integrate_stage1_clip.py`

**Pipeline:**

```
Stage 1 (YOLOv11 Detector)
  ↓ Detects objects: rock_toe, slope_drain, toe_drain, vegetation
  ↓ Outputs: bounding boxes + confidence scores
  ↓
[Extract bounding boxes + full image context]
  ↓ For each detected object:
  ↓   - Extract bbox coordinates
  ↓   - Get full image
  ↓   - Identify spatial relationships (above/below/at end of)
  ↓
Stage 2 (CLIP ViT-L/14 Classifier)
  ↓ Generate text prompts for all conditions
  ↓ Compute CLIP similarity scores
  ↓ Select condition with highest similarity
  ↓
[Combine Results]
  ↓ Output: {object_type} {condition}
  ↓ Example: "toe drain blocked", "slope drain damaged"
```

**Text Prompts for Inference:**

1. Generate prompts for all 9 classes
2. For each detected object:
   - Generate prompts: "toe drain normal", "toe drain blocked", "toe drain damaged"
   - Include spatial context: "toe drain normal at bottom of image"
   - Add multi-object context if multiple objects detected
3. Use CLIP to compute image-text similarity
4. Return class with highest similarity score

**Stage 1 Model:**
- Path: `../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt`
- Performance: 82.3% mAP@0.5
- Classes: rock_toe, slope_drain, toe_drain, vegetation (4 base classes)

---

### Phase 3: Documentation and Progress Tracking

#### 3.1 Update Progress Documentation

**File:** `stage2_conditional/CLIP_FINETUNING_PROGRESS.md`

**Document:**

1. **Decision Rationale**
   - Why switched from Qwen2-VL to CLIP
   - Memory constraints (42GB limit)
   - Model size comparison (14GB vs 890MB)
   - Training time comparison (24-48h vs 3-5h)

2. **Model Selection**
   - Why ViT-L/14 over ViT-B/32 or ViT-H/14
   - Expected accuracy and performance
   - Trade-offs and justifications

3. **Training Results**
   - Zero-shot baseline results
   - Fine-tuned model results
   - Per-class accuracy improvements
   - Spatial reasoning improvements
   - Blocked condition improvements

4. **Challenges and Solutions**
   - Class imbalance → Weighted loss + oversampling
   - Blocked condition → Focal loss + oversampling
   - Spatial reasoning → Enhanced prompts
   - MPS device → float32, no pin_memory

#### 3.2 Create Quick Start Guide

**File:** `stage2_conditional/CLIP_QUICK_START.md`

**Include:**

1. **Installation**
   ```bash
   cd stage2_conditional
   pip install transformers torch torchvision pillow scikit-learn tqdm
   ```

2. **Zero-Shot Testing**
   ```bash
   python scripts/evaluate_clip.py --zero-shot --split valid
   ```

3. **Fine-Tuning**
   ```bash
   python scripts/finetune_clip_conditional.py \
       --model openai/clip-vit-large-patch14 \
       --batch_size 16 \
       --epochs 15 \
       --lr 2e-5
   ```

4. **Evaluation**
   ```bash
   python scripts/evaluate_clip.py \
       --model_path models/clip_conditional_final \
       --split test
   ```

5. **Integration with Stage 1**
   ```bash
   python scripts/integrate_stage1_clip.py \
       --yolo_model ../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt \
       --clip_model models/clip_conditional_final \
       --split valid \
       --num_images 10
   ```

6. **Expected Results**
   - Training time: ~3-5 hours on M2 Max
   - Memory usage: ~4GB
   - Accuracy: >90% (vs 70-75% zero-shot)

---

## 📁 File Structure

```
stage2_conditional/
├── scripts/
│   ├── finetune_clip_conditional.py    # Main training script ⭐
│   ├── clip_dataset.py                  # Dataset loader ⭐
│   ├── evaluate_clip.py                 # Evaluation script ⭐
│   ├── integrate_stage1_clip.py        # Stage 1 + Stage 2 integration ⭐
│   └── test_clip_zeroshot.py           # Zero-shot baseline (optional)
├── models/
│   └── clip_conditional_final/          # Saved fine-tuned models
├── experiments/
│   ├── clip_zeroshot_results.json      # Zero-shot results
│   ├── clip_training_logs.json         # Training logs
│   └── clip_evaluation_results.json    # Final evaluation
├── CLIP_FINETUNING_PROGRESS.md          # Progress tracking
└── CLIP_QUICK_START.md                  # Quick start guide
```

---

## 🚀 Key Advantages of CLIP ViT-L/14 vs Qwen2-VL

| Aspect | CLIP ViT-L/14 | Qwen2-VL 7B | Advantage |
|--------|---------------|-------------|-----------|
| **Model Size** | ~890MB | ~14GB | **16x smaller** |
| **Memory Usage** | ~4GB | ~42GB+ | **Fits on M2 Max** |
| **Training Time** | ~3-5 hours | ~24-48 hours | **5-10x faster** |
| **Model Loading** | <5 seconds | ~30 seconds | **6x faster** |
| **Inference Speed** | ~0.5s/image | ~2-3s/image | **4-6x faster** |
| **Architecture** | Simpler | Complex | **Easier to fine-tune** |
| **Community** | Well-established | Newer | **More examples** |
| **MPS Support** | Excellent | Poor (offloading) | **No issues** |

---

## 📊 Success Metrics

### Training Success
- [x] Model loads without memory errors
- [ ] Training completes in <5 hours
- [ ] Loss decreases consistently
- [ ] Validation accuracy improves
- [ ] Model saves successfully

### Accuracy Success
- [ ] Overall accuracy >90% (target)
- [ ] Spatial reasoning >85% (target)
- [ ] Rare classes >70% (Toe drain variants)
- [ ] Blocked condition >60% (currently 0%)
- [ ] Normal condition >90%
- [ ] Damaged condition >85%

### Integration Success
- [ ] Stage 1 + Stage 2 pipeline works end-to-end
- [ ] Inference time <2s per image
- [ ] Results properly formatted
- [ ] Spatial relationships preserved

### Deployment Success
- [ ] Model size acceptable (<1GB)
- [ ] Memory usage acceptable (<5GB)
- [ ] Inference speed acceptable (<2s)
- [ ] Ready for production

---

## 🎯 Implementation Checklist

### Must-Have Features

#### 1. Class Imbalance Handling ⭐ CRITICAL
- [ ] Weighted loss with class weights (7.04x ratio)
- [ ] Oversampling of rare classes (Toe drain variants)
- [ ] Focal loss for hard examples (blocked condition)
- [ ] Balanced validation set

#### 2. Spatial Reasoning ⭐ CRITICAL
- [ ] Enhanced prompts with Y-position info
- [ ] Multi-object context (all objects in image)
- [ ] Spatial relationship patterns (above/below/at end of)
- [ ] Position encoding in prompts

#### 3. Text Prompt Strategy ⭐ CRITICAL
- [ ] Format: `"{object_type} {condition}"`
- [ ] Include spatial context: `"{object_type} {condition} at {position}"`
- [ ] Multi-object prompts when multiple objects present
- [ ] Test with/without spatial context

#### 4. Integration with Stage 1 ⭐ CRITICAL
- [ ] Load YOLO detector from Stage 1
- [ ] Extract bounding boxes + full image context
- [ ] Pass to CLIP for conditional classification
- [ ] Combine results: `{object_type} {condition}`

#### 5. Evaluation Metrics ⭐ CRITICAL
- [ ] Overall accuracy (>90% target)
- [ ] Per-class accuracy (especially rare classes)
- [ ] Spatial reasoning accuracy (>85% target)
- [ ] F1-scores (handles imbalance)
- [ ] Confusion matrix

#### 6. Training Configuration ⭐ CRITICAL
- [ ] Model: `openai/clip-vit-large-patch14` (ViT-L/14)
- [ ] Batch size: 16-32
- [ ] Learning rate: 2e-5
- [ ] Epochs: 15
- [ ] Device: MPS (M2 Max GPU)
- [ ] Dtype: float32 (MPS compatibility)

---

## 📈 Expected Training Progress

### Epoch 1-5 (Initial Learning)
- Loss: ~3.0 → ~1.5
- Validation Accuracy: ~75% → ~82%
- Learning: Basic class recognition

### Epoch 6-10 (Refinement)
- Loss: ~1.5 → ~0.8
- Validation Accuracy: ~82% → ~88%
- Learning: Spatial relationships, rare classes

### Epoch 11-15 (Fine-Tuning)
- Loss: ~0.8 → ~0.5
- Validation Accuracy: ~88% → >90%
- Learning: Edge cases, blocked condition

### Final Model
- Loss: <0.5
- Validation Accuracy: >90%
- Test Accuracy: >90%
- Ready for deployment

---

## 🔧 Troubleshooting

### If Accuracy <90%:
1. Increase epochs to 20
2. Try fine-tuning both vision + text encoders
3. Increase oversampling for rare classes
4. Add more spatial context to prompts
5. Try ViT-H/14 if ViT-L/14 not sufficient

### If Memory Issues:
1. Reduce batch size to 8
2. Use gradient accumulation (accumulation_steps=2)
3. Switch to ViT-B/16 (but accuracy may drop)

### If Training Too Slow:
1. Use ViT-B/16 (faster but lower accuracy)
2. Reduce epochs to 10
3. Increase batch size if memory allows

---

## 🎓 Next Steps After Fine-Tuning

1. **Evaluate on Test Set**
   - Measure final accuracy
   - Compare with zero-shot baseline
   - Analyze per-class performance

2. **Analyze Results**
   - Identify remaining weak classes
   - Measure spatial reasoning improvements
   - Check blocked condition improvements

3. **Integrate with Stage 1**
   - Test end-to-end pipeline
   - Measure inference time
   - Validate results

4. **Deploy to Production**
   - Save final model
   - Create deployment script
   - Document usage

5. **Optional: Quantization**
   - Consider INT8 quantization for speed
   - Measure accuracy retention
   - Deploy quantized model if acceptable

---

**Last Updated:** December 15, 2025  
**Status:** Ready for Implementation  
**Next Action:** Build `finetune_clip_conditional.py` and `clip_dataset.py`

