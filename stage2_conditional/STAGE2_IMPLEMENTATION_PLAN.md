# 🚀 Stage 2 Implementation Plan: Qwen2-VL 7B for Conditional Classification

**Date:** 2025-01-27  
**Status:** Ready to Begin  
**Focus:** Data Analysis → Fine-Tuning → Integration → Evaluation  
**Quantization:** After all models are built and evaluated

---

## 🎯 Objectives

### Primary Goals
1. **Accuracy:** Achieve >90% accuracy for conditional classification
2. **Spatial Reasoning:** Correctly identify spatial relationships (>85% accuracy)
3. **Conditional Classification:** Classify 9 conditional classes accurately
4. **Integration:** Seamlessly integrate with Stage 1 object detection model

### Secondary Goals (After Accuracy Target Met)
5. **Quantization:** Optimize model for deployment (INT8/INT4)
6. **Production Ready:** Deploy on Nvidia A30 server

---

## 📊 Dataset Information

### Stage 2 Dataset
- **Location:** `../quen2-vl.yolov11/`
- **Total Images:** 290 (218 train, 47 valid, 25 test)
- **Resolution:** 640×640 (stretch)
- **Format:** YOLO polygon annotations
- **Classes:** 9 conditional classes
  - `Toe drain`, `Toe drain- Blocked`, `Toe drain- Damaged`
  - `rock toe`, `rock toe damaged`
  - `slope drain`, `slope drain blocked`, `slope drain damaged`
  - `vegetation`

### Stage 1 Integration
- **Best Model:** `../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt`
- **Performance:** 82.3% mAP@0.5
- **Classes Detected:** 4 (rock_toe, slope_drain, toe_drain, vegetation)

---

## 🏗️ Architecture Overview

### Two-Stage Pipeline

```
Stage 1 (Object Detection)
  ↓
[Detects: slope_drain, toe_drain, rock_toe, vegetation]
  ↓
[Extract bounding boxes + full image context]
  ↓
Stage 2 (Conditional Classification)
  ↓
[Qwen2-VL 7B classifies condition with spatial reasoning]
  ↓
[Output: 9 conditional classes with confidence]
```

### Key Components
1. **Stage 1 Detector:** YOLOv11 (82.3% mAP) - detects base objects
2. **Crop Extraction:** Extract object regions from full images
3. **Context Preparation:** Full image + all detections + spatial relationships
4. **Qwen2-VL 7B:** Classifies condition using visual + spatial reasoning
5. **Post-processing:** Format output for downstream use

---

## 📋 Implementation Phases

### Phase 1: Dataset Analysis (Week 1, Days 1-3)

**Goal:** Understand dataset characteristics for optimal fine-tuning

#### Tasks:
1. **Comprehensive Dataset Analysis**
   - Class distribution per split
   - Condition distribution (normal/blocked/damaged)
   - Object type distribution
   - Bounding box statistics (position, size, area)
   - Spatial relationship analysis
   - Class co-occurrence patterns
   - Image resolution verification
   - Dataset challenges identification

2. **Visual Inspection**
   - Sample image visualization with annotations
   - Quality check
   - Verify full images (not crops)

**Deliverables:**
- `scripts/analyze_stage2_dataset.py` - Analysis script
- `scripts/inspect_stage2_dataset.py` - Visualization script
- `metadata/dataset_analysis.json` - Analysis results
- `metadata/dataset_analysis_report.md` - Detailed report

---

### Phase 2: Qwen2-VL 7B Research & Setup (Week 1, Days 4-5)

**Goal:** Research and set up Qwen2-VL 7B for fine-tuning

#### Tasks:
1. **Model Research**
   - Qwen2-VL 7B architecture understanding
   - Fine-tuning methods (LoRA, QLoRA, full fine-tuning)
   - Prompt engineering for spatial reasoning
   - Best practices for conditional classification

2. **Environment Setup**
   - Install dependencies
   - Test model loading
   - Verify GPU/device compatibility

**Deliverables:**
- `requirements.txt` - Dependencies
- `metadata/qwen2vl_research.md` - Research notes
- `scripts/test_model_loading.py` - Model loading test

---

### Phase 3: Zero-Shot Testing (Week 2, Days 1-2)

**Goal:** Test Qwen2-VL 7B zero-shot performance before fine-tuning

#### Tasks:
1. **Zero-Shot Evaluation**
   - Test on validation set (47 images)
   - Build spatial reasoning prompts
   - Measure accuracy metrics
   - Analyze failure cases

2. **Prompt Engineering**
   - Develop optimal prompts for spatial reasoning
   - Test different prompt strategies
   - Optimize for conditional classification

**Deliverables:**
- `scripts/test_qwen2vl_zeroshot.py` - Zero-shot testing script
- `experiments/zeroshot_results.json` - Results
- `experiments/zeroshot_analysis.md` - Analysis report
- `metadata/optimal_prompts.py` - Best prompt templates

**Success Criteria:**
- Zero-shot accuracy >80% (target)
- Spatial reasoning working
- Inference time <3s per image

---

### Phase 4: LoRA Fine-Tuning (Week 2, Days 3-5)

**Goal:** Fine-tune Qwen2-VL 7B for maximum accuracy

#### Tasks:
1. **Fine-Tuning Setup**
   - Prepare training dataset
   - Set up LoRA configuration
   - Create training prompts with spatial reasoning
   - Configure training arguments

2. **Training**
   - Train with LoRA (efficient fine-tuning)
   - Monitor training metrics
   - Validate on validation set
   - Save checkpoints

3. **Evaluation**
   - Test fine-tuned model
   - Compare with zero-shot
   - Analyze improvements

**Deliverables:**
- `scripts/finetune_qwen2vl_lora.py` - Fine-tuning script
- `models/qwen2vl_lora_final/` - Fine-tuned model
- `experiments/training_logs.json` - Training logs
- `experiments/finetuning_results.md` - Results report

**Success Criteria:**
- Fine-tuned accuracy >90%
- Spatial reasoning >85%
- Conditional classification >88% per condition

---

### Phase 5: Stage 1 Integration (Week 3, Days 1-3)

**Goal:** Integrate Qwen2-VL 7B with Stage 1 detector

#### Tasks:
1. **Integration Pipeline**
   - Load Stage 1 detector
   - Run detection on images
   - Extract crops + context
   - Run Qwen2-VL classification
   - Combine results

2. **End-to-End Testing**
   - Test on sample images
   - Test on validation set
   - Measure end-to-end performance
   - Validate spatial reasoning

**Deliverables:**
- `scripts/integrate_stage1_stage2.py` - Integration script
- `scripts/end_to_end_test.py` - End-to-end testing
- `experiments/integration_results.json` - Results
- `experiments/integration_analysis.md` - Analysis

**Success Criteria:**
- End-to-end pipeline working
- Accuracy maintained (>90%)
- Spatial reasoning working
- Inference time acceptable

---

### Phase 6: Model Evaluation & Comparison (Week 3, Days 4-5)

**Goal:** Comprehensive evaluation and model selection

#### Tasks:
1. **Comprehensive Evaluation**
   - Test on test set (25 images)
   - Per-class accuracy
   - Per-condition accuracy
   - Per-object-type accuracy
   - Spatial reasoning accuracy
   - Error analysis

2. **Model Comparison**
   - Compare zero-shot vs fine-tuned
   - Analyze improvements
   - Document best model

**Deliverables:**
- `scripts/evaluate_qwen2vl.py` - Evaluation script
- `experiments/final_evaluation.json` - Results
- `experiments/final_evaluation_report.md` - Comprehensive report
- `models/best_model/` - Best performing model

**Success Criteria:**
- Test set accuracy >90%
- All metrics meet targets
- Model ready for quantization

---

### Phase 7: Quantization (Week 4 - After All Models Built)

**Goal:** Quantize best model for deployment

#### Tasks:
1. **INT8 Quantization**
   - Quantize to INT8
   - Test accuracy retention
   - Benchmark performance

2. **INT4 Quantization (Optional)**
   - Quantize to INT4 if needed
   - Evaluate trade-offs

**Deliverables:**
- `scripts/quantize_qwen2vl.py` - Quantization script
- `models/qwen2vl_int8/` - INT8 quantized model
- `models/qwen2vl_int4/` - INT4 quantized model (if created)
- `experiments/quantization_results.md` - Results

**Success Criteria:**
- INT8 accuracy retention >95%
- Model size reduced
- Inference speed improved

---

## 📁 Project Structure

```
stage2_conditional/
├── README.md                          # Stage 2 overview
├── STAGE2_IMPLEMENTATION_PLAN.md      # This file
├── requirements.txt                    # Dependencies
├── scripts/
│   ├── analyze_stage2_dataset.py      # Dataset analysis
│   ├── inspect_stage2_dataset.py      # Visual inspection
│   ├── test_qwen2vl_zeroshot.py       # Zero-shot testing
│   ├── finetune_qwen2vl_lora.py       # LoRA fine-tuning
│   ├── integrate_stage1_stage2.py     # Integration pipeline
│   ├── end_to_end_test.py             # End-to-end testing
│   ├── evaluate_qwen2vl.py             # Comprehensive evaluation
│   └── quantize_qwen2vl.py            # Quantization (later)
├── experiments/
│   ├── zeroshot_results.json          # Zero-shot results
│   ├── training_logs.json             # Training logs
│   ├── integration_results.json        # Integration results
│   └── final_evaluation.json           # Final evaluation
├── models/
│   ├── qwen2vl_lora_final/            # Fine-tuned model
│   ├── best_model/                    # Best performing model
│   ├── qwen2vl_int8/                  # INT8 quantized (later)
│   └── qwen2vl_int4/                  # INT4 quantized (later)
├── metadata/
│   ├── dataset_analysis.json          # Dataset statistics
│   ├── dataset_analysis_report.md     # Analysis report
│   ├── qwen2vl_research.md            # Research notes
│   └── optimal_prompts.py             # Best prompts
├── labeled/                           # Labeled data (if needed)
└── raw_crops/                         # Extracted crops (if needed)
```

---

## 🔧 Technical Specifications

### Qwen2-VL 7B Configuration

**Model:** `Qwen/Qwen2-VL-7B-Instruct`

**Fine-Tuning:**
- Method: LoRA (QLoRA)
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.1
- Target modules: q_proj, k_proj, v_proj, o_proj

**Training:**
- Learning rate: 2e-4
- Batch size: 1 (with gradient accumulation: 4)
- Epochs: 3-5
- Optimizer: AdamW
- Precision: FP16

**Inference:**
- Max new tokens: 50
- Temperature: 0.7
- Top-p: 0.9

### Spatial Reasoning Prompts

**Template:**
```
Analyze this infrastructure inspection image.

Detected objects: {list_of_objects}

Focus on: {target_object}

Classify condition considering:
1. Visual appearance: damaged, blocked, or normal?
2. Spatial relationships:
   - Is toe drain at bottom/end of slope drain?
   - Is rock toe above toe drain?
   - Relative positioning?
3. Context: Overall infrastructure state

Class: {class_name}
```

---

## 📊 Success Metrics

### Accuracy Targets
- **Overall Accuracy:** >90%
- **Spatial Reasoning:** >85%
- **Conditional Classification:**
  - Normal: >90%
  - Damaged: >85%
  - Blocked: >85%
- **Per-Object-Type:**
  - slope_drain: >90%
  - toe_drain: >85%
  - rock_toe: >85%

### Performance Targets
- **Inference Time:** <2s per image (FP16), <1s (INT8)
- **Memory Usage:** <16GB VRAM (FP16), <8GB (INT8)
- **Model Size:** <14GB (FP16), <7GB (INT8), <4GB (INT4)

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
cd stage2_conditional
pip install -r requirements.txt
```

### 2. Analyze Dataset
```bash
python scripts/analyze_stage2_dataset.py
```

### 3. Test Zero-Shot
```bash
python scripts/test_qwen2vl_zeroshot.py
```

### 4. Fine-Tune
```bash
python scripts/finetune_qwen2vl_lora.py
```

### 5. Integrate with Stage 1
```bash
python scripts/integrate_stage1_stage2.py
```

---

## 📝 Notes

- **Quantization:** Will be done after all models are built and evaluated
- **Focus:** Accuracy and spatial reasoning first, optimization later
- **Dataset:** Full images at 640×640 are ready for Qwen2-VL 7B
- **Integration:** Stage 1 model path is configured

---

**Last Updated:** 2025-01-27  
**Status:** Ready for Implementation


