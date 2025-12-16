# 🎯 Comprehensive Cursor Prompt: Stage 2 Qwen2-VL 7B Implementation

**Use this prompt in Cursor to build the complete Stage 2 system**

---

## 📋 Complete Implementation Prompt

```
I need to build a complete Stage 2 conditional classification system using Qwen2-VL 7B for infrastructure inspection. This is a multi-stage pipeline that integrates with an existing Stage 1 object detection model.

## Project Context

**Stage 1 (Complete):**
- Best Model: `yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt`
- Performance: 82.3% mAP@0.5
- Detects: rock_toe, slope_drain, toe_drain, vegetation (4 classes)

**Stage 2 (To Build):**
- Dataset: `quen2-vl.yolov11/` (290 images, 640×640, 9 conditional classes)
- Model: Qwen2-VL 7B (Qwen/Qwen2-VL-7B-Instruct)
- Goal: Classify conditions (normal, blocked, damaged) with spatial reasoning
- Target: >90% accuracy, >85% spatial reasoning

**Project Structure:**
- All Stage 2 work goes in: `stage2_conditional/`
- Dataset is at: `quen2-vl.yolov11/` (relative to project root)

## Requirements

### Phase 1: Dataset Analysis (Priority 1)

Create `stage2_conditional/scripts/analyze_stage2_dataset.py` that:

1. **Parse YOLO Polygon Annotations**
   - Read label files from train/valid/test splits
   - Convert polygon coordinates to bounding boxes (x_center, y_center, width, height)
   - Handle normalized coordinates (0.0-1.0)

2. **Class Distribution Analysis**
   - Calculate per-class counts for all 9 classes
   - Calculate per-split distribution (train/valid/test)
   - Calculate condition distribution (normal/blocked/damaged)
   - Calculate object type distribution (toe_drain/slope_drain/rock_toe/vegetation)
   - Identify class imbalance (max/min ratio)

3. **Bounding Box Statistics**
   - For each class: mean/std/min/max for area, width, height
   - Y-position analysis (top/middle/bottom of image)
   - Identify spatial patterns (e.g., "toe drain usually at bottom")

4. **Spatial Relationship Analysis**
   - Analyze relationships between objects in same image
   - Identify "above/below" relationships
   - Calculate distances between objects
   - Document key patterns:
     - "toe drain at bottom/end of slope drain"
     - "rock toe above toe drain"
     - Relative positioning patterns

5. **Co-occurrence Analysis**
   - Which classes appear together in images
   - Frequency of co-occurrence
   - Common object combinations

6. **Image Statistics**
   - Verify resolution (should be 640×640)
   - Calculate instances per image (mean, std, min, max)
   - Check image quality

7. **Dataset Challenges Identification**
   - Class imbalance severity
   - Data size assessment (low/moderate/sufficient)
   - Recommendations for fine-tuning

8. **Output**
   - Save to `stage2_conditional/metadata/dataset_analysis.json`
   - Print comprehensive report to console
   - Generate markdown report: `stage2_conditional/metadata/dataset_analysis_report.md`

**Class Names:**
```python
CLASS_NAMES = [
    'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
    'rock toe', 'rock toe damaged',
    'slope drain', 'slope drain blocked', 'slope drain damaged',
    'vegetation'
]
```

### Phase 2: Qwen2-VL 7B Zero-Shot Testing (Priority 2)

Create `stage2_conditional/scripts/test_qwen2vl_zeroshot.py` that:

1. **Model Loading**
   - Load Qwen2-VL 7B: `Qwen/Qwen2-VL-7B-Instruct`
   - Load processor
   - Handle device (CUDA/MPS/CPU)
   - Use FP16 if CUDA available

2. **Spatial Reasoning Prompt Engineering**
   - Build prompts that emphasize spatial relationships
   - Include all detected objects for context
   - Ask model to consider:
     - Visual appearance (damaged, blocked, normal)
     - Spatial relationships (positions, above/below)
     - Context (overall infrastructure state)
   - Format: Image + text prompt for Qwen2-VL

3. **Zero-Shot Evaluation**
   - Test on validation set (47 images)
   - For each image:
     - Load image and ground truth annotations
     - For each annotated object:
       - Build spatial reasoning prompt
       - Run Qwen2-VL inference
       - Extract predicted class
       - Compare with ground truth

4. **Metrics Calculation**
   - Overall accuracy
   - Per-class accuracy (9 classes)
   - Per-condition accuracy (normal/blocked/damaged)
   - Per-object-type accuracy (toe_drain/slope_drain/rock_toe/vegetation)
   - Inference time per object
   - Error analysis (confusion matrix)

5. **Output**
   - Save to `stage2_conditional/experiments/zeroshot_results.json`
   - Print comprehensive results
   - Generate analysis report

**Prompt Template:**
```
Analyze this infrastructure inspection image.

Detected objects: {list}

Focus on: {target_object}

Classify condition considering:
1. Visual appearance: damaged, blocked, or normal?
2. Spatial relationships: positions relative to other objects
3. Context: overall infrastructure state

Class: {class_name}
```

### Phase 3: LoRA Fine-Tuning (Priority 3)

Create `stage2_conditional/scripts/finetune_qwen2vl_lora.py` that:

1. **Dataset Preparation**
   - Create Dataset class for Qwen2-VL format
   - Load images and annotations
   - Build training prompts with spatial reasoning
   - Format: Image + prompt → target class name

2. **LoRA Configuration**
   - Use PEFT library for LoRA
   - LoRA rank: 16
   - LoRA alpha: 32
   - LoRA dropout: 0.1
   - Target modules: q_proj, k_proj, v_proj, o_proj

3. **Training Setup**
   - Learning rate: 2e-4
   - Batch size: 1 (with gradient accumulation: 4)
   - Epochs: 3-5
   - FP16 training if CUDA
   - Evaluation on validation set
   - Save checkpoints

4. **Training Prompts**
   - Include spatial reasoning context
   - List all objects in image for context
   - Emphasize visual + spatial + context analysis
   - Target: exact class name from 9 classes

5. **Output**
   - Save model to `stage2_conditional/models/qwen2vl_lora_final/`
   - Save training logs
   - Evaluate on validation set after training

### Phase 4: Stage 1 Integration (Priority 4)

Create `stage2_conditional/scripts/integrate_stage1_stage2.py` that:

1. **Load Both Models**
   - Stage 1: Load YOLO model from `yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt`
   - Stage 2: Load fine-tuned Qwen2-VL 7B (or base if not fine-tuned)

2. **End-to-End Pipeline**
   - Stage 1: Run detection on image → get bounding boxes + classes
   - Extract spatial context from all detections
   - Stage 2: For each detection:
     - Build prompt with spatial context
     - Run Qwen2-VL classification
     - Get conditional class

3. **Spatial Context Extraction**
   - Extract positions of all detected objects
   - Identify relationships (above/below, distances)
   - Build context string for prompts

4. **Testing**
   - Test on sample images
   - Test on validation set
   - Measure end-to-end performance
   - Validate spatial reasoning

5. **Output**
   - Save results to `stage2_conditional/experiments/integration_results.json`
   - Print summary statistics

### Phase 5: Comprehensive Evaluation (Priority 5)

Create `stage2_conditional/scripts/evaluate_qwen2vl.py` that:

1. **Test Set Evaluation**
   - Test on test set (25 images)
   - Calculate all metrics
   - Compare zero-shot vs fine-tuned

2. **Detailed Metrics**
   - Overall accuracy
   - Per-class accuracy
   - Per-condition accuracy
   - Per-object-type accuracy
   - Spatial reasoning accuracy
   - Confusion matrix

3. **Error Analysis**
   - Identify failure cases
   - Analyze common errors
   - Document improvements needed

4. **Output**
   - Save to `stage2_conditional/experiments/final_evaluation.json`
   - Generate comprehensive report

## Technical Requirements

### Dependencies
- transformers>=4.37.0
- torch>=2.0.0
- peft>=0.7.0 (for LoRA)
- ultralytics>=8.0.0 (for Stage 1)
- All other dependencies in requirements.txt

### File Structure
All files should be in `stage2_conditional/`:
- Scripts in `scripts/`
- Results in `experiments/`
- Models in `models/`
- Metadata in `metadata/`

### Error Handling
- Handle missing files gracefully
- Provide clear error messages
- Continue processing if individual items fail
- Log all errors for analysis

### Code Quality
- Add docstrings to all functions
- Use type hints
- Add progress bars (tqdm)
- Save intermediate results
- Make scripts executable

## Implementation Order

1. **First:** Dataset analysis script (understand data)
2. **Second:** Zero-shot testing (baseline performance)
3. **Third:** Fine-tuning (improve accuracy)
4. **Fourth:** Integration (end-to-end pipeline)
5. **Fifth:** Evaluation (comprehensive testing)

## Success Criteria

- Dataset analysis: Complete statistics and insights
- Zero-shot: >80% accuracy (target)
- Fine-tuned: >90% accuracy (target)
- Integration: End-to-end pipeline working
- Evaluation: Comprehensive metrics

## Notes

- Quantization will be done later (after all models built)
- Focus on accuracy and spatial reasoning first
- Use full images (640×640) for Qwen2-VL context
- Spatial reasoning is critical for this task

Please implement all scripts with proper error handling, logging, and documentation.
```

---

## 🎯 Simplified Quick Prompt (Copy-Paste Ready)

```
Build a complete Stage 2 conditional classification system using Qwen2-VL 7B for infrastructure inspection.

**Context:**
- Stage 1 model: `yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt` (82.3% mAP)
- Stage 2 dataset: `quen2-vl.yolov11/` (290 images, 9 classes, 640×640)
- Goal: Classify conditions (normal/blocked/damaged) with spatial reasoning
- Target: >90% accuracy

**Tasks:**

1. **Dataset Analysis** (`stage2_conditional/scripts/analyze_stage2_dataset.py`):
   - Parse YOLO polygon annotations → bounding boxes
   - Calculate class/condition/object-type distributions
   - Analyze spatial relationships (above/below, distances)
   - Identify dataset challenges
   - Save JSON + markdown report

2. **Zero-Shot Testing** (`stage2_conditional/scripts/test_qwen2vl_zeroshot.py`):
   - Load Qwen2-VL 7B model
   - Build spatial reasoning prompts
   - Test on validation set (47 images)
   - Calculate accuracy metrics (overall, per-class, per-condition)
   - Save results

3. **LoRA Fine-Tuning** (`stage2_conditional/scripts/finetune_qwen2vl_lora.py`):
   - Setup LoRA (r=16, alpha=32)
   - Create training dataset with spatial reasoning prompts
   - Train for 3-5 epochs (lr=2e-4)
   - Save fine-tuned model

4. **Stage 1 Integration** (`stage2_conditional/scripts/integrate_stage1_stage2.py`):
   - Load Stage 1 YOLO model
   - Run detection → extract spatial context
   - Run Stage 2 Qwen2-VL classification
   - Test end-to-end pipeline

5. **Evaluation** (`stage2_conditional/scripts/evaluate_qwen2vl.py`):
   - Test on test set
   - Comprehensive metrics
   - Error analysis

**Key Requirements:**
- Spatial reasoning prompts must emphasize relationships ("toe drain at end of slope drain")
- Use full images (not just crops) for Qwen2-VL context
- Handle all 9 conditional classes
- Save all results to experiments/ and metadata/
- Add proper error handling and logging

**Class Names:** ['Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged', 'rock toe', 'rock toe damaged', 'slope drain', 'slope drain blocked', 'slope drain damaged', 'vegetation']

Implement all scripts with documentation and error handling.
```

---

## 📝 Usage Instructions

1. **Copy the simplified prompt above**
2. **Paste into Cursor chat**
3. **Let Cursor implement all scripts**
4. **Run scripts in order:**
   - First: `python scripts/analyze_stage2_dataset.py`
   - Second: `python scripts/test_qwen2vl_zeroshot.py`
   - Third: `python scripts/finetune_qwen2vl_lora.py`
   - Fourth: `python scripts/integrate_stage1_stage2.py`
   - Fifth: `python scripts/evaluate_qwen2vl.py`

---

**This prompt will guide Cursor to build the complete Stage 2 system!**


