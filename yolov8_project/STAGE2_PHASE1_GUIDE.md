# Stage 2 Phase 1: Model Selection & Experimentation Guide

**Status:** Ready to Execute  
**Goal:** Test and compare Vision-Language Models to select the best one for conditional classification

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd yolov8_project
pip install -r requirements_stage2.txt
```

**Note:** If you're using Apple Silicon (M1/M2), you may need to install PyTorch with MPS support:
```bash
pip install torch torchvision torchaudio
```

### Step 2: Test a Single Model

Test Qwen2-VL (recommended) on 10 sample images:
```bash
python scripts/test_vlm_models.py --model qwen2-vl --images 10
```

### Step 3: Test All Models

Test all candidate models:
```bash
python scripts/test_vlm_models.py --model all --images 10
```

**Note:** This will download large model files (~14-26GB each). Ensure you have:
- Sufficient disk space (50GB+ recommended)
- Stable internet connection
- Sufficient RAM/VRAM for model loading

### Step 4: Compare Results

Generate comparison report:
```bash
python scripts/compare_vlm_models.py --results vlm_test_results.json
```

This creates `STAGE2_MODEL_COMPARISON.md` with detailed analysis.

---

## 📋 Detailed Instructions

### Testing Individual Models

#### Qwen2-VL 7B (Recommended)
```bash
python scripts/test_vlm_models.py --model qwen2-vl --images 20
```

#### InternVL2 8B
```bash
python scripts/test_vlm_models.py --model internvl2 --images 20
```

#### LLaVA-NeXT 13B
```bash
python scripts/test_vlm_models.py --model llava --images 20
```

### Testing Parameters

- `--model`: Model to test (`qwen2-vl`, `internvl2`, `llava`, or `all`)
- `--images`: Number of images to test on (default: 10)
- `--output`: Output JSON file path (default: `vlm_test_results.json`)

### Expected Output

The script will:
1. Download the model (first run only)
2. Load the model into memory
3. Test on sample images from Stage 2 dataset
4. Calculate accuracy, inference time, and memory usage
5. Save results to JSON file

**Example output:**
```
📦 Loading Qwen2-VL 7B...
   Using device: mps
✅ Qwen2-VL 7B loaded successfully!

🧪 Testing Qwen2-VL 7B on 10 images...
   [1/10] image1.jpg: GT=normal, Pred=normal, Correct=True
   [2/10] image2.jpg: GT=damaged, Pred=damaged, Correct=True
   ...

📊 Qwen2-VL 7B Results:
   Accuracy: 85.00%
   Avg Inference Time: 1.234s
   Memory Usage: 12.45 GB
   Load Time: 45.67s
```

---

## 📊 Understanding Results

### Metrics Explained

1. **Accuracy**: Overall classification accuracy (target: >80%)
2. **Avg Inference Time**: Time per image (target: <2s)
3. **Memory Usage**: Peak memory consumption
4. **Load Time**: Model loading time (one-time cost)

### Per-Condition Accuracy

The script tracks accuracy for each condition:
- `normal`: Normal condition classification
- `damaged`: Damage detection accuracy
- `blocked`: Blockage detection accuracy

### Per-Object-Type Accuracy

The script tracks accuracy for each object type:
- `rock_toe`: Rock toe condition classification
- `slope_drain`: Slope drain condition classification
- `toe_drain`: Toe drain condition classification

---

## 🎯 Success Criteria

A model meets Phase 1 success criteria if:
- ✅ **Accuracy ≥ 80%** (zero-shot)
- ✅ **Inference time < 2s** per image
- ✅ **Supports quantization** (INT4/INT8)
- ✅ **Memory usage < 16GB** (FP16) or < 8GB (quantized)

---

## ⚠️ Troubleshooting

### Issue: "transformers not installed"
**Solution:**
```bash
pip install transformers torch pillow
```

### Issue: "Out of memory"
**Solutions:**
1. Reduce number of test images: `--images 5`
2. Use CPU instead of GPU (slower but less memory)
3. Test models one at a time instead of all at once

### Issue: "Model download fails"
**Solutions:**
1. Check internet connection
2. Ensure sufficient disk space (50GB+)
3. Try downloading manually from HuggingFace

### Issue: "Device not found" (CUDA/MPS)
**Solutions:**
- **CUDA:** Ensure NVIDIA GPU drivers are installed
- **MPS (Apple Silicon):** Should work automatically on M1/M2/M3
- **CPU:** Will fall back automatically but will be slower

### Issue: "Stage 2 dataset not found"
**Solution:** Ensure Stage 2 dataset is at:
```
../STEP 2- Conditional classes.v1-stage-2--1.yolov11/
```

---

## 📁 Output Files

After running tests, you'll have:

1. **`vlm_test_results.json`**: Raw test results in JSON format
2. **`STAGE2_MODEL_COMPARISON.md`**: Detailed comparison report (after running compare script)

---

## 🔄 Next Steps After Phase 1

Once you've selected the best model:

1. **Phase 2: Dataset Preparation**
   - Extract object crops from Stage 1 detections
   - Organize by conditional classes
   - Create train/val/test splits

2. **Phase 3: Fine-Tuning** (if needed)
   - Fine-tune selected model if zero-shot <85%
   - Use LoRA for efficient fine-tuning

3. **Phase 4: Integration**
   - Integrate with Stage 1 detection pipeline
   - Test end-to-end system

4. **Phase 5: Quantization**
   - Quantize model for A30 deployment
   - Optimize inference speed

---

## 📚 Additional Resources

- **VLM Research:** See `STAGE2_VLM_RESEARCH.md` for detailed model information
- **Model Documentation:**
  - Qwen2-VL: https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
  - InternVL2: https://huggingface.co/OpenGVLab/InternVL2-8B
  - LLaVA: https://huggingface.co/llava-hf/llava-1.5-13b-hf

---

**Last Updated:** 2025-01-27  
**Status:** Ready for Testing


