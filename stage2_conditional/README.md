# Stage 2: Conditional Classification with Qwen2-VL 7B

**Status:** Ready for Implementation  
**Focus:** Data Analysis → Fine-Tuning → Integration → Evaluation

---

## 🎯 Overview

Stage 2 implements conditional classification using Qwen2-VL 7B to classify infrastructure objects into 9 conditional classes:
- Normal, blocked, and damaged conditions for toe drain, slope drain, and rock toe
- Spatial reasoning to understand object relationships

---

## 📁 Structure

```
stage2_conditional/
├── README.md                          # This file
├── STAGE2_IMPLEMENTATION_PLAN.md      # Detailed plan
├── requirements.txt                   # Dependencies
├── scripts/                           # Implementation scripts
│   ├── analyze_stage2_dataset.py      # Dataset analysis
│   ├── test_qwen2vl_zeroshot.py       # Zero-shot testing
│   ├── finetune_qwen2vl_lora.py       # LoRA fine-tuning
│   └── integrate_stage1_stage2.py     # Integration pipeline
├── experiments/                       # Results and logs
├── models/                            # Trained models
├── metadata/                          # Analysis reports
└── ...
```

---

## 🚀 Quick Start

### 1. Setup
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
python scripts/integrate_stage1_stage2.py --split valid --num_images 10
```

---

## 📊 Goals

- **Accuracy:** >90% overall
- **Spatial Reasoning:** >85%
- **Conditional Classification:** >88% per condition
- **Integration:** Seamless with Stage 1 detector

---

**See STAGE2_IMPLEMENTATION_PLAN.md for detailed plan**


