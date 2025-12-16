# Stage 2: Vision-Language Model Research for Conditional Classification

**Date:** 2025-01-27  
**Status:** Phase 1 - Model Selection & Experimentation

---

## 🎯 Objective

Identify the best Vision-Language Model (VLM) for conditional classification of infrastructure objects. The model must:
- Classify object conditions: `normal`, `damaged`, `blocked`
- Use contextual reasoning (spatial relationships, image context)
- Support quantization for Nvidia A30 deployment
- Achieve >80% zero-shot accuracy

---

## 📊 Candidate Models

### 1. Qwen2-VL 7B ⭐ **PRIMARY RECOMMENDATION**

**Model:** `Qwen/Qwen2-VL-7B-Instruct`

**Strengths:**
- ✅ **Best reasoning capability** - State-of-the-art spatial understanding
- ✅ **Excellent quantization** - Supports INT4/INT8 (4GB quantized)
- ✅ **Strong visual grounding** - Understands spatial relationships
- ✅ **Context-aware** - Can reason about object relationships
- ✅ **Fast inference** - Optimized architecture

**Weaknesses:**
- ⚠️ Requires ~14GB VRAM (FP16) / ~4GB (INT4)
- ⚠️ May need fine-tuning for domain-specific tasks

**Quantization:**
- INT4: ~4GB, minimal accuracy loss
- INT8: ~7GB, near-FP16 accuracy

**Best For:**
- Conditional classification with context
- Spatial reasoning (e.g., "toe drain at bottom of slope drain")
- Production deployment with quantization

**HuggingFace:** https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct

---

### 2. InternVL2 8B

**Model:** `OpenGVLab/InternVL2-8B`

**Strengths:**
- ✅ **Best visual detail recognition** - Excellent at fine-grained features
- ✅ **Strong on damaged/blocked detection** - Good at identifying defects
- ✅ **High resolution support** - Can process detailed images

**Weaknesses:**
- ⚠️ Larger model size (~16GB FP16)
- ⚠️ Slower inference than Qwen2-VL
- ⚠️ Less efficient quantization

**Best For:**
- Fine-grained visual analysis
- Damage detection requiring high detail
- When visual precision is priority

**HuggingFace:** https://huggingface.co/OpenGVLab/InternVL2-8B

---

### 3. LLaVA-NeXT 13B

**Model:** `llava-hf/llava-1.5-13b-hf`

**Strengths:**
- ✅ **Strong reasoning** - Good at multi-step reasoning
- ✅ **Explainable** - Provides reasoning explanations
- ✅ **Well-documented** - Extensive community support

**Weaknesses:**
- ⚠️ Large model (~26GB FP16)
- ⚠️ Slower inference
- ⚠️ May be overkill for classification task

**Best For:**
- When explainability is important
- Complex reasoning tasks
- Research and experimentation

**HuggingFace:** https://huggingface.co/llava-hf/llava-1.5-13b-hf

---

### 4. CogVLM2 19B

**Model:** `THUDM/cogvlm2-19b-chat`

**Strengths:**
- ✅ **Excellent visual grounding** - Strong spatial understanding
- ✅ **Large context window** - Can process full images with context

**Weaknesses:**
- ⚠️ Very large (~38GB FP16)
- ⚠️ Slow inference
- ⚠️ Resource-intensive

**Best For:**
- Research scenarios with abundant resources
- When maximum accuracy is needed regardless of speed

**HuggingFace:** https://huggingface.co/THUDM/cogvlm2-19b-chat

---

### 5. Florence-2 Large

**Model:** `microsoft/Florence-2-large`

**Strengths:**
- ✅ **Efficient** - Smaller, faster than others
- ✅ **Unified tasks** - Can do multiple vision tasks
- ✅ **Good quantization** - Efficient deployment

**Weaknesses:**
- ⚠️ May lack reasoning depth of larger models
- ⚠️ Less context-aware than Qwen2-VL

**Best For:**
- Resource-constrained environments
- Fast inference requirements
- When efficiency > reasoning depth

**HuggingFace:** https://huggingface.co/microsoft/Florence-2-large

---

## 📋 Comparison Matrix

| Model | Size (FP16) | Quantized (INT4) | Reasoning | Visual Detail | Speed | Quantization | Recommendation |
|-------|-------------|------------------|-----------|---------------|-------|--------------|----------------|
| **Qwen2-VL 7B** | 14GB | 4GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ **BEST** |
| InternVL2 8B | 16GB | 6GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Alternative |
| LLaVA-NeXT 13B | 26GB | 8GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | Research |
| CogVLM2 19B | 38GB | 12GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ | Research |
| Florence-2 Large | 8GB | 3GB | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Efficient |

---

## 🧪 Testing Strategy

### Phase 1: Zero-Shot Testing
1. Test each candidate on 10-20 sample images from Stage 2 dataset
2. Use standardized prompts for conditional classification
3. Measure accuracy, inference time, memory usage
4. Compare reasoning quality (subjective)

### Phase 2: Prompt Engineering
1. Optimize prompts for best-performing models
2. Test context-aware prompts
3. Evaluate spatial relationship understanding

### Phase 3: Fine-Tuning (if needed)
1. If zero-shot <80%, proceed with LoRA fine-tuning
2. Fine-tune on prepared conditional class dataset
3. Validate accuracy improvement

---

## 📝 Test Prompts

### Standard Prompt Template
```
"Analyze this [object_type] (rock_toe/slope_drain/toe_drain). 
Is it: A) Normal, B) Damaged, C) Blocked? 
Consider visible damage, erosion, vegetation, or debris."
```

### Context-Aware Prompt Template
```
"This [object_type] is located [context description]. 
For example, a toe drain at the bottom of a slope drain. 
What is its condition: normal, damaged, or blocked?"
```

### Multi-Step Reasoning Prompt
```
"Step 1: Identify if this is a [object_type].
Step 2: Check for cracks, erosion, or structural damage → indicates 'damaged'.
Step 3: Check for vegetation, debris, or blockages → indicates 'blocked'.
Step 4: If neither, classify as 'normal'.
What is the final classification?"
```

---

## 🎯 Success Criteria

**Model Selection Phase:**
- ✅ Test at least 3 VLM candidates
- ✅ Identify best model with >80% zero-shot accuracy
- ✅ Model supports quantization (INT4/INT8)
- ✅ Inference time <2s per image (unquantized)
- ✅ Memory usage <16GB (FP16) or <8GB (quantized)

**Final System:**
- ✅ Conditional classification accuracy >85%
- ✅ End-to-end pipeline working
- ✅ Quantized model ready for deployment

---

## 📚 References

- Qwen2-VL Paper: https://arxiv.org/abs/2409.12169
- InternVL2 Paper: https://arxiv.org/abs/2402.03371
- LLaVA Paper: https://arxiv.org/abs/2304.08485
- CogVLM2 Paper: https://arxiv.org/abs/2401.06061
- Florence-2 Paper: https://arxiv.org/abs/2311.06242

---

## 🔄 Next Steps

1. **Setup Testing Framework** - Create `test_vlm_models.py`
2. **Test Top 3 Candidates** - Qwen2-VL, InternVL2, LLaVA-NeXT
3. **Compare Results** - Generate comparison report
4. **Select Best Model** - Based on accuracy, speed, quantization
5. **Proceed to Phase 2** - Dataset preparation and fine-tuning

---

**Last Updated:** 2025-01-27  
**Status:** Research Complete - Ready for Testing


