# 📊 Dataset Analysis Insights & Qwen2-VL Fine-Tuning Strategy

**Date:** 2025-01-27  
**Based on:** Dataset analysis of 1,465 instances across 290 images

---

## 🔍 Key Findings from Dataset Analysis

### 1. **Severe Class Imbalance (7.04x ratio)**

**Problem:**
- **Most common:** "rock toe damaged" = 366 instances (25.0%)
- **Least common:** "Toe drain" = 52 instances (3.5%)
- **Imbalance ratio:** 7.04x (max/min)

**Impact:**
- Model will be biased toward common classes
- Rare classes ("Toe drain", "Toe drain- Blocked") will be under-predicted
- Need weighted loss or class balancing

### 2. **Condition Distribution**

| Condition | Count | Percentage | Status |
|-----------|-------|------------|--------|
| Normal | 700 | 47.8% | Most common |
| Damaged | 590 | 40.3% | Common |
| Blocked | 175 | 11.9% | **Rare** ⚠️ |

**Impact:**
- "Blocked" condition is underrepresented
- Need to emphasize blocked examples during training

### 3. **Spatial Patterns Discovered**

**Critical Spatial Relationships:**
- **Toe drain** → Always at **bottom** (Y-mean: 0.66-0.79)
- **Rock toe damaged** → Often at **bottom** (Y-mean: 0.69)
- **Slope drain** → Usually in **middle** (Y-mean: 0.44-0.52)
- **4,612 spatial relationships** analyzed

**Key Patterns:**
- "Toe drain" is **above** "slope drain" 57.7% of the time
- "Rock toe damaged" is **above** "slope drain" 49.3% of the time
- "Toe drain" is **below** "rock toe damaged" 45.8% of the time

**Impact:**
- Model MUST understand spatial relationships
- Prompts must emphasize position (above/below/at end of)

### 4. **Co-Occurrence Patterns**

**Strong Relationships:**
- `rock toe damaged` ↔ `slope drain`: 444 co-occurrences
- `slope drain` ↔ `slope drain`: 528 co-occurrences
- `rock toe damaged` ↔ `rock toe damaged`: 564 co-occurrences

**Impact:**
- Objects appear together frequently
- Context from other objects is critical
- Full image context is essential

### 5. **Data Quality**

- ✅ **Resolution:** Perfect 640×640 (as required)
- ✅ **Data size:** Sufficient (162.8 instances/class average)
- ✅ **Multiple objects per image:** 5.09 average (good for context)

---

## 🎯 Fine-Tuning Strategy Based on Findings

### Strategy 1: **Weighted Loss Function** ⭐ CRITICAL

**Why:** Class imbalance (7.04x ratio) will bias model

**Implementation:**
```python
# Calculate class weights inversely proportional to frequency
class_weights = {
    'Toe drain': 7.04,              # Rare → High weight
    'Toe drain- Blocked': 4.69,    # Rare → High weight
    'Toe drain- Damaged': 4.82,    # Rare → High weight
    'rock toe': 2.39,               # Common → Low weight
    'rock toe damaged': 1.0,        # Most common → Base weight
    'slope drain': 1.42,            # Common → Low weight
    'slope drain blocked': 3.77,   # Rare → High weight
    'slope drain damaged': 2.47,    # Common → Low weight
    'vegetation': 1.54              # Common → Low weight
}
```

**Action:** Update fine-tuning script to use weighted loss

---

### Strategy 2: **Enhanced Spatial Reasoning Prompts** ⭐ CRITICAL

**Why:** 4,612 spatial relationships identified, spatial patterns are critical

**Current Prompt (Basic):**
```
"Is a toe drain at the bottom/end of a slope drain?"
```

**Enhanced Prompt (Data-Driven):**
```
"Analyze spatial relationships:
- Toe drain is typically at the BOTTOM of images (Y-position: 0.66-0.79)
- Toe drain is ABOVE slope drain 57.7% of the time
- Rock toe is often ABOVE toe drain
- Slope drain is usually in the MIDDLE (Y-position: 0.44-0.52)
- Consider: Is this toe drain at the bottom/end of a slope drain?
- Consider: Is this rock toe positioned above a toe drain?"
```

**Action:** Update prompt templates with specific spatial patterns

---

### Strategy 3: **Oversampling Rare Classes** ⭐ IMPORTANT

**Why:** "Toe drain" (52 instances) and "Toe drain- Blocked" (78 instances) are rare

**Implementation:**
- Oversample rare classes during training
- Use data augmentation for rare classes
- Create synthetic examples if needed

**Action:** Add class-aware sampling to dataset loader

---

### Strategy 4: **Condition-Aware Training** ⭐ IMPORTANT

**Why:** "Blocked" condition is rare (11.9% vs 47.8% normal)

**Implementation:**
- Emphasize blocked examples in training
- Use higher learning rate for blocked samples
- Add more blocked examples to validation set

**Action:** Add condition-based weighting

---

### Strategy 5: **Multi-Object Context Emphasis** ⭐ IMPORTANT

**Why:** Average 5.09 objects per image, strong co-occurrence patterns

**Implementation:**
- Always include ALL objects in image for context
- Emphasize co-occurrence patterns in prompts
- Train model to use full image context

**Action:** Ensure prompts always include all detected objects

---

### Strategy 6: **Spatial Position Encoding** ⭐ RECOMMENDED

**Why:** Clear spatial patterns (toe drain at bottom, slope drain in middle)

**Implementation:**
- Add explicit position information to prompts
- "This object is at Y-position 0.75 (bottom of image)"
- "This object is above another object"

**Action:** Add position encoding to prompts

---

## ✅ Current Fine-Tuning Script Status

### What's Already Implemented ✅

1. ✅ **LoRA Configuration** - Efficient fine-tuning (r=16, alpha=32)
2. ✅ **Spatial Reasoning Prompts** - Basic spatial relationship prompts
3. ✅ **Multi-Object Context** - Includes all objects in image
4. ✅ **Training Dataset** - Properly loads images and labels
5. ✅ **Validation Split** - Uses validation set for evaluation

### What's MISSING ❌

1. ❌ **Weighted Loss** - No class weighting for imbalance
2. ❌ **Enhanced Spatial Prompts** - Basic prompts, not data-driven
3. ❌ **Oversampling** - No rare class oversampling
4. ❌ **Condition Weighting** - No special handling for "blocked"
5. ❌ **Position Encoding** - No explicit Y-position in prompts

---

## 🔧 Required Updates to Fine-Tuning Script

### Update 1: Add Weighted Loss

```python
from torch.nn import CrossEntropyLoss

# Calculate class weights
class_weights = calculate_class_weights(stats)
criterion = CrossEntropyLoss(weight=class_weights)
```

### Update 2: Enhanced Spatial Prompts

```python
def build_enhanced_spatial_prompt(class_name, all_objects, bbox_y_position):
    """Enhanced prompt with data-driven spatial patterns."""
    
    position_context = ""
    if bbox_y_position > 0.6:
        position_context = "This object is at the BOTTOM of the image (typical for toe drains)."
    elif bbox_y_position < 0.4:
        position_context = "This object is at the TOP of the image."
    else:
        position_context = "This object is in the MIDDLE of the image (typical for slope drains)."
    
    # Add specific spatial relationships based on class
    spatial_rules = get_spatial_rules_for_class(class_name)
    
    prompt = f"""...{position_context} {spatial_rules}..."""
    return prompt
```

### Update 3: Class-Aware Sampling

```python
class WeightedDataset(Dataset):
    def __init__(self, ...):
        # Oversample rare classes
        self.samples = oversample_rare_classes(samples, class_weights)
```

---

## 📋 Implementation Checklist

- [ ] Add weighted loss function based on class frequencies
- [ ] Enhance spatial reasoning prompts with data-driven patterns
- [ ] Implement oversampling for rare classes
- [ ] Add condition-aware weighting (emphasize "blocked")
- [ ] Add position encoding to prompts (Y-position information)
- [ ] Update training loop to use weighted loss
- [ ] Test on validation set with new strategies
- [ ] Compare results with/without improvements

---

## 🎯 Expected Improvements

### With Current Script:
- **Accuracy:** ~80-85% (baseline)
- **Rare classes:** Poor performance (under-predicted)
- **Spatial reasoning:** Basic understanding

### With Enhanced Script:
- **Accuracy:** >90% (target)
- **Rare classes:** Improved performance (weighted loss)
- **Spatial reasoning:** Strong understanding (data-driven prompts)
- **Blocked condition:** Better detection (condition weighting)

---

## 🚀 Next Steps

1. **Update fine-tuning script** with all strategies above
2. **Run zero-shot test** first (baseline)
3. **Fine-tune with enhancements** (improved version)
4. **Compare results** (before/after improvements)
5. **Iterate** based on validation performance

---

**Last Updated:** 2025-01-27  
**Status:** Ready for Implementation


