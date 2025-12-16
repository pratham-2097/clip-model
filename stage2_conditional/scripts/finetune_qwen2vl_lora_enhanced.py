#!/usr/bin/env python3
"""
Enhanced Fine-tune Qwen2-VL 7B with LoRA for conditional classification
Addresses class imbalance, spatial reasoning, and rare class handling
Based on dataset analysis findings
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from tqdm import tqdm
import numpy as np
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

DATASET_DIR = Path(__file__).parent.parent.parent / "quen2-vl.yolov11"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR = Path(__file__).parent.parent / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
    'rock toe', 'rock toe damaged',
    'slope drain', 'slope drain blocked', 'slope drain damaged',
    'vegetation'
]

# Class weights based on dataset analysis (inverse frequency)
# Most common: rock toe damaged (366) → weight 1.0
# Least common: Toe drain (52) → weight 7.04
CLASS_WEIGHTS = {
    'Toe drain': 7.04,
    'Toe drain- Blocked': 4.69,
    'Toe drain- Damaged': 4.82,
    'rock toe': 2.39,
    'rock toe damaged': 1.0,  # Most common, base weight
    'slope drain': 1.42,
    'slope drain blocked': 3.77,
    'slope drain damaged': 2.47,
    'vegetation': 1.54
}

# Condition weights (blocked is rare)
CONDITION_WEIGHTS = {
    'normal': 1.0,
    'damaged': 1.19,
    'blocked': 4.0  # Rare, high weight
}

def get_spatial_rules_for_class(class_name: str) -> str:
    """Get data-driven spatial rules based on dataset analysis."""
    
    rules = {
        'Toe drain': """Spatial patterns from data:
- Toe drain is typically at the BOTTOM of images (Y-position: 0.66-0.79)
- Toe drain is ABOVE slope drain 57.7% of the time
- Toe drain is often BELOW rock toe damaged (45.8% of co-occurrences)
- Consider: Is this toe drain at the bottom/end of a slope drain?""",
        
        'Toe drain- Blocked': """Spatial patterns from data:
- Blocked toe drain is at the BOTTOM (Y-position: 0.79)
- Often appears with slope drain (89 co-occurrences)
- Consider: Is this blocked toe drain at the bottom/end of a slope drain?""",
        
        'Toe drain- Damaged': """Spatial patterns from data:
- Damaged toe drain is at the BOTTOM (Y-position: 0.77)
- Often ABOVE slope drain blocked (86.2% of co-occurrences)
- Consider: Is this damaged toe drain at the bottom/end of a slope drain?""",
        
        'rock toe': """Spatial patterns from data:
- Rock toe is in the MIDDLE of images (Y-position: 0.53)
- Often ABOVE slope drain (46.0% of co-occurrences)
- Consider: Is this rock toe positioned above a toe drain?""",
        
        'rock toe damaged': """Spatial patterns from data:
- Rock toe damaged is at the BOTTOM (Y-position: 0.69)
- Strong co-occurrence with slope drain (444 instances)
- Often ABOVE slope drain (49.3% of co-occurrences)
- Consider: Is this rock toe damaged above a toe drain or slope drain?""",
        
        'slope drain': """Spatial patterns from data:
- Slope drain is in the MIDDLE of images (Y-position: 0.44)
- Often has objects ABOVE it (toe drain, rock toe)
- Consider: Are there objects above or below this slope drain?""",
        
        'slope drain blocked': """Spatial patterns from data:
- Blocked slope drain is in the MIDDLE (Y-position: 0.51)
- Often has rock toe ABOVE it (61.7% of co-occurrences)
- Consider: Are there objects above this blocked slope drain?""",
        
        'slope drain damaged': """Spatial patterns from data:
- Damaged slope drain is in the MIDDLE (Y-position: 0.52)
- Often has rock toe damaged ABOVE it (50.0% of co-occurrences)
- Consider: Are there objects above or below this damaged slope drain?""",
        
        'vegetation': """Spatial patterns from data:
- Vegetation is in the MIDDLE (Y-position: 0.57)
- Often appears with multiple objects
- Consider: What is the spatial context of this vegetation?"""
    }
    
    return rules.get(class_name, "Consider spatial relationships with other objects.")

def get_condition_weight(class_name: str) -> float:
    """Get condition weight for a class."""
    if 'Blocked' in class_name or 'blocked' in class_name:
        return CONDITION_WEIGHTS['blocked']
    elif 'Damaged' in class_name or 'damaged' in class_name:
        return CONDITION_WEIGHTS['damaged']
    else:
        return CONDITION_WEIGHTS['normal']

def parse_polygon_to_bbox(polygon_coords: List[float]) -> Tuple[float, float, float, float]:
    """Convert polygon coordinates to bounding box (x_center, y_center, width, height)."""
    coords = np.array(polygon_coords).reshape(-1, 2)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def build_enhanced_training_prompt(class_name: str, all_objects: List[str] = None, bbox_y: float = None) -> str:
    """Build enhanced training prompt with data-driven spatial reasoning."""
    
    objects_context = ""
    if all_objects:
        objects_context = f"\nOther objects in image: {', '.join(all_objects)}"
    
    # Position context
    position_context = ""
    if bbox_y is not None:
        if bbox_y > 0.6:
            position_context = "\nPosition: This object is at the BOTTOM of the image (typical for toe drains and rock toe damaged)."
        elif bbox_y < 0.4:
            position_context = "\nPosition: This object is at the TOP of the image."
        else:
            position_context = "\nPosition: This object is in the MIDDLE of the image (typical for slope drains)."
    
    # Spatial rules
    spatial_rules = get_spatial_rules_for_class(class_name)
    
    prompt = f"""Analyze this infrastructure inspection image and classify the condition of the object.{position_context}

Consider:
1. Visual appearance: Is it damaged, blocked, or in normal condition?
   - Look for cracks, wear, obstructions, or structural issues
   - Assess the physical state carefully
2. Spatial relationships with other objects:
{spatial_rules}
3. Context: What is the overall state of the infrastructure?{objects_context}

Classify as: {class_name}"""
    
    return prompt

class EnhancedConditionalClassificationDataset(Dataset):
    """Enhanced dataset with oversampling and position encoding."""
    
    def __init__(self, dataset_dir: Path, split: str = 'train', processor=None, oversample_rare=True):
        self.dataset_dir = dataset_dir
        self.split = split
        self.processor = processor
        self.oversample_rare = oversample_rare
        
        self.images_dir = self.dataset_dir / split / 'images'
        self.labels_dir = self.dataset_dir / split / 'labels'
        
        # Load all image-label pairs with bbox info
        self.samples = []
        for img_file in sorted(self.images_dir.glob('*.jpg')):
            label_file = self.labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                # Read all labels for context
                all_classes = []
                class_bboxes = {}  # Store bbox for each class
                
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts and len(parts) >= 3:
                            try:
                                class_idx = int(parts[0])
                                if 0 <= class_idx < len(CLASS_NAMES):
                                    class_name = CLASS_NAMES[class_idx]
                                    all_classes.append(class_name)
                                    
                                    # Parse bbox
                                    polygon_coords = [float(x) for x in parts[1:]]
                                    if len(polygon_coords) >= 6:
                                        _, y_center, _, _ = parse_polygon_to_bbox(polygon_coords)
                                        class_bboxes[class_name] = y_center
                            except (ValueError, IndexError):
                                continue
                
                # Create sample for each object
                for class_name in all_classes:
                    bbox_y = class_bboxes.get(class_name, 0.5)  # Default to middle
                    weight = CLASS_WEIGHTS.get(class_name, 1.0)
                    condition_weight = get_condition_weight(class_name)
                    total_weight = weight * condition_weight
                    
                    self.samples.append({
                        'image_path': img_file,
                        'class_name': class_name,
                        'all_objects': all_classes,
                        'bbox_y': bbox_y,
                        'weight': total_weight
                    })
        
        # Oversample rare classes
        if self.oversample_rare and split == 'train':
            self.samples = self._oversample_rare_classes(self.samples)
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _oversample_rare_classes(self, samples: List[Dict]) -> List[Dict]:
        """Oversample rare classes based on weights."""
        # Calculate target count (use most common class as baseline)
        class_counts = Counter([s['class_name'] for s in samples])
        max_count = max(class_counts.values())
        
        # Oversample each class to reach target
        oversampled = []
        for sample in samples:
            oversampled.append(sample)
            
            # Oversample based on weight (higher weight = more oversampling)
            weight = sample['weight']
            if weight > 2.0:  # Only oversample rare classes
                oversample_factor = int(weight)
                for _ in range(oversample_factor - 1):
                    oversampled.append(sample)
        
        return oversampled
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Build enhanced prompt
        prompt = build_enhanced_training_prompt(
            sample['class_name'],
            [obj for obj in sample['all_objects'] if obj != sample['class_name']],
            sample['bbox_y']
        )
        
        # Format for Qwen2-VL
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]
        
        # Process with processor (updated API)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs - ensure proper format for Qwen2-VL
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Remove batch dimension from tensors (processor adds [1, ...] but collator expects [...])
        # This allows the DataCollator to properly batch multiple samples
        for key in ['input_ids', 'attention_mask', 'pixel_values']:
            if key in inputs and isinstance(inputs[key], torch.Tensor):
                if inputs[key].dim() > 1 and inputs[key].shape[0] == 1:
                    inputs[key] = inputs[key].squeeze(0)  # [1, ...] -> [...]
        
        # Ensure image_grid_thw is a tensor with shape [1, 3] for single image
        # Keep batch dimension here as it will be stacked by collator
        if 'image_grid_thw' in inputs:
            grid_thw = inputs['image_grid_thw']
            if isinstance(grid_thw, torch.Tensor):
                # Ensure shape is [1, 3]
                if grid_thw.dim() == 1:
                    if len(grid_thw) == 3:
                        inputs['image_grid_thw'] = grid_thw.unsqueeze(0)  # [3] -> [1, 3]
                    else:
                        inputs['image_grid_thw'] = torch.tensor([[1, 1, 1]], dtype=torch.long)
                elif grid_thw.dim() == 2:
                    # Already [1, 3] - keep as is
                    if grid_thw.shape[0] != 1:
                        inputs['image_grid_thw'] = grid_thw[:1]  # Take first
                elif grid_thw.dim() > 2:
                    inputs['image_grid_thw'] = torch.tensor([[1, 1, 1]], dtype=torch.long)
            else:
                inputs['image_grid_thw'] = torch.tensor([[1, 1, 1]], dtype=torch.long)
        else:
            inputs['image_grid_thw'] = torch.tensor([[1, 1, 1]], dtype=torch.long)
        
        # Add target label
        target_text = sample['class_name']
        target_inputs = self.processor(
            text=[target_text],
            padding=True,
            return_tensors="pt"
        )
        
        # Combine inputs
        labels = target_inputs['input_ids']
        # Remove batch dimension from labels too
        if labels.dim() > 1 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        inputs['labels'] = labels
        inputs['weight'] = torch.tensor(sample['weight'], dtype=torch.float32)
        
        return inputs

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

class WeightedTrainer(Trainer):
    """Custom trainer with weighted loss."""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        weights = inputs.get("weight", None)
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        
        # Compute per-sample loss
        per_sample_loss = loss_fct(flat_logits, flat_labels)
        
        # Apply weights if available
        if weights is not None:
            # Reshape weights to match loss shape
            weight_tensor = weights.view(-1, 1).expand_as(per_sample_loss.view(-1, 1))
            per_sample_loss = per_sample_loss * weight_tensor.squeeze()
        
        # Average
        loss = per_sample_loss.mean()
        
        return (loss, outputs) if return_outputs else loss

def main():
    print("="*80)
    print("ENHANCED QWEN2-VL 7B LoRA FINE-TUNING")
    print("With Class Weighting, Spatial Reasoning, and Oversampling")
    print("="*80)
    
    # Check for available compute device (CUDA for NVIDIA, MPS for Apple Silicon, or CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load dataset analysis
    analysis_file = METADATA_DIR / "dataset_analysis.json"
    if analysis_file.exists():
        with open(analysis_file, 'r') as f:
            analysis = json.load(f)
        print(f"\n📊 Dataset Analysis Loaded:")
        print(f"   Total instances: {analysis.get('total_instances', 'N/A')}")
        print(f"   Class imbalance ratio: 7.04x")
    
    # Load model
    print("\n1. Loading Qwen2-VL 7B model...")
    print("   ⚠️  Note: 7B model is large. Using CPU offloading for MPS to fit in memory.")
    try:
        # For MPS, use CPU offloading to handle large model
        # This splits the model between CPU and MPS to fit in memory
        if device == "mps":
            # Use CPU offloading: most layers on CPU, only active layers on MPS
            # This is slower but will fit in memory
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float32,
                device_map="cpu",  # Load to CPU first
                low_cpu_mem_usage=True
            )
            # Enable gradient checkpointing to save memory
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
            print("   ℹ️  Model loaded to CPU. Will use CPU for training (MPS memory insufficient).")
        elif device == "cuda":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            # CPU
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
        
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Setup LoRA
    print("\n2. Setting up LoRA...")
    model = setup_lora(model)
    print("✅ LoRA configured")
    
    # Create datasets with enhancements
    print("\n3. Creating enhanced datasets...")
    train_dataset = EnhancedConditionalClassificationDataset(
        DATASET_DIR, split='train', processor=processor, oversample_rare=True
    )
    val_dataset = EnhancedConditionalClassificationDataset(
        DATASET_DIR, split='valid', processor=processor, oversample_rare=False
    )
    print(f"✅ Train: {len(train_dataset)} samples (with oversampling)")
    print(f"✅ Val: {len(val_dataset)} samples")
    
    # Training arguments
    print("\n4. Configuring training...")
    # For MPS, we're using CPU due to memory constraints, so adjust device accordingly
    training_device = "cpu" if device == "mps" else device
    
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "qwen2vl_lora_enhanced_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=(device == "cuda"),  # Only use fp16 for CUDA
        gradient_checkpointing=True,  # Enable to save memory
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=50,
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0  # Reduce memory usage
    )
    
    # Custom data collator that handles image_grid_thw properly
    class Qwen2VLDataCollator(DataCollatorForLanguageModeling):
        def __call__(self, features):
            # Handle image_grid_thw separately before calling parent
            image_grid_thws = []
            for feature in features:
                if 'image_grid_thw' in feature:
                    grid_thw = feature.pop('image_grid_thw')
                    # Ensure it's a tensor with shape [1, 3]
                    if isinstance(grid_thw, torch.Tensor):
                        if grid_thw.dim() == 1 and len(grid_thw) == 3:
                            image_grid_thws.append(grid_thw.unsqueeze(0))  # [3] -> [1, 3]
                        elif grid_thw.dim() == 2:
                            # Already [1, 3] or [N, 3]
                            if grid_thw.shape[0] == 1:
                                image_grid_thws.append(grid_thw)
                            else:
                                image_grid_thws.append(grid_thw[:1])  # Take first
                        else:
                            image_grid_thws.append(torch.tensor([[1, 1, 1]], dtype=torch.long))
                    else:
                        # Not a tensor - create default
                        image_grid_thws.append(torch.tensor([[1, 1, 1]], dtype=torch.long))
                else:
                    image_grid_thws.append(torch.tensor([[1, 1, 1]], dtype=torch.long))
            
            # Call parent collator
            batch = super().__call__(features)
            
            # Stack image_grid_thw tensors: [tensor([1,3]), tensor([1,3])] -> tensor([2, 3])
            if image_grid_thws:
                batch['image_grid_thw'] = torch.cat(image_grid_thws, dim=0)  # Stack along batch dimension
            
            return batch
    
    data_collator = Qwen2VLDataCollator(
        tokenizer=processor.tokenizer,
        mlm=False
    )
    
    # Trainer with weighted loss
    print("\n5. Initializing weighted trainer...")
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n6. Starting training with enhancements...")
    print("   - Class-weighted loss (handles 7.04x imbalance)")
    print("   - Enhanced spatial reasoning prompts")
    print("   - Oversampling of rare classes")
    print("   - Position encoding in prompts")
    print("="*80)
    trainer.train()
    
    # Save final model
    print("\n7. Saving final model...")
    final_model_dir = MODELS_DIR / "qwen2vl_lora_enhanced_final"
    model.save_pretrained(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))
    print(f"✅ Model saved to {final_model_dir}")
    
    # Save training info
    training_info = {
        'model': 'Qwen2-VL-7B-Instruct',
        'fine_tuning_method': 'LoRA Enhanced',
        'enhancements': [
            'Class-weighted loss (7.04x imbalance handling)',
            'Enhanced spatial reasoning prompts (data-driven)',
            'Oversampling of rare classes',
            'Position encoding (Y-position in prompts)',
            'Condition-aware weighting (blocked: 4.0x)'
        ],
        'lora_r': 16,
        'lora_alpha': 32,
        'learning_rate': 2e-4,
        'epochs': 3,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'final_model_path': str(final_model_dir),
        'class_weights': CLASS_WEIGHTS,
        'condition_weights': CONDITION_WEIGHTS
    }
    
    info_file = EXPERIMENTS_DIR / "training_info_enhanced.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✅ Training info saved to {info_file}")
    print("\n" + "="*80)
    print("✅ Enhanced fine-tuning complete!")

if __name__ == '__main__':
    main()

