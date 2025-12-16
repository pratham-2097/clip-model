#!/usr/bin/env python3
"""
Fine-tune Qwen2-VL 7B with LoRA for conditional classification
Optimized for spatial reasoning and conditional classification
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
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

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

DATASET_DIR = Path(__file__).parent.parent.parent / "quen2-vl.yolov11"
MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
    'rock toe', 'rock toe damaged',
    'slope drain', 'slope drain blocked', 'slope drain damaged',
    'vegetation'
]

def build_training_prompt(class_name: str, all_objects: List[str] = None) -> str:
    """Build training prompt for conditional classification with spatial reasoning."""
    
    objects_context = ""
    if all_objects:
        objects_context = f"\nOther objects in image: {', '.join(all_objects)}"
    
    prompt = f"""Analyze this infrastructure inspection image and classify the condition of the object.

Consider:
1. Visual appearance: Is it damaged, blocked, or in normal condition?
   - Look for cracks, wear, obstructions, or structural issues
2. Spatial relationships with other objects:
   - Is a toe drain at the bottom/end of a slope drain?
   - Is a rock toe positioned above a toe drain?
   - What is the relative positioning?
3. Context: What is the overall state of the infrastructure?{objects_context}

Classify as: {class_name}"""
    
    return prompt

class ConditionalClassificationDataset(Dataset):
    """Dataset for conditional classification fine-tuning."""
    
    def __init__(self, dataset_dir: Path, split: str = 'train', processor=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.processor = processor
        
        self.images_dir = self.dataset_dir / split / 'images'
        self.labels_dir = self.dataset_dir / split / 'labels'
        
        # Load all image-label pairs
        self.samples = []
        for img_file in sorted(self.images_dir.glob('*.jpg')):
            label_file = self.labels_dir / (img_file.stem + '.txt')
            if label_file.exists():
                # Read all labels for context
                all_classes = []
                with open(label_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_idx = int(parts[0])
                                if 0 <= class_idx < len(CLASS_NAMES):
                                    all_classes.append(CLASS_NAMES[class_idx])
                            except ValueError:
                                continue
                
                # Create sample for each object
                for class_name in all_classes:
                    self.samples.append({
                        'image_path': img_file,
                        'class_name': class_name,
                        'all_objects': all_classes
                    })
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Build prompt
        prompt = build_training_prompt(
            sample['class_name'],
            [obj for obj in sample['all_objects'] if obj != sample['class_name']]
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
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Add target label
        target_text = sample['class_name']
        target_inputs = self.processor(
            text=[target_text],
            padding=True,
            return_tensors="pt"
        )
        
        # Combine inputs
        inputs['labels'] = target_inputs['input_ids']
        
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

def main():
    print("="*80)
    print("QWEN2-VL 7B LoRA FINE-TUNING")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("\n1. Loading Qwen2-VL 7B model...")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Setup LoRA
    print("\n2. Setting up LoRA...")
    model = setup_lora(model)
    print("✅ LoRA configured")
    
    # Create datasets
    print("\n3. Creating datasets...")
    train_dataset = ConditionalClassificationDataset(
        DATASET_DIR, split='train', processor=processor
    )
    val_dataset = ConditionalClassificationDataset(
        DATASET_DIR, split='valid', processor=processor
    )
    print(f"✅ Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Training arguments
    print("\n4. Configuring training...")
    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "qwen2vl_lora_checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=device == "cuda",
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        warmup_steps=50,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False
    )
    
    # Trainer
    print("\n5. Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\n6. Starting training...")
    print("="*80)
    trainer.train()
    
    # Save final model
    print("\n7. Saving final model...")
    final_model_dir = MODELS_DIR / "qwen2vl_lora_final"
    model.save_pretrained(str(final_model_dir))
    processor.save_pretrained(str(final_model_dir))
    print(f"✅ Model saved to {final_model_dir}")
    
    # Save training info
    training_info = {
        'model': 'Qwen2-VL-7B-Instruct',
        'fine_tuning_method': 'LoRA',
        'lora_r': 16,
        'lora_alpha': 32,
        'learning_rate': 2e-4,
        'epochs': 3,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'final_model_path': str(final_model_dir)
    }
    
    info_file = EXPERIMENTS_DIR / "training_info.json"
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✅ Training info saved to {info_file}")
    print("\n" + "="*80)
    print("✅ Fine-tuning complete!")

if __name__ == '__main__':
    main()

