"""
CLIP Dataset Loader for Conditional Classification

Features:
- Load YOLO format annotations
- Handle class imbalance with oversampling
- Generate text prompts with spatial context
- Support train/valid/test splits
- Multi-object context support
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from collections import defaultdict

class CLIPConditionalDataset(Dataset):
    """Dataset for CLIP fine-tuning on conditional classification."""
    
    # Class names from data.yaml
    CLASS_NAMES = [
        'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
        'rock toe', 'rock toe damaged',
        'slope drain', 'slope drain blocked', 'slope drain damaged',
        'vegetation'
    ]
    
    # Class weights (inversely proportional to frequency)
    CLASS_WEIGHTS = {
        'Toe drain': 7.04,              # 52 instances
        'Toe drain- Blocked': 4.69,     # 78 instances  
        'Toe drain- Damaged': 4.82,     # 76 instances
        'rock toe': 2.39,                # 153 instances
        'rock toe damaged': 1.0,         # 366 instances (base)
        'slope drain': 1.42,             # 257 instances
        'slope drain blocked': 3.77,    # 97 instances
        'slope drain damaged': 2.47,     # 148 instances
        'vegetation': 1.54               # 238 instances
    }
    
    # Spatial patterns from dataset analysis
    SPATIAL_PATTERNS = {
        'Toe drain': 'at bottom of image',
        'Toe drain- Blocked': 'at bottom of image',
        'Toe drain- Damaged': 'at bottom of image',
        'slope drain': 'in middle of image',
        'slope drain blocked': 'in middle of image',
        'slope drain damaged': 'in middle of image',
        'rock toe': 'above toe drain',
        'rock toe damaged': 'above toe drain',
        'vegetation': 'on slope',
    }
    
    def __init__(
        self,
        dataset_dir: str,
        split: str = 'train',
        processor=None,
        oversample: bool = True,
        spatial_context: bool = True,
        oversample_factor: float = 2.5,
    ):
        """
        Args:
            dataset_dir: Path to dataset (e.g., '../quen2-vl.yolov11/')
            split: 'train', 'valid', or 'test'
            processor: CLIP processor for image and text
            oversample: Whether to oversample rare classes
            spatial_context: Whether to include spatial context in prompts
            oversample_factor: How much to oversample rare classes (2.5 = 250% of base)
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.processor = processor
        self.spatial_context = spatial_context
        self.oversample = oversample and split == 'train'  # Only oversample training
        
        # Set up paths
        self.images_dir = self.dataset_dir / split / 'images'
        self.labels_dir = self.dataset_dir / split / 'labels'
        
        # Load all samples
        self.samples = self._load_samples()
        
        # Apply oversampling if enabled
        if self.oversample:
            self.samples = self._apply_oversampling(self.samples, oversample_factor)
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        self._print_class_distribution()
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the dataset."""
        samples = []
        
        for label_file in self.labels_dir.glob('*.txt'):
            image_file = self.images_dir / f"{label_file.stem}.jpg"
            
            if not image_file.exists():
                continue
            
            # Read all annotations in this image
            image_annotations = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # Convert polygon to bbox (x_center, y_center, width, height)
                        coords = [float(x) for x in parts[1:]]
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        x_center = (x_min + x_max) / 2
                        y_center = (y_min + y_max) / 2
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        image_annotations.append({
                            'class_id': class_id,
                            'bbox': [x_center, y_center, width, height],
                            'y_position': y_center,
                        })
            
            # Create one sample per annotation
            for i, ann in enumerate(image_annotations):
                samples.append({
                    'image_path': str(image_file),
                    'class_id': ann['class_id'],
                    'class_name': self.CLASS_NAMES[ann['class_id']],
                    'bbox': ann['bbox'],
                    'y_position': ann['y_position'],
                    'all_objects': [self.CLASS_NAMES[a['class_id']] for a in image_annotations],
                    'object_index': i,
                    'total_objects': len(image_annotations),
                })
        
        return samples
    
    def _apply_oversampling(self, samples: List[Dict], factor: float) -> List[Dict]:
        """Oversample rare classes to balance the dataset."""
        if not samples:
            return samples
        
        # Group samples by class
        class_samples = defaultdict(list)
        for sample in samples:
            class_samples[sample['class_name']].append(sample)
        
        if not class_samples:
            return samples
        
        # Calculate target count (based on most common class * factor)
        max_count = max(len(s) for s in class_samples.values())
        target_count = int(max_count * factor / len(class_samples))
        
        # Oversample each class
        balanced_samples = []
        for class_name, class_samples_list in class_samples.items():
            weight = self.CLASS_WEIGHTS.get(class_name, 1.0)
            # Oversample proportional to weight
            oversample_count = int(target_count * min(weight, 3.0))  # Cap at 3x
            
            if len(class_samples_list) < oversample_count:
                # Oversample by repeating
                repeats = oversample_count // len(class_samples_list)
                remainder = oversample_count % len(class_samples_list)
                balanced_samples.extend(class_samples_list * repeats)
                balanced_samples.extend(np.random.choice(class_samples_list, remainder, replace=False))
            else:
                # Use all samples
                balanced_samples.extend(class_samples_list)
        
        # Shuffle
        np.random.shuffle(balanced_samples)
        return balanced_samples
    
    def _print_class_distribution(self):
        """Print class distribution for debugging."""
        class_counts = defaultdict(int)
        for sample in self.samples:
            class_counts[sample['class_name']] += 1
        
        print(f"\nClass distribution ({self.split}):")
        for class_name in self.CLASS_NAMES:
            count = class_counts.get(class_name, 0)
            print(f"  {class_name:30s}: {count:4d}")
    
    def _generate_text_prompt(self, sample: Dict) -> List[str]:
        """Generate text prompts for all classes (for contrastive learning)."""
        prompts = []
        
        for class_name in self.CLASS_NAMES:
            # Basic prompt
            prompt = f"a photo of {class_name}"
            
            # Add spatial context if enabled
            if self.spatial_context and class_name in self.SPATIAL_PATTERNS:
                spatial_info = self.SPATIAL_PATTERNS[class_name]
                prompt = f"a photo of {class_name} {spatial_info}"
            
            # Add multi-object context if multiple objects in image
            if sample['total_objects'] > 1 and len(sample['all_objects']) > 1:
                other_objects = [obj for obj in sample['all_objects'] if obj != sample['class_name']]
                if other_objects and len(other_objects) <= 3:
                    other_str = ', '.join(other_objects[:3])
                    prompt = f"{prompt}, with {other_str} nearby"
            
            prompts.append(prompt)
        
        return prompts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Generate text prompts for all classes
        text_prompts = self._generate_text_prompt(sample)
        
        # Process with CLIP processor
        if self.processor is not None:
            inputs = self.processor(
                text=text_prompts,
                images=image,
                return_tensors="pt",
                padding="max_length",  # Use max_length padding for consistent sizes
                max_length=77,  # CLIP's max sequence length
                truncation=True
            )
            
            # Remove extra batch dimension from image processing
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(0)
            # Keep text inputs as is (they already have the right shape)
        else:
            inputs = {
                'image': image,
                'text': text_prompts,
            }
        
        # Add label
        inputs['labels'] = torch.tensor(sample['class_id'], dtype=torch.long)
        inputs['class_name'] = sample['class_name']
        
        return inputs


def create_dataloaders(
    dataset_dir: str,
    processor,
    batch_size: int = 16,
    oversample: bool = True,
    spatial_context: bool = True,
    num_workers: int = 0,  # Set to 0 for MPS compatibility
):
    """Create train, validation, and test dataloaders."""
    
    # Create datasets
    train_dataset = CLIPConditionalDataset(
        dataset_dir,
        split='train',
        processor=processor,
        oversample=oversample,
        spatial_context=spatial_context,
    )
    
    valid_dataset = CLIPConditionalDataset(
        dataset_dir,
        split='valid',
        processor=processor,
        oversample=False,  # No oversampling for validation
        spatial_context=spatial_context,
    )
    
    test_dataset = CLIPConditionalDataset(
        dataset_dir,
        split='test',
        processor=processor,
        oversample=False,  # No oversampling for test
        spatial_context=spatial_context,
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't support pin_memory
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # Test the dataset loader
    from transformers import CLIPProcessor
    
    dataset_dir = '../../quen2-vl.yolov11'
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    
    print("Creating datasets...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        dataset_dir,
        processor,
        batch_size=4,
        oversample=True,
        spatial_context=True,
    )
    
    print("\nTesting train loader...")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Sample class: {batch['class_name'][0]}")
        break
    
    print("\n✅ Dataset loader working!")

