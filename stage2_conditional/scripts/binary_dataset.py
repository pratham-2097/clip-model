"""
Binary Dataset Loader with YOLO Integration

Loads detections from YOLO Stage 1 model and prepares data for hierarchical binary classification:
- Runs YOLO on images to get detections
- Extracts image crops for each detection
- Computes spatial features
- Maps 9 classes → binary (NORMAL vs CONDITIONAL)
- Provides data for CLIP ViT-B/32 training
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import pickle

from spatial_features import extract_spatial_features


class BinaryConditionalDataset(Dataset):
    """
    Hierarchical binary dataset for NORMAL vs CONDITIONAL classification.
    
    Features:
    - Loads YOLO detections
    - Extracts spatial features
    - Provides CLIP-ready image crops
    - Binary labels: 0=NORMAL, 1=CONDITIONAL
    """
    
    # Label mapping: 9 classes → binary
    LABEL_MAPPING = {
        'Toe drain': 0,              # NORMAL
        'Toe drain- Blocked': 1,     # CONDITIONAL
        'Toe drain- Damaged': 1,     # CONDITIONAL
        'rock toe': 0,               # NORMAL
        'rock toe damaged': 1,       # CONDITIONAL
        'slope drain': 0,            # NORMAL
        'slope drain blocked': 1,    # CONDITIONAL
        'slope drain damaged': 1,    # CONDITIONAL
        'vegetation': 0,             # NORMAL
    }
    
    # Object type mapping for embedding
    OBJECT_TYPE_MAPPING = {
        'rock_toe': 0,
        'slope_drain': 1,
        'toe_drain': 2,
        'vegetation': 3,
    }
    
    # YOLO class names (Stage 1)
    YOLO_CLASS_NAMES = ['rock_toe', 'slope_drain', 'toe_drain', 'vegetation']
    
    # Ground truth class names (Stage 2 - from data.yaml)
    GT_CLASS_NAMES = [
        'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
        'rock toe', 'rock toe damaged',
        'slope drain', 'slope drain blocked', 'slope drain damaged',
        'vegetation'
    ]
    
    def __init__(
        self,
        dataset_dir: str,
        split: str = 'train',
        yolo_model_path: str = None,
        processor = None,
        cache_dir: str = None,
        use_cache: bool = True,
        conf_threshold: float = 0.25,
    ):
        """
        Args:
            dataset_dir: Path to dataset (e.g., 'quen2-vl.yolov11/')
            split: 'train', 'valid', or 'test'
            yolo_model_path: Path to Stage 1 YOLO model
            processor: CLIP processor for image preprocessing
            cache_dir: Directory to cache detections
            use_cache: Whether to use cached detections
            conf_threshold: YOLO confidence threshold
        """
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.processor = processor
        self.conf_threshold = conf_threshold
        
        # Paths
        self.images_dir = self.dataset_dir / split / 'images'
        self.labels_dir = self.dataset_dir / split / 'labels'
        
        # Cache
        if cache_dir is None:
            cache_dir = self.dataset_dir.parent / 'cache' / 'binary_detections'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f'{split}_detections.pkl'
        
        # Load or generate samples
        if use_cache and self.cache_file.exists():
            print(f"Loading cached detections from {self.cache_file}...")
            with open(self.cache_file, 'rb') as f:
                self.samples = pickle.load(f)
            print(f"✅ Loaded {len(self.samples)} cached samples")
        else:
            print(f"Generating detections for {split} split...")
            if yolo_model_path is None:
                # Default YOLO model path
                yolo_model_path = self.dataset_dir.parent / 'yolov8_project' / 'runs' / 'detect' / 'yolov11_expanded_finetune_aug_reduced' / 'weights' / 'best.pt'
            self.samples = self._generate_samples_from_yolo(yolo_model_path)
            
            # Cache the samples
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"✅ Cached detections to {self.cache_file}")
        
        self._print_statistics()
    
    def _generate_samples_from_yolo(self, yolo_model_path: str) -> List[Dict]:
        """
        Run YOLO on all images and generate samples with spatial features.
        
        This uses ground truth labels, not YOLO predictions, for class labels.
        YOLO is only used to simulate the Stage 1 detection pipeline.
        """
        from ultralytics import YOLO
        
        # Load YOLO model
        print(f"Loading YOLO model from {yolo_model_path}...")
        yolo_model = YOLO(str(yolo_model_path))
        
        samples = []
        image_files = list(self.images_dir.glob('*.jpg'))
        
        print(f"Processing {len(image_files)} images...")
        for img_file in tqdm(image_files, desc=f"Generating {self.split} samples"):
            # Load ground truth annotations
            label_file = self.labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                continue
            
            # Parse ground truth labels
            gt_objects = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        class_name = self.GT_CLASS_NAMES[class_id]
                        
                        # Convert polygon to bbox
                        coords = [float(x) for x in parts[1:]]
                        x_coords = coords[0::2]
                        y_coords = coords[1::2]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        gt_objects.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'bbox_norm': [x_min, y_min, x_max, y_max],  # Normalized
                        })
            
            if not gt_objects:
                continue
            
            # Run YOLO to get detections (simulating Stage 1)
            img = Image.open(img_file)
            img_width, img_height = img.size
            
            results = yolo_model(str(img_file), conf=self.conf_threshold, verbose=False)
            
            # Extract YOLO detections
            yolo_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    object_type = self.YOLO_CLASS_NAMES[cls_id]
                    
                    yolo_detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'object_type': object_type,
                        'confidence': conf,
                    })
            
            # Match ground truth objects with YOLO detections
            # Use IoU matching to pair them
            for gt_obj in gt_objects:
                # Convert normalized gt bbox to pixel coordinates
                gt_x1 = gt_obj['bbox_norm'][0] * img_width
                gt_y1 = gt_obj['bbox_norm'][1] * img_height
                gt_x2 = gt_obj['bbox_norm'][2] * img_width
                gt_y2 = gt_obj['bbox_norm'][3] * img_height
                gt_bbox_pix = [gt_x1, gt_y1, gt_x2, gt_y2]
                
                # Find best matching YOLO detection (highest IoU)
                best_match = None
                best_iou = 0.3  # Minimum IoU threshold
                
                for yolo_det in yolo_detections:
                    # Compute IoU
                    from spatial_features import compute_iou
                    # Normalize YOLO bbox for IoU computation
                    y_bbox_norm = [
                        yolo_det['bbox'][0] / img_width,
                        yolo_det['bbox'][1] / img_height,
                        yolo_det['bbox'][2] / img_width,
                        yolo_det['bbox'][3] / img_height,
                    ]
                    iou = compute_iou(gt_obj['bbox_norm'], y_bbox_norm)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_match = yolo_det
                
                # If no match, use ground truth bbox with default confidence
                if best_match is None:
                    best_match = {
                        'bbox': gt_bbox_pix,
                        'object_type': self._gt_to_yolo_type(gt_obj['class_name']),
                        'confidence': 0.5,  # Default confidence
                    }
                
                # Extract spatial features
                spatial_features = extract_spatial_features(
                    target_detection=best_match,
                    all_detections=yolo_detections if yolo_detections else [best_match],
                    image_shape=(img_height, img_width)
                )
                
                # Map to binary label
                binary_label = self.LABEL_MAPPING[gt_obj['class_name']]
                
                # Get object type ID
                object_type_id = self.OBJECT_TYPE_MAPPING[best_match['object_type']]
                
                # Create sample
                sample = {
                    'image_path': str(img_file),
                    'bbox': best_match['bbox'],
                    'object_type': best_match['object_type'],
                    'object_type_id': object_type_id,
                    'spatial_features': spatial_features,
                    'binary_label': binary_label,
                    'original_class': gt_obj['class_name'],
                    'yolo_confidence': best_match['confidence'],
                }
                
                samples.append(sample)
        
        print(f"Generated {len(samples)} samples from {len(image_files)} images")
        return samples
    
    def _gt_to_yolo_type(self, gt_class_name: str) -> str:
        """Map ground truth class name to YOLO object type."""
        if 'toe drain' in gt_class_name.lower():
            return 'toe_drain'
        elif 'slope drain' in gt_class_name.lower():
            return 'slope_drain'
        elif 'rock' in gt_class_name.lower():
            return 'rock_toe'
        else:
            return 'vegetation'
    
    def _print_statistics(self):
        """Print dataset statistics."""
        print(f"\n{'='*60}")
        print(f"Binary Dataset Statistics - {self.split.upper()} Split")
        print(f"{'='*60}")
        print(f"Total samples: {len(self.samples)}")
        
        # Count binary labels
        normal_count = sum(1 for s in self.samples if s['binary_label'] == 0)
        conditional_count = sum(1 for s in self.samples if s['binary_label'] == 1)
        print(f"\nBinary label distribution:")
        print(f"  NORMAL:      {normal_count:4d} ({100*normal_count/len(self.samples):.1f}%)")
        print(f"  CONDITIONAL: {conditional_count:4d} ({100*conditional_count/len(self.samples):.1f}%)")
        print(f"  Balance ratio: {max(normal_count, conditional_count)/min(normal_count, conditional_count):.2f}x")
        
        # Count object types
        print(f"\nObject type distribution:")
        for obj_type in ['rock_toe', 'slope_drain', 'toe_drain', 'vegetation']:
            count = sum(1 for s in self.samples if s['object_type'] == obj_type)
            print(f"  {obj_type:15s}: {count:4d}")
        
        print(f"{'='*60}\n")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Crop image using bbox
        bbox = sample['bbox']
        crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        
        # Process with CLIP processor
        if self.processor is not None:
            inputs = self.processor(images=crop, return_tensors="pt")
            pixel_values = inputs['pixel_values'].squeeze(0)  # Remove batch dim
        else:
            # Return PIL image if no processor
            pixel_values = crop
        
        return {
            'pixel_values': pixel_values,
            'object_type_id': torch.tensor(sample['object_type_id'], dtype=torch.long),
            'spatial_features': torch.tensor(sample['spatial_features'], dtype=torch.float32),
            'labels': torch.tensor(sample['binary_label'], dtype=torch.long),
            'object_type': sample['object_type'],
            'original_class': sample['original_class'],
        }


def create_binary_dataloaders(
    dataset_dir: str,
    yolo_model_path: str,
    processor,
    batch_size: int = 32,
    num_workers: int = 0,
    use_cache: bool = True,
):
    """Create train, validation, and test dataloaders."""
    
    # Create datasets
    train_dataset = BinaryConditionalDataset(
        dataset_dir,
        split='train',
        yolo_model_path=yolo_model_path,
        processor=processor,
        use_cache=use_cache,
    )
    
    valid_dataset = BinaryConditionalDataset(
        dataset_dir,
        split='valid',
        yolo_model_path=yolo_model_path,
        processor=processor,
        use_cache=use_cache,
    )
    
    test_dataset = BinaryConditionalDataset(
        dataset_dir,
        split='test',
        yolo_model_path=yolo_model_path,
        processor=processor,
        use_cache=use_cache,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # MPS doesn't support pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    
    test_loader = DataLoader(
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
    yolo_model_path = '../../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt'
    
    print("Testing binary dataset loader...")
    print("Loading CLIP processor...")
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    print("\nCreating dataloaders...")
    train_loader, valid_loader, test_loader = create_binary_dataloaders(
        dataset_dir,
        yolo_model_path,
        processor,
        batch_size=8,
        use_cache=True,
    )
    
    print("\nTesting train loader...")
    for batch in train_loader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Pixel values shape: {batch['pixel_values'].shape}")
        print(f"Object type IDs shape: {batch['object_type_id'].shape}")
        print(f"Spatial features shape: {batch['spatial_features'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        print(f"Sample object types: {batch['object_type'][:3]}")
        print(f"Sample original classes: {batch['original_class'][:3]}")
        print(f"Sample labels: {batch['labels'][:3]}")
        break
    
    print("\n✅ Binary dataset loader working!")

