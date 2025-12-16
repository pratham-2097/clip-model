"""
Stage 1 + Stage 2 Integration Script

Pipeline:
1. Stage 1 (YOLOv11) detects objects → bounding boxes
2. Stage 2 (CLIP) classifies conditions → normal/damaged/blocked
3. Combine results → {object_type} {condition}

Handles spatial relationships and multi-object context.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import CLIPProcessor
from ultralytics import YOLO
from tqdm import tqdm
import json

from finetune_clip_conditional import CLIPClassifier
from clip_dataset import CLIPConditionalDataset


class Stage1Stage2Pipeline:
    """Combined pipeline for object detection + conditional classification."""
    
    # Stage 1 class mapping (YOLOv11)
    STAGE1_CLASSES = {
        0: 'rock_toe',
        1: 'slope_drain',
        2: 'toe_drain',
        3: 'vegetation',
    }
    
    # Stage 2 class mapping (CLIP)
    STAGE2_CLASSES = CLIPConditionalDataset.CLASS_NAMES
    
    def __init__(
        self,
        yolo_model_path: str,
        clip_model_path: str,
        clip_base_model: str = 'openai/clip-vit-large-patch14',
        device: str = 'mps',
    ):
        """
        Args:
            yolo_model_path: Path to Stage 1 YOLOv11 model
            clip_model_path: Path to Stage 2 fine-tuned CLIP model
            clip_base_model: Base CLIP model name
            device: Device to run on ('mps', 'cuda', or 'cpu')
        """
        self.device = device
        
        # Load Stage 1 (YOLO)
        print(f"Loading Stage 1 model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load Stage 2 (CLIP)
        print(f"Loading Stage 2 model: {clip_model_path}")
        self.clip_processor = CLIPProcessor.from_pretrained(
            Path(clip_model_path) / 'clip_model'
        )
        self.clip_model = CLIPClassifier(clip_base_model, num_classes=9)
        checkpoint = torch.load(
            Path(clip_model_path) / 'best_model.pt',
            map_location=device
        )
        self.clip_model.load_state_dict(checkpoint['model_state_dict'])
        self.clip_model = self.clip_model.to(device)
        self.clip_model.eval()
        
        print(f"✅ Pipeline loaded successfully")
    
    def detect_objects(self, image_path: str, conf_threshold: float = 0.25) -> List[Dict]:
        """
        Stage 1: Detect objects using YOLOv11.
        
        Returns:
            List of detections with bbox, class, confidence
        """
        # Run detection
        results = self.yolo_model(image_path, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract bbox info
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Get class name
                object_type = self.STAGE1_CLASSES.get(cls_id, 'unknown')
                
                # Calculate center and normalized position
                img_height, img_width = result.orig_shape
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'bbox_norm': [x_center, y_center],
                    'object_type': object_type,
                    'confidence': conf,
                    'class_id': cls_id,
                })
        
        return detections
    
    def classify_condition(
        self,
        image_path: str,
        detections: List[Dict],
    ) -> List[Dict]:
        """
        Stage 2: Classify condition for each detected object using CLIP.
        
        Args:
            image_path: Path to image
            detections: List of detections from Stage 1
        
        Returns:
            Detections with added 'condition' field
        """
        if not detections:
            return []
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Process each detection
        for detection in detections:
            object_type = detection['object_type']
            
            # Generate text prompts for all possible conditions
            # Match the object type with CLIP classes
            candidate_classes = []
            for clip_class in self.STAGE2_CLASSES:
                # Check if this CLIP class matches the detected object type
                clip_lower = clip_class.lower()
                obj_lower = object_type.lower().replace('_', ' ')
                
                if obj_lower in clip_lower or clip_lower.startswith(obj_lower):
                    candidate_classes.append(clip_class)
            
            if not candidate_classes:
                # Fallback: assume normal condition
                detection['condition'] = 'normal'
                detection['clip_class'] = object_type
                detection['clip_confidence'] = 0.0
                continue
            
            # Generate prompts for candidate classes
            prompts = [f"a photo of {cls}" for cls in candidate_classes]
            
            # Add spatial context
            y_pos = detection['bbox_norm'][1]
            if y_pos < 0.33:
                spatial = "at top of image"
            elif y_pos > 0.66:
                spatial = "at bottom of image"
            else:
                spatial = "in middle of image"
            
            prompts = [f"{p} {spatial}" for p in prompts]
            
            # Process with CLIP
            inputs = self.clip_processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.clip_model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=inputs['pixel_values'],
                )
                
                # Get probabilities
                probs = torch.softmax(outputs['logits'], dim=1)
                
                # Find best matching candidate
                candidate_indices = [self.STAGE2_CLASSES.index(c) for c in candidate_classes]
                candidate_probs = probs[0, candidate_indices]
                best_idx = candidate_probs.argmax().item()
                best_class = candidate_classes[best_idx]
                best_conf = candidate_probs[best_idx].item()
            
            # Extract condition from class name
            if 'blocked' in best_class.lower():
                condition = 'blocked'
            elif 'damaged' in best_class.lower():
                condition = 'damaged'
            else:
                condition = 'normal'
            
            detection['condition'] = condition
            detection['clip_class'] = best_class
            detection['clip_confidence'] = best_conf
        
        return detections
    
    def process_image(self, image_path: str) -> Dict:
        """
        Process single image through full pipeline.
        
        Returns:
            Dict with detections and results
        """
        # Stage 1: Detect objects
        detections = self.detect_objects(image_path)
        
        # Stage 2: Classify conditions
        detections = self.classify_condition(image_path, detections)
        
        # Build results
        results = []
        for detection in detections:
            results.append({
                'object_type': detection['object_type'],
                'condition': detection['condition'],
                'combined_class': f"{detection['object_type']} {detection['condition']}",
                'bbox': detection['bbox'],
                'yolo_confidence': detection['confidence'],
                'clip_confidence': detection.get('clip_confidence', 0.0),
                'clip_class': detection.get('clip_class', ''),
            })
        
        return {
            'image_path': image_path,
            'num_detections': len(results),
            'detections': results,
        }
    
    def visualize_results(self, image_path: str, results: Dict, output_path: str):
        """Visualize results with bounding boxes and labels."""
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw detections
        for detection in results['detections']:
            x1, y1, x2, y2 = [int(x) for x in detection['bbox']]
            label = detection['combined_class']
            conf = detection['yolo_confidence']
            
            # Color based on condition
            if detection['condition'] == 'normal':
                color = (0, 255, 0)  # Green
            elif detection['condition'] == 'damaged':
                color = (255, 165, 0)  # Orange
            else:  # blocked
                color = (255, 0, 0)  # Red
            
            # Draw bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{label} ({conf:.2f})"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Save
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img)


def main():
    parser = argparse.ArgumentParser(description='Run Stage 1 + Stage 2 pipeline')
    parser.add_argument('--yolo_model', type=str,
                       default='../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt',
                       help='Path to Stage 1 YOLO model')
    parser.add_argument('--clip_model', type=str,
                       default='../models/clip_conditional_final',
                       help='Path to Stage 2 CLIP model')
    parser.add_argument('--dataset_dir', type=str, default='../quen2-vl.yolov11',
                       help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'],
                       help='Which split to process')
    parser.add_argument('--num_images', type=int, default=10,
                       help='Number of images to process (-1 for all)')
    parser.add_argument('--output_dir', type=str, default='../experiments/integration_results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true', default=True,
                       help='Save visualizations')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Create pipeline
    print(f"\n{'='*80}")
    print("STAGE 1 + STAGE 2 INTEGRATION PIPELINE")
    print(f"{'='*80}\n")
    
    pipeline = Stage1Stage2Pipeline(
        args.yolo_model,
        args.clip_model,
        device=device,
    )
    
    # Get images
    images_dir = Path(args.dataset_dir) / args.split / 'images'
    image_files = list(images_dir.glob('*.jpg'))
    
    if args.num_images > 0:
        image_files = image_files[:args.num_images]
    
    print(f"\nProcessing {len(image_files)} images from {args.split} split...")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
    
    # Process images
    all_results = []
    for image_path in tqdm(image_files, desc="Processing"):
        # Process
        results = pipeline.process_image(str(image_path))
        all_results.append(results)
        
        # Visualize
        if args.visualize and results['num_detections'] > 0:
            vis_path = vis_dir / f"{image_path.stem}_result.jpg"
            pipeline.visualize_results(str(image_path), results, str(vis_path))
    
    # Save results
    results_file = output_dir / f'integration_results_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    
    total_detections = sum(r['num_detections'] for r in all_results)
    print(f"\nTotal images processed: {len(all_results)}")
    print(f"Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / len(all_results):.2f}")
    
    # Count by condition
    condition_counts = {'normal': 0, 'damaged': 0, 'blocked': 0}
    for result in all_results:
        for detection in result['detections']:
            condition_counts[detection['condition']] += 1
    
    print(f"\nDetections by condition:")
    for condition, count in condition_counts.items():
        pct = 100. * count / total_detections if total_detections > 0 else 0
        print(f"  {condition.capitalize():10s}: {count:4d} ({pct:.1f}%)")
    
    print(f"\n✅ Results saved to {results_file}")
    if args.visualize:
        print(f"✅ Visualizations saved to {vis_dir}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()

