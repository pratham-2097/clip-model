#!/usr/bin/env python3
"""
Integrate Stage 1 object detection with Stage 2 conditional classification
End-to-end pipeline: Detection → Classification
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Paths
STAGE1_MODEL_PATH = Path(__file__).parent.parent.parent / "yolov8_project" / "runs" / "detect" / "yolov11_expanded_finetune_aug_reduced" / "weights" / "best.pt"
STAGE2_MODEL_PATH = Path(__file__).parent.parent / "models" / "qwen2vl_lora_final"
DATASET_DIR = Path(__file__).parent.parent.parent / "quen2-vl.yolov11"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Stage 1 class mapping (4 classes)
STAGE1_CLASSES = ['rock_toe', 'slope_drain', 'toe_drain', 'vegetation']

# Stage 2 class mapping (9 classes)
STAGE2_CLASSES = [
    'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
    'rock toe', 'rock toe damaged',
    'slope drain', 'slope drain blocked', 'slope drain damaged',
    'vegetation'
]

def load_stage1_model(model_path: Path):
    """Load Stage 1 object detection model."""
    if not model_path.exists():
        raise FileNotFoundError(f"Stage 1 model not found at {model_path}")
    
    model = YOLO(str(model_path))
    print(f"✅ Stage 1 model loaded: {model_path.name}")
    return model

def load_stage2_model(model_path: Path):
    """Load Stage 2 conditional classification model."""
    if not model_path.exists():
        print(f"⚠️  Stage 2 model not found at {model_path}")
        print("   Using base Qwen2-VL 7B model instead")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    else:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(str(model_path))
        print(f"✅ Stage 2 model loaded: {model_path.name}")
    
    return model, processor

def extract_spatial_context(detections, image_size: Tuple[int, int]) -> str:
    """Extract spatial context from Stage 1 detections."""
    if not detections or len(detections) == 0:
        return "No objects detected"
    
    context_parts = []
    for det in detections:
        class_name = STAGE1_CLASSES[det['class']] if det['class'] < len(STAGE1_CLASSES) else f"Class_{det['class']}"
        bbox = det['bbox']
        x_center = bbox[0] / image_size[0]  # Normalized
        y_center = bbox[1] / image_size[1]  # Normalized
        
        position = "bottom" if y_center > 0.6 else "top" if y_center < 0.4 else "middle"
        context_parts.append(f"{class_name} at {position} (y={y_center:.2f})")
    
    return "; ".join(context_parts)

def build_classification_prompt(target_object: str, stage1_class: str, spatial_context: str, all_detections: List[Dict]) -> str:
    """Build prompt for conditional classification."""
    
    detected_objects = [STAGE1_CLASSES[d['class']] for d in all_detections if d['class'] < len(STAGE1_CLASSES)]
    objects_list = ", ".join(detected_objects) if detected_objects else "No other objects"
    
    prompt = f"""Analyze this infrastructure inspection image.

Stage 1 detected objects: {objects_list}
Spatial context: {spatial_context}

Focus on: {target_object} (detected as {stage1_class})

Classify its condition considering:
1. Visual appearance: damaged, blocked, or normal?
2. Spatial relationships:
   - Is toe drain at bottom/end of slope drain?
   - Is rock toe above toe drain?
   - Relative positioning?
3. Context: Overall infrastructure state

Classify as one of:
- Toe drain
- Toe drain- Blocked
- Toe drain- Damaged
- rock toe
- rock toe damaged
- slope drain
- slope drain blocked
- slope drain damaged
- vegetation

Class name:"""
    
    return prompt

def run_end_to_end(image_path: Path, stage1_model, stage2_model, processor):
    """Run end-to-end pipeline on a single image."""
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_size = image.size
    
    # Stage 1: Object Detection
    results = stage1_model(str(image_path), conf=0.25)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            
            # Convert to center format
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]
            
            detections.append({
                'class': cls,
                'confidence': conf,
                'bbox': (x_center, y_center, width, height)
            })
    
    # Stage 2: Conditional Classification
    spatial_context = extract_spatial_context(detections, image_size)
    
    classifications = []
    for det in detections:
        stage1_class = STAGE1_CLASSES[det['class']] if det['class'] < len(STAGE1_CLASSES) else f"Class_{det['class']}"
        
        # Map Stage 1 class to possible Stage 2 classes
        if 'toe_drain' in stage1_class:
            target_object = "toe drain"
        elif 'slope_drain' in stage1_class:
            target_object = "slope drain"
        elif 'rock_toe' in stage1_class:
            target_object = "rock toe"
        elif 'vegetation' in stage1_class:
            target_object = "vegetation"
        else:
            target_object = stage1_class
        
        # Build prompt
        prompt = build_classification_prompt(
            target_object,
            stage1_class,
            spatial_context,
            detections
        )
        
        # Run Qwen2-VL
        try:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            ).to(stage2_model.device)
            
            generated_ids = stage2_model.generate(**inputs, max_new_tokens=50)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            predicted_class = output_text.strip()
            
            classifications.append({
                'stage1_class': stage1_class,
                'stage1_confidence': det['confidence'],
                'stage2_class': predicted_class,
                'bbox': det['bbox']
            })
        except Exception as e:
            print(f"⚠️  Error classifying {stage1_class}: {e}")
            classifications.append({
                'stage1_class': stage1_class,
                'stage1_confidence': det['confidence'],
                'stage2_class': 'ERROR',
                'bbox': det['bbox'],
                'error': str(e)
            })
    
    return {
        'image': str(image_path),
        'detections': detections,
        'classifications': classifications
    }

def test_end_to_end(split='valid', num_images=10):
    """Test end-to-end pipeline."""
    
    print("="*80)
    print("STAGE 1 + STAGE 2 END-TO-END TESTING")
    print("="*80)
    
    # Load models
    print("\n1. Loading models...")
    stage1_model = load_stage1_model(STAGE1_MODEL_PATH)
    stage2_model, processor = load_stage2_model(STAGE2_MODEL_PATH)
    print("✅ Models loaded")
    
    # Get test images
    images_dir = DATASET_DIR / split / 'images'
    labels_dir = DATASET_DIR / split / 'labels'
    
    image_files = list(images_dir.glob('*.jpg'))[:num_images]
    print(f"\n2. Testing on {len(image_files)} images from {split} split...")
    
    results = []
    for img_file in tqdm(image_files, desc="Processing"):
        result = run_end_to_end(img_file, stage1_model, stage2_model, processor)
        results.append(result)
    
    # Save results
    output_file = OUTPUT_DIR / "integration_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")
    
    # Print summary
    print("\n3. Summary:")
    total_detections = sum(len(r['detections']) for r in results)
    total_classifications = sum(len(r['classifications']) for r in results)
    print(f"   Total detections: {total_detections}")
    print(f"   Total classifications: {total_classifications}")
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid', 'test'])
    parser.add_argument('--num_images', type=int, default=10)
    args = parser.parse_args()
    
    test_end_to_end(split=args.split, num_images=args.num_images)

