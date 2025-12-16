#!/usr/bin/env python3
"""
Stage 2 inference utilities for CLIP binary classifier.
Handles loading the hierarchical binary classifier and running conditional classification.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor

# Add stage2_conditional scripts to path
stage2_path = Path(__file__).parent.parent.parent / "stage2_conditional" / "scripts"
if str(stage2_path) not in sys.path:
    sys.path.insert(0, str(stage2_path))

from binary_model import HierarchicalBinaryClassifier
from spatial_features import extract_spatial_features


# Stage 2 model paths (relative to yolov8_project directory)
STAGE2_MODEL_PATHS = {
    "CLIP-B32-Binary": "../stage2_conditional/models/clip_binary_fast/best_model.pt",
}

# Object type mapping (matches training)
OBJECT_TYPE_MAPPING = {
    'rock_toe': 0,
    'rock toe': 0,
    'slope_drain': 1,
    'slope drain': 1,
    'toe_drain': 2,
    'toe drain': 2,
    'Toe drain': 2,
    'vegetation': 3,
}

# Binary label names
BINARY_LABELS = {
    0: 'NORMAL',
    1: 'CONDITIONAL'
}


def get_device() -> str:
    """
    Auto-detect the best available device.
    """
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def load_stage2_model(
    model_type: str = "CLIP-B32-Binary",
    base_path: Optional[Path] = None
) -> Tuple[HierarchicalBinaryClassifier, CLIPProcessor, str]:
    """
    Load Stage 2 CLIP binary classifier.
    
    Args:
        model_type: Model type (currently only "CLIP-B32-Binary")
        base_path: Base path to yolov8_project directory (parent of ui/)
    
    Returns:
        Tuple of (model, processor, model_path)
    """
    if model_type not in STAGE2_MODEL_PATHS:
        raise ValueError(f"Invalid model type: {model_type}")
    
    if base_path is None:
        # Default: assume running from ui directory, go up to parent
        current = Path(__file__).parent
        if current.name == "ui":
            base_path = current.parent  # yolov8_project
        else:
            base_path = current
    
    # Get model path
    model_path = base_path / STAGE2_MODEL_PATHS[model_type]
    model_path = model_path.expanduser().resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Stage 2 model weights not found at {model_path}\n"
            f"Please train the model first:\n"
            f"  cd stage2_conditional/scripts\n"
            f"  python3 train_binary_clip.py --epochs 8 --batch_size 32"
        )
    
    # Load CLIP processor
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    
    # Create model
    model = HierarchicalBinaryClassifier(
        clip_model_name='openai/clip-vit-base-patch32',
        num_object_types=4,
        object_type_embed_dim=32,
        spatial_feature_dim=9,
        hidden_dims=[256, 128],
        dropout=0.3,
    )
    
    # Load checkpoint
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, processor, str(model_path)


def normalize_object_type(obj_type: str) -> str:
    """
    Normalize object type name for consistent mapping.
    """
    # Handle variations
    obj_type_lower = obj_type.lower().strip()
    
    # Map variations to standard names
    if 'rock' in obj_type_lower:
        return 'rock_toe'
    elif 'slope' in obj_type_lower:
        return 'slope_drain'
    elif 'toe' in obj_type_lower:
        return 'toe_drain'
    elif 'veg' in obj_type_lower:
        return 'vegetation'
    else:
        return obj_type_lower


def run_stage2_inference(
    model: HierarchicalBinaryClassifier,
    processor: CLIPProcessor,
    image: Image.Image,
    detections: List[Dict],
    device: Optional[str] = None
) -> List[Dict]:
    """
    Run Stage 2 classification on detected objects.
    
    Args:
        model: Loaded Stage 2 model
        processor: CLIP processor
        image: Original PIL image
        detections: List of detections from Stage 1, each with:
            - 'class' or 'class_name': str (object type)
            - 'bbox': dict with {'x1', 'y1', 'x2', 'y2'} or list [x1, y1, x2, y2]
            - 'confidence': float
        device: Device to run on (auto-detected if None)
    
    Returns:
        List of detections with added 'condition' field (NORMAL/CONDITIONAL)
    """
    if device is None:
        device = get_device()
    
    if not detections:
        return detections
    
    # Get image size
    img_width, img_height = image.size
    
    # Convert detections to format expected by spatial feature extractor
    all_yolo_detections = []
    for det in detections:
        # Handle both dict and list bbox formats
        bbox_dict = det.get('bbox', {})
        if isinstance(bbox_dict, dict):
            bbox = [bbox_dict['x1'], bbox_dict['y1'], bbox_dict['x2'], bbox_dict['y2']]
        else:
            bbox = bbox_dict  # Already a list
        
        # Handle both 'class' and 'class_name' keys
        obj_type = normalize_object_type(det.get('class_name', det.get('class', 'unknown')))
        
        all_yolo_detections.append({
            'bbox': bbox,
            'object_type': obj_type,
            'confidence': det['confidence']
        })
    
    # Process each detection
    results = []
    for idx, det in enumerate(detections):
        try:
            # Handle both dict and list bbox formats
            bbox_dict = det.get('bbox', {})
            if isinstance(bbox_dict, dict):
                bbox = [bbox_dict['x1'], bbox_dict['y1'], bbox_dict['x2'], bbox_dict['y2']]
            else:
                bbox = bbox_dict  # Already a list
            
            # Handle both 'class' and 'class_name' keys
            obj_type = normalize_object_type(det.get('class_name', det.get('class', 'unknown')))
            
            # Extract image crop
            x1, y1, x2, y2 = map(int, bbox)
            crop = image.crop((x1, y1, x2, y2))
            
            # Preprocess with CLIP
            inputs = processor(images=crop, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # Get object type ID
            obj_type_id = OBJECT_TYPE_MAPPING.get(obj_type, 0)
            obj_type_tensor = torch.tensor([obj_type_id], dtype=torch.long).to(device)
            
            # Extract spatial features
            spatial_features = extract_spatial_features(
                target_detection=all_yolo_detections[idx],
                all_detections=all_yolo_detections,
                image_shape=(img_height, img_width)
            )
            spatial_tensor = torch.tensor(spatial_features, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Run Stage 2 inference
            with torch.no_grad():
                logits = model(pixel_values, obj_type_tensor, spatial_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_class = logits.argmax(dim=1).item()
                confidence = probs[0, pred_class].item()
            
            # Get condition label
            condition = BINARY_LABELS[pred_class]
            
            # Add to results
            result = det.copy()
            result['condition'] = condition
            result['condition_confidence'] = float(confidence)
            results.append(result)
            
        except Exception as e:
            # If Stage 2 fails for this detection, keep original without condition
            print(f"Warning: Stage 2 failed for detection {idx}: {e}")
            result = det.copy()
            result['condition'] = 'UNKNOWN'
            result['condition_confidence'] = 0.0
            results.append(result)
    
    return results


def format_detection_label(detection: Dict, include_stage2: bool = True) -> str:
    """
    Format detection label for display.
    
    Args:
        detection: Detection dict with 'class' or 'class_name' and optionally 'condition'
        include_stage2: Whether to include Stage 2 condition
    
    Returns:
        Formatted label string
    """
    # Handle both 'class' and 'class_name' keys
    label = detection.get('class_name', detection.get('class', 'unknown'))
    
    if include_stage2 and 'condition' in detection:
        condition = detection['condition']
        label = f"{label} ({condition})"
    
    return label


def get_condition_color(condition: str) -> Tuple[int, int, int]:
    """
    Get color for condition status.
    
    Args:
        condition: 'NORMAL', 'CONDITIONAL', or 'UNKNOWN'
    
    Returns:
        RGB tuple
    """
    colors = {
        'NORMAL': (0, 255, 0),       # Green
        'CONDITIONAL': (255, 165, 0), # Orange
        'UNKNOWN': (128, 128, 128)    # Gray
    }
    return colors.get(condition, (255, 0, 0))  # Red for error


if __name__ == '__main__':
    # Test loading
    print("Testing Stage 2 model loading...")
    try:
        model, processor, path = load_stage2_model()
        print(f"✅ Model loaded successfully from {path}")
        print(f"✅ Device: {get_device()}")
        print(f"✅ Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    except Exception as e:
        print(f"❌ Error: {e}")

