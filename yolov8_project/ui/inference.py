#!/usr/bin/env python3
"""
Inference utility functions for YOLO object detection models.
Handles model loading and running inference on images.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO


# Model paths relative to yolov8_project directory
MODEL_PATHS = {
    "YOLOv8": "runs/detect/finetune_phase/weights/best.pt",
    "YOLOv11": "runs/detect/yolov11_finetune_phase/weights/best.pt",
}


def get_device() -> str:
    """
    Auto-detect the best available device for inference.
    Returns: 'mps' (Mac), 'cuda:0' (NVIDIA GPU), or 'cpu'
    """
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    except:
        return "cpu"


def load_model(model_type: str, base_path: Optional[Path] = None) -> Tuple[YOLO, str]:
    """
    Load a YOLO model based on model type.
    
    Args:
        model_type: Either "YOLOv8" or "YOLOv11"
        base_path: Base path to the project directory. If None, uses current working directory.
    
    Returns:
        Tuple of (model, model_path)
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model_type is invalid
    """
    if model_type not in MODEL_PATHS:
        raise ValueError(f"Invalid model type: {model_type}. Must be 'YOLOv8' or 'YOLOv11'")
    
    if base_path is None:
        base_path = Path.cwd()
    
    model_path = base_path / MODEL_PATHS[model_type]
    model_path = model_path.expanduser().resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model = YOLO(str(model_path))
    return model, str(model_path)


def run_inference(
    model: YOLO,
    image: Image.Image,
    conf_threshold: float = 0.25,
    device: Optional[str] = None
) -> Dict:
    """
    Run inference on a single image.
    
    Args:
        model: Loaded YOLO model
        image: PIL Image object
        conf_threshold: Confidence threshold (0.0-1.0)
        device: Device to use ('mps', 'cuda:0', 'cpu'). If None, auto-detects.
    
    Returns:
        Dictionary containing:
        - 'results': Ultralytics results object
        - 'annotated_image': PIL Image with bounding boxes drawn
        - 'detections': List of detection dictionaries
        - 'class_summary': Dictionary with per-class statistics
    
    Raises:
        ValueError: If image is invalid or conf_threshold is out of range
        RuntimeError: If inference fails
    """
    # Validate inputs
    if image is None:
        raise ValueError("Image cannot be None")
    
    if not isinstance(image, Image.Image):
        raise ValueError(f"Expected PIL Image, got {type(image)}")
    
    if not (0.0 <= conf_threshold <= 1.0):
        raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {conf_threshold}")
    
    if device is None:
        device = get_device()
    
    try:
        # Convert PIL Image to numpy array for YOLO
        img_array = np.array(image)
        
        if img_array.size == 0:
            raise ValueError("Image array is empty")
        
        # Run prediction
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            device=device,
            verbose=False,
        )
    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}") from e
    
    # Get the first (and only) result
    result = results[0]
    
    # Get annotated image (YOLO automatically draws boxes)
    annotated_img = result.plot()
    annotated_image = Image.fromarray(annotated_img)
    
    # Extract detection details
    detections = []
    class_counts = {}
    class_confidences = {}
    
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detection = {
                "object_id": i + 1,
                "class": class_name,
                "confidence": conf,
                "confidence_pct": conf * 100,
                "bbox": {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                },
            }
            detections.append(detection)
            
            # Update class statistics
            if class_name not in class_counts:
                class_counts[class_name] = 0
                class_confidences[class_name] = []
            
            class_counts[class_name] += 1
            class_confidences[class_name].append(conf)
    
    # Create per-class summary
    class_summary = {}
    for class_name in class_counts.keys():
        confidences = class_confidences[class_name]
        class_summary[class_name] = {
            "count": class_counts[class_name],
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
        }
    
    return {
        "results": result,
        "annotated_image": annotated_image,
        "detections": detections,
        "class_summary": class_summary,
        "total_detections": len(detections),
        "device_used": device,
    }

