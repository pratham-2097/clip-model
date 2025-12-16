#!/usr/bin/env python3
"""
Test Vision-Language Models (VLMs) for conditional classification.

This script tests multiple VLM candidates on Stage 2 dataset images
to identify the best model for conditional classification.

Usage:
    python scripts/test_vlm_models.py --model qwen2-vl --images 20
    python scripts/test_vlm_models.py --model all --images 10
"""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from PIL import Image
import psutil
import os

# Try importing transformers (required for VLMs)
try:
    from transformers import (
        AutoProcessor,
        AutoModelForVision2Seq,
        Qwen2VLForConditionalGeneration,
        AutoTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers not installed. Install with: pip install transformers torch pillow")

# Stage 2 dataset path
STAGE2_DATASET = Path(__file__).parent.parent.parent / "STEP 2- Conditional classes.v1-stage-2--1.yolov11"
STAGE2_IMAGES = STAGE2_DATASET / "train" / "images"
STAGE2_LABELS = STAGE2_DATASET / "train" / "labels"

# Conditional class mapping (9 classes → 3 conditions per object)
CONDITIONAL_CLASSES = {
    'Toe drain': 'normal',
    'Toe drain- Blocked': 'blocked',
    'Toe drain- Damaged': 'damaged',
    'rock toe': 'normal',
    'rock toe damaged': 'damaged',
    'slope drain': 'normal',
    'slope drain blocked': 'blocked',
    'slope drain damaged': 'damaged',
    'vegetation': 'normal',  # No conditional states
}

# Object type mapping
OBJECT_TYPES = {
    'Toe drain': 'toe_drain',
    'Toe drain- Blocked': 'toe_drain',
    'Toe drain- Damaged': 'toe_drain',
    'rock toe': 'rock_toe',
    'rock toe damaged': 'rock_toe',
    'slope drain': 'slope_drain',
    'slope drain blocked': 'slope_drain',
    'slope drain damaged': 'slope_drain',
    'vegetation': 'vegetation',
}


class VLMTester:
    """Base class for testing Vision-Language Models."""
    
    def __init__(self, model_name: str, model_id: str):
        self.model_name = model_name
        self.model_id = model_id
        self.processor = None
        self.model = None
        self.device = None
        self.loaded = False
        
    def load_model(self, device: str = "auto"):
        """Load the VLM model and processor."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")
        
        print(f"📦 Loading {self.model_name}...")
        
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"   Using device: {self.device}")
        
        try:
            # Load processor/tokenizer
            if "qwen2-vl" in self.model_id.lower():
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            else:
                # Generic loading for other models
                self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
            
            if self.device != "cuda" and self.device != "mps":
                self.model = self.model.to(self.device)
            
            self.loaded = True
            print(f"✅ {self.model_name} loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading {self.model_name}: {e}")
            traceback.print_exc()
            raise
    
    def classify_condition(self, image: Image.Image, object_type: str, prompt_template: str = None) -> Tuple[str, float]:
        """
        Classify the condition of an object in an image.
        
        Args:
            image: PIL Image
            object_type: One of 'rock_toe', 'slope_drain', 'toe_drain'
            prompt_template: Optional custom prompt
            
        Returns:
            (condition, confidence): Tuple of condition string and confidence score
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default prompt template
        if prompt_template is None:
            prompt_template = (
                f"Analyze this {object_type}. "
                "Is it: A) Normal, B) Damaged, C) Blocked? "
                "Consider visible damage, erosion, vegetation, or debris. "
                "Respond with only: Normal, Damaged, or Blocked."
            )
        
        try:
            # Prepare inputs
            if "qwen2-vl" in self.model_id.lower():
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt_template}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = self.processor.process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.device)
                
                # Generate
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            else:
                # Generic processing for other models
                inputs = self.processor(images=image, text=prompt_template, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Parse response
            response_lower = response.lower().strip()
            if "normal" in response_lower and "damaged" not in response_lower and "blocked" not in response_lower:
                condition = "normal"
            elif "damaged" in response_lower:
                condition = "damaged"
            elif "blocked" in response_lower:
                condition = "blocked"
            else:
                condition = "normal"  # Default
            
            # Simple confidence (could be improved with logits)
            confidence = 0.8 if condition in response_lower else 0.5
            
            return condition, confidence
            
        except Exception as e:
            print(f"⚠️  Error during classification: {e}")
            traceback.print_exc()
            return "normal", 0.0
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)


def load_yolo_labels(label_path: Path) -> List[Tuple[int, str]]:
    """Load YOLO format labels and return class indices and names."""
    if not label_path.exists():
        return []
    
    classes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_idx = int(parts[0])
                classes.append(class_idx)
    
    return classes


def get_ground_truth_condition(label_path: Path, class_names: List[str]) -> Optional[str]:
    """Extract ground truth condition from label file."""
    classes = load_yolo_labels(label_path)
    if not classes:
        return None
    
    # Get most common class (in case of multiple objects)
    if classes:
        class_idx = max(set(classes), key=classes.count)
        if class_idx < len(class_names):
            class_name = class_names[class_idx]
            return CONDITIONAL_CLASSES.get(class_name, 'normal')
    
    return None


def test_model(
    tester: VLMTester,
    image_paths: List[Path],
    label_paths: List[Path],
    class_names: List[str],
    num_images: int = 10
) -> Dict:
    """Test a VLM model on sample images."""
    
    print(f"\n🧪 Testing {tester.model_name} on {num_images} images...")
    
    # Load model
    start_time = time.time()
    tester.load_model()
    load_time = time.time() - start_time
    
    initial_memory = tester.get_memory_usage()
    
    results = {
        'model_name': tester.model_name,
        'model_id': tester.model_id,
        'load_time': load_time,
        'initial_memory_gb': initial_memory,
        'predictions': [],
        'metrics': {
            'total': 0,
            'correct': 0,
            'accuracy': 0.0,
            'avg_inference_time': 0.0,
            'per_condition': {'normal': {'correct': 0, 'total': 0},
                            'damaged': {'correct': 0, 'total': 0},
                            'blocked': {'correct': 0, 'total': 0}},
            'per_object_type': {'rock_toe': {'correct': 0, 'total': 0},
                               'slope_drain': {'correct': 0, 'total': 0},
                               'toe_drain': {'correct': 0, 'total': 0}}
        }
    }
    
    inference_times = []
    
    # Test on sample images
    for i, (img_path, label_path) in enumerate(zip(image_paths[:num_images], label_paths[:num_images])):
        if not img_path.exists():
            continue
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Get ground truth
        gt_condition = get_ground_truth_condition(label_path, class_names)
        if gt_condition is None:
            continue
        
        # Determine object type from ground truth class
        classes = load_yolo_labels(label_path)
        if classes:
            class_idx = max(set(classes), key=classes.count)
            if class_idx < len(class_names):
                class_name = class_names[class_idx]
                object_type = OBJECT_TYPES.get(class_name, 'unknown')
            else:
                object_type = 'unknown'
        else:
            object_type = 'unknown'
        
        # Skip vegetation (no conditional classification)
        if object_type == 'vegetation':
            continue
        
        # Classify
        start_inf = time.time()
        pred_condition, confidence = tester.classify_condition(image, object_type)
        inf_time = time.time() - start_inf
        inference_times.append(inf_time)
        
        # Check correctness
        is_correct = (pred_condition.lower() == gt_condition.lower())
        
        results['predictions'].append({
            'image': str(img_path.name),
            'object_type': object_type,
            'ground_truth': gt_condition,
            'prediction': pred_condition,
            'confidence': confidence,
            'correct': is_correct,
            'inference_time': inf_time
        })
        
        # Update metrics
        results['metrics']['total'] += 1
        if is_correct:
            results['metrics']['correct'] += 1
            results['metrics']['per_condition'][gt_condition]['correct'] += 1
            results['metrics']['per_object_type'][object_type]['correct'] += 1
        
        results['metrics']['per_condition'][gt_condition]['total'] += 1
        results['metrics']['per_object_type'][object_type]['total'] += 1
        
        print(f"   [{i+1}/{num_images}] {img_path.name}: GT={gt_condition}, Pred={pred_condition}, Correct={is_correct}")
    
    # Calculate final metrics
    if results['metrics']['total'] > 0:
        results['metrics']['accuracy'] = results['metrics']['correct'] / results['metrics']['total']
        results['metrics']['avg_inference_time'] = sum(inference_times) / len(inference_times) if inference_times else 0.0
    
    final_memory = tester.get_memory_usage()
    results['final_memory_gb'] = final_memory
    results['peak_memory_gb'] = final_memory  # Simplified
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test VLM models for conditional classification")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen2-vl",
        choices=["qwen2-vl", "internvl2", "llava", "all"],
        help="Model to test"
    )
    parser.add_argument(
        "--images",
        type=int,
        default=10,
        help="Number of images to test on"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vlm_test_results.json",
        help="Output JSON file for results"
    )
    args = parser.parse_args()
    
    # Check if transformers is available
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Error: transformers library not installed.")
        print("   Install with: pip install transformers torch pillow")
        return
    
    # Check Stage 2 dataset
    if not STAGE2_IMAGES.exists():
        print(f"❌ Error: Stage 2 dataset not found at {STAGE2_IMAGES}")
        return
    
    # Load class names from data.yaml
    data_yaml = STAGE2_DATASET / "data.yaml"
    class_names = []
    if data_yaml.exists():
        import yaml
        with open(data_yaml, 'r') as f:
            data = yaml.safe_load(f)
            class_names = data.get('names', [])
    
    # Get image and label paths
    image_paths = sorted(list(STAGE2_IMAGES.glob("*.jpg")))
    label_paths = []
    for img_path in image_paths:
        label_path = STAGE2_LABELS / (img_path.stem + ".txt")
        label_paths.append(label_path)
    
    if not image_paths:
        print(f"❌ Error: No images found in {STAGE2_IMAGES}")
        return
    
    print(f"📊 Found {len(image_paths)} images in Stage 2 dataset")
    print(f"   Testing on {min(args.images, len(image_paths))} images\n")
    
    # Model configurations
    models = {
        "qwen2-vl": {
            "name": "Qwen2-VL 7B",
            "id": "Qwen/Qwen2-VL-7B-Instruct"
        },
        "internvl2": {
            "name": "InternVL2 8B",
            "id": "OpenGVLab/InternVL2-8B"
        },
        "llava": {
            "name": "LLaVA-NeXT 13B",
            "id": "llava-hf/llava-1.5-13b-hf"
        }
    }
    
    # Test models
    all_results = []
    
    if args.model == "all":
        models_to_test = list(models.keys())
    else:
        models_to_test = [args.model]
    
    for model_key in models_to_test:
        if model_key not in models:
            print(f"⚠️  Warning: Unknown model {model_key}, skipping...")
            continue
        
        model_config = models[model_key]
        tester = VLMTester(model_config["name"], model_config["id"])
        
        try:
            results = test_model(tester, image_paths, label_paths, class_names, args.images)
            all_results.append(results)
            
            # Print summary
            print(f"\n📊 {model_config['name']} Results:")
            print(f"   Accuracy: {results['metrics']['accuracy']:.2%}")
            print(f"   Avg Inference Time: {results['metrics']['avg_inference_time']:.3f}s")
            print(f"   Memory Usage: {results['final_memory_gb']:.2f} GB")
            print(f"   Load Time: {results['load_time']:.2f}s")
            
        except Exception as e:
            print(f"❌ Error testing {model_config['name']}: {e}")
            traceback.print_exc()
            continue
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # Print comparison
    if len(all_results) > 1:
        print("\n📊 Model Comparison:")
        print(f"{'Model':<20} {'Accuracy':<12} {'Inference':<12} {'Memory':<10}")
        print("-" * 60)
        for result in all_results:
            print(f"{result['model_name']:<20} "
                  f"{result['metrics']['accuracy']:>10.2%} "
                  f"{result['metrics']['avg_inference_time']:>10.3f}s "
                  f"{result['final_memory_gb']:>8.2f}GB")


if __name__ == "__main__":
    main()


