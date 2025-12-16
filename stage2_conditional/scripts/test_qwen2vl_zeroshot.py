#!/usr/bin/env python3
"""
Test Qwen2-VL 7B zero-shot performance on Stage 2 dataset
Tests conditional classification with spatial reasoning
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import time
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

DATASET_DIR = Path(__file__).parent.parent.parent / "quen2-vl.yolov11"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
    'rock toe', 'rock toe damaged',
    'slope drain', 'slope drain blocked', 'slope drain damaged',
    'vegetation'
]

def build_spatial_reasoning_prompt(image_path: str, all_objects: List[str], target_object: str) -> str:
    """Build prompt for spatial reasoning and conditional classification."""
    
    objects_list = ", ".join(all_objects) if all_objects else "No other objects detected"
    
    prompt = f"""Analyze this infrastructure inspection image.

Detected objects in the image: {objects_list}

Focus on the object: {target_object}

Classify its condition considering:
1. Visual appearance: Is it damaged, blocked, or in normal condition?
   - Look for cracks, wear, obstructions, or structural issues
   - Assess the physical state of the object
2. Spatial relationships: Where is it positioned relative to other objects?
   - Is a toe drain at the bottom or end of a slope drain?
   - Is a rock toe positioned above a toe drain?
   - What is the relative positioning and context?
3. Context: What is the overall state of the infrastructure?
   - Consider surrounding conditions
   - Assess environmental factors

Respond with ONLY the class name from this exact list:
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

def test_zero_shot(model, processor, split='valid', num_images=None):
    """Test zero-shot performance."""
    
    images_dir = DATASET_DIR / split / 'images'
    labels_dir = DATASET_DIR / split / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"❌ Error: {split} directory not found")
        return None
    
    results = {
        'correct': 0,
        'total': 0,
        'per_class': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_condition': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'per_object_type': defaultdict(lambda: {'correct': 0, 'total': 0}),
        'inference_times': [],
        'errors': [],
        'confusion': defaultdict(lambda: defaultdict(int))
    }
    
    image_files = list(images_dir.glob('*.jpg'))
    if num_images:
        image_files = image_files[:num_images]
    
    print(f"Testing on {len(image_files)} images from {split} split...")
    
    for img_file in tqdm(image_files, desc=f"Testing {split}"):
        label_file = labels_dir / (img_file.stem + '.txt')
        
        if not label_file.exists():
            continue
        
        # Load image
        try:
            image = Image.open(img_file).convert('RGB')
        except Exception as e:
            print(f"⚠️  Could not load {img_file}: {e}")
            continue
        
        # Read ground truth
        gt_classes = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_idx = int(parts[0])
                        if 0 <= class_idx < len(CLASS_NAMES):
                            gt_classes.append(CLASS_NAMES[class_idx])
                    except ValueError:
                        continue
        
        # Test each object
        for gt_class in gt_classes:
            # Build prompt
            prompt = build_spatial_reasoning_prompt(
                str(img_file), 
                gt_classes,
                gt_class
            )
            
            # Inference
            start_time = time.time()
            try:
                # Use simplified API compatible with latest transformers
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Apply chat template
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Process inputs (new API)
                inputs = processor(
                    text=[text],
                    images=[image],
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)
                
                # Generate
                generated_ids = model.generate(**inputs, max_new_tokens=50)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
                inference_time = time.time() - start_time
                results['inference_times'].append(inference_time)
                
                # Check if correct
                predicted = output_text.strip()
                
                # Try to match predicted class
                is_correct = False
                matched_class = None
                for class_name in CLASS_NAMES:
                    if class_name.lower() in predicted.lower() or predicted.lower() in class_name.lower():
                        is_correct = (class_name == gt_class)
                        matched_class = class_name
                        break
                
                results['total'] += 1
                results['per_class'][gt_class]['total'] += 1
                results['confusion'][gt_class][matched_class or predicted] += 1
                
                # Extract condition and object type for analysis
                if 'Blocked' in gt_class or 'blocked' in gt_class:
                    results['per_condition']['blocked']['total'] += 1
                elif 'Damaged' in gt_class or 'damaged' in gt_class:
                    results['per_condition']['damaged']['total'] += 1
                else:
                    results['per_condition']['normal']['total'] += 1
                
                if 'Toe drain' in gt_class:
                    results['per_object_type']['toe_drain']['total'] += 1
                elif 'slope drain' in gt_class.lower():
                    results['per_object_type']['slope_drain']['total'] += 1
                elif 'rock toe' in gt_class.lower():
                    results['per_object_type']['rock_toe']['total'] += 1
                elif 'vegetation' in gt_class.lower():
                    results['per_object_type']['vegetation']['total'] += 1
                
                if is_correct:
                    results['correct'] += 1
                    results['per_class'][gt_class]['correct'] += 1
                    
                    if 'Blocked' in gt_class or 'blocked' in gt_class:
                        results['per_condition']['blocked']['correct'] += 1
                    elif 'Damaged' in gt_class or 'damaged' in gt_class:
                        results['per_condition']['damaged']['correct'] += 1
                    else:
                        results['per_condition']['normal']['correct'] += 1
                    
                    if 'Toe drain' in gt_class:
                        results['per_object_type']['toe_drain']['correct'] += 1
                    elif 'slope drain' in gt_class.lower():
                        results['per_object_type']['slope_drain']['correct'] += 1
                    elif 'rock toe' in gt_class.lower():
                        results['per_object_type']['rock_toe']['correct'] += 1
                    elif 'vegetation' in gt_class.lower():
                        results['per_object_type']['vegetation']['correct'] += 1
                else:
                    results['errors'].append({
                        'image': img_file.name,
                        'gt': gt_class,
                        'predicted': predicted,
                        'matched': matched_class
                    })
            
            except Exception as e:
                results['errors'].append({
                    'image': img_file.name,
                    'gt': gt_class,
                    'error': str(e)
                })
                print(f"⚠️  Error processing {img_file.name}: {e}")
    
    return results

def print_results(results: Dict):
    """Print comprehensive results."""
    
    print("\n" + "="*80)
    print("QWEN2-VL 7B ZERO-SHOT RESULTS")
    print("="*80)
    
    if results['total'] == 0:
        print("❌ No results to display")
        return
    
    # Overall accuracy
    overall_acc = results['correct'] / results['total'] * 100
    print(f"\nOverall Accuracy: {results['correct']}/{results['total']} = {overall_acc:.2f}%")
    
    if results['inference_times']:
        avg_time = sum(results['inference_times']) / len(results['inference_times'])
        print(f"Average Inference Time: {avg_time:.2f}s per object")
        print(f"Total Inference Time: {sum(results['inference_times']):.2f}s")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for class_name in CLASS_NAMES:
        if results['per_class'][class_name]['total'] > 0:
            correct = results['per_class'][class_name]['correct']
            total = results['per_class'][class_name]['total']
            acc = correct / total * 100
            print(f"  {class_name:30s}: {correct:3d}/{total:3d} = {acc:5.2f}%")
    
    # Per-condition accuracy
    print("\nPer-Condition Accuracy:")
    for condition in ['normal', 'damaged', 'blocked']:
        if results['per_condition'][condition]['total'] > 0:
            correct = results['per_condition'][condition]['correct']
            total = results['per_condition'][condition]['total']
            acc = correct / total * 100
            print(f"  {condition:15s}: {correct:3d}/{total:3d} = {acc:5.2f}%")
    
    # Per-object-type accuracy
    print("\nPer-Object-Type Accuracy:")
    for obj_type in ['toe_drain', 'slope_drain', 'rock_toe', 'vegetation']:
        if results['per_object_type'][obj_type]['total'] > 0:
            correct = results['per_object_type'][obj_type]['correct']
            total = results['per_object_type'][obj_type]['total']
            acc = correct / total * 100
            print(f"  {obj_type:15s}: {correct:3d}/{total:3d} = {acc:5.2f}%")
    
    # Error analysis
    if results['errors']:
        print(f"\nErrors: {len(results['errors'])}")
        print("Sample errors (first 10):")
        for error in results['errors'][:10]:
            print(f"  Image: {error['image']}")
            print(f"    GT: {error.get('gt', 'N/A')}")
            print(f"    Predicted: {error.get('predicted', 'N/A')}")
            if 'error' in error:
                print(f"    Error: {error['error']}")
    
    print("\n" + "="*80)

def main():
    print("Loading Qwen2-VL 7B model...")
    
    # Check for available compute device (CUDA for NVIDIA, MPS for Apple Silicon, or CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    try:
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        
        print("Loading model and processor...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if device in ["cuda", "mps"] else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        
        print("✅ Model loaded successfully")
        
        print("\nTesting zero-shot performance on validation set...")
        results = test_zero_shot(model, processor, split='valid', num_images=47)
        
        if results:
            print_results(results)
            
            # Save results
            output_file = OUTPUT_DIR / "zeroshot_results.json"
            
            # Convert to JSON-serializable
            json_results = {
                'overall_accuracy': results['correct'] / results['total'] if results['total'] > 0 else 0,
                'correct': results['correct'],
                'total': results['total'],
                'per_class': {k: dict(v) for k, v in results['per_class'].items()},
                'per_condition': {k: dict(v) for k, v in results['per_condition'].items()},
                'per_object_type': {k: dict(v) for k, v in results['per_object_type'].items()},
                'avg_inference_time': sum(results['inference_times']) / len(results['inference_times']) if results['inference_times'] else 0,
                'total_inference_time': sum(results['inference_times']),
                'errors': results['errors'][:50]  # Save first 50 errors
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"\n✅ Results saved to {output_file}")
        else:
            print("❌ No results generated")
    
    except ImportError as e:
        print(f"❌ Error: Required packages not installed")
        print(f"Install with: pip install transformers torch accelerate")
        print(f"Error details: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

