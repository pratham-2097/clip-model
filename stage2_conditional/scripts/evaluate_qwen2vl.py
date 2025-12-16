#!/usr/bin/env python3
"""
Comprehensive evaluation of Qwen2-VL 7B for conditional classification
Tests on test set and generates detailed metrics
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

DATASET_DIR = Path(__file__).parent.parent.parent / "quen2-vl.yolov11"
MODELS_DIR = Path(__file__).parent.parent / "models"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
    'rock toe', 'rock toe damaged',
    'slope drain', 'slope drain blocked', 'slope drain damaged',
    'vegetation'
]

def load_model(model_path: Path = None):
    """Load Qwen2-VL 7B model (fine-tuned or base)."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    if model_path and model_path.exists():
        print(f"Loading fine-tuned model from {model_path}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(str(model_path))
    else:
        print("Loading base Qwen2-VL 7B model")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    
    return model, processor

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

def evaluate_model(model, processor, split='test'):
    """Comprehensive evaluation on test set."""
    
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
        'predictions': [],
        'ground_truth': [],
        'inference_times': []
    }
    
    image_files = list(images_dir.glob('*.jpg'))
    print(f"Evaluating on {len(image_files)} images from {split} split...")
    
    for img_file in tqdm(image_files, desc=f"Evaluating {split}"):
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
                ).to(model.device)
                
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
                
                # Match predicted class
                predicted = output_text.strip()
                matched_class = None
                for class_name in CLASS_NAMES:
                    if class_name.lower() in predicted.lower() or predicted.lower() in class_name.lower():
                        matched_class = class_name
                        break
                
                if matched_class is None:
                    matched_class = predicted  # Keep original if no match
                
                results['predictions'].append(matched_class)
                results['ground_truth'].append(gt_class)
                
                # Update metrics
                is_correct = (matched_class == gt_class)
                results['total'] += 1
                results['per_class'][gt_class]['total'] += 1
                
                if is_correct:
                    results['correct'] += 1
                    results['per_class'][gt_class]['correct'] += 1
                
                # Condition metrics
                if 'Blocked' in gt_class or 'blocked' in gt_class:
                    results['per_condition']['blocked']['total'] += 1
                    if is_correct:
                        results['per_condition']['blocked']['correct'] += 1
                elif 'Damaged' in gt_class or 'damaged' in gt_class:
                    results['per_condition']['damaged']['total'] += 1
                    if is_correct:
                        results['per_condition']['damaged']['correct'] += 1
                else:
                    results['per_condition']['normal']['total'] += 1
                    if is_correct:
                        results['per_condition']['normal']['correct'] += 1
                
                # Object type metrics
                if 'Toe drain' in gt_class:
                    results['per_object_type']['toe_drain']['total'] += 1
                    if is_correct:
                        results['per_object_type']['toe_drain']['correct'] += 1
                elif 'slope drain' in gt_class.lower():
                    results['per_object_type']['slope_drain']['total'] += 1
                    if is_correct:
                        results['per_object_type']['slope_drain']['correct'] += 1
                elif 'rock toe' in gt_class.lower():
                    results['per_object_type']['rock_toe']['total'] += 1
                    if is_correct:
                        results['per_object_type']['rock_toe']['correct'] += 1
                elif 'vegetation' in gt_class.lower():
                    results['per_object_type']['vegetation']['total'] += 1
                    if is_correct:
                        results['per_object_type']['vegetation']['correct'] += 1
            
            except Exception as e:
                print(f"⚠️  Error processing {img_file.name}: {e}")
                continue
    
    return results

def generate_confusion_matrix(results: Dict, output_dir: Path):
    """Generate and save confusion matrix."""
    
    if not results['predictions'] or not results['ground_truth']:
        print("⚠️  No predictions to generate confusion matrix")
        return
    
    # Create confusion matrix
    cm = confusion_matrix(results['ground_truth'], results['predictions'], labels=CLASS_NAMES)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Qwen2-VL 7B Conditional Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_file = output_dir / "confusion_matrix.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {output_file}")
    plt.close()

def print_evaluation_report(results: Dict):
    """Print comprehensive evaluation report."""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE EVALUATION REPORT - QWEN2-VL 7B")
    print("="*80)
    
    if results['total'] == 0:
        print("❌ No results to display")
        return
    
    # Overall accuracy
    overall_acc = results['correct'] / results['total'] * 100
    print(f"\n📊 OVERALL METRICS")
    print("-"*80)
    print(f"Overall Accuracy: {results['correct']}/{results['total']} = {overall_acc:.2f}%")
    
    if results['inference_times']:
        avg_time = sum(results['inference_times']) / len(results['inference_times'])
        print(f"Average Inference Time: {avg_time:.2f}s per object")
        print(f"Total Inference Time: {sum(results['inference_times']):.2f}s")
    
    # Per-class accuracy
    print(f"\n📈 PER-CLASS ACCURACY")
    print("-"*80)
    for class_name in CLASS_NAMES:
        if results['per_class'][class_name]['total'] > 0:
            correct = results['per_class'][class_name]['correct']
            total = results['per_class'][class_name]['total']
            acc = correct / total * 100
            status = "✅" if acc >= 85 else "⚠️" if acc >= 70 else "❌"
            print(f"  {status} {class_name:30s}: {correct:3d}/{total:3d} = {acc:5.2f}%")
    
    # Per-condition accuracy
    print(f"\n📊 PER-CONDITION ACCURACY")
    print("-"*80)
    for condition in ['normal', 'damaged', 'blocked']:
        if results['per_condition'][condition]['total'] > 0:
            correct = results['per_condition'][condition]['correct']
            total = results['per_condition'][condition]['total']
            acc = correct / total * 100
            status = "✅" if acc >= 85 else "⚠️" if acc >= 70 else "❌"
            print(f"  {status} {condition:15s}: {correct:3d}/{total:3d} = {acc:5.2f}%")
    
    # Per-object-type accuracy
    print(f"\n📊 PER-OBJECT-TYPE ACCURACY")
    print("-"*80)
    for obj_type in ['toe_drain', 'slope_drain', 'rock_toe', 'vegetation']:
        if results['per_object_type'][obj_type]['total'] > 0:
            correct = results['per_object_type'][obj_type]['correct']
            total = results['per_object_type'][obj_type]['total']
            acc = correct / total * 100
            status = "✅" if acc >= 85 else "⚠️" if acc >= 70 else "❌"
            print(f"  {status} {obj_type:15s}: {correct:3d}/{total:3d} = {acc:5.2f}%")
    
    # Success criteria check
    print(f"\n🎯 SUCCESS CRITERIA CHECK")
    print("-"*80)
    print(f"  Overall Accuracy >90%: {'✅' if overall_acc >= 90 else '❌'} ({overall_acc:.2f}%)")
    
    condition_accs = []
    for condition in ['normal', 'damaged', 'blocked']:
        if results['per_condition'][condition]['total'] > 0:
            acc = results['per_condition'][condition]['correct'] / results['per_condition'][condition]['total'] * 100
            condition_accs.append(acc)
            print(f"  {condition.capitalize()} >85%: {'✅' if acc >= 85 else '❌'} ({acc:.2f}%)")
    
    print("\n" + "="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to fine-tuned model (default: use base model)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    args = parser.parse_args()
    
    model_path = Path(args.model_path) if args.model_path else None
    if model_path and not model_path.exists():
        # Try in models directory
        model_path = MODELS_DIR / args.model_path
        if not model_path.exists():
            model_path = None
    
    print("="*80)
    print("COMPREHENSIVE EVALUATION - QWEN2-VL 7B")
    print("="*80)
    
    # Load model
    print("\n1. Loading model...")
    model, processor = load_model(model_path)
    print("✅ Model loaded")
    
    # Evaluate
    print(f"\n2. Evaluating on {args.split} split...")
    results = evaluate_model(model, processor, split=args.split)
    
    if results:
        # Print report
        print_evaluation_report(results)
        
        # Generate confusion matrix
        print("\n3. Generating confusion matrix...")
        generate_confusion_matrix(results, EXPERIMENTS_DIR)
        
        # Save results
        output_file = EXPERIMENTS_DIR / f"final_evaluation_{args.split}.json"
        
        json_results = {
            'overall_accuracy': results['correct'] / results['total'] if results['total'] > 0 else 0,
            'correct': results['correct'],
            'total': results['total'],
            'per_class': {k: dict(v) for k, v in results['per_class'].items()},
            'per_condition': {k: dict(v) for k, v in results['per_condition'].items()},
            'per_object_type': {k: dict(v) for k, v in results['per_object_type'].items()},
            'avg_inference_time': sum(results['inference_times']) / len(results['inference_times']) if results['inference_times'] else 0,
            'predictions': results['predictions'],
            'ground_truth': results['ground_truth']
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✅ Results saved to {output_file}")
    else:
        print("❌ No results generated")

if __name__ == '__main__':
    main()

