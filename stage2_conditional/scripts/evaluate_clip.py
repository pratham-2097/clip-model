"""
CLIP Model Evaluation Script

Comprehensive evaluation with:
- Overall accuracy
- Per-class accuracy (handles imbalance)
- Per-condition accuracy (normal/damaged/blocked)
- Confusion matrix
- F1-scores
- Zero-shot vs fine-tuned comparison
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
import numpy as np
from transformers import CLIPProcessor
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from clip_dataset import create_dataloaders, CLIPConditionalDataset
from finetune_clip_conditional import CLIPClassifier


def evaluate_model(
    model: CLIPClassifier,
    data_loader,
    device: str,
    class_names: List[str],
) -> Dict:
    """Comprehensive evaluation of the model."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_class_names = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            
            # Predictions
            _, predicted = outputs['logits'].max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_class_names.extend(batch['class_name'])
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall metrics
    accuracy = 100. * np.mean(all_preds == all_labels)
    
    # Per-class metrics
    per_class_correct = {}
    per_class_total = {}
    per_class_accuracy = {}
    
    for i, class_name in enumerate(class_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class_total[class_name] = mask.sum()
            per_class_correct[class_name] = (all_preds[mask] == all_labels[mask]).sum()
            per_class_accuracy[class_name] = 100. * per_class_correct[class_name] / per_class_total[class_name]
        else:
            per_class_total[class_name] = 0
            per_class_correct[class_name] = 0
            per_class_accuracy[class_name] = 0.0
    
    # Per-condition metrics
    condition_metrics = {}
    for condition in ['normal', 'blocked', 'damaged']:
        # Find classes with this condition
        condition_mask = np.array([condition.lower() in name.lower() for name in all_class_names])
        if condition_mask.sum() > 0:
            condition_preds = all_preds[condition_mask]
            condition_labels = all_labels[condition_mask]
            condition_accuracy = 100. * np.mean(condition_preds == condition_labels)
            condition_metrics[condition] = {
                'accuracy': condition_accuracy,
                'total': condition_mask.sum(),
            }
    
    # Per-object-type metrics
    object_type_metrics = {}
    for obj_type in ['toe_drain', 'slope_drain', 'rock_toe', 'vegetation']:
        # Find classes with this object type
        type_mask = np.array([obj_type.replace('_', ' ').lower() in name.lower() 
                             for name in all_class_names])
        if type_mask.sum() > 0:
            type_preds = all_preds[type_mask]
            type_labels = all_labels[type_mask]
            type_accuracy = 100. * np.mean(type_preds == type_labels)
            object_type_metrics[obj_type] = {
                'accuracy': type_accuracy,
                'total': type_mask.sum(),
            }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # F1 scores
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'per_class_total': per_class_total,
        'per_class_correct': per_class_correct,
        'condition_metrics': condition_metrics,
        'object_type_metrics': object_type_metrics,
        'confusion_matrix': cm.tolist(),
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'total_samples': len(all_labels),
        'correct_predictions': int((all_preds == all_labels).sum()),
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: Path):
    """Plot confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved to {output_path}")


def print_evaluation_results(results: Dict, split_name: str):
    """Print evaluation results in a nice format."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {split_name.upper()} SET")
    print(f"{'='*80}")
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:     {results['accuracy']:.2f}%")
    print(f"  F1 (macro):   {results['f1_macro']:.4f}")
    print(f"  F1 (weighted):{results['f1_weighted']:.4f}")
    print(f"  Correct:      {results['correct_predictions']}/{results['total_samples']}")
    
    print(f"\n📊 Per-Class Accuracy:")
    for class_name in CLIPConditionalDataset.CLASS_NAMES:
        acc = results['per_class_accuracy'].get(class_name, 0)
        total = results['per_class_total'].get(class_name, 0)
        correct = results['per_class_correct'].get(class_name, 0)
        print(f"  {class_name:30s}: {acc:6.2f}% ({correct:3d}/{total:3d})")
    
    if 'condition_metrics' in results:
        print(f"\n📊 Per-Condition Accuracy:")
        for condition, metrics in results['condition_metrics'].items():
            acc = metrics['accuracy']
            total = metrics['total']
            print(f"  {condition.capitalize():10s}: {acc:6.2f}% ({total:4d} samples)")
    
    if 'object_type_metrics' in results:
        print(f"\n📊 Per-Object-Type Accuracy:")
        for obj_type, metrics in results['object_type_metrics'].items():
            acc = metrics['accuracy']
            total = metrics['total']
            print(f"  {obj_type:15s}: {acc:6.2f}% ({total:4d} samples)")
    
    # Check targets
    print(f"\n🎯 Target Achievement:")
    if results['accuracy'] >= 90.0:
        print(f"  ✅ Overall Accuracy (>90%): ACHIEVED ({results['accuracy']:.2f}%)")
    else:
        print(f"  ❌ Overall Accuracy (>90%): NOT MET ({results['accuracy']:.2f}%)")
    
    # Check rare classes
    rare_classes = ['Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged']
    rare_acc = np.mean([results['per_class_accuracy'].get(c, 0) for c in rare_classes])
    if rare_acc >= 70.0:
        print(f"  ✅ Rare Classes (>70%): ACHIEVED ({rare_acc:.2f}%)")
    else:
        print(f"  ❌ Rare Classes (>70%): NOT MET ({rare_acc:.2f}%)")
    
    # Check blocked condition
    if 'condition_metrics' in results and 'blocked' in results['condition_metrics']:
        blocked_acc = results['condition_metrics']['blocked']['accuracy']
        if blocked_acc >= 60.0:
            print(f"  ✅ Blocked Condition (>60%): ACHIEVED ({blocked_acc:.2f}%)")
        else:
            print(f"  ❌ Blocked Condition (>60%): NOT MET ({blocked_acc:.2f}%)")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate CLIP model')
    parser.add_argument('--dataset_dir', type=str, default='../quen2-vl.yolov11',
                       help='Path to dataset directory')
    parser.add_argument('--model_path', type=str, default='../models/clip_conditional_final',
                       help='Path to trained model')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14',
                       help='Base CLIP model name')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--zero_shot', action='store_true',
                       help='Evaluate zero-shot (no fine-tuning)')
    parser.add_argument('--output_dir', type=str, default='../experiments',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    
    # Load processor
    print(f"\nLoading processor...")
    if args.zero_shot:
        processor = CLIPProcessor.from_pretrained(args.model_name)
    else:
        model_path = Path(args.model_path)
        processor = CLIPProcessor.from_pretrained(model_path / 'clip_model')
    
    # Create dataloader
    print(f"\nLoading {args.split} dataset...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        args.dataset_dir,
        processor,
        batch_size=args.batch_size,
        oversample=False,  # No oversampling for evaluation
        spatial_context=True,
    )
    
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'valid':
        data_loader = valid_loader
    else:
        data_loader = test_loader
    
    # Load model
    print(f"\nLoading model...")
    if args.zero_shot:
        print("  Using zero-shot (pre-trained) model")
        model = CLIPClassifier(args.model_name, num_classes=9)
    else:
        print(f"  Loading fine-tuned model from {args.model_path}")
        model = CLIPClassifier(args.model_name, num_classes=9)
        checkpoint = torch.load(Path(args.model_path) / 'best_model.pt', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded model from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.2f}%)")
    
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print(f"\n{'='*80}")
    print(f"EVALUATING ON {args.split.upper()} SET")
    print(f"{'='*80}\n")
    
    results = evaluate_model(
        model,
        data_loader,
        device,
        CLIPConditionalDataset.CLASS_NAMES,
    )
    
    # Print results
    print_evaluation_results(results, args.split)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.zero_shot:
        results_file = output_dir / f'clip_zeroshot_{args.split}_results.json'
    else:
        results_file = output_dir / f'clip_finetuned_{args.split}_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Results saved to {results_file}")
    
    # Plot confusion matrix
    cm = np.array(results['confusion_matrix'])
    if args.zero_shot:
        cm_file = output_dir / f'clip_zeroshot_{args.split}_confusion_matrix.png'
    else:
        cm_file = output_dir / f'clip_finetuned_{args.split}_confusion_matrix.png'
    
    plot_confusion_matrix(cm, CLIPConditionalDataset.CLASS_NAMES, cm_file)


if __name__ == '__main__':
    main()

