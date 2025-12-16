"""
Evaluation Script for Hierarchical Binary Classifier

Comprehensive evaluation with:
- Overall accuracy
- Per-class metrics (NORMAL vs CONDITIONAL)
- Per-object-type breakdown
- Confusion matrix
- Precision/Recall/F1 scores
"""

import argparse
import json
from pathlib import Path
import torch
from transformers import CLIPProcessor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from binary_model import HierarchicalBinaryClassifier
from binary_dataset import create_binary_dataloaders


def evaluate_with_breakdown(
    model,
    data_loader,
    device,
    split_name='test',
):
    """Evaluate with detailed breakdown by object type."""
    model.eval()
    
    # Overall metrics
    correct = 0
    total = 0
    
    # Per-class
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    
    # Per-object-type
    obj_type_correct = {}
    obj_type_total = {}
    
    # Confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Collect all predictions
    all_preds = []
    all_labels = []
    all_obj_types = []
    
    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(device)
            object_type_ids = batch['object_type_id'].to(device)
            spatial_features = batch['spatial_features'].to(device)
            labels = batch['labels'].to(device)
            obj_types = batch['object_type']
            
            # Forward pass
            logits = model(pixel_values, object_type_ids, spatial_features)
            _, predicted = logits.max(1)
            
            # Store
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_obj_types.extend(obj_types)
            
            # Overall
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                obj_type = obj_types[i]
                
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                
                # Per-object-type
                if obj_type not in obj_type_total:
                    obj_type_total[obj_type] = 0
                    obj_type_correct[obj_type] = 0
                obj_type_total[obj_type] += 1
                if pred == label:
                    obj_type_correct[obj_type] += 1
                
                # Confusion matrix
                if label == 0 and pred == 0:
                    tn += 1
                elif label == 0 and pred == 1:
                    fp += 1
                elif label == 1 and pred == 0:
                    fn += 1
                elif label == 1 and pred == 1:
                    tp += 1
    
    # Calculate metrics
    accuracy = 100. * correct / total
    normal_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    conditional_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    # Precision, Recall, F1
    precision_cond = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_cond = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_cond = 2 * precision_cond * recall_cond / (precision_cond + recall_cond) if (precision_cond + recall_cond) > 0 else 0
    
    precision_norm = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_norm = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_norm = 2 * precision_norm * recall_norm / (precision_norm + recall_norm) if (precision_norm + recall_norm) > 0 else 0
    
    # Per-object-type accuracy
    obj_type_acc = {}
    for obj_type in obj_type_total:
        obj_type_acc[obj_type] = 100. * obj_type_correct[obj_type] / obj_type_total[obj_type]
    
    return {
        'accuracy': accuracy,
        'normal_accuracy': normal_acc,
        'conditional_accuracy': conditional_acc,
        'precision_normal': precision_norm,
        'precision_conditional': precision_cond,
        'recall_normal': recall_norm,
        'recall_conditional': recall_cond,
        'f1_normal': f1_norm,
        'f1_conditional': f1_cond,
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
        'object_type_accuracy': obj_type_acc,
        'object_type_total': obj_type_total,
        'correct': correct,
        'total': total,
    }


def print_results(results, split_name):
    """Print evaluation results."""
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {split_name.upper()} SET")
    print(f"{'='*80}\n")
    
    print(f"📊 Overall Metrics:")
    print(f"  Accuracy: {results['accuracy']:.2f}%")
    print(f"  Correct:  {results['correct']}/{results['total']}")
    
    print(f"\n📊 Per-Class Accuracy:")
    print(f"  NORMAL:      {results['normal_accuracy']:.2f}%")
    print(f"  CONDITIONAL: {results['conditional_accuracy']:.2f}%")
    
    print(f"\n📊 Precision/Recall/F1:")
    print(f"  NORMAL:")
    print(f"    Precision: {results['precision_normal']:.3f}")
    print(f"    Recall:    {results['recall_normal']:.3f}")
    print(f"    F1-score:  {results['f1_normal']:.3f}")
    print(f"  CONDITIONAL:")
    print(f"    Precision: {results['precision_conditional']:.3f}")
    print(f"    Recall:    {results['recall_conditional']:.3f}")
    print(f"    F1-score:  {results['f1_conditional']:.3f}")
    
    print(f"\n📊 Confusion Matrix:")
    cm = results['confusion_matrix']
    print(f"  [[TN={cm[0][0]:3d}, FP={cm[0][1]:3d}],")
    print(f"   [FN={cm[1][0]:3d}, TP={cm[1][1]:3d}]]")
    
    print(f"\n📊 Per-Object-Type Accuracy:")
    for obj_type in sorted(results['object_type_accuracy'].keys()):
        acc = results['object_type_accuracy'][obj_type]
        total = results['object_type_total'][obj_type]
        print(f"  {obj_type:15s}: {acc:6.2f}% ({total:3d} samples)")
    
    # Success criteria
    print(f"\n🎯 Success Criteria:")
    if results['accuracy'] >= 85.0:
        print(f"  ✅ Overall Accuracy (≥85%): ACHIEVED ({results['accuracy']:.2f}%)")
    else:
        print(f"  ❌ Overall Accuracy (≥85%): NOT MET ({results['accuracy']:.2f}%)")
    
    if results['recall_conditional'] >= 0.85:
        print(f"  ✅ CONDITIONAL Recall (≥0.85): ACHIEVED ({results['recall_conditional']:.2f})")
    else:
        print(f"  ❌ CONDITIONAL Recall (≥0.85): NOT MET ({results['recall_conditional']:.2f})")
    
    if results['normal_accuracy'] >= 80.0 and results['conditional_accuracy'] >= 80.0:
        print(f"  ✅ No Class Collapse (both >80%): ACHIEVED")
    else:
        print(f"  ❌ No Class Collapse (both >80%): NOT MET")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate hierarchical binary classifier')
    parser.add_argument('--dataset_dir', type=str, default='../../quen2-vl.yolov11',
                       help='Path to dataset directory')
    parser.add_argument('--yolo_model', type=str,
                       default='../../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt',
                       help='Path to Stage 1 YOLO model')
    parser.add_argument('--model_path', type=str, default='../models/clip_binary_fast',
                       help='Path to trained model')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
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
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    
    # Create dataloaders
    print(f"\nLoading {args.split} dataset...")
    train_loader, valid_loader, test_loader = create_binary_dataloaders(
        args.dataset_dir,
        args.yolo_model,
        processor,
        batch_size=args.batch_size,
        use_cache=True,
    )
    
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'valid':
        data_loader = valid_loader
    else:
        data_loader = test_loader
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = HierarchicalBinaryClassifier(
        clip_model_name=args.clip_model,
        num_object_types=4,
        object_type_embed_dim=32,
        spatial_feature_dim=9,
        hidden_dims=[256, 128],
        dropout=0.3,
    )
    
    checkpoint = torch.load(Path(args.model_path) / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_acc: {checkpoint['val_acc']:.2f}%)")
    
    # Evaluate
    print(f"\n{'='*80}")
    print(f"EVALUATING ON {args.split.upper()} SET")
    print(f"{'='*80}\n")
    
    results = evaluate_with_breakdown(model, data_loader, device, args.split)
    
    # Print results
    print_results(results, args.split)
    
    # Save results
    output_dir = Path(args.model_path)
    results_file = output_dir / f'{args.split}_results.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Results saved to {results_file}")


if __name__ == '__main__':
    main()

