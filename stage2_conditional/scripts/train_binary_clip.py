"""
Training Script for Hierarchical Binary Classifier

Fast training (10-20 minutes) using:
- Frozen CLIP ViT-B/32
- Only MLP head + object type embeddings trained (~175K params)
- Binary classification: NORMAL vs CONDITIONAL
- Weighted BCE loss
- Live epoch updates

Target: 85-90% accuracy in 8 epochs
"""

import os
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from tqdm import tqdm
import numpy as np

from binary_model import HierarchicalBinaryClassifier
from binary_dataset import create_binary_dataloaders

# Add models directory to path for registry
models_dir = Path(__file__).parent.parent / "models"
sys.path.insert(0, str(models_dir))
from register_model import add_model


class WeightedBCELoss(nn.Module):
    """Weighted binary cross-entropy loss."""
    
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
    
    def forward(self, logits, targets):
        # Move pos_weight to same device as logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        
        # Binary cross-entropy with logits
        loss = nn.functional.cross_entropy(logits, targets)
        return loss


def train_epoch(
    model: HierarchicalBinaryClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: WeightedBCELoss,
    device: str,
    epoch: int,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        pixel_values = batch['pixel_values'].to(device)
        object_type_ids = batch['object_type_id'].to(device)
        spatial_features = batch['spatial_features'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(pixel_values, object_type_ids, spatial_features)
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total,
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(
    model: HierarchicalBinaryClassifier,
    data_loader: DataLoader,
    device: str,
    split_name: str = 'valid',
) -> dict:
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    
    # Per-class metrics
    class_correct = {0: 0, 1: 0}  # NORMAL, CONDITIONAL
    class_total = {0: 0, 1: 0}
    
    # Confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
            # Move to device
            pixel_values = batch['pixel_values'].to(device)
            object_type_ids = batch['object_type_id'].to(device)
            spatial_features = batch['spatial_features'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(pixel_values, object_type_ids, spatial_features)
            
            # Predictions
            _, predicted = logits.max(1)
            
            # Overall accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
                
                # Confusion matrix (0=NORMAL, 1=CONDITIONAL)
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
    
    # Per-class accuracy
    normal_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    conditional_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    # Precision, Recall, F1
    precision_cond = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_cond = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_cond = 2 * precision_cond * recall_cond / (precision_cond + recall_cond) if (precision_cond + recall_cond) > 0 else 0
    
    precision_norm = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_norm = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_norm = 2 * precision_norm * recall_norm / (precision_norm + recall_norm) if (precision_norm + recall_norm) > 0 else 0
    
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
        'confusion_matrix': [[tn, fp], [fn, tp]],
        'correct': correct,
        'total': total,
    }


def main():
    parser = argparse.ArgumentParser(description='Train hierarchical binary classifier')
    parser.add_argument('--dataset_dir', type=str, default='../../quen2-vl.yolov11',
                       help='Path to dataset directory')
    parser.add_argument('--yolo_model', type=str,
                       default='../../yolov8_project/runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt',
                       help='Path to Stage 1 YOLO model')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                       help='CLIP model name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=8,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--output_dir', type=str, default='../models/clip_binary_fast',
                       help='Output directory for model')
    parser.add_argument('--use_cache', action='store_true', default=True,
                       help='Use cached YOLO detections')
    
    args = parser.parse_args()
    
    # Setup device
    if torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Using MPS (M2 Max GPU)")
    elif torch.cuda.is_available():
        device = 'cuda'
        print("✅ Using CUDA")
    else:
        device = 'cpu'
        print("⚠️  Using CPU")
    
    print(f"\n{'='*80}")
    print("HIERARCHICAL BINARY CLASSIFIER TRAINING")
    print("Frozen CLIP + Spatial Features + Object Type → NORMAL vs CONDITIONAL")
    print(f"{'='*80}")
    print(f"Model: {args.clip_model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*80}\n")
    
    # Load processor
    print("Loading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, valid_loader, test_loader = create_binary_dataloaders(
        args.dataset_dir,
        args.yolo_model,
        processor,
        batch_size=args.batch_size,
        use_cache=args.use_cache,
    )
    
    # Create model
    print(f"\nLoading model: {args.clip_model}...")
    model = HierarchicalBinaryClassifier(
        clip_model_name=args.clip_model,
        num_object_types=4,
        object_type_embed_dim=32,
        spatial_feature_dim=9,
        hidden_dims=[256, 128],
        dropout=0.3,
    )
    model = model.to(device)
    model.print_parameter_summary()
    
    # Setup loss (slight weight for CONDITIONAL if imbalanced)
    criterion = WeightedBCELoss(pos_weight=1.0)
    
    # Setup optimizer (only trainable parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
    training_start = datetime.now()
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_metrics': [],
    }
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Evaluate
        val_metrics = evaluate(model, valid_loader, device, 'valid')
        val_acc = val_metrics['accuracy']
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        training_history['val_metrics'].append(val_metrics)
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Acc:  {train_acc:.2f}%")
        print(f"  Val Acc:    {val_acc:.2f}%")
        print(f"  Normal Acc: {val_metrics['normal_accuracy']:.2f}%")
        print(f"  Conditional Acc: {val_metrics['conditional_accuracy']:.2f}%")
        print(f"  Conditional Recall: {val_metrics['recall_conditional']:.2f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_metrics': val_metrics,
            }, output_dir / 'best_model.pt')
            
            print(f"\n✅ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}\n")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    test_metrics = evaluate(model, test_loader, device, 'test')
    test_acc = test_metrics['accuracy']
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"  Normal Accuracy: {test_metrics['normal_accuracy']:.2f}%")
    print(f"  Conditional Accuracy: {test_metrics['conditional_accuracy']:.2f}%")
    print(f"\nPrecision/Recall:")
    print(f"  Normal - P: {test_metrics['precision_normal']:.3f}, R: {test_metrics['recall_normal']:.3f}, F1: {test_metrics['f1_normal']:.3f}")
    print(f"  Conditional - P: {test_metrics['precision_conditional']:.3f}, R: {test_metrics['recall_conditional']:.3f}, F1: {test_metrics['f1_conditional']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={test_metrics['confusion_matrix'][0][0]}, FP={test_metrics['confusion_matrix'][0][1]}],")
    print(f"   [FN={test_metrics['confusion_matrix'][1][0]}, TP={test_metrics['confusion_matrix'][1][1]}]]")
    
    # Save results
    results = {
        'model_name': args.clip_model,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_metrics': test_metrics,
        'training_history': training_history,
        'args': vars(args),
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Training complete!")
    print(f"✅ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✅ Test accuracy: {test_acc:.2f}%")
    print(f"✅ Model saved to: {output_dir}")
    
    # Get git commit hash if available
    git_commit = None
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except:
        pass
    
    # Get dataset info
    dataset_info = {
        "train_samples": len(train_loader.dataset),
        "valid_samples": len(valid_loader.dataset),
        "test_samples": len(test_loader.dataset),
    }
    
    # Count normal vs conditional
    try:
        normal_count = sum(1 for s in train_loader.dataset.samples if s['binary_label'] == 0)
        conditional_count = sum(1 for s in train_loader.dataset.samples if s['binary_label'] == 1)
        dataset_info["normal_count"] = normal_count
        dataset_info["conditional_count"] = conditional_count
        dataset_info["balance_ratio"] = max(normal_count, conditional_count) / min(normal_count, conditional_count) if min(normal_count, conditional_count) > 0 else 0
    except:
        pass
    
    # Get model file size
    model_file = output_dir / 'best_model.pt'
    file_size_mb = model_file.stat().st_size / (1024 * 1024) if model_file.exists() else None
    
    # Calculate training time
    training_time_seconds = (datetime.now() - training_start).total_seconds() if 'training_start' in locals() else None
    
    # Register model in registry
    print(f"\n📝 Registering model in registry...")
    try:
        # Calculate relative path from models directory
        models_base = Path(__file__).parent.parent / "models"
        output_dir_abs = Path(args.output_dir).resolve()
        if not output_dir_abs.is_absolute():
            output_dir_abs = (Path(__file__).parent.parent / args.output_dir).resolve()
        
        # Get relative path
        try:
            model_path_rel = output_dir_abs.relative_to(models_base) / "best_model.pt"
        except ValueError:
            # If not relative, use absolute path from models directory
            model_path_rel = Path(args.output_dir) / "best_model.pt"
        
        model_path = str(model_path_rel).replace("\\", "/")  # Normalize path
        
        add_model(
            model_path=model_path,
            model_type="CLIP-B32-Binary",
            name=f"CLIP ViT-B/32 Binary - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            performance={
                "validation_accuracy": best_val_acc,
                "test_accuracy": test_acc,
                "normal_accuracy": test_metrics['normal_accuracy'],
                "conditional_accuracy": test_metrics['conditional_accuracy'],
                "conditional_recall": test_metrics['recall_conditional'],
                "conditional_precision": test_metrics['precision_conditional'],
                "f1_conditional": test_metrics['f1_conditional'],
                "training_time_seconds": training_time_seconds,
                "inference_time_ms_per_object": 100  # Estimated
            },
            training_config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "weight_decay": args.weight_decay,
                "optimizer": "AdamW",
                "loss": "WeightedBCE",
                "device": device
            },
            dataset_info=dataset_info,
            notes=f"Trained with {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}",
            git_commit=git_commit,
            file_size_mb=file_size_mb,
            status="active"
        )
        print(f"✅ Model registered successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Could not register model: {e}")
        print(f"   You can register manually using: python3 register_model.py add --model-path {model_path}")
    
    # Check if target met
    if test_acc >= 85.0:
        print(f"\n🎉 SUCCESS! Target accuracy (≥85%) ACHIEVED!")
    elif test_acc >= 80.0:
        print(f"\n✅ Good! Close to target (≥80%)")
    else:
        print(f"\n⚠️  Below target. Consider:")
        print(f"   - Training for more epochs")
        print(f"   - Adjusting learning rate")
        print(f"   - Checking spatial features")


if __name__ == '__main__':
    main()

