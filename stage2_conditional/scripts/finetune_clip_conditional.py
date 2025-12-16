"""
CLIP Fine-Tuning for Conditional Classification

Fine-tunes CLIP ViT-L/14 for 9-class conditional classification with:
- Class imbalance handling (weighted loss)
- Spatial reasoning prompts
- Oversampling of rare classes
- MPS (M2 Max GPU) support

Target: >90% accuracy overall, >85% spatial reasoning
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm
import numpy as np
from datetime import datetime

from clip_dataset import create_dataloaders, CLIPConditionalDataset


class CLIPClassifier(nn.Module):
    """CLIP model using contrastive image-text similarity for classification."""
    
    def __init__(self, model_name: str = 'openai/clip-vit-large-patch14', num_classes: int = 9):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.num_classes = num_classes
        
        # Temperature parameter for scaling logits (learnable)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
        """
        Forward pass using CLIP's contrastive learning approach.
        
        Args:
            input_ids: [batch_size, num_classes, seq_len] - text prompts for all classes
            attention_mask: [batch_size, num_classes, seq_len]
            pixel_values: [batch_size, 3, 224, 224] - images
            labels: [batch_size] - ground truth class indices
        """
        batch_size = pixel_values.shape[0]
        
        # Get image embeddings
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        image_embeds = vision_outputs[1]  # [batch_size, embed_dim]
        image_embeds = self.clip.visual_projection(image_embeds)
        
        # Normalize image embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Get text embeddings for all classes
        # input_ids shape: [batch_size, num_classes, seq_len]
        # Reshape to process all texts at once
        input_ids_flat = input_ids.view(batch_size * self.num_classes, -1)
        attention_mask_flat = attention_mask.view(batch_size * self.num_classes, -1)
        
        text_outputs = self.clip.text_model(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        text_embeds = text_outputs[1]  # [batch_size * num_classes, embed_dim]
        text_embeds = self.clip.text_projection(text_embeds)
        
        # Normalize text embeddings
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Reshape back to [batch_size, num_classes, embed_dim]
        text_embeds = text_embeds.view(batch_size, self.num_classes, -1)
        
        # Compute similarity between image and all text prompts
        # image_embeds: [batch_size, embed_dim]
        # text_embeds: [batch_size, num_classes, embed_dim]
        # logits: [batch_size, num_classes]
        logits = torch.einsum('be,bce->bc', image_embeds, text_embeds)
        
        # Scale by temperature
        logits = logits * self.temperature.exp()
        
        loss = None
        if labels is not None:
            # Will compute weighted loss outside
            loss = F.cross_entropy(logits, labels, reduction='none')
        
        return {
            'loss': loss,
            'logits': logits,
            'image_embeds': image_embeds,
        }
    
    def freeze_vision_encoder(self):
        """Freeze vision encoder for faster training."""
        for param in self.clip.vision_model.parameters():
            param.requires_grad = False
        print("✅ Vision encoder frozen")
    
    def unfreeze_vision_encoder(self):
        """Unfreeze vision encoder for full fine-tuning."""
        for param in self.clip.vision_model.parameters():
            param.requires_grad = True
        print("✅ Vision encoder unfrozen")
    
    def freeze_text_encoder(self):
        """Freeze text encoder."""
        for param in self.clip.text_model.parameters():
            param.requires_grad = False
        print("✅ Text encoder frozen")
    
    def freeze_all_except_temperature(self):
        """Freeze everything except temperature parameter - fastest training."""
        for param in self.clip.parameters():
            param.requires_grad = False
        self.temperature.requires_grad = True
        print("✅ All CLIP parameters frozen, only temperature trainable")


class WeightedFocalLoss(nn.Module):
    """Focal loss with class weights for imbalanced datasets."""
    
    def __init__(self, class_weights: Dict[str, float], alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, class_names: List[str]) -> torch.Tensor:
        """
        Args:
            inputs: Logits [batch_size, num_classes]
            targets: Class indices [batch_size]
            class_names: List of class names for weighting [batch_size]
        """
        # Standard cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Focal term
        p_t = torch.exp(-ce_loss)
        focal_term = (1 - p_t) ** self.gamma
        
        # Class weights
        weights = torch.tensor([self.class_weights.get(name, 1.0) for name in class_names], 
                              device=inputs.device, dtype=inputs.dtype)
        
        # Combined loss
        loss = self.alpha * focal_term * ce_loss * weights
        return loss.mean()


def train_epoch(
    model: CLIPClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: WeightedFocalLoss,
    device: str,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        class_names = batch['class_name']
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        
        # Compute weighted focal loss
        loss = criterion(outputs['logits'], labels, class_names)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        _, predicted = outputs['logits'].max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total,
            'lr': scheduler.get_last_lr()[0],
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(
    model: CLIPClassifier,
    data_loader: DataLoader,
    device: str,
    split_name: str = 'valid',
) -> Tuple[float, Dict]:
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    per_class_correct = {name: 0 for name in CLIPConditionalDataset.CLASS_NAMES}
    per_class_total = {name: 0 for name in CLIPConditionalDataset.CLASS_NAMES}
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f"Evaluating {split_name}")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            class_names = batch['class_name']
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
            
            # Predictions
            _, predicted = outputs['logits'].max(1)
            
            # Overall accuracy
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Per-class accuracy
            for i, (pred, label, class_name) in enumerate(zip(predicted, labels, class_names)):
                per_class_total[class_name] += 1
                if pred == label:
                    per_class_correct[class_name] += 1
    
    # Calculate metrics
    accuracy = 100. * correct / total
    
    per_class_accuracy = {}
    for class_name in CLIPConditionalDataset.CLASS_NAMES:
        if per_class_total[class_name] > 0:
            per_class_accuracy[class_name] = 100. * per_class_correct[class_name] / per_class_total[class_name]
        else:
            per_class_accuracy[class_name] = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
        'correct': correct,
        'total': total,
    }
    
    return accuracy, metrics


def main():
    parser = argparse.ArgumentParser(description='Fine-tune CLIP for conditional classification')
    parser.add_argument('--dataset_dir', type=str, default='../../quen2-vl.yolov11',
                       help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='openai/clip-vit-large-patch14',
                       help='CLIP model name')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=100,
                       help='Number of warmup steps')
    parser.add_argument('--output_dir', type=str, default='../models/clip_conditional_final',
                       help='Output directory for model')
    parser.add_argument('--oversample', action='store_true', default=True,
                       help='Oversample rare classes')
    parser.add_argument('--spatial_context', action='store_true', default=True,
                       help='Include spatial context in prompts')
    parser.add_argument('--freeze_vision', action='store_true', default=False,
                       help='Freeze vision encoder (train only classifier)')
    parser.add_argument('--eval_every', type=int, default=100,
                       help='Evaluate every N steps')
    
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
    print("CLIP FINE-TUNING FOR CONDITIONAL CLASSIFICATION")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Oversample: {args.oversample}")
    print(f"Spatial context: {args.spatial_context}")
    print(f"Freeze vision encoder: {args.freeze_vision}")
    print(f"{'='*80}\n")
    
    # Load processor
    print("Loading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, valid_loader, test_loader = create_dataloaders(
        args.dataset_dir,
        processor,
        batch_size=args.batch_size,
        oversample=args.oversample,
        spatial_context=args.spatial_context,
    )
    
    # Create model
    print(f"\nLoading CLIP model: {args.model_name}...")
    model = CLIPClassifier(args.model_name, num_classes=9)
    
    # Freeze encoders if requested
    if args.freeze_vision:
        model.freeze_vision_encoder()
    
    # Always freeze text encoder (we're doing image classification)
    model.freeze_text_encoder()
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Setup weighted focal loss
    criterion = WeightedFocalLoss(
        class_weights=CLIPConditionalDataset.CLASS_WEIGHTS,
        alpha=0.25,
        gamma=2.0,
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # Setup scheduler (cosine with warmup)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
        anneal_strategy='cos',
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")
    
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
            model, train_loader, optimizer, scheduler, criterion, device, epoch
        )
        
        # Evaluate
        val_acc, val_metrics = evaluate(model, valid_loader, device, 'valid')
        
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
        print(f"\n  Per-class accuracy:")
        for class_name, acc in val_metrics['per_class_accuracy'].items():
            print(f"    {class_name:30s}: {acc:6.2f}%")
        
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
            
            # Save CLIP model separately
            model.clip.save_pretrained(output_dir / 'clip_model')
            processor.save_pretrained(output_dir / 'clip_model')
            
            print(f"\n✅ Saved best model (val_acc: {val_acc:.2f}%)")
    
    # Final evaluation on test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}\n")
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    test_acc, test_metrics = evaluate(model, test_loader, device, 'test')
    
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print(f"\nPer-class accuracy:")
    for class_name, acc in test_metrics['per_class_accuracy'].items():
        print(f"  {class_name:30s}: {acc:6.2f}%")
    
    # Save results
    results = {
        'model_name': args.model_name,
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
    
    # Check if target met
    if test_acc >= 90.0:
        print(f"\n🎉 SUCCESS! Target accuracy (>90%) achieved!")
    elif test_acc >= 85.0:
        print(f"\n✅ Good! Close to target (>85%)")
    else:
        print(f"\n⚠️  Below target. Consider:")
        print(f"   - Training for more epochs")
        print(f"   - Unfreezing vision encoder")
        print(f"   - Using ViT-H/14 for higher accuracy")


if __name__ == '__main__':
    main()

