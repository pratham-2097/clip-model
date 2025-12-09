#!/usr/bin/env python3
"""
Quick YOLOv11 Improvement Script
Focus: Improve precision (reduce false positives) and toe_drain detection
Uses MPS acceleration for fastest training
"""

from pathlib import Path
from ultralytics import YOLO
import time
import torch

def main():
    # Check device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("="*80)
    print("QUICK YOLOv11 IMPROVEMENT TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print("Goals:")
    print("  1. Improve precision (reduce false positives)")
    print("  2. Improve toe_drain detection (target: ≥66.72% to match YOLOv8)")
    print("="*80 + "\n")
    
    # Paths
    data_yaml = Path("data.yaml").resolve()
    project_dir = Path("runs/detect").resolve()
    
    if not data_yaml.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        return
    
    # Load existing best YOLOv11 model as starting point
    existing_model = Path("runs/detect/yolov11_finetune_phase/weights/best.pt")
    if existing_model.exists():
        print(f"📦 Loading existing YOLOv11 model: {existing_model}")
        model = YOLO(str(existing_model))
    else:
        print("📦 Starting from pretrained YOLOv11-S")
        model = YOLO("yolo11s.pt")
    
    print("\n" + "="*80)
    print("TRAINING WITH IMPROVED SETTINGS")
    print("="*80)
    print("Strategy:")
    print("  • Higher class weight for toe_drain (cls=0.8)")
    print("  • Copy-paste augmentation (0.3) for minority classes")
    print("  • Label smoothing (0.05) for better generalization")
    print("  • Optimized for precision and toe_drain")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # Single-phase training with optimized settings
    results = model.train(
        data=str(data_yaml),
        epochs=60,  # Quick training
        imgsz=640,
        batch=8,
        device=device,
        project=str(project_dir),
        name="yolov11_improved",
        # Optimizer
        optimizer="AdamW",
        lr0=0.0005,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Loss weights - increase cls weight to improve precision
        cls=0.8,  # Higher weight for classification (reduces false positives)
        box=7.5,
        dfl=1.5,
        # Augmentation - copy-paste helps minority classes
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,  # Helps toe_drain and vegetation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        # Regularization
        label_smoothing=0.05,
        # Other
        patience=50,
        save=True,
        plots=True,
        verbose=True,
    )
    
    train_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"⏱️  Training time: {train_time/60:.2f} minutes")
    print(f"📁 Best model: {results.save_dir}/weights/best.pt")
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    val_results = model.val(
        data=str(data_yaml),
        split="val",
        device=device,
        save_json=True,
        plots=True,
    )
    
    # Print metrics
    print("\n📊 FINAL METRICS")
    print("="*80)
    print(f"mAP@0.5:        {val_results.box.map50:.4f} ({val_results.box.map50*100:.2f}%)")
    print(f"mAP@[0.5:0.95]: {val_results.box.map:.4f} ({val_results.box.map*100:.2f}%)")
    print(f"Precision:      {val_results.box.mp:.4f} ({val_results.box.mp*100:.2f}%)")
    print(f"Recall:         {val_results.box.mr:.4f} ({val_results.box.mr*100:.2f}%)")
    print(f"F1-Score:       {2 * (val_results.box.mp * val_results.box.mr) / (val_results.box.mp + val_results.box.mr) if (val_results.box.mp + val_results.box.mr) > 0 else 0:.4f}")
    
    # Per-class metrics
    print("\n📈 PER-CLASS METRICS")
    print("-" * 80)
    class_names = ['rock_toe', 'slope_drain', 'toe_drain', 'vegetation']
    print(f"{'Class':<20} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        cls_precision, cls_recall, cls_ap50, cls_ap = val_results.box.class_result(i)
        print(f"{class_name:<20} {cls_ap50:<12.4f} {cls_precision:<12.4f} {cls_recall:<12.4f}")
    
    # Compare with targets
    print("\n🎯 IMPROVEMENT CHECK")
    print("="*80)
    toe_drain_precision, toe_drain_recall, toe_drain_ap50, _ = val_results.box.class_result(2)
    
    print(f"Precision: {val_results.box.mp*100:.2f}% (Target: >75% to match YOLOv8)")
    print(f"toe_drain mAP@0.5: {toe_drain_ap50*100:.2f}% (Target: ≥66.72% to match YOLOv8)")
    
    if val_results.box.mp >= 0.75:
        print("✅ Precision target achieved!")
    else:
        print(f"⚠️  Precision still below target (need {0.75 - val_results.box.mp:.2%} more)")
    
    if toe_drain_ap50 >= 0.6672:
        print("✅ toe_drain target achieved!")
    else:
        print(f"⚠️  toe_drain still below target (need {0.6672 - toe_drain_ap50:.2%} more)")
    
    print("="*80)
    print(f"\n✅ Model saved at: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()

