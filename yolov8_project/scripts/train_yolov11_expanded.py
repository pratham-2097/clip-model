#!/usr/bin/env python3
"""
Train YOLOv11 model on expanded merged dataset for maximum accuracy.
Uses advanced training strategies: two-phase fine-tuning with extended epochs,
learning rate scheduling, and optimized hyperparameters.
"""

from pathlib import Path
from ultralytics import YOLO
import time

def main():
    # Paths - using merged dataset
    data_yaml = Path("dataset_merged/data.yaml").resolve()
    project_dir = Path("runs/detect").resolve()
    
    print("="*80)
    print("YOLOv11 TRAINING - EXPANDED DATASET (MAXIMUM ACCURACY)")
    print("="*80)
    print(f"📊 Data config: {data_yaml}")
    print(f"📁 Project directory: {project_dir}")
    print(f"🎯 Goal: Maximum object detection accuracy and confidence")
    print("="*80 + "\n")
    
    # Check if data.yaml exists
    if not data_yaml.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        print(f"💡 Make sure you've run the consolidation script first")
        return
    
    # Verify dataset exists
    train_images = Path("dataset_merged/train/images")
    if not train_images.exists():
        print(f"❌ Error: Training images not found at {train_images}")
        return
    
    train_count = len(list(train_images.glob("*.jpg")))
    val_count = len(list(Path("dataset_merged/val/images").glob("*.jpg"))) if Path("dataset_merged/val/images").exists() else 0
    print(f"📊 Dataset: {train_count} training images, {val_count} validation images\n")
    
    # Phase 1: Freeze backbone training
    print("\n" + "="*80)
    print("PHASE 1: FREEZE BACKBONE TRAINING")
    print("="*80)
    print("Training detection heads with frozen backbone (10 layers)...")
    print("Focus: Learning feature representations for object detection\n")
    
    # Load YOLOv11-S model
    model = YOLO("yolo11s.pt")
    
    # Phase 1: Freeze first 10 layers, train heads
    start_time = time.time()
    results_phase1 = model.train(
        data=str(data_yaml),
        epochs=20,  # Increased from 15 for better convergence
        imgsz=640,
        batch=8,
        device="mps",
        project=str(project_dir),
        name="yolov11_expanded_freeze",
        freeze=10,  # Freeze first 10 layers
        lr0=0.002,  # Initial learning rate
        lrf=0.01,   # Final learning rate factor (lr0 * lrf)
        optimizer="SGD",
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,  # Warmup for stable training
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=50,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        # Augmentation for better generalization
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,    # Image HSV-Saturation augmentation
        hsv_v=0.4,    # Image HSV-Value augmentation
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1, # Image translation (+/- fraction)
        scale=0.5,    # Image scale (+/- gain)
        shear=0.0,    # Image shear (+/- deg)
        perspective=0.0, # Image perspective (+/- fraction)
        flipud=0.0,   # Image flip up-down (probability)
        fliplr=0.5,   # Image flip left-right (probability)
        mosaic=1.0,   # Image mosaic (probability)
        mixup=0.0,    # Image mixup (probability)
        copy_paste=0.0, # Segment copy-paste (probability)
    )
    phase1_time = time.time() - start_time
    
    print(f"\n✅ Phase 1 completed in {phase1_time/60:.2f} minutes")
    print(f"📁 Phase 1 model saved at: {results_phase1.save_dir}")
    print(f"📊 Phase 1 mAP@0.5: {results_phase1.box.map50:.4f} ({results_phase1.box.map50*100:.2f}%)")
    
    # Phase 2: Full fine-tuning with extended epochs
    print("\n" + "="*80)
    print("PHASE 2: FULL FINE-TUNING (EXTENDED)")
    print("="*80)
    print("Fine-tuning entire model with optimized hyperparameters...")
    print("Extended epochs for maximum accuracy convergence\n")
    
    # Load the best model from phase 1
    phase1_best = Path(results_phase1.save_dir) / "weights" / "best.pt"
    if not phase1_best.exists():
        phase1_best = Path(results_phase1.save_dir) / "weights" / "last.pt"
        print(f"⚠️  Using last.pt instead of best.pt")
    
    model_phase2 = YOLO(str(phase1_best))
    
    start_time = time.time()
    results_phase2 = model_phase2.train(
        data=str(data_yaml),
        epochs=150,  # Extended epochs for maximum accuracy
        imgsz=640,
        batch=8,
        device="mps",
        project=str(project_dir),
        name="yolov11_expanded_finetune",
        freeze=None,  # Unfreeze everything
        lr0=0.0005,   # Lower initial LR for fine-tuning
        lrf=0.01,     # Final LR factor
        optimizer="AdamW",  # AdamW for better convergence
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # Same augmentation as phase 1
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        # Loss weights for better detection
        box=7.5,      # Box loss gain
        cls=0.5,      # Class loss gain
        dfl=1.5,      # DFL loss gain
    )
    phase2_time = time.time() - start_time
    
    total_time = phase1_time + phase2_time
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"⏱️  Phase 1 time: {phase1_time/60:.2f} minutes")
    print(f"⏱️  Phase 2 time: {phase2_time/60:.2f} minutes")
    print(f"⏱️  Total time: {total_time/60:.2f} minutes ({total_time/3600:.2f} hours)")
    print(f"📁 Best model saved at: {results_phase2.save_dir}/weights/best.pt")
    print("="*80)
    
    # Print final metrics
    print("\n📊 FINAL VALIDATION METRICS")
    print("="*80)
    print(f"mAP@0.5:        {results_phase2.box.map50:.4f} ({results_phase2.box.map50*100:.2f}%)")
    print(f"mAP@[0.5:0.95]: {results_phase2.box.map:.4f} ({results_phase2.box.map*100:.2f}%)")
    print(f"Precision:      {results_phase2.box.mp:.4f} ({results_phase2.box.mp*100:.2f}%)")
    print(f"Recall:         {results_phase2.box.mr:.4f} ({results_phase2.box.mr*100:.2f}%)")
    print(f"F1-Score:       {(2 * results_phase2.box.mp * results_phase2.box.mr / (results_phase2.box.mp + results_phase2.box.mr)):.4f}" if (results_phase2.box.mp + results_phase2.box.mr) > 0 else "F1-Score:       N/A")
    
    # Per-class metrics
    print("\n📈 PER-CLASS METRICS")
    print("="*80)
    class_names = ['rock_toe', 'slope_drain', 'toe_drain', 'vegetation']
    print(f"{'Class':<20} {'mAP@0.5':<12} {'mAP@[0.5:0.95]':<15} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        try:
            cls_precision, cls_recall, cls_ap50, cls_ap = results_phase2.box.class_result(i)
            print(
                f"{class_name:<20} "
                f"{cls_ap50:<12.4f} "
                f"{cls_ap:<15.4f} "
                f"{cls_precision:<12.4f} "
                f"{cls_recall:<12.4f}"
            )
        except Exception as e:
            print(f"{class_name:<20} Error: {e}")
    
    print("="*80)
    
    # Performance assessment
    map50 = results_phase2.box.map50
    print("\n🎯 PERFORMANCE ASSESSMENT")
    print("="*80)
    if map50 >= 0.85:
        print("✅ EXCELLENT: Model achieves outstanding detection accuracy!")
    elif map50 >= 0.80:
        print("✅ VERY GOOD: Model achieves high detection accuracy!")
    elif map50 >= 0.75:
        print("✅ GOOD: Model achieves solid detection accuracy.")
    else:
        print("⚠️  MODERATE: Model may benefit from more training or data.")
    print("="*80)

if __name__ == "__main__":
    main()

