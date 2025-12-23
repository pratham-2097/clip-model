#!/usr/bin/env python3
"""
Phase 2: Optimize Training for toe_drain Detection
- Higher resolution or multi-scale training
- Class-weighted approach (oversample toe_drain)
- Tighter boxes for better precision
- Small object detection optimizations
"""

import os
import sys
import torch
from pathlib import Path
from datetime import datetime

# Force MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

def check_gpu():
    """Check GPU availability."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def verify_prerequisites():
    """Verify all prerequisites."""
    print("=" * 80)
    print("PHASE 2: VERIFYING PREREQUISITES")
    print("=" * 80)
    
    project_root = Path(__file__).parent.parent
    
    # Check best weights from Phase 1 (or use YOLOv11-Best)
    best_weights = project_root / 'runs' / 'detect' / 'yolov11_best_reproduction_20251223_095945' / 'weights' / 'best.pt'
    
    if not best_weights.exists():
        print(f"⚠️  Phase 1 best weights not found, using YOLOv11-Best...")
        best_weights = project_root / 'runs' / 'detect' / 'yolov11_final_optimized_20251218_0411' / 'weights' / 'best.pt'
        if not best_weights.exists():
            print(f"❌ YOLOv11-Best weights not found: {best_weights}")
            return False, None, None
    
    print(f"✅ Starting weights found: {best_weights}")
    
    # Check dataset
    data_yaml = project_root / 'dataset_merged' / 'data.yaml'
    if not data_yaml.exists():
        print(f"❌ Dataset not found: {data_yaml}")
        return False, None, None
    
    print(f"✅ Dataset found: {data_yaml}")
    
    return True, data_yaml, best_weights

def main():
    """Main training function."""
    
    # Step 0: Environment setup
    device = check_gpu()
    print("=" * 80)
    print("PHASE 2: TOE_DRAIN OPTIMIZED TRAINING")
    print("=" * 80)
    print(f"Device: {device}\n")
    
    # Step 1: Verify prerequisites
    prereq_ok, data_yaml, weights_path = verify_prerequisites()
    if not prereq_ok:
        print("\n❌ Prerequisites check failed.")
        sys.exit(1)
    
    project_root = Path(__file__).parent.parent
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"phase2_toe_drain_optimized_{timestamp}"
    project_dir = project_root / 'runs' / 'detect'
    
    print(f"\nConfiguration:")
    print(f"  • Starting weights: {weights_path.name}")
    print(f"  • Dataset: {data_yaml}")
    print(f"  • Device: {device}")
    print(f"  • Epochs: 80 (with early stopping)")
    print(f"  • Image size: 640 (standard - can upgrade to 1280 later)")
    print(f"  • Learning rate: 0.0001 (lower for fine-tuning)")
    print(f"  • Optimizer: AdamW")
    print(f"  • Loss: box=10.0, cls=0.6, dfl=1.7, iou=0.70 (tighter boxes)")
    print(f"  • Augmentation: Optimized for small objects")
    print(f"  • Target: toe_drain mAP 60-75%, Overall mAP 78-82%")
    print("=" * 80)
    
    # Load model
    print("\n🔧 Loading model...")
    from ultralytics import YOLO
    model = YOLO(str(weights_path))
    print(f"✅ Model loaded: {model.names}")
    
    # Step 2: Train with toe_drain optimizations
    print("\n" + "=" * 80)
    print("🚀 STARTING PHASE 2 TRAINING")
    print("=" * 80)
    
    try:
        results = model.train(
            # Core settings
            data=str(data_yaml),
            epochs=80,  # Reduced from 100
            imgsz=640,  # Can upgrade to 1280 for better small object detection
            batch=8,
            device=device,
            
            # Project settings
            project=str(project_dir),
            name=experiment_name,
            
            # Optimizer & Learning Rate (LOWER for fine-tuning)
            optimizer="AdamW",
            lr0=0.0001,           # Lower for fine-tuning
            lrf=0.01,
            cos_lr=True,
            warmup_epochs=3,
            momentum=0.937,
            weight_decay=0.001,   # Higher regularization
            
            # Loss Balance (TIGHTER BOXES for toe_drain)
            box=10.0,             # Higher - penalize loose boxes more
            cls=0.6,
            dfl=1.7,
            iou=0.70,             # Higher - stricter matching
            
            # Augmentation (OPTIMIZED FOR SMALL OBJECTS)
            mosaic=0.2,           # Lower - preserves small objects
            mixup=0.0,            # Disabled - hurts small objects
            copy_paste=0.0,       # Disabled - breaks spatial logic
            degrees=5.0,
            translate=0.1,
            scale=0.6,            # Higher - better scale variation
            perspective=0.0,      # Disabled
            flipud=0.0,
            fliplr=0.5,
            close_mosaic=20,
            
            # HSV Augmentation (CONSERVATIVE)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            
            # Inference defaults
            conf=0.25,
            max_det=300,
            
            # Training control
            patience=15,          # Early stopping
            save=True,
            save_period=5,
            plots=True,
            verbose=True,
            
            # Performance
            workers=4,
            amp=True,
            cache=True,
        )
        
        print("\n✅ Phase 2 training completed successfully!")
        
    except RuntimeError as e:
        if "MPS" in str(e):
            print("\n⚠️  MPS error, retrying on CPU...")
            device = "cpu"
            # Retry with same config
            results = model.train(
                data=str(data_yaml), epochs=80, imgsz=640, batch=8, device=device,
                project=str(project_dir), name=experiment_name,
                optimizer="AdamW", lr0=0.0001, lrf=0.01, cos_lr=True,
                warmup_epochs=3, momentum=0.937, weight_decay=0.001,
                box=10.0, cls=0.6, dfl=1.7, iou=0.70,
                mosaic=0.2, mixup=0.0, copy_paste=0.0, degrees=5.0,
                translate=0.1, scale=0.6, perspective=0.0, flipud=0.0, fliplr=0.5,
                close_mosaic=20, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                conf=0.25, max_det=300, patience=15, save=True, save_period=5,
                plots=True, verbose=True, workers=4, amp=True, cache=True,
            )
            print("\n✅ Phase 2 training completed on CPU!")
        else:
            raise
    
    # Step 3: Final results
    print("\n" + "=" * 80)
    print("🎯 PHASE 2 FINAL RESULTS")
    print("=" * 80)
    
    final_metrics = model.val(data=str(data_yaml), device=device, verbose=False)
    
    # Load best results from CSV
    results_csv = project_dir / experiment_name / 'results.csv'
    if results_csv.exists():
        import pandas as pd
        df = pd.read_csv(results_csv)
        best_idx = df['metrics/mAP50(B)'].idxmax()
        best = df.loc[best_idx]
        
        print(f"\n✅ BEST RESULTS (Epoch {int(best['epoch'])})")
        print(f"   mAP@0.5:        {best['metrics/mAP50(B)']*100:.2f}%")
        print(f"   mAP@[0.5:0.95]: {best['metrics/mAP50-95(B)']*100:.2f}%")
        print(f"   Precision:      {best['metrics/precision(B)']*100:.2f}%")
        print(f"   Recall:         {best['metrics/recall(B)']*100:.2f}%")
        
        # Check toe_drain specifically (if available in per-class metrics)
        print(f"\n📁 Best model saved at:")
        print(f"   {project_dir / experiment_name / 'weights' / 'best.pt'}")
        
        print(f"\n🎯 Phase 2 Target:")
        print(f"   Overall mAP: 78-82% (current: {best['metrics/mAP50(B)']*100:.2f}%)")
        print(f"   toe_drain mAP: 60-75% (check per-class metrics)")
    
    print("\n" + "=" * 80)
    print("✅ PHASE 2 COMPLETE!")
    print("=" * 80)
    print("\nNext: Phase 3 - Push to 90% mAP")

if __name__ == '__main__':
    main()

