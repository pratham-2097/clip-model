#!/usr/bin/env python3
"""
Train YOLOv11 model on the same dataset for comparison with YOLOv8.
Uses similar training strategy: two-phase (freeze/unfreeze) approach.
"""

from pathlib import Path
from ultralytics import YOLO
import time

def main():
    # Paths
    data_yaml = Path("data.yaml").resolve()
    project_dir = Path("runs/detect").resolve()
    
    print("="*80)
    print("YOLOv11 TRAINING - COMPARISON WITH YOLOv8")
    print("="*80)
    print(f"📊 Data config: {data_yaml}")
    print(f"📁 Project directory: {project_dir}")
    print("="*80 + "\n")
    
    # Check if data.yaml exists
    if not data_yaml.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml}")
        return
    
    # Phase 1: Freeze backbone (similar to YOLOv8 training)
    print("\n" + "="*80)
    print("PHASE 1: FREEZE BACKBONE TRAINING")
    print("="*80)
    print("Training detection heads with frozen backbone (10 layers)...\n")
    
    # Load YOLOv11-S model (Small variant for fair comparison)
    model = YOLO("yolo11s.pt")  # YOLOv11-S pretrained weights
    
    # Phase 1: Freeze first 10 layers
    start_time = time.time()
    results_phase1 = model.train(
        data=str(data_yaml),
        epochs=15,
        imgsz=640,
        batch=8,
        device="mps",
        project=str(project_dir),
        name="yolov11_freeze_phase",
        freeze=10,  # Freeze first 10 layers
        lr0=0.002,  # Same as YOLOv8 Phase A
        optimizer="SGD",  # Same as YOLOv8 Phase A
        patience=50,
        save=True,
        plots=True,
        verbose=True,
    )
    phase1_time = time.time() - start_time
    
    print(f"\n✅ Phase 1 completed in {phase1_time/60:.2f} minutes")
    print(f"📁 Phase 1 model saved at: {results_phase1.save_dir}")
    
    # Phase 2: Full fine-tuning
    print("\n" + "="*80)
    print("PHASE 2: FULL FINE-TUNING")
    print("="*80)
    print("Fine-tuning entire model with lower learning rate...\n")
    
    # Load the best model from phase 1
    phase1_best = Path(results_phase1.save_dir) / "weights" / "best.pt"
    if not phase1_best.exists():
        phase1_best = Path(results_phase1.save_dir) / "weights" / "last.pt"
    
    model_phase2 = YOLO(str(phase1_best))
    
    start_time = time.time()
    results_phase2 = model_phase2.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device="mps",
        project=str(project_dir),
        name="yolov11_finetune_phase",
        freeze=None,  # Unfreeze everything
        lr0=0.0005,  # Same as YOLOv8 Phase B
        optimizer="AdamW",  # Same as YOLOv8 Phase B
        patience=50,
        save=True,
        plots=True,
        verbose=True,
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
    print("="*80)

if __name__ == "__main__":
    main()

