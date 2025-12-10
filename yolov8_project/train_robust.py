#!/usr/bin/env python3
"""
Robust YOLOv11 Training Script
Handles interruptions and ensures training completes 150 epochs
"""
import subprocess
import sys
import os
import time
import csv
from pathlib import Path

# Training configuration
MODEL_PATH = "runs/detect/yolov11_expanded_finetune/weights/best.pt"
DATA_YAML = "dataset_merged/data.yaml"
OUTPUT_NAME = "yolov11_expanded_finetune_v5"
RESULTS_DIR = f"runs/detect/{OUTPUT_NAME}"
RESULTS_CSV = f"{RESULTS_DIR}/results.csv"

# Training parameters
EPOCHS = 150
BATCH = 8
IMGSZ = 640
DEVICE = "mps"
LR0 = 0.00035
LRF = 0.01
OPTIMIZER = "AdamW"

def get_current_epoch():
    """Get the last completed epoch from results.csv"""
    if not os.path.exists(RESULTS_CSV):
        return 0
    
    try:
        with open(RESULTS_CSV, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_epoch = int(rows[-1]['epoch'])
                return last_epoch
    except Exception as e:
        print(f"Error reading results.csv: {e}")
    
    return 0

def build_train_command(resume=False, resume_from=None):
    """Build the training command"""
    cmd = [
        "yolo", "detect", "train",
        f"model={MODEL_PATH}",
        f"data={DATA_YAML}",
        f"epochs={EPOCHS}",
        f"imgsz={IMGSZ}",
        f"batch={BATCH}",
        f"device={DEVICE}",
        f"lr0={LR0}",
        f"lrf={LRF}",
        f"optimizer={OPTIMIZER}",
        "cos_lr=True",
        "warmup_epochs=3",
        "hsv_h=0.015",
        "hsv_s=0.7",
        "hsv_v=0.4",
        "translate=0.1",
        "scale=0.5",
        "fliplr=0.5",
        "mosaic=0.7",
        "mixup=0.1",
        "close_mosaic=50",
        "patience=30",
        "box=7.5",
        "cls=0.5",
        "dfl=1.5",
        f"name={OUTPUT_NAME}",
        "project=runs/detect"
    ]
    
    if resume and resume_from:
        weights_path = f"{RESULTS_DIR}/weights/epoch{resume_from}.pt"
        if os.path.exists(weights_path):
            cmd.append(f"resume={weights_path}")
            print(f"🔄 Resuming from epoch {resume_from}")
        else:
            # Try last.pt
            last_pt = f"{RESULTS_DIR}/weights/last.pt"
            if os.path.exists(last_pt):
                cmd.append(f"resume={last_pt}")
                print(f"🔄 Resuming from last checkpoint")
    
    return cmd

def check_training_complete():
    """Check if training has reached 150 epochs"""
    current_epoch = get_current_epoch()
    return current_epoch >= EPOCHS

def monitor_training(process):
    """Monitor training process and handle interruptions"""
    print("📊 Training started. Monitoring progress...")
    print(f"📁 Results will be saved to: {RESULTS_DIR}")
    print()
    
    last_epoch = 0
    consecutive_errors = 0
    max_errors = 3
    
    try:
        while True:
            # Check if process is still running
            if process.poll() is not None:
                return_code = process.returncode
                
                if return_code == 0:
                    print("✅ Training completed successfully!")
                    return True
                else:
                    print(f"⚠️  Process exited with code {return_code}")
                    consecutive_errors += 1
                    
                    if consecutive_errors >= max_errors:
                        print("❌ Too many consecutive errors. Stopping.")
                        return False
                    
                    # Check current progress
                    current_epoch = get_current_epoch()
                    if current_epoch >= EPOCHS:
                        print("✅ Training reached 150 epochs!")
                        return True
                    
                    if current_epoch > last_epoch:
                        print(f"📈 Progress: {current_epoch}/{EPOCHS} epochs completed")
                        print("🔄 Restarting training from checkpoint...")
                        return "restart"
            
            # Check progress
            current_epoch = get_current_epoch()
            if current_epoch > last_epoch:
                print(f"📈 Epoch {current_epoch}/{EPOCHS} completed")
                last_epoch = current_epoch
                
                if current_epoch >= EPOCHS:
                    print("✅ Training reached 150 epochs!")
                    return True
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\n⚠️  Monitoring interrupted. Training may continue in background.")
        return "interrupted"

def main():
    """Main training loop with restart capability"""
    print("=" * 80)
    print("🚀 Robust YOLOv11 Training - 150 Epochs")
    print("=" * 80)
    print()
    
    # Check if training already complete
    if check_training_complete():
        print("✅ Training already completed (150 epochs reached)")
        return
    
    # Get current progress
    current_epoch = get_current_epoch()
    if current_epoch > 0:
        print(f"📊 Found existing training: {current_epoch}/{EPOCHS} epochs completed")
        print(f"🔄 Will resume from checkpoint")
    else:
        print("🆕 Starting fresh training")
    
    print()
    
    max_restarts = 10
    restart_count = 0
    
    while restart_count < max_restarts:
        # Build command
        resume = current_epoch > 0
        cmd = build_train_command(resume=resume, resume_from=current_epoch)
        
        print(f"▶️  Starting training (attempt {restart_count + 1})...")
        print(f"Command: {' '.join(cmd)}")
        print()
        
        try:
            # Start training
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Monitor
            result = monitor_training(process)
            
            if result is True:
                print("✅ Training completed successfully!")
                break
            elif result == "restart":
                restart_count += 1
                current_epoch = get_current_epoch()
                print(f"\n🔄 Restarting... (attempt {restart_count + 1}/{max_restarts})")
                time.sleep(5)  # Wait before restart
                continue
            else:
                print("❌ Training failed")
                break
                
        except Exception as e:
            print(f"❌ Error: {e}")
            restart_count += 1
            
            if restart_count < max_restarts:
                current_epoch = get_current_epoch()
                print(f"🔄 Retrying... (attempt {restart_count + 1}/{max_restarts})")
                time.sleep(10)
            else:
                print("❌ Max restart attempts reached")
                break
    
    # Final check
    final_epoch = get_current_epoch()
    if final_epoch >= EPOCHS:
        print()
        print("=" * 80)
        print("🎉 SUCCESS: Training completed 150 epochs!")
        print(f"📁 Final model: {RESULTS_DIR}/weights/best.pt")
        print("=" * 80)
    else:
        print()
        print("=" * 80)
        print(f"⚠️  Training stopped at epoch {final_epoch}/{EPOCHS}")
        print("📁 Checkpoint saved. You can resume manually.")
        print("=" * 80)

if __name__ == "__main__":
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Activate venv and run
    main()

