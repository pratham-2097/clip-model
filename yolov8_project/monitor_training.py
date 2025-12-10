#!/usr/bin/env python3
"""
Live Training Monitor for YOLOv11
Shows real-time training progress, metrics, and ETA
"""
import time
import csv
import os
from datetime import datetime, timedelta

RESULTS_FILE = "runs/detect/yolov11_expanded_finetune_v4/results.csv"
TOTAL_EPOCHS = 150

def get_latest_metrics():
    """Read latest metrics from results.csv"""
    if not os.path.exists(RESULTS_FILE):
        return None
    
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        return rows[-1]

def format_metric(value):
    """Format metric as percentage"""
    try:
        return f"{float(value)*100:.2f}%"
    except:
        return "N/A"

def calculate_eta(current_epoch, total_time_seconds, total_epochs):
    """Calculate estimated time remaining"""
    if current_epoch <= 1:
        return "Calculating..."
    
    avg_time_per_epoch = total_time_seconds / current_epoch
    remaining_epochs = total_epochs - current_epoch
    eta_seconds = avg_time_per_epoch * remaining_epochs
    
    eta = timedelta(seconds=int(eta_seconds))
    return str(eta)

def display_status(metrics):
    """Display formatted training status"""
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("=" * 80)
    print("🚀 YOLOv11 Training Monitor - Live Status")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not metrics:
        print("⏳ Waiting for training to start...")
        return
    
    epoch = int(metrics['epoch'])
    total_time = float(metrics['time'])
    eta = calculate_eta(epoch, total_time, TOTAL_EPOCHS)
    
    # Progress bar
    progress = (epoch / TOTAL_EPOCHS) * 100
    bar_length = 50
    filled = int(bar_length * epoch / TOTAL_EPOCHS)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"📊 Progress: Epoch {epoch}/{TOTAL_EPOCHS} ({progress:.1f}%)")
    print(f"[{bar}]")
    print()
    
    # Time information
    elapsed = timedelta(seconds=int(total_time))
    print(f"⏱️  Elapsed: {elapsed}")
    print(f"⏳ ETA: {eta}")
    print()
    
    # Metrics
    print("📈 Validation Metrics:")
    print(f"   mAP@0.5:     {format_metric(metrics['metrics/mAP50(B)'])}")
    print(f"   mAP@[0.5:0.95]: {format_metric(metrics['metrics/mAP50-95(B)'])}")
    print(f"   Precision:   {format_metric(metrics['metrics/precision(B)'])}")
    print(f"   Recall:      {format_metric(metrics['metrics/recall(B)'])}")
    print()
    
    # Losses
    print("📉 Losses:")
    print(f"   Box Loss:    {float(metrics['val/box_loss']):.4f}")
    print(f"   Class Loss:  {float(metrics['val/cls_loss']):.4f}")
    print(f"   DFL Loss:    {float(metrics['val/dfl_loss']):.4f}")
    print()
    
    # Learning rate
    lr = float(metrics['lr/pg0'])
    print(f"📚 Learning Rate: {lr:.6f}")
    print()
    
    # Comparison with baselines
    map50 = float(metrics['metrics/mAP50(B)'])
    map50_95 = float(metrics['metrics/mAP50-95(B)'])
    
    print("🎯 Comparison with Baselines:")
    print(f"   YOLOv8-S:    mAP@0.5=76.2%, mAP@[0.5:0.95]=51.5%")
    print(f"   YOLOv11-S:   mAP@0.5=75.9%, mAP@[0.5:0.95]=51.1%")
    print(f"   Current:     mAP@0.5={format_metric(metrics['metrics/mAP50(B)'])}, mAP@[0.5:0.95]={format_metric(metrics['metrics/mAP50-95(B)'])}")
    
    if map50 > 0.762 and map50_95 > 0.515:
        print("   ✅ SURPASSED BOTH BASELINES!")
    elif map50 > 0.762:
        print("   ⚠️  Surpassed mAP@0.5, but mAP@[0.5:0.95] needs improvement")
    elif map50_95 > 0.515:
        print("   ⚠️  Surpassed mAP@[0.5:0.95], but mAP@0.5 needs improvement")
    else:
        print("   ⏳ Still training to surpass baselines...")
    
    print()
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring (training will continue)")
    print("=" * 80)

def main():
    """Main monitoring loop"""
    print("Starting training monitor...")
    print("Monitoring:", RESULTS_FILE)
    print()
    
    last_epoch = 0
    
    try:
        while True:
            metrics = get_latest_metrics()
            
            if metrics:
                current_epoch = int(metrics['epoch'])
                if current_epoch != last_epoch:
                    display_status(metrics)
                    last_epoch = current_epoch
                    
                    # Check if training is complete
                    if current_epoch >= TOTAL_EPOCHS:
                        print("\n🎉 Training Complete!")
                        break
            else:
                print("⏳ Waiting for training to start...")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped. Training continues in background.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()

