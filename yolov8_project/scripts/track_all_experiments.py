#!/usr/bin/env python3
"""
Track and compare all experiment results.
"""
import csv
import json
from pathlib import Path
from datetime import datetime

EXPERIMENTS = {
    "yolov11_expanded_finetune_lr_low": "1.1 Lower Learning Rate",
    "yolov11_expanded_finetune_aug_reduced": "1.2 Reduced Augmentation",
    "yolov11_baseline_finetune_merged": "2.1 Start from Baseline",
    "yolov11_expanded_finetune_sgd": "2.2 SGD Optimizer",
    "yolov11_expanded_finetune_v52": "Previous Best (v52)",
}

BASELINES = {
    "YOLOv8-S": {"mAP50": 0.7617, "mAP50_95": 0.5153},
    "YOLOv11-S": {"mAP50": 0.7593, "mAP50_95": 0.5111},
}

def get_best_metrics(results_file):
    """Extract best metrics from results.csv"""
    if not Path(results_file).exists():
        return None
    
    best_map50 = 0
    best_map50_95 = 0
    best_epoch = 0
    best_precision = 0
    best_recall = 0
    
    with open(results_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                map50 = float(row['metrics/mAP50(B)'])
                map50_95 = float(row['metrics/mAP50-95(B)'])
                precision = float(row['metrics/precision(B)'])
                recall = float(row['metrics/recall(B)'])
                epoch = int(row['epoch'])
                
                if map50 > best_map50:
                    best_map50 = map50
                    best_map50_95 = map50_95
                    best_precision = precision
                    best_recall = recall
                    best_epoch = epoch
            except (ValueError, KeyError):
                continue
    
    if best_epoch == 0:
        return None
    
    return {
        "epoch": best_epoch,
        "mAP50": best_map50,
        "mAP50_95": best_map50_95,
        "precision": best_precision,
        "recall": best_recall,
    }

def main():
    """Track all experiments and generate comparison"""
    results = {}
    
    print("=" * 80)
    print("Experiment Results Tracker")
    print("=" * 80)
    print()
    
    for exp_name, exp_desc in EXPERIMENTS.items():
        results_file = f"runs/detect/{exp_name}/results.csv"
        metrics = get_best_metrics(results_file)
        
        if metrics:
            results[exp_name] = {
                "description": exp_desc,
                **metrics
            }
            print(f"✅ {exp_desc}")
            print(f"   Epoch: {metrics['epoch']}")
            print(f"   mAP@0.5: {metrics['mAP50']*100:.2f}%")
            print(f"   mAP@[0.5:0.95]: {metrics['mAP50_95']*100:.2f}%")
            print()
        else:
            print(f"⏳ {exp_desc}: No results yet")
            print()
    
    # Compare against baselines
    print("=" * 80)
    print("Comparison with Baselines")
    print("=" * 80)
    print()
    
    # Create comparison table
    print(f"{'Model':<40} {'mAP@0.5':<12} {'mAP@[0.5:0.95]':<15} {'Status':<20}")
    print("-" * 80)
    
    for baseline_name, baseline_metrics in BASELINES.items():
        print(f"{baseline_name:<40} {baseline_metrics['mAP50']*100:>10.2f}%  {baseline_metrics['mAP50_95']*100:>13.2f}%  {'Baseline':<20}")
    
    print()
    
    for exp_name, exp_data in results.items():
        map50 = exp_data['mAP50']
        map50_95 = exp_data['mAP50_95']
        
        # Check if surpasses baselines
        surpasses_yolov8 = map50 > BASELINES["YOLOv8-S"]["mAP50"] and map50_95 > BASELINES["YOLOv8-S"]["mAP50_95"]
        surpasses_yolov11 = map50 > BASELINES["YOLOv11-S"]["mAP50"] and map50_95 > BASELINES["YOLOv11-S"]["mAP50_95"]
        
        if surpasses_yolov8:
            status = "✅ Surpasses Both!"
        elif surpasses_yolov11:
            status = "✅ Surpasses YOLOv11-S"
        else:
            status = "❌ Below Baselines"
        
        print(f"{exp_data['description']:<40} {map50*100:>10.2f}%  {map50_95*100:>13.2f}%  {status:<20}")
    
    # Save results to JSON
    output_file = "experiments/results_summary.json"
    Path("experiments").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "baselines": BASELINES,
            "experiments": results
        }, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()

