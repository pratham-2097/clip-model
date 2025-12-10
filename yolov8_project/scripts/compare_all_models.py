#!/usr/bin/env python3
"""
Final comparison script - validates all models and creates comparison table.
"""
import subprocess
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

def validate_model(model_path, data_yaml):
    """Run validation on a model and extract metrics"""
    try:
        result = subprocess.run(
            ["yolo", "detect", "val", f"model={model_path}", f"data={data_yaml}", "device=mps"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Parse output for metrics
        output = result.stdout
        metrics = {}
        
        for line in output.split('\n'):
            if 'all' in line and 'Box(P' in line:
                # Parse the metrics line
                parts = line.split()
                if len(parts) >= 6:
                    try:
                        metrics['precision'] = float(parts[3])
                        metrics['recall'] = float(parts[4])
                        metrics['mAP50'] = float(parts[5])
                        metrics['mAP50_95'] = float(parts[6])
                    except (ValueError, IndexError):
                        pass
        
        return metrics
    except Exception as e:
        print(f"Error validating {model_path}: {e}")
        return None

def main():
    """Compare all models"""
    print("=" * 80)
    print("Final Model Comparison")
    print("=" * 80)
    print()
    
    data_yaml = "dataset_merged/data.yaml"
    results = {}
    
    # Validate all models
    for exp_name, exp_desc in EXPERIMENTS.items():
        model_path = f"runs/detect/{exp_name}/weights/best.pt"
        
        if not Path(model_path).exists():
            print(f"⏳ {exp_desc}: Model not ready yet")
            continue
        
        print(f"🔍 Validating {exp_desc}...")
        metrics = validate_model(model_path, data_yaml)
        
        if metrics:
            results[exp_name] = {
                "description": exp_desc,
                **metrics
            }
            print(f"   ✅ mAP@0.5: {metrics['mAP50']*100:.2f}%, mAP@[0.5:0.95]: {metrics['mAP50_95']*100:.2f}%")
        else:
            print(f"   ❌ Validation failed")
        print()
    
    # Create comparison table
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Model':<45} {'mAP@0.5':<12} {'mAP@[0.5:0.95]':<15} {'Status':<20}")
    print("-" * 80)
    
    # Baselines
    for baseline_name, baseline_metrics in BASELINES.items():
        print(f"{baseline_name:<45} {baseline_metrics['mAP50']*100:>10.2f}%  {baseline_metrics['mAP50_95']*100:>13.2f}%  {'Baseline':<20}")
    
    print()
    
    # Experiments
    best_model = None
    best_map50 = 0
    
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
        
        print(f"{exp_data['description']:<45} {map50*100:>10.2f}%  {map50_95*100:>13.2f}%  {status:<20}")
        
        if map50 > best_map50:
            best_map50 = map50
            best_model = exp_name
    
    print()
    print("=" * 80)
    if best_model and results[best_model]['mAP50'] > BASELINES["YOLOv8-S"]["mAP50"]:
        print(f"🏆 WINNER: {results[best_model]['description']}")
        print(f"   Model: runs/detect/{best_model}/weights/best.pt")
        print(f"   mAP@0.5: {results[best_model]['mAP50']*100:.2f}%")
        print(f"   mAP@[0.5:0.95]: {results[best_model]['mAP50_95']*100:.2f}%")
    else:
        print("⚠️  No model surpassed baselines yet")
        if best_model:
            print(f"   Best so far: {results[best_model]['description']}")
    print("=" * 80)
    
    # Save results
    output_file = "experiments/final_comparison.json"
    Path("experiments").mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "baselines": BASELINES,
            "experiments": results,
            "best_model": best_model
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()

