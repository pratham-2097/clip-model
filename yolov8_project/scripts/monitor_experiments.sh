#!/bin/bash
# Monitor all running experiments

cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
source .venv/bin/activate

echo "=========================================="
echo "Experiment Status Monitor"
echo "=========================================="
echo ""

experiments=(
    "yolov11_expanded_finetune_lr_low:experiments/lr_low.log"
    "yolov11_expanded_finetune_aug_reduced:experiments/aug_reduced.log"
    "yolov11_baseline_finetune_merged:experiments/baseline_merged.log"
    "yolov11_expanded_finetune_sgd:experiments/sgd.log"
)

for exp_info in "${experiments[@]}"; do
    IFS=':' read -r exp_name log_file <<< "$exp_info"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📊 $exp_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if process is running
    if pgrep -f "yolo detect train.*$exp_name" > /dev/null; then
        echo "✅ Status: RUNNING"
    else
        echo "⏸️  Status: STOPPED"
    fi
    
    # Check results file
    results_file="runs/detect/$exp_name/results.csv"
    if [ -f "$results_file" ]; then
        last_epoch=$(tail -n 1 "$results_file" | cut -d',' -f1)
        if [ ! -z "$last_epoch" ] && [ "$last_epoch" != "epoch" ]; then
            echo "📈 Epoch: $last_epoch/100"
            
            # Get latest metrics
            tail -n 1 "$results_file" | awk -F',' '{
                printf "   mAP@0.5: %.2f%%\n", $8*100
                printf "   mAP@[0.5:0.95]: %.2f%%\n", $9*100
                printf "   Precision: %.2f%%\n", $6*100
                printf "   Recall: %.2f%%\n", $7*100
            }'
        else
            echo "⏳ Initializing..."
        fi
    else
        echo "⏳ No results yet"
    fi
    
    echo ""
done

echo "=========================================="
echo "Baseline Targets:"
echo "  YOLOv8-S: mAP@0.5=76.17%, mAP@[0.5:0.95]=51.53%"
echo "  YOLOv11-S: mAP@0.5=75.93%, mAP@[0.5:0.95]=51.11%"
echo "=========================================="

