#!/bin/bash
# Quick training status checker

cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
source .venv/bin/activate

RESULTS_FILE="runs/detect/yolov11_expanded_finetune_v5/results.csv"

echo "=========================================="
echo "Training Status Check"
echo "=========================================="
echo ""

# Check if process is running
if pgrep -f "yolo detect train.*yolov11_expanded_finetune_v5" > /dev/null; then
    echo "✅ Training process is RUNNING"
    echo ""
    PROCESS_COUNT=$(pgrep -f "yolo detect train.*yolov11_expanded_finetune_v5" | wc -l | tr -d ' ')
    echo "   Active processes: $PROCESS_COUNT"
else
    echo "❌ Training process is NOT running"
    echo ""
fi

# Check results file
if [ -f "$RESULTS_FILE" ]; then
    echo "✅ Results file exists"
    LAST_EPOCH=$(tail -n 1 "$RESULTS_FILE" | cut -d',' -f1)
    if [ ! -z "$LAST_EPOCH" ] && [ "$LAST_EPOCH" != "epoch" ]; then
        echo "   Current epoch: $LAST_EPOCH / 150"
        
        # Get latest metrics
        tail -n 1 "$RESULTS_FILE" | awk -F',' '{
            printf "   mAP@0.5: %.2f%%\n", $8*100
            printf "   mAP@[0.5:0.95]: %.2f%%\n", $9*100
            printf "   Precision: %.2f%%\n", $6*100
            printf "   Recall: %.2f%%\n", $7*100
        }'
    else
        echo "   ⏳ Training just started, waiting for first epoch..."
    fi
else
    echo "⏳ Results file not created yet (training may be initializing)"
fi

echo ""
echo "=========================================="
echo "To view live progress, run:"
echo "  tail -f training_output.log"
echo "=========================================="

