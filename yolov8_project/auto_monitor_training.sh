#!/bin/bash
# Auto-monitor training with auto-restart on failure

cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
source .venv/bin/activate

RESULTS_FILE="runs/detect/yolov11_expanded_finetune_v5/results.csv"
TARGET_EPOCHS=150
CHECK_INTERVAL=60  # Check every 60 seconds
MAX_RESTARTS=5

echo "=========================================="
echo "Auto-Monitoring Training (150 Epochs)"
echo "=========================================="
echo "Will check every $CHECK_INTERVAL seconds"
echo "Max auto-restarts: $MAX_RESTARTS"
echo "Press Ctrl+C to stop monitoring (training continues)"
echo "=========================================="
echo ""

restart_count=0

while [ $restart_count -lt $MAX_RESTARTS ]; do
    # Check if training is running
    if ! pgrep -f "yolo detect train.*yolov11_expanded_finetune_v5" > /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Training process stopped!"
        
        # Check current epoch
        if [ -f "$RESULTS_FILE" ]; then
            LAST_EPOCH=$(tail -n 1 "$RESULTS_FILE" | cut -d',' -f1)
            if [ ! -z "$LAST_EPOCH" ] && [ "$LAST_EPOCH" != "epoch" ]; then
                echo "   Last completed epoch: $LAST_EPOCH / $TARGET_EPOCHS"
                
                if [ "$LAST_EPOCH" -ge "$TARGET_EPOCHS" ]; then
                    echo "   ✅ Training completed 150 epochs!"
                    break
                fi
                
                echo "   🔄 Attempting to restart... (attempt $((restart_count + 1))/$MAX_RESTARTS)"
                restart_count=$((restart_count + 1))
                
                # Restart training
                yolo detect train \
                    model="runs/detect/yolov11_expanded_finetune/weights/best.pt" \
                    data="dataset_merged/data.yaml" \
                    epochs=150 \
                    imgsz=640 \
                    batch=8 \
                    device=mps \
                    lr0=0.00035 \
                    lrf=0.01 \
                    optimizer=AdamW \
                    cos_lr=True \
                    warmup_epochs=3 \
                    hsv_h=0.015 \
                    hsv_s=0.7 \
                    hsv_v=0.4 \
                    translate=0.1 \
                    scale=0.5 \
                    fliplr=0.5 \
                    mosaic=0.7 \
                    mixup=0.1 \
                    close_mosaic=50 \
                    patience=30 \
                    box=7.5 \
                    cls=0.5 \
                    dfl=1.5 \
                    name=yolov11_expanded_finetune_v5 \
                    project=runs/detect \
                    resume="runs/detect/yolov11_expanded_finetune_v5/weights/last.pt" \
                    >> training_output.log 2>&1 &
                
                sleep 10
            else
                echo "   ⏳ No epochs completed yet, waiting..."
            fi
        else
            echo "   ⏳ Results file not found, waiting..."
        fi
    else
        # Training is running, show progress
        if [ -f "$RESULTS_FILE" ]; then
            LAST_EPOCH=$(tail -n 1 "$RESULTS_FILE" | cut -d',' -f1)
            if [ ! -z "$LAST_EPOCH" ] && [ "$LAST_EPOCH" != "epoch" ]; then
                PROGRESS=$((LAST_EPOCH * 100 / TARGET_EPOCHS))
                echo "[$(date '+%H:%M:%S')] ✅ Running - Epoch $LAST_EPOCH/$TARGET_EPOCHS ($PROGRESS%)"
                
                if [ "$LAST_EPOCH" -ge "$TARGET_EPOCHS" ]; then
                    echo ""
                    echo "=========================================="
                    echo "🎉 SUCCESS: Training completed 150 epochs!"
                    echo "=========================================="
                    break
                fi
            fi
        fi
    fi
    
    sleep $CHECK_INTERVAL
done

if [ $restart_count -ge $MAX_RESTARTS ]; then
    echo ""
    echo "❌ Max restart attempts reached. Please check manually."
fi

