#!/bin/bash
# Run all model improvement experiments in parallel
# This script starts multiple training runs simultaneously

cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
source .venv/bin/activate

echo "=========================================="
echo "Starting All Model Improvement Experiments"
echo "=========================================="
echo ""

# Kill any existing training processes
pkill -f "yolo detect train" 2>/dev/null
sleep 2

# Step 1.1: Lower Learning Rate
echo "🚀 Starting Experiment 1.1: Lower Learning Rate (lr0=0.0002)"
nohup yolo detect train \
    model="runs/detect/yolov11_expanded_finetune/weights/best.pt" \
    data="dataset_merged/data.yaml" \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    device=mps \
    lr0=0.0002 \
    lrf=0.01 \
    optimizer=AdamW \
    cos_lr=True \
    warmup_epochs=3 \
    patience=50 \
    mosaic=0.7 \
    mixup=0.1 \
    close_mosaic=30 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    name=yolov11_expanded_finetune_lr_low \
    project=runs/detect \
    > experiments/lr_low.log 2>&1 &

sleep 5

# Step 1.2: Reduced Augmentation
echo "🚀 Starting Experiment 1.2: Reduced Augmentation"
nohup yolo detect train \
    model="runs/detect/yolov11_expanded_finetune/weights/best.pt" \
    data="dataset_merged/data.yaml" \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    device=mps \
    lr0=0.00035 \
    lrf=0.01 \
    optimizer=AdamW \
    cos_lr=True \
    warmup_epochs=3 \
    patience=50 \
    mosaic=0.3 \
    mixup=0.0 \
    close_mosaic=20 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    name=yolov11_expanded_finetune_aug_reduced \
    project=runs/detect \
    > experiments/aug_reduced.log 2>&1 &

sleep 5

# Step 2.1: Start from YOLOv11-S Baseline
echo "🚀 Starting Experiment 2.1: Start from YOLOv11-S Baseline"
nohup yolo detect train \
    model="runs/detect/yolov11_finetune_phase/weights/best.pt" \
    data="dataset_merged/data.yaml" \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    device=mps \
    lr0=0.00035 \
    lrf=0.01 \
    optimizer=AdamW \
    cos_lr=True \
    warmup_epochs=3 \
    patience=50 \
    mosaic=0.7 \
    mixup=0.1 \
    close_mosaic=30 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    name=yolov11_baseline_finetune_merged \
    project=runs/detect \
    > experiments/baseline_merged.log 2>&1 &

sleep 5

# Step 2.2: SGD Optimizer
echo "🚀 Starting Experiment 2.2: SGD Optimizer"
nohup yolo detect train \
    model="runs/detect/yolov11_expanded_finetune/weights/best.pt" \
    data="dataset_merged/data.yaml" \
    epochs=100 \
    imgsz=640 \
    batch=8 \
    device=mps \
    lr0=0.01 \
    lrf=0.1 \
    optimizer=SGD \
    momentum=0.937 \
    cos_lr=True \
    warmup_epochs=3 \
    patience=50 \
    mosaic=0.7 \
    mixup=0.1 \
    close_mosaic=30 \
    box=7.5 \
    cls=0.5 \
    dfl=1.5 \
    name=yolov11_expanded_finetune_sgd \
    project=runs/detect \
    > experiments/sgd.log 2>&1 &

echo ""
echo "=========================================="
echo "✅ All 4 experiments started!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  ./scripts/monitor_experiments.sh"
echo ""
echo "Check individual logs:"
echo "  tail -f experiments/lr_low.log"
echo "  tail -f experiments/aug_reduced.log"
echo "  tail -f experiments/baseline_merged.log"
echo "  tail -f experiments/sgd.log"

