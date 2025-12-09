#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

yolo detect train \
  data="$PROJECT_ROOT/data.yaml" \
  model=yolov8s.pt \
  epochs=50 \
  imgsz=640 \
  batch=8 \
  device=mps



