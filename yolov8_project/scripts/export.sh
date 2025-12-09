#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS="${1:-$PROJECT_ROOT/runs/detect/train/weights/best.pt}"

yolo export \
  model="$WEIGHTS" \
  format=onnx \
  opset=12



