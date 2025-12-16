#!/bin/bash

#===============================================================================
# Launch Script for 2-Stage Detection & Classification UI
# Stage 1: YOLO Object Detection
# Stage 2: CLIP Conditional Classification
#===============================================================================

echo "======================================================================"
echo "  2-STAGE OBJECT DETECTION & CLASSIFICATION UI"
echo "======================================================================"
echo ""
echo "Stage 1: YOLO Object Detection (rock_toe, slope_drain, toe_drain, vegetation)"
echo "Stage 2: CLIP Binary Classifier (NORMAL vs CONDITIONAL)"
echo ""
echo "======================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Change to parent directory (yolov8_project) so imports work correctly
cd "$SCRIPT_DIR/.."

# Check if Stage 1 model exists (relative to parent directory)
STAGE1_MODEL="runs/detect/yolov11_expanded_finetune_aug_reduced/weights/best.pt"
if [ ! -f "$STAGE1_MODEL" ]; then
    echo "❌ ERROR: Stage 1 model not found at:"
    echo "   $STAGE1_MODEL"
    echo ""
    echo "Please ensure the YOLOv11-Best model is trained and available."
    echo ""
    exit 1
fi

echo "✅ Stage 1 model found: YOLOv11-Best"

# Check if Stage 2 model exists (relative to parent directory)
STAGE2_MODEL="../stage2_conditional/models/clip_binary_fast/best_model.pt"
if [ ! -f "$STAGE2_MODEL" ]; then
    echo "⚠️  WARNING: Stage 2 model not found at:"
    echo "   $STAGE2_MODEL"
    echo ""
    echo "Stage 2 will not be available. To train Stage 2:"
    echo "   cd ../../stage2_conditional/scripts"
    echo "   python3 train_binary_clip.py --epochs 8 --batch_size 32"
    echo ""
    echo "Continuing with Stage 1 only..."
    echo ""
else
    echo "✅ Stage 2 model found: CLIP ViT-B/32 Binary"
fi

# Check Python dependencies
echo ""
echo "Checking dependencies..."

python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Streamlit not found"
    echo ""
    echo "Install with one of these commands:"
    echo "  pip install streamlit"
    echo "  pip3 install streamlit"
    echo "  python3 -m pip install streamlit"
    echo ""
    exit 1
fi

python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: PyTorch not found"
    echo "Install with: pip install torch"
    exit 1
fi

python3 -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Ultralytics not found"
    echo "Install with: pip install ultralytics"
    exit 1
fi

python3 -c "import transformers" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  WARNING: Transformers not found (needed for Stage 2)"
    echo "Install with: pip install transformers"
fi

echo "✅ Dependencies OK"
echo ""

# Launch Streamlit
echo "======================================================================"
echo "  LAUNCHING UI..."
echo "======================================================================"
echo ""
echo "The UI will open in your default browser at: http://localhost:8501"
echo ""
echo "Features:"
echo "  • Select Stage 1 model (YOLOv8, YOLOv11, YOLOv11-Best)"
echo "  • Select Stage 2 model (CLIP-B32-Binary or None)"
echo "  • Upload images for analysis"
echo "  • View detections with condition status"
echo "  • See NORMAL (green) and CONDITIONAL (orange) results"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================================================"
echo ""

# Run Streamlit with the Stage 2 enhanced app
# Try streamlit command first, fallback to python3 -m streamlit
if command -v streamlit &> /dev/null; then
    echo "Using 'streamlit' command..."
    streamlit run ui/app_stage2.py \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false \
        --theme.base light \
        --theme.primaryColor "#1f77b4"
else
    echo "Using 'python3 -m streamlit' (streamlit command not in PATH)..."
    python3 -m streamlit run ui/app_stage2.py \
        --server.port 8501 \
        --server.address localhost \
        --browser.gatherUsageStats false \
        --theme.base light \
        --theme.primaryColor "#1f77b4"
fi

