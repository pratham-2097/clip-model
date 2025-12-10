#!/bin/bash
# Start Streamlit server for YOLO Object Detection UI

cd "/Users/prathamprabhu/Desktop/CLIP model/yolov8_project"
source .venv/bin/activate

echo "=========================================="
echo "Starting YOLO Object Detection UI"
echo "=========================================="
echo ""
echo "Server will start at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

python3 -m streamlit run ui/app.py

