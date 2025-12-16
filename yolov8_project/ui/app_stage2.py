#!/usr/bin/env python3
"""
Enhanced Streamlit web application with Stage 1 + Stage 2 pipeline.
Stage 1: YOLO object detection
Stage 2: CLIP conditional classification (NORMAL vs CONDITIONAL)
"""

import streamlit as st
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
import time

# Add parent directory to path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import modules - try both absolute and relative imports
try:
    from ui.inference import load_model, run_inference, get_device as get_stage1_device
except ImportError:
    # Fallback: import directly from current directory
    from inference import load_model, run_inference, get_device as get_stage1_device

try:
    from ui.stage2_inference import (
        load_stage2_model,
        run_stage2_inference,
        format_detection_label,
        get_condition_color,
        get_device as get_stage2_device
    )
except ImportError:
    # Fallback: import directly from current directory
    from stage2_inference import (
        load_stage2_model,
        run_stage2_inference,
        format_detection_label,
        get_condition_color,
        get_device as get_stage2_device
    )


# Page configuration
st.set_page_config(
    page_title="2-Stage Object Detection & Classification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stage-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .normal-badge {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .conditional-badge {
        background-color: #ffa500;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def cached_load_stage1_model(model_type: str):
    """Cached Stage 1 (YOLO) model loading."""
    try:
        base_path = Path(__file__).parent.parent
        model, model_path = load_model(model_type, base_path=base_path)
        return model, model_path, None
    except Exception as e:
        return None, None, str(e)


@st.cache_resource
def cached_load_stage2_model(model_type: str):
    """Cached Stage 2 (CLIP) model loading."""
    if model_type == "None":
        return None, None, None, None
    
    try:
        # Base path should be parent directory (yolov8_project) when running from there
        # If running from ui directory, use parent; otherwise use current
        current_dir = Path(__file__).parent
        if current_dir.name == "ui":
            base_path = current_dir.parent  # yolov8_project
        else:
            base_path = current_dir
        model, processor, model_path = load_stage2_model(model_type, base_path=base_path)
        return model, processor, model_path, None
    except Exception as e:
        return None, None, None, str(e)


def draw_detections(
    image: Image.Image,
    detections: list,
    include_stage2: bool = False
) -> Image.Image:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: PIL Image
        detections: List of detection dicts
        include_stage2: Whether to include Stage 2 condition labels
    
    Returns:
        Image with drawn detections
    """
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    for det in detections:
        # Handle both dict and list bbox formats
        bbox = det['bbox']
        if isinstance(bbox, dict):
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
        else:
            x1, y1, x2, y2 = map(int, bbox)
        
        # Get color based on condition
        if include_stage2 and 'condition' in det:
            color = get_condition_color(det['condition'])
        else:
            color = (0, 255, 0)  # Default green
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Prepare label
        label = format_detection_label(det, include_stage2)
        conf = det['confidence']
        label_text = f"{label} {conf:.2f}"
        
        # Draw label background
        bbox_label = draw.textbbox((x1, y1 - 25), label_text, font=font)
        draw.rectangle(bbox_label, fill=color)
        
        # Draw label text
        draw.text((x1, y1 - 25), label_text, fill=(255, 255, 255), font=font)
    
    return draw_image


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 2-Stage Detection & Classification</h1>', unsafe_allow_html=True)
    st.markdown("**Stage 1**: Object Detection (YOLO) | **Stage 2**: Conditional Classification (CLIP)")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Stage 1 Model Selection
        st.markdown('<p class="stage-header">Stage 1: Object Detection</p>', unsafe_allow_html=True)
        stage1_model = st.selectbox(
            "Select YOLO Model:",
            ["YOLOv11-Best", "YOLOv8", "YOLOv11"],
            help="Stage 1 detects objects (rock_toe, slope_drain, toe_drain, vegetation)",
            index=0
        )
        
        # Stage 2 Model Selection
        st.markdown('<p class="stage-header">Stage 2: Classification</p>', unsafe_allow_html=True)
        stage2_model = st.selectbox(
            "Select Classifier:",
            ["CLIP-B32-Binary", "None"],
            help="Stage 2 classifies objects as NORMAL or CONDITIONAL",
            index=0
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for Stage 1 detections"
        )
        
        st.markdown("---")
        
        # Model Information
        st.markdown("### 📊 Model Status")
        
        # Load Stage 1
        stage1, stage1_path, stage1_error = cached_load_stage1_model(stage1_model)
        
        if stage1_error:
            st.error(f"Stage 1: ❌ {stage1_error}")
        else:
            st.success(f"Stage 1: ✅ {stage1_model} loaded")
            st.caption(f"Device: {get_stage1_device()}")
        
        # Load Stage 2
        if stage2_model != "None":
            stage2, processor, stage2_path, stage2_error = cached_load_stage2_model(stage2_model)
            
            if stage2_error:
                st.error(f"Stage 2: ❌ {stage2_error}")
                st.info("💡 Train the model first:\n```bash\ncd stage2_conditional/scripts\npython3 train_binary_clip.py\n```")
            else:
                st.success(f"Stage 2: ✅ {stage2_model} loaded")
                st.caption(f"Device: {get_stage2_device()}")
                st.caption(f"Accuracy: 80.47% test, 86.54% val")
        else:
            st.info("Stage 2: Not selected (Stage 1 only)")
            stage2, processor = None, None
    
    # Initialize session state for tracking analysis
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    if 'result_image' not in st.session_state:
        st.session_state.result_image = None
    
    # Main content
    uploaded_file = st.file_uploader(
        "📤 Choose an image to analyze...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for analysis"
    )
    
    # Reset analysis state if new image uploaded
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.get('last_uploaded_file', None):
            st.session_state.analysis_complete = False
            st.session_state.detections = []
            st.session_state.result_image = None
            st.session_state.last_uploaded_file = uploaded_file.name
    
    # Show uploaded image only if analysis not complete
    if uploaded_file is not None and not st.session_state.analysis_complete:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.subheader("ℹ️ Ready to Analyze")
            st.markdown("""
            **Stage 1: Object Detection (YOLO)**
            - Detects infrastructure objects in images
            - Classes: rock_toe, slope_drain, toe_drain, vegetation
            - Best model: YOLOv11-Best (82.3% mAP)
            
            **Stage 2: Conditional Classification (CLIP)**
            - Classifies detected objects as NORMAL or CONDITIONAL
            - Uses frozen CLIP ViT-B/32 + spatial reasoning
            - Accuracy: 80.47% test, 86.54% validation
            - Speed: ~100ms per object
            
            **NORMAL** 🟢: Object is clearly visible and unobstructed
            
            **CONDITIONAL** 🟠: Object is blocked, damaged, uneven, or not clearly visible
            """)
    
    # Run inference button
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if stage1 is None:
                st.error("❌ Stage 1 model not loaded. Cannot proceed.")
                return
            
            with st.spinner("Running 2-stage pipeline..."):
                # Stage 1: Object Detection
                st.markdown("### 🔍 Stage 1: Object Detection")
                stage1_start = time.time()
                
                results = run_inference(
                    stage1,
                    image,
                    conf_threshold=conf_threshold,
                    device=get_stage1_device()
                )
                
                stage1_time = time.time() - stage1_start
                detections = results['detections']
                
                st.success(f"✅ Stage 1 complete: {len(detections)} objects detected in {stage1_time:.2f}s")
                
                # Stage 2: Conditional Classification
                if stage2 is not None and len(detections) > 0:
                    st.markdown("### 🎯 Stage 2: Conditional Classification")
                    stage2_start = time.time()
                    
                    detections = run_stage2_inference(
                        stage2,
                        processor,
                        image,
                        detections,
                        device=get_stage2_device()
                    )
                    
                    stage2_time = time.time() - stage2_start
                    st.success(f"✅ Stage 2 complete: Classifications added in {stage2_time:.2f}s")
                    
                    # Count conditions
                    normal_count = sum(1 for d in detections if d.get('condition') == 'NORMAL')
                    conditional_count = sum(1 for d in detections if d.get('condition') == 'CONDITIONAL')
                    
                    # Display metrics
                    metric_cols = st.columns(3)
                    with metric_cols[0]:
                        st.metric("🟢 NORMAL", normal_count)
                    with metric_cols[1]:
                        st.metric("🟠 CONDITIONAL", conditional_count)
                    with metric_cols[2]:
                        st.metric("⏱️ Total Time", f"{stage1_time + stage2_time:.2f}s")
                else:
                    stage2_time = 0
                
                # Draw detections and store in session state
                result_image = draw_detections(
                    image,
                    detections,
                    include_stage2=(stage2 is not None)
                )
                
                # Store results in session state
                st.session_state.analysis_complete = True
                st.session_state.detections = detections
                st.session_state.result_image = result_image
                st.session_state.stage2_enabled = (stage2 is not None)
                
                # Force rerun to show results
                st.rerun()
    
    # Display results after analysis (full-width large image)
    if st.session_state.analysis_complete and st.session_state.result_image is not None:
        st.markdown("---")
        st.markdown("### 📊 Detection Results")
        
        # Display large detection results image (full width)
        st.image(
            st.session_state.result_image,
            caption="Detection Results with Bounding Boxes and Labels",
            use_container_width=True
        )
        
        # Detection details
        st.markdown("### 📋 Detection Details")
        
        if st.session_state.detections:
            for i, det in enumerate(st.session_state.detections, 1):
                with st.expander(f"Detection {i}: {format_detection_label(det, st.session_state.stage2_enabled)}"):
                    cols = st.columns(2)
                    
                    with cols[0]:
                        # Handle both 'class' and 'class_name' keys
                        class_name = det.get('class_name', det.get('class', 'unknown'))
                        st.write(f"**Object Type:** {class_name}")
                        st.write(f"**Detection Confidence:** {det['confidence']:.3f}")
                        
                        # Handle both dict and list bbox formats
                        bbox = det['bbox']
                        if isinstance(bbox, dict):
                            bbox_list = [bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']]
                        else:
                            bbox_list = bbox
                        st.write(f"**Bounding Box:** [{bbox_list[0]:.0f}, {bbox_list[1]:.0f}, {bbox_list[2]:.0f}, {bbox_list[3]:.0f}]")
                    
                    with cols[1]:
                        if 'condition' in det:
                            condition = det['condition']
                            conf = det.get('condition_confidence', 0)
                            
                            if condition == 'NORMAL':
                                st.markdown(f'<span class="normal-badge">NORMAL ({conf:.2%})</span>', unsafe_allow_html=True)
                                st.write("**Status:** Object is in good condition")
                            elif condition == 'CONDITIONAL':
                                st.markdown(f'<span class="conditional-badge">CONDITIONAL ({conf:.2%})</span>', unsafe_allow_html=True)
                                st.write("**Status:** Object may be blocked, damaged, uneven, or not clearly visible")
                            else:
                                st.write(f"**Condition:** {condition}")
        else:
            st.info("No objects detected. Try lowering the confidence threshold.")
    
    elif uploaded_file is None:
        st.info("👆 Upload an image to get started")
        
        # Example info
        col1, col2 = st.columns(2)
        with col2:
            st.subheader("ℹ️ How It Works")
            st.markdown("""
            **Stage 1: Object Detection (YOLO)**
            - Detects infrastructure objects in images
            - Classes: rock_toe, slope_drain, toe_drain, vegetation
            - Best model: YOLOv11-Best (82.3% mAP)
            
            **Stage 2: Conditional Classification (CLIP)**
            - Classifies detected objects as NORMAL or CONDITIONAL
            - Uses frozen CLIP ViT-B/32 + spatial reasoning
            - Accuracy: 80.47% test, 86.54% validation
            - Speed: ~100ms per object
            
            **NORMAL** 🟢: Object is clearly visible and unobstructed
            
            **CONDITIONAL** 🟠: Object is blocked, damaged, uneven, or not clearly visible
            """)


if __name__ == "__main__":
    main()

