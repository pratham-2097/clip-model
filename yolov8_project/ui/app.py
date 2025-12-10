#!/usr/bin/env python3
"""
Streamlit web application for YOLO object detection.
Allows users to upload images and run inference with YOLOv8 or YOLOv11 models.
"""

import streamlit as st
from pathlib import Path
from PIL import Image
import sys

# Add parent directory to path to import inference module
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import inference module
from ui.inference import load_model, run_inference, get_device


# Page configuration
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def cached_load_model(model_type: str):
    """
    Cached model loading to avoid reloading models on every interaction.
    """
    try:
        base_path = Path(__file__).parent.parent
        model, model_path = load_model(model_type, base_path=base_path)
        return model, model_path, None
    except FileNotFoundError as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Error loading model: {str(e)}"


def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 YOLO Object Detection Demo</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_type = st.radio(
            "Select Model:",
            ["YOLOv11-Best", "YOLOv8", "YOLOv11"],
            help="Choose a model for object detection. YOLOv11-Best (82.3% mAP) is the recommended model.",
            index=0  # Default to best model
        )
        
        # Confidence threshold
        conf_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence score for detections (0.0 = all detections, 1.0 = only very confident)"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        
        # Load model (cached)
        model, model_path, error = cached_load_model(model_type)
        
        if error:
            st.error(f"❌ {error}")
            st.info("💡 Make sure the model weights are available at the expected path.")
        elif model:
            st.success(f"✅ {model_type} loaded")
            st.caption(f"Path: {model_path}")
            
            # Show model performance info for best model
            if model_type == "YOLOv11-Best":
                st.info("🏆 **Best Model:** 82.3% mAP@0.5, 53.7% mAP@[0.5:0.95]")
            
            # Show device info
            device = get_device()
            device_emoji = "🖥️" if device == "cpu" else "⚡" if device == "mps" else "🎮"
            st.caption(f"{device_emoji} Device: {device.upper()}")
        else:
            st.warning("⚠️ Model not loaded")
        
        st.markdown("---")
        st.markdown("### 📝 About")
        st.info("""
        **Classes Detected:**
        - rock_toe
        - slope_drain
        - toe_drain
        - vegetation
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image to run object detection"
        )
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                # Convert to RGB if necessary (handles RGBA, P, etc.)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Run detection button
                if model and not error:
                    if st.button("🚀 Run Detection", type="primary", use_container_width=True):
                        with st.spinner("Running inference... This may take a few seconds."):
                            try:
                                # Run inference
                                inference_result = run_inference(
                                    model=model,
                                    image=image,
                                    conf_threshold=conf_threshold,
                                )
                                
                                # Store results in session state
                                st.session_state['inference_result'] = inference_result
                                st.session_state['original_image'] = image
                                st.session_state['model_type'] = model_type
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error during inference: {str(e)}")
                                st.exception(e)  # Show full error for debugging
                                st.session_state.pop('inference_result', None)
                else:
                    st.warning("⚠️ Please ensure the model is loaded before running detection.")
            except Exception as e:
                st.error(f"❌ Error loading image: {str(e)}")
                st.info("💡 Please make sure you uploaded a valid image file (JPG, PNG, etc.)")
    
    with col2:
        st.header("📊 Detection Results")
        
        if 'inference_result' in st.session_state and 'original_image' in st.session_state:
            result = st.session_state['inference_result']
            original_image = st.session_state['original_image']
            model_used = st.session_state.get('model_type', 'Unknown')
            
            # Display model info
            st.caption(f"Model: {model_used} | Confidence Threshold: {conf_threshold:.2f}")
            
            # Display annotated image
            st.subheader("Annotated Image")
            try:
                st.image(result['annotated_image'], caption="Detections with Bounding Boxes", use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying annotated image: {str(e)}")
            
            # Summary metrics
            st.subheader("Summary")
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Total Detections", result['total_detections'])
            with col_metric2:
                st.metric("Device Used", result['device_used'].upper())
            with col_metric3:
                num_classes = len(result['class_summary'])
                st.metric("Classes Detected", num_classes)
            
            # Per-object detection details
            if result['detections']:
                st.subheader("📋 Per-Object Detections")
                
                # Create table data
                table_data = []
                for det in result['detections']:
                    table_data.append({
                        "Object #": det['object_id'],
                        "Class": det['class'],
                        "Confidence": f"{det['confidence_pct']:.2f}%",
                        "Bounding Box": f"({det['bbox']['x1']}, {det['bbox']['y1']}) → ({det['bbox']['x2']}, {det['bbox']['y2']})"
                    })
                
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Per-class summary
                st.subheader("📈 Per-Class Summary")
                
                summary_data = []
                for class_name, stats in result['class_summary'].items():
                    summary_data.append({
                        "Class": class_name,
                        "Count": stats['count'],
                        "Avg Confidence": f"{stats['avg_confidence'] * 100:.2f}%",
                        "Min Confidence": f"{stats['min_confidence'] * 100:.2f}%",
                        "Max Confidence": f"{stats['max_confidence'] * 100:.2f}%",
                    })
                
                # Sort by count (descending)
                summary_data.sort(key=lambda x: x['Count'], reverse=True)
                
                st.dataframe(
                    summary_data,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download results button
                st.markdown("---")
                if st.button("🔄 Clear Results", use_container_width=True):
                    st.session_state.pop('inference_result', None)
                    st.session_state.pop('original_image', None)
                    st.session_state.pop('model_type', None)
                    st.rerun()
            else:
                st.warning("⚠️ No detections found. Try lowering the confidence threshold.")
                st.info("💡 The model didn't detect any objects above the confidence threshold. Try:")
                st.info("   • Lowering the confidence threshold in the sidebar")
                st.info("   • Uploading a different image")
                st.info("   • Checking if the image contains the expected objects (rock_toe, slope_drain, toe_drain, vegetation)")
        else:
            st.info("👆 Upload an image and click 'Run Detection' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>YOLO Object Detection System | Infrastructure Component Detection</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

