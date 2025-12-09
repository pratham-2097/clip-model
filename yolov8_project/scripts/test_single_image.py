#!/usr/bin/env python3
"""
Quick test script for running inference on a single image or folder of new images.
Shows detailed results with confidence scores.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model on new images")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image file",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="test_images",
        help="Path to folder of images to test",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/finetune_phase/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0-1.0)",
    )
    args = parser.parse_args()
    
    # Load model
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        print(f"❌ Error: Model weights not found at {weights_path}")
        return
    
    print(f"📦 Loading model from: {weights_path}")
    model = YOLO(str(weights_path))
    print("✅ Model loaded successfully!\n")
    
    # Determine input source
    if args.image:
        source = Path(args.image).expanduser().resolve()
        if not source.exists():
            print(f"❌ Error: Image not found at {source}")
            return
        print(f"🖼️  Testing on single image: {source.name}\n")
    else:
        source = Path(args.folder).expanduser().resolve()
        if not source.exists():
            print(f"❌ Error: Folder not found at {source}")
            print(f"💡 Tip: Create a 'test_images' folder and add your images there")
            return
        print(f"📁 Testing on folder: {source}\n")
    
    # Run inference
    results = model.predict(
        source=str(source),
        save=True,
        conf=args.conf,
        save_txt=True,
        save_conf=True,
        project="outputs",
        name="test_results",
        device="mps",
    )
    
    # Print detailed results
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    total_detections = 0
    class_counts = {}
    class_confidences = {}
    
    for i, result in enumerate(results):
        image_name = Path(result.path).name
        boxes = result.boxes
        
        if boxes is not None and len(boxes) > 0:
            print(f"\n📸 Image {i+1}: {image_name}")
            print(f"   Detections: {len(boxes)}")
            
            for j, box in enumerate(boxes):
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                print(f"   [{j+1}] {class_name:15s} | Confidence: {conf:.2%} | Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})")
                
                total_detections += 1
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                if class_name not in class_confidences:
                    class_confidences[class_name] = []
                class_confidences[class_name].append(conf)
        else:
            print(f"\n📸 Image {i+1}: {image_name}")
            print("   ⚠️  No detections found")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    
    if class_counts:
        print("\nDetections by class:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            avg_conf = sum(class_confidences[class_name]) / len(class_confidences[class_name])
            print(f"  {class_name:15s}: {count:3d} detections (avg confidence: {avg_conf:.2%})")
    
    print(f"\n✅ Results saved to: outputs/test_results/")
    print(f"   - Annotated images: outputs/test_results/*.jpg")
    print(f"   - Label files: outputs/test_results/labels/*.txt")
    print("="*70)

if __name__ == "__main__":
    main()


