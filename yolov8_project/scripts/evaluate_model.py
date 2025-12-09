#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Tests the model on validation/test data and displays detailed metrics including:
- Overall mAP, Precision, Recall
- Per-class metrics
- Confusion matrix
- Visual predictions
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLOv8 model with comprehensive metrics"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/finetune_phase/weights/best.pt",
        help="Path to model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data.yaml",
        help="Path to data.yaml file",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "train"],
        default="val",
        help="Dataset split to evaluate on (val, test, or train)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for mAP calculation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (mps, cpu, or cuda:0)",
    )
    args = parser.parse_args()
    
    # Load model
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        print(f"❌ Error: Model weights not found at {weights_path}")
        return
    
    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        print(f"❌ Error: Data config not found at {data_path}")
        return
    
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"📦 Loading model from: {weights_path}")
    print(f"📊 Data config: {data_path}")
    print(f"🔍 Evaluating on: {args.split} split")
    print(f"⚙️  Confidence threshold: {args.conf}")
    print(f"⚙️  IoU threshold: {args.iou}")
    print("="*80 + "\n")
    
    model = YOLO(str(weights_path))
    
    # Run validation
    print("🔄 Running evaluation...\n")
    results = model.val(
        data=str(data_path),
        split=args.split,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=True,
        save_hybrid=True,
        plots=True,
    )
    
    # Extract metrics
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    # Overall metrics
    print("\n📊 OVERALL METRICS")
    print("-" * 80)
    print(f"  mAP@0.5:        {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
    print(f"  mAP@[0.5:0.95]: {results.box.map:.4f} ({results.box.map*100:.2f}%)")
    print(f"  Precision:      {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
    print(f"  Recall:         {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
    
    # Per-class metrics
    print("\n📈 PER-CLASS METRICS")
    print("-" * 80)
    print(f"{'Class':<20} {'mAP@0.5':<12} {'mAP@[0.5:0.95]':<15} {'Precision':<12} {'Recall':<12}")
    print("-" * 80)
    
    class_names = model.names
    
    for i, class_name in class_names.items():
        # class_result returns (precision, recall, ap50, ap)
        cls_precision, cls_recall, cls_ap50, cls_ap = results.box.class_result(i)
        print(
            f"{class_name:<20} "
            f"{cls_ap50:<12.4f} "
            f"{cls_ap:<15.4f} "
            f"{cls_precision:<12.4f} "
            f"{cls_recall:<12.4f}"
        )
    
    # Performance assessment
    print("\n" + "="*80)
    print("PERFORMANCE ASSESSMENT")
    print("="*80)
    
    map50 = results.box.map50
    map = results.box.map
    precision = results.box.mp
    recall = results.box.mr
    
    print("\n✅ STRENGTHS:")
    if map50 >= 0.7:
        print(f"  • Excellent overall detection (mAP@0.5 = {map50*100:.1f}%)")
    elif map50 >= 0.5:
        print(f"  • Good overall detection (mAP@0.5 = {map50*100:.1f}%)")
    
    if map >= 0.4:
        print(f"  • Good bounding box precision (mAP@[0.5:0.95] = {map*100:.1f}%)")
    
    if precision >= 0.7:
        print(f"  • High precision - few false positives ({precision*100:.1f}%)")
    
    if recall >= 0.7:
        print(f"  • High recall - finding most objects ({recall*100:.1f}%)")
    
    print("\n⚠️  AREAS FOR IMPROVEMENT:")
    if map50 < 0.7:
        print(f"  • Overall detection accuracy could be higher (current: {map50*100:.1f}%)")
    
    if map < 0.4:
        print(f"  • Bounding box precision needs work (current: {map*100:.1f}%)")
        print("    → Consider more training data or tighter annotations")
    
    if precision < 0.6:
        print(f"  • Too many false positives (precision: {precision*100:.1f}%)")
        print("    → Consider increasing confidence threshold or more training data")
    
    if recall < 0.6:
        print(f"  • Missing too many objects (recall: {recall*100:.1f}%)")
        print("    → Consider decreasing confidence threshold or more training data")
    
    gap = map50 - map
    if gap > 0.3:
        print(f"  • Large gap between mAP@0.5 and mAP@[0.5:0.95] ({gap:.2f})")
        print("    → Bounding boxes are not tight enough around objects")
    
    # Class-specific insights
    print("\n📋 CLASS-SPECIFIC INSIGHTS:")
    print("-" * 80)
    for i, class_name in class_names.items():
        _, _, cls_ap50, _ = results.box.class_result(i)
        map50_val = cls_ap50

        if map50_val >= 0.8:
            status = "✅ Excellent"
        elif map50_val >= 0.6:
            status = "✅ Good"
        elif map50_val >= 0.4:
            status = "⚠️  Moderate"
        else:
            status = "❌ Poor"
        
        print(f"  {class_name:<20} {status:<15} (mAP@0.5: {map50_val*100:.1f}%)")
    
    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"📁 Results directory: {Path(results.save_dir)}")
    print(f"  • Confusion matrix: {Path(results.save_dir)}/confusion_matrix.png")
    print(f"  • PR curves: {Path(results.save_dir)}/PR_curve.png")
    print(f"  • Validation predictions: {Path(results.save_dir)}/val_batch*.jpg")
    print(f"  • JSON results: {Path(results.save_dir)}/predictions.json")
    print("="*80)
    
    # Summary for clear images
    print("\n" + "="*80)
    print("EXPECTED PERFORMANCE ON CLEAR IMAGES")
    print("="*80)
    print("\nBased on validation metrics, when provided with clear, well-lit images")
    print("similar to your training data, the model should:")
    print(f"\n  • Detect ~{recall*100:.0f}% of all objects (Recall)")
    print(f"  • Have ~{precision*100:.0f}% of detections be correct (Precision)")
    print(f"  • Achieve ~{map50*100:.0f}% mAP@0.5 (overall detection quality)")
    print(f"  • Have ~{map*100:.0f}% mAP@[0.5:0.95] (bounding box precision)")
    print("\n⚠️  Note: Performance may drop on images with different characteristics")
    print("   (lighting, angles, object sizes) compared to training data.")
    print("="*80)

if __name__ == "__main__":
    main()

