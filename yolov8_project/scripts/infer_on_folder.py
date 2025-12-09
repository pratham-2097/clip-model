#!/usr/bin/env python3

"""
Batch inference utility for running YOLOv8 detections on a folder of images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on a folder of images.")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/finetune_phase/weights/best.pt",
        help="Path to YOLOv8 weights.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        default="dataset/images/val",
        help="Folder of images to run inference on.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="outputs",
        help="Where to store predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Execution device (e.g. mps, cpu, 0 for first CUDA GPU).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights_path = Path(args.weights).expanduser().resolve()
    input_dir = Path(args.input_folder).expanduser().resolve()
    output_dir = Path(args.output_folder).expanduser()

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found at {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    print(f"Loading model from: {weights_path}")
    print(f"Running inference on: {input_dir}")
    print(f"Output will be saved to: {output_dir}/results")
    
    results = model.predict(
        source=str(input_dir),
        save=True,
        project=str(output_dir),
        name="results",
        device=args.device,
        conf=0.25,  # Confidence threshold
        save_txt=True,  # Save labels
        save_conf=True,  # Save confidence scores
    )
    
    # Print summary
    print("\n" + "="*60)
    print("INFERENCE SUMMARY")
    print("="*60)
    total_detections = 0
    class_counts = {}
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            num_detections = len(boxes)
            total_detections += num_detections
            for box in boxes:
                cls = int(box.cls[0])
                class_name = model.names[cls]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {total_detections}")
    print("\nDetections by class:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count}")
    print(f"\nResults saved to: {output_dir}/results")
    print("="*60)


if __name__ == "__main__":
    main()


