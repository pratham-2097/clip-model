#!/usr/bin/env python3
"""
Comprehensive Stage 2 Dataset Analysis for Qwen2-VL 7B
Analyzes class distribution, bounding boxes, spatial relationships, and data quality
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

DATASET_DIR = Path(__file__).parent.parent.parent / "quen2-vl.yolov11"
OUTPUT_DIR = Path(__file__).parent.parent / "metadata"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged',
    'rock toe', 'rock toe damaged',
    'slope drain', 'slope drain blocked', 'slope drain damaged',
    'vegetation'
]

def parse_polygon_to_bbox(polygon_coords: List[float]) -> Tuple[float, float, float, float]:
    """Convert polygon coordinates to bounding box (x_center, y_center, width, height)."""
    coords = np.array(polygon_coords).reshape(-1, 2)
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    
    return x_center, y_center, width, height

def analyze_dataset():
    """Comprehensive dataset analysis."""
    
    stats = {
        'class_counts': defaultdict(int),
        'split_counts': defaultdict(lambda: defaultdict(int)),
        'bbox_stats': defaultdict(list),
        'spatial_relationships': [],
        'image_sizes': [],
        'instances_per_image': [],
        'class_co_occurrence': defaultdict(lambda: defaultdict(int)),
        'condition_distribution': defaultdict(int),
        'object_type_distribution': defaultdict(int),
    }
    
    # Analyze each split
    for split in ['train', 'valid', 'test']:
        labels_dir = DATASET_DIR / split / 'labels'
        images_dir = DATASET_DIR / split / 'images'
        
        if not labels_dir.exists():
            print(f"⚠️  {split}/labels directory not found")
            continue
        
        split_instances = 0
        
        for label_file in sorted(labels_dir.glob('*.txt')):
            image_file = images_dir / (label_file.stem + '.jpg')
            if not image_file.exists():
                # Try alternative naming
                image_file = images_dir / label_file.stem.replace('.txt', '.jpg')
            
            # Get image size
            if image_file.exists():
                try:
                    img = Image.open(image_file)
                    stats['image_sizes'].append(img.size)
                except Exception as e:
                    print(f"⚠️  Could not open {image_file}: {e}")
            
            # Parse labels
            objects_in_image = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    
                    try:
                        class_idx = int(parts[0])
                        polygon_coords = [float(x) for x in parts[1:]]
                        
                        if len(polygon_coords) < 6:  # Need at least 3 points
                            continue
                        
                        # Convert to bbox
                        x_center, y_center, width, height = parse_polygon_to_bbox(polygon_coords)
                        
                        stats['class_counts'][class_idx] += 1
                        stats['split_counts'][split][class_idx] += 1
                        stats['bbox_stats'][class_idx].append({
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height,
                            'area': width * height
                        })
                        
                        # Extract condition and object type
                        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
                        
                        if 'Blocked' in class_name or 'blocked' in class_name:
                            stats['condition_distribution']['blocked'] += 1
                        elif 'Damaged' in class_name or 'damaged' in class_name:
                            stats['condition_distribution']['damaged'] += 1
                        else:
                            stats['condition_distribution']['normal'] += 1
                        
                        # Object type
                        if 'Toe drain' in class_name:
                            stats['object_type_distribution']['toe_drain'] += 1
                        elif 'slope drain' in class_name.lower():
                            stats['object_type_distribution']['slope_drain'] += 1
                        elif 'rock toe' in class_name.lower():
                            stats['object_type_distribution']['rock_toe'] += 1
                        elif 'vegetation' in class_name.lower():
                            stats['object_type_distribution']['vegetation'] += 1
                        
                        objects_in_image.append({
                            'class': class_idx,
                            'class_name': class_name,
                            'bbox': (x_center, y_center, width, height)
                        })
                        split_instances += 1
                    except (ValueError, IndexError) as e:
                        print(f"⚠️  Error parsing line in {label_file}: {e}")
                        continue
            
            if objects_in_image:
                stats['instances_per_image'].append(len(objects_in_image))
                
                # Co-occurrence
                classes_in_image = [obj['class'] for obj in objects_in_image]
                for i, class1 in enumerate(classes_in_image):
                    for class2 in classes_in_image[i+1:]:
                        stats['class_co_occurrence'][class1][class2] += 1
                        stats['class_co_occurrence'][class2][class1] += 1
                
                # Spatial relationships
                for i, obj1 in enumerate(objects_in_image):
                    for obj2 in objects_in_image[i+1:]:
                        rel = analyze_spatial_relationship(obj1, obj2)
                        if rel:
                            stats['spatial_relationships'].append(rel)
    
    return stats

def analyze_spatial_relationship(obj1: Dict, obj2: Dict) -> Dict:
    """Analyze spatial relationship between two objects."""
    x1, y1, w1, h1 = obj1['bbox']
    x2, y2, w2, h2 = obj2['bbox']
    
    # Determine relationship
    relationship = {
        'class1': obj1['class_name'],
        'class2': obj2['class_name'],
        'vertical': None,
        'horizontal': None,
        'distance': float(np.sqrt((x1-x2)**2 + (y1-y2)**2)),
    }
    
    # Vertical relationship
    if y1 < y2 - 0.05:  # obj1 is above obj2
        relationship['vertical'] = 'above'
    elif y1 > y2 + 0.05:  # obj1 is below obj2
        relationship['vertical'] = 'below'
    else:
        relationship['vertical'] = 'same_level'
    
    # Horizontal relationship
    if abs(x1 - x2) < 0.1:  # Vertically aligned
        relationship['horizontal'] = 'aligned'
    elif x1 < x2:
        relationship['horizontal'] = 'left'
    else:
        relationship['horizontal'] = 'right'
    
    return relationship

def print_analysis_report(stats: Dict):
    """Print comprehensive analysis report."""
    
    print("="*80)
    print("STAGE 2 DATASET ANALYSIS FOR QWEN2-VL 7B")
    print("="*80)
    
    # 1. Class Distribution
    print("\n1. CLASS DISTRIBUTION")
    print("-"*80)
    total_instances = sum(stats['class_counts'].values())
    
    print(f"\nOverall Distribution (Total: {total_instances} instances):")
    for class_idx in sorted(stats['class_counts'].keys()):
        count = stats['class_counts'][class_idx]
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
        print(f"  [{class_idx:2d}] {class_name:30s}: {count:4d} ({percentage:5.1f}%)")
    
    # Per-split distribution
    print("\nPer-Split Distribution:")
    for split in ['train', 'valid', 'test']:
        if split not in stats['split_counts']:
            continue
        print(f"\n  {split.upper()}:")
        split_total = sum(stats['split_counts'][split].values())
        for class_idx in sorted(stats['split_counts'][split].keys()):
            count = stats['split_counts'][split][class_idx]
            percentage = (count / split_total * 100) if split_total > 0 else 0
            class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
            print(f"    [{class_idx:2d}] {class_name:30s}: {count:4d} ({percentage:5.1f}%)")
    
    # 2. Condition Distribution
    print("\n\n2. CONDITION DISTRIBUTION")
    print("-"*80)
    total_conditions = sum(stats['condition_distribution'].values())
    print(f"\nTotal Conditions: {total_conditions}")
    for condition, count in sorted(stats['condition_distribution'].items()):
        percentage = (count / total_conditions * 100) if total_conditions > 0 else 0
        print(f"  {condition:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # 3. Object Type Distribution
    print("\n\n3. OBJECT TYPE DISTRIBUTION")
    print("-"*80)
    total_objects = sum(stats['object_type_distribution'].values())
    print(f"\nTotal Objects: {total_objects}")
    for obj_type, count in sorted(stats['object_type_distribution'].items()):
        percentage = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"  {obj_type:15s}: {count:4d} ({percentage:5.1f}%)")
    
    # 4. Bounding Box Statistics
    print("\n\n4. BOUNDING BOX STATISTICS")
    print("-"*80)
    
    for class_idx in sorted(stats['bbox_stats'].keys()):
        bboxes = stats['bbox_stats'][class_idx]
        if not bboxes:
            continue
        
        areas = [b['area'] for b in bboxes]
        y_centers = [b['y_center'] for b in bboxes]
        widths = [b['width'] for b in bboxes]
        heights = [b['height'] for b in bboxes]
        
        class_name = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class_{class_idx}"
        print(f"\n  {class_name}:")
        print(f"    Count: {len(bboxes)}")
        print(f"    Area:     mean={np.mean(areas):.4f}, std={np.std(areas):.4f}, "
              f"min={np.min(areas):.4f}, max={np.max(areas):.4f}")
        print(f"    Y-Position: mean={np.mean(y_centers):.4f}, std={np.std(y_centers):.4f}")
        print(f"      → {'Bottom' if np.mean(y_centers) > 0.6 else 'Top' if np.mean(y_centers) < 0.4 else 'Middle'} of image")
        print(f"    Width:    mean={np.mean(widths):.4f}, std={np.std(widths):.4f}")
        print(f"    Height:   mean={np.mean(heights):.4f}, std={np.std(heights):.4f}")
    
    # 5. Spatial Relationships
    print("\n\n5. SPATIAL RELATIONSHIPS")
    print("-"*80)
    
    # Group by class pairs
    relationship_counts = defaultdict(lambda: defaultdict(int))
    for rel in stats['spatial_relationships']:
        pair = tuple(sorted([rel['class1'], rel['class2']]))
        relationship_counts[pair][rel['vertical']] += 1
    
    print("\n  Key Spatial Relationships:")
    for pair in sorted(relationship_counts.keys()):
        rels = relationship_counts[pair]
        total = sum(rels.values())
        if total > 5:  # Only show significant relationships
            print(f"\n    {pair[0]} <-> {pair[1]} (n={total}):")
            for direction, count in sorted(rels.items(), key=lambda x: -x[1]):
                percentage = (count / total * 100) if total > 0 else 0
                print(f"      {direction:15s}: {count:3d} ({percentage:5.1f}%)")
    
    # 6. Co-occurrence Analysis
    print("\n\n6. CLASS CO-OCCURRENCE")
    print("-"*80)
    
    print("\n  Classes that appear together in images:")
    for class1 in sorted(stats['class_co_occurrence'].keys()):
        co_occurrences = stats['class_co_occurrence'][class1]
        if co_occurrences:
            class1_name = CLASS_NAMES[class1] if class1 < len(CLASS_NAMES) else f"Class_{class1}"
            print(f"\n    {class1_name}:")
            for class2, count in sorted(co_occurrences.items(), key=lambda x: -x[1])[:5]:
                class2_name = CLASS_NAMES[class2] if class2 < len(CLASS_NAMES) else f"Class_{class2}"
                print(f"      with {class2_name:30s}: {count:3d} times")
    
    # 7. Image Statistics
    print("\n\n7. IMAGE STATISTICS")
    print("-"*80)
    
    if stats['image_sizes']:
        sizes = np.array(stats['image_sizes'])
        print(f"  Image sizes: {sizes.shape[0]} images analyzed")
        print(f"    Width:  mean={np.mean(sizes[:, 0]):.0f}, std={np.std(sizes[:, 0]):.0f}")
        print(f"    Height: mean={np.mean(sizes[:, 1]):.0f}, std={np.std(sizes[:, 1]):.0f}")
        if np.mean(sizes[:, 0]) == 640 and np.mean(sizes[:, 1]) == 640:
            print("    ✅ Resolution correct: 640×640")
        else:
            print(f"    ⚠️  Expected 640×640, got {np.mean(sizes[:, 0]):.0f}×{np.mean(sizes[:, 1]):.0f}")
    
    if stats['instances_per_image']:
        instances = np.array(stats['instances_per_image'])
        print(f"\n  Instances per image:")
        print(f"    Mean: {np.mean(instances):.2f}")
        print(f"    Std:  {np.std(instances):.2f}")
        print(f"    Min:  {np.min(instances)}")
        print(f"    Max:  {np.max(instances)}")
    
    # 8. Dataset Challenges
    print("\n\n8. DATASET CHALLENGES & RECOMMENDATIONS")
    print("-"*80)
    
    # Class imbalance
    counts = list(stats['class_counts'].values())
    if counts:
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n  Class Imbalance:")
        print(f"    Ratio (max/min): {imbalance_ratio:.2f}x")
        if imbalance_ratio > 3:
            print(f"    ⚠️  SEVERE IMBALANCE - Need class balancing or weighted loss")
        elif imbalance_ratio > 2:
            print(f"    ⚠️  Moderate imbalance - Consider class weights")
        else:
            print(f"    ✅ Relatively balanced")
    
    # Data size
    total_images = sum(len(stats['split_counts'][s]) for s in ['train', 'valid', 'test'])
    avg_per_class = total_instances / len(stats['class_counts']) if stats['class_counts'] else 0
    print(f"\n  Data Size:")
    print(f"    Total images: {total_images}")
    print(f"    Average per class: {avg_per_class:.1f} instances")
    if avg_per_class < 30:
        print(f"    ⚠️  LOW DATA - Qwen2-VL 7B's zero-shot capability is critical")
        print(f"    ✅ Recommendation: Use few-shot learning and LoRA fine-tuning")
    elif avg_per_class < 50:
        print(f"    ⚠️  MODERATE DATA - LoRA fine-tuning recommended")
    else:
        print(f"    ✅ Sufficient data for fine-tuning")
    
    # Spatial reasoning requirements
    print(f"\n  Spatial Reasoning Requirements:")
    print(f"    Total relationships analyzed: {len(stats['spatial_relationships'])}")
    print(f"    ✅ Models need to understand spatial context")
    print(f"    ✅ Qwen2-VL 7B can process full images with all objects for context")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    print("Analyzing Stage 2 dataset for Qwen2-VL 7B...")
    print(f"Dataset directory: {DATASET_DIR}")
    
    if not DATASET_DIR.exists():
        print(f"❌ Error: Dataset directory not found at {DATASET_DIR}")
        print("Please ensure the dataset is exported to 'quen2-vl.yolov11/' in the project root")
        sys.exit(1)
    
    stats = analyze_dataset()
    print_analysis_report(stats)
    
    # Save statistics to JSON
    output_file = OUTPUT_DIR / "dataset_analysis.json"
    
    # Convert to JSON-serializable format
    json_stats = {
        'class_counts': dict(stats['class_counts']),
        'split_counts': {k: dict(v) for k, v in stats['split_counts'].items()},
        'total_instances': sum(stats['class_counts'].values()),
        'condition_distribution': dict(stats['condition_distribution']),
        'object_type_distribution': dict(stats['object_type_distribution']),
        'spatial_relationships_count': len(stats['spatial_relationships']),
        'instances_per_image': {
            'mean': float(np.mean(stats['instances_per_image'])) if stats['instances_per_image'] else 0,
            'std': float(np.std(stats['instances_per_image'])) if stats['instances_per_image'] else 0,
            'min': int(np.min(stats['instances_per_image'])) if stats['instances_per_image'] else 0,
            'max': int(np.max(stats['instances_per_image'])) if stats['instances_per_image'] else 0,
        },
        'image_resolution': {
            'mean_width': float(np.mean([s[0] for s in stats['image_sizes']])) if stats['image_sizes'] else 0,
            'mean_height': float(np.mean([s[1] for s in stats['image_sizes']])) if stats['image_sizes'] else 0,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_stats, f, indent=2)
    
    print(f"\n✅ Analysis saved to {output_file}")


