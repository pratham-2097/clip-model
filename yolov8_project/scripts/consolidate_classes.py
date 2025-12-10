#!/usr/bin/env python3
"""
Consolidate 9 conditional classes from Stage 2 dataset to 4 main object detection classes.

Maps:
- 'Toe drain', 'Toe drain- Blocked', 'Toe drain- Damaged' → 'toe_drain'
- 'rock toe', 'rock toe damaged' → 'rock_toe'
- 'slope drain', 'slope drain blocked', 'slope drain damaged' → 'slope_drain'
- 'vegetation' → 'vegetation'
"""

import argparse
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple
import yaml


# Target class names (4 main classes)
TARGET_CLASSES = ['rock_toe', 'slope_drain', 'toe_drain', 'vegetation']


def load_class_mapping(data_yaml_path: Path) -> Dict[int, int]:
    """
    Create mapping from old class indices to new class indices.
    
    Args:
        data_yaml_path: Path to data.yaml file with old class names
    
    Returns:
        Dictionary mapping old_index -> new_index
    """
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    old_class_names = data['names']
    print(f"\n📋 Original classes ({len(old_class_names)}):")
    for i, name in enumerate(old_class_names):
        print(f"   [{i}] {name}")
    
    # Create mapping based on class name matching
    mapping = {}
    for old_idx, old_name in enumerate(old_class_names):
        old_name_lower = old_name.lower().strip()
        
        # Map to target classes based on name matching
        if 'rock toe' in old_name_lower or 'rocktoe' in old_name_lower:
            new_idx = 0  # rock_toe
        elif 'slope drain' in old_name_lower or 'slopedrain' in old_name_lower:
            new_idx = 1  # slope_drain
        elif 'toe drain' in old_name_lower or 'toedrain' in old_name_lower:
            new_idx = 2  # toe_drain
        elif 'vegetation' in old_name_lower:
            new_idx = 3  # vegetation
        else:
            raise ValueError(f"Unknown class name: {old_name}")
        
        mapping[old_idx] = new_idx
        print(f"   [{old_idx}] '{old_name}' → [{new_idx}] '{TARGET_CLASSES[new_idx]}'")
    
    return mapping


def update_label_file(label_path: Path, class_mapping: Dict[int, int], stats: Dict) -> bool:
    """
    Update class indices in a YOLO label file.
    
    Args:
        label_path: Path to label file
        class_mapping: Dictionary mapping old_index -> new_index
        stats: Statistics dictionary to update
    
    Returns:
        True if file was updated, False if empty or error
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return False
        
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if not parts:
                continue
            
            old_class_idx = int(parts[0])
            
            # Update class index
            if old_class_idx in class_mapping:
                new_class_idx = class_mapping[old_class_idx]
                parts[0] = str(new_class_idx)
                updated_lines.append(' '.join(parts) + '\n')
                
                # Update statistics
                stats['old_classes'][old_class_idx] += 1
                stats['new_classes'][new_class_idx] += 1
            else:
                # Keep original if not in mapping (shouldn't happen)
                updated_lines.append(line + '\n')
                print(f"⚠️  Warning: Class index {old_class_idx} not in mapping for {label_path}")
        
        # Write updated content
        with open(label_path, 'w') as f:
            f.writelines(updated_lines)
        
        return True
    
    except Exception as e:
        print(f"❌ Error processing {label_path}: {e}")
        return False


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    class_mapping: Dict[int, int],
    split: str = 'train'
) -> Dict:
    """
    Process all label files in a dataset split.
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output directory for consolidated dataset
        class_mapping: Class index mapping
        split: Dataset split ('train', 'valid', 'test')
    
    Returns:
        Statistics dictionary
    """
    stats = {
        'old_classes': defaultdict(int),
        'new_classes': defaultdict(int),
        'files_processed': 0,
        'files_skipped': 0,
    }
    
    labels_dir = input_dir / split / 'labels'
    images_dir = input_dir / split / 'images'
    
    if not labels_dir.exists():
        print(f"⚠️  {split}/labels directory not found, skipping...")
        return stats
    
    # Create output directories
    output_labels_dir = output_dir / split / 'labels'
    output_images_dir = output_dir / split / 'images'
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all label files
    label_files = list(labels_dir.glob('*.txt'))
    print(f"\n📁 Processing {len(label_files)} label files in {split}/...")
    
    for label_file in label_files:
        # Copy label file to output
        output_label_path = output_labels_dir / label_file.name
        shutil.copy2(label_file, output_label_path)
        
        # Update class indices
        if update_label_file(output_label_path, class_mapping, stats):
            stats['files_processed'] += 1
            
            # Copy corresponding image file
            image_name = label_file.stem + '.jpg'
            image_path = images_dir / image_name
            if image_path.exists():
                output_image_path = output_images_dir / image_name
                shutil.copy2(image_path, output_image_path)
        else:
            stats['files_skipped'] += 1
    
    return stats


def update_data_yaml(output_dir: Path, original_yaml_path: Path):
    """
    Create updated data.yaml with 4 classes.
    
    Args:
        output_dir: Output dataset directory
        original_yaml_path: Original data.yaml path
    """
    output_yaml_path = output_dir / 'data.yaml'
    
    # Read original to get path structure
    with open(original_yaml_path, 'r') as f:
        original_data = yaml.safe_load(f)
    
    # Create new data.yaml
    new_data = {
        'train': f'./{output_dir.name}/train/images',
        'val': f'./{output_dir.name}/valid/images',
        'test': f'./{output_dir.name}/test/images',
        'nc': 4,
        'names': TARGET_CLASSES,
    }
    
    # Preserve roboflow info if present
    if 'roboflow' in original_data:
        new_data['roboflow'] = original_data['roboflow']
    
    with open(output_yaml_path, 'w') as f:
        yaml.dump(new_data, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created updated data.yaml at {output_yaml_path}")


def generate_statistics(all_stats: Dict[str, Dict], original_class_names: list):
    """
    Generate and print statistics report.
    
    Args:
        all_stats: Dictionary with stats for each split
        original_class_names: Original class names from data.yaml
    """
    print("\n" + "="*80)
    print("STATISTICS REPORT")
    print("="*80)
    
    # Aggregate statistics
    total_old = defaultdict(int)
    total_new = defaultdict(int)
    total_files = 0
    
    for split, stats in all_stats.items():
        for old_idx, count in stats['old_classes'].items():
            total_old[old_idx] += count
        for new_idx, count in stats['new_classes'].items():
            total_new[new_idx] += count
        total_files += stats['files_processed']
    
    # Original class distribution
    print("\n📊 ORIGINAL CLASS DISTRIBUTION (9 classes):")
    print("-" * 80)
    for old_idx in sorted(total_old.keys()):
        class_name = original_class_names[old_idx] if old_idx < len(original_class_names) else f"Class {old_idx}"
        count = total_old[old_idx]
        print(f"  [{old_idx}] {class_name:30s}: {count:4d} instances")
    
    # Consolidated class distribution
    print("\n📊 CONSOLIDATED CLASS DISTRIBUTION (4 classes):")
    print("-" * 80)
    for new_idx in sorted(total_new.keys()):
        class_name = TARGET_CLASSES[new_idx]
        count = total_new[new_idx]
        print(f"  [{new_idx}] {class_name:30s}: {count:4d} instances")
    
    # Summary
    print("\n📈 SUMMARY:")
    print("-" * 80)
    print(f"  Total label files processed: {total_files}")
    print(f"  Original classes: {len(total_old)}")
    print(f"  Consolidated classes: {len(total_new)}")
    print(f"  Total instances: {sum(total_new.values())}")
    
    print("="*80)


def merge_with_existing(
    consolidated_dir: Path,
    existing_dataset_dir: Path,
    output_merged_dir: Path
):
    """
    Merge consolidated dataset with existing dataset.
    
    Args:
        consolidated_dir: Consolidated Stage 2 dataset
        existing_dataset_dir: Existing yolov8_project dataset
        output_merged_dir: Output directory for merged dataset
    """
    print("\n" + "="*80)
    print("MERGING WITH EXISTING DATASET")
    print("="*80)
    
    # Create merged directory structure
    for split in ['train', 'val', 'test']:
        (output_merged_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_merged_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy existing dataset
    print("\n📁 Copying existing dataset...")
    existing_images = existing_dataset_dir / 'dataset' / 'images'
    existing_labels = existing_dataset_dir / 'dataset' / 'labels'
    
    for split in ['train', 'val', 'test']:
        existing_split_images = existing_images / split
        existing_split_labels = existing_labels / split
        
        if existing_split_images.exists():
            for img_file in existing_split_images.glob('*.jpg'):
                shutil.copy2(img_file, output_merged_dir / split / 'images' / img_file.name)
        
        if existing_split_labels.exists():
            for label_file in existing_split_labels.glob('*.txt'):
                shutil.copy2(label_file, output_merged_dir / split / 'labels' / label_file.name)
    
    # Copy consolidated dataset (handle filename conflicts)
    print("\n📁 Copying consolidated dataset...")
    conflict_count = 0
    
    for split in ['train', 'val', 'test']:
        consolidated_images = consolidated_dir / split / 'images'
        consolidated_labels = consolidated_dir / split / 'labels'
        
        if consolidated_images.exists():
            for img_file in consolidated_images.glob('*.jpg'):
                dest_img = output_merged_dir / split / 'images' / img_file.name
                dest_label = output_merged_dir / split / 'labels' / (img_file.stem + '.txt')
                
                # Check for conflicts
                if dest_img.exists():
                    conflict_count += 1
                    # Rename with prefix
                    new_name = f"stage2_{img_file.name}"
                    dest_img = output_merged_dir / split / 'images' / new_name
                    dest_label = output_merged_dir / split / 'labels' / (new_name.replace('.jpg', '.txt'))
                
                shutil.copy2(img_file, dest_img)
                
                # Copy corresponding label
                label_file = consolidated_labels / (img_file.stem + '.txt')
                if label_file.exists():
                    shutil.copy2(label_file, dest_label)
    
    if conflict_count > 0:
        print(f"⚠️  {conflict_count} filename conflicts resolved with 'stage2_' prefix")
    
    # Create merged data.yaml
    merged_yaml = output_merged_dir / 'data.yaml'
    with open(merged_yaml, 'w') as f:
        yaml.dump({
            'train': './train/images',
            'val': './val/images',
            'test': './test/images',
            'nc': 4,
            'names': TARGET_CLASSES,
        }, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Merged dataset created at {output_merged_dir}")
    print(f"✅ Merged data.yaml created at {merged_yaml}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate 9 conditional classes to 4 main object detection classes"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to Stage 2 dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./dataset_consolidated',
        help='Output directory for consolidated dataset'
    )
    parser.add_argument(
        '--merge_with_existing',
        action='store_true',
        help='Merge consolidated dataset with existing yolov8_project dataset'
    )
    parser.add_argument(
        '--existing_dataset_dir',
        type=str,
        default='.',
        help='Path to yolov8_project directory (for merging)'
    )
    parser.add_argument(
        '--merged_output_dir',
        type=str,
        default='./dataset_merged',
        help='Output directory for merged dataset'
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    existing_dataset_dir = Path(args.existing_dataset_dir).expanduser().resolve()
    merged_output_dir = Path(args.merged_output_dir).expanduser().resolve()
    
    # Validate input
    data_yaml_path = input_dir / 'data.yaml'
    if not data_yaml_path.exists():
        print(f"❌ Error: data.yaml not found at {data_yaml_path}")
        return
    
    print("="*80)
    print("CLASS CONSOLIDATION SCRIPT")
    print("="*80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load class mapping
    class_mapping = load_class_mapping(data_yaml_path)
    
    # Get original class names for statistics
    with open(data_yaml_path, 'r') as f:
        original_data = yaml.safe_load(f)
    original_class_names = original_data['names']
    
    # Process each split
    all_stats = {}
    for split in ['train', 'valid', 'test']:
        stats = process_dataset(input_dir, output_dir, class_mapping, split)
        all_stats[split] = stats
    
    # Update data.yaml
    update_data_yaml(output_dir, data_yaml_path)
    
    # Generate statistics
    generate_statistics(all_stats, original_class_names)
    
    # Merge with existing dataset if requested
    if args.merge_with_existing:
        merge_with_existing(output_dir, existing_dataset_dir, merged_output_dir)
        
        # Count final merged dataset
        print("\n📊 FINAL MERGED DATASET STATISTICS:")
        print("-" * 80)
        for split in ['train', 'val', 'test']:
            images_dir = merged_output_dir / split / 'images'
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg')))
                label_count = len(list((merged_output_dir / split / 'labels').glob('*.txt')))
                print(f"  {split:6s}: {image_count:4d} images, {label_count:4d} labels")
    
    print("\n✅ Consolidation complete!")
    print(f"✅ Consolidated dataset: {output_dir}")
    if args.merge_with_existing:
        print(f"✅ Merged dataset: {merged_output_dir}")


if __name__ == '__main__':
    main()

