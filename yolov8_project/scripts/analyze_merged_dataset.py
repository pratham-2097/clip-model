#!/usr/bin/env python3
"""
Analyze the merged dataset to show final class distribution.
"""

from pathlib import Path
from collections import defaultdict


def analyze_dataset(dataset_dir: Path):
    """Analyze dataset and show class distribution."""
    # Check for both possible structures
    labels_dir = dataset_dir / 'labels'
    if not labels_dir.exists():
        # Try direct structure (train/labels, val/labels, etc.)
        labels_dir = None
    
    class_counts = defaultdict(int)
    total_files = 0
    total_instances = 0
    
    class_names = ['rock_toe', 'slope_drain', 'toe_drain', 'vegetation']
    
    for split in ['train', 'val', 'test']:
        # Try both structures: split/labels or labels/split
        split_labels = dataset_dir / split / 'labels'
        if not split_labels.exists():
            split_labels = dataset_dir / 'labels' / split
        if not split_labels.exists():
            continue
        
        split_class_counts = defaultdict(int)
        split_files = 0
        split_instances = 0
        
        for label_file in split_labels.glob('*.txt'):
            split_files += 1
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if parts:
                        class_idx = int(parts[0])
                        if 0 <= class_idx < len(class_names):
                            split_class_counts[class_idx] += 1
                            split_instances += 1
        
        if split_files > 0:
            print(f"\n{split.upper()} Split:")
            print(f"  Files: {split_files}")
            print(f"  Total instances: {split_instances}")
            print(f"  Per-class distribution:")
            for class_idx in sorted(split_class_counts.keys()):
                count = split_class_counts[class_idx]
                class_counts[class_idx] += count
                print(f"    [{class_idx}] {class_names[class_idx]:20s}: {count:4d} instances")
            total_files += split_files
            total_instances += split_instances
    
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    print(f"Total files: {total_files}")
    print(f"Total instances: {total_instances}")
    print(f"\nOverall class distribution:")
    for class_idx in sorted(class_counts.keys()):
        count = class_counts[class_idx]
        percentage = (count / total_instances * 100) if total_instances > 0 else 0
        print(f"  [{class_idx}] {class_names[class_idx]:20s}: {count:4d} instances ({percentage:5.1f}%)")


if __name__ == '__main__':
    import sys
    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./dataset_merged')
    analyze_dataset(dataset_dir)

