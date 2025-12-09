#!/usr/bin/env python3

"""
Oversample under-represented classes in the training split by duplicating their
images/labels up to a target occurrence count.

Usage:
    python scripts/duplicate_minority.py --target 20

The script assumes the dataset lives under `dataset/images/train` and
`dataset/labels/train` relative to the project root (matching our data.yaml).
The labels may be YOLO detection or segmentation format; files are copied as-is.

Idempotency:
    Existing duplicate files with the `_dup{n}` suffix are ignored when
    calculating counts, so you can safely re-run the script after deleting
    unwanted duplicates.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_IMG_DIR = PROJECT_ROOT / "dataset/images/train"
TRAIN_LBL_DIR = PROJECT_ROOT / "dataset/labels/train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Duplicate minority-class samples.")
    parser.add_argument(
        "--target",
        type=int,
        default=20,
        help="Minimum desired count per class.",
    )
    return parser.parse_args()


def load_label_classes(label_path: Path) -> set[int]:
    classes: set[int] = set()
    with label_path.open() as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            try:
                cls = int(parts[0])
            except (ValueError, IndexError):
                continue
            classes.add(cls)
    return classes


def main() -> None:
    args = parse_args()

    if not TRAIN_IMG_DIR.exists() or not TRAIN_LBL_DIR.exists():
        raise FileNotFoundError("Expected dataset directories under dataset/images/train and dataset/labels/train.")

    # Build list of base (non-duplicated) samples for counting
    label_files = [
        p for p in TRAIN_LBL_DIR.glob("*.txt") if "_dup" not in p.stem
    ]

    counts: dict[int, int] = {}
    files_by_class: dict[int, list[Path]] = {}
    for lbl_path in label_files:
        classes = load_label_classes(lbl_path)
        for cls in classes:
            counts[cls] = counts.get(cls, 0) + 1
            files_by_class.setdefault(cls, []).append(lbl_path)

    print("Current per-class counts (unique files):")
    for cls, count in sorted(counts.items()):
        print(f"  Class {cls}: {count}")

    for cls, count in sorted(counts.items()):
        need = max(0, args.target - count)
        if need == 0:
            continue
        sources = files_by_class.get(cls, [])
        if not sources:
            continue
        print(f"Duplicating class {cls}: need {need} additional samples.")
        for i in range(need):
            src_lbl = sources[i % len(sources)]
            stem = src_lbl.stem
            src_img = TRAIN_IMG_DIR / f"{stem}.jpg"
            if not src_img.exists():
                # Try alternate common extensions
                for ext in [".png", ".jpeg", ".JPG", ".PNG"]:
                    candidate = TRAIN_IMG_DIR / f"{stem}{ext}"
                    if candidate.exists():
                        src_img = candidate
                        break
            dst_suffix = f"_dup{i}"
            dst_lbl = TRAIN_LBL_DIR / f"{stem}{dst_suffix}.txt"
            dst_img = src_img.with_name(f"{src_img.stem}{dst_suffix}{src_img.suffix}")
            shutil.copy2(src_lbl, dst_lbl)
            shutil.copy2(src_img, dst_img)

    print("Duplication complete.")


if __name__ == "__main__":
    main()



