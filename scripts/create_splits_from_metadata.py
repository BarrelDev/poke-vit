#!/usr/bin/env python3
"""
Create stratified train/val/test splits from a metadata CSV for the pokemon-dataset-1000.

This script reads a CSV with at least columns `label` and `image_path` (relative to the dataset root),
performs a per-class stratified split (honoring min samples per split), and writes out:
 - CSV files: `train.csv`, `val.csv`, `test.csv` (columns: image_path,label)
 - Optionally copies image files into `train/<label>/`, `val/<label>/`, `test/<label>/` under the output directory.

Usage examples:
  python scripts/create_splits_from_metadata.py \
    --metadata pokemon-dataset-1000/metadata.csv \
    --dataset-root pokemon-dataset-1000 \
    --out-dir pokemon-dataset-1000 \
    --copy

The default ratios are 0.8/0.1/0.1. For classes with few images the script ensures at least one
example in train when possible and keeps val/test >=1 when class size allows.
"""
import argparse
import csv
import random
from pathlib import Path
import shutil
from collections import defaultdict


def read_metadata(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # expect at least 'label' and 'image_path'
            if 'label' not in r or 'image_path' not in r:
                raise RuntimeError('metadata.csv must contain label and image_path columns')
            rows.append({'label': r['label'], 'image_path': r['image_path']})
    return rows


def allocate_splits(per_class, train_ratio, val_ratio, test_ratio, seed=42):
    random.seed(seed)
    splits = {'train': [], 'val': [], 'test': []}
    for label, items in per_class.items():
        n = len(items)
        if n == 0:
            continue
        random.shuffle(items)
        # If very small, allocate minimally
        if n < 5:
            # prefer: train = n-2, val=1, test=1 when possible
            if n == 1:
                splits['train'].extend(items)
            elif n == 2:
                splits['train'].append(items[0]); splits['val'].append(items[1])
            else:
                # n==3 or 4
                splits['train'].extend(items[: max(1, n-2)])
                if n >= 2:
                    splits['val'].append(items[-2])
                if n >= 3:
                    splits['test'].append(items[-1])
        else:
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            n_test = n - n_train - n_val
            # adjust if rounding caused zero test
            if n_test <= 0:
                n_test = 1
                if n_train + n_val + n_test > n:
                    # reduce train first
                    if n_train > 1:
                        n_train -= 1
                    else:
                        n_val = max(1, n_val - 1)
            start = 0
            splits['train'].extend(items[start:start + n_train])
            start += n_train
            splits['val'].extend(items[start:start + n_val])
            start += n_val
            splits['test'].extend(items[start:])

    return splits


def write_csv(entries, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        for img, lbl in entries:
            writer.writerow([img, lbl])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default="pokemon-dataset-1000/metadata.csv", help='Path to metadata.csv')
    parser.add_argument('--dataset-root', default='pokemon-dataset-1000', help='Root that image paths in metadata are relative to')
    parser.add_argument('--out-dir', default='pokemon-dataset-1000', help='Directory where train/val/test dirs and CSVs will be created')
    parser.add_argument('--train-ratio', type=float, default=0.64)
    parser.add_argument('--val-ratio', type=float, default=0.16)
    parser.add_argument('--test-ratio', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy', action='store_true', help='Copy image files into train/val/test folders (default: only write CSVs)')
    args = parser.parse_args()

    meta_path = Path(args.metadata)
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(meta_path)
    per_class = defaultdict(list)
    for r in rows:
        per_class[r['label']].append((r['image_path'], r['label']))

    splits = allocate_splits(per_class, args.train_ratio, args.val_ratio, args.test_ratio, seed=args.seed)

    # Write CSVs
    train_csv = out_dir / 'train.csv'
    val_csv = out_dir / 'val.csv'
    test_csv = out_dir / 'test.csv'

    write_csv(splits['train'], train_csv)
    write_csv(splits['val'], val_csv)
    write_csv(splits['test'], test_csv)

    print(f'Wrote CSVs: {train_csv}, {val_csv}, {test_csv}')

    if args.copy:
        for split_name in ('train', 'val', 'test'):
            for img_rel, lbl in splits[split_name]:
                src = dataset_root / img_rel
                if not src.exists():
                    print(f'Warning: source image not found: {src}')
                    continue
                dst_dir = out_dir / split_name / lbl
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / src.name
                shutil.copy2(src, dst)
        print(f'Copied image files into {out_dir}/train, {out_dir}/val, {out_dir}/test')


if __name__ == '__main__':
    main()
