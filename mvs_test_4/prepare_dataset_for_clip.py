#!/usr/bin/env python3
"""Prepare dataset JSONL splits for CLIP finetuning.

Reads: data/frames.jsonl (or a provided file)
Writes: data/train.jsonl, data/val.jsonl, data/test.jsonl

Usage:
  python prepare_dataset_for_clip.py --data-dir ./data --train 0.8 --val 0.1 --test 0.1
"""
import argparse
import json
import random
from pathlib import Path
from typing import List


def read_jsonl(p: Path) -> List[dict]:
    rows = []
    with p.open('r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                # try to skip malformed
                continue
    return rows


def write_jsonl(p: Path, rows: List[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--source', type=str, default=None, help='source JSONL/JSON filename in data-dir (auto-detected if omitted)')
    p.add_argument('--train', type=float, default=0.8)
    p.add_argument('--val', type=float, default=0.1)
    p.add_argument('--test', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--labels', nargs='*', default=['SLOWER','IDLE','FASTER'], help='filter labels to keep (optional)')
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    # auto-detect common filenames if source not provided
    candidates = [
        args.source,
        'frames.jsonl', 'frames.json', 'frame.json', 'data.jsonl', 'dataset.jsonl'
    ]
    src = None
    for c in candidates:
        if not c:
            continue
        pth = data_dir / c
        if pth.exists():
            src = pth
            break
    # if user didn't provide source name, try any json/jsonl file in dir
    if src is None and args.source is None:
        for ext in ('*.jsonl','*.json'):
            files = list(data_dir.glob(ext))
            if files:
                src = files[0]; break

    if src is None:
        raise FileNotFoundError(f"No source JSON/JSONL found in: {data_dir}")

    # read rows: support JSONL (one JSON per line) or a JSON array
    rows = []
    try:
        # try JSONL first
        rows = read_jsonl(src)
        # if file had single JSON array and our read_jsonl returned 1 element that's a list, unwrap
        if len(rows) == 1 and isinstance(rows[0], list):
            rows = rows[0]
    except Exception:
        # fallback: try json.load for array files
        try:
            with src.open('r', encoding='utf-8') as f:
                obj = json.load(f)
            if isinstance(obj, list):
                rows = obj
            else:
                # not a list, attempt to coerce
                rows = [obj]
        except Exception as e:
            raise RuntimeError(f"Failed to read dataset file {src}: {e}")

    print(f"Read {len(rows)} rows from {src}")

    # filter by labels if provided
    if args.labels:
        labels_set = set(args.labels)
        rows = [r for r in rows if r.get('action_id') in labels_set]
        print(f"Kept {len(rows)} rows after filtering labels {args.labels}")

    random.Random(args.seed).shuffle(rows)

    n = len(rows)
    n_train = int(round(n * args.train))
    n_val = int(round(n * args.val))
    n_test = n - n_train - n_val

    train = rows[:n_train]
    val = rows[n_train:n_train+n_val]
    test = rows[n_train+n_val:]

    write_jsonl(data_dir / 'train.jsonl', train)
    write_jsonl(data_dir / 'val.jsonl', val)
    write_jsonl(data_dir / 'test.jsonl', test)

    print(f"Wrote train={len(train)}, val={len(val)}, test={len(test)} to {data_dir}")


if __name__ == '__main__':
    main()
