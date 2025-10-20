#!/usr/bin/env python3
"""
Simple dataset oversampler: duplicate entries matching a target action to increase their frequency.

Usage:
    python oversample_dataset.py \
        --input enhanced_intersection_dataset.json \
        --action "speed up and go faster" \
        --multiplier 10 \
        --output enhanced_intersection_dataset_oversampled.json

This will replicate every entry whose `action` equals the provided action string
so that the minority class appears `multiplier` times as often (i.e. duplicates added = multiplier-1).
"""
import argparse
import json
import random
from pathlib import Path


def oversample(input_path: Path, output_path: Path, action_value: str, multiplier: int, shuffle: bool, seed: int):
    with input_path.open("r") as f:
        data = json.load(f)

    total = len(data)
    matches = [d for d in data if (d.get("action") or "").strip().lower() == action_value.strip().lower()]
    non_matches = [d for d in data if (d.get("action") or "").strip().lower() != action_value.strip().lower()]

    if not matches:
        print(f"No entries found with action='{action_value}'. Nothing to do.")
        return

    print(f"Found {len(matches)} matching entries out of {total} total.")

    # Create duplicates
    duplicates = []
    for _ in range(multiplier - 1):
        # shallow copy is fine because we'll dump to JSON; keep original metadata
        duplicates.extend([dict(d) for d in matches])

    new_data = data + duplicates

    if shuffle:
        random.seed(seed)
        random.shuffle(new_data)

    with output_path.open("w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Wrote oversampled dataset to {output_path} (size: {len(new_data)} ; multiplier: {multiplier})")


def main():
    p = argparse.ArgumentParser(description="Oversample minority-class entries in a dataset JSON")
    p.add_argument("--input", required=True, help="Path to input dataset JSON")
    p.add_argument("--output", required=False, help="Output JSON path", default="enhanced_intersection_dataset_oversampled.json")
    p.add_argument("--action", required=True, help="Action string to oversample (exact match, case-insensitive)")
    p.add_argument("--multiplier", type=int, default=10, help="How many times to replicate the minority class (>=1)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false", help="Do not shuffle output entries")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")

    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if args.multiplier < 1:
        raise SystemExit("multiplier must be >= 1")

    oversample(inp, out, args.action, args.multiplier, args.shuffle, args.seed)


if __name__ == "__main__":
    main()
