#!/usr/bin/env python3
"""
Script to download and subsample Amazon Reviews 2023 dataset from Hugging Face.
Optimized for local MacBook runs — downloads small subsets first.

Usage:
  python scripts/download_data.py --subset electronics --max_samples 20000
"""

import argparse
from datasets import load_dataset
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 subset")
    parser.add_argument("--subset", type=str, default="electronics",
                        help="Category subset (electronics, books, etc.)")
    parser.add_argument("--max_samples", type=int, default=100_000,
                        help="Max number of examples to keep")
    parser.add_argument("--output_dir", type=str, default="data/raw",
                        help="Where to save the downloaded dataset")
    
    args = parser.parse_args()

    # Capitalize first letter for config name (e.g., electronics → Electronics)
    category = args.subset.capitalize()
    config_name = f"raw_review_{category}"

    print(f"Downloading Amazon Reviews 2023 - {config_name} subset...")
    print(f"Keeping at most {args.max_samples:,} samples")

    # Load the dataset (no split specified = returns DatasetDict with 'full')
    ds_dict = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        name=config_name,
        trust_remote_code=True,
        num_proc=os.cpu_count() or 4
    )

    # Access the only available split: 'full'
    if "full" not in ds_dict:
        raise ValueError("Expected 'full' split not found. Available splits: " + str(list(ds_dict.keys())))

    ds = ds_dict["full"]

    print(f"Loaded 'full' split with {len(ds):,} examples")

    # Subsample if needed (shuffle for randomness)
    if len(ds) > args.max_samples:
        ds = ds.shuffle(seed=42).select(range(args.max_samples))
        print(f"Subsampled to {len(ds):,} examples")

    # Save to disk (Arrow format — efficient)
    output_path = Path(args.output_dir) / f"amazon_reviews_{category.lower()}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    ds.save_to_disk(str(output_path))
    print(f"Saved to: {output_path}")

    # Quick stats & example
    print("\nDataset info:")
    print(ds)
    print("\nFirst example:")
    print(ds[0])

if __name__ == "__main__":
    main()