"""
CLI utility to download, prepare, and verify the MAESTRO dataset or sample audio/MIDI pairs.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.data_downloader import prepare_maestro_dataset
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MAESTRO Piano Dataset")
    parser.add_argument(
        "--dataset-root", type=str, default="data/maestro", help="Dataset directory root"
    )
    parser.add_argument(
        "--mode", type=str, default="sample", choices=["sample", "metadata_only", "full"],
        help="Dataset prep mode: 'sample' (working subset), 'metadata_only', or 'full'"
    )
    parser.add_argument(
        "--num-samples", type=int, default=4, help="Number of sample pieces to generate in sample mode"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"\n=======================================================")
    print(f" MAESTRO Dataset Preparation")
    print(f" Target Directory : {args.dataset_root}")
    print(f" Preparation Mode : {args.mode}")
    print(f"=======================================================\n")

    root = prepare_maestro_dataset(
        dataset_root=args.dataset_root,
        mode=args.mode,
        num_samples=args.num_samples,
        verbose=True,
    )

    print(f"\n[Completed] Dataset successfully prepared in: {root}")


if __name__ == "__main__":
    main()
