"""
CLI Entry point for model training.
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.data_downloader import prepare_maestro_dataset
from src.dataset.maestro_dataset import MaestroDataset
from src.training.trainer import train_model
from src.utils.config import get_device, load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CRNN Piano Note Recognition & Transcription Model"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to custom config.yaml"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override initial learning rate"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Compute device ('cuda', 'cpu', 'mps', 'auto')"
    )
    parser.add_argument(
        "--dataset-root", type=str, default=None, help="Path to MAESTRO dataset root"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Run quick 3-epoch test with small subset"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides.setdefault("training", {})["learning_rate"] = args.lr
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.dataset_root is not None:
        overrides.setdefault("dataset", {})["dataset_root"] = args.dataset_root

    if args.quick_test:
        overrides.setdefault("training", {})["epochs"] = 3
        overrides.setdefault("training", {})["batch_size"] = 4

    config = load_config(args.config, overrides=overrides)
    device = get_device(config.training.device, verbose=True)

    dataset_root = Path(config.dataset.dataset_root)
    if not dataset_root.exists() or not list(dataset_root.rglob("*.wav")):
        print(f"[Dataset] No audio files found in {dataset_root}. Preparing sample subset...")
        prepare_maestro_dataset(dataset_root, mode="sample", num_samples=4)

    model, history = train_model(config, device=device)

    try:
        from src.evaluation.plotter import plot_training_history
        plot_path = Path(config.paths.output_dir) / "training_curves.png"
        plot_training_history(history, save_path=plot_path)
        print(f"[Plot] Saved training curve plots to {plot_path}")
    except Exception as e:
        print(f"Notice: Could not generate training plot: {e}")

    print("\n[Done] Training completed successfully.")


if __name__ == "__main__":
    main()
