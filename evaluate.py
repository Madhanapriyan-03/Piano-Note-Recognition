"""
CLI Entry point for model evaluation and transcription metrics.
"""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.maestro_dataset import create_dataloaders
from src.evaluation.evaluator import evaluate_dataset
from src.evaluation.plotter import plot_transcription_comparison
from src.models.crnn_transcriber import build_model
from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.label_processor import LabelProcessor
from src.utils.config import get_device, load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CRNN Piano Transcription Model")
    parser.add_argument(
        "--model", type=str, default="models/best_model.pt", help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--split", type=str, default="validation", choices=["train", "validation", "test"], help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Compute device ('cuda', 'cpu', 'mps', 'auto')"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    device = get_device(args.device, verbose=True)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file '{model_path}' does not exist. Train a model first via `python train.py`.")
        sys.exit(1)

    print(f"\n[Evaluation] Loading model from {model_path}...")
    model = build_model(config).to(device)
    import torch
    checkpoint = torch.load(str(model_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)

    model.eval()

    audio_proc = AudioProcessor.from_config(config)
    label_proc = LabelProcessor.from_config(config)
    train_loader, val_loader, test_loader = create_dataloaders(config)

    dataloader = val_loader if args.split == "validation" else (test_loader if args.split == "test" else train_loader)

    print(f"[Evaluation] Running benchmark on {args.split} set ({len(dataloader.dataset)} segments)...")
    results = evaluate_dataset(
        model=model,
        dataloader=dataloader,
        device=device,
        label_processor=label_proc,
        frame_rate=audio_proc.frame_rate,
        onset_threshold=float(config.inference.onset_threshold),
        frame_threshold=float(config.inference.frame_threshold),
        onset_tolerance=float(config.inference.onset_tolerance),
    )

    print("\n=======================================================")
    print(f" TRANSCRIPTION EVALUATION RESULTS ({args.split.upper()} SET)")
    print("=======================================================")
    print(f" Frame-level Precision : {results['frame_precision'] * 100:.2f}%")
    print(f" Frame-level Recall    : {results['frame_recall'] * 100:.2f}%")
    print(f" Frame-level F1-Score  : {results['frame_f1'] * 100:.2f}%")
    print("-------------------------------------------------------")
    print(f" Note Onset Precision  : {results['note_onset_precision'] * 100:.2f}%")
    print(f" Note Onset Recall     : {results['note_onset_recall'] * 100:.2f}%")
    print(f" Note Onset F1-Score   : {results['note_onset_f1'] * 100:.2f}%")
    print(f" Total Evaluated Clips : {results['evaluated_segments']}")
    print("=======================================================\n")

    out_dir = Path(config.paths.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"evaluation_{args.split}.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] Evaluation metrics saved to {res_path}")


if __name__ == "__main__":
    main()
