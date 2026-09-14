"""
Evaluation metrics, benchmarking, and visualization plotting modules.
"""

from src.evaluation.evaluator import evaluate_transcription, evaluate_dataset
from src.evaluation.plotter import (
    plot_training_history,
    plot_transcription_comparison,
    plot_spectrogram_and_notes,
)

__all__ = [
    "evaluate_transcription",
    "evaluate_dataset",
    "plot_training_history",
    "plot_transcription_comparison",
    "plot_spectrogram_and_notes",
]
