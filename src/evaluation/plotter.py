"""
Visualization and plotting utilities for training metrics, piano rolls, and spectrograms.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.midi_utils import NoteEvent, midi_to_note_name


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor("#0F172A")

    for ax in axes.flat:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#E2E8F0")
        ax.yaxis.label.set_color("#E2E8F0")
        ax.title.set_color("#F8FAFC")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")

    # 1. Total Loss
    axes[0, 0].plot(epochs, history.get("train_loss", []), label="Train Loss", color="#38BDF8", linewidth=2)
    axes[0, 0].plot(epochs, history.get("val_loss", []), label="Val Loss", color="#F43F5E", linewidth=2, linestyle="--")
    axes[0, 0].set_title("Total Loss Progression", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # 2. Sub-Losses
    axes[0, 1].plot(epochs, history.get("train_onset_loss", []), label="Train Onset Loss", color="#FBBF24", linewidth=1.8)
    axes[0, 1].plot(epochs, history.get("train_frame_loss", []), label="Train Frame Loss", color="#34D399", linewidth=1.8)
    axes[0, 1].plot(epochs, history.get("val_onset_loss", []), label="Val Onset Loss", color="#F59E0B", linewidth=1.8, linestyle=":")
    axes[0, 1].plot(epochs, history.get("val_frame_loss", []), label="Val Frame Loss", color="#10B981", linewidth=1.8, linestyle=":")
    axes[0, 1].set_title("Multi-Task Component Losses", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("BCE Loss")
    axes[0, 1].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC", fontsize=8)

    # 3. Validation F1 Scores
    axes[1, 0].plot(epochs, history.get("val_f1_frame", []), label="Frame F1", color="#A78BFA", linewidth=2)
    axes[1, 0].plot(epochs, history.get("val_f1_onset", []), label="Onset F1", color="#EC4899", linewidth=2)
    axes[1, 0].set_title("Validation Transcription F1-Score", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score [0-1]")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    # 4. Learning Rate Schedule
    axes[1, 1].plot(epochs, history.get("learning_rate", []), label="Learning Rate", color="#38BDF8", linewidth=2)
    axes[1, 1].set_title("Optimizer Learning Rate Schedule", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

    plt.tight_layout()

    if save_path:
        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    return fig


def plot_transcription_comparison(
    ref_notes: List[NoteEvent],
    est_notes: List[NoteEvent],
    duration: float,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, dpi=150)
    fig.patch.set_facecolor("#0F172A")

    for ax, title, notes, color in [
        (ax1, "Ground Truth MIDI Notes", ref_notes, "#38BDF8"),
        (ax2, "Model Predicted Notes", est_notes, "#F43F5E"),
    ]:
        ax.set_facecolor("#1E293B")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#E2E8F0")
        ax.yaxis.label.set_color("#E2E8F0")
        ax.title.set_color("#F8FAFC")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("MIDI Pitch")
        ax.set_ylim(20, 109)

        for n in notes:
            ax.broken_barh(
                [(n.onset, max(0.05, n.offset - n.onset))],
                (n.pitch - 0.4, 0.8),
                facecolors=color,
                edgecolors="#FFFFFF",
                linewidth=0.5,
                alpha=0.9,
            )

    ax2.set_xlabel("Time (seconds)")
    ax2.set_xlim(0, max(duration, 1.0))

    plt.tight_layout()

    if save_path:
        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    return fig


def plot_spectrogram_and_notes(
    mel_spec: np.ndarray,
    note_events: List[NoteEvent],
    duration: float,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    n_mels, n_frames = mel_spec.shape
    im = ax.imshow(
        mel_spec,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, n_mels],
        cmap="magma",
    )

    ax.tick_params(colors="#94A3B8")
    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    ax.set_title("Input Log-Mel Spectrogram & Detected Note Onsets", fontweight="bold")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel Frequency Bin")

    for n in note_events:
        ax.axvline(x=n.onset, color="#38BDF8", linestyle="--", alpha=0.6, linewidth=1.2)
        ax.text(
            n.onset + 0.02,
            n_mels * 0.85,
            n.note_name,
            color="#FFFFFF",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", alpha=0.7, edgecolor="#38BDF8"),
        )

    plt.colorbar(im, ax=ax, pad=0.02, label="Normalized Log Magnitude")
    plt.tight_layout()

    if save_path:
        out = Path(save_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

    return fig
