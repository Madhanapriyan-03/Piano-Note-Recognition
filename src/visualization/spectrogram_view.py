"""
Audio waveform and Mel-spectrogram rendering helpers.
"""

from typing import Optional, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(
    audio: np.ndarray,
    sr: int = 16000,
    title: str = "Audio Waveform",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5), dpi=150)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    time_axis = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(time_axis, audio, color="#38BDF8", linewidth=0.8, alpha=0.85)

    ax.tick_params(colors="#94A3B8")
    ax.xaxis.label.set_color("#E2E8F0")
    ax.yaxis.label.set_color("#E2E8F0")
    ax.title.set_color("#F8FAFC")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.grid(True, linestyle="--", alpha=0.2, color="#475569")

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, max(1.0, len(audio) / sr))
    ax.set_ylim(-1.05, 1.05)

    plt.tight_layout()
    return fig


def plot_interactive_spectrogram(
    mel_spec: np.ndarray,
    sr: int = 16000,
    duration: float = 3.0,
    title: str = "Log-Mel Spectrogram",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=150)
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#1E293B")

    n_mels = mel_spec.shape[0]
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

    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel Frequency Bins")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(colors="#94A3B8")
    cbar.set_label("Normalized Log Power", color="#E2E8F0")

    plt.tight_layout()
    return fig
