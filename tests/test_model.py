"""
Tests for CRNN neural network model forward pass and output dimensions.
"""

import pytest
import torch

from src.models.crnn_transcriber import CRNNTranscriber, build_model
from src.utils.config import load_config


def test_crnn_forward_pass():
    batch_size = 2
    n_mels = 229
    n_frames = 94
    num_pitches = 88

    model = CRNNTranscriber(
        in_channels=1,
        n_mels=n_mels,
        num_pitches=num_pitches,
        cnn_channels=[16, 32, 64],
        lstm_hidden=64,
        lstm_layers=1,
    )

    x = torch.randn(batch_size, 1, n_mels, n_frames)
    outputs = model(x)

    assert "onset_logits" in outputs
    assert "frame_logits" in outputs
    assert "onset_probs" in outputs
    assert "frame_probs" in outputs

    assert outputs["onset_logits"].shape == (batch_size, n_frames, num_pitches)
    assert outputs["frame_logits"].shape == (batch_size, n_frames, num_pitches)
    assert outputs["onset_probs"].shape == (batch_size, n_frames, num_pitches)
    assert outputs["frame_probs"].shape == (batch_size, n_frames, num_pitches)

    assert torch.all(outputs["onset_probs"] >= 0.0) and torch.all(outputs["onset_probs"] <= 1.0)
    assert torch.all(outputs["frame_probs"] >= 0.0) and torch.all(outputs["frame_probs"] <= 1.0)


def test_build_model_from_config():
    cfg = load_config()
    model = build_model(cfg)
    assert isinstance(model, CRNNTranscriber)
    assert model.num_pitches == 88
