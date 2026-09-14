"""
Tests for single training step, loss computation, and backpropagation.
"""

import pytest
import torch
import torch.nn as nn

from src.models.crnn_transcriber import CRNNTranscriber


def test_single_training_step():
    device = torch.device("cpu")
    model = CRNNTranscriber(
        in_channels=1,
        n_mels=229,
        num_pitches=88,
        cnn_channels=[16, 32, 64],
        lstm_hidden=64,
        lstm_layers=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_onset = nn.BCEWithLogitsLoss()
    criterion_frame = nn.BCEWithLogitsLoss()

    batch_size = 2
    n_frames = 50
    mels = torch.randn(batch_size, 1, 229, n_frames, device=device)
    target_frames = torch.randint(0, 2, (batch_size, n_frames, 88)).float().to(device)
    target_onsets = torch.randint(0, 2, (batch_size, n_frames, 88)).float().to(device)

    model.train()
    optimizer.zero_grad()

    outputs = model(mels)
    loss_onset = criterion_onset(outputs["onset_logits"], target_onsets)
    loss_frame = criterion_frame(outputs["frame_logits"], target_frames)
    total_loss = loss_onset + loss_frame

    assert not torch.isnan(total_loss)
    assert total_loss.item() > 0

    total_loss.backward()

    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None

    optimizer.step()
