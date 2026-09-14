"""
Tests for audio preprocessing and spectrogram extraction.
"""

import numpy as np
import pytest

from src.preprocessing.audio_processor import AudioProcessor


def test_mel_spectrogram_shape():
    sr = 16000
    hop_length = 512
    n_mels = 229
    duration = 3.0

    processor = AudioProcessor(
        sample_rate=sr,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    audio = np.random.randn(int(sr * duration)).astype(np.float32)
    mel_spec = processor.compute_mel_spectrogram(audio)

    assert mel_spec.shape[0] == n_mels
    expected_frames = int(round(duration * (sr / hop_length)))
    assert abs(mel_spec.shape[1] - expected_frames) <= 2


def test_spectrogram_slicing():
    processor = AudioProcessor()
    mel_spec = np.random.randn(229, 200).astype(np.float32)

    segment_frames = 94
    hop_frames = 47

    slices = processor.slice_spectrogram(mel_spec, segment_frames, hop_frames)
    assert len(slices) >= 3

    for seg, start, end in slices:
        assert seg.shape == (229, segment_frames)
        assert start < end
