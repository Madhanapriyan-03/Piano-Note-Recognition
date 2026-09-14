"""
Tests for audio I/O and synthesis utilities.
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest

from src.utils.audio_io import (
    generate_synthesized_piano_tone,
    load_audio,
    normalize_audio,
    resample_audio,
    save_audio,
)


def test_piano_tone_synthesis():
    sr = 16000
    duration = 1.0
    tone = generate_synthesized_piano_tone(midi_pitch=60, duration=duration, sr=sr)
    assert len(tone) == int(sr * duration)
    assert np.max(np.abs(tone)) <= 1.0
    assert np.max(np.abs(tone)) > 0.1


def test_audio_save_and_load():
    sr = 16000
    tone = generate_synthesized_piano_tone(midi_pitch=64, duration=0.5, sr=sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        save_audio(tmp_path, tone, sr=sr)
        assert tmp_path.exists()

        loaded_audio, loaded_sr = load_audio(tmp_path, target_sr=sr)
        assert loaded_sr == sr
        assert len(loaded_audio) == len(tone)
        corr = np.corrcoef(tone, loaded_audio)[0, 1]
        assert corr > 0.98
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_audio_resampling():
    orig_sr = 16000
    target_sr = 22050
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1.0, orig_sr))

    resampled = resample_audio(audio, orig_sr=orig_sr, target_sr=target_sr)
    expected_len = int(round(len(audio) * target_sr / orig_sr))
    assert abs(len(resampled) - expected_len) <= 2
