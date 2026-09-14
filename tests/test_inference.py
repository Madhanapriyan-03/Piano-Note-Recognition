"""
Tests for end-to-end inference transcriber using pretrained PianoTranscription.
"""

from pathlib import Path
import numpy as np
import pytest

from src.inference.transcriber import PianoTranscriber
from src.utils.audio_io import generate_synthesized_piano_tone
from src.utils.config import load_config


def test_end_to_end_transcription_synthetic():
    config = load_config()
    transcriber = PianoTranscriber(checkpoint_path=None, config=config, device="cpu")

    tone = generate_synthesized_piano_tone(midi_pitch=60, duration=1.0, sr=16000)

    results = transcriber.transcribe_audio(tone)

    assert "notes" in results
    assert "note_events" in results
    assert "sequence_chain" in results
    assert "duration" in results
    assert "mel_spectrogram" in results
    assert "audio" in results

    assert abs(results["duration"] - 1.0) < 0.05
    assert isinstance(results["sequence_chain"], str)
    assert isinstance(results["notes"], list)
