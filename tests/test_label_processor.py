"""
Tests for label generation and prediction-to-notes decoding.
"""

import numpy as np
import pytest

from src.preprocessing.label_processor import LabelProcessor
from src.utils.midi_utils import NoteEvent


def test_label_processor_targets():
    processor = LabelProcessor(min_midi=21, max_midi=108)
    frame_rate = 31.25  # 16000 / 512
    total_frames = 100

    notes = [
        NoteEvent(pitch=60, onset=0.5, offset=1.5),
        NoteEvent(pitch=64, onset=0.5, offset=1.5),
    ]

    frame_roll, onset_roll = processor.notes_to_targets(notes, total_frames, frame_rate)

    assert frame_roll.shape == (total_frames, 88)
    assert onset_roll.shape == (total_frames, 88)

    c4_idx = processor.pitch_to_index(60)
    e4_idx = processor.pitch_to_index(64)

    assert frame_roll[20, c4_idx] == 1.0
    assert frame_roll[20, e4_idx] == 1.0
    assert onset_roll[16, c4_idx] >= 0.5
    assert onset_roll[16, e4_idx] >= 0.5


def test_predictions_to_notes():
    processor = LabelProcessor()
    total_frames = 100
    frame_rate = 31.25

    frame_probs = np.zeros((total_frames, 88), dtype=np.float32)
    onset_probs = np.zeros((total_frames, 88), dtype=np.float32)

    c4_idx = processor.pitch_to_index(60)
    onset_probs[10, c4_idx] = 0.95
    frame_probs[10:30, c4_idx] = 0.85

    detected = processor.predictions_to_notes(
        frame_probs,
        onset_probs,
        frame_rate=frame_rate,
        onset_threshold=0.5,
        frame_threshold=0.4,
    )

    assert len(detected) == 1
    note = detected[0]
    assert note.pitch == 60
    assert note.note_name == "C4"
    assert abs(note.onset - (10 / frame_rate)) < 0.05
    assert abs(note.offset - (30 / frame_rate)) < 0.05
