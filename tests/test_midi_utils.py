"""
Tests for MIDI utilities, pitch conversions, note formatting, and parsing.
"""

import tempfile
from pathlib import Path
import pytest

from src.utils.midi_utils import (
    NoteEvent,
    format_note_sequence,
    format_timestamp,
    midi_to_note_name,
    note_name_to_midi,
    notes_to_midi_file,
    parse_midi_file,
)


def test_midi_pitch_conversions():
    test_cases = [
        (60, "C4"),
        (69, "A4"),
        (21, "A0"),
        (108, "C8"),
        (61, "C#4"),
        (71, "B4"),
    ]
    for pitch, name in test_cases:
        assert midi_to_note_name(pitch) == name
        assert note_name_to_midi(name) == pitch

    assert note_name_to_midi("Db4") == 61
    assert note_name_to_midi("Eb4") == 63
    assert note_name_to_midi("Bb3") == 58


def test_timestamp_and_sequence_formatting():
    assert format_timestamp(0.52) == "00:00.52"
    assert format_timestamp(65.4) == "01:05.40"

    notes = [
        NoteEvent(pitch=60, onset=0.5, offset=1.0),
        NoteEvent(pitch=64, onset=1.0, offset=1.5),
        NoteEvent(pitch=67, onset=1.5, offset=2.0),
    ]
    seq_str = format_note_sequence(notes)
    assert seq_str == "C4 -> E4 -> G4"


def test_midi_file_roundtrip():
    notes_in = [
        NoteEvent(pitch=60, onset=0.0, offset=0.5, velocity=0.8),
        NoteEvent(pitch=64, onset=0.5, offset=1.0, velocity=0.7),
        NoteEvent(pitch=67, onset=1.0, offset=1.5, velocity=0.9),
    ]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        notes_to_midi_file(notes_in, tmp_path)
        assert tmp_path.exists()
        assert tmp_path.stat().st_size > 0

        parsed_notes = parse_midi_file(tmp_path)
        assert len(parsed_notes) == 3

        for orig, parsed in zip(notes_in, parsed_notes):
            assert orig.pitch == parsed.pitch
            assert abs(orig.onset - parsed.onset) < 0.05
            assert abs(orig.offset - parsed.offset) < 0.05
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
