"""
MIDI utilities: parsing, pitch-to-note conversion, timeline formatting, and MIDI export.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
import re

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

NOTE_ALIASES: Dict[str, str] = {
    "DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#",
    "C#": "C#", "D#": "D#", "F#": "F#", "G#": "G#", "A#": "A#",
}


@dataclass
class NoteEvent:
    """Represents a discrete musical note event."""
    pitch: int                  # MIDI pitch number (21 = A0 ... 108 = C8)
    onset: float                # Onset time in seconds
    offset: float               # Offset time in seconds
    velocity: Union[int, float] = 100  # MIDI velocity (0-127) or normalized [0, 1]
    confidence: Optional[float] = None # Prediction probability / confidence [0, 1] if available

    @property
    def note_name(self) -> str:
        return midi_to_note_name(self.pitch)

    @property
    def duration(self) -> float:
        return max(0.0, self.offset - self.onset)

    def to_dict(self) -> dict:
        vel_val = int(self.velocity) if self.velocity > 1 or isinstance(self.velocity, int) else int(round(float(self.velocity) * 127))
        d = {
            "note": self.note_name,
            "pitch": int(self.pitch),
            "onset": round(float(self.onset), 3),
            "offset": round(float(self.offset), 3),
            "duration": round(float(self.duration), 3),
            "velocity": vel_val,
            "onset_formatted": format_timestamp(self.onset),
            "offset_formatted": format_timestamp(self.offset),
        }
        if self.confidence is not None:
            d["confidence"] = round(float(self.confidence), 3)
        return d


def midi_to_note_name(pitch: int) -> str:
    """Convert MIDI pitch integer (21..108) to scientific notation (e.g. 60 -> 'C4')."""
    if not (0 <= pitch <= 127):
        return f"Pitch{pitch}"
    octave = (pitch // 12) - 1
    note_index = pitch % 12
    return f"{NOTE_NAMES[note_index]}{octave}"


def note_name_to_midi(name: str) -> int:
    """Convert note name (e.g. 'C4', 'C#4', 'Db4') to MIDI pitch integer."""
    clean_name = name.strip().upper()
    match = re.match(r"^([A-G][#B]?)(-?\d+)$", clean_name)
    if not match:
        raise ValueError(f"Invalid note name format: '{name}'")

    note_letter, octave_str = match.groups()
    if len(note_letter) > 1 and note_letter[1] == "B":
        note_letter = NOTE_ALIASES.get(note_letter, note_letter)

    note_index = NOTE_NAMES.index(note_letter)
    octave = int(octave_str)
    pitch = (octave + 1) * 12 + note_index

    if not (0 <= pitch <= 127):
        raise ValueError(f"Pitch out of standard MIDI range (0-127): {pitch}")
    return pitch


def format_timestamp(seconds: float) -> str:
    """Format seconds into 'MM:SS.ss'."""
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    return f"{minutes:02d}:{rem_seconds:05.2f}"


def format_note_sequence(
    note_events: List[NoteEvent],
    max_notes: Optional[int] = 30,
    separator: str = " -> ",
) -> str:
    """Format note events into a linear sequence chain (e.g. 'C4 -> E4 -> G4 -> C5')."""
    if not note_events:
        return "No notes detected"

    sorted_notes = sorted(note_events, key=lambda n: n.onset)
    names = [n.note_name for n in sorted_notes]

    if max_notes and len(names) > max_notes:
        displayed = names[:max_notes]
        return separator.join(displayed) + f"{separator}... (+{len(names) - max_notes} more)"

    return separator.join(names)


def parse_midi_file(midi_path: Union[str, Path]) -> List[NoteEvent]:
    """Parse MIDI file and extract all note events."""
    path_str = str(Path(midi_path).resolve())
    notes: List[NoteEvent] = []

    try:
        import pretty_midi
        midi_data = pretty_midi.PrettyMIDI(path_str)
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append(
                        NoteEvent(
                            pitch=int(note.pitch),
                            onset=float(note.start),
                            offset=float(note.end),
                            velocity=float(note.velocity) / 127.0,
                            confidence=1.0,
                        )
                    )
    except Exception:
        try:
            import mido
            mid = mido.MidiFile(path_str)
            active_notes: Dict[int, float] = {}
            current_time = 0.0

            for msg in mid:
                current_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    active_notes[msg.note] = current_time
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time = active_notes.pop(msg.note)
                        notes.append(
                            NoteEvent(
                                pitch=int(msg.note),
                                onset=start_time,
                                offset=current_time,
                                velocity=float(getattr(msg, "velocity", 64)) / 127.0,
                                confidence=1.0,
                            )
                        )
        except Exception as e:
            raise RuntimeError(f"Failed to parse MIDI file '{path_str}': {e}")

    notes.sort(key=lambda n: (n.onset, n.pitch))
    return notes


def notes_to_midi_file(
    note_events: List[NoteEvent],
    output_path: Union[str, Path],
    instrument_program: int = 0,
    tempo: float = 120.0,
) -> Path:
    """Convert a list of NoteEvents into a standard General MIDI file."""
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=instrument_program, is_drum=False, name="Piano")

        for event in note_events:
            if 0 <= event.pitch <= 127:
                vel = int(np_clip(event.velocity if event.velocity > 1 else event.velocity * 127.0, 1, 127))
                pm_note = pretty_midi.Note(
                    velocity=vel,
                    pitch=int(event.pitch),
                    start=float(event.onset),
                    end=float(max(event.offset, event.onset + 0.05)),
                )
                piano.notes.append(pm_note)

        midi.instruments.append(piano)
        midi.write(str(out_path))
    except Exception:
        import mido
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo)))
        track.append(mido.Message("program_change", program=instrument_program, time=0))

        events = []
        for note in note_events:
            if 0 <= note.pitch <= 127:
                vel = int(np_clip(note.velocity if note.velocity > 1 else note.velocity * 127.0, 1, 127))
                events.append((note.onset, "note_on", int(note.pitch), vel))
                events.append((max(note.offset, note.onset + 0.05), "note_off", int(note.pitch), 0))

        events.sort(key=lambda x: x[0])

        last_time = 0.0
        ticks_per_second = mid.ticks_per_beat * (tempo / 60.0)

        for event_time, event_type, pitch, vel in events:
            delta_seconds = max(0.0, event_time - last_time)
            delta_ticks = int(delta_seconds * ticks_per_second)
            last_time = event_time
            track.append(mido.Message(event_type, note=pitch, velocity=vel, time=delta_ticks))

        mid.save(str(out_path))

    return out_path


def np_clip(val: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, val))
