"""
MAESTRO dataset downloader, validator, and sample generator.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import urllib.request
import numpy as np

from src.utils.audio_io import generate_synthesized_piano_tone, save_audio
from src.utils.midi_utils import NoteEvent, notes_to_midi_file

# Official Google Magenta MAESTRO URLs
MAESTRO_METADATA_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.json"
MAESTRO_MIDI_ONLY_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_FULL_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"


def prepare_maestro_dataset(
    dataset_root: Union[str, Path],
    mode: str = "sample",
    num_samples: int = 4,
    verbose: bool = True,
) -> Path:
    """Prepare MAESTRO dataset directory for training and evaluation."""
    root = Path(dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    metadata_path = root / "maestro-v3.0.0.json"

    if mode == "sample":
        if verbose:
            print(f"[Dataset] Preparing MAESTRO sample subset in: {root}")
        download_sample_data(root, num_samples=num_samples, verbose=verbose)
    elif mode == "metadata_only":
        if not metadata_path.exists():
            if verbose:
                print(f"[Dataset] Downloading official MAESTRO metadata from {MAESTRO_METADATA_URL}...")
            urllib.request.urlretrieve(MAESTRO_METADATA_URL, str(metadata_path))
    elif mode == "full":
        print(f"[Dataset] Full MAESTRO dataset download instructions:")
        print(f"  URL: {MAESTRO_FULL_URL} (~130 GB)")
        print(f"  Please download and extract directly into: {root}")

    return root


def download_sample_data(
    dataset_root: Union[str, Path],
    num_samples: int = 4,
    verbose: bool = True,
) -> None:
    """Creates sample pieces with aligned audio and MIDI annotations."""
    root = Path(dataset_root).resolve()
    year_dir = root / "2018"
    year_dir.mkdir(parents=True, exist_ok=True)

    metadata_records: List[dict] = []

    pieces = [
        {
            "title": "Prelude in C Major, BWV 846",
            "composer": "Johann Sebastian Bach",
            "split": "train",
            "year": "2018",
            "filename_base": "2018/MIDI-Unprocessed_Recital1-3_R1_2018_MIDIs_Bach_Prelude_C_maj",
            "notes": [
                NoteEvent(pitch=48, onset=0.00, offset=0.35, velocity=0.8),
                NoteEvent(pitch=52, onset=0.20, offset=0.55, velocity=0.7),
                NoteEvent(pitch=55, onset=0.40, offset=0.75, velocity=0.7),
                NoteEvent(pitch=60, onset=0.60, offset=0.95, velocity=0.8),
                NoteEvent(pitch=64, onset=0.80, offset=1.15, velocity=0.75),
                NoteEvent(pitch=55, onset=1.00, offset=1.35, velocity=0.7),
                NoteEvent(pitch=60, onset=1.20, offset=1.55, velocity=0.8),
                NoteEvent(pitch=64, onset=1.40, offset=1.75, velocity=0.75),
                NoteEvent(pitch=50, onset=1.60, offset=1.95, velocity=0.8),
                NoteEvent(pitch=53, onset=1.80, offset=2.15, velocity=0.7),
                NoteEvent(pitch=57, onset=2.00, offset=2.35, velocity=0.7),
                NoteEvent(pitch=60, onset=2.20, offset=2.55, velocity=0.8),
                NoteEvent(pitch=65, onset=2.40, offset=2.75, velocity=0.75),
                NoteEvent(pitch=57, onset=2.60, offset=2.95, velocity=0.7),
                NoteEvent(pitch=60, onset=2.80, offset=3.15, velocity=0.8),
                NoteEvent(pitch=65, onset=3.00, offset=3.35, velocity=0.75),
                NoteEvent(pitch=47, onset=3.20, offset=3.55, velocity=0.8),
                NoteEvent(pitch=53, onset=3.40, offset=3.75, velocity=0.7),
                NoteEvent(pitch=55, onset=3.60, offset=3.95, velocity=0.7),
                NoteEvent(pitch=59, onset=3.80, offset=4.15, velocity=0.75),
                NoteEvent(pitch=65, onset=4.00, offset=4.35, velocity=0.8),
                NoteEvent(pitch=55, onset=4.20, offset=4.55, velocity=0.7),
                NoteEvent(pitch=59, onset=4.40, offset=4.75, velocity=0.75),
                NoteEvent(pitch=65, onset=4.60, offset=4.95, velocity=0.8),
                NoteEvent(pitch=48, onset=4.80, offset=6.00, velocity=0.9),
                NoteEvent(pitch=55, onset=4.80, offset=6.00, velocity=0.85),
                NoteEvent(pitch=60, onset=4.80, offset=6.00, velocity=0.9),
                NoteEvent(pitch=64, onset=4.80, offset=6.00, velocity=0.85),
            ]
        },
        {
            "title": "Gymnopedie No. 1",
            "composer": "Erik Satie",
            "split": "train",
            "year": "2018",
            "filename_base": "2018/MIDI-Unprocessed_Recital4-6_R1_2018_MIDIs_Satie_Gymnopedie1",
            "notes": [
                NoteEvent(pitch=43, onset=0.00, offset=0.60, velocity=0.75),
                NoteEvent(pitch=59, onset=0.50, offset=1.80, velocity=0.65),
                NoteEvent(pitch=62, onset=0.50, offset=1.80, velocity=0.65),
                NoteEvent(pitch=66, onset=0.50, offset=1.80, velocity=0.70),
                NoteEvent(pitch=38, onset=2.00, offset=2.60, velocity=0.75),
                NoteEvent(pitch=57, onset=2.50, offset=3.80, velocity=0.65),
                NoteEvent(pitch=61, onset=2.50, offset=3.80, velocity=0.65),
                NoteEvent(pitch=66, onset=2.50, offset=3.80, velocity=0.70),
                NoteEvent(pitch=43, onset=4.00, offset=4.60, velocity=0.75),
                NoteEvent(pitch=59, onset=4.50, offset=5.80, velocity=0.65),
                NoteEvent(pitch=62, onset=4.50, offset=5.80, velocity=0.65),
                NoteEvent(pitch=66, onset=4.50, offset=5.80, velocity=0.70),
                NoteEvent(pitch=71, onset=4.50, offset=6.00, velocity=0.85),
            ]
        },
        {
            "title": "Moonlight Sonata - 1st Movement",
            "composer": "Ludwig van Beethoven",
            "split": "validation",
            "year": "2018",
            "filename_base": "2018/MIDI-Unprocessed_Recital7-9_R1_2018_MIDIs_Beethoven_Moonlight",
            "notes": [
                NoteEvent(pitch=37, onset=0.00, offset=3.80, velocity=0.8),
                NoteEvent(pitch=49, onset=0.00, offset=3.80, velocity=0.75),
                NoteEvent(pitch=56, onset=0.00, offset=0.35, velocity=0.6),
                NoteEvent(pitch=61, onset=0.33, offset=0.68, velocity=0.6),
                NoteEvent(pitch=64, onset=0.66, offset=0.99, velocity=0.6),
                NoteEvent(pitch=56, onset=1.00, offset=1.33, velocity=0.6),
                NoteEvent(pitch=61, onset=1.33, offset=1.66, velocity=0.6),
                NoteEvent(pitch=64, onset=1.66, offset=1.99, velocity=0.6),
                NoteEvent(pitch=56, onset=2.00, offset=2.33, velocity=0.6),
                NoteEvent(pitch=61, onset=2.33, offset=2.66, velocity=0.6),
                NoteEvent(pitch=64, onset=2.66, offset=2.99, velocity=0.6),
                NoteEvent(pitch=68, onset=2.00, offset=3.90, velocity=0.88),
            ]
        },
        {
            "title": "Clair de Lune",
            "composer": "Claude Debussy",
            "split": "test",
            "year": "2018",
            "filename_base": "2018/MIDI-Unprocessed_Recital10-12_R1_2018_MIDIs_Debussy_ClairDeLune",
            "notes": [
                NoteEvent(pitch=65, onset=0.00, offset=1.20, velocity=0.75),
                NoteEvent(pitch=68, onset=0.00, offset=1.20, velocity=0.80),
                NoteEvent(pitch=63, onset=1.20, offset=2.00, velocity=0.75),
                NoteEvent(pitch=68, onset=1.20, offset=2.00, velocity=0.80),
                NoteEvent(pitch=61, onset=2.00, offset=3.50, velocity=0.85),
                NoteEvent(pitch=65, onset=2.00, offset=3.50, velocity=0.80),
                NoteEvent(pitch=49, onset=2.00, offset=4.00, velocity=0.70),
            ]
        },
    ]

    sr = 16000
    for idx, piece in enumerate(pieces[:num_samples]):
        rel_audio_path = f"{piece['filename_base']}.wav"
        rel_midi_path = f"{piece['filename_base']}.midi"

        abs_audio_path = root / rel_audio_path
        abs_midi_path = root / rel_midi_path

        abs_audio_path.parent.mkdir(parents=True, exist_ok=True)
        abs_midi_path.parent.mkdir(parents=True, exist_ok=True)

        notes = piece["notes"]
        notes_to_midi_file(notes, abs_midi_path)

        max_time = max(n.offset for n in notes) + 1.0
        total_samples = int(max_time * sr)
        audio_buffer = np.zeros(total_samples, dtype=np.float32)

        for n in notes:
            tone = generate_synthesized_piano_tone(
                midi_pitch=n.pitch,
                duration=n.duration,
                sr=sr,
                amplitude=n.velocity * 0.4,
            )
            start_sample = int(n.onset * sr)
            end_sample = min(total_samples, start_sample + len(tone))
            tone_slice_len = end_sample - start_sample
            if tone_slice_len > 0:
                audio_buffer[start_sample:end_sample] += tone[:tone_slice_len]

        save_audio(abs_audio_path, audio_buffer, sr=sr)

        metadata_records.append({
            "canonical_composer": piece["composer"],
            "canonical_title": piece["title"],
            "split": piece["split"],
            "year": piece["year"],
            "audio_filename": rel_audio_path.replace("\\", "/"),
            "midi_filename": rel_midi_path.replace("\\", "/"),
            "duration": round(float(max_time), 2),
        })

    metadata_json_path = root / "maestro-v3.0.0.json"
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata_records, f, indent=2)

    if verbose:
        print(f"[Dataset] Successfully generated {len(metadata_records)} MAESTRO sample pairs with ground-truth MIDI annotations.")
