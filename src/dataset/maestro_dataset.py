"""
MAESTRO Dataset and PyTorch DataLoader implementations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.audio_processor import AudioProcessor
from src.preprocessing.label_processor import LabelProcessor
from src.utils.audio_io import load_audio
from src.utils.midi_utils import NoteEvent, parse_midi_file


class MaestroDataset(Dataset):
    """
    MAESTRO (MIDI and Audio Edited for Synchronous Tracks and Organization) Dataset.

    Loads aligned acoustic piano audio and MIDI annotations, segmenting long recordings
    into fixed-duration training slices with aligned multi-pitch target matrices.
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        split: str = "train",
        segment_duration: float = 3.0,
        hop_duration: float = 1.5,
        audio_processor: Optional[AudioProcessor] = None,
        label_processor: Optional[LabelProcessor] = None,
        subset_ratio: float = 1.0,
        cache_spectrograms: bool = False,
    ):
        self.dataset_root = Path(dataset_root).resolve()
        self.split = split.lower()
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.audio_proc = audio_processor or AudioProcessor()
        self.label_proc = label_processor or LabelProcessor()
        self.subset_ratio = subset_ratio
        self.cache_spectrograms = cache_spectrograms

        # Calculate exact number of frames per segment
        self.segment_samples = int(self.segment_duration * self.audio_proc.sample_rate)
        # Approximate frame count in segment
        self.segment_frames = int(round(self.segment_duration * self.audio_proc.frame_rate))

        self.samples: List[Dict] = []
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        self._index_dataset()

    def _index_dataset(self) -> None:
        """Scan metadata and audio/MIDI files to create segmented slice index."""
        metadata_path = self.dataset_root / "maestro-v3.0.0.json"
        records = []

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                records = json.load(f)
        else:
            # Fallback: scan directory recursively for matching audio and MIDI files
            wav_files = list(self.dataset_root.rglob("*.wav"))
            for wav_path in wav_files:
                midi_candidates = [
                    wav_path.with_suffix(".midi"),
                    wav_path.with_suffix(".mid"),
                ]
                midi_path = next((m for m in midi_candidates if m.exists()), None)
                if midi_path:
                    records.append({
                        "canonical_composer": "Unknown",
                        "canonical_title": wav_path.stem,
                        "split": "train",
                        "audio_filename": str(wav_path.relative_to(self.dataset_root)),
                        "midi_filename": str(midi_path.relative_to(self.dataset_root)),
                    })

        # Filter by split if specified
        if self.split != "all":
            split_records = [r for r in records if r.get("split", "train").lower() == self.split]
            # If requested split is empty, fall back to all records
            if not split_records and records:
                split_records = records
        else:
            split_records = records

        # Subset slicing if requested
        if 0.0 < self.subset_ratio < 1.0:
            count = max(1, int(len(split_records) * self.subset_ratio))
            split_records = split_records[:count]

        # Index segments for each piece
        for record in split_records:
            audio_path = self.dataset_root / record["audio_filename"]
            midi_path = self.dataset_root / record["midi_filename"]

            if not audio_path.exists() or not midi_path.exists():
                continue

            try:
                # Parse all notes from MIDI
                all_notes = parse_midi_file(midi_path)
                if not all_notes:
                    continue

                # Get duration
                audio_np, sr = load_audio(audio_path, target_sr=self.audio_proc.sample_rate)
                total_duration = len(audio_np) / self.audio_proc.sample_rate

                # Create overlapping time segments
                current_time = 0.0
                while current_time < total_duration:
                    end_time = current_time + self.segment_duration

                    # Extract notes active within [current_time, end_time]
                    segment_notes = []
                    for n in all_notes:
                        if n.offset > current_time and n.onset < end_time:
                            # Shift note relative to segment start
                            rel_onset = max(0.0, n.onset - current_time)
                            rel_offset = min(self.segment_duration, n.offset - current_time)
                            segment_notes.append(
                                NoteEvent(
                                    pitch=n.pitch,
                                    onset=rel_onset,
                                    offset=rel_offset,
                                    velocity=n.velocity,
                                    confidence=1.0,
                                )
                            )

                    self.samples.append({
                        "audio_path": audio_path,
                        "start_time": current_time,
                        "end_time": end_time,
                        "start_sample": int(current_time * self.audio_proc.sample_rate),
                        "end_sample": int(end_time * self.audio_proc.sample_rate),
                        "notes": segment_notes,
                        "composer": record.get("canonical_composer", "Unknown"),
                        "title": record.get("canonical_title", "Unknown"),
                    })

                    current_time += self.hop_duration
                    if end_time >= total_duration:
                        break

            except Exception as e:
                print(f"Warning: Failed to index track '{audio_path.name}': {e}")
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            mel_tensor: [1, n_mels, T_frames] (float32)
            frame_targets: [T_frames, 88] (float32)
            onset_targets: [T_frames, 88] (float32)
            metadata: dict
        """
        if self.cache_spectrograms and idx in self._cache:
            mel_tensor, frame_target_tensor, onset_target_tensor = self._cache[idx]
            return mel_tensor, frame_target_tensor, onset_target_tensor, self.samples[idx]

        sample_info = self.samples[idx]
        audio_path = sample_info["audio_path"]
        start_samp = sample_info["start_sample"]
        end_samp = sample_info["end_sample"]

        # Load full audio and extract segment
        audio_full, _ = load_audio(
            audio_path,
            target_sr=self.audio_proc.sample_rate,
            mono=True,
            normalize=self.audio_proc.normalize,
        )

        segment_audio = audio_full[start_samp:end_samp]
        if len(segment_audio) < self.segment_samples:
            # Pad segment if short
            padded = np.zeros(self.segment_samples, dtype=np.float32)
            padded[:len(segment_audio)] = segment_audio
            segment_audio = padded

        # Compute Log-Mel Spectrogram [n_mels, T_frames]
        mel_spec = self.audio_proc.compute_mel_spectrogram(segment_audio)
        n_mels, num_frames = mel_spec.shape

        # Generate target multi-pitch matrices [num_frames, 88]
        frame_roll, onset_roll = self.label_proc.notes_to_targets(
            note_events=sample_info["notes"],
            total_frames=num_frames,
            frame_rate=self.audio_proc.frame_rate,
        )

        # Convert to PyTorch tensors
        # Add channel dimension to spectrogram: [1, n_mels, num_frames]
        mel_tensor = torch.from_numpy(mel_spec).unsqueeze(0).float()
        frame_target_tensor = torch.from_numpy(frame_roll).float()
        onset_target_tensor = torch.from_numpy(onset_roll).float()

        if self.cache_spectrograms:
            self._cache[idx] = (mel_tensor, frame_target_tensor, onset_target_tensor)

        return mel_tensor, frame_target_tensor, onset_target_tensor, sample_info


def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test PyTorch DataLoaders based on configuration.

    Args:
        config: Central configuration object.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    audio_proc = AudioProcessor.from_config(config)
    label_proc = LabelProcessor.from_config(config)

    root_path = Path(config.dataset.dataset_root)
    segment_dur = float(config.dataset.segment_duration)
    hop_dur = float(config.dataset.hop_duration)
    batch_size = int(config.training.batch_size)
    num_workers = int(config.dataset.num_workers)

    train_ds = MaestroDataset(
        dataset_root=root_path,
        split="train",
        segment_duration=segment_dur,
        hop_duration=hop_dur,
        audio_processor=audio_proc,
        label_processor=label_proc,
    )

    val_ds = MaestroDataset(
        dataset_root=root_path,
        split="validation",
        segment_duration=segment_dur,
        hop_duration=hop_dur,
        audio_processor=audio_proc,
        label_processor=label_proc,
    )

    test_ds = MaestroDataset(
        dataset_root=root_path,
        split="test",
        segment_duration=segment_dur,
        hop_duration=hop_dur,
        audio_processor=audio_proc,
        label_processor=label_proc,
    )

    def custom_collate(batch):
        mels = torch.stack([item[0] for item in batch], dim=0)
        frames = torch.stack([item[1] for item in batch], dim=0)
        onsets = torch.stack([item[2] for item in batch], dim=0)
        metas = [item[3] for item in batch]
        return mels, frames, onsets, metas

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate,
        drop_last=(len(train_ds) > batch_size),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate,
    )

    return train_loader, val_loader, test_loader
