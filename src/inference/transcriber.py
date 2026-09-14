"""
End-to-end Inference Engine for Automatic Piano Transcription using Pretrained CRNN.
"""

import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Union
import urllib.error
import urllib.request
import numpy as np
import torch
from piano_transcription_inference import PianoTranscription

from src.preprocessing.audio_processor import AudioProcessor
from src.utils.audio_io import load_audio
from src.utils.config import get_device, load_config
from src.utils.midi_utils import NoteEvent, format_note_sequence, notes_to_midi_file


CHECKPOINT_URLS = [
    "https://zenodo.org/records/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1",
    "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth?download=1",
    "https://zenodo.org/records/4034264/files/CRNN_note_F1=0.9677_pedal_F1=0.9186.pth?download=1",
]
DEFAULT_CHECKPOINT_FILENAME = "note_F1=0.9677_pedal_F1=0.9186.pth"
MIN_VALID_CHECKPOINT_SIZE = 160_000_000  # ~164 MB expected size is 171,966,578 bytes


def resolve_or_download_checkpoint(
    custom_path: Optional[Union[str, Path]] = None,
    target_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    min_size_bytes: int = MIN_VALID_CHECKPOINT_SIZE,
) -> Path:
    """
    Ensure the pretrained PianoTranscription CRNN model checkpoint is available locally.
    Discovers existing cached weights across candidate locations, or downloads them
    safely with streaming chunks and size validation.

    Candidate check order:
      1. Explicit custom_path argument
      2. PIANO_TRANSCRIPTION_CHECKPOINT environment variable
      3. Standard user home cache directory (~/piano_transcription_inference_data/)
      4. Project local models directory (PROJECT_ROOT/models/)
    """
    candidates = []

    if custom_path:
        p = Path(custom_path).resolve()
        if p.is_file() and p.stat().st_size >= min_size_bytes:
            return p
        candidates.append(p)

    env_path = os.environ.get("PIANO_TRANSCRIPTION_CHECKPOINT")
    if env_path:
        p = Path(env_path).resolve()
        if p.is_file() and p.stat().st_size >= min_size_bytes:
            return p
        candidates.append(p)

    # Standard home cache directory used by piano_transcription_inference
    home_dir = Path.home() / "piano_transcription_inference_data"
    home_ckpt = home_dir / target_filename
    if home_ckpt.is_file() and home_ckpt.stat().st_size >= min_size_bytes:
        return home_ckpt
    candidates.append(home_ckpt)

    # Project root models directory
    project_root = Path(__file__).resolve().parent.parent.parent
    project_ckpt = project_root / "models" / target_filename
    if project_ckpt.is_file() and project_ckpt.stat().st_size >= min_size_bytes:
        return project_ckpt
    candidates.append(project_ckpt)

    # Check if any candidate exists with valid size
    for cand in candidates:
        if cand.is_file() and cand.stat().st_size >= min_size_bytes:
            return cand

    # Determine best download destination
    dest_path = home_ckpt
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        dest_path = project_ckpt
        dest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest_path.with_suffix(".pth.download.tmp")

    print(f"[Model Loader] Downloading pretrained piano transcription checkpoint to {dest_path} (~165 MB)...")
    last_error = None

    for url in CHECKPOINT_URLS:
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "curl/7.68.0"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1 MB

                with open(tmp_path, "wb") as f_out:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r[Model Loader] Progress: {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB ({pct:.1f}%)", end="", flush=True)

                print()  # Newline after progress bar

            # Verify downloaded file size
            if tmp_path.exists() and tmp_path.stat().st_size >= min_size_bytes:
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
                tmp_path.rename(dest_path)
                print(f"[Model Loader] Successfully downloaded and cached checkpoint ({dest_path.stat().st_size / (1024*1024):.1f} MB).")
                return dest_path
            else:
                actual_size = tmp_path.stat().st_size if tmp_path.exists() else 0
                raise ValueError(f"Downloaded file size ({actual_size} bytes) was smaller than expected minimum ({min_size_bytes} bytes).")

        except Exception as e:
            last_error = e
            print(f"\n[Model Loader] Download failed from {url}: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    raise RuntimeError(
        f"Failed to automatically download pretrained model checkpoint. Last error: {last_error}"
    )


class PianoTranscriber:
    """
    High-level transcription engine powered by the pretrained PianoTranscription
    neural network (Kong et al.). Loads audio, runs deep acoustic inference,
    extracts note and pedal events, and generates structured outputs.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config: Optional[object] = None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        self.config = config or load_config()

        if device is None:
            self.device_str = "cpu"
        elif isinstance(device, torch.device):
            self.device_str = "cuda" if device.type == "cuda" else "cpu"
        else:
            self.device_str = str(device).lower()

        # Resolve or automatically download model checkpoint
        self.checkpoint_path = resolve_or_download_checkpoint(checkpoint_path)

        self.model = PianoTranscription(
            device=self.device_str,
            checkpoint_path=str(self.checkpoint_path),
        )

        self.audio_proc = AudioProcessor.from_config(self.config)

    def transcribe_audio(
        self,
        audio_or_path: Union[str, Path, np.ndarray],
        onset_threshold: Optional[float] = None,
        frame_threshold: Optional[float] = None,
        min_note_duration: Optional[float] = None,
        midi_path: Optional[Union[str, Path]] = None,
    ) -> Dict:
        """
        Transcribe an audio file or waveform into piano note events.

        Args:
            audio_or_path: Path to audio file or 1D numpy array.
            onset_threshold: Optional onset sensitivity threshold (default 0.3).
            frame_threshold: Optional frame sustain threshold (default 0.1).
            min_note_duration: Minimum duration in seconds to keep a detected note.
            midi_path: Optional path to directly write the generated MIDI file.

        Returns:
            Dict containing detected notes, NoteEvent list, sequence chain,
            duration, spectrogram, waveform, and pedal events.
        """
        if onset_threshold is not None:
            self.model.onset_threshold = float(onset_threshold)
        if frame_threshold is not None:
            self.model.frame_threshold = float(frame_threshold)

        # Load audio safely without relying on broken package load_audio
        if isinstance(audio_or_path, (str, Path)):
            audio_np, sr = load_audio(
                audio_or_path,
                target_sr=16000,
                mono=True,
                normalize=True,
            )
        else:
            audio_np = np.asarray(audio_or_path, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=-1)
            sr = 16000

        duration = len(audio_np) / float(sr)

        # Compute Log-Mel Spectrogram for visualization
        mel_spec = self.audio_proc.compute_mel_spectrogram(audio_np)

        # Run transcription with pretrained model
        midi_out_str = str(Path(midi_path).resolve()) if midi_path else None
        trans_dict = self.model.transcribe(audio_np, midi_path=midi_out_str)

        est_note_events = trans_dict.get("est_note_events", [])
        est_pedal_events = trans_dict.get("est_pedal_events", [])

        min_dur = (
            min_note_duration
            if min_note_duration is not None
            else float(getattr(self.config.inference, "min_note_duration", 0.05))
        )

        note_events: List[NoteEvent] = []
        for ev in est_note_events:
            pitch = int(ev["midi_note"])
            onset = float(ev["onset_time"])
            offset = float(ev["offset_time"])
            velocity = int(ev.get("velocity", 100))

            if (offset - onset) >= min_dur:
                note_events.append(
                    NoteEvent(
                        pitch=pitch,
                        onset=onset,
                        offset=offset,
                        velocity=velocity,
                    )
                )

        note_events.sort(key=lambda n: (n.onset, n.pitch))
        sequence_chain = format_note_sequence(note_events)

        return {
            "notes": [n.to_dict() for n in note_events],
            "note_events": note_events,
            "raw_note_events": est_note_events,
            "pedal_events": est_pedal_events,
            "sequence_chain": sequence_chain,
            "duration": duration,
            "mel_spectrogram": mel_spec,
            "audio": audio_np,
            "midi_path": midi_out_str,
        }

    def export_midi(
        self,
        note_events: Union[List[NoteEvent], List[dict]],
        output_path: Union[str, Path],
        pedal_events: Optional[List[dict]] = None,
    ) -> Path:
        """
        Export note events to a standard General MIDI file.
        """
        out_path = Path(output_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from piano_transcription_inference.utilities import write_events_to_midi

            raw_events = []
            for n in note_events:
                if isinstance(n, NoteEvent):
                    raw_events.append({
                        "midi_note": int(n.pitch),
                        "onset_time": float(n.onset),
                        "offset_time": float(n.offset),
                        "velocity": int(n.velocity if n.velocity > 1 else n.velocity * 127),
                    })
                elif isinstance(n, dict):
                    raw_events.append({
                        "midi_note": int(n.get("midi_note", n.get("pitch", 60))),
                        "onset_time": float(n.get("onset_time", n.get("onset", 0.0))),
                        "offset_time": float(n.get("offset_time", n.get("offset", 0.0))),
                        "velocity": int(n.get("velocity", 100)),
                    })
            write_events_to_midi(
                start_time=0,
                note_events=raw_events,
                pedal_events=pedal_events,
                midi_path=str(out_path),
            )
            return out_path
        except Exception:
            if note_events and isinstance(note_events[0], dict):
                note_events_objs = [
                    NoteEvent(
                        pitch=int(n.get("pitch", n.get("midi_note", 60))),
                        onset=float(n.get("onset", n.get("onset_time", 0.0))),
                        offset=float(n.get("offset", n.get("offset_time", 0.0))),
                        velocity=int(n.get("velocity", 100)),
                    )
                    for n in note_events
                ]
            else:
                note_events_objs = note_events  # type: ignore
            return notes_to_midi_file(note_events_objs, out_path)
