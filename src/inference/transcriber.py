"""
End-to-end Inference Engine for Automatic Piano Transcription using Pretrained CRNN.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from piano_transcription_inference import PianoTranscription

from src.preprocessing.audio_processor import AudioProcessor
from src.utils.audio_io import load_audio
from src.utils.config import get_device, load_config
from src.utils.midi_utils import NoteEvent, format_note_sequence, notes_to_midi_file


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

        ckpt_str = str(checkpoint_path) if checkpoint_path else None
        self.model = PianoTranscription(
            device=self.device_str,
            checkpoint_path=ckpt_str,
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
