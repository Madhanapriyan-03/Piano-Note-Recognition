"""
Label processing: Converting ground-truth MIDI notes into frame-level multi-pitch target matrices.
"""

from typing import List, Optional, Tuple
import numpy as np
from src.utils.midi_utils import NoteEvent, midi_to_note_name


class LabelProcessor:
    """
    Transforms discrete note events (pitch, onset, offset) into frame-level multi-pitch target tensors
    and converts prediction matrices back to discrete NoteEvent instances.
    """

    def __init__(
        self,
        min_midi: int = 21,
        max_midi: int = 108,
        onset_tolerance_frames: int = 1,
    ):
        self.min_midi = min_midi
        self.max_midi = max_midi
        self.num_pitches = max_midi - min_midi + 1
        self.onset_tolerance_frames = onset_tolerance_frames

    @classmethod
    def from_config(cls, config) -> "LabelProcessor":
        cfg = config.model if hasattr(config, "model") else config
        return cls(
            min_midi=getattr(cfg, "min_midi", 21),
            max_midi=getattr(cfg, "max_midi", 108),
        )

    def pitch_to_index(self, pitch: int) -> Optional[int]:
        if self.min_midi <= pitch <= self.max_midi:
            return pitch - self.min_midi
        return None

    def index_to_pitch(self, idx: int) -> int:
        return idx + self.min_midi

    def notes_to_targets(
        self,
        note_events: List[NoteEvent],
        total_frames: int,
        frame_rate: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert list of NoteEvents into multi-pitch target matrices."""
        frame_roll = np.zeros((total_frames, self.num_pitches), dtype=np.float32)
        onset_roll = np.zeros((total_frames, self.num_pitches), dtype=np.float32)

        for note in note_events:
            idx = self.pitch_to_index(note.pitch)
            if idx is None:
                continue

            onset_frame = int(round(note.onset * frame_rate))
            offset_frame = int(round(note.offset * frame_rate))
            offset_frame = max(onset_frame + 1, offset_frame)

            start_f = max(0, min(total_frames, onset_frame))
            end_f = max(0, min(total_frames, offset_frame))

            if start_f < total_frames and end_f > start_f:
                frame_roll[start_f:end_f, idx] = 1.0

            if 0 <= onset_frame < total_frames:
                onset_roll[onset_frame, idx] = 1.0
                for offset in range(1, self.onset_tolerance_frames + 1):
                    if onset_frame - offset >= 0:
                        onset_roll[onset_frame - offset, idx] = max(
                            onset_roll[onset_frame - offset, idx], 0.5
                        )
                    if onset_frame + offset < total_frames:
                        onset_roll[onset_frame + offset, idx] = max(
                            onset_roll[onset_frame + offset, idx], 0.5
                        )

        return frame_roll, onset_roll

    def predictions_to_notes(
        self,
        frame_probs: np.ndarray,
        onset_probs: np.ndarray,
        frame_rate: float,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.4,
        min_note_duration: float = 0.05,
    ) -> List[NoteEvent]:
        """Convert frame and onset probability arrays into discrete NoteEvents."""
        total_frames, num_pitches = frame_probs.shape
        detected_notes: List[NoteEvent] = []

        for p_idx in range(num_pitches):
            pitch = self.index_to_pitch(p_idx)
            f_prob = frame_probs[:, p_idx]
            o_prob = onset_probs[:, p_idx]

            in_note = False
            onset_f = 0
            confidences = []
            onset_conf = 0.0

            for t in range(total_frames):
                is_onset = (o_prob[t] >= onset_threshold) and (
                    t == 0 or o_prob[t] >= o_prob[t - 1]
                ) and (t == total_frames - 1 or o_prob[t] >= o_prob[t + 1])
                is_frame_active = f_prob[t] >= frame_threshold

                if is_onset:
                    if in_note:
                        offset_f = t
                        duration_sec = (offset_f - onset_f) / frame_rate
                        if duration_sec >= min_note_duration:
                            avg_conf = float(np.mean(confidences)) if confidences else float(onset_conf)
                            detected_notes.append(
                                NoteEvent(
                                    pitch=pitch,
                                    onset=onset_f / frame_rate,
                                    offset=offset_f / frame_rate,
                                    velocity=min(1.0, float(onset_conf * 0.9 + 0.1)),
                                    confidence=float(0.6 * onset_conf + 0.4 * avg_conf),
                                )
                            )
                    in_note = True
                    onset_f = t
                    onset_conf = float(o_prob[t])
                    confidences = [float(f_prob[t])]
                elif in_note:
                    if is_frame_active:
                        confidences.append(float(f_prob[t]))
                    else:
                        offset_f = t
                        duration_sec = (offset_f - onset_f) / frame_rate
                        if duration_sec >= min_note_duration:
                            avg_conf = float(np.mean(confidences)) if confidences else float(onset_conf)
                            detected_notes.append(
                                NoteEvent(
                                    pitch=pitch,
                                    onset=onset_f / frame_rate,
                                    offset=offset_f / frame_rate,
                                    velocity=min(1.0, float(onset_conf * 0.9 + 0.1)),
                                    confidence=float(0.6 * onset_conf + 0.4 * avg_conf),
                                )
                            )
                        in_note = False
                        confidences = []

            if in_note:
                offset_f = total_frames
                duration_sec = (offset_f - onset_f) / frame_rate
                if duration_sec >= min_note_duration:
                    avg_conf = float(np.mean(confidences)) if confidences else float(onset_conf)
                    detected_notes.append(
                        NoteEvent(
                            pitch=pitch,
                            onset=onset_f / frame_rate,
                            offset=offset_f / frame_rate,
                            velocity=min(1.0, float(onset_conf * 0.9 + 0.1)),
                            confidence=float(0.6 * onset_conf + 0.4 * avg_conf),
                        )
                    )

        detected_notes.sort(key=lambda n: (n.onset, n.pitch))
        return detected_notes
