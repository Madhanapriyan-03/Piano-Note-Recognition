"""
Audio input/output and signal processing utilities.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import scipy.io.wavfile
import scipy.signal


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, resample, and normalize.
    """
    path_str = str(Path(file_path).resolve())

    audio = None
    sr = None
    try:
        import soundfile as sf
        audio, sr = sf.read(path_str, dtype="float32")
    except Exception:
        pass

    if audio is None:
        try:
            import librosa
            audio, sr = librosa.load(path_str, sr=target_sr, mono=mono)
            if normalize:
                audio = normalize_audio(audio)
            return audio.astype(np.float32), int(sr)
        except Exception:
            pass

    if audio is None:
        try:
            sr, raw_audio = scipy.io.wavfile.read(path_str)
            if raw_audio.dtype == np.int16:
                audio = raw_audio.astype(np.float32) / 32768.0
            elif raw_audio.dtype == np.int32:
                audio = raw_audio.astype(np.float32) / 2147483648.0
            elif raw_audio.dtype == np.uint8:
                audio = (raw_audio.astype(np.float32) - 128.0) / 128.0
            else:
                audio = raw_audio.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file '{path_str}': {e}")

    if mono and audio.ndim > 1:
        audio = np.mean(audio, axis=-1)

    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if normalize:
        audio = normalize_audio(audio)

    return audio.astype(np.float32), int(sr)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D audio."""
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return scipy.signal.resample_poly(audio, up, down).astype(np.float32)


def normalize_audio(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio signal."""
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        return (audio / max_val) * peak
    return audio


def save_audio(
    file_path: Union[str, Path],
    audio: np.ndarray,
    sr: int = 16000,
) -> None:
    """Save audio numpy array to WAV."""
    out_path = Path(file_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
        sf.write(str(out_path), audio, sr, subtype="PCM_16")
    except Exception:
        scaled = np.clip(audio, -1.0, 1.0)
        int16_audio = (scaled * 32767.0).astype(np.int16)
        scipy.io.wavfile.write(str(out_path), sr, int16_audio)


def generate_synthesized_piano_tone(
    midi_pitch: int,
    duration: float = 1.0,
    sr: int = 16000,
    amplitude: float = 0.8,
) -> np.ndarray:
    """Synthesize acoustic piano tone with harmonic envelope."""
    freq = 440.0 * (2.0 ** ((midi_pitch - 69.0) / 12.0))
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    attack_samples = max(1, int(0.005 * sr))
    envelope = np.exp(-3.0 * t / duration)
    if len(t) > attack_samples:
        envelope[:attack_samples] = np.linspace(0, 1.0, attack_samples)

    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.50 * np.sin(2 * np.pi * 2 * freq * t)
    signal += 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    signal += 0.12 * np.sin(2 * np.pi * 4 * freq * t)
    signal += 0.06 * np.sin(2 * np.pi * 5 * freq * t)

    audio = signal * envelope * amplitude
    return normalize_audio(audio.astype(np.float32))
