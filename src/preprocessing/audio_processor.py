"""
Audio preprocessing: STFT and Log-Mel Spectrogram computation.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from src.utils.audio_io import load_audio, normalize_audio


class AudioProcessor:
    """
    Unified Audio Preprocessing pipeline.

    Computes standardized Log-Mel Spectrograms and segments audio for training and inference.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: Optional[int] = 2048,
        window: str = "hann",
        n_mels: int = 229,
        f_min: float = 30.0,
        f_max: float = 8000.0,
        log_eps: float = 1e-6,
        normalize_audio: bool = True,
    ):
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.window = str(window)
        self.n_mels = int(n_mels)
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.log_eps = float(log_eps)
        self.normalize = bool(normalize_audio)

        # Frame rate in frames per second
        self.frame_rate = float(self.sample_rate) / float(self.hop_length)

    @classmethod
    def from_config(cls, config) -> "AudioProcessor":
        """Instantiate AudioProcessor directly from configuration object."""
        cfg = config.audio if hasattr(config, "audio") else config
        return cls(
            sample_rate=getattr(cfg, "sample_rate", 16000),
            n_fft=getattr(cfg, "n_fft", 2048),
            hop_length=getattr(cfg, "hop_length", 512),
            win_length=getattr(cfg, "win_length", 2048),
            window=getattr(cfg, "window", "hann"),
            n_mels=getattr(cfg, "n_mels", 229),
            f_min=getattr(cfg, "f_min", 30.0),
            f_max=getattr(cfg, "f_max", 8000.0),
            log_eps=getattr(cfg, "log_eps", 1e-6),
            normalize_audio=getattr(cfg, "normalize_audio", True),
        )

    def process_file(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Load audio file and compute log-mel spectrogram.

        Returns:
            mel_spec: 2D numpy array [n_mels, T_frames]
            audio: 1D normalized audio array
            duration: total duration in seconds
        """
        audio, sr = load_audio(
            file_path,
            target_sr=self.sample_rate,
            mono=True,
            normalize=self.normalize,
        )
        mel_spec = self.compute_mel_spectrogram(audio)
        duration = len(audio) / self.sample_rate
        return mel_spec, audio, duration

    def compute_mel_spectrogram(self, audio: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute Log-Mel Spectrogram from raw 1D audio waveform.

        Args:
            audio: 1D numpy array or torch.Tensor of audio samples.

        Returns:
            mel_spectrogram: 2D numpy array with shape [n_mels, T_frames].
        """
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy().astype(np.float32)
        else:
            audio_np = np.asarray(audio, dtype=np.float32)

        if self.normalize:
            audio_np = normalize_audio(audio_np)

        try:
            import librosa
            # Standard Mel-filterbank spectrogram
            mel_basis = librosa.feature.melspectrogram(
                y=audio_np,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                n_mels=self.n_mels,
                fmin=self.f_min,
                fmax=self.f_max,
                power=2.0,
                center=True,
                pad_mode="reflect",
            )
            # Log compression
            log_mel = np.log(np.maximum(mel_basis, self.log_eps))
            # Standardize / zero-mean unit-variance per utterance or scale
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
            return log_mel.astype(np.float32)
        except Exception:
            # Fallback using scipy STFT and standard triangular Mel filterbank
            return self._compute_mel_scipy(audio_np)

    def _compute_mel_scipy(self, audio: np.ndarray) -> np.ndarray:
        """SciPy-based Mel spectrogram fallback."""
        import scipy.signal

        # Windowed STFT
        window_func = scipy.signal.get_window(self.window, self.win_length)
        _, _, Zxx = scipy.signal.stft(
            audio,
            fs=self.sample_rate,
            window=window_func,
            nperseg=self.win_length,
            noverlap=self.win_length - self.hop_length,
            nfft=self.n_fft,
            boundary="zeros",
            padded=True,
        )
        power_spec = np.abs(Zxx) ** 2  # Shape: [1 + n_fft//2, T_frames]

        # Mel filterbank
        mel_filters = self._create_mel_filterbank(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
        )
        mel_spec = np.dot(mel_filters, power_spec)
        log_mel = np.log(np.maximum(mel_spec, self.log_eps))
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        return log_mel.astype(np.float32)

    def _create_mel_filterbank(
        self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
    ) -> np.ndarray:
        """Generate Mel triangular filterbank matrix [n_mels, 1 + n_fft//2]."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        num_fft_bins = 1 + n_fft // 2
        fbank = np.zeros((n_mels, num_fft_bins), dtype=np.float32)

        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                if f_m > f_m_minus and k < num_fft_bins:
                    fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus > f_m and k < num_fft_bins:
                    fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        return fbank

    def time_to_frame(self, time_sec: float) -> int:
        """Convert a time in seconds to the nearest spectrogram frame index."""
        return int(round(time_sec * self.frame_rate))

    def frame_to_time(self, frame_idx: int) -> float:
        """Convert a spectrogram frame index to time in seconds."""
        return float(frame_idx) / self.frame_rate

    def slice_spectrogram(
        self,
        mel_spec: np.ndarray,
        segment_frames: int,
        hop_frames: int,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Slice full-length spectrogram [n_mels, Total_Frames] into fixed-frame windows.

        Returns:
            List of tuples: (window_spec [n_mels, segment_frames], start_frame, end_frame)
        """
        n_mels, total_frames = mel_spec.shape
        segments = []

        if total_frames <= segment_frames:
            # Pad with zeros along time dimension
            padded = np.zeros((n_mels, segment_frames), dtype=mel_spec.dtype)
            padded[:, :total_frames] = mel_spec
            segments.append((padded, 0, total_frames))
            return segments

        start = 0
        while start < total_frames:
            end = start + segment_frames
            if end > total_frames:
                # Last window: take last segment_frames
                start_adj = max(0, total_frames - segment_frames)
                segment = mel_spec[:, start_adj:total_frames]
                if segment.shape[1] < segment_frames:
                    padded = np.zeros((n_mels, segment_frames), dtype=mel_spec.dtype)
                    padded[:, :segment.shape[1]] = segment
                    segments.append((padded, start_adj, total_frames))
                else:
                    segments.append((segment, start_adj, total_frames))
                break
            else:
                segment = mel_spec[:, start:end]
                segments.append((segment, start, end))
                start += hop_frames

        return segments
