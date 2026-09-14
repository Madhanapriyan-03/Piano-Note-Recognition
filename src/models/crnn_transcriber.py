"""
CRNN (Convolutional Recurrent Neural Network) Architecture for Automatic Piano Transcription.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    2D Convolution Block with Batch Normalization, Activation, and Frequency-only Pooling.
    Frequency pooling preserves full temporal resolution (frame-level precision).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        pool_freq: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()
        self.pool_freq = pool_freq
        if pool_freq > 1:
            self.pool = nn.MaxPool2d(kernel_size=(pool_freq, 1), stride=(pool_freq, 1))
        else:
            self.pool = nn.Identity()
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class CRNNTranscriber(nn.Module):
    """
    Deep Learning CRNN Architecture for Automatic Piano Transcription.

    Pipeline:
      Log-Mel Spectrogram [B, 1, F, T]
            ↓
      2D-CNN Feature Extractor (4 Conv blocks with frequency pooling)
            ↓
      Reshape to Temporal Sequence [B, T, D_feat]
            ↓
      2-layer Bidirectional LSTM [B, T, 2 * H]
            ↓
      Multi-task Heads:
        1. Onset Detection Head -> [B, T, 88] (logits & probabilities)
        2. Frame Activation Head -> [B, T, 88] (logits & probabilities)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_mels: int = 229,
        num_pitches: int = 88,
        cnn_channels: Optional[List[int]] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.num_pitches = num_pitches
        channels = cnn_channels or [32, 64, 128, 128]

        conv_layers = []
        curr_in = in_channels
        curr_freq = n_mels

        conv_layers.append(ConvBlock(curr_in, channels[0], pool_freq=2, dropout=0.1))
        curr_freq = curr_freq // 2

        conv_layers.append(ConvBlock(channels[0], channels[1], pool_freq=2, dropout=0.15))
        curr_freq = curr_freq // 2

        conv_layers.append(ConvBlock(channels[1], channels[2], pool_freq=2, dropout=0.2))
        curr_freq = curr_freq // 2

        if len(channels) > 3:
            conv_layers.append(ConvBlock(channels[2], channels[3], pool_freq=1, dropout=0.2))
            feat_channels = channels[3]
        else:
            feat_channels = channels[2]

        self.cnn = nn.Sequential(*conv_layers)
        self.flattened_dim = feat_channels * curr_freq

        self.proj = nn.Sequential(
            nn.Linear(self.flattened_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        rnn_out_dim = lstm_hidden * 2

        self.onset_head = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_pitches),
        )

        self.frame_head = nn.Sequential(
            nn.Linear(rnn_out_dim + num_pitches, 128),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_pitches),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        feat = self.cnn(x)
        b, c, f_prime, t = feat.shape

        feat = feat.permute(0, 3, 1, 2).contiguous()
        feat = feat.view(b, t, c * f_prime)

        proj_feat = self.proj(feat)

        lstm_out, _ = self.lstm(proj_feat)

        onset_logits = self.onset_head(lstm_out)
        onset_probs = torch.sigmoid(onset_logits)

        frame_input = torch.cat([lstm_out, onset_logits.detach()], dim=-1)
        frame_logits = self.frame_head(frame_input)
        frame_probs = torch.sigmoid(frame_logits)

        return {
            "onset_logits": onset_logits,
            "frame_logits": frame_logits,
            "onset_probs": onset_probs,
            "frame_probs": frame_probs,
        }


def build_model(config) -> CRNNTranscriber:
    audio_cfg = config.audio if hasattr(config, "audio") else config
    model_cfg = config.model if hasattr(config, "model") else config

    n_mels = getattr(audio_cfg, "n_mels", 229)
    num_pitches = getattr(model_cfg, "num_pitches", 88)
    cnn_channels = getattr(model_cfg, "cnn_channels", [32, 64, 128, 128])
    lstm_hidden = getattr(model_cfg, "lstm_hidden", 128)
    lstm_layers = getattr(model_cfg, "lstm_layers", 2)
    dropout = getattr(model_cfg, "dropout", 0.25)

    model = CRNNTranscriber(
        in_channels=1,
        n_mels=n_mels,
        num_pitches=num_pitches,
        cnn_channels=cnn_channels,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )
    return model
