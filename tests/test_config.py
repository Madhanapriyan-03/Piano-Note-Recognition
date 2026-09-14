"""
Tests for configuration loading and device management.
"""

from pathlib import Path
import pytest
import torch

from src.utils.config import get_device, load_config


def test_load_default_config():
    cfg = load_config()
    assert hasattr(cfg, "audio")
    assert hasattr(cfg, "model")
    assert hasattr(cfg, "training")
    assert hasattr(cfg, "dataset")
    assert hasattr(cfg, "inference")
    assert hasattr(cfg, "paths")

    assert cfg.audio.sample_rate == 16000
    assert cfg.model.num_pitches == 88
    assert cfg.model.min_midi == 21
    assert cfg.model.max_midi == 108


def test_config_overrides():
    overrides = {
        "training": {"epochs": 99, "batch_size": 32},
        "audio": {"sample_rate": 22050},
    }
    cfg = load_config(overrides=overrides)
    assert cfg.training.epochs == 99
    assert cfg.training.batch_size == 32
    assert cfg.audio.sample_rate == 22050
    assert cfg.model.num_pitches == 88


def test_get_device():
    cpu_device = get_device("cpu", verbose=False)
    assert cpu_device.type == "cpu"

    auto_device = get_device("auto", verbose=False)
    assert isinstance(auto_device, torch.device)
