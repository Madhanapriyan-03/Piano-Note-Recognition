"""
Configuration loader and device management.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
import torch


class ConfigDict(dict):
    """Dictionary subclass supporting attribute-style dot access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            if isinstance(value, dict) and not isinstance(value, ConfigDict):
                value = ConfigDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"Configuration has no attribute '{key}'")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> ConfigDict:
    """
    Load YAML configuration file and apply any runtime overrides.
    """
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path.resolve()}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    if overrides:
        _deep_update(config_data, overrides)

    return _to_config_dict(config_data)


def _deep_update(target: dict, source: dict) -> None:
    """Recursively update a dictionary."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _to_config_dict(data: Any) -> Any:
    """Recursively convert standard dicts to ConfigDict."""
    if isinstance(data, dict):
        return ConfigDict({k: _to_config_dict(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [_to_config_dict(item) for item in data]
    return data


def get_device(device_str: str = "auto", verbose: bool = True) -> torch.device:
    """Detect and return the appropriate torch.device."""
    device_str = device_str.lower().strip()

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            device = torch.device(device_str)
        else:
            if verbose:
                print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif device_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            if verbose:
                print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if verbose:
        if device.type == "cuda":
            device_name = torch.cuda.get_device_name(device)
            vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            print(f"[Device] Using CUDA GPU: {device_name} ({vram_gb:.2f} GB VRAM)")
        elif device.type == "mps":
            print("[Device] Using Apple Silicon GPU (MPS)")
        else:
            print("[Device] Using CPU (Portable & Standard Execution)")

    return device
