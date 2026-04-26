"""Device detection and GPU memory probes.

Designed to degrade gracefully on Mac MPS / CPU (where peak-memory APIs aren't
available), returning None instead of crashing.
"""

import torch


def detect_device():
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def reset_peak_memory(device):
    """Reset CUDA peak-memory tracking. No-op on MPS/CPU."""
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device):
    """Return peak GPU memory in GB, or None if not measurable on this device."""
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / 1e9
    return None


def device_label(device):
    """Human-readable hardware label, e.g. 'cuda:NVIDIA A100' or 'mps' or 'cpu'."""
    if device.startswith("cuda"):
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return device
