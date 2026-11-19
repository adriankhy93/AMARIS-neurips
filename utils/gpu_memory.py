"""Small helpers for recording peak PyTorch GPU allocator memory."""

import torch


def reset_peak_gpu_memory() -> None:
    """Reset per-device peak counters for the current pipeline stage."""
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        try:
            torch.cuda.reset_peak_memory_stats(index)
        except RuntimeError:
            # CPU tests may mock is_available() without initializing a device.
            return


def peak_gpu_memory_bytes() -> list[dict[str, int]]:
    """Return current and peak allocator memory for every visible GPU, in bytes."""
    if not torch.cuda.is_available():
        return []
    readings = []
    for index in range(torch.cuda.device_count()):
        try:
            readings.append(
                {
                    "device_index": index,
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                }
            )
        except RuntimeError:
            return []
    return readings
