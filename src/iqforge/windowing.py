"""Cutting a recording into fixed-length sliding windows, and representations."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from iqforge.io import IQForgeError, Recording

#: Supported representations (`--repr`).
REPRESENTATIONS = ("iq2ch", "complex", "magphase")

#: How many windows to hold in memory at once.
DEFAULT_BATCH_WINDOWS = 512


def window_count(num_samples: int, window: int, stride: int) -> int:
    """Return how many whole windows a recording yields.

    The trailing partial window is dropped; nothing is padded.

    Args:
        num_samples: Total samples in the recording.
        window: Window length in samples.
        stride: Step between consecutive windows, in samples.

    Returns:
        `floor((num_samples - window) / stride) + 1`, or 0 if negative.
    """
    if num_samples < window:
        return 0
    return (num_samples - window) // stride + 1


def window_starts(num_samples: int, window: int, stride: int) -> np.ndarray:
    """Return the starting sample index of every window."""
    return np.arange(window_count(num_samples, window, stride), dtype=np.int64) * stride


def validate_window_params(window: int, stride: int) -> None:
    """Validate the windowing parameters.

    Raises:
        IQForgeError: If window or stride is not positive.
    """
    if window <= 0:
        raise IQForgeError(f"--window must be positive, got {window}.")
    if stride <= 0:
        raise IQForgeError(f"--stride must be positive, got {stride}.")


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Normalize each window separately to unit power.

    `x = x / sqrt(mean(|x|^2))`. Zero-power windows are not divided; they are
    returned as zeros.

    Args:
        windows: An `(n, window)` complex64 array.

    Returns:
        The normalized array, same shape.
    """
    rms = np.sqrt(np.mean(np.abs(windows) ** 2, axis=1, keepdims=True))
    scale = np.divide(1.0, rms, out=np.zeros_like(rms), where=rms > 0)
    return (windows * scale).astype(np.complex64)


def to_representation(windows: np.ndarray, representation: str) -> np.ndarray:
    """Convert complex windows into the requested representation.

    Args:
        windows: An `(n, window)` complex64 array.
        representation: One of `iq2ch`, `complex`, `magphase`.

    Returns:
        `(n, 2, window)` float32 for `iq2ch` and `magphase`,
        `(n, window)` complex64 for `complex`.

    Raises:
        IQForgeError: If the representation is not recognised.
    """
    if representation == "complex":
        return windows.astype(np.complex64)
    if representation == "iq2ch":
        return np.stack([windows.real, windows.imag], axis=1).astype(np.float32)
    if representation == "magphase":
        return np.stack([np.abs(windows), np.angle(windows)], axis=1).astype(np.float32)
    raise IQForgeError(
        f"Unknown representation '{representation}'. Supported: {', '.join(REPRESENTATIONS)}."
    )


def iter_window_batches(
    rec: Recording,
    window: int,
    stride: int,
    indices: np.ndarray | None = None,
    batch_windows: int = DEFAULT_BATCH_WINDOWS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Read a recording window by window, in batches.

    The whole recording is never loaded at once: each batch reads only the
    sample range it covers.

    Args:
        rec: The opened recording.
        window: Window length.
        stride: Step between windows.
        indices: Only produce these window indices (None means all of them).
        batch_windows: Windows per batch.

    Yields:
        `(idx, batch)` where `idx` holds the window indices and `batch` is a
        `(k, window)` complex64 array.
    """
    starts = window_starts(rec.num_samples, window, stride)
    selected = np.arange(starts.size) if indices is None else np.asarray(indices, dtype=np.int64)

    for begin in range(0, selected.size, batch_windows):
        chunk = selected[begin : begin + batch_windows]
        if chunk.size == 0:
            continue
        first, last = int(starts[chunk[0]]), int(starts[chunk[-1]])
        block = rec.read(start=first, count=last + window - first)
        offsets = starts[chunk] - first
        batch = np.empty((chunk.size, window), dtype=np.complex64)
        for row, offset in enumerate(offsets):
            batch[row] = block[offset : offset + window]
        yield chunk, batch
