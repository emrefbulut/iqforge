"""Kaydı sabit uzunlukta kayan pencerelere bölme ve temsil dönüşümleri."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from iqforge.io import IQForgeError, Recording

#: Desteklenen temsil biçimleri (`--repr`).
REPRESENTATIONS = ("iq2ch", "complex", "magphase")

#: Tek seferde belleğe alınacak pencere sayısı.
DEFAULT_BATCH_WINDOWS = 512


def window_count(num_samples: int, window: int, stride: int) -> int:
    """Kaydın kaç tam pencereye bölüneceğini verir.

    Sondaki eksik pencere atılır; padding yapılmaz.

    Args:
        num_samples: Kayıttaki toplam örnek sayısı.
        window: Pencere uzunluğu (örnek).
        stride: Pencereler arası adım (örnek).

    Returns:
        `floor((num_samples - window) / stride) + 1`, negatifse 0.
    """
    if num_samples < window:
        return 0
    return (num_samples - window) // stride + 1


def window_starts(num_samples: int, window: int, stride: int) -> np.ndarray:
    """Pencerelerin başlangıç örnek indislerini verir."""
    return np.arange(window_count(num_samples, window, stride), dtype=np.int64) * stride


def validate_window_params(window: int, stride: int) -> None:
    """Pencere parametrelerini doğrular.

    Raises:
        IQForgeError: Pencere veya adım pozitif değilse.
    """
    if window <= 0:
        raise IQForgeError(f"--window pozitif olmalı, {window} verildi.")
    if stride <= 0:
        raise IQForgeError(f"--stride pozitif olmalı, {stride} verildi.")


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Her pencereyi ayrı ayrı birim güce normalize eder.

    `x = x / sqrt(mean(|x|^2))`. Sıfır güçlü pencerelerde bölme yapılmaz,
    pencere sıfır olarak döner.

    Args:
        windows: `(n, window)` complex64 dizi.

    Returns:
        Aynı şekilde normalize edilmiş dizi.
    """
    rms = np.sqrt(np.mean(np.abs(windows) ** 2, axis=1, keepdims=True))
    scale = np.divide(1.0, rms, out=np.zeros_like(rms), where=rms > 0)
    return (windows * scale).astype(np.complex64)


def to_representation(windows: np.ndarray, representation: str) -> np.ndarray:
    """Kompleks pencereleri istenen temsile çevirir.

    Args:
        windows: `(n, window)` complex64 dizi.
        representation: `iq2ch`, `complex` veya `magphase`.

    Returns:
        `iq2ch` ve `magphase` için `(n, 2, window)` float32,
        `complex` için `(n, window)` complex64.

    Raises:
        IQForgeError: Temsil tanınmıyorsa.
    """
    if representation == "complex":
        return windows.astype(np.complex64)
    if representation == "iq2ch":
        return np.stack([windows.real, windows.imag], axis=1).astype(np.float32)
    if representation == "magphase":
        return np.stack([np.abs(windows), np.angle(windows)], axis=1).astype(np.float32)
    raise IQForgeError(
        f"Bilinmeyen temsil '{representation}'. Desteklenenler: {', '.join(REPRESENTATIONS)}."
    )


def iter_window_batches(
    rec: Recording,
    window: int,
    stride: int,
    indices: np.ndarray | None = None,
    batch_windows: int = DEFAULT_BATCH_WINDOWS,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Kaydı pencereler halinde, parça parça okur.

    Kaydın tamamı belleğe alınmaz: her partide yalnızca o partinin kapsadığı
    örnek aralığı okunur.

    Args:
        rec: Açılmış kayıt.
        window: Pencere uzunluğu.
        stride: Adım.
        indices: Yalnızca bu pencere indisleri üretilsin (None ise tümü).
        batch_windows: Parti başına pencere sayısı.

    Yields:
        `(idx, batch)` — `idx` pencere indisleri, `batch` `(k, window)` complex64.
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
