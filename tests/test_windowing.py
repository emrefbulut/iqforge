"""sigkit.windowing testleri."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from sigkit.io import SigkitError
from sigkit.windowing import (
    iter_window_batches,
    normalize_windows,
    to_representation,
    validate_window_params,
    window_count,
    window_starts,
)


@pytest.mark.parametrize(
    ("num_samples", "window", "stride", "expected"),
    [
        (65_536, 1024, 512, 127),  # floor((65536-1024)/512)+1
        (1024, 1024, 512, 1),
        (1023, 1024, 512, 0),  # eksik pencere atılır, padding yok
        (2048, 1024, 1024, 2),
        (2047, 1024, 1024, 1),  # sondaki eksik pencere düşer
        (10_000, 1000, 1000, 10),
    ],
)
def test_window_count_matches_formula(
    num_samples: int, window: int, stride: int, expected: int
) -> None:
    """Pencere sayısı SPEC §5.2'deki formülle birebir aynı olmalı."""
    assert window_count(num_samples, window, stride) == expected


def test_window_starts_never_exceed_record() -> None:
    """Hiçbir pencere kaydın sonunu aşmamalı."""
    starts = window_starts(10_000, 1024, 512)
    assert starts[0] == 0
    assert starts[-1] + 1024 <= 10_000
    assert np.all(np.diff(starts) == 512)


def test_validate_window_params_rejects_non_positive() -> None:
    """Sıfır veya negatif pencere/adım açık hata vermeli."""
    with pytest.raises(SigkitError, match="--window"):
        validate_window_params(0, 512)
    with pytest.raises(SigkitError, match="--stride"):
        validate_window_params(1024, -1)


def test_normalize_gives_unit_power() -> None:
    """Her pencere ayrı ayrı birim güce normalize edilmeli."""
    rng = np.random.default_rng(0)
    windows = (rng.standard_normal((5, 256)) + 1j * rng.standard_normal((5, 256))).astype(
        np.complex64
    ) * np.array([[0.01], [1.0], [100.0], [3.0], [0.5]])

    out = normalize_windows(windows.astype(np.complex64))
    power = np.mean(np.abs(out) ** 2, axis=1)

    assert np.allclose(power, 1.0, atol=1e-5)
    assert out.dtype == np.complex64


def test_zero_power_window_returns_zeros_not_nan() -> None:
    """Sıfır güçlü pencerede bölme hatası olmamalı, sıfır dönmeli (SPEC §5.5)."""
    windows = np.zeros((2, 64), dtype=np.complex64)
    windows[1] = 1.0 + 0j

    out = normalize_windows(windows)

    assert np.all(out[0] == 0)
    assert not np.any(np.isnan(out))
    assert np.isclose(np.mean(np.abs(out[1]) ** 2), 1.0)


def test_iq2ch_channels_are_real_and_imaginary() -> None:
    """iq2ch kanal 0 = I, kanal 1 = Q olmalı."""
    windows = np.array([[1 + 2j, 3 - 4j]], dtype=np.complex64)
    out = to_representation(windows, "iq2ch")

    assert out.shape == (1, 2, 2)
    assert out.dtype == np.float32
    assert out[0, 0].tolist() == [1.0, 3.0]
    assert out[0, 1].tolist() == [2.0, -4.0]


def test_complex_representation_is_unchanged() -> None:
    """complex temsili ham hali korumalı."""
    windows = np.array([[1 + 2j, 3 - 4j]], dtype=np.complex64)
    out = to_representation(windows, "complex")

    assert out.shape == (1, 2)
    assert out.dtype == np.complex64
    assert np.array_equal(out, windows)


def test_magphase_round_trips_to_original() -> None:
    """magphase kanalları genlik ve faz olmalı; birleştirince orijinali vermeli."""
    rng = np.random.default_rng(1)
    windows = (rng.standard_normal((3, 32)) + 1j * rng.standard_normal((3, 32))).astype(
        np.complex64
    )
    out = to_representation(windows, "magphase")

    assert out.shape == (3, 2, 32)
    rebuilt = out[:, 0] * np.exp(1j * out[:, 1])
    assert np.allclose(rebuilt, windows, atol=1e-5)


def test_unknown_representation_is_explicit() -> None:
    """Tanınmayan temsil desteklenenleri listelemeli."""
    with pytest.raises(SigkitError) as exc:
        to_representation(np.zeros((1, 4), dtype=np.complex64), "iq3ch")
    assert "iq2ch" in str(exc.value) and "magphase" in str(exc.value)


def test_batched_reading_matches_direct_slicing(
    tmp_path: Path, make_recording: Callable[..., object], noise: Callable[..., np.ndarray]
) -> None:
    """Parça parça okuma, kaydın tamamını dilimlemeyle birebir aynı sonucu vermeli."""
    samples = noise(20_000, seed=3)
    rec = make_recording(tmp_path, samples)

    window, stride = 1024, 512
    starts = window_starts(rec.num_samples, window, stride)  # type: ignore[attr-defined]
    collected = np.zeros((starts.size, window), dtype=np.complex64)
    for chunk, batch in iter_window_batches(rec, window, stride, batch_windows=7):  # type: ignore[arg-type]
        collected[chunk] = batch

    for i, start in enumerate(starts):
        assert np.allclose(collected[i], samples[start : start + window], atol=1e-6)


def test_batched_reading_honours_index_selection(
    tmp_path: Path, make_recording: Callable[..., object], noise: Callable[..., np.ndarray]
) -> None:
    """Yalnızca istenen pencere indisleri üretilmeli."""
    samples = noise(8192, seed=4)
    rec = make_recording(tmp_path, samples)
    wanted = np.array([0, 3, 4, 9])

    produced = [
        int(i)
        for chunk, _ in iter_window_batches(rec, 1024, 512, indices=wanted, batch_windows=3)  # type: ignore[arg-type]
        for i in chunk
    ]

    assert produced == wanted.tolist()
