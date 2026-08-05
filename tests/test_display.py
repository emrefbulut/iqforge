"""Tests for iqforge.display. All synthetic, no network access."""

from __future__ import annotations

import numpy as np
import pytest
from rich.style import Style

from iqforge.display import (
    UPPER_HALF_BLOCK,
    VIRIDIS,
    _pool_max,
    _pool_mean_axis,
    colormap,
    compute_spectrogram,
    spectrogram_panel,
)

SAMPLE_RATE = 1_024_000.0


def tone(offset_hz: float, n: int = 32_768, amplitude: float = 1.0) -> np.ndarray:
    """Generate a pure complex tone at the given offset."""
    t = np.arange(n) / SAMPLE_RATE
    return (amplitude * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def test_spectrogram_axes_are_centred_and_ordered() -> None:
    """The frequency axis spans -fs/2…+fs/2 and is ascending."""
    freqs, times, power_db = compute_spectrogram(tone(0.0), SAMPLE_RATE, nfft=1024)

    assert freqs.size == 1024
    assert np.all(np.diff(freqs) > 0)
    assert freqs[0] == pytest.approx(-SAMPLE_RATE / 2)
    assert power_db.shape == (freqs.size, times.size)
    assert times[0] > 0 and times[-1] < 32_768 / SAMPLE_RATE


@pytest.mark.parametrize("offset", [100_000.0, -200_000.0, 250_000.0, 0.0])
def test_tone_lands_on_the_expected_bin(offset: float) -> None:
    """A known tone lands in the right bin, sign included."""
    freqs, _, power_db = compute_spectrogram(tone(offset), SAMPLE_RATE, nfft=1024)
    peak = freqs[int(np.argmax(power_db.mean(axis=1)))]

    bin_width = freqs[1] - freqs[0]
    assert peak == pytest.approx(offset, abs=bin_width)


def test_short_input_gives_actionable_error() -> None:
    """Fewer samples than nfft must say what to do about it."""
    with pytest.raises(ValueError) as exc:
        compute_spectrogram(tone(0.0, n=100), SAMPLE_RATE, nfft=1024)
    message = str(exc.value)
    assert "1024" in message and "--nfft" in message


def test_pool_max_preserves_narrowband_peaks() -> None:
    """Pooling takes the maximum when shrinking; a single-bin peak survives."""
    arr = np.full((64, 8), -60.0)
    arr[37, :] = 0.0
    pooled = _pool_max(arr, 16, axis=0)

    assert pooled.shape == (16, 8)
    assert pooled.max() == pytest.approx(0.0)
    # The peak must land in exactly one pooled row and not bleed into neighbours.
    assert (pooled.max(axis=1) > -60.0).sum() == 1


def test_pool_max_upsamples_when_target_is_larger() -> None:
    """A target longer than the input is filled by nearest neighbour."""
    arr = np.arange(4, dtype=float).reshape(4, 1)
    assert _pool_max(arr, 8, axis=0).ravel().tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_pool_mean_axis_matches_bucket_means() -> None:
    """Axis pooling returns the mean of each bucket."""
    values = np.arange(10, dtype=float)
    assert _pool_mean_axis(values, 5).tolist() == [0.5, 2.5, 4.5, 6.5, 8.5]


def test_colormap_endpoints_and_shape() -> None:
    """The colour map returns the viridis endpoints and preserves shape."""
    assert colormap(np.array(0.0)).tolist() == list(VIRIDIS[0])
    assert colormap(np.array(1.0)).tolist() == list(VIRIDIS[-1])
    assert colormap(np.zeros((3, 4))).shape == (3, 4, 3)
    # Out-of-range values must clip rather than raise.
    assert colormap(np.array([-5.0, 5.0])).tolist() == [list(VIRIDIS[0]), list(VIRIDIS[-1])]


def test_panel_geometry_and_frequency_labels() -> None:
    """The panel has the requested rows and columns, top row highest frequency."""
    freqs, times, power_db = compute_spectrogram(tone(100_000.0), SAMPLE_RATE, nfft=1024)
    panel = spectrogram_panel(freqs, times, power_db, width=40, height=10)

    lines = panel.plain.rstrip("\n").split("\n")
    plot_rows = [ln for ln in lines if UPPER_HALF_BLOCK in ln]
    assert len(plot_rows) == 10
    assert all(ln.count(UPPER_HALF_BLOCK) == 40 for ln in plot_rows)

    labels = [float(ln.split()[0]) for ln in plot_rows]
    assert labels == sorted(labels, reverse=True), "the top row must be the highest frequency"


def test_panel_colours_vary_when_signal_is_present() -> None:
    """A panel over real signal must not come out uniform.

    If the percentiles are not computed after pooling, every pixel clips to the
    upper bound and the panel collapses to one colour; this test catches that
    regression.
    """
    freqs, times, power_db = compute_spectrogram(tone(100_000.0), SAMPLE_RATE, nfft=1024)
    panel = spectrogram_panel(freqs, times, power_db, width=40, height=10)

    # Axis labels carry a plain string style ("dim"); coloured cells carry a Style.
    cells = [s.style for s in panel.spans if isinstance(s.style, Style)]
    colours = {(s.color.triplet, s.bgcolor.triplet) for s in cells}
    assert len(cells) == 40 * 10
    assert len(colours) > 1, "the spectrogram collapsed to a single colour"
