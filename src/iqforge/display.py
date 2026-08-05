"""Terminal spectrogram and power-over-time plot.

Drawing uses the Unicode upper half block (`▀`): each character cell carries two
vertical pixels — the top half painted in the foreground colour, the bottom half
in the background colour. That works in terminals without a graphics protocol.
"""

from __future__ import annotations

import numpy as np
from rich.color import Color
from rich.console import Group, RenderableType
from rich.style import Style
from rich.text import Text
from scipy import signal

from iqforge.io import Recording

#: RGB anchors sampled at 32 points from matplotlib's viridis palette.
#: Embedded so there is no matplotlib dependency at runtime.
VIRIDIS: tuple[tuple[int, int, int], ...] = (
    (68, 1, 84), (71, 13, 96), (72, 24, 106), (72, 35, 116),
    (71, 46, 124), (69, 56, 130), (66, 65, 134), (62, 74, 137),
    (58, 84, 140), (54, 93, 141), (50, 101, 142), (46, 109, 142),
    (43, 117, 142), (40, 125, 142), (37, 132, 142), (34, 140, 141),
    (31, 148, 140), (30, 156, 137), (32, 163, 134), (37, 171, 130),
    (46, 179, 124), (58, 186, 118), (72, 193, 110), (88, 199, 101),
    (108, 205, 90), (127, 211, 78), (147, 215, 65), (168, 219, 52),
    (192, 223, 37), (213, 226, 26), (234, 229, 26), (253, 231, 37),
)  # fmt: skip

UPPER_HALF_BLOCK = "▀"
SPARK_BLOCKS = " ▁▂▃▄▅▆▇█"

#: Percentiles used for the lower and upper bound of the colour scale.
CLIP_PERCENTILES = (5.0, 99.0)

FREQ_LABEL_WIDTH = 9


def colormap(values: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to viridis-like RGB colours.

    Args:
        values: Array of any shape with values in [0, 1].

    Returns:
        A uint8 RGB array of shape `values.shape + (3,)`.
    """
    anchors = np.asarray(VIRIDIS, dtype=np.float64)
    pos = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    lo = np.floor(pos).astype(np.intp)
    hi = np.minimum(lo + 1, len(anchors) - 1)
    frac = (pos - lo)[..., None]
    rgb = anchors[lo] * (1.0 - frac) + anchors[hi] * frac
    return np.round(rgb).astype(np.uint8)


def compute_spectrogram(
    samples: np.ndarray, sample_rate: float, nfft: int = 1024
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the spectrogram in dB via STFT.

    Args:
        samples: Complex samples.
        sample_rate: Sample rate in Hz.
        nfft: FFT length.

    Returns:
        `(freqs, times, power_db)`. `freqs` is ascending and relative to the
        centre frequency (Hz); `times` is in seconds; `power_db` has shape
        `(len(freqs), len(times))`.

    Raises:
        ValueError: If there are fewer samples than `nfft`.
    """
    if samples.size < nfft:
        raise ValueError(
            f"A spectrogram needs at least {nfft} samples, got {samples.size}. "
            f"Lower --nfft or raise --samples."
        )

    freqs, times, zxx = signal.stft(
        samples,
        fs=sample_rate,
        nperseg=nfft,
        noverlap=nfft // 2,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    freqs = np.fft.fftshift(freqs)
    zxx = np.fft.fftshift(zxx, axes=0)
    power_db = 20.0 * np.log10(np.abs(zxx) + 1e-12)
    return freqs, times, power_db


def _bucket_starts(n_in: int, n_out: int) -> np.ndarray:
    """Start indices that divide an axis of length `n_in` into `n_out` buckets."""
    return (np.arange(n_out) * n_in) // n_out


def _pool_max(arr: np.ndarray, n_out: int, axis: int) -> np.ndarray:
    """Reduce an axis to `n_out`, taking the maximum when shrinking.

    Maximum rather than mean, because a narrowband tone averaged together with
    its neighbouring noise bins disappears; the maximum keeps it visible.
    """
    n_in = arr.shape[axis]
    if n_out >= n_in:
        idx = np.clip((np.arange(n_out) * n_in) // n_out, 0, n_in - 1)
        return np.take(arr, idx, axis=axis)
    return np.maximum.reduceat(arr, _bucket_starts(n_in, n_out), axis=axis)


def _pool_mean_axis(values: np.ndarray, n_out: int) -> np.ndarray:
    """Reduce a one-dimensional axis to `n_out`, taking each bucket's mean."""
    n_in = values.size
    if n_out >= n_in:
        idx = np.clip((np.arange(n_out) * n_in) // n_out, 0, n_in - 1)
        return values[idx]
    starts = _bucket_starts(n_in, n_out)
    sums = np.add.reduceat(values, starts)
    counts = np.diff(np.append(starts, n_in))
    return sums / counts


def _format_offset_mhz(hz: float) -> str:
    """Format an offset from the centre frequency in MHz, with a sign."""
    return f"{hz / 1e6:+.3f}"


def spectrogram_panel(
    freqs: np.ndarray,
    times: np.ndarray,
    power_db: np.ndarray,
    *,
    width: int,
    height: int,
) -> Text:
    """Render the spectrogram as text drawn with half-block characters.

    Args:
        freqs: Frequency axis relative to the centre frequency (Hz), ascending.
        times: Time axis in seconds.
        power_db: Power in dB, shaped `(freq, time)`.
        width: Character width of the plot area, excluding axis labels.
        height: Character height of the plot area; vertical resolution is 2×.

    Returns:
        A coloured `rich.text.Text`.
    """
    # Flip the frequency axis so the top row is the highest frequency.
    flipped = power_db[::-1, :]
    freqs_desc = freqs[::-1]

    pixels = _pool_max(_pool_max(flipped, 2 * height, axis=0), width, axis=1)
    pixel_freqs = _pool_mean_axis(freqs_desc, 2 * height)

    # Percentiles are computed AFTER pooling, i.e. over the values actually
    # printed. Computed from the full-resolution array instead, nearly every
    # max-pooled pixel would sit above the upper bound and the image would come
    # out uniform.
    vmin, vmax = np.percentile(pixels, CLIP_PERCENTILES)
    if vmax <= vmin:
        vmax = vmin + 1.0
    rgb = colormap((pixels - vmin) / (vmax - vmin))

    text = Text()
    for row in range(height):
        top, bottom = rgb[2 * row], rgb[2 * row + 1]
        label = _format_offset_mhz((pixel_freqs[2 * row] + pixel_freqs[2 * row + 1]) / 2.0)
        text.append(f"{label:>{FREQ_LABEL_WIDTH - 1}} ", style="dim")
        for col in range(width):
            style = Style(
                color=Color.from_rgb(*top[col].tolist()),
                bgcolor=Color.from_rgb(*bottom[col].tolist()),
            )
            text.append(UPPER_HALF_BLOCK, style=style)
        text.append("\n")

    text.append(_time_axis(times, width))
    return text


def _time_axis(times: np.ndarray, width: int) -> Text:
    """Draw the horizontal time axis: tick marks plus labels."""
    n_ticks = max(2, min(6, width // 12))
    cols = [round(i * (width - 1) / (n_ticks - 1)) for i in range(n_ticks)]
    labels = [f"{times[round(c * (times.size - 1) / (width - 1))]:.3f}" for c in cols]

    ruler = [" "] * width
    for c in cols:
        ruler[c] = "┬"
    axis = Text(" " * FREQ_LABEL_WIDTH + "".join(ruler) + "\n", style="dim")

    line = [" "] * (width + len(labels[-1]))
    for c, label in zip(cols, labels, strict=True):
        pos = min(max(c - len(label) // 2, 0), width - 1)
        line[pos : pos + len(label)] = list(label)
    axis.append(" " * FREQ_LABEL_WIDTH + "".join(line).rstrip() + " s\n", style="dim")
    return axis


def power_panel(samples: np.ndarray, times: np.ndarray, width: int) -> Text:
    """Draw power over time as a one-line sparkline.

    Args:
        samples: Complex samples.
        times: The spectrogram's time axis (used only for length alignment).
        width: Sparkline width in characters.

    Returns:
        A coloured `rich.text.Text`.
    """
    del times
    usable = (samples.size // width) * width
    if usable == 0:
        return Text()
    power = np.abs(samples[:usable].reshape(width, -1)) ** 2
    power_db = 10.0 * np.log10(power.mean(axis=1) + 1e-12)

    lo, hi = power_db.min(), power_db.max()
    if hi <= lo:
        hi = lo + 1.0
    levels = np.clip(
        ((power_db - lo) / (hi - lo) * (len(SPARK_BLOCKS) - 1)).round().astype(int),
        0,
        len(SPARK_BLOCKS) - 1,
    )

    text = Text(f"{'power':>{FREQ_LABEL_WIDTH - 1}} ", style="dim")
    text.append("".join(SPARK_BLOCKS[i] for i in levels), style="bright_cyan")
    text.append(f"\n{'':>{FREQ_LABEL_WIDTH}}{lo:.1f} … {hi:.1f} dB\n", style="dim")
    return text


def render_inspect(
    rec: Recording,
    samples: np.ndarray,
    start: int,
    nfft: int,
    *,
    width: int,
    height: int,
) -> RenderableType:
    """Build the whole `iqforge inspect` output: header, spectrogram, power.

    Args:
        rec: The opened recording.
        samples: The complex samples to display.
        start: Index of the first sample within the recording.
        nfft: FFT length.
        width: Total available character width.
        height: Character height of the spectrogram.

    Returns:
        A renderable object for `rich`.
    """
    freqs, times, power_db = compute_spectrogram(samples, rec.sample_rate, nfft)
    times = times + start / rec.sample_rate
    # One column is left blank: a full-width line wraps in some terminals (and
    # in rich's legacy Windows mode).
    plot_width = max(16, width - FREQ_LABEL_WIDTH - 1)

    center_mhz = (rec.center_frequency or 0.0) / 1e6
    header = Text.assemble(
        (rec.meta_path.name, "bold"),
        ("  centre ", "dim"),
        (f"{center_mhz:.6g} MHz", "cyan"),
        ("  rate ", "dim"),
        (f"{rec.sample_rate / 1e6:.6g} MS/s", "cyan"),
        ("  samples ", "dim"),
        (f"{start}…{start + samples.size}", "cyan"),
        ("  nfft ", "dim"),
        (f"{nfft}", "cyan"),
        ("\nvertical: offset from centre (MHz)   horizontal: time (s)\n", "dim"),
    )
    return Group(
        header,
        spectrogram_panel(freqs, times, power_db, width=plot_width, height=height),
        power_panel(samples, times, plot_width),
    )
