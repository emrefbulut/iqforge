"""Verify the terminal spectrogram against a matplotlib-drawn reference.

Uses exactly the same STFT as `iqforge inspect`
(`iqforge.display.compute_spectrogram`), draws the result to a PNG, and checks
numerically that both drawing paths show the same structure:

  * Is the reference tone's peak frequency exactly +100 kHz?
  * Does the time span where the BPSK/QPSK bursts' in-band power rises match the
    `core:sample_start` / `core:sample_count` in the annotation?

Both questions are answered separately on the full-resolution STFT (the data
matplotlib draws) and on the pooled pixel matrix printed to the terminal. If the
two paths disagree, one of them is broken.

Usage:
    python scripts/verify_spectrogram.py -o artifacts/spectrogram_bpsk_01.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from iqforge.display import (  # noqa: E402
    CLIP_PERCENTILES,
    _pool_max,
    _pool_mean_axis,
    compute_spectrogram,
)
from iqforge.io import Recording, load  # noqa: E402

REF_TONE_HZ = 100_000.0
#: Length of the burst edge ramp in the example recordings (scripts/make_example.py).
BURST_RAMP = 512
#: The terminal view's default character grid (COLUMNS=100, 24 rows).
TERM_WIDTH, TERM_HEIGHT = 91, 24

#: Block length of the power curve, in samples. Each block's time is its CENTRE.
POWER_BLOCK = 512


def power_curve(
    samples: np.ndarray, start: int, sample_rate: float, block: int = POWER_BLOCK
) -> tuple[np.ndarray, np.ndarray]:
    """Return block-averaged power over time.

    The time axis corresponds to block CENTRES. Using block starts instead
    shifts the curve left by half a block (0.25 ms by default) and puts a
    systematic error into the comparison with the annotation edges.

    Args:
        samples: Complex samples.
        start: Index of the first sample within the recording.
        sample_rate: Sample rate in Hz.
        block: Block length to average over, in samples.

    Returns:
        `(times, power_db)` - block centre times in seconds and power in dB.
    """
    usable = (samples.size // block) * block
    blocks = np.abs(samples[:usable].reshape(-1, block)) ** 2
    centres = (np.arange(blocks.shape[0]) + 0.5) * block + start
    return centres / sample_rate, 10.0 * np.log10(blocks.mean(axis=1) + 1e-12)


def half_power_edges(
    times: np.ndarray, power_db: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    """Find the step edges at the half-power point, at sub-block resolution.

    Floor and plateau levels are estimated by percentile; the threshold is their
    midpoint in linear power. Linear interpolation between the two neighbouring
    blocks that straddle the threshold gives a resolution finer than the block
    spacing.

    Returns:
        `(rising_edge, falling_edge)` in seconds, or `(None, None)` if no edge
        was found.
    """
    floor_db, plateau_db = np.percentile(power_db, 10), np.percentile(power_db, 90)
    threshold_db = 10.0 * np.log10((10 ** (floor_db / 10) + 10 ** (plateau_db / 10)) / 2.0)

    above = power_db > threshold_db
    if not above.any() or above.all():
        return None, None

    def _cross(i: int) -> float:
        """Interpolate the moment the threshold is crossed between i-1 and i."""
        y0, y1 = power_db[i - 1], power_db[i]
        frac = (threshold_db - y0) / (y1 - y0)
        return float(times[i - 1] + frac * (times[i] - times[i - 1]))

    rises = np.flatnonzero(~above[:-1] & above[1:]) + 1
    falls = np.flatnonzero(above[:-1] & ~above[1:]) + 1
    if rises.size == 0 or falls.size == 0:
        return None, None
    return _cross(int(rises[0])), _cross(int(falls[-1]))


def band_power_db(freqs: np.ndarray, power_db: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Return total power in a frequency band over time, in dB."""
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        raise ValueError(f"No STFT bin falls in the {lo:.0f}…{hi:.0f} Hz band.")
    return 10.0 * np.log10(np.sum(10.0 ** (power_db[mask] / 10.0), axis=0))


def active_span(values_db: np.ndarray, times: np.ndarray) -> tuple[float, float]:
    """Find the time span where a power series exceeds the floor/plateau midpoint.

    Args:
        values_db: Time series in dB.
        times: Time axis of the same length, in seconds.

    Returns:
        `(start, end)` in seconds, or `(nan, nan)` if nothing exceeds it.
    """
    floor, plateau = np.percentile(values_db, 10), np.percentile(values_db, 90)
    above = np.flatnonzero(values_db > (floor + plateau) / 2.0)
    if above.size == 0:
        return float("nan"), float("nan")
    return float(times[above[0]]), float(times[above[-1]])


def terminal_pixels(freqs: np.ndarray, power_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same pooling as `display.spectrogram_panel` and return the pixels.

    Returns:
        `(pixels, pixel_freqs)` where `pixels` is `(2*TERM_HEIGHT, TERM_WIDTH)`
        with the highest frequency on top, and `pixel_freqs` holds each pixel's
        offset frequency in Hz.
    """
    flipped = power_db[::-1, :]
    pixels = _pool_max(_pool_max(flipped, 2 * TERM_HEIGHT, axis=0), TERM_WIDTH, axis=1)
    return pixels, _pool_mean_axis(freqs[::-1], 2 * TERM_HEIGHT)


def draw_png(
    rec: Recording,
    samples: np.ndarray,
    start: int,
    freqs: np.ndarray,
    times: np.ndarray,
    power_db: np.ndarray,
    out_path: Path,
) -> None:
    """Draw the spectrogram, a frequency slice and the power curve into one PNG."""
    vmin, vmax = np.percentile(power_db, CLIP_PERCENTILES)
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 11), height_ratios=[3, 1.4, 1.2], constrained_layout=True
    )

    ax = axes[0]
    mesh = ax.pcolormesh(
        times, freqs / 1e3, power_db, cmap="viridis", vmin=vmin, vmax=vmax, shading="nearest"
    )
    # Rotate the label explicitly: matplotlib's colorbar default is -90 and the
    # text reads upside down.
    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.set_label("power (dB)", rotation=90, va="bottom", labelpad=14)
    # The reference marker is drawn only at the edges: a full-width line would
    # cover the single-bin tone it is supposed to point at.
    for xmin, xmax in ((0.0, 0.035), (0.965, 1.0)):
        ax.axhline(REF_TONE_HZ / 1e3, xmin=xmin, xmax=xmax, color="red", lw=1.6)
    ax.plot(
        [],
        [],
        color="red",
        lw=1.6,
        label=f"ref_tone expected: +{REF_TONE_HZ / 1e3:.0f} kHz (edge markers)",
    )
    for a in rec.annotations:
        if a.label == "ref_tone" or a.freq_lower_edge is None:
            continue
        t0, t1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
        if t1 < times[0] or t0 > times[-1]:
            continue  # an annotation outside the window would stretch the axis
        lo = (a.freq_lower_edge - (rec.center_frequency or 0.0)) / 1e3
        hi = (a.freq_upper_edge - (rec.center_frequency or 0.0)) / 1e3
        ax.add_patch(
            plt.Rectangle(
                (t0, lo), t1 - t0, hi - lo, fill=False, ec="white", ls="--", lw=1.4, alpha=0.9
            )
        )
        ax.text(t0, hi + 8, a.label, color="white", fontsize=9, weight="bold")
    ax.set_xlim(times[0], times[-1])
    ax.set_ylabel("offset from centre (kHz)")
    ax.set_title(
        f"{rec.meta_path.name} - centre {(rec.center_frequency or 0) / 1e6:.6g} MHz, "
        f"{rec.sample_rate / 1e6:.6g} MS/s\n"
        "white dashed: annotation ranges    red edge markers: expected ref_tone frequency"
    )
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    spectrum = 10.0 * np.log10(np.mean(10.0 ** (power_db / 10.0), axis=1))
    ax.plot(freqs / 1e3, spectrum, lw=0.8, color="#2a788e")
    ax.axvline(REF_TONE_HZ / 1e3, color="red", ls="--", lw=1.0)
    peak_hz = freqs[int(np.argmax(spectrum))]
    ax.annotate(
        f"peak: {peak_hz / 1e3:+.3f} kHz",
        xy=(peak_hz / 1e3, spectrum.max()),
        xytext=(peak_hz / 1e3 + 60, spectrum.max() - 4),
        arrowprops={"arrowstyle": "->", "color": "red"},
        color="red",
        fontsize=9,
    )
    ax.set_xlabel("offset from centre (kHz)")
    ax.set_ylabel("mean power (dB)")
    ax.grid(alpha=0.3)

    ax = axes[2]
    t_power, p_db = power_curve(samples, start, rec.sample_rate)
    ax.plot(t_power, p_db, lw=0.7, color="#440154")
    rise, fall = half_power_edges(t_power, p_db)
    for edge in (rise, fall):
        if edge is not None:
            ax.axvline(edge, color="tab:red", ls=":", lw=1.2)
    for a in rec.annotations:
        t0, t1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
        if a.label == "ref_tone" or t1 < times[0] or t0 > times[-1]:
            continue
        ax.axvspan(t0, t1, alpha=0.15, color="tab:orange")
        ax.text(max(t0, times[0]), ax.get_ylim()[1], f" {a.label}", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("total power (dB)")
    ax.set_xlim(times[0], times[-1])
    ax.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The dpi is high enough for the single-bin reference tone to occupy at
    # least one raster pixel (nfft=1024 rows over a panel about 6 inches tall).
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def report(
    rec: Recording,
    samples: np.ndarray,
    start: int,
    freqs: np.ndarray,
    times: np.ndarray,
    power_db: np.ndarray,
) -> bool:
    """Compare the two drawing paths and report whether every check passed."""
    pixels, pixel_freqs = terminal_pixels(freqs, power_db)
    bin_width = float(freqs[1] - freqs[0])
    ok = True

    print("\n--- reference tone (+100 kHz) ---")
    spectrum = 10.0 * np.log10(np.mean(10.0 ** (power_db / 10.0), axis=1))
    peak_full = float(freqs[int(np.argmax(spectrum))])
    row = int(np.argmax(pixels.max(axis=1)))
    peak_term = float(pixel_freqs[row])
    row_span = 0.5 * abs(pixel_freqs[0] - pixel_freqs[-1]) / (len(pixel_freqs) - 1) * 2

    print(
        f"  matplotlib (full resolution): peak {peak_full / 1e3:+.3f} kHz  (bin {bin_width:.0f} Hz)"
    )
    print(
        f"  terminal (pooled)          : peak row {peak_term / 1e3:+.1f} kHz  "
        f"(row height ~{row_span / 1e3:.1f} kHz)"
    )
    if abs(peak_full - REF_TONE_HZ) > bin_width:
        drift = abs(peak_full - REF_TONE_HZ)
        print(f"  FAIL: the full-resolution peak is {drift:.0f} Hz away from +100 kHz")
        ok = False
    else:
        print("  OK: the full-resolution peak is at +100 kHz (within one bin)")
    if abs(peak_term - REF_TONE_HZ) > row_span:
        print("  FAIL: the terminal peak row does not contain +100 kHz")
        ok = False
    else:
        print("  OK: the terminal peak row contains +100 kHz")

    print("\n--- burst time spans ---")
    for a in rec.annotations:
        if a.label == "ref_tone" or a.freq_lower_edge is None:
            continue
        center = rec.center_frequency or 0.0
        lo, hi = a.freq_lower_edge - center, a.freq_upper_edge - center
        exp0, exp1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
        if exp0 > times[-1] or exp1 < times[0]:
            print(f"  {a.label}: outside the displayed window, skipped")
            continue

        got0, got1 = active_span(band_power_db(freqs, power_db, lo, hi), times)
        rows = (pixel_freqs >= lo) & (pixel_freqs <= hi)
        pixel_times = _pool_mean_axis(times, TERM_WIDTH)
        term0, term1 = active_span(pixels[rows].max(axis=0), pixel_times)

        print(
            f"  {a.label}: annotation {exp0:.4f}…{exp1:.4f} s  "
            f"({lo / 1e3:+.1f}…{hi / 1e3:+.1f} kHz)"
        )
        print(
            f"      matplotlib measurement: {got0:.4f}…{got1:.4f} s  "
            f"(diff {(got0 - exp0) * 1e3:+.1f} / {(got1 - exp1) * 1e3:+.1f} ms)"
        )
        print(f"      terminal measurement  : {term0:.4f}…{term1:.4f} s")
        if abs(got0 - exp0) > 0.005 or abs(got1 - exp1) > 0.005:
            print("      FAIL: outside the 5 ms tolerance")
            ok = False
        else:
            print("      OK: within the 5 ms tolerance")

    ok &= _report_power_edges(rec, samples, start, times)
    return ok


def ramp_half_power_offset(ramp: int) -> float:
    """Distance of the Hann ramp's half-power point from the ramp start, in samples.

    `scripts/make_example.py` softens the burst edges with a
    `np.hanning(2*ramp)` amplitude ramp. The power curve crosses the threshold
    at that ramp's half-power point, while the annotation marks the START of the
    ramp. That difference is the expected offset.
    """
    env_power = np.hanning(2 * ramp)[:ramp] ** 2
    i = int(np.flatnonzero(env_power >= 0.5)[0])
    y0, y1 = env_power[i - 1], env_power[i]
    return float((i - 1) + (0.5 - y0) / (y1 - y0))


def _report_power_edges(rec: Recording, samples: np.ndarray, start: int, times: np.ndarray) -> bool:
    """Compare the power curve's step edges with the annotation edges numerically.

    The edges are measured at two block lengths. As the block shrinks the
    measurement must converge on the ramp's analytic half-power point; if it
    does not, there is a systematic timing error the ramp cannot explain.
    """
    print("\n--- power curve step edges vs annotation ---")
    bursts = [a for a in rec.annotations if a.label != "ref_tone"]
    if len(bursts) != 1:
        print(f"  {len(bursts)} burst annotations; this check expects exactly one.")
        return True

    a = bursts[0]
    exp0, exp1 = a.sample_start / rec.sample_rate, a.sample_end / rec.sample_rate
    if exp0 < times[0] or exp1 > times[-1]:
        print("  the burst lies outside the displayed window, skipped")
        return True

    expected = ramp_half_power_offset(BURST_RAMP)
    print(f"  annotation           : {exp0:.5f}…{exp1:.5f} s")
    print(
        f"  expected offset      : {expected:+.1f} / {-expected:+.1f} samples - "
        f"the half-power point of the {BURST_RAMP}-sample Hann ramp"
    )

    fine = 32
    results: dict[int, tuple[float, float]] = {}
    for block in (POWER_BLOCK, fine):
        t_p, p_db = power_curve(samples, start, rec.sample_rate, block=block)
        rise, fall = half_power_edges(t_p, p_db)
        if rise is None or fall is None:
            print(f"  FAIL: no step edge found for block={block}")
            return False
        n0 = (rise - exp0) * rec.sample_rate
        n1 = (fall - exp1) * rec.sample_rate
        results[block] = (n0, n1)
        tag = "drawn curve" if block == POWER_BLOCK else "fine grid"
        print(
            f"  block={block:<4} ({tag}): {rise:.5f}…{fall:.5f} s  "
            f"diff {(rise - exp0) * 1e3:+.3f} / {(fall - exp1) * 1e3:+.3f} ms  "
            f"({n0:+.0f} / {n1:+.0f} samples)"
        )

    coarse_n0, coarse_n1 = results[POWER_BLOCK]
    fine_n0, fine_n1 = results[fine]
    print(
        f"  block grid effect    : {abs(coarse_n0 - fine_n0):.0f} / "
        f"{abs(coarse_n1 - fine_n1):.0f} samples - a {POWER_BLOCK}-sample block squeezes "
        f"the {BURST_RAMP}-sample ramp into one block and pushes the edge outwards"
    )

    if not (fine_n0 > 0 > fine_n1):
        print("  FAIL: the sign of the offset does not match the ramp (outwards, not inwards)")
        return False
    tolerance = 0.1 * BURST_RAMP
    if abs(fine_n0 - expected) > tolerance or abs(fine_n1 + expected) > tolerance:
        print(
            f"  FAIL: the fine-grid measurement is more than {tolerance:.0f} samples away from "
            "the ramp's half-power point - something other than the ramp is at work"
        )
        return False
    print("  OK: on the fine grid the offset converges on the ramp's half-power point;")
    print("      what remains is block grid resolution, not a systematic error")
    return True


def main() -> int:
    """Run the verification; return 0 when every check passes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="examples/bpsk_01.sigmf-meta")
    parser.add_argument("-o", "--output", default="artifacts/spectrogram_bpsk_01.png")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=None, help="default: the whole recording")
    parser.add_argument("--nfft", type=int, default=1024)
    args = parser.parse_args()

    rec = load(args.path)
    data = rec.read(start=args.start, count=args.samples)
    freqs, times, power_db = compute_spectrogram(data, rec.sample_rate, args.nfft)
    times = times + args.start / rec.sample_rate

    out = Path(args.output)
    draw_png(rec, data, args.start, freqs, times, power_db, out)
    print(f"PNG written: {out}")
    print(
        f"window: samples {args.start}…{args.start + data.size} "
        f"({times[0]:.4f}…{times[-1]:.4f} s), nfft={args.nfft}"
    )

    ok = report(rec, data, args.start, freqs, times, power_db)
    print("\nRESULT:", "every check passed" if ok else "AT LEAST ONE CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
