"""Generate the synthetic SigMF example recordings under examples/.

This script is run once; the resulting `examples/*.sigmf-*` files are treated as
a fixed reference by later work and are not regenerated.

Why more than one recording: SPEC §5.6 requires that windows from the same
recording go to the same split. With a single recording that rule cannot be
exercised, and `build` risks silently falling back to window-level splitting.
The dataset therefore holds sixteen recordings: eight BPSK and eight QPSK.

Every recording contains:
  * A continuous pure tone, exactly +100 kHz from the centre frequency
    (`ref_tone`). The Phase 2 spectrogram verification uses it as a reference.
  * A single modulated burst (`bpsk` OR `qpsk`), which gives the recording its
    identity.
  * Low-level complex Gaussian noise.

Only the modulation differs between the classes. To close off shortcut cues,
these are identical in both:
  * symbol rate (64 kBd) and RRC roll-off (0.35), hence the same bandwidth
    (86.4 kHz)
  * burst duration (20480 samples) and mean power
  * the carrier offset pool - each class uses the same four offsets the same
    number of times, so the carrier frequency carries no class information
What varies from recording to recording: the noise seed, the symbol sequence,
the burst position in time, and the carrier offset.

The sample rate is deliberately 1.024 MHz: with a 1024-point FFT the bin spacing
is exactly 1 kHz, so the +100 kHz reference tone lands exactly on bin +100.

Usage:
    python scripts/make_example.py          # skips if the files exist
    python scripts/make_example.py --force  # deliberately regenerates
"""

from __future__ import annotations

import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sigmf
from sigmf import SigMFFile

BASE_SEED = 20240101
SAMPLE_RATE = 1_024_000.0  # Hz
CENTER_FREQ = 2_450_000_000.0  # Hz
NUM_SAMPLES = 32_768  # 0.032 s per recording

REF_TONE_OFFSET = 100_000.0  # Hz - the known reference signal
REF_TONE_AMPLITUDE = 0.25

# Burst parameters shared by both classes.
SYMBOL_RATE = 64_000.0  # Bd
RRC_BETA = 0.35
BURST_RMS = 0.22  # every burst is normalized to this mean power
BURST_COUNT = 20_480  # samples; same in every recording, 40 labelled windows at 1024/512
BURST_RAMP = 512  # samples; envelope rise and fall time

#: How many recordings to generate per (class, carrier offset) pair.
#: TWO is required so that a single offset's recordings can be shared between
#: train and test. With one, the within-split independence guarantee (§5.6)
#: forces every recording at that offset into the same split and the model is
#: always evaluated on an unseen carrier.
RECORDS_PER_CELL = 2

#: Occupied bandwidth in Hz; the annotation frequency edges derive from this.
OCCUPIED_BW = SYMBOL_RATE * (1.0 + RRC_BETA)

NOISE_SIGMA = 0.02

OUT_DIR = Path(__file__).resolve().parent.parent / "examples"


@dataclass(frozen=True)
class RecordPlan:
    """Parameters of a single recording to generate.

    Attributes:
        name: File name without extension; carries the recording's identity.
        modulation: `"bpsk"` or `"qpsk"`.
        carrier_offset: Burst offset from the centre frequency, in Hz.
        burst_start: Sample index where the burst starts.
        seed: Randomness seed specific to this recording.
    """

    name: str
    modulation: str
    carrier_offset: float
    burst_start: int
    seed: int


#: The carrier offset pool. Each class uses each offset `RECORDS_PER_CELL`
#: times, so the carrier frequency carries no class information.
CARRIER_OFFSETS = (-280_000.0, -180_000.0, 180_000.0, 280_000.0)

#: The burst start pool. Bursts always begin in the first half of the recording,
#: leaving at least 4096 samples of signal-free tail afterwards so the reference
#: tone measurement (the Phase 1/2 verification) has a clean region to work in.
BURST_STARTS = (1_024, 3_072, 5_120, 7_168)


def _build_plans(records_per_cell: int = RECORDS_PER_CELL) -> list[RecordPlan]:
    """Build the plan for 2 classes x 4 offsets x `records_per_cell` recordings.

    The burst start is rotated so that it is independent of both the offset and
    the class: each class uses each start the same number of times, so the burst
    position carries no class information either.

    Args:
        records_per_cell: Recordings per (class, offset) pair. The default is
            what `examples/` was generated with and must not change. The leakage
            experiment asks for more so that each offset group has enough
            recordings to be shared across splits.
    """
    plans: list[RecordPlan] = []
    for class_shift, modulation in enumerate(("bpsk", "qpsk")):
        counter = 0
        for offset_index, offset in enumerate(CARRIER_OFFSETS):
            for repeat in range(records_per_cell):
                counter += 1
                start_index = (2 * offset_index + repeat + class_shift) % len(BURST_STARTS)
                plans.append(
                    RecordPlan(
                        name=f"{modulation}_{counter:02d}",
                        modulation=modulation,
                        carrier_offset=offset,
                        burst_start=BURST_STARTS[start_index],
                        seed=BASE_SEED + len(plans),
                    )
                )
    return plans


PLANS = _build_plans()


def rrc_taps(sps: int, span: int = 8, beta: float = RRC_BETA) -> np.ndarray:
    """Build root-raised-cosine (RRC) pulse-shaping coefficients.

    Args:
        sps: Samples per symbol.
        span: How many symbols the filter spans.
        beta: Roll-off factor.

    Returns:
        A float64 coefficient array normalized to unit energy.
    """
    n = np.arange(-span * sps / 2, span * sps / 2 + 1, dtype=np.float64)
    t = n / sps
    taps = np.empty_like(t)
    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1.0 / (4.0 * beta)):
            taps[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(
                np.pi * ti * (1.0 + beta)
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            taps[i] = num / den
    return taps / np.sqrt(np.sum(taps**2))


def _envelope(length: int, ramp: int) -> np.ndarray:
    """Build a cosine-ramped envelope that softens the burst edges."""
    env = np.ones(length, dtype=np.float64)
    r = np.hanning(2 * ramp)
    env[:ramp] = r[:ramp]
    env[-ramp:] = r[ramp:]
    return env


def _shaped_burst(symbols: np.ndarray, sps: int, length: int) -> np.ndarray:
    """Shape symbols with the RRC filter and return a burst of the given length.

    The burst is multiplied by the softened envelope and then normalized so its
    mean power is exactly `BURST_RMS**2`. That makes different modulations, and
    different recordings, identical in power.

    Args:
        symbols: Complex symbol sequence.
        sps: Samples per symbol.
        length: Desired burst length in samples.

    Returns:
        A complex128 array of `length` samples with mean power `BURST_RMS**2`.
    """
    upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
    upsampled[::sps] = symbols
    shaped = np.convolve(upsampled, rrc_taps(sps), mode="same")
    if len(shaped) < length:
        shaped = np.pad(shaped, (0, length - len(shaped)))
    burst = shaped[:length] * _envelope(length, BURST_RAMP)
    return burst * (BURST_RMS / np.sqrt(np.mean(np.abs(burst) ** 2)))


def _symbols(modulation: str, count: int, rng: np.random.Generator) -> np.ndarray:
    """Draw random constellation symbols for the given modulation.

    Raises:
        ValueError: If the modulation is not recognised.
    """
    if modulation == "bpsk":
        return 2.0 * rng.integers(0, 2, count) - 1.0 + 0j
    if modulation == "qpsk":
        return np.exp(1j * (np.pi / 4.0 + rng.integers(0, 4, count) * np.pi / 2.0))
    raise ValueError(f"Unknown modulation '{modulation}'. Supported: bpsk, qpsk.")


def build_signal(plan: RecordPlan, noise_sigma: float = NOISE_SIGMA) -> np.ndarray:
    """Generate the complex samples of one recording.

    Args:
        plan: The recording's parameters.
        noise_sigma: Standard deviation per quadrature of the additive complex
            Gaussian noise. The default is what `examples/` was generated with
            and must not change — those files are frozen. The leakage
            experiment overrides it to sweep SNR, which is why this is a
            parameter rather than a constant read from the module.
    """
    rng = np.random.default_rng(plan.seed)
    t = np.arange(NUM_SAMPLES, dtype=np.float64) / SAMPLE_RATE
    x = np.zeros(NUM_SAMPLES, dtype=np.complex128)

    # The reference tone: exactly +100 kHz from centre, for the whole recording.
    x += REF_TONE_AMPLITUDE * np.exp(2j * np.pi * REF_TONE_OFFSET * t)

    sps = int(round(SAMPLE_RATE / SYMBOL_RATE))
    burst = _shaped_burst(_symbols(plan.modulation, BURST_COUNT // sps + 8, rng), sps, BURST_COUNT)
    seg = slice(plan.burst_start, plan.burst_start + BURST_COUNT)
    x[seg] += burst * np.exp(2j * np.pi * plan.carrier_offset * t[seg])

    x += noise_sigma * (rng.standard_normal(NUM_SAMPLES) + 1j * rng.standard_normal(NUM_SAMPLES))
    return x.astype(np.complex64)


def write_record(plan: RecordPlan, samples: np.ndarray, out_dir: Path) -> Path:
    """Write the samples and metadata as a SigMF recording pair.

    Args:
        plan: The recording's parameters.
        samples: `complex64` samples.
        out_dir: Output directory.

    Returns:
        Path of the written `.sigmf-meta` file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{plan.name}.sigmf-data"
    meta_path = out_dir / f"{plan.name}.sigmf-meta"

    interleaved = np.empty(samples.size * 2, dtype=np.float32)
    interleaved[0::2] = samples.real
    interleaved[1::2] = samples.imag
    interleaved.tofile(data_path)

    meta = SigMFFile(
        data_file=str(data_path),
        global_info={
            sigmf.DATATYPE_KEY: "cf32_le",
            sigmf.SAMPLE_RATE_KEY: SAMPLE_RATE,
            sigmf.AUTHOR_KEY: "iqforge",
            sigmf.DESCRIPTION_KEY: (
                f"iqforge synthetic example recording '{plan.name}'. Continuous reference "
                f"tone (+100 kHz from centre), one {plan.modulation.upper()} burst, "
                "low-level AWGN."
            ),
            sigmf.HW_KEY: "synthetic (scripts/make_example.py)",
            sigmf.RECORDER_KEY: "iqforge scripts/make_example.py",
            sigmf.VERSION_KEY: "1.0.0",
        },
    )
    meta.add_capture(
        0,
        metadata={
            sigmf.FREQUENCY_KEY: CENTER_FREQ,
            sigmf.DATETIME_KEY: dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
            .isoformat()
            .replace("+00:00", "Z"),
        },
    )
    meta.add_annotation(
        0,
        NUM_SAMPLES,
        metadata={
            sigmf.LABEL_KEY: "ref_tone",
            sigmf.FREQ_LOWER_EDGE_KEY: CENTER_FREQ + REF_TONE_OFFSET - 500.0,
            sigmf.FREQ_UPPER_EDGE_KEY: CENTER_FREQ + REF_TONE_OFFSET + 500.0,
            sigmf.COMMENT_KEY: (
                "Known reference signal: a pure tone exactly +100000 Hz from the centre "
                "frequency, continuous for the whole recording. The spectrogram "
                "verification uses it. It is a measurement reference rather than a class, "
                "and is excluded from labelling by --exclude-label."
            ),
        },
    )
    meta.add_annotation(
        plan.burst_start,
        BURST_COUNT,
        metadata={
            sigmf.LABEL_KEY: plan.modulation,
            sigmf.FREQ_LOWER_EDGE_KEY: CENTER_FREQ + plan.carrier_offset - OCCUPIED_BW / 2.0,
            sigmf.FREQ_UPPER_EDGE_KEY: CENTER_FREQ + plan.carrier_offset + OCCUPIED_BW / 2.0,
            sigmf.COMMENT_KEY: (
                f"{plan.modulation.upper()}, {SYMBOL_RATE / 1e3:.0f} kBd, RRC beta={RRC_BETA}, "
                f"bandwidth {OCCUPIED_BW / 1e3:.1f} kHz, mean power {BURST_RMS**2:.4f}, "
                f"carrier {plan.carrier_offset / 1e3:+.0f} kHz"
            ),
        },
    )

    meta.tofile(str(meta_path), skip_validate=False, pretty=True)
    return meta_path


def main() -> None:
    """Generate every recording and print a summary.

    Does nothing if any recording already exists: the example dataset is
    generated once and then stays fixed. Use `--force` to regenerate on purpose.
    """
    force = "--force" in sys.argv
    existing = [p for p in PLANS if (OUT_DIR / f"{p.name}.sigmf-meta").exists()]
    if existing and not force:
        print(f"{len(existing)} recording(s) already exist in {OUT_DIR} - skipping generation.")
        print("To regenerate on purpose: python scripts/make_example.py --force")
        return
    if force:
        for stale in sorted(OUT_DIR.glob("*.sigmf-*")):
            stale.unlink()

    total_bytes = 0
    print(f"{'recording':<10} {'modulation':<11} {'carrier':>11} {'burst (s)':>18} {'power':>9}")
    for plan in PLANS:
        samples = build_signal(plan)
        meta_path = write_record(plan, samples, OUT_DIR)
        total_bytes += meta_path.with_suffix(".sigmf-data").stat().st_size

        seg = samples[plan.burst_start : plan.burst_start + BURST_COUNT]
        t0 = plan.burst_start / SAMPLE_RATE
        t1 = (plan.burst_start + BURST_COUNT) / SAMPLE_RATE
        print(
            f"{plan.name:<10} {plan.modulation:<11} {plan.carrier_offset / 1e3:>+8.0f} kHz "
            f"{t0:>8.4f}…{t1:<8.4f} {np.mean(np.abs(seg) ** 2):>9.6f}"
        )

    print(
        f"\n{len(PLANS)} recordings, {NUM_SAMPLES} samples each "
        f"({NUM_SAMPLES / SAMPLE_RATE:.4f} s), {total_bytes / 1e6:.2f} MB total"
    )
    print(f"reference tone: {REF_TONE_OFFSET:+.0f} Hz from centre (every recording)")
    print(f"burst bandwidth: {OCCUPIED_BW / 1e3:.1f} kHz (every recording)")


if __name__ == "__main__":
    main()
