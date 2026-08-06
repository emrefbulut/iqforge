"""Tests for iqforge.io. All synthetic, no network access."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from helpers import write_record
from iqforge.io import Annotation, IQForgeError, Recording, load

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.sigmf-meta"))


def _burst(rec: Recording) -> Annotation:
    """Return the recording's single modulated burst annotation (not ref_tone)."""
    bursts = [a for a in rec.annotations if a.label != "ref_tone"]
    assert len(bursts) == 1, f"{rec.meta_path.name}: expected exactly one burst"
    return bursts[0]


@pytest.fixture
def tone() -> np.ndarray:
    """A small deterministic complex test signal."""
    n = np.arange(4096)
    return (0.5 * np.exp(2j * np.pi * 0.05 * n)).astype(np.complex64)


@pytest.mark.parametrize("datatype", ["cf32_le", "ci16_le", "ci8"])
def test_roundtrip_all_supported_datatypes(tmp_path: Path, tone: np.ndarray, datatype: str) -> None:
    """All three supported datatypes read back as correctly scaled complex64."""
    meta = write_record(tmp_path, tone, datatype)
    rec = load(meta)

    assert rec.datatype == datatype
    assert rec.num_samples == tone.size
    samples = rec.read()
    assert samples.dtype == np.complex64
    tolerance = {"cf32_le": 1e-6, "ci16_le": 1e-4, "ci8": 1e-2}[datatype]
    assert np.allclose(samples, tone, atol=tolerance)


def test_metadata_fields(tmp_path: Path, tone: np.ndarray) -> None:
    """Sample rate, centre frequency, sample count and duration read correctly."""
    meta = write_record(tmp_path, tone, sample_rate=2_000_000.0, center_freq=915e6)
    rec = load(meta)

    assert rec.sample_rate == 2_000_000.0
    assert rec.center_frequency == 915e6
    assert rec.num_samples == 4096
    assert rec.duration_seconds == pytest.approx(4096 / 2_000_000.0)


def test_partial_read(tmp_path: Path, tone: np.ndarray) -> None:
    """start/count returns the right slice and never runs past the end."""
    meta = write_record(tmp_path, tone)
    rec = load(meta)

    chunk = rec.read(start=1000, count=256)
    assert chunk.size == 256
    assert np.allclose(chunk, tone[1000:1256], atol=1e-6)

    assert rec.read(start=4000, count=10_000).size == 96
    assert rec.read(start=rec.num_samples).size == 0

    with pytest.raises(IQForgeError, match="outside the recording"):
        rec.read(start=-1)


def test_unsupported_datatype_is_explicit(tmp_path: Path, tone: np.ndarray) -> None:
    """An unsupported datatype is never guessed; the message is actionable."""
    meta = write_record(tmp_path, tone, datatype="cf64_le")
    with pytest.raises(IQForgeError) as exc:
        load(meta)
    message = str(exc.value)
    assert "cf64_le" in message
    assert "cf32_le" in message and "ci16_le" in message and "ci8" in message


def test_missing_sample_rate_is_an_error(tmp_path: Path, tone: np.ndarray) -> None:
    """A missing sample rate raises rather than defaulting to something."""
    meta = write_record(tmp_path, tone, sample_rate=None)
    with pytest.raises(IQForgeError, match="core:sample_rate"):
        load(meta)


def test_missing_files_are_reported(tmp_path: Path, tone: np.ndarray) -> None:
    """Missing metadata and data files each raise their own clear error."""
    with pytest.raises(IQForgeError, match="metadata file not found"):
        load(tmp_path / "missing.sigmf-meta")

    meta = write_record(tmp_path, tone)
    meta.with_suffix(".sigmf-data").unlink()
    with pytest.raises(IQForgeError, match="data file not found"):
        load(meta)


def test_truncated_data_file_is_rejected(tmp_path: Path, tone: np.ndarray) -> None:
    """A data file that is not a whole number of samples counts as corrupt."""
    meta = write_record(tmp_path, tone)
    data = meta.with_suffix(".sigmf-data")
    data.write_bytes(data.read_bytes()[:-3])
    with pytest.raises(IQForgeError, match="may be corrupt"):
        load(meta)


def test_annotations_are_parsed_and_sorted(tmp_path: Path, tone: np.ndarray) -> None:
    """Annotations are read and sorted by their start index."""
    meta = write_record(
        tmp_path,
        tone,
        annotations=[
            {"core:sample_start": 2000, "core:sample_count": 500, "core:label": "b"},
            {
                "core:sample_start": 0,
                "core:sample_count": 1000,
                "core:label": "a",
                "core:freq_lower_edge": 99e6,
                "core:freq_upper_edge": 101e6,
            },
        ],
    )
    rec = load(meta)

    assert [a.label for a in rec.annotations] == ["a", "b"]
    assert rec.annotations[0].sample_end == 1000
    assert rec.annotations[0].freq_upper_edge == 101e6


@pytest.mark.skipif(not EXAMPLES, reason="examples/ recordings have not been generated")
def test_example_set_has_sixteen_records_balanced_by_class() -> None:
    """The example dataset is eight bpsk plus eight qpsk recordings."""
    labels = [_burst(load(p)).label for p in EXAMPLES]

    assert len(EXAMPLES) == 16
    assert labels.count("bpsk") == 8
    assert labels.count("qpsk") == 8
    assert len({p.stem for p in EXAMPLES}) == 16, "recording names must be unique"


@pytest.mark.skipif(not EXAMPLES, reason="examples/ recordings have not been generated")
def test_every_class_offset_cell_has_two_records() -> None:
    """Every (class, carrier offset) pair must have exactly TWO recordings.

    This is the precondition for the Phase 4 verification gate to work. With a
    single recording, the within-split independence guarantee (SPEC §5.6) would
    force every recording at that offset into the same split; train and test
    could not share an offset and the model would always be evaluated on an
    unseen carrier.
    """
    cells: dict[tuple[str, int], int] = {}
    for path in EXAMPLES:
        rec = load(path)
        a = _burst(rec)
        centre = round((a.freq_lower_edge + a.freq_upper_edge) / 2 - rec.center_frequency)
        key = (a.label, centre)
        cells[key] = cells.get(key, 0) + 1

    assert len(cells) == 8, f"expected 2 classes x 4 offsets, found {len(cells)} cells"
    assert set(cells.values()) == {2}, f"every cell must hold 2 recordings: {cells}"


@pytest.mark.skipif(not EXAMPLES, reason="examples/ recordings have not been generated")
def test_example_records_share_metadata_and_fit_size_budget() -> None:
    """Every recording shares the same basic metadata and the set fits in 6 MB."""
    total = 0
    for path in EXAMPLES:
        rec = load(path)
        assert rec.datatype == "cf32_le"
        assert rec.sample_rate == 1_024_000.0
        assert rec.center_frequency == 2_450_000_000.0
        assert rec.num_samples == 32_768
        assert rec.duration_seconds == pytest.approx(0.032)
        assert {a.label for a in rec.annotations} == {"ref_tone", _burst(rec).label}
        total += rec.data_path.stat().st_size + rec.meta_path.stat().st_size

    assert total < 6_000_000, f"the example dataset exceeds 6 MB: {total / 1e6:.2f} MB"


@pytest.mark.skipif(not EXAMPLES, reason="examples/ recordings have not been generated")
def test_example_bursts_are_equal_in_bandwidth_and_duration() -> None:
    """All bursts share the same bandwidth and the same duration.

    The classes must differ only by modulation; a difference in bandwidth or
    duration would hand the classifier a shortcut.
    """
    widths = {_burst(load(p)).freq_upper_edge - _burst(load(p)).freq_lower_edge for p in EXAMPLES}
    counts = {_burst(load(p)).sample_count for p in EXAMPLES}

    assert widths == {86_400.0}
    assert counts == {20_480}


@pytest.mark.skipif(not EXAMPLES, reason="examples/ recordings have not been generated")
def test_carrier_offset_carries_no_class_information() -> None:
    """Both classes must draw from the same pool of carrier offsets.

    Otherwise the network learns the carrier frequency rather than the
    modulation, and the Phase 4 accuracy measurement means nothing.
    """
    by_class: dict[str, set[float]] = {}
    for path in EXAMPLES:
        rec = load(path)
        a = _burst(rec)
        centre = (a.freq_lower_edge + a.freq_upper_edge) / 2 - rec.center_frequency
        by_class.setdefault(a.label, set()).add(round(centre))

    assert by_class["bpsk"] == by_class["qpsk"], (
        f"carrier offsets differ between classes: {by_class}"
    )
    assert len(by_class["bpsk"]) == 4


@pytest.mark.skipif(not EXAMPLES, reason="examples/ recordings have not been generated")
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_reference_tone_is_exactly_plus_100_khz(path: Path) -> None:
    """In every recording the reference tone sits exactly +100 kHz from centre.

    Later phases — the spectrogram verification in particular — rest on this.
    """
    rec = load(path)
    ref = next(a for a in rec.annotations if a.label == "ref_tone")
    assert ref.sample_start == 0
    assert ref.sample_count == rec.num_samples
    assert (ref.freq_lower_edge + ref.freq_upper_edge) / 2 - rec.center_frequency == 100_000.0

    # The quiet tail after the burst, holding the tone alone. 4096 samples means
    # a 250 Hz bin width, far more than enough to resolve +100 kHz.
    quiet_start = _burst(rec).sample_end
    quiet = rec.read(start=quiet_start, count=rec.num_samples - quiet_start)
    assert quiet.size >= 4096, "not enough quiet signal to measure the tone"

    spectrum = np.abs(np.fft.fftshift(np.fft.fft(quiet)))
    freqs = np.fft.fftshift(np.fft.fftfreq(quiet.size, d=1.0 / rec.sample_rate))
    bin_width = rec.sample_rate / quiet.size

    # The peak must be at +100 kHz WITH ITS SIGN. Swapping I and Q
    # (x -> j*conj(x)) moves the tone to -100 kHz, so the signed frequency is
    # checked rather than |frequency|.
    peak_offset = freqs[int(np.argmax(spectrum))]
    assert peak_offset == pytest.approx(100_000.0, abs=bin_width)
    assert peak_offset > 0, f"reference tone found at a negative frequency ({peak_offset:.0f} Hz)"

    # The mirror bin at -100 kHz must be clearly weaker: on its own this is the
    # check that catches an I/Q swap.
    power_plus = spectrum[int(np.argmin(np.abs(freqs - 100_000.0)))]
    power_minus = spectrum[int(np.argmin(np.abs(freqs + 100_000.0)))]
    assert power_plus > 100.0 * power_minus, (
        f"+100 kHz / -100 kHz power ratio too low: {power_plus / power_minus:.1f}x"
    )


def test_declared_version_survives_the_sigmf_library(tmp_path: Path) -> None:
    """The version reported must be the file's, not the reader's.

    `SigMFFile(metadata=...)` mutates the dict it is given, replacing
    `core:version` with the spec version the installed library implements. Real
    captures declaring 1.0.0 were being reported as 1.2.6, which is the one
    number a compatibility investigation cannot afford to have wrong.
    """
    samples = (np.arange(16) + 1j * np.arange(16)).astype(np.complex64)
    meta_path = write_record(tmp_path, samples, name="declared")
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    raw["global"]["core:version"] = "1.0.0"
    meta_path.write_text(json.dumps(raw), encoding="utf-8")

    rec = load(meta_path)

    assert rec.declared_version == "1.0.0"
