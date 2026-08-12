"""Tests for `iqforge audit`.

The behaviour these lock down is mostly about what the report refuses to say.
A check that silently turns "not examined" into "passed" is the failure mode the
command exists to prevent, so the vocabulary and the summary arithmetic are
tested as hard as the measurements.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from iqforge.audit import (
    WIDTH,
    AuditReport,
    Finding,
    RecordFeatures,
    Status,
    _pack_path,
    _split_findings,
    audit_dataset,
    difficulty_verdict,
    processing_gain_finding,
    render_json,
    render_text,
    separability,
)


def _report(findings: list[Finding]) -> AuditReport:
    return AuditReport(
        tool_version="0.0.0",
        generated="2026-01-01T00:00:00Z",
        mode="test",
        input_path="ds/",
        input_lines=["1 recording"],
        fingerprint="sha256:00000000",
        findings=findings,
        did_not_check=["something"],
        verdict="unknown - test",
    )


def _splits(records_per_split: dict[str, list[str]], counts: dict[str, int] | None = None) -> dict:
    return {
        name: {
            "count": (counts or {}).get(name, len(ids)),
            "shards": [],
            "labels": [],
            "records": [{"id": rid, "label": "a"} for rid in ids],
        }
        for name, ids in records_per_split.items()
    }


# --------------------------------------------------------------------------
# Split findings
# --------------------------------------------------------------------------


def test_disjoint_recordings_pass_by_proof_not_by_sampling() -> None:
    """Overlap is settled from structure; no windows are read to decide it."""
    findings = _split_findings(_splits({"train": ["a", "b"], "test": ["c"]}), 1024, 512)
    assert [f.status for f in findings] == [Status.PASS_PROOF, Status.PASS_PROOF]
    overlap = next(f for f in findings if f.check == "cross-split overlap")
    assert "impossible" in overlap.detail
    assert "sampled" in overlap.detail


def test_a_recording_in_two_splits_is_a_leak_with_the_shared_sample_count() -> None:
    findings = _split_findings(_splits({"train": ["a", "b"], "test": ["a"]}), 1024, 512)
    assert all(f.status is Status.LEAK for f in findings)
    assert "512 samples" in findings[1].detail


def test_missing_provenance_is_not_checked_rather_than_passed() -> None:
    """A split with windows but no recordings cannot be audited, and says so."""
    splits = _splits({"train": ["a"], "test": []}, counts={"train": 10, "test": 10})
    findings = _split_findings(splits, 1024, 512)
    assert all(f.status is Status.NOT_CHECKED for f in findings)
    assert "not recoverable" in findings[0].detail


def test_zero_stride_overlap_is_reported_as_proof() -> None:
    """Cross-split overlap is impossible when windows are disjoint too."""
    findings = _split_findings(_splits({"train": ["a"], "test": ["b"]}), 1024, 1024)
    assert findings[0].status is Status.PASS_PROOF


# --------------------------------------------------------------------------
# Separability
# --------------------------------------------------------------------------


def test_separability_finds_a_perfectly_predictive_axis() -> None:
    values = [0.0, 0.1, 0.2, 10.0, 10.1, 10.2]
    labels = ["a", "a", "a", "b", "b", "b"]
    score, chance = separability(values, labels)
    assert score == pytest.approx(1.0)
    assert chance == pytest.approx(0.5)


def test_separability_of_an_uninformative_axis_is_near_chance() -> None:
    values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    labels = ["a", "b", "a", "b", "a", "b"]
    score, _ = separability(values, labels)
    assert score == pytest.approx(0.0)


def test_chance_uses_the_largest_class() -> None:
    _, chance = separability([0.0, 1.0, 2.0, 3.0], ["a", "a", "a", "b"])
    assert chance == pytest.approx(0.75)


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def _features(offsets: list[float], labels: list[str]) -> list[RecordFeatures]:
    return [
        RecordFeatures(record_id=f"r{i}", label=label, carrier_offset_hz=offset)
        for i, (offset, label) in enumerate(zip(offsets, labels, strict=True))
    ]


def test_a_separable_axis_makes_the_verdict_ceiling() -> None:
    features = _features([0.0, 1.0, 2.0, 500.0, 501.0, 502.0], ["a"] * 3 + ["b"] * 3)
    verdict, is_ceiling = difficulty_verdict(features)
    assert is_ceiling
    assert verdict.startswith("ceiling")
    assert "carrier offset" in verdict


def test_an_unseparable_axis_gives_unknown_never_measurable() -> None:
    """There is no `measurable` outcome: that would need a trained model."""
    features = _features([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], ["a", "b"] * 3)
    verdict, is_ceiling = difficulty_verdict(features)
    assert not is_ceiling
    assert verdict.startswith("unknown")
    assert "measurable" not in verdict.split(" - ")[0]


def test_processing_gain_is_not_guessed_without_a_bandwidth() -> None:
    finding = processing_gain_finding([RecordFeatures(record_id="r", label="a")])
    assert finding.status is Status.NOT_CHECKED
    assert "undetermined" in finding.detail


def test_processing_gain_reports_the_gap_between_wideband_and_in_band() -> None:
    """The pilot's lesson: a 19.5 kHz signal in 7.68 MHz hides ~26 dB."""
    features = [
        RecordFeatures(
            record_id="r",
            label="a",
            occupied_bw_hz=19_500.0,
            sample_rate=7_680_000.0,
            bw_source="spectrum",
        )
    ]
    finding = processing_gain_finding(features)
    assert finding.status is Status.PASS_SAMPLE
    # 10*log10(7.68e6 / 19.5e3) = 25.95 dB.
    assert "26.0 dB" in finding.detail
    assert "spectrum" in finding.detail


# --------------------------------------------------------------------------
# Report vocabulary and arithmetic
# --------------------------------------------------------------------------


def test_not_checked_is_never_counted_as_passed() -> None:
    report = _report(
        [
            Finding(Status.PASS_PROOF, "a", "x"),
            Finding(Status.PASS_SAMPLE, "b", "x"),
            Finding(Status.RISK, "c", "x"),
            Finding(Status.NOT_CHECKED, "d", "x"),
            Finding(Status.NOT_CHECKED, "e", "x"),
        ]
    )
    assert report.summary == {"leaks": 0, "passed": 2, "risk": 1, "not checked": 2}


def test_the_summary_line_shows_not_checked_separately() -> None:
    report = _report([Finding(Status.PASS_PROOF, "a", "x"), Finding(Status.NOT_CHECKED, "b", "y")])
    line = next(line for line in render_text(report).splitlines() if line.startswith("SUMMARY"))
    assert "1 passed" in line
    assert "1 not checked" in line


def test_the_report_never_says_clean() -> None:
    """The whole point. `PASS` is scoped; `clean` is not a word this tool owns."""
    report = _report([Finding(Status.PASS_PROOF, "a", "x")])
    assert "clean" not in render_text(report).lower()
    assert "clean" not in render_json(report).lower()


def test_what_was_not_checked_is_always_printed() -> None:
    text = render_text(_report([Finding(Status.PASS_PROOF, "a", "x")]))
    assert "WHAT THIS DID NOT CHECK" in text
    assert "- something" in text


def test_the_quotable_width_is_78_columns() -> None:
    """Pinned to the literal.

    The wrapping test below compared against `WIDTH` itself, which made it a
    tautology: raising the constant to 200 kept it green. The number is part of
    the format's contract with anyone pasting the block into a paper, so it is
    asserted here rather than derived.
    """
    assert WIDTH == 78


def test_every_line_fits_the_quotable_width() -> None:
    """The block is meant to be pasted into fixed-width contexts unaltered."""
    report = _report(
        [
            Finding(Status.RISK, "axis: mean power", "d " * 80),
            Finding(Status.NOT_CHECKED, "in-band SNR", "why " * 40),
        ]
    )
    report.input_path = "C:/a-very/deep/path/" * 8
    report.next_step = ["iqforge measure-leakage " + "x/" * 60, "because " * 30]
    for line in render_text(report).splitlines():
        assert len(line) <= WIDTH, line


def test_the_text_block_is_pure_ascii() -> None:
    """It goes into LaTeX and onto non-UTF-8 consoles; both reject the rest."""
    report = _report([Finding(Status.PASS_SAMPLE, "axis: mean power", "60% of 30 (chance 33%)")])
    render_text(report).encode("ascii")


def test_paths_break_on_separators_not_mid_token() -> None:
    lines = _pack_path("aaa/bbb/ccc/ddd", 8)
    assert all(len(line) <= 8 for line in lines)
    assert "".join(lines) == "aaa/bbb/ccc/ddd"


def test_json_carries_the_same_four_buckets_and_the_caveats() -> None:
    report = _report([Finding(Status.RISK, "a", "x"), Finding(Status.NOT_CHECKED, "b", "y")])
    payload = json.loads(render_json(report))
    assert payload["summary"] == {"leak": 0, "pass": 0, "risk": 1, "not_checked": 1}
    assert payload["did_not_check"] == ["something"]


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_audit_of_a_built_dataset_proves_disjointness(tmp_path: Path) -> None:
    manifest = {
        "manifest_schema": 1,
        "config": {"window": 1024, "stride": 512},
        "label_map": {"a": 0, "b": 1},
        "source_files": [],
        "splits": _splits({"train": ["a"], "val": ["b"], "test": ["c"]}),
    }
    report = audit_dataset(tmp_path, manifest, "0.0.0")
    statuses = {f.check: f.status for f in report.findings}
    assert statuses["recording disjointness"] is Status.PASS_PROOF
    assert statuses["task difficulty"] is Status.NOT_CHECKED
    assert report.summary["not checked"] >= 1


def test_audit_reports_50_percent_overlap_from_the_config(tmp_path: Path) -> None:
    manifest = {
        "manifest_schema": 1,
        "config": {"window": 1024, "stride": 512},
        "label_map": {},
        "source_files": [],
        "splits": _splits({"train": ["a"], "test": ["b"]}),
    }
    report = audit_dataset(tmp_path, manifest, "0.0.0")
    assert any("50% overlap" in line for line in report.input_lines)


def test_capture_time_axis_is_not_checked_without_datetime() -> None:
    from iqforge.audit import _capture_time_finding

    features = [RecordFeatures(record_id=f"r{i}", label="a") for i in range(4)]
    finding = _capture_time_finding(features)
    assert finding.status is Status.NOT_CHECKED
    assert "core:datetime" in finding.detail


def test_overlapping_air_time_between_recordings_is_a_leak() -> None:
    from iqforge.audit import _time_overlap

    start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    features = [
        RecordFeatures(
            record_id="a",
            label="x",
            capture_time=start,
            duration_samples=1000,
            sample_rate=100.0,
        ),
        RecordFeatures(
            record_id="b",
            label="x",
            capture_time=start + dt.timedelta(seconds=3),
            duration_samples=1000,
            sample_rate=100.0,
        ),
    ]
    finding = _time_overlap(features)
    assert finding.status is Status.LEAK
    assert "a / b" in finding.detail


def test_non_overlapping_air_time_passes_by_sample_not_by_proof() -> None:
    """Air time comes from metadata, so it can only ever be a sampled pass."""
    from iqforge.audit import _time_overlap

    start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    features = [
        RecordFeatures(
            record_id="a",
            label="x",
            capture_time=start,
            duration_samples=100,
            sample_rate=100.0,
        ),
        RecordFeatures(
            record_id="b",
            label="x",
            capture_time=start + dt.timedelta(seconds=60),
            duration_samples=100,
            sample_rate=100.0,
        ),
    ]
    assert _time_overlap(features).status is Status.PASS_SAMPLE


def test_sample_spread_covers_the_whole_recording(tmp_path: Path) -> None:
    """A prefix would miss a burst that starts late; the spread must not."""
    from iqforge.audit import _sample_spread

    class FakeRecording:
        num_samples = 50_000_000

        def read(self, start: int, count: int) -> np.ndarray:
            # Encode the offset in the samples so coverage is checkable.
            return np.arange(start, start + count, dtype=np.float64).astype(np.complex64)

    samples = _sample_spread(FakeRecording())  # type: ignore[arg-type]
    assert samples.size <= 2_000_000
    assert float(samples.real.max()) > 0.9 * FakeRecording.num_samples
