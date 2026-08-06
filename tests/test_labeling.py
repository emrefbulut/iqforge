"""Tests for iqforge.labeling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from iqforge.io import Annotation, IQForgeError, Recording
from iqforge.labeling import (
    DEFAULT_EXCLUDE_LABELS,
    UNLABELED,
    AnnotationLabelSurvey,
    annotation_field_value,
    carrier_offset_hz,
    dominant_label,
    label_from_annotations,
    label_from_csv,
    label_from_dirname,
    load_label_csv,
    resolve_exclude_labels,
)
from iqforge.windowing import window_starts

WINDOW, STRIDE = 1024, 512


def _annotation(start: int, count: int, label: str) -> dict:
    return {"core:sample_start": start, "core:sample_count": count, "core:label": label}


@pytest.fixture
def record(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> Callable[..., Recording]:
    """Factory producing an 8192-sample recording with the given annotations."""

    def _make(annotations: list[dict], name: str = "rec", directory: Path | None = None):
        return make_recording(
            directory or tmp_path, noise(8192, seed=1), name=name, annotations=annotations
        )

    return _make


def test_window_label_comes_from_its_centre(record: Callable[..., Recording]) -> None:
    """A window's label comes from the range its centre falls in (SPEC §5.3)."""
    # Window 0 has its centre at 512, window 1 at 1024.
    rec = record([_annotation(0, 1000, "a")])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), False)

    assert labels[0] == "a", "centre 512 falls inside [0,1000)"
    assert labels[1] is None, "centre 1024 falls outside [0,1000)"
    assert stats.labeled == 1
    assert stats.unmatched == len(labels) - 1


def test_excluded_annotation_does_not_create_ambiguity(record: Callable[..., Recording]) -> None:
    """An excluded annotation must not count towards overlap.

    ref_tone spans the whole recording; without exclusion every window would
    fall in two ranges and all of them would be dropped as ambiguous.
    """
    rec = record([_annotation(0, 8192, "ref_tone"), _annotation(0, 4096, "bpsk")])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    excluded, stats = label_from_annotations(
        rec, starts, WINDOW, frozenset(DEFAULT_EXCLUDE_LABELS), False
    )
    assert stats.ambiguous == 0
    assert set(filter(None, excluded)) == {"bpsk"}
    assert stats.excluded_labels == {"ref_tone"}

    kept, kept_stats = label_from_annotations(rec, starts, WINDOW, frozenset(), False)
    assert kept_stats.ambiguous > 0, "without exclusion there must be an overlap"
    assert all(label is None for label in kept[: kept_stats.ambiguous])


def test_ambiguous_windows_are_dropped_not_guessed(record: Callable[..., Recording]) -> None:
    """Windows still overlapping after exclusion are dropped, never guessed."""
    rec = record([_annotation(0, 4096, "a"), _annotation(0, 4096, "b")])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), False)

    assert stats.ambiguous == 7  # windows whose centre lies in [0,4096)
    assert stats.labeled == 0
    assert all(label is None for label in labels[:7])


def test_keep_unlabeled_turns_misses_into_a_class(record: Callable[..., Recording]) -> None:
    """--keep-unlabeled labels unmatched windows instead of dropping them."""
    rec = record([_annotation(0, 1000, "a")])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), True)

    assert labels[1] == UNLABELED
    assert None not in labels
    assert stats.labeled == len(labels)


def test_annotations_without_label_are_ignored(record: Callable[..., Recording]) -> None:
    """An annotation with no core:label cannot be a source of labels."""
    rec = record([{"core:sample_start": 0, "core:sample_count": 4096}])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), False)

    assert all(label is None for label in labels)
    assert stats.labeled == 0


def test_dirname_labels_use_parent_folder(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> None:
    """The dirname source labels every window with the directory name."""
    folder = tmp_path / "device_a"
    rec = make_recording(folder, noise(4096, seed=2))
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_dirname(rec, starts, frozenset())

    assert set(labels) == {"device_a"}
    assert stats.labeled == starts.size


def test_dirname_respects_exclusion(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> None:
    """An excluded directory name must not label any window."""
    rec = make_recording(tmp_path / "junk", noise(4096, seed=2))
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_dirname(rec, starts, frozenset({"junk"}))

    assert all(label is None for label in labels)
    assert stats.excluded_labels == {"junk"}


def test_csv_labels_match_by_filename(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> None:
    """The CSV source matches on file name."""
    rec = make_recording(tmp_path, noise(4096, seed=2), name="capture_7")
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("filename,label\ncapture_7.sigmf-meta,wifi\n", encoding="utf-8")

    table = load_label_csv(csv_path)
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)
    labels, stats = label_from_csv(rec, starts, table, frozenset())

    assert set(labels) == {"wifi"}
    assert stats.labeled == starts.size


def test_csv_errors_are_actionable(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> None:
    """Missing file, missing columns and missing recording each raise clearly."""
    with pytest.raises(IQForgeError, match="Label file not found"):
        load_label_csv(tmp_path / "missing.csv")

    bad = tmp_path / "bad.csv"
    bad.write_text("file,tag\na,b\n", encoding="utf-8")
    with pytest.raises(IQForgeError) as exc:
        load_label_csv(bad)
    assert "filename" in str(exc.value) and "label" in str(exc.value)

    good = tmp_path / "good.csv"
    good.write_text("filename,label\nother.sigmf-meta,wifi\n", encoding="utf-8")
    rec = make_recording(tmp_path, noise(4096, seed=2), name="capture_7")
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)
    with pytest.raises(IQForgeError, match="not in the label CSV"):
        label_from_csv(rec, starts, load_label_csv(good), frozenset())


def test_dominant_label_breaks_ties_deterministically() -> None:
    """The dominant label has the most windows; ties break alphabetically."""
    assert dominant_label(["a", "a", "b", None]) == "a"
    assert dominant_label(["b", "a"]) == "a"
    assert dominant_label(["z", "z", "a", "a"]) == "a"
    assert dominant_label([None, None]) is None


def test_annotation_field_value_reads_arbitrary_sigmf_keys(
    record: Callable[..., Recording],
) -> None:
    """--balance-by must read any SigMF key, not a fixed list of fields."""
    rec = record(
        [
            {
                "core:sample_start": 0,
                "core:sample_count": 4096,
                "core:label": "bpsk",
                "core:freq_lower_edge": 2_450_136_800.0,
                "custom:antenna": "yagi",
            }
        ]
    )

    assert (
        annotation_field_value(rec, "core:freq_lower_edge", "bpsk", frozenset()) == 2_450_136_800.0
    )
    assert annotation_field_value(rec, "custom:antenna", "bpsk", frozenset()) == "yagi"
    assert annotation_field_value(rec, "core:datatype", "bpsk", frozenset()) == "cf32_le"
    assert annotation_field_value(rec, "no:such-field", "bpsk", frozenset()) is None


def test_annotation_field_skips_excluded_annotations(record: Callable[..., Recording]) -> None:
    """The field comes from the annotation that gave the label, not from ref_tone."""
    rec = record(
        [
            {
                "core:sample_start": 0,
                "core:sample_count": 8192,
                "core:label": "ref_tone",
                "core:freq_lower_edge": 1.0,
            },
            {
                "core:sample_start": 0,
                "core:sample_count": 4096,
                "core:label": "bpsk",
                "core:freq_lower_edge": 2.0,
            },
        ]
    )

    value = annotation_field_value(rec, "core:freq_lower_edge", "bpsk", frozenset({"ref_tone"}))

    assert value == 2.0


def test_carrier_offset_is_centre_minus_capture_frequency(
    record: Callable[..., Recording],
) -> None:
    """The carrier offset is the band midpoint minus the capture centre frequency."""
    rec = record(
        [
            {
                "core:sample_start": 0,
                "core:sample_count": 4096,
                "core:label": "bpsk",
                "core:freq_lower_edge": 100_136_800.0,
                "core:freq_upper_edge": 100_223_200.0,
            }
        ]
    )

    assert carrier_offset_hz(rec, "bpsk", frozenset()) == pytest.approx(180_000.0)


def test_carrier_offset_is_none_without_frequency_edges(
    record: Callable[..., Recording],
) -> None:
    """Without frequency edges the offset must not be invented."""
    rec = record([_annotation(0, 4096, "bpsk")])

    assert carrier_offset_hz(rec, "bpsk", frozenset()) is None


def test_exclude_label_default_is_ref_tone() -> None:
    """Without `--exclude-label`, ref_tone is excluded."""
    assert resolve_exclude_labels(None) == frozenset({"ref_tone"})
    assert resolve_exclude_labels([]) == frozenset({"ref_tone"})
    assert resolve_exclude_labels(["a", "b"]) == frozenset({"a", "b"})


def test_survey_names_the_field_that_holds_the_text() -> None:
    """A failed annotation labelling must say what it DID find.

    Recorders disagree about where the class goes -- OmniSIG writes it to
    `core:description` -- and being told only "no labels" leaves the user
    staring at a file that visibly has eight annotations.
    """
    rec = SimpleNamespace(
        annotations=[
            Annotation(sample_start=0, sample_count=10, raw={"core:description": "LTE"}),
            Annotation(sample_start=10, sample_count=10, raw={"core:description": "CDMA"}),
        ]
    )
    survey = AnnotationLabelSurvey()
    survey.observe(rec)  # type: ignore[arg-type]

    hint = survey.hint()
    assert "2 annotation(s)" in hint
    assert "0 carry a non-empty 'core:label'" in hint
    assert "core:description" in hint
    assert "'LTE'" in hint
    # The behaviour must not change: the hint tells, it does not relabel.
    assert "will not guess" in hint


def test_survey_does_not_point_at_structural_fields() -> None:
    """Frequency edges and sample indices are not candidate label fields."""
    rec = SimpleNamespace(
        annotations=[
            Annotation(
                sample_start=0,
                sample_count=10,
                raw={"core:sample_start": 0, "core:freq_lower_edge": 1.0, "core:uuid": "abc-123"},
            )
        ]
    )
    survey = AnnotationLabelSurvey()
    survey.observe(rec)  # type: ignore[arg-type]

    assert survey.text_fields == Counter()


def test_survey_reports_no_annotations_at_all() -> None:
    """A recording with no annotations gets a different, correct explanation."""
    survey = AnnotationLabelSurvey()
    survey.observe(SimpleNamespace(annotations=[]))  # type: ignore[arg-type]

    assert "none of them has any annotation" in survey.hint()
