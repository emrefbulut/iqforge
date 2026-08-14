"""Tests for iqforge.labeling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from helpers import write_record
from iqforge.io import Annotation, IQForgeError, Recording, load
from iqforge.labeling import (
    DEFAULT_EXCLUDE_LABELS,
    UNLABELED,
    AnnotationLabelSurvey,
    annotation_field_value,
    carrier_offset_hz,
    csv_collapse_error,
    csv_declared_labels,
    dirname_at_level,
    dirname_level_warning,
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


def _layout(root: Path, paths: list[str]) -> list[Path]:
    """Create empty SigMF recordings at the given relative directories."""
    samples = (np.arange(32) + 1j * np.arange(32)).astype(np.complex64)
    return [write_record(root / p, samples, name="cap") for p in paths]


def test_warns_when_the_label_level_is_a_run_counter(tmp_path: Path) -> None:
    """The layout that started this: class above, run counter below.

    `CH0/rec1/cap.sigmf-meta` labels by `rec1`, which names the run, not the
    class. Nothing about that is invalid, so nothing failed -- the dataset just
    came out meaningless. The warning has to name both levels, because the
    point is to let the user settle it at a glance.
    """
    metas = _layout(
        tmp_path,
        [f"CH{ch}/rec{n}" for ch in (0, 93, 186) for n in (1, 2, 3)],
    )

    warning = dirname_level_warning(metas, 1)

    assert warning is not None
    assert "rec1" in warning and "rec3" in warning
    assert "CH0" in warning and "CH93" in warning and "CH186" in warning
    assert "--dirname-level 2" in warning


def test_no_warning_once_the_right_level_is_chosen(tmp_path: Path) -> None:
    """Passing --dirname-level 2 on the same layout is silent."""
    metas = _layout(tmp_path, [f"CH{ch}/rec{n}" for ch in (0, 93, 186) for n in (1, 2, 3)])

    assert dirname_level_warning(metas, 2) is None


def test_no_warning_for_a_flat_numbered_class_layout(tmp_path: Path) -> None:
    """`device_01/`, `device_02/` … is a numbered CLASS layout, not a counter.

    A false positive here would be the worst outcome: the labels are correct,
    there is no better level to point at, and a warning would only teach the
    user to ignore warnings.
    """
    metas = _layout(tmp_path, [f"device_{n:02d}" for n in range(1, 9)])

    assert dirname_level_warning(metas, 1) is None


def test_no_warning_for_the_dash7_indoor_layout(tmp_path: Path) -> None:
    """`indoor_loc1/` … `indoor_loc10/` flat: the counter IS the class."""
    metas = _layout(tmp_path, [f"indoor_loc{n}" for n in range(1, 11)])

    assert dirname_level_warning(metas, 1) is None


def test_no_warning_when_numbers_are_values_rather_than_a_count(tmp_path: Path) -> None:
    """Non-contiguous numbers carry meaning; they are not a run counter."""
    metas = _layout(tmp_path, [f"session{s}/snr_{v}" for s in (1, 2) for v in (10, 20, 30)])

    assert dirname_level_warning(metas, 1) is None


def test_no_warning_for_ordinary_word_labels(tmp_path: Path) -> None:
    """bpsk/qpsk never trips the check, at any level."""
    metas = _layout(tmp_path, [f"{cls}/take{n}" for cls in ("bpsk", "qpsk") for n in (1, 2)])

    assert dirname_level_warning(metas, 2) is None


def test_dirname_level_reaching_past_the_root_is_an_error(tmp_path: Path) -> None:
    """An impossible level says so instead of silently labelling by a drive."""
    (meta,) = _layout(tmp_path, ["only"])

    with pytest.raises(IQForgeError, match="reaches past the top"):
        dirname_at_level(meta, 99)

    with pytest.raises(IQForgeError, match="1 or more"):
        dirname_at_level(meta, 0)


def test_label_from_dirname_honours_the_level(tmp_path: Path) -> None:
    """Level 2 labels by the grandparent directory."""
    (meta,) = _layout(tmp_path, ["CH93/rec4"])
    rec = load(meta)
    starts = np.array([0], dtype=np.int64)

    assert label_from_dirname(rec, starts, frozenset(), level=1)[0] == ["rec4"]
    assert label_from_dirname(rec, starts, frozenset(), level=2)[0] == ["CH93"]


def test_label_csv_matches_on_relative_path_when_names_collide(tmp_path):
    """310 of 312 recordings were silently given one label by name-only matching."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,label\nsess1/rrh1/3.sigmf-meta,indoor\nsess2/rrh1/3.sigmf-meta,outdoor\n",
        encoding="utf-8",
    )
    table = load_label_csv(csv_path)
    meta = write_record(tmp_path / "sess1" / "rrh1", np.zeros(4096, dtype=np.complex64), name="3")
    rec = load(meta)
    labels, _ = label_from_csv(
        rec, np.array([0]), table, frozenset(), record_id="sess1/rrh1/3.sigmf-meta"
    )
    assert labels == ["indoor"]


def test_label_csv_refuses_a_name_two_rows_disagree_on(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "filename,label\nsess1/3.sigmf-meta,indoor\nsess2/3.sigmf-meta,outdoor\n", encoding="utf-8"
    )
    table = load_label_csv(csv_path)
    meta = write_record(tmp_path / "other", np.zeros(4096, dtype=np.complex64), name="3")
    rec = load(meta)
    with pytest.raises(IQForgeError, match="more than one"):
        label_from_csv(rec, np.array([0]), table, frozenset())


def test_label_csv_still_matches_a_flat_layout_by_name(tmp_path):
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("filename,label\nrec.sigmf-meta,bpsk\n", encoding="utf-8")
    table = load_label_csv(csv_path)
    meta = write_record(tmp_path / "d", np.zeros(4096, dtype=np.complex64))
    labels, _ = label_from_csv(load(meta), np.array([0]), table, frozenset())
    assert labels == ["bpsk"]


# --------------------------------------------------------------------------
# Label-source collapse (no threshold, no notion of "too imbalanced")
# --------------------------------------------------------------------------


def _nested_label_csv(path, rows):
    path.write_text("filename,label\n" + "".join(f"{k},{v}\n" for k, v in rows), encoding="utf-8")
    return path


def test_csv_declared_labels_reads_the_rows_not_the_lookup(tmp_path):
    """The LoRaIQ scenario: one file name under many directories, four labels."""
    csv_path = _nested_label_csv(
        tmp_path / "labels.csv",
        [
            ("s1/rrh1/3.sigmf-meta", "drone_los"),
            ("s2/rrh1/3.sigmf-meta", "indoor"),
            ("s3/rrh1/3.sigmf-meta", "pedestrian_nlos"),
            ("s4/rrh1/3.sigmf-meta", "pedestrian_partial_los"),
        ],
    )
    built = [
        "s1/rrh1/3.sigmf-meta",
        "s2/rrh1/3.sigmf-meta",
        "s3/rrh1/3.sigmf-meta",
        "s4/rrh1/3.sigmf-meta",
    ]
    assert len(csv_declared_labels(csv_path, built, frozenset())) == 4


def test_a_collapsed_lookup_is_an_error_naming_the_lost_labels(tmp_path):
    """310 of 312 recordings carried one label and the build said nothing."""
    csv_path = _nested_label_csv(
        tmp_path / "labels.csv",
        [("s1/1.sigmf-meta", "a"), ("s2/1.sigmf-meta", "b"), ("s3/1.sigmf-meta", "c")],
    )
    declared = csv_declared_labels(csv_path, ["s1/1.sigmf-meta", "s2/1.sigmf-meta"], frozenset())
    message = csv_collapse_error(declared, {"a"}, csv_path)
    assert message is not None
    assert "b" in message
    assert "labels.csv" in message


def test_no_error_when_every_declared_label_survives(tmp_path):
    csv_path = _nested_label_csv(
        tmp_path / "labels.csv", [("a.sigmf-meta", "x"), ("b.sigmf-meta", "y")]
    )
    declared = csv_declared_labels(csv_path, ["a.sigmf-meta", "b.sigmf-meta"], frozenset())
    assert csv_collapse_error(declared, {"x", "y"}, csv_path) is None


def test_an_extremely_skewed_but_complete_labelling_is_not_an_error(tmp_path):
    """Rare-event detection is a normal thing to build. No threshold on skew."""
    rows = [(f"r{i}.sigmf-meta", "background") for i in range(200)]
    rows.append(("rare.sigmf-meta", "event"))
    csv_path = _nested_label_csv(tmp_path / "labels.csv", rows)
    built = [k for k, _ in rows]
    declared = csv_declared_labels(csv_path, built, frozenset())
    assert csv_collapse_error(declared, {"background", "event"}, csv_path) is None


def test_excluded_labels_do_not_look_like_a_collapse(tmp_path):
    csv_path = _nested_label_csv(
        tmp_path / "labels.csv",
        [("a.sigmf-meta", "x"), ("b.sigmf-meta", "y"), ("c.sigmf-meta", "ref_tone")],
    )
    built = ["a.sigmf-meta", "b.sigmf-meta", "c.sigmf-meta"]
    declared = csv_declared_labels(csv_path, built, frozenset({"ref_tone"}))
    assert declared == {"x", "y"}
    assert csv_collapse_error(declared, {"x", "y"}, csv_path) is None


def test_a_csv_covering_a_superset_does_not_trigger_the_check(tmp_path):
    """A table for a whole corpus, used to build one subdirectory."""
    csv_path = _nested_label_csv(
        tmp_path / "labels.csv",
        [("s1/a.sigmf-meta", "x"), ("s2/b.sigmf-meta", "y"), ("s3/c.sigmf-meta", "z")],
    )
    declared = csv_declared_labels(csv_path, ["s1/a.sigmf-meta"], frozenset())
    assert declared == {"x"}
    assert csv_collapse_error(declared, {"x"}, csv_path) is None
