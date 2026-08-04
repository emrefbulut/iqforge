"""iqforge.labeling testleri."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from iqforge.io import IQForgeError, Recording
from iqforge.labeling import (
    DEFAULT_EXCLUDE_LABELS,
    UNLABELED,
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
    """Verilen annotation'larla 8192 örneklik bir kayıt üreten fabrika."""

    def _make(annotations: list[dict], name: str = "rec", directory: Path | None = None):
        return make_recording(
            directory or tmp_path, noise(8192, seed=1), name=name, annotations=annotations
        )

    return _make


def test_window_label_comes_from_its_centre(record: Callable[..., Recording]) -> None:
    """Pencerenin etiketi merkezinin düştüğü aralıktan gelmeli (SPEC §5.3)."""
    # Pencere 0'ın merkezi 512, pencere 1'in merkezi 1024.
    rec = record([_annotation(0, 1000, "a")])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), False)

    assert labels[0] == "a", "merkez 512, [0,1000) içinde"
    assert labels[1] is None, "merkez 1024, [0,1000) dışında"
    assert stats.labeled == 1
    assert stats.unmatched == len(labels) - 1


def test_excluded_annotation_does_not_create_ambiguity(record: Callable[..., Recording]) -> None:
    """Dışlanan annotation çakışma sayımına girmemeli.

    ref_tone kaydın tamamını kapsıyor; dışlanmazsa her pencere iki aralığa
    düşer ve hepsi belirsiz sayılıp atılırdı.
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
    assert kept_stats.ambiguous > 0, "dışlama olmadan çakışma olmalı"
    assert all(label is None for label in kept[: kept_stats.ambiguous])


def test_ambiguous_windows_are_dropped_not_guessed(record: Callable[..., Recording]) -> None:
    """Dışlamadan sonra hâlâ çakışan pencereler atılmalı, biri seçilmemeli."""
    rec = record([_annotation(0, 4096, "a"), _annotation(0, 4096, "b")])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), False)

    assert stats.ambiguous == 7  # merkezleri [0,4096) içinde kalan pencereler
    assert stats.labeled == 0
    assert all(label is None for label in labels[:7])


def test_keep_unlabeled_turns_misses_into_a_class(record: Callable[..., Recording]) -> None:
    """--keep-unlabeled eşleşmeyen pencereleri atmak yerine etiketlemeli."""
    rec = record([_annotation(0, 1000, "a")])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), True)

    assert labels[1] == UNLABELED
    assert None not in labels
    assert stats.labeled == len(labels)


def test_annotations_without_label_are_ignored(record: Callable[..., Recording]) -> None:
    """core:label taşımayan annotation etiket kaynağı olamaz."""
    rec = record([{"core:sample_start": 0, "core:sample_count": 4096}])
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_annotations(rec, starts, WINDOW, frozenset(), False)

    assert all(label is None for label in labels)
    assert stats.labeled == 0


def test_dirname_labels_use_parent_folder(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> None:
    """dirname kaynağı klasör adını tüm pencerelere vermeli."""
    folder = tmp_path / "device_a"
    rec = make_recording(folder, noise(4096, seed=2))
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_dirname(rec, starts, frozenset())

    assert set(labels) == {"device_a"}
    assert stats.labeled == starts.size


def test_dirname_respects_exclusion(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> None:
    """Dışlanan klasör adı hiçbir pencereye etiket olmamalı."""
    rec = make_recording(tmp_path / "junk", noise(4096, seed=2))
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)

    labels, stats = label_from_dirname(rec, starts, frozenset({"junk"}))

    assert all(label is None for label in labels)
    assert stats.excluded_labels == {"junk"}


def test_csv_labels_match_by_filename(
    tmp_path: Path, make_recording: Callable[..., Recording], noise: Callable[..., np.ndarray]
) -> None:
    """CSV kaynağı dosya adına göre eşleşmeli."""
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
    """Eksik dosya, eksik sütun ve eksik kayıt için ayrı ayrı açık hata verilmeli."""
    with pytest.raises(IQForgeError, match="Etiket dosyası bulunamadı"):
        load_label_csv(tmp_path / "yok.csv")

    bad = tmp_path / "bad.csv"
    bad.write_text("dosya,etiket\na,b\n", encoding="utf-8")
    with pytest.raises(IQForgeError) as exc:
        load_label_csv(bad)
    assert "filename" in str(exc.value) and "label" in str(exc.value)

    good = tmp_path / "good.csv"
    good.write_text("filename,label\nbaska.sigmf-meta,wifi\n", encoding="utf-8")
    rec = make_recording(tmp_path, noise(4096, seed=2), name="capture_7")
    starts = window_starts(rec.num_samples, WINDOW, STRIDE)
    with pytest.raises(IQForgeError, match="etiket CSV'sinde yok"):
        label_from_csv(rec, starts, load_label_csv(good), frozenset())


def test_dominant_label_breaks_ties_deterministically() -> None:
    """Baskın etiket en çok pencereye sahip olan; eşitlikte alfabetik ilk."""
    assert dominant_label(["a", "a", "b", None]) == "a"
    assert dominant_label(["b", "a"]) == "a"
    assert dominant_label(["z", "z", "a", "a"]) == "a"
    assert dominant_label([None, None]) is None


def test_annotation_field_value_reads_arbitrary_sigmf_keys(
    record: Callable[..., Recording],
) -> None:
    """--balance-by herhangi bir SigMF anahtarını okuyabilmeli, sabit alan listesi olmamalı."""
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
    assert annotation_field_value(rec, "yok:alan", "bpsk", frozenset()) is None


def test_annotation_field_skips_excluded_annotations(record: Callable[..., Recording]) -> None:
    """Alan, kayda etiketini veren annotation'dan okunmalı; ref_tone'dan değil."""
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
    """Taşıyıcı ofseti annotation bandının ortası eksi capture merkez frekansı."""
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
    """Frekans sınırları yoksa ofset uydurulmamalı."""
    rec = record([_annotation(0, 4096, "bpsk")])

    assert carrier_offset_hz(rec, "bpsk", frozenset()) is None


def test_exclude_label_default_is_ref_tone() -> None:
    """--exclude-label verilmezse ref_tone dışlanmalı."""
    assert resolve_exclude_labels(None) == frozenset({"ref_tone"})
    assert resolve_exclude_labels([]) == frozenset({"ref_tone"})
    assert resolve_exclude_labels(["a", "b"]) == frozenset({"a", "b"})
