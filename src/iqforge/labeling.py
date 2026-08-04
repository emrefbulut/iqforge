"""Pencere etiketleme: SigMF annotation, klasör adı ve CSV kaynakları."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iqforge.io import Annotation, IQForgeError, Recording

#: Desteklenen etiket kaynakları (`--labels`).
LABEL_SOURCES = ("annotations", "dirname", "csv")

#: `--exclude-label` verilmezse dışlanan etiketler. Örnek kayıtlardaki `ref_tone`
#: bir sınıf değil ölçüm referansıdır ve kaydın tamamını kapsadığı için her
#: pencereyle çakışır; ayrıntı SPEC §5.3.
DEFAULT_EXCLUDE_LABELS = ("ref_tone",)

#: `--keep-unlabeled` verildiğinde etiketsiz pencerelere verilen etiket.
UNLABELED = "unlabeled"


@dataclass
class LabelingStats:
    """Bir kaydın etiketlenmesinde ne olduğunu özetler.

    Attributes:
        total: Kayıttaki toplam pencere sayısı.
        labeled: Etiket alan pencere sayısı.
        unmatched: Hiçbir annotation aralığına düşmediği için atılan pencereler.
        ambiguous: Dışlamadan sonra birden fazla aralığa düştüğü için atılanlar.
        excluded_labels: Bu kayıtta fiilen dışlanan etiketler.
    """

    total: int = 0
    labeled: int = 0
    unmatched: int = 0
    ambiguous: int = 0
    excluded_labels: set[str] = field(default_factory=set)

    def merge(self, other: LabelingStats) -> None:
        """Başka bir kaydın istatistiklerini bu nesneye ekler."""
        self.total += other.total
        self.labeled += other.labeled
        self.unmatched += other.unmatched
        self.ambiguous += other.ambiguous
        self.excluded_labels |= other.excluded_labels


def label_from_annotations(
    rec: Recording,
    starts: np.ndarray,
    window: int,
    exclude_labels: frozenset[str],
    keep_unlabeled: bool,
) -> tuple[list[str | None], LabelingStats]:
    """Pencereleri SigMF annotation'larına göre etiketler.

    Bir pencerenin etiketi, pencere MERKEZİNİN hangi annotation aralığına
    düştüğüyle belirlenir. `exclude_labels` içindeki annotation'lar hiç dikkate
    alınmaz — çakışma sayımına da girmezler.

    Dışlamadan sonra bir pencere hâlâ birden fazla aralığa düşüyorsa
    etiketlenemez sayılır ve atılır; sessizce biri seçilmez (SPEC §5.3).

    Args:
        rec: Açılmış kayıt.
        starts: Pencere başlangıç indisleri.
        window: Pencere uzunluğu.
        exclude_labels: Dikkate alınmayacak etiketler.
        keep_unlabeled: True ise eşleşmeyen pencereler `UNLABELED` etiketi alır.

    Returns:
        `(labels, stats)` — `labels` her pencere için etiket veya None.
    """
    usable = [
        a
        for a in rec.annotations
        if a.label is not None and a.label not in exclude_labels and a.sample_count > 0
    ]
    stats = LabelingStats(
        total=starts.size,
        excluded_labels={
            a.label for a in rec.annotations if a.label is not None and a.label in exclude_labels
        },
    )

    centres = starts + window // 2
    labels: list[str | None] = []
    for centre in centres:
        matches = [a.label for a in usable if a.sample_start <= centre < a.sample_end]
        if len(matches) == 1:
            labels.append(matches[0])
            stats.labeled += 1
        elif len(matches) == 0:
            labels.append(UNLABELED if keep_unlabeled else None)
            stats.unmatched += 1
            stats.labeled += int(keep_unlabeled)
        else:
            labels.append(None)
            stats.ambiguous += 1
    return labels, stats


def label_from_dirname(
    rec: Recording, starts: np.ndarray, exclude_labels: frozenset[str]
) -> tuple[list[str | None], LabelingStats]:
    """Kaydın bulunduğu klasörün adını tüm pencerelere etiket olarak verir."""
    name = rec.meta_path.resolve().parent.name
    stats = LabelingStats(total=starts.size)
    if name in exclude_labels:
        stats.excluded_labels.add(name)
        stats.unmatched = starts.size
        return [None] * starts.size, stats
    stats.labeled = starts.size
    return [name] * starts.size, stats


def load_label_csv(path: Path) -> dict[str, str]:
    """`filename,label` sütunlu CSV'yi okur.

    `filename` alanı yol ayracı içerebilir; eşleştirme dosya adına göre yapılır.

    Raises:
        IQForgeError: Dosya yoksa veya beklenen sütunlar eksikse.
    """
    if not path.exists():
        raise IQForgeError(f"Etiket dosyası bulunamadı: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"filename", "label"} <= fields:
            raise IQForgeError(
                f"'{path.name}' içinde 'filename' ve 'label' sütunları olmalı. "
                f"Bulunan sütunlar: {', '.join(sorted(fields)) or '(yok)'}."
            )
        return {Path(row["filename"]).name: row["label"] for row in reader if row.get("filename")}


def label_from_csv(
    rec: Recording,
    starts: np.ndarray,
    table: dict[str, str],
    exclude_labels: frozenset[str],
) -> tuple[list[str | None], LabelingStats]:
    """CSV tablosundan kaydın etiketini bulup tüm pencerelere verir.

    Raises:
        IQForgeError: Kayıt CSV'de yoksa.
    """
    candidates = (rec.meta_path.name, rec.meta_path.stem, rec.data_path.name)
    label = next((table[c] for c in candidates if c in table), None)
    if label is None:
        raise IQForgeError(
            f"'{rec.meta_path.name}' etiket CSV'sinde yok. "
            f"CSV'nin 'filename' sütununda şunlardan biri bulunmalı: {', '.join(candidates)}."
        )

    stats = LabelingStats(total=starts.size)
    if label in exclude_labels:
        stats.excluded_labels.add(label)
        stats.unmatched = starts.size
        return [None] * starts.size, stats
    stats.labeled = starts.size
    return [label] * starts.size, stats


def dominant_label(labels: list[str | None]) -> str | None:
    """Bir kaydın baskın etiketini verir (en çok pencereye sahip etiket).

    Katmanlı bölme kayıt bazında yapılır; birden fazla etiket içeren kayıtlar
    için hangi katmana ait olduğunu bu belirler. Eşitlikte alfabetik olarak ilk
    etiket seçilir, böylece sonuç deterministiktir.
    """
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        return None
    best = max(counts.values())
    return sorted(label for label, n in counts.items() if n == best)[0]


def labelled_annotation(
    rec: Recording, label: str, exclude_labels: frozenset[str]
) -> Annotation | None:
    """Kayda etiketini veren annotation'ı bulur.

    Önce `core:label` alanı `label` ile eşleşen annotation aranır (annotations
    kaynağı). Bulunamazsa — `dirname`/`csv` kaynaklarında etiket annotation'dan
    gelmez — dışlanmamış tek bir annotation varsa o kullanılır.

    Returns:
        Bulunan `Annotation` veya None.
    """
    for annotation in rec.annotations:
        if annotation.label == label:
            return annotation
    usable = [a for a in rec.annotations if a.label not in exclude_labels]
    return usable[0] if len(usable) == 1 else None


def annotation_field_value(rec: Recording, field: str, label: str, exclude: frozenset[str]) -> Any:
    """Bir kayıttan, dengeleme için kullanılacak metadata alanının değerini okur.

    Alan sırayla şuralarda aranır:
      1. Kayda etiketini veren annotation'ın ham SigMF sözlüğü.
      2. `global` bölümü (`core:hw` gibi kayıt geneli alanlar için).

    Args:
        rec: Açılmış kayıt.
        field: SigMF anahtarı, ör. `core:freq_lower_edge` veya `core:hw`.
        label: Kaydın baskın etiketi.
        exclude: Etiketlemede dışlanan etiketler.

    Returns:
        Alanın değeri; hiçbir yerde bulunamazsa None.
    """
    annotation = labelled_annotation(rec, label, exclude)
    if annotation is not None and field in annotation.raw:
        return annotation.raw[field]
    return rec.global_info.get(field)


def carrier_offset_hz(rec: Recording, label: str, exclude: frozenset[str]) -> float | None:
    """Kaydın burst'ünün merkez frekansa göre taşıyıcı ofsetini verir (Hz).

    Annotation'ın frekans sınırlarının ortası ile capture merkez frekansının
    farkıdır. Sınırlar veya merkez frekans yoksa None döner.
    """
    annotation = labelled_annotation(rec, label, exclude)
    if annotation is None or rec.center_frequency is None:
        return None
    if annotation.freq_lower_edge is None or annotation.freq_upper_edge is None:
        return None
    centre = (annotation.freq_lower_edge + annotation.freq_upper_edge) / 2.0
    return float(centre - rec.center_frequency)


def resolve_exclude_labels(values: list[str] | None) -> frozenset[str]:
    """`--exclude-label` değerlerini çözer; verilmemişse varsayılanı kullanır."""
    if values is None or not values:
        return frozenset(DEFAULT_EXCLUDE_LABELS)
    return frozenset(values)
