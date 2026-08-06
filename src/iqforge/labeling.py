"""Window labelling from three sources: SigMF annotations, directory name, CSV."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from iqforge.io import Annotation, IQForgeError, Recording

#: Supported label sources (`--labels`).
LABEL_SOURCES = ("annotations", "dirname", "csv")

#: Labels excluded when `--exclude-label` is not given. In the bundled example
#: recordings `ref_tone` is a measurement reference rather than a class, and it
#: spans the whole recording so it overlaps every window; see SPEC §5.3.
DEFAULT_EXCLUDE_LABELS = ("ref_tone",)

#: Label given to unmatched windows when `--keep-unlabeled` is set.
UNLABELED = "unlabeled"


#: Annotation keys that hold structure rather than a name, so finding text in
#: them says nothing about where a label might be hiding.
_NON_LABEL_KEYS = frozenset(
    {
        "core:label",
        "core:sample_start",
        "core:sample_count",
        "core:freq_lower_edge",
        "core:freq_upper_edge",
        "core:uuid",
        "core:generator",
        "core:comment",
    }
)


@dataclass
class AnnotationLabelSurvey:
    """What the annotations look like, gathered so a failure can explain itself.

    `--labels annotations` reads `core:label` and nothing else, on purpose: a
    tool that guesses which field means "class" will eventually guess wrong and
    silently mislabel a dataset. But when nothing matches, the user deserves to
    know WHY rather than being told only that it did not work. Real recorders
    disagree about where the class goes -- OmniSIG writes it to
    `core:description` -- so the message names what was actually found.

    Attributes:
        recordings: Recordings scanned.
        annotations: Annotations seen across them.
        with_label: Annotations carrying a non-empty `core:label`.
        text_fields: Other keys whose value is non-empty text, and how often.
        examples: One sample value per such key, to make the hint concrete.
    """

    recordings: int = 0
    annotations: int = 0
    with_label: int = 0
    text_fields: Counter[str] = field(default_factory=Counter)
    examples: dict[str, str] = field(default_factory=dict)

    def observe(self, rec: Recording) -> None:
        """Fold one recording's annotations into the survey."""
        self.recordings += 1
        for annotation in rec.annotations:
            self.annotations += 1
            if annotation.label:
                self.with_label += 1
            for key, value in annotation.raw.items():
                if key in _NON_LABEL_KEYS or not isinstance(value, str) or not value.strip():
                    continue
                self.text_fields[key] += 1
                self.examples.setdefault(key, value.strip())

    def hint(self) -> str:
        """A sentence or three explaining what the annotations contain."""
        if self.annotations == 0:
            return (
                f"Scanned {self.recordings} recording(s); none of them has any annotation. "
                "Label from the directory name (--labels dirname) or a CSV (--labels csv)."
            )
        lines = [
            f"Scanned {self.recordings} recording(s) with {self.annotations} annotation(s); "
            f"{self.with_label} carry a non-empty 'core:label'."
        ]
        if self.with_label and not self.text_fields:
            lines.append(
                "The labels exist, so every window was dropped for another reason: check the "
                "unmatched and ambiguous counts above, and --exclude-label."
            )
            return " ".join(lines)
        if self.text_fields:
            found = ", ".join(
                f"'{key}' ({count}x, e.g. {self.examples[key]!r})"
                for key, count in self.text_fields.most_common(3)
            )
            lines.append(f"Text was found in {found}.")
            lines.append(
                "iqforge labels from 'core:label' only and will not guess which other field "
                "means 'class' -- guessing wrong mislabels a dataset silently. If one of the "
                "fields above is your class, copy it into 'core:label', or label from the "
                "directory name (--labels dirname) or a CSV (--labels csv)."
            )
        return " ".join(lines)


@dataclass
class LabelingStats:
    """Summary of what happened while labelling one recording.

    Attributes:
        total: Total windows in the recording.
        labeled: Windows that received a label.
        unmatched: Windows dropped because they fell in no annotation range.
        ambiguous: Windows dropped because, after exclusions, they still fell in
            more than one range.
        excluded_labels: Labels actually excluded in this recording.
    """

    total: int = 0
    labeled: int = 0
    unmatched: int = 0
    ambiguous: int = 0
    excluded_labels: set[str] = field(default_factory=set)

    def merge(self, other: LabelingStats) -> None:
        """Fold another recording's statistics into this one."""
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
    """Label windows from the recording's SigMF annotations.

    A window's label is decided by which annotation range its CENTRE falls in.
    Annotations in `exclude_labels` are ignored entirely — they do not even
    count towards overlap.

    If a window still falls in more than one range after exclusion it is treated
    as unlabelled and dropped; one is never picked silently (SPEC §5.3).

    Args:
        rec: The opened recording.
        starts: Window start indices.
        window: Window length.
        exclude_labels: Labels to ignore.
        keep_unlabeled: If True, unmatched windows get the `UNLABELED` label.

    Returns:
        `(labels, stats)` where `labels` holds a label or None per window.
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
    """Use the name of the recording's parent directory as the label."""
    name = rec.meta_path.resolve().parent.name
    stats = LabelingStats(total=starts.size)
    if name in exclude_labels:
        stats.excluded_labels.add(name)
        stats.unmatched = starts.size
        return [None] * starts.size, stats
    stats.labeled = starts.size
    return [name] * starts.size, stats


def load_label_csv(path: Path) -> dict[str, str]:
    """Read a CSV with `filename,label` columns.

    The `filename` field may contain path separators; matching is done on the
    file name alone.

    Raises:
        IQForgeError: If the file is missing or the expected columns are absent.
    """
    if not path.exists():
        raise IQForgeError(f"Label file not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if not {"filename", "label"} <= fields:
            raise IQForgeError(
                f"'{path.name}' must have 'filename' and 'label' columns. "
                f"Columns found: {', '.join(sorted(fields)) or '(none)'}."
            )
        return {Path(row["filename"]).name: row["label"] for row in reader if row.get("filename")}


def label_from_csv(
    rec: Recording,
    starts: np.ndarray,
    table: dict[str, str],
    exclude_labels: frozenset[str],
) -> tuple[list[str | None], LabelingStats]:
    """Look the recording up in the CSV table and label every window with it.

    Raises:
        IQForgeError: If the recording is not listed in the CSV.
    """
    candidates = (rec.meta_path.name, rec.meta_path.stem, rec.data_path.name)
    label = next((table[c] for c in candidates if c in table), None)
    if label is None:
        raise IQForgeError(
            f"'{rec.meta_path.name}' is not in the label CSV. The 'filename' column must "
            f"contain one of: {', '.join(candidates)}."
        )

    stats = LabelingStats(total=starts.size)
    if label in exclude_labels:
        stats.excluded_labels.add(label)
        stats.unmatched = starts.size
        return [None] * starts.size, stats
    stats.labeled = starts.size
    return [label] * starts.size, stats


def dominant_label(labels: list[str | None]) -> str | None:
    """Return a recording's dominant label — the one with the most windows.

    Stratified splitting works at the recording level, so this decides which
    stratum a recording holding several labels belongs to. Ties are broken
    alphabetically, which keeps the result deterministic.
    """
    counts = Counter(label for label in labels if label is not None)
    if not counts:
        return None
    best = max(counts.values())
    return sorted(label for label, n in counts.items() if n == best)[0]


def labelled_annotation(
    rec: Recording, label: str, exclude_labels: frozenset[str]
) -> Annotation | None:
    """Find the annotation that gave the recording its label.

    First looks for an annotation whose `core:label` matches `label` (the
    annotations source). Failing that — with the `dirname` and `csv` sources the
    label does not come from an annotation — it uses the single non-excluded
    annotation, if there is exactly one.

    Returns:
        The matching `Annotation`, or None.
    """
    for annotation in rec.annotations:
        if annotation.label == label:
            return annotation
    usable = [a for a in rec.annotations if a.label not in exclude_labels]
    return usable[0] if len(usable) == 1 else None


def annotation_field_value(rec: Recording, field: str, label: str, exclude: frozenset[str]) -> Any:
    """Read the metadata field used for balancing from a recording.

    The field is looked up, in order, in:
      1. the raw SigMF dictionary of the annotation that gave the label,
      2. the `global` section, for recording-wide fields such as `core:hw`.

    Args:
        rec: The opened recording.
        field: A SigMF key, e.g. `core:freq_lower_edge` or `core:hw`.
        label: The recording's dominant label.
        exclude: Labels excluded during labelling.

    Returns:
        The field's value, or None if it is not found anywhere.
    """
    annotation = labelled_annotation(rec, label, exclude)
    if annotation is not None and field in annotation.raw:
        return annotation.raw[field]
    return rec.global_info.get(field)


def carrier_offset_hz(rec: Recording, label: str, exclude: frozenset[str]) -> float | None:
    """Return the burst's carrier offset from the centre frequency, in Hz.

    This is the midpoint of the annotation's frequency edges minus the capture
    centre frequency. Returns None if either is missing.
    """
    annotation = labelled_annotation(rec, label, exclude)
    if annotation is None or rec.center_frequency is None:
        return None
    if annotation.freq_lower_edge is None or annotation.freq_upper_edge is None:
        return None
    centre = (annotation.freq_lower_edge + annotation.freq_upper_edge) / 2.0
    return float(centre - rec.center_frequency)


def resolve_exclude_labels(values: list[str] | None) -> frozenset[str]:
    """Resolve `--exclude-label` values, falling back to the default."""
    if values is None or not values:
        return frozenset(DEFAULT_EXCLUDE_LABELS)
    return frozenset(values)
